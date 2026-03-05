# Step7_direct_pose_from_rgbd.py
# ──────────────────────────────────────────────────────────────
# 캘리브레이션(T_C0_Ci, K, D) + RGB-D 프레임만으로 직접 3D 포즈 추정
#
# ML 모델 불필요. numpy + opencv만 사용.
# --ref_ply 지정 시: ICP로 CAD 모델 매칭 → 정확한 회전값
# --ref_ply 미지정 시: PLY 파일 불필요, PCA로 회전 추정 (180° 모호성 있음)
#
# ★ 원리:
#   1. 각 카메라 depth → 3D 역투영 (핀홀 + 왜곡 보정)
#   2. T_C0_Ci로 cam0 좌표계(실세계 mm)에 모두 합침
#   3. RANSAC → 테이블 평면 제거
#   4. BFS 클러스터링 → 객체 점군 추출 (가장 큰 덩어리)
#   5a. [PCA]  공분산 고유벡터 = 회전  (--ref_ply 없을 때, 180° 모호성)
#   5b. [ICP]  CAD 모델 점군 매칭 → 정확한 회전  (--ref_ply 있을 때)
#
# ★ 위치는 어떻게?
#   depth 역투영 + 멀티카메라 융합 → 테이블 제거 → 객체 centroid
#
# ★ 회전은 어떻게?
#   [PCA] 객체 점군의 공분산 행렬 → 고유벡터 3개 = 주축 3개
#   [ICP] CAD 모델(--ref_ply)과 점군 매칭 → 180° 모호성 없는 정확한 회전
#
# 입력:
#   - RGB-D 프레임: data/rgbd_capture/cam{0,1,2}/rgb_NNNNNN.jpg, depth_NNNNNN.png
#   - 캘리브레이션: T_C0_C1.npy, T_C0_C2.npy
#   - 내부파라미터: cam{0,1,2}.npz (color_K, color_D, depth_scale_m_per_unit)
#
# 출력:
#   - 객체 위치 (x, y, z) mm — cam0 좌표계
#   - 객체 회전 (Euler, Quaternion, Axis-Angle, Rotation Matrix)
#   - OBB 크기 (가로, 세로, 높이) mm
#   - PLY 파일 (융합 점군, 객체 점군)
#   - JSON 결과 파일
#   - 좌표 프레임 시각화 PNG
#
# 필요 패키지: numpy, opencv-python, matplotlib
# ──────────────────────────────────────────────────────────────
"""
사용법 (pose_step7_direct/ 폴더에서 실행):

# PCA 방식 (기본, ref_ply 없어도 동작, 180° 모호성 있음)
python Step7_direct_pose_from_rgbd.py \
  --rgbd_dir    ../data/rgbd_capture \
  --calib_dir   ../data/cube_session_01/calib_out_cube \
  --frame 0

# ICP 방식 (CAD 모델로 정확한 회전 추정, 권장)
python Step7_direct_pose_from_rgbd.py \
  --rgbd_dir    ../data/rgbd_capture \
  --calib_dir   ../data/cube_session_01/calib_out_cube \
  --ref_ply     ../data/3d_ply/tiger.ply \
  --frame 0

# 특정 프레임 + depth 범위 조정
python Step7_direct_pose_from_rgbd.py \
  --rgbd_dir    ../data/rgbd_capture \
  --calib_dir   ../data/cube_session_01/calib_out_cube \
  --ref_ply     ../data/3d_ply/tiger.ply \
  --frame 5 --z_min 0.15 --z_max 0.8
"""

import os
import json
import argparse
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


# ================================================================
#  1. 캘리브레이션 로드
# ================================================================

def load_intrinsics(intrinsics_dir: str, cam_idx: int):
    """cam{i}.npz → K(3x3), D(distortion), depth_scale(m/unit)."""
    p = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    data = np.load(p, allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64) if "color_D" in data else np.zeros(5)
    ds = float(data["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in data else 0.001
    return K, D, ds


def load_extrinsics(calib_dir: str, cam_indices: List[int]):
    """T_C0_C{i}.npy → {cam_idx: 4x4 SE3}. cam0=identity."""
    T = {}
    for ci in cam_indices:
        if ci == 0:
            T[ci] = np.eye(4, dtype=np.float64)
        else:
            p = os.path.join(calib_dir, f"T_C0_C{ci}.npy")
            T[ci] = np.load(p).astype(np.float64)
    return T


# ================================================================
#  2. Depth → 3D 역투영
# ================================================================

def depth_to_points(
    depth_u16: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    depth_scale: float,
    z_min: float = 0.1,
    z_max: float = 1.0,
    stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    depth 이미지 → 카메라 프레임 3D 점군.

    원리:
      1) 픽셀 (u,v)에서 depth 값 z를 읽음
      2) cv2.undistortPoints()로 렌즈 왜곡 보정 → 정규화 좌표 (x_n, y_n)
      3) X = x_n * z,  Y = y_n * z,  Z = z  (핀홀 모델 역투영)

    stride로 서브샘플링 (stride=2이면 4배 빠름, 점 수 1/4).
    """
    h, w = depth_u16.shape
    v_grid, u_grid = np.mgrid[0:h:stride, 0:w:stride]
    u_flat = u_grid.ravel().astype(np.float64)
    v_flat = v_grid.ravel().astype(np.float64)

    z = depth_u16[v_grid, u_grid].ravel().astype(np.float64) * depth_scale
    valid = (z > z_min) & (z < z_max)
    u_v, v_v, z_v = u_flat[valid], v_flat[valid], z[valid]

    if len(z_v) == 0:
        return np.empty((0, 3)), np.empty((0, 2))

    # 왜곡 보정: (u,v) → 정규화 좌표 (x_n, y_n)
    pts_2d = np.column_stack([u_v, v_v]).reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts_2d, K, D).reshape(-1, 2)

    # 3D: X = x_n * z,  Y = y_n * z,  Z = z
    xyz = np.column_stack([undist[:, 0] * z_v, undist[:, 1] * z_v, z_v])
    return xyz, np.column_stack([u_v, v_v])


def depth_to_colored_points(
    depth_u16: np.ndarray,
    rgb_bgr: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    depth_scale: float,
    z_min: float = 0.1,
    z_max: float = 1.0,
    stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """depth → 3D 점 + RGB 색상."""
    pts, uv = depth_to_points(depth_u16, K, D, depth_scale, z_min, z_max, stride)
    if len(pts) == 0:
        return pts, np.empty((0, 3))
    u_idx = uv[:, 0].astype(int)
    v_idx = uv[:, 1].astype(int)
    colors = rgb_bgr[v_idx, u_idx][:, ::-1].astype(np.float64) / 255.0  # BGR→RGB, 0~1
    return pts, colors


# ================================================================
#  3. 멀티카메라 융합
# ================================================================

def fuse_multicam(
    rgbd_dir: str,
    frame_idx: int,
    cam_indices: List[int],
    K_map: dict,
    D_map: dict,
    ds_map: dict,
    T_map: dict,
    z_min: float = 0.1,
    z_max: float = 1.0,
    stride: int = 2,
    pad: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    3대 카메라 RGB-D → cam0 좌표계 3D 점군 융합.

    원리:
      cam_i의 점 p_i를 T_C0_Ci로 변환:
        p_cam0 = R_C0_Ci @ p_cam_i + t_C0_Ci
      → 모든 점이 cam0 실세계 좌표계에 통합됨.
    """
    fid = f"{frame_idx:0{pad}d}"
    all_pts, all_cols = [], []

    for ci in cam_indices:
        rgb_path = os.path.join(rgbd_dir, f"cam{ci}", f"rgb_{fid}.jpg")
        depth_path = os.path.join(rgbd_dir, f"cam{ci}", f"depth_{fid}.png")
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"  cam{ci}: skip (파일 없음)")
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_u16 is None:
            continue

        pts_cam, cols = depth_to_colored_points(
            depth_u16, rgb_bgr, K_map[ci], D_map[ci], ds_map[ci],
            z_min, z_max, stride,
        )
        if len(pts_cam) == 0:
            continue

        # cam_i → cam0 변환
        T = T_map[ci]
        pts_cam0 = pts_cam @ T[:3, :3].T + T[:3, 3]

        all_pts.append(pts_cam0)
        all_cols.append(cols)

        t_mm = T[:3, 3] * 1000
        print(f"  cam{ci}: {len(pts_cam):,} pts  "
              f"T=({'I' if ci == 0 else f'{t_mm[0]:.0f},{t_mm[1]:.0f},{t_mm[2]:.0f}mm'})")

    if not all_pts:
        raise RuntimeError("유효한 RGB-D 프레임 없음")

    return np.concatenate(all_pts), np.concatenate(all_cols)


# ================================================================
#  4. 아웃라이어 제거 (SOR)
# ================================================================

def statistical_outlier_removal(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    std_ratio: float = 1.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """축별 IQR + 중심 거리 기반 아웃라이어 제거."""
    n = len(points)
    if n < 10:
        return points, colors

    mask = np.ones(n, dtype=bool)
    for axis in range(3):
        vals = points[:, axis]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        mask &= (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)

    centroid = points[mask].mean(axis=0) if mask.sum() > 0 else points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    mu_d = dists[mask].mean() if mask.sum() > 0 else dists.mean()
    std_d = dists[mask].std() if mask.sum() > 0 else dists.std()
    mask &= dists < (mu_d + std_ratio * std_d)

    removed = n - mask.sum()
    print(f"  SOR: {n:,} → {mask.sum():,} ({removed:,} removed)")
    return points[mask], (colors[mask] if colors is not None else None)


# ================================================================
#  5. RANSAC 평면 검출
# ================================================================

def ransac_plane(
    points: np.ndarray,
    threshold: float = 0.005,
    n_iterations: int = 2000,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    RANSAC 평면 검출.

    원리:
      1) 랜덤 3점 선택 → 외적으로 평면 법선(normal) 계산
      2) 모든 점의 평면까지 거리 = |n·p + d|
      3) threshold 이내 점 수가 최대인 평면 선택
      4) SVD로 최종 정밀화

    반환: (normal, d, inlier_mask)  —  n·p + d = 0
    """
    rng = np.random.RandomState(42)
    n = len(points)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    best_normal = np.array([0.0, 1.0, 0.0])
    best_d = 0.0

    for _ in range(n_iterations):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normal /= norm_len
        d = -normal.dot(p0)

        dists = np.abs(points @ normal + d)
        mask = dists < threshold
        count = mask.sum()
        if count > best_count:
            best_count = count
            best_mask = mask
            best_normal = normal
            best_d = d

    # SVD 정밀화
    if best_count > 3:
        inlier_pts = points[best_mask]
        centroid = inlier_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid)
        best_normal = Vt[2]
        best_d = -best_normal.dot(centroid)
        best_mask = np.abs(points @ best_normal + best_d) < threshold

    return best_normal, best_d, best_mask


# ================================================================
#  6. 객체 추출 (테이블 제거 + BFS 클러스터링)
# ================================================================

def extract_object(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    plane_threshold: float = 0.008,
    min_height: float = 0.01,
    min_cluster: int = 200,
    cluster_voxel: float = 0.015,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    테이블 평면 제거 → 객체 점군 추출.

    원리:
      1) RANSAC으로 가장 큰 평면(= 테이블) 검출
      2) 평면 위쪽 점만 남김 (signed distance > min_height)
      3) 남은 점들을 voxel grid에 매핑
      4) 26-연결 BFS로 연결된 덩어리(cluster) 찾기
      5) 가장 큰 cluster = 객체
    """
    # 서브샘플 후 RANSAC
    max_ransac = 30000
    sub_stride = max(1, len(points) // max_ransac)
    sub_pts = points[::sub_stride] if len(points) > max_ransac else points

    normal, d_plane, _ = ransac_plane(sub_pts, threshold=plane_threshold)
    print(f"  Plane normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})")

    # 평면 기준 분리
    plane_mask = np.abs(points @ normal + d_plane) < plane_threshold
    signed_dist = points @ normal + d_plane
    above = signed_dist > min_height
    below = signed_dist < -min_height

    # 점이 적은 쪽 = 테이블 위 객체
    obj_mask = above if (above.sum() > 0 and above.sum() < below.sum()) else below
    obj_mask = obj_mask & ~plane_mask

    if obj_mask.sum() < min_cluster:
        obj_mask = ~plane_mask

    obj_pts = points[obj_mask]
    obj_cols = colors[obj_mask] if colors is not None else None
    print(f"  After plane removal: {obj_mask.sum():,} pts (table: {plane_mask.sum():,})")

    if len(obj_pts) < min_cluster:
        return obj_pts, obj_cols

    # BFS 클러스터링 (26-연결)
    voxel_keys = (obj_pts / cluster_voxel).astype(np.int32)
    key_to_idx: Dict[tuple, List[int]] = {}
    for i in range(len(voxel_keys)):
        key = tuple(voxel_keys[i].tolist())
        key_to_idx.setdefault(key, []).append(i)

    occupied = set(key_to_idx.keys())
    visited = set()
    best_cluster: List[int] = []

    for start_key in key_to_idx:
        if start_key in visited:
            continue
        queue = deque([start_key])
        visited.add(start_key)
        cluster: List[int] = []
        while queue:
            cur = queue.popleft()
            cluster.extend(key_to_idx[cur])
            cx, cy, cz = cur
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nb = (cx + dx, cy + dy, cz + dz)
                        if nb in occupied and nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) >= min_cluster:
        idx = np.array(best_cluster)
        obj_pts = obj_pts[idx]
        obj_cols = obj_cols[idx] if obj_cols is not None else None

    print(f"  Object cluster: {len(obj_pts):,} pts")
    return obj_pts, obj_cols


# ================================================================
#  7. PCA 포즈 추정
# ================================================================

def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    """R → Euler XYZ (degrees)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z]))


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """R → quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w, x, y, z = 0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotation_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """R → (axis, angle_deg)."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    axis /= (np.linalg.norm(axis) + 1e-12)
    return axis, np.degrees(angle)


def estimate_pose(points: np.ndarray) -> dict:
    """
    PCA 기반 포즈 추정.

    원리:
      1) centroid = 점군 평균 → 이것이 객체의 3D 위치
      2) 공분산 행렬 C = (P - μ)ᵀ(P - μ) / (n-1)
      3) 고유분해: C = V Λ Vᵀ
         - λ₁ > λ₂ > λ₃ (내림차순 정렬)
         - v₁ = 분산 가장 큰 방향 = 객체의 길이 방향
         - v₂ = 두 번째 = 너비 방향
         - v₃ = 가장 작은 = 높이 방향
      4) R = [v₁ | v₂ | v₃] → 이것이 객체의 회전 행렬
      5) det(R) < 0이면 v₃ 뒤집어서 오른손 좌표계 보장
      6) OBB = R 방향으로 투영한 min/max 차이 = 객체 크기
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = (centered.T @ centered) / (len(points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 큰 순서로 정렬
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 오른손 좌표계 보장
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    R = eigenvectors
    projected = centered @ R
    extents = projected.max(axis=0) - projected.min(axis=0)

    euler = rotation_to_euler(R)
    quat = rotation_to_quaternion(R)
    axis, angle = rotation_to_axis_angle(R)

    return {
        "position_m": centroid,
        "position_mm": centroid * 1000,
        "rotation_matrix": R,
        "euler_xyz_deg": euler,
        "quaternion_wxyz": quat,
        "axis_angle": {"axis": axis, "angle_deg": angle},
        "obb_extents_m": extents,
        "obb_extents_mm": extents * 1000,
        "eigenvalues": eigenvalues,
    }


# ================================================================
#  7b. PLY 로드 + ICP 포즈 추정 (CAD 모델 기반 회전)
# ================================================================

def load_ply_points(path: str, max_pts: int = 5000) -> np.ndarray:
    """
    PLY 파일에서 XYZ 점군 로드 (ASCII / binary-little-endian 자동 감지).
    max_pts 이하로 균일 다운샘플링 → ICP 속도 유지.
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("utf-8", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        header = "\n".join(header_lines)
        n_verts = int(next(
            l.split()[-1] for l in header_lines if l.startswith("element vertex")
        ))
        is_binary = "binary_little_endian" in header

        props = []
        in_vert = False
        for l in header_lines:
            if l.startswith("element vertex"):
                in_vert = True; continue
            if l.startswith("element") and "vertex" not in l:
                in_vert = False
            if in_vert and l.startswith("property"):
                parts = l.split()
                props.append((parts[1], parts[2]))  # (type, name)

        _dtype_map = {
            "float": np.float32, "float32": np.float32,
            "double": np.float64, "float64": np.float64,
            "int": np.int32, "int32": np.int32,
            "uint": np.uint32, "uint32": np.uint32,
            "short": np.int16, "ushort": np.uint16,
            "uchar": np.uint8, "char": np.int8,
        }

        if is_binary:
            dt = np.dtype([(name, _dtype_map.get(ptype, np.float32))
                           for ptype, name in props])
            raw = f.read(n_verts * dt.itemsize)
            arr = np.frombuffer(raw, dtype=dt)
        else:
            rows = []
            for _ in range(n_verts):
                row = f.readline().decode("utf-8").split()
                rows.append(row[:len(props)])
            arr_np = np.array(rows, dtype=np.float32)
            arr = np.zeros(len(arr_np), dtype=[(name, np.float32) for _, name in props])
            for i, (_, name) in enumerate(props):
                if i < arr_np.shape[1]:
                    arr[name] = arr_np[:, i]

    names = [name for _, name in props]
    xyz = np.column_stack([
        arr["x"].astype(np.float64) if "x" in names else np.zeros(n_verts),
        arr["y"].astype(np.float64) if "y" in names else np.zeros(n_verts),
        arr["z"].astype(np.float64) if "z" in names else np.zeros(n_verts),
    ])
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) > max_pts:
        idx = np.linspace(0, len(xyz) - 1, max_pts, dtype=int)
        xyz = xyz[idx]
    return xyz


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    init_R: Optional[np.ndarray] = None,
    init_t: Optional[np.ndarray] = None,
    max_iter: int = 60,
    tol: float = 1e-5,
    max_dist_frac: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    순수 numpy Point-to-Point ICP.

    알고리즘 (매 반복):
      1) source를 현재 R,t로 변환
      2) target에서 최근접 이웃 탐색 (브루트포스, 서브샘플)
      3) max_dist_frac 이내 대응점만 사용
      4) SVD로 rigid transform 계산:
           H = (src - c_s)ᵀ (tgt - c_t)
           U,S,Vᵀ = SVD(H)
           R = VᵀᵀUᵀ  (det<0이면 Vᵀ[-1]*=-1)
           t = c_t - R @ c_s
      5) RMSE 수렴 판정

    반환: (R, t, rmse)
    """
    N_SRC = min(1200, len(source))
    N_TGT = min(2000, len(target))
    rng = np.random.RandomState(0)

    src = source[rng.choice(len(source), N_SRC, replace=False)].copy()
    tgt = target[rng.choice(len(target), N_TGT, replace=False)].copy()

    R = init_R.copy() if init_R is not None else np.eye(3)
    t = init_t.copy() if init_t is not None else np.zeros(3)
    prev_rmse = np.inf
    valid = np.ones(N_SRC, dtype=bool)
    nn_d2 = np.zeros(N_SRC)

    for _ in range(max_iter):
        # 변환 적용
        src_t = src @ R.T + t

        # 최근접 이웃 (브루트포스)
        diff = src_t[:, None, :] - tgt[None, :, :]   # (N_SRC, N_TGT, 3)
        d2 = (diff ** 2).sum(axis=2)                  # (N_SRC, N_TGT)
        nn_idx = d2.argmin(axis=1)
        nn_d2 = d2[np.arange(N_SRC), nn_idx]

        # 거리 임계값 필터
        max_d2 = np.percentile(nn_d2, max_dist_frac * 100)
        valid = nn_d2 <= max_d2
        if valid.sum() < 6:
            break

        s_valid = src_t[valid]
        t_valid = tgt[nn_idx[valid]]

        # SVD rigid transform
        c_s = s_valid.mean(axis=0)
        c_t = t_valid.mean(axis=0)
        H = (s_valid - c_s).T @ (t_valid - c_t)
        U, _, Vt = np.linalg.svd(H)
        R_step = Vt.T @ U.T
        if np.linalg.det(R_step) < 0:
            Vt[-1] *= -1
            R_step = Vt.T @ U.T

        t_step = c_t - R_step @ c_s
        R = R_step @ R
        t = R_step @ t + t_step

        rmse = float(np.sqrt(nn_d2[valid].mean()))
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    final_rmse = float(np.sqrt(nn_d2[valid].mean())) if valid.sum() > 0 else np.inf
    return R, t, final_rmse


def estimate_pose_icp(
    pts_object: np.ndarray,
    ref_pts: np.ndarray,
    max_iter: int = 60,
) -> dict:
    """
    ICP 기반 6DoF 포즈 추정.

    원리:
      - 위치: pts_object의 centroid (3D 평균, PCA와 동일하게 정확)
      - 회전: ICP로 CAD 모델(ref_pts)에 객체 점군을 맞추어 회전행렬 추출

    180° 모호성 해소:
      PCA 초기화 후 4가지 축 부호 조합 × ICP → 최저 RMSE 선택
      → ref_ply의 형상 정보가 올바른 방향을 결정함

    스케일 정규화:
      두 점군 모두 zero-centroid + unit RMS로 정규화
      → PLY 단위(mm/m/임의)와 무관하게 동작
    """
    # 위치 = centroid (정확함, ICP 불필요)
    centroid = pts_object.mean(axis=0)

    # ── 스케일 정규화 ──────────────────────────────
    c_obj = pts_object.mean(axis=0)
    obj_n = pts_object - c_obj
    s_obj = float(np.sqrt((obj_n ** 2).sum(axis=1).mean())) + 1e-12
    obj_n /= s_obj

    c_ref = ref_pts.mean(axis=0)
    ref_n = ref_pts - c_ref
    s_ref = float(np.sqrt((ref_n ** 2).sum(axis=1).mean())) + 1e-12
    ref_n /= s_ref

    # ── PCA 주축 → 초기 회전 ─────────────────────
    def _pca_axes(pts):
        c = pts - pts.mean(axis=0)
        cov = (c.T @ c) / (len(pts) - 1)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]
        if np.linalg.det(vecs) < 0:
            vecs[:, 2] *= -1
        return vecs

    E_obj = _pca_axes(obj_n)
    E_ref = _pca_axes(ref_n)
    R_pca = E_ref @ E_obj.T

    # ── 4 부호 조합 × ICP (180° 모호성 해소) ──────
    # PCA 고유벡터는 부호가 불확정이므로 4가지 valid combo를 모두 시도
    sign_combos = [
        np.diag([1.0,  1.0,  1.0]),
        np.diag([-1.0, -1.0,  1.0]),
        np.diag([-1.0,  1.0, -1.0]),
        np.diag([1.0,  -1.0, -1.0]),
    ]
    best_R, best_rmse = np.eye(3), np.inf
    for S in sign_combos:
        R_init = R_pca @ S
        R_icp, _, rmse = icp_point_to_point(obj_n, ref_n, init_R=R_init, max_iter=max_iter)
        if rmse < best_rmse:
            best_rmse = rmse
            best_R = R_icp

    R = best_R

    # OBB (ICP 회전 적용)
    centered = pts_object - centroid
    projected = centered @ R
    extents = projected.max(axis=0) - projected.min(axis=0)

    euler = rotation_to_euler(R)
    quat = rotation_to_quaternion(R)
    axis_aa, angle_aa = rotation_to_axis_angle(R)

    return {
        "position_m": centroid,
        "position_mm": centroid * 1000,
        "rotation_matrix": R,
        "euler_xyz_deg": euler,
        "quaternion_wxyz": quat,
        "axis_angle": {"axis": axis_aa, "angle_deg": angle_aa},
        "obb_extents_m": extents,
        "obb_extents_mm": extents * 1000,
        "eigenvalues": np.array([best_rmse, best_rmse, best_rmse]),
        "icp_rmse_normalized": float(best_rmse),
    }


# ================================================================
#  8. PLY 저장
# ================================================================

def save_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
    """ASCII PLY 저장."""
    n = len(points)
    has_color = colors is not None and len(colors) == n
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            line = f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}"
            if has_color:
                r, g, b = np.clip(colors[i] * 255, 0, 255).astype(int)
                line += f" {r} {g} {b}"
            f.write(line + "\n")
    print(f"  [SAVE] {path}  ({n:,} pts)")


# ================================================================
#  9. 좌표 프레임 시각화
# ================================================================

def visualize_pose(
    T_map: dict,
    pose: dict,
    out_path: str,
):
    """cam0/cam1/cam2 + 객체 포즈를 3D 좌표 프레임으로 시각화."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Direct Pose from RGB-D  (cam0 frame, mm)", fontsize=13, pad=15)
    ax.set_xlabel("X (mm)", fontsize=10)
    ax.set_ylabel("Y (mm)", fontsize=10)
    ax.set_zlabel("Z (mm)", fontsize=10)

    def draw_axes(R, t, length, label, lw=2, alpha=1.0):
        colors = ["#e74c3c", "#27ae60", "#2980b9"]
        names = ["X", "Y", "Z"]
        for i in range(3):
            v = R[:, i] * length
            ax.quiver(t[0], t[1], t[2], v[0], v[1], v[2],
                      color=colors[i], linewidth=lw, arrow_length_ratio=0.12, alpha=alpha)
            tip = t + v * 1.15
            ax.text(tip[0], tip[1], tip[2], names[i],
                    fontsize=6, color=colors[i], fontweight="bold", alpha=alpha)
        if label:
            ax.text(t[0], t[1], t[2] - length * 0.4, label,
                    fontsize=9, fontweight="bold", ha="center")

    def draw_cam(R, t, size, color, label):
        s = size
        local = np.array([
            [0,0,0],[-s,-s*0.7,s*1.5],[s,-s*0.7,s*1.5],[s,s*0.7,s*1.5],[-s,s*0.7,s*1.5]
        ])
        pts = (R @ local.T).T + t
        for i in range(1,5):
            j = (i%4)+1
            ax.add_collection3d(Poly3DCollection(
                [[pts[0],pts[i],pts[j]]], alpha=0.12, facecolor=color,
                edgecolor=color, linewidth=0.5))
        ax.add_collection3d(Poly3DCollection(
            [[pts[1],pts[2],pts[3],pts[4]]], alpha=0.08, facecolor=color,
            edgecolor=color, linewidth=0.5))
        ax.text(t[0], t[1], t[2]-size*1.5, label,
                fontsize=9, fontweight="bold", ha="center", color=color)

    def draw_obb(R, center, extents, color):
        h = extents / 2
        c = np.array([[-h[0],-h[1],-h[2]],[h[0],-h[1],-h[2]],[h[0],h[1],-h[2]],[-h[0],h[1],-h[2]],
                       [-h[0],-h[1],h[2]],[h[0],-h[1],h[2]],[h[0],h[1],h[2]],[-h[0],h[1],h[2]]])
        corners = (R @ c.T).T + center
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i,j in edges:
            ax.plot3D(*zip(corners[i],corners[j]), color=color, lw=1.0, alpha=0.5)
        faces = [[corners[j] for j in f] for f in
                 [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]]
        ax.add_collection3d(Poly3DCollection(
            faces, alpha=0.06, facecolor=color, edgecolor=color, linewidth=0.3))

    # cam0 원점
    draw_axes(np.eye(3), np.zeros(3), 70, "cam0 (ref)", lw=3)

    # cam1, cam2
    cam_colors = {1: "#e67e22", 2: "#8e44ad"}
    for ci in T_map:
        if ci == 0:
            continue
        R_ci = T_map[ci][:3, :3]
        t_ci = T_map[ci][:3, 3] * 1000
        draw_cam(R_ci, t_ci, 30, cam_colors.get(ci, "#95a5a6"), f"cam{ci}")
        draw_axes(R_ci, t_ci, 40, "", lw=1.5, alpha=0.5)
        # 베이스라인
        ax.plot3D([0,t_ci[0]], [0,t_ci[1]], [0,t_ci[2]],
                  "--", color=cam_colors.get(ci, "#95a5a6"), lw=0.8, alpha=0.3)

    # 객체
    obj_pos = pose["position_mm"]
    obj_R = pose["rotation_matrix"]
    obj_obb = pose["obb_extents_mm"]
    euler = pose["euler_xyz_deg"]

    draw_axes(obj_R, obj_pos, 55, "", lw=3.5)
    draw_obb(obj_R, obj_pos, obj_obb, "#c0392b")

    # 라벨
    ax.text(obj_pos[0], obj_pos[1]-50, obj_pos[2]+75,
            "Object", fontsize=11, fontweight="bold", color="#c0392b", ha="center")
    ax.text(obj_pos[0], obj_pos[1]-50, obj_pos[2]+55,
            f"({obj_pos[0]:.1f}, {obj_pos[1]:.1f}, {obj_pos[2]:.1f}) mm",
            fontsize=8, color="#2c3e50", ha="center")
    ax.text(obj_pos[0], obj_pos[1]-50, obj_pos[2]+38,
            f"euler ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg",
            fontsize=8, color="#2c3e50", ha="center")
    ax.text(obj_pos[0], obj_pos[1]-50, obj_pos[2]+21,
            f"OBB {obj_obb[0]:.1f} x {obj_obb[1]:.1f} x {obj_obb[2]:.1f} mm",
            fontsize=7, color="#c0392b", ha="center")

    # cam0→object
    ax.plot3D([0, obj_pos[0]], [0, obj_pos[1]], [0, obj_pos[2]],
              ":", color="#c0392b", lw=1, alpha=0.3)
    mid = obj_pos / 2
    ax.text(mid[0], mid[1], mid[2],
            f"{np.linalg.norm(obj_pos):.0f} mm", fontsize=7, color="#c0392b")

    # 등비 축
    all_pts = [[0,0,0], obj_pos.tolist()]
    for ci in T_map:
        if ci != 0:
            all_pts.append((T_map[ci][:3, 3] * 1000).tolist())
    pts = np.array(all_pts)
    c = pts.mean(axis=0)
    r = max((pts.max(axis=0) - pts.min(axis=0)).max() / 2 * 1.3, 1.0)
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)
    ax.view_init(elev=25, azim=-55)

    # 하단 범례
    fig.text(0.5, 0.01, "Axis color:  X = Red   Y = Green   Z = Blue",
             fontsize=9, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="#ecf0f1", ec="#bdc3c7"))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {out_path}")


# ================================================================
#  10. Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step7: RGB-D + Calibration → Direct 3D Pose Estimation"
    )
    parser.add_argument("--rgbd_dir", required=True, help="RGB-D frames (cam0/cam1/cam2)")
    parser.add_argument("--calib_dir", required=True, help="T_C0_C1.npy, T_C0_C2.npy")
    parser.add_argument("--intrinsics_dir", default="../intrinsics", help="cam{i}.npz")
    parser.add_argument("--frame", type=int, default=0, help="Frame index")
    parser.add_argument("--z_min", type=float, default=0.1, help="Min depth (m)")
    parser.add_argument("--z_max", type=float, default=1.0, help="Max depth (m)")
    parser.add_argument("--stride", type=int, default=2, help="Depth subsampling stride")
    parser.add_argument("--ref_ply", default=None,
                        help="CAD 모델 PLY 경로 (지정 시 ICP 회전 사용, 미지정 시 PCA fallback)")
    parser.add_argument("--icp_iter", type=int, default=60, help="ICP 최대 반복 횟수")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    t_start = time.time()

    # ─── 캘리브레이션 ───
    print("=" * 60)
    print(" Step 1: Load Calibration")
    print("=" * 60)

    # 카메라 탐색
    cam_indices = sorted([
        int(d.replace("cam", ""))
        for d in os.listdir(args.rgbd_dir)
        if d.startswith("cam") and os.path.isdir(os.path.join(args.rgbd_dir, d))
    ])
    print(f"  Cameras: {cam_indices}")

    K_map, D_map, ds_map = {}, {}, {}
    for ci in cam_indices:
        K, D, ds = load_intrinsics(args.intrinsics_dir, ci)
        K_map[ci], D_map[ci], ds_map[ci] = K, D, ds
        print(f"  cam{ci}: fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    T_map = load_extrinsics(args.calib_dir, cam_indices)
    for ci in cam_indices:
        if ci == 0:
            print(f"  T_C0_C0: Identity")
        else:
            t = T_map[ci][:3, 3] * 1000
            print(f"  T_C0_C{ci}: t=({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) mm")

    # ─── 멀티카메라 융합 ───
    print(f"\n{'=' * 60}")
    print(f" Step 2: Multi-camera Fusion (frame {args.frame})")
    print("=" * 60)

    pts_fused, cols_fused = fuse_multicam(
        args.rgbd_dir, args.frame, cam_indices,
        K_map, D_map, ds_map, T_map,
        args.z_min, args.z_max, args.stride,
    )
    print(f"  Fused: {len(pts_fused):,} pts")

    # SOR
    pts_fused, cols_fused = statistical_outlier_removal(pts_fused, cols_fused)

    # ─── 테이블 제거 + 객체 추출 ───
    print(f"\n{'=' * 60}")
    print(f" Step 3: Table Removal + Object Extraction")
    print("=" * 60)

    obj_pts, obj_cols = extract_object(pts_fused, cols_fused)

    if len(obj_pts) < 50:
        print("[ERROR] Object not found (too few points)")
        return

    # ─── 포즈 추정 (ICP or PCA) ───
    if args.ref_ply is not None:
        print(f"\n{'=' * 60}")
        print(f" Step 4: ICP Pose Estimation  (ref: {os.path.basename(args.ref_ply)})")
        print("=" * 60)
        ref_pts = load_ply_points(args.ref_ply, max_pts=5000)
        print(f"  Reference PLY: {len(ref_pts):,} pts")
        pose_pca = estimate_pose(obj_pts)   # 비교용
        pose = estimate_pose_icp(obj_pts, ref_pts, max_iter=args.icp_iter)
        rotation_method = "icp"
        R_diff = pose["rotation_matrix"].T @ pose_pca["rotation_matrix"]
        angle_diff = float(np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))))
        print(f"  ICP RMSE (normalized): {pose['icp_rmse_normalized']:.5f}")
        print(f"  PCA vs ICP angle diff: {angle_diff:.1f} deg")
    else:
        print(f"\n{'=' * 60}")
        print(f" Step 4: PCA Pose Estimation  (--ref_ply 미지정, 180° 모호성 있음)")
        print("=" * 60)
        pose = estimate_pose(obj_pts)
        rotation_method = "pca"
        angle_diff = 0.0

    pos_mm = pose["position_mm"]
    euler = pose["euler_xyz_deg"]
    obb_mm = pose["obb_extents_mm"]
    quat = pose["quaternion_wxyz"]
    ax_ang = pose["axis_angle"]

    print(f"\n  Position:   ({pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}) mm")
    print(f"  Euler XYZ:  ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg")
    print(f"  Quaternion: ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
    print(f"  Axis-Angle: ({ax_ang['axis'][0]:.3f}, {ax_ang['axis'][1]:.3f}, {ax_ang['axis'][2]:.3f}) "
          f"{ax_ang['angle_deg']:.1f} deg")
    print(f"  OBB size:   ({obb_mm[0]:.1f}, {obb_mm[1]:.1f}, {obb_mm[2]:.1f}) mm")
    print(f"  Distance:   {np.linalg.norm(pos_mm):.1f} mm from cam0")

    # ─── 저장 ───
    print(f"\n{'=' * 60}")
    print(f" Step 5: Save Results")
    print("=" * 60)

    out_dir = args.out_dir or "./output"
    os.makedirs(out_dir, exist_ok=True)

    # PLY
    n_save = min(len(pts_fused), 100000)
    stride_save = max(1, len(pts_fused) // n_save)
    save_ply(
        os.path.join(out_dir, f"cam0_fused_frame{args.frame:06d}.ply"),
        pts_fused[::stride_save], cols_fused[::stride_save] if cols_fused is not None else None,
    )
    save_ply(
        os.path.join(out_dir, f"object_frame{args.frame:06d}.ply"),
        obj_pts, obj_cols,
    )

    # JSON
    result = {
        "frame": args.frame,
        "coordinate_frame": "cam0 (real-world meters)",
        "cameras": cam_indices,
        "fused_points": len(pts_fused),
        "object_points": len(obj_pts),
        "pose": {
            "rotation_method": rotation_method,
            "ref_ply": args.ref_ply,
            "position_m": pose["position_m"].tolist(),
            "position_mm": pose["position_mm"].tolist(),
            "rotation_matrix": pose["rotation_matrix"].tolist(),
            "euler_xyz_deg": pose["euler_xyz_deg"].tolist(),
            "quaternion_wxyz": pose["quaternion_wxyz"].tolist(),
            "axis_angle": {
                "axis": pose["axis_angle"]["axis"].tolist(),
                "angle_deg": float(pose["axis_angle"]["angle_deg"]),
            },
            "obb_extents_m": pose["obb_extents_m"].tolist(),
            "obb_extents_mm": pose["obb_extents_mm"].tolist(),
            **({"icp_rmse_normalized": pose["icp_rmse_normalized"],
                "pca_vs_icp_angle_diff_deg": round(angle_diff, 2)}
               if rotation_method == "icp" else {}),
        },
        "elapsed_sec": round(time.time() - t_start, 2),
    }

    json_path = os.path.join(out_dir, f"pose_frame{args.frame:06d}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [SAVE] {json_path}")

    # 시각화
    vis_path = os.path.join(out_dir, f"pose_frame{args.frame:06d}.png")
    visualize_pose(T_map, pose, vis_path)

    # ─── 최종 요약 ───
    print(f"\n{'=' * 60}")
    print(f" Result Summary")
    print("=" * 60)
    print(f"  Object Position: ({pos_mm[0]:+.1f}, {pos_mm[1]:+.1f}, {pos_mm[2]:+.1f}) mm")
    print(f"  Object Rotation: ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}) deg")
    print(f"  Object Size:     ({obb_mm[0]:.1f} x {obb_mm[1]:.1f} x {obb_mm[2]:.1f}) mm")
    print(f"  Elapsed:         {time.time() - t_start:.1f} s")
    print(f"  Output:          {os.path.abspath(out_dir)}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
