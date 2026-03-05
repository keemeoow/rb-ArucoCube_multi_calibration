# Step6_estimate_pose_from_ply.py
# ──────────────────────────────────────────────────────────────
# ArUco 큐브 멀티카메라 캘리브레이션(T_C0_Ci, K, D)을 사용하여
# 3가지 PLY 소스(SAM3D, COLMAP, gs2mesh)의 3D Pose 추정 + 정합
#
# 파이프라인:
#   캘리브레이션(T_C0_C1, T_C0_C2) + 내부파라미터(K, D)
#     → RGB-D 역투영 → cam0 프레임 점군 생성
#     → 테이블 평면 RANSAC 제거 → 객체 점군 추출
#     → gs2mesh ↔ cam0 객체 직접 ICP 정합 → T_cam0_colmap
#     → 3개 PLY 모두 cam0 좌표계(실제 미터)로 변환
#
# 입력:
#   1. tiger.ply               — SAM3D 객체 (3DGS 좌표계)
#   2. point_cloud_cleaned.ply — COLMAP SfM 전체 씬
#   3. gs2mesh *.ply           — GS2Mesh 객체 메시 (COLMAP 좌표계)
#   + 캘리브레이션: T_C0_C1.npy, T_C0_C2.npy, cam{0,1,2}.npz
#   + RGB-D 프레임: data/rgbd_capture/cam{0,1,2}/
#
# 출력:
#   - 각 모델의 pose (cam0 좌표계, position + orientation)
#   - T_cam0_colmap.npy (재사용 가능)
#   - 정합된 PLY 파일 (cam0 좌표계)
#   - JSON 결과 파일
#
# 필요 패키지: numpy, opencv-python (open3d/scipy 불필요)
# ──────────────────────────────────────────────────────────────
"""
사용법 (pose_step6_ply/ 폴더에서 실행):

# ★ 권장: 캘리브레이션 직접 사용 (T_C0_Ci + K,D + RGB-D)
python Step6_estimate_pose_from_ply.py \
  --object_ply  ../data/3d_ply/tiger.ply \
  --scene_ply   ../data/3d_ply/point_cloud_cleaned.ply \
  --mesh_ply    "../data/3d_ply/tiger_figure_custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_20_cleaned_mesh.ply" \
  --calib_dir   ../data/cube_session_01/calib_out_cube \
  --intrinsics_dir ../intrinsics \
  --rgbd_dir    ../data/rgbd_capture

# 이미 계산된 T_cam0_colmap.npy 재사용
python Step6_estimate_pose_from_ply.py \
  --object_ply  ../data/3d_ply/tiger.ply \
  --scene_ply   ../data/3d_ply/point_cloud_cleaned.ply \
  --mesh_ply    "..." \
  --T_multicam_colmap ./output/T_cam0_colmap.npy

# 멀티캠 PLY 사용 (간접 방식)
python Step6_estimate_pose_from_ply.py \
  --object_ply  ../data/3d_ply/tiger.ply \
  --scene_ply   ../data/3d_ply/point_cloud_cleaned.ply \
  --mesh_ply    "..." \
  --multicam_ply ../data/rgbd_capture/ply/frame_000000.ply
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
#  1. 캘리브레이션 I/O
# ================================================================
def load_intrinsics(npz_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """카메라 내부파라미터 로드. 반환: (K_3x3, D_1d, depth_scale_m)."""
    data = np.load(npz_path)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64).ravel()
    depth_scale = float(data["depth_scale_m_per_unit"])
    return K, D, depth_scale


def load_extrinsic(npy_path: str) -> np.ndarray:
    """외부파라미터 T_C0_Ci (4x4 SE3) 로드."""
    T = np.load(npy_path).astype(np.float64)
    assert T.shape == (4, 4), f"Expected 4x4, got {T.shape}"
    return T


# ================================================================
#  2. PLY I/O (순수 numpy, open3d 불필요)
# ================================================================
_PLY_DTYPE_MAP = {
    "float": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
    "uchar": np.uint8, "uint8": np.uint8,
    "char": np.int8, "int8": np.int8,
    "short": np.int16, "int16": np.int16,
    "ushort": np.uint16, "uint16": np.uint16,
    "int": np.int32, "int32": np.int32,
    "uint": np.uint32, "uint32": np.uint32,
}


def load_ply(path: str) -> dict:
    """
    PLY 로더 (binary little-endian + ASCII 자동 감지).
    반환: {"xyz": Nx3, "normals": Nx3|None, "colors": Nx3 uint8|None,
           "n_faces": int, "n_vertices": int, "properties": [str]}
    """
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = 0
        n_faces = 0
        vertex_props = []
        in_vertex = True
        is_ascii = False

        for line in header_lines:
            if "format ascii" in line:
                is_ascii = True
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
                in_vertex = True
            elif line.startswith("element face"):
                n_faces = int(line.split()[-1])
                in_vertex = False
            elif line.startswith("property") and in_vertex and "list" not in line:
                parts = line.split()
                if len(parts) >= 3:
                    vertex_props.append(
                        (parts[2], _PLY_DTYPE_MAP.get(parts[1], np.float32))
                    )

        if is_ascii:
            lines_data = [
                f.readline().decode("ascii", errors="ignore").strip()
                for _ in range(n_vertices)
            ]
            prop_names = [p[0] for p in vertex_props]
            arr = np.empty((n_vertices, len(prop_names)), dtype=np.float64)
            for i, ln in enumerate(lines_data):
                vals = ln.split()
                for j in range(min(len(prop_names), len(vals))):
                    arr[i, j] = float(vals[j])
            data = {prop_names[j]: arr[:, j] for j in range(len(prop_names))}
        else:
            dt = np.dtype(vertex_props)
            raw = f.read(n_vertices * dt.itemsize)
            raw_data = np.frombuffer(raw, dtype=dt, count=n_vertices)
            data = {p[0]: raw_data[p[0]] for p in vertex_props}

    prop_names = [p[0] for p in vertex_props]

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)

    normals = None
    if all(k in prop_names for k in ("nx", "ny", "nz")):
        normals = np.column_stack(
            [data["nx"], data["ny"], data["nz"]]
        ).astype(np.float64)

    colors = None
    if all(k in prop_names for k in ("red", "green", "blue")):
        colors = np.column_stack([data["red"], data["green"], data["blue"]])
        if colors.dtype != np.uint8:
            colors = (
                (colors * 255).astype(np.uint8)
                if colors.max() <= 1.0
                else colors.astype(np.uint8)
            )
    elif "f_dc_0" in prop_names:
        # 3DGS SH band-0 → RGB 변환
        sh0 = np.column_stack(
            [data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]]
        ).astype(np.float64)
        C0 = 0.28209479177387814
        rgb_float = np.clip(0.5 + C0 * sh0, 0.0, 1.0)
        colors = (rgb_float * 255).astype(np.uint8)

    return {
        "xyz": xyz,
        "normals": normals,
        "colors": colors,
        "n_faces": n_faces,
        "n_vertices": n_vertices,
        "properties": prop_names,
    }


def save_ply(path: str, points: np.ndarray,
             colors: Optional[np.ndarray] = None) -> None:
    """ASCII PLY 저장."""
    n = len(points)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write(
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            )
        f.write("end_header\n")
        for i in range(n):
            line = f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}"
            if colors is not None:
                r, g, b = colors[i].astype(int)
                line += f" {r} {g} {b}"
            f.write(line + "\n")


# ================================================================
#  3. Depth → 3D 역투영 + 멀티캠 융합
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
    Depth → 3D 카메라 프레임 점군 역투영.
    cv2.undistortPoints로 렌즈 왜곡 보정 후 핀홀 역투영.
    반환: (Nx3 xyz, Nx2 pixel_coords)
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

    pts_2d = np.column_stack([u_v, v_v]).reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts_2d, K, D).reshape(-1, 2)

    xyz = np.column_stack([undist[:, 0] * z_v, undist[:, 1] * z_v, z_v])
    return xyz, np.column_stack([u_v, v_v])


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """SE3 변환: p' = R @ p + t."""
    return (T[:3, :3] @ points.T).T + T[:3, 3]


def statistical_outlier_removal(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    std_ratio: float = 1.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    축별 IQR + 중심 거리 기반 Statistical Outlier Removal.
    """
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

    return points[mask], (colors[mask] if colors is not None else None)


def build_cam0_cloud(
    calib_dir: str,
    intrinsics_dir: str,
    rgbd_dir: str,
    frame_idx: int = 0,
    n_cams: int = 3,
    z_min: float = 0.1,
    z_max: float = 0.80,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    캘리브레이션(T_C0_Ci) + 내부파라미터(K, D) + RGB-D → cam0 융합 점군.
    """
    all_pts, all_cols = [], []
    T_C0_Ci = {0: np.eye(4, dtype=np.float64)}
    for ci in range(1, n_cams):
        T_C0_Ci[ci] = load_extrinsic(
            os.path.join(calib_dir, f"T_C0_C{ci}.npy")
        )

    frame_str = f"{frame_idx:06d}"
    for ci in range(n_cams):
        K, D, ds = load_intrinsics(
            os.path.join(intrinsics_dir, f"cam{ci}.npz")
        )
        depth_path = os.path.join(rgbd_dir, f"cam{ci}", f"depth_{frame_str}.png")
        rgb_path = os.path.join(rgbd_dir, f"cam{ci}", f"rgb_{frame_str}.jpg")

        if not os.path.exists(depth_path):
            print(f"    [WARN] {depth_path} 없음, skip cam{ci}")
            continue

        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if depth_u16 is None:
            print(f"    [WARN] cam{ci} depth 로드 실패")
            continue

        pts_cam, px = depth_to_points(
            depth_u16, K, D, ds, z_min=z_min, z_max=z_max, stride=stride
        )
        if len(pts_cam) == 0:
            continue

        pts_cam0 = transform_points(pts_cam, T_C0_Ci[ci])
        all_pts.append(pts_cam0)

        if rgb is not None:
            u_idx = px[:, 0].astype(int)
            v_idx = px[:, 1].astype(int)
            all_cols.append(rgb[v_idx, u_idx][:, ::-1])  # BGR → RGB

        print(
            f"    cam{ci}: {len(pts_cam):,} pts  "
            f"(K={K[0,0]:.0f},{K[1,1]:.0f}  ds={ds}  "
            f"T={'I' if ci==0 else 'T_C0_C'+str(ci)})"
        )

    if not all_pts:
        return np.empty((0, 3)), None

    xyz = np.vstack(all_pts)
    colors = np.vstack(all_cols) if all_cols else None

    n_before = len(xyz)
    xyz, colors = statistical_outlier_removal(xyz, colors, std_ratio=1.5)
    print(f"    SOR: {n_before:,} → {len(xyz):,} ({n_before - len(xyz):,} 제거)")

    return xyz, colors


# ================================================================
#  4. RANSAC 평면 검출 + 객체 추출
# ================================================================
def ransac_plane(
    points: np.ndarray,
    threshold: float = 0.005,
    n_iterations: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    RANSAC 평면 검출. 반환: (normal, d, inlier_mask).
    평면 방정식: normal · p + d = 0
    """
    rng = np.random.RandomState(seed)
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

    # SVD refinement
    if best_count > 3:
        inlier_pts = points[best_mask]
        centroid = inlier_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid)
        best_normal = Vt[2]
        best_d = -best_normal.dot(centroid)
        best_mask = np.abs(points @ best_normal + best_d) < threshold

    return best_normal, best_d, best_mask


def extract_object_from_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    plane_threshold: float = 0.008,
    min_height_above_plane: float = 0.01,
    min_cluster_points: int = 200,
    cluster_voxel: float = 0.015,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    테이블 평면 RANSAC 제거 → 그 위의 객체 점군 추출 (BFS 클러스터링).
    1. RANSAC 서브샘플(30K)로 평면 검출, 전체에 적용
    2. signed distance로 평면 위/아래 분리 (적은 쪽 = 객체)
    3. voxel 기반 26-연결 BFS → 최대 클러스터 선택
    """
    # RANSAC subsample
    max_ransac = 30000
    sub_stride = max(1, len(points) // max_ransac)
    sub_pts = points[::sub_stride] if len(points) > max_ransac else points

    normal, d_plane, _ = ransac_plane(
        sub_pts, threshold=plane_threshold, n_iterations=2000
    )

    # 전체 점에 평면 적용
    plane_mask = np.abs(points @ normal + d_plane) < plane_threshold
    signed_dist = points @ normal + d_plane
    above = signed_dist > min_height_above_plane
    below = signed_dist < -min_height_above_plane

    # 점이 적은 쪽 = 객체 (테이블 위)
    if above.sum() > 0 and below.sum() > 0:
        obj_mask = above if above.sum() < below.sum() else below
    elif above.sum() > 0:
        obj_mask = above
    else:
        obj_mask = below

    obj_mask = obj_mask & ~plane_mask

    if obj_mask.sum() < min_cluster_points:
        obj_mask = ~plane_mask

    obj_pts = points[obj_mask]
    obj_cols = colors[obj_mask] if colors is not None else None

    if len(obj_pts) < min_cluster_points:
        return obj_pts, obj_cols

    # Voxel BFS clustering (26-connectivity)
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

    if len(best_cluster) < min_cluster_points:
        return obj_pts, obj_cols

    idx = np.array(best_cluster)
    return obj_pts[idx], (obj_cols[idx] if obj_cols is not None else None)


# ================================================================
#  5. Rotation 유틸리티
# ================================================================
def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    """R → Euler XYZ intrinsic (degrees)."""
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
        w, x, y, z = 0.25 / s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
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
    if abs(angle) < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2*np.sin(angle))
    return axis / np.linalg.norm(axis), float(np.degrees(angle))


# ================================================================
#  6. PCA Pose 추정 + OBB
# ================================================================
def estimate_pose_pca(points: np.ndarray) -> dict:
    """
    PCA → position(centroid) + orientation(eigenvectors) + extents.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = (centered.T @ centered) / (len(points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    R = eigenvectors
    projected = centered @ R
    extents = projected.max(axis=0) - projected.min(axis=0)

    return {
        "centroid": centroid,
        "rotation_matrix": R,
        "euler_xyz_deg": rotation_to_euler(R),
        "quaternion_wxyz": rotation_to_quaternion(R),
        "axis_angle": dict(
            zip(("axis", "angle_deg"), rotation_to_axis_angle(R))
        ),
        "eigenvalues": eigenvalues,
        "extents": extents,
    }


def compute_obb(points: np.ndarray) -> dict:
    """PCA 기반 Oriented Bounding Box."""
    pose = estimate_pose_pca(points)
    R = pose["rotation_matrix"]
    centroid = pose["centroid"]
    projected = (points - centroid) @ R
    obb_min, obb_max = projected.min(axis=0), projected.max(axis=0)
    obb_size = obb_max - obb_min
    obb_center_local = (obb_min + obb_max) / 2
    return {
        "center": centroid + R @ obb_center_local,
        "size": obb_size,
        "axes": R,
        "half_extents": obb_size / 2,
    }


# ================================================================
#  7. 정합 (Voxel downsample, Umeyama, ICP)
# ================================================================
def voxel_downsample(
    points: np.ndarray,
    voxel_size: float,
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.shape[0] == 0 or voxel_size <= 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    mins = keys.min(axis=0)
    shifted = keys - mins
    dims = shifted.max(axis=0) + 1
    flat = shifted[:, 0] * (dims[1] * dims[2]) + shifted[:, 1] * dims[2] + shifted[:, 2]
    _, inverse, counts = np.unique(flat, return_inverse=True, return_counts=True)
    n = len(counts)
    sum_pts = np.zeros((n, 3), dtype=np.float64)
    np.add.at(sum_pts, inverse, points)
    pts_out = sum_pts / counts[:, None]
    cols_out = None
    if colors is not None:
        sum_cols = np.zeros((n, 3), dtype=np.float64)
        np.add.at(sum_cols, inverse, colors.astype(np.float64))
        cols_out = (sum_cols / counts[:, None]).astype(np.uint8)
    return pts_out, cols_out


def _nearest_neighbor(
    src: np.ndarray, dst: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Brute-force NN (chunked, scipy 불필요)."""
    n_src = len(src)
    indices = np.empty(n_src, dtype=np.int64)
    distances = np.empty(n_src, dtype=np.float64)
    chunk = 2000
    for i in range(0, n_src, chunk):
        end = min(i + chunk, n_src)
        diff = dst[np.newaxis, :, :] - src[i:end, np.newaxis, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        min_idx = np.argmin(dist_sq, axis=1)
        indices[i:end] = min_idx
        distances[i:end] = np.sqrt(dist_sq[np.arange(end - i), min_idx])
    return indices, distances


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama: dst ≈ scale * R @ src + t.
    반환: (scale, R_3x3, t_3)
    """
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    var_src = np.sum(src_c ** 2) / n
    H = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.diag([1.0, 1.0, np.sign(d)])
    R = U @ D @ Vt
    scale = np.sum(S * np.diag(D)) / var_src if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def icp_point_to_point(
    src: np.ndarray,
    dst: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    max_correspondence_dist: float = 0.1,
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, dict]:
    """
    ICP Point-to-Point. src → dst 정합.
    반환: (scale, R_3x3, t_3, stats)
    """
    src_h = src.copy()
    total_scale, total_R, total_t = 1.0, np.eye(3), np.zeros(3)
    prev_error = float("inf")
    stats = {
        "iterations": 0, "final_rmse": 0.0,
        "inlier_ratio": 0.0, "converged": False,
    }

    for it in range(max_iterations):
        indices, distances = _nearest_neighbor(src_h, dst)
        mask = distances < max_correspondence_dist
        if mask.sum() < 10:
            break

        s, R, t = umeyama_alignment(src_h[mask], dst[indices[mask]], with_scale)
        src_h = s * (R @ src_h.T).T + t
        total_scale *= s
        total_R = R @ total_R
        total_t = s * R @ total_t + t

        mean_error = distances[mask].mean()
        rmse = np.sqrt((distances[mask] ** 2).mean())
        stats.update(
            iterations=it + 1,
            final_rmse=float(rmse),
            inlier_ratio=float(mask.sum() / len(mask)),
        )
        if abs(prev_error - mean_error) < tolerance:
            stats["converged"] = True
            break
        prev_error = mean_error

    return total_scale, total_R, total_t, stats


# ================================================================
#  8. 보조 함수 (COLMAP 씬 크롭, 스케일 추정)
# ================================================================
def extract_object_from_scene(
    scene_xyz: np.ndarray,
    center: np.ndarray,
    size: np.ndarray,
    margin: float = 0.3,
    scene_colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """씬에서 객체 주변 AABB 영역 추출."""
    half = size / 2 * (1 + margin)
    mask = np.all(
        (scene_xyz >= center - half) & (scene_xyz <= center + half), axis=1
    )
    return (
        scene_xyz[mask],
        scene_colors[mask] if scene_colors is not None else None,
        mask,
    )


def estimate_scale_ratio(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """bbox 대각선 비율로 스케일 추정. (pts_a * scale ≈ pts_b)"""
    diag_a = np.linalg.norm(pts_a.max(0) - pts_a.min(0))
    diag_b = np.linalg.norm(pts_b.max(0) - pts_b.min(0))
    return diag_b / diag_a if diag_a > 1e-10 else 1.0


# ================================================================
#  9. 출력 유틸
# ================================================================
def _print_bbox(xyz: np.ndarray, prefix: str = "") -> None:
    mn, mx = xyz.min(0), xyz.max(0)
    c, sz = xyz.mean(0), mx - mn
    print(f"{prefix}Centroid: ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
    print(f"{prefix}Size:     ({sz[0]:.4f}, {sz[1]:.4f}, {sz[2]:.4f})  "
          f"diag={np.linalg.norm(sz):.4f}")


def _print_pose(pose: dict, prefix: str = "") -> None:
    c = pose["centroid"]
    e = pose["euler_xyz_deg"]
    q = pose["quaternion_wxyz"]
    ax = pose["axis_angle"]
    ext = pose["extents"]
    print(f"{prefix}Position:  ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
    print(f"{prefix}Euler XYZ: ({e[0]:.1f}\u00b0, {e[1]:.1f}\u00b0, {e[2]:.1f}\u00b0)")
    print(f"{prefix}Quat wxyz: ({q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f})")
    print(f"{prefix}Axis-Angle: ({ax['axis'][0]:.3f}, {ax['axis'][1]:.3f}, "
          f"{ax['axis'][2]:.3f}) {ax['angle_deg']:.1f}\u00b0")
    print(f"{prefix}OBB size:  ({ext[0]:.4f}, {ext[1]:.4f}, {ext[2]:.4f})")


def _pose_to_dict(pose: dict, label: str = "") -> dict:
    return {
        "label": label,
        "position_m": pose["centroid"].tolist(),
        "rotation_matrix": pose["rotation_matrix"].tolist(),
        "euler_xyz_deg": pose["euler_xyz_deg"].tolist(),
        "quaternion_wxyz": pose["quaternion_wxyz"].tolist(),
        "axis_angle": {
            "axis": pose["axis_angle"]["axis"].tolist(),
            "angle_deg": float(pose["axis_angle"]["angle_deg"]),
        },
        "obb_extents": pose["extents"].tolist(),
        "eigenvalues": pose["eigenvalues"].tolist(),
    }


# ================================================================
#  10. 메인 파이프라인
# ================================================================
# PCA 부호 모호성 해결용 flip 조합
_FLIP_COMBOS = [
    np.diag([1.0, 1.0, 1.0]),
    np.diag([-1.0, -1.0, 1.0]),
    np.diag([-1.0, 1.0, -1.0]),
    np.diag([1.0, -1.0, -1.0]),
]


def run_pose_estimation(
    object_ply_path: str,
    scene_ply_path: str,
    mesh_ply_path: str,
    out_dir: str,
    voxel_size: float = 0.005,
    icp_max_dist: float = 0.1,
    icp_iterations: int = 50,
    T_multicam_colmap: Optional[np.ndarray] = None,
    multicam_ply_path: Optional[str] = None,
    calib_dir: Optional[str] = None,
    intrinsics_dir: Optional[str] = None,
    rgbd_dir: Optional[str] = None,
    frame_idx: int = 0,
) -> dict:
    """
    전체 파이프라인:
      1. PLY 로드 + 다운샘플링
      2. PCA pose (각 모델)
      3. SAM3D → gs2mesh 정합 (Scale + PCA + ICP)
      4. cam0 점군 생성 + 객체 추출 + gs2mesh 직접 정합
      5. 3개 PLY → cam0 좌표계 변환 + 저장
    """
    results = {}
    t0 = time.time()

    # ── Step 1: PLY 로드 ──────────────────────────────────────
    print("=" * 60)
    print(" Step 1: PLY 파일 로드")
    print("=" * 60)

    print(f"\n[1/3] SAM3D: {os.path.basename(object_ply_path)}")
    obj = load_ply(object_ply_path)
    obj_xyz, obj_colors = obj["xyz"], obj["colors"]
    print(f"  Points: {len(obj_xyz):,}  Color: {obj_colors is not None}")
    _print_bbox(obj_xyz, "  ")

    print(f"\n[2/3] COLMAP scene: {os.path.basename(scene_ply_path)}")
    scene = load_ply(scene_ply_path)
    scene_xyz, scene_colors = scene["xyz"], scene["colors"]
    print(f"  Points: {len(scene_xyz):,}  Color: {scene_colors is not None}")
    _print_bbox(scene_xyz, "  ")

    print(f"\n[3/3] gs2mesh: {os.path.basename(mesh_ply_path)}")
    mesh = load_ply(mesh_ply_path)
    mesh_xyz, mesh_colors = mesh["xyz"], mesh["colors"]
    print(f"  Vertices: {len(mesh_xyz):,}  Faces: {mesh['n_faces']:,}  "
          f"Color: {mesh_colors is not None}")
    _print_bbox(mesh_xyz, "  ")

    # ── Step 2: 다운샘플링 ────────────────────────────────────
    obj_diag = np.linalg.norm(obj_xyz.max(0) - obj_xyz.min(0))
    mesh_diag = np.linalg.norm(mesh_xyz.max(0) - mesh_xyz.min(0))
    scene_diag = np.linalg.norm(scene_xyz.max(0) - scene_xyz.min(0))

    obj_vox = max(obj_diag * 0.01, 0.001)
    mesh_vox = max(mesh_diag * 0.01, 0.001)
    scene_vox = max(scene_diag * 0.005, 0.01)

    print(f"\n{'=' * 60}")
    print(f" Step 2: Voxel downsample")
    print(f"{'=' * 60}")

    obj_ds, obj_ds_c = voxel_downsample(obj_xyz, obj_vox, obj_colors)
    mesh_ds, mesh_ds_c = voxel_downsample(mesh_xyz, mesh_vox, mesh_colors)
    scene_ds, scene_ds_c = voxel_downsample(scene_xyz, scene_vox, scene_colors)
    print(f"  SAM3D:   {len(obj_xyz):,} → {len(obj_ds):,} (voxel={obj_vox*1000:.1f}mm)")
    print(f"  gs2mesh: {len(mesh_xyz):,} → {len(mesh_ds):,} (voxel={mesh_vox*1000:.1f}mm)")
    print(f"  COLMAP:  {len(scene_xyz):,} → {len(scene_ds):,} (voxel={scene_vox*1000:.1f}mm)")

    # ── Step 3: PCA Pose ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" Step 3: PCA 기반 Pose 추정")
    print(f"{'=' * 60}")

    obj_pose = estimate_pose_pca(obj_ds)
    mesh_pose = estimate_pose_pca(mesh_ds)
    print("\n  [SAM3D]")
    _print_pose(obj_pose, "    ")
    print("\n  [gs2mesh]")
    _print_pose(mesh_pose, "    ")

    results["sam3d_pose"] = _pose_to_dict(obj_pose, "sam3d_object")
    results["gs2mesh_pose"] = _pose_to_dict(mesh_pose, "gs2mesh")

    # ── Step 4: COLMAP crop ───────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" Step 4: COLMAP 씬에서 객체 영역 추출")
    print(f"{'=' * 60}")

    mesh_center = mesh_pose["centroid"]
    mesh_size = mesh_xyz.max(0) - mesh_xyz.min(0)
    scene_crop, scene_crop_c, _ = extract_object_from_scene(
        scene_ds, mesh_center, mesh_size, margin=0.5
    )
    print(f"  gs2mesh 중심: ({mesh_center[0]:.3f}, {mesh_center[1]:.3f}, {mesh_center[2]:.3f})")
    print(f"  COLMAP crop: {len(scene_crop):,} / {len(scene_ds):,}")

    if len(scene_crop) > 100:
        crop_pose = estimate_pose_pca(scene_crop)
        print("\n  [COLMAP crop PCA]")
        _print_pose(crop_pose, "    ")
        results["colmap_crop_pose"] = _pose_to_dict(crop_pose, "colmap_crop")

    # ── Step 5: SAM3D → gs2mesh 정합 ─────────────────────────
    print(f"\n{'=' * 60}")
    print(f" Step 5: SAM3D → gs2mesh 정합 (Scale + PCA + ICP)")
    print(f"{'=' * 60}")

    scale_ratio = estimate_scale_ratio(obj_ds, mesh_ds)
    print(f"  스케일 비율: {scale_ratio:.4f}  "
          f"(SAM3D diag={obj_diag:.4f}  gs2mesh diag={mesh_diag:.4f})")

    # PCA 초기 정렬 + axis flip 최적화
    obj_centered = (obj_ds - obj_pose["centroid"]) * scale_ratio
    mesh_centered = mesh_ds - mesh_pose["centroid"]
    R_obj_pca = obj_pose["rotation_matrix"]
    R_mesh_pca = mesh_pose["rotation_matrix"]

    best_rmse, best_R_init = float("inf"), R_mesh_pca @ R_obj_pca.T
    for flip in _FLIP_COMBOS:
        R_cand = R_mesh_pca @ flip @ R_obj_pca.T
        rotated = (R_cand @ obj_centered.T).T
        step_s = max(1, len(rotated) // 2000)
        step_d = max(1, len(mesh_centered) // 2000)
        _, dists = _nearest_neighbor(rotated[::step_s], mesh_centered[::step_d])
        rmse = np.sqrt((dists ** 2).mean())
        if rmse < best_rmse:
            best_rmse, best_R_init = rmse, R_cand

    print(f"  PCA 초기 RMSE: {best_rmse:.4f}")
    obj_init = (best_R_init @ obj_centered.T).T + mesh_pose["centroid"]

    # ICP (스케일 고정)
    icp_dist = mesh_diag * 0.1
    scale_icp, R_icp, t_icp, icp_stats = icp_point_to_point(
        obj_init, mesh_ds,
        max_iterations=icp_iterations,
        max_correspondence_dist=icp_dist,
        with_scale=False,
    )
    print(f"\n  [ICP] iter={icp_stats['iterations']}  "
          f"converged={icp_stats['converged']}  "
          f"RMSE={icp_stats['final_rmse']:.6f}  "
          f"inlier={icp_stats['inlier_ratio']*100:.1f}%")

    obj_aligned = (R_icp @ obj_init.T).T + t_icp
    R_total = R_icp @ best_R_init
    t_total = R_icp @ mesh_pose["centroid"] + t_icp

    results["registration"] = {
        "method": "pca_align_icp",
        "source": "sam3d_object",
        "target": "gs2mesh",
        "scale_ratio": float(scale_ratio),
        "R_total": R_total.tolist(),
        "t_total": t_total.tolist(),
        "rmse": float(icp_stats["final_rmse"]),
        "inlier_ratio": float(icp_stats["inlier_ratio"]),
        "converged": icp_stats["converged"],
    }

    # ── Step 6: gs2mesh ↔ COLMAP crop 교차 검증 ──────────────
    if len(scene_crop) > 100:
        print(f"\n{'=' * 60}")
        print(f" Step 6: gs2mesh ↔ COLMAP crop 교차 검증")
        print(f"{'=' * 60}")
        _, nn_d = _nearest_neighbor(
            mesh_ds[::max(1, len(mesh_ds)//3000)],
            scene_crop[::max(1, len(scene_crop)//3000)],
        )
        cross_rmse = np.sqrt((nn_d ** 2).mean())
        print(f"  gs2mesh → COLMAP crop RMSE: {cross_rmse:.6f}")
        results["cross_validation"] = {
            "gs2mesh_vs_colmap_crop_rmse": float(cross_rmse),
        }

    # ── Step 7: cam0 점군 → 객체 추출 → COLMAP 정합 ──────────
    mc_xyz = None
    mc_colors = None

    # 7-A) 캘리브레이션 직접 사용
    if calib_dir and intrinsics_dir and rgbd_dir:
        print(f"\n{'=' * 60}")
        print(f" Step 7-A: 캘리브레이션 → cam0 점군 생성")
        print(f"{'=' * 60}")
        print(f"  calib_dir:      {calib_dir}")
        print(f"  intrinsics_dir: {intrinsics_dir}")
        print(f"  rgbd_dir:       {rgbd_dir}")
        print(f"  frame_idx:      {frame_idx}")

        # 캘리브레이션 정보 출력
        for ci in range(3):
            npz_p = os.path.join(intrinsics_dir, f"cam{ci}.npz")
            if os.path.exists(npz_p):
                K_i, D_i, ds_i = load_intrinsics(npz_p)
                print(f"  cam{ci}: fx={K_i[0,0]:.1f} fy={K_i[1,1]:.1f} "
                      f"cx={K_i[0,2]:.1f} cy={K_i[1,2]:.1f} ds={ds_i}")
        for ci in range(1, 3):
            npy_p = os.path.join(calib_dir, f"T_C0_C{ci}.npy")
            if os.path.exists(npy_p):
                t_vec = load_extrinsic(npy_p)[:3, 3]
                print(f"  T_C0_C{ci}: t=({t_vec[0]*1000:.1f}, "
                      f"{t_vec[1]*1000:.1f}, {t_vec[2]*1000:.1f}) mm")

        print(f"\n  [RGB-D → cam0 융합]")
        mc_xyz, mc_colors = build_cam0_cloud(
            calib_dir, intrinsics_dir, rgbd_dir,
            frame_idx=frame_idx, z_min=0.1, z_max=0.80, stride=1,
        )
        print(f"\n  cam0 총 점수: {len(mc_xyz):,}")
        _print_bbox(mc_xyz, "  ")

        os.makedirs(out_dir, exist_ok=True)
        cam0_ply = os.path.join(out_dir, "cam0_fused_from_calib.ply")
        save_step = max(1, len(mc_xyz) // 100000)
        save_ply(
            cam0_ply, mc_xyz[::save_step],
            mc_colors[::save_step] if mc_colors is not None else None,
        )
        print(f"  [SAVE] {cam0_ply} ({len(mc_xyz[::save_step]):,} pts)")

        results["cam0_cloud"] = {
            "method": "direct_calibration",
            "n_points": len(mc_xyz),
            "frame_idx": frame_idx,
        }

    # 7-B) 기생성 멀티캠 PLY
    elif multicam_ply_path:
        print(f"\n{'=' * 60}")
        print(f" Step 7-B: 멀티캠 PLY 로드 (cam0 프레임)")
        print(f"{'=' * 60}")
        multicam = load_ply(multicam_ply_path)
        mc_xyz, mc_colors = multicam["xyz"], multicam["colors"]
        print(f"  Points: {len(mc_xyz):,}")
        _print_bbox(mc_xyz, "  ")

    # 7-C) 객체 추출 + gs2mesh 직접 정합
    if mc_xyz is not None and len(mc_xyz) > 0 and T_multicam_colmap is None:
        print(f"\n  [Step 7-C: 객체 추출 → gs2mesh 직접 정합]")

        # RANSAC 테이블 제거 + BFS 클러스터링
        obj_cam0, obj_cam0_c = extract_object_from_cloud(
            mc_xyz, mc_colors,
            plane_threshold=0.008,
            min_height_above_plane=0.01,
            min_cluster_points=500,
            cluster_voxel=0.008,
        )
        print(f"  cam0 전체: {len(mc_xyz):,} → 객체: {len(obj_cam0):,}")
        _print_bbox(obj_cam0, "  객체 ")

        os.makedirs(out_dir, exist_ok=True)
        save_ply(os.path.join(out_dir, "cam0_object_extracted.ply"),
                 obj_cam0, obj_cam0_c)

        # 스케일 추정: cam0_obj_diag / gs2mesh_diag
        cam0_obj_diag = np.linalg.norm(obj_cam0.max(0) - obj_cam0.min(0))
        scale_m2c = cam0_obj_diag / mesh_diag
        print(f"\n  [스케일] cam0 obj diag={cam0_obj_diag*1000:.1f}mm  "
              f"gs2mesh diag={mesh_diag:.4f}  scale={scale_m2c:.6f}")

        mesh_scaled = mesh_ds * scale_m2c

        # 다운샘플 (정합용)
        ovox = max(cam0_obj_diag * 0.015, 0.002)
        mvox = max(mesh_diag * scale_m2c * 0.015, 0.002)
        obj_cam0_ds, _ = voxel_downsample(obj_cam0, ovox)
        mesh_sc_ds, _ = voxel_downsample(mesh_scaled, mvox)
        print(f"  cam0 객체 DS: {len(obj_cam0):,} → {len(obj_cam0_ds):,}")
        print(f"  gs2mesh DS:   {len(mesh_scaled):,} → {len(mesh_sc_ds):,}")

        # PCA 초기 정렬 (object↔object)
        pca_o = estimate_pose_pca(obj_cam0_ds)
        pca_m = estimate_pose_pca(mesh_sc_ds)
        R_o, R_m = pca_o["rotation_matrix"], pca_m["rotation_matrix"]
        o_cen = obj_cam0_ds - pca_o["centroid"]
        m_cen = mesh_sc_ds - pca_m["centroid"]

        best_br_rmse, best_R_br = float("inf"), R_o @ R_m.T
        for flip in _FLIP_COMBOS:
            R_c = R_o @ flip @ R_m.T
            rotated = (R_c @ m_cen.T).T
            sa, sb = max(1, len(rotated)//2000), max(1, len(o_cen)//2000)
            _, dd = _nearest_neighbor(rotated[::sa], o_cen[::sb])
            rmse_c = np.sqrt((dd ** 2).mean())
            if rmse_c < best_br_rmse:
                best_br_rmse, best_R_br = rmse_c, R_c

        print(f"\n  [PCA 초기] RMSE={best_br_rmse*1000:.1f} mm")
        mesh_rotated = (best_R_br @ m_cen.T).T + pca_o["centroid"]

        # ICP (스케일 고정)
        br_icp_dist = cam0_obj_diag * 0.20
        s_br, R_br, t_br, stats_br = icp_point_to_point(
            mesh_rotated, obj_cam0_ds,
            max_iterations=100,
            max_correspondence_dist=br_icp_dist,
            with_scale=False,
        )
        print(f"  [ICP] iter={stats_br['iterations']}  "
              f"converged={stats_br['converged']}  "
              f"RMSE={stats_br['final_rmse']*1000:.2f}mm  "
              f"inlier={stats_br['inlier_ratio']*100:.1f}%")

        # Similarity transform 합성: cam0_pt = scale * R_total @ mesh_pt + t_total
        total_scale = scale_m2c
        total_R_br = R_br @ best_R_br
        total_t_br = (
            R_br @ (pca_o["centroid"] - best_R_br @ pca_m["centroid"]) + t_br
        )

        # 검증
        mesh_verify = total_scale * (total_R_br @ mesh_ds.T).T + total_t_br
        sv = max(1, len(mesh_verify) // 3000)
        sv2 = max(1, len(obj_cam0_ds) // 3000)
        _, vnn = _nearest_neighbor(mesh_verify[::sv], obj_cam0_ds[::sv2])
        v_rmse = np.sqrt((vnn ** 2).mean())
        print(f"  [검증 RMSE: {v_rmse*1000:.1f} mm]")

        # T_cam0_colmap 저장
        T_cam0_colmap = np.eye(4, dtype=np.float64)
        T_cam0_colmap[:3, :3] = total_scale * total_R_br
        T_cam0_colmap[:3, 3] = total_t_br

        T_save_path = os.path.join(out_dir, "T_cam0_colmap.npy")
        np.save(T_save_path, T_cam0_colmap)
        print(f"  [SAVE] {T_save_path}")

        T_multicam_colmap = T_cam0_colmap

        results["bridge_colmap_to_cam0"] = {
            "method": "object_direct_registration",
            "scale": float(total_scale),
            "cam0_obj_diag_mm": float(cam0_obj_diag * 1000),
            "gs2mesh_diag": float(mesh_diag),
            "cam0_obj_points": len(obj_cam0),
            "rmse_m": float(v_rmse),
            "icp_converged": stats_br["converged"],
            "icp_inlier_ratio": stats_br["inlier_ratio"],
        }

    # ── Step 8: cam0 좌표계 Pose ──────────────────────────────
    if T_multicam_colmap is not None:
        print(f"\n{'=' * 60}")
        print(f" Step 8: cam0 좌표계 최종 Pose")
        print(f"{'=' * 60}")

        sR = T_multicam_colmap[:3, :3]
        t_mc = T_multicam_colmap[:3, 3]
        mc_scale = np.linalg.norm(sR, axis=0).mean()
        R_mc = sR / mc_scale

        pos_mc = mc_scale * R_mc @ mesh_pose["centroid"] + t_mc
        R_obj_mc = R_mc @ mesh_pose["rotation_matrix"]
        if np.linalg.det(R_obj_mc) < 0:
            R_obj_mc[:, 2] *= -1

        euler_mc = rotation_to_euler(R_obj_mc)
        quat_mc = rotation_to_quaternion(R_obj_mc)
        extents_cam0 = mesh_pose["extents"] * mc_scale

        print(f"  Scale:    {mc_scale:.6f}")
        print(f"  Position: ({pos_mc[0]*1000:.1f}, {pos_mc[1]*1000:.1f}, "
              f"{pos_mc[2]*1000:.1f}) mm")
        print(f"  Euler:    ({euler_mc[0]:.1f}\u00b0, {euler_mc[1]:.1f}\u00b0, "
              f"{euler_mc[2]:.1f}\u00b0)")
        print(f"  OBB:      ({extents_cam0[0]*1000:.1f}, {extents_cam0[1]*1000:.1f}, "
              f"{extents_cam0[2]*1000:.1f}) mm")

        results["multicam_pose"] = {
            "coordinate_frame": "cam0 (from calibration T_C0_Ci)",
            "position_m": pos_mc.tolist(),
            "position_mm": (pos_mc * 1000).tolist(),
            "rotation_matrix": R_obj_mc.tolist(),
            "euler_xyz_deg": euler_mc.tolist(),
            "quaternion_wxyz": quat_mc.tolist(),
            "obb_extents_m": extents_cam0.tolist(),
            "obb_extents_mm": (extents_cam0 * 1000).tolist(),
            "transform_scale": float(mc_scale),
        }

    # ── Step 9: 최종 요약 ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" 최종 결과 요약")
    print(f"{'=' * 60}")

    print(f"\n  [gs2mesh Pose (COLMAP 좌표계)]")
    c = mesh_pose["centroid"]
    e = mesh_pose["euler_xyz_deg"]
    print(f"  Position:  ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
    print(f"  Euler XYZ: ({e[0]:.1f}\u00b0, {e[1]:.1f}\u00b0, {e[2]:.1f}\u00b0)")

    if T_multicam_colmap is not None and "multicam_pose" in results:
        pm = results["multicam_pose"]["position_mm"]
        em = results["multicam_pose"]["euler_xyz_deg"]
        ex = results["multicam_pose"].get("obb_extents_mm", [0, 0, 0])
        print(f"\n  [cam0 좌표계 — 캘리브레이션 기준]")
        print(f"  Position: ({pm[0]:.1f}, {pm[1]:.1f}, {pm[2]:.1f}) mm")
        print(f"  Euler:    ({em[0]:.1f}\u00b0, {em[1]:.1f}\u00b0, {em[2]:.1f}\u00b0)")
        print(f"  OBB:      ({ex[0]:.1f}, {ex[1]:.1f}, {ex[2]:.1f}) mm")

    # ── Step 10: PLY 저장 ─────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    aligned_path = os.path.join(out_dir, "sam3d_aligned_to_colmap.ply")
    save_ply(aligned_path, obj_aligned, obj_ds_c)
    print(f"\n[SAVE] {aligned_path}")

    if T_multicam_colmap is not None:
        sR = T_multicam_colmap[:3, :3]
        t_sv = T_multicam_colmap[:3, 3]

        mesh_cam0 = (sR @ mesh_ds.T).T + t_sv
        mesh_cam0_path = os.path.join(out_dir, "gs2mesh_in_cam0.ply")
        save_ply(mesh_cam0_path, mesh_cam0, mesh_ds_c)
        print(f"[SAVE] {mesh_cam0_path}")

        sam3d_cam0 = (sR @ obj_aligned.T).T + t_sv
        sam3d_cam0_path = os.path.join(out_dir, "sam3d_in_cam0.ply")
        save_ply(sam3d_cam0_path, sam3d_cam0, obj_ds_c)
        print(f"[SAVE] {sam3d_cam0_path}")

        scene_cam0 = (sR @ scene_ds.T).T + t_sv
        scene_cam0_path = os.path.join(out_dir, "colmap_scene_in_cam0.ply")
        save_ply(scene_cam0_path, scene_cam0, scene_ds_c)
        print(f"[SAVE] {scene_cam0_path}")

        results["saved_cam0_plys"] = {
            "gs2mesh_in_cam0": os.path.abspath(mesh_cam0_path),
            "sam3d_in_cam0": os.path.abspath(sam3d_cam0_path),
            "colmap_scene_in_cam0": os.path.abspath(scene_cam0_path),
        }

    results["elapsed_sec"] = round(time.time() - t0, 2)
    out_json = os.path.join(out_dir, "pose_estimation_results.json")
    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"[SAVE] {out_json}")

    return results


# ================================================================
#  Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step6: ArUco 큐브 캘리브레이션 기반 3D Pose 추정 + PLY 정합"
    )

    # PLY 입력
    parser.add_argument("--object_ply", required=True,
                        help="SAM3D 객체 PLY (tiger.ply)")
    parser.add_argument("--scene_ply", required=True,
                        help="COLMAP 전체 씬 PLY (point_cloud_cleaned.ply)")
    parser.add_argument("--mesh_ply", required=True,
                        help="gs2mesh 객체 메시 PLY")

    # 캘리브레이션 직접 사용 (권장)
    parser.add_argument("--calib_dir", type=str, default=None,
                        help="캘리브레이션 폴더 (T_C0_C1.npy, T_C0_C2.npy)")
    parser.add_argument("--intrinsics_dir", type=str, default=None,
                        help="내부파라미터 폴더 (cam0.npz, cam1.npz, cam2.npz)")
    parser.add_argument("--rgbd_dir", type=str, default=None,
                        help="RGB-D 프레임 폴더 (cam0/, cam1/, cam2/)")
    parser.add_argument("--frame_idx", type=int, default=0,
                        help="사용할 프레임 인덱스")

    # 대안
    parser.add_argument("--multicam_ply", type=str, default=None,
                        help="멀티캠 depth PLY (cam0 프레임)")
    parser.add_argument("--T_multicam_colmap", type=str, default=None,
                        help="COLMAP→cam0 변환 4x4 npy (기계산)")

    # 파라미터
    parser.add_argument("--out_dir", default=None,
                        help="출력 폴더 (기본: object_ply/../pose_out/)")
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--icp_max_dist", type=float, default=0.1)
    parser.add_argument("--icp_iterations", type=int, default=50)

    args = parser.parse_args()

    out_dir = args.out_dir or "./output"

    T_mc = None
    if args.T_multicam_colmap:
        T_mc = np.load(args.T_multicam_colmap).astype(np.float64)
        print(f"[INFO] T_multicam_colmap: {args.T_multicam_colmap}")

    run_pose_estimation(
        object_ply_path=args.object_ply,
        scene_ply_path=args.scene_ply,
        mesh_ply_path=args.mesh_ply,
        out_dir=out_dir,
        voxel_size=args.voxel_size,
        icp_max_dist=args.icp_max_dist,
        icp_iterations=args.icp_iterations,
        T_multicam_colmap=T_mc,
        multicam_ply_path=args.multicam_ply,
        calib_dir=args.calib_dir,
        intrinsics_dir=args.intrinsics_dir,
        rgbd_dir=args.rgbd_dir,
        frame_idx=args.frame_idx,
    )


if __name__ == "__main__":
    main()
