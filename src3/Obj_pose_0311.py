#!/usr/bin/env python3
"""
Obj_pose_0311.py - 멀티뷰 카메라 기반 물체 6-DOF 포즈 추정
============================================================
파이프라인 A (ICP 기반 - 기본):
  1. 데이터 로드 (카메라 내부/외부 파라미터, RGB-D 이미지)
  2. 각 카메라에서 depth 역투영 -> 3D 점군 생성
  3. cam0 좌표계로 멀티뷰 점군 통합
  4. GLB 참조 모델 로드 및 점군 변환
  5. 스케일 정합 (bounding box 비교)
  6. 초기 정합 (FPFH + RANSAC 전역 정합)
  7. 정밀 정합 (Point-to-Plane ICP, 다단계)
  8. 포즈 추출 (R, t, euler, quaternion)

파이프라인 B (SAM3D 통합 - --sam3d_dir 지정 시):
  SAM3D 정규화 좌표계의 rotation/translation/scale을 읽어
  카메라 좌표계(cam_i)를 거쳐 cam0 월드좌표계로 변환.
  선택적으로 관측 점군과 ICP 정밀 정합 수행.

  SAM3D 좌표 변환 체인:
    P_canonical (SAM3D 정규화)
      ↓  scale * R_sam3d @ P + t_sam3d
    P_sam3d (SAM3D 출력)
      ↓  T_gl2cv (OpenGL Y-up → OpenCV Y-down)
    P_cam_i (카메라_i 좌표계)
      ↓  T_C0_Ci
    P_cam0 (월드좌표계)

좌표계: cam0 (OpenCV: X-right, Y-down, Z-forward)

사용법:
  # --- ICP 기반 ---
  python Obj_pose_0311.py
  python Obj_pose_0311.py --frame 3 --ref_model data/reference_knife.glb
  python Obj_pose_0311.py --frame 3 --seg_mode hsv
  python Obj_pose_0311.py --frame 0 --ref_length_mm 165

  # --- SAM3D 통합 ---
  python Obj_pose_0311.py --sam3d_dir path/to/sam3d_output --sam3d_cam 0
  python Obj_pose_0311.py --sam3d_dir path/to/sam3d_output --sam3d_cam 1 --sam3d_refine_icp
  python Obj_pose_0311.py --sam3d_ply sam3d_knife.ply --sam3d_pose pose.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CAP_DIR = _THIS_DIR / "data/object_capture"
_DEFAULT_CAL_DIR = _THIS_DIR / "data/cube_session_01/calib_out_cube"
_DEFAULT_INT_DIR = _THIS_DIR / "data/_intrinsics"
_DEFAULT_OUT_DIR = _THIS_DIR / "Obj_pose_0311_output"
_DEFAULT_REF_MODEL = _THIS_DIR / "data/reference_knife.glb"

# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraCalibration:
    """카메라 캘리브레이션 데이터."""
    camera_id: int
    K: np.ndarray               # 3x3 intrinsic
    D: np.ndarray               # 왜곡 계수
    depth_scale: float           # depth -> 미터 변환 스케일
    T_cam0_cam: np.ndarray       # 4x4 cam -> cam0 변환 행렬


@dataclass
class FrameData:
    """한 카메라의 한 프레임 RGB-D 데이터."""
    camera_id: int
    rgb_path: Path
    depth_path: Path
    bgr: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None


@dataclass
class MultiViewSession:
    """멀티뷰 캡처 세션."""
    camera_ids: Tuple[int, ...]
    pad: int
    calibrations: Dict[int, CameraCalibration] = field(default_factory=dict)
    frames: Dict[int, FrameData] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 1: 데이터 로드
# ---------------------------------------------------------------------------

def load_intrinsics(intrin_dir: Path, camera_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """카메라 내부 파라미터 로드 (color_K, color_D, depth_scale)."""
    npz_path = intrin_dir / f"cam{camera_id}.npz"
    data = np.load(str(npz_path), allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64) if "color_D" in data else np.zeros(5, dtype=np.float64)
    depth_scale = float(data.get("depth_scale_m_per_unit", 0.001))
    return K, D, depth_scale


def load_extrinsics(calib_dir: Path, camera_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    """카메라 외부 파라미터 로드 (T_C0_Ci 변환 행렬)."""
    transforms: Dict[int, np.ndarray] = {}
    for cid in camera_ids:
        if cid == 0:
            transforms[cid] = np.eye(4, dtype=np.float64)
        else:
            npy_path = calib_dir / f"T_C0_C{cid}.npy"
            transforms[cid] = np.load(str(npy_path)).astype(np.float64)
    return transforms


def discover_cameras(capture_dir: Path) -> Tuple[int, ...]:
    """캡처 디렉토리에서 cam* 폴더를 찾아 카메라 ID 목록을 반환."""
    camera_ids = []
    for d in sorted(glob.glob(str(capture_dir / "cam*"))):
        try:
            cid = int(Path(d).name.replace("cam", ""))
        except ValueError:
            continue
        if glob.glob(str(Path(d) / "rgb_*.jpg")):
            camera_ids.append(cid)
    if not camera_ids:
        raise RuntimeError(f"cam* 폴더를 찾지 못함: {capture_dir}")
    return tuple(camera_ids)


def frame_pad(capture_dir: Path, camera_id: int) -> int:
    """프레임 번호 패딩 자릿수 파악."""
    files = glob.glob(str(capture_dir / f"cam{camera_id}" / "rgb_*.jpg"))
    if not files:
        return 6
    stem = Path(files[0]).stem.replace("rgb_", "")
    return len(stem)


def load_session(args: argparse.Namespace) -> MultiViewSession:
    """캘리브레이션, 프레임 데이터를 모두 로드하여 세션 구성."""
    capture_dir = Path(args.capture_dir)
    calib_dir = Path(args.calib_dir)
    intrin_dir = Path(args.intrinsics_dir)

    camera_ids = discover_cameras(capture_dir)
    pad = frame_pad(capture_dir, camera_ids[0])
    transforms = load_extrinsics(calib_dir, camera_ids)

    session = MultiViewSession(camera_ids=camera_ids, pad=pad)

    for cid in camera_ids:
        K, D, depth_scale = load_intrinsics(intrin_dir, cid)
        session.calibrations[cid] = CameraCalibration(
            camera_id=cid,
            K=K, D=D,
            depth_scale=depth_scale,
            T_cam0_cam=transforms[cid],
        )

    fid = f"{args.frame:0{pad}d}"
    for cid in camera_ids:
        rgb_path = capture_dir / f"cam{cid}" / f"rgb_{fid}.jpg"
        depth_path = capture_dir / f"cam{cid}" / f"depth_{fid}.png"
        bgr = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.exists() else None
        session.frames[cid] = FrameData(
            camera_id=cid,
            rgb_path=rgb_path,
            depth_path=depth_path,
            bgr=bgr,
            depth=depth,
        )
        status = "OK" if bgr is not None and depth is not None else "MISSING"
        print(f"    cam{cid}: {status}  rgb={rgb_path.name}  depth={depth_path.name}")

    return session


# ---------------------------------------------------------------------------
# Step 2: Depth -> 3D 점군 생성
# ---------------------------------------------------------------------------

def depth_to_pointcloud(
    depth_u16: np.ndarray,
    bgr: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    depth_scale: float,
    mask: Optional[np.ndarray] = None,
    z_min: float = 0.1,
    z_max: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """depth + color 이미지 -> 3D 점군 + 색상.

    Args:
        depth_u16: uint16 depth 이미지
        bgr: BGR 컬러 이미지
        K: 3x3 카메라 내부 행렬
        D: 왜곡 계수
        depth_scale: depth 단위 -> 미터 변환 스케일
        mask: 마스크 (True인 영역만 점군 생성)
        z_min, z_max: 유효 깊이 범위 (미터)

    Returns:
        points (N,3), colors (N,3) in camera coordinate
    """
    h, w = depth_u16.shape[:2]
    if mask is not None and mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)

    vg, ug = np.mgrid[0:h, 0:w]
    z = depth_u16.astype(np.float64) * depth_scale

    valid = (z > z_min) & (z < z_max)
    if mask is not None:
        valid &= mask

    if not valid.any():
        return np.empty((0, 3), np.float64), np.empty((0, 3), np.float64)

    z_val = z[valid]
    u_val = ug[valid].astype(np.float64)
    v_val = vg[valid].astype(np.float64)

    # 왜곡 보정 후 정규화 좌표
    pts_2d = np.column_stack([u_val, v_val]).reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts_2d, K, D).reshape(-1, 2)

    x = undist[:, 0] * z_val
    y = undist[:, 1] * z_val
    points = np.column_stack([x, y, z_val])

    # 색상 추출 (BGR -> RGB, 0~1)
    colors = bgr[v_val.astype(int), u_val.astype(int)][:, ::-1] / 255.0

    return points, colors


# ---------------------------------------------------------------------------
# Step 3: 멀티뷰 점군 통합 (cam0 좌표계)
# ---------------------------------------------------------------------------

def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """4x4 변환 행렬로 점군 변환."""
    if len(points) == 0:
        return points
    return points @ T[:3, :3].T + T[:3, 3]


def merge_multiview_pointclouds(
    session: MultiViewSession,
    z_min: float = 0.1,
    z_max: float = 1.5,
    masks: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """멀티뷰 depth를 cam0 좌표계로 변환 및 통합.

    Args:
        session: 멀티뷰 세션
        z_min, z_max: 유효 깊이 범위
        masks: 카메라별 세그멘테이션 마스크 (None이면 전체)

    Returns:
        merged_points (N,3), merged_colors (N,3) in cam0 coordinate
    """
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    for cid in session.camera_ids:
        frame = session.frames[cid]
        calib = session.calibrations[cid]
        if frame.bgr is None or frame.depth is None:
            print(f"    cam{cid}: 이미지 없음, 건너뜀")
            continue

        mask = masks.get(cid) if masks is not None else None
        points_cam, colors = depth_to_pointcloud(
            frame.depth, frame.bgr,
            calib.K, calib.D, calib.depth_scale,
            mask=mask, z_min=z_min, z_max=z_max,
        )
        if len(points_cam) == 0:
            print(f"    cam{cid}: 유효 점군 없음")
            continue

        # cam -> cam0 좌표계 변환
        points_cam0 = transform_points(points_cam, calib.T_cam0_cam)
        all_points.append(points_cam0)
        all_colors.append(colors)
        print(f"    cam{cid}: {len(points_cam0):,} pts -> cam0")

    if not all_points:
        raise RuntimeError("모든 카메라에서 점군 생성 실패")

    return np.concatenate(all_points), np.concatenate(all_colors)


# ---------------------------------------------------------------------------
# 세그멘테이션: 물체 영역 추출
# ---------------------------------------------------------------------------

def segment_by_depth_roi(
    session: MultiViewSession,
    z_min: float,
    z_max: float,
    plane_threshold: float = 0.005,
    cluster_eps: float = 0.005,
    cluster_min_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """depth ROI + 테이블 제거 + DBSCAN 최대 클러스터 선택."""
    import open3d as o3d

    # 전체 depth 점군 수집
    points, colors = merge_multiview_pointclouds(session, z_min, z_max)

    # RANSAC 평면(테이블) 제거
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(points) < 50:
        raise RuntimeError("평면 제거를 수행하기에 점군이 너무 적음")
    _, inliers = pcd.segment_plane(
        distance_threshold=plane_threshold, ransac_n=3, num_iterations=1000)
    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[inliers] = True
    points = points[~inlier_mask]
    colors = colors[~inlier_mask]
    print(f"    테이블 제거 후: {len(points):,} pts")

    # IQR 이상치 제거
    points, colors = statistical_outlier_removal(points, colors)

    # DBSCAN 최대 클러스터
    points, colors = keep_largest_cluster(points, colors, cluster_eps, cluster_min_samples)

    return points, colors


def compute_hsv_mask(
    bgr: np.ndarray,
    h_range: Sequence[int] = (15, 35),
    s_min: int = 80,
    v_min: int = 80,
) -> np.ndarray:
    """HSV 컬러 기반 마스크 생성 (노란색 손잡이 검출)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= h_range[0]) & (hsv[:, :, 0] <= h_range[1]) &
        (hsv[:, :, 1] >= s_min) &
        (hsv[:, :, 2] >= v_min)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)


def largest_connected_component(mask: np.ndarray, min_area: int = 500) -> Optional[np.ndarray]:
    """마스크에서 가장 큰 연결 요소 추출."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    best_label, best_area = -1, 0
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_label, best_area = lbl, area
    return (labels == best_label) if best_label >= 0 else None


def segment_by_hsv_anchor(
    session: MultiViewSession,
    z_min: float, z_max: float,
    h_range: Sequence[int] = (15, 35),
    s_min: int = 80, v_min: int = 80,
    min_area: int = 500,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """HSV 앵커(노란 손잡이) + depth 기반 물체 마스크 세그멘테이션.

    Returns:
        points, colors, anchor_center (cam0 좌표)
    """
    anchor_points_all: List[np.ndarray] = []
    masks: Dict[int, np.ndarray] = {}

    for cid in session.camera_ids:
        frame = session.frames[cid]
        calib = session.calibrations[cid]
        if frame.bgr is None or frame.depth is None:
            continue

        # 1) 노란색 영역 검출
        yellow_mask = compute_hsv_mask(frame.bgr, h_range, s_min, v_min)
        anchor_mask = largest_connected_component(yellow_mask, min_area)
        if anchor_mask is None:
            continue

        # 2) 앵커 3D 점 수집
        anchor_pts, _ = depth_to_pointcloud(
            frame.depth, frame.bgr, calib.K, calib.D, calib.depth_scale,
            mask=anchor_mask, z_min=z_min, z_max=z_max)
        if len(anchor_pts) == 0:
            continue
        anchor_pts_cam0 = transform_points(anchor_pts, calib.T_cam0_cam)
        anchor_points_all.append(anchor_pts_cam0)

        # 3) 앵커 depth 기반으로 물체 마스크 확장
        depth_m = frame.depth.astype(np.float64) * calib.depth_scale
        anchor_depths = depth_m[anchor_mask & (depth_m > z_min) & (depth_m < z_max)]
        if len(anchor_depths) == 0:
            masks[cid] = anchor_mask
            continue

        anchor_depth = float(np.median(anchor_depths))
        # 앵커 주변 depth 범위로 물체 영역 확장
        depth_ok = (
            (frame.depth > 0) &
            (depth_m > max(z_min, anchor_depth - 0.020)) &
            (depth_m < min(z_max, anchor_depth + 0.020))
        )
        # 앵커 주변 ROI 제한
        ys, xs = np.where(anchor_mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            h, w = frame.depth.shape[:2]
            roi_mask = np.zeros((h, w), dtype=bool)
            margin = 120
            roi_mask[max(0, cy - margin):min(h, cy + margin),
                     max(0, cx - 3 * margin):min(w, cx + 3 * margin)] = True
            object_mask = depth_ok & roi_mask
        else:
            object_mask = depth_ok

        # 앵커와 연결된 component만 유지
        object_mask = object_mask | anchor_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        object_mask = cv2.morphologyEx(
            object_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
        comp = largest_connected_component(object_mask, min_area // 2)
        if comp is not None:
            object_mask = comp

        masks[cid] = object_mask
        print(f"    cam{cid}: 앵커 {len(anchor_pts):,} pts, 마스크 {object_mask.sum():,} px")

    if not anchor_points_all:
        raise RuntimeError("노란 손잡이 앵커 검출 실패 (모든 카메라)")

    anchor_center = np.concatenate(anchor_points_all).mean(axis=0)
    print(f"    앵커 중심 (cam0): "
          f"({anchor_center[0]*1000:.1f}, {anchor_center[1]*1000:.1f}, {anchor_center[2]*1000:.1f}) mm")

    # 마스크 기반 점군 생성
    if masks:
        points, colors = merge_multiview_pointclouds(session, z_min, z_max, masks=masks)
    else:
        raise RuntimeError("물체 마스크 생성 실패")

    # 이상치 제거 + 클러스터링
    points, colors = statistical_outlier_removal(points, colors)
    points, colors = keep_largest_cluster(points, colors, eps=0.005, min_samples=10)

    return points, colors, anchor_center


# ---------------------------------------------------------------------------
# 점군 후처리
# ---------------------------------------------------------------------------

def statistical_outlier_removal(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    std_ratio: float = 1.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """IQR + 거리 기반 이상치 제거."""
    n = len(points)
    if n < 10:
        return points, colors

    mask = np.ones(n, dtype=bool)
    for axis in range(3):
        vals = points[:, axis]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        mask &= (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)

    center = points[mask].mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    mean_d, std_d = dists[mask].mean(), dists[mask].std()
    mask &= dists < mean_d + std_ratio * std_d

    print(f"    SOR: {n:,} -> {mask.sum():,}")
    return points[mask], (colors[mask] if colors is not None else None)


def keep_largest_cluster(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    eps: float = 0.005,
    min_samples: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """DBSCAN으로 가장 큰 클러스터만 유지."""
    from sklearn.cluster import DBSCAN

    if len(points) < min_samples:
        raise RuntimeError("클러스터링을 수행하기에 점군이 너무 적음")

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    valid = labels[labels >= 0]
    if len(valid) == 0:
        raise RuntimeError("DBSCAN 클러스터 없음")

    unique, counts = np.unique(valid, return_counts=True)
    best = unique[np.argmax(counts)]
    mask = labels == best
    print(f"    DBSCAN: {len(unique)} 클러스터, 최대={mask.sum():,} pts 채택")
    return points[mask], (colors[mask] if colors is not None else None)


# ---------------------------------------------------------------------------
# Step 4: GLB 참조 모델 로드
# ---------------------------------------------------------------------------

def load_reference_model(path: str, num_samples: int = 50000) -> "open3d.geometry.PointCloud":
    """GLB/PLY/OBJ 파일을 Open3D PointCloud로 로드.

    GLB(OpenGL Y-up) -> OpenCV(Y-down) 좌표계 변환 포함.
    중심점을 원점으로 이동.
    """
    import open3d as o3d

    ref_path = Path(path)
    ext = ref_path.suffix.lower()

    if ext == ".ply":
        mesh = o3d.io.read_triangle_mesh(str(ref_path))
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
        else:
            pcd = o3d.io.read_point_cloud(str(ref_path))
    elif ext in {".glb", ".gltf", ".obj"}:
        import trimesh

        scene = trimesh.load(str(ref_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            mesh_tm = (scene.to_geometry()
                       if hasattr(scene, "to_geometry")
                       else scene.dump(concatenate=True))
        else:
            mesh_tm = scene

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_tm.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_tm.faces)
        if mesh_tm.visual is not None and hasattr(mesh_tm.visual, "vertex_colors"):
            vc = mesh_tm.visual.vertex_colors[:, :3] / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vc)
        mesh_o3d.compute_vertex_normals()
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=num_samples)
    else:
        raise ValueError(f"지원하지 않는 참조 모델 형식: {ext}")

    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise RuntimeError(f"참조 모델이 비어있음: {path}")

    # OpenGL(Y-up) -> OpenCV(Y-down): Y, Z 반전
    if ext in {".glb", ".gltf"}:
        points[:, 1] *= -1.0
        points[:, 2] *= -1.0
        print("  [좌표계 변환] OpenGL(Y-up) -> OpenCV(Y-down): Y,Z 반전")

    # 중심점을 원점으로
    centroid = points.mean(axis=0)
    points -= centroid
    pcd.points = o3d.utility.Vector3dVector(points)

    bbox = points.max(axis=0) - points.min(axis=0)
    print(f"  참조 모델: {len(points):,} pts")
    print(f"  bbox: {bbox[0]:.4f} x {bbox[1]:.4f} x {bbox[2]:.4f}")
    return pcd


# ---------------------------------------------------------------------------
# SAM3D 좌표계 변환 모듈
# ---------------------------------------------------------------------------
# SAM3D 좌표 변환 체인:
#
#   [SAM3D canonical space]
#       ↓  T_sam3d = scale * R_6d @ P + translation
#   [SAM3D output space]  (OpenGL: Y-up, Z-toward-viewer)
#       ↓  T_gl2cv = diag(1, -1, -1)  (Y, Z 반전)
#   [Camera_i space]  (OpenCV: Y-down, Z-forward)
#       ↓  T_C0_Ci  (외부 캘리브레이션)
#   [cam0 world space]  (최종 포즈)
# ---------------------------------------------------------------------------

# OpenGL(Y-up) ↔ OpenCV(Y-down) 변환 행렬
T_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])  # 4x4
T_CV2GL = T_GL2CV.copy()  # 자기 역행렬


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    """6D rotation representation → 3x3 회전 행렬.

    Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"
    입력: (6,) 벡터 [a1(3), a2(3)]
    출력: (3,3) 정규직교 회전 행렬
    """
    a1 = rot_6d[:3].astype(np.float64)
    a2 = rot_6d[3:6].astype(np.float64)

    # Gram-Schmidt 정규직교화
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)

    return np.column_stack([b1, b2, b3])


def load_sam3d_pose(
    pose_source: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """SAM3D 포즈 파라미터 로드.

    지원 형식:
      - .json: {"rotation": [...], "translation": [...], "scale": ...}
      - .npz/.npy: rotation, translation, scale 키
      - .pt/.pth: PyTorch state_dict

    Returns:
        R_sam3d (3x3), t_sam3d (3,), scale (float)
    """
    ext = Path(pose_source).suffix.lower()

    if ext == ".json":
        with open(pose_source) as f:
            data = json.load(f)
        # rotation: 3x3, 6D, 또는 quaternion
        rot_raw = np.array(data["rotation"], dtype=np.float64)
        if rot_raw.shape == (3, 3):
            R = rot_raw
        elif rot_raw.shape == (6,):
            R = rotation_6d_to_matrix(rot_raw)
        elif rot_raw.shape == (4,):  # quaternion wxyz or xyzw
            R = _quat_to_matrix(rot_raw)
        else:
            R = rot_raw.reshape(3, 3)

        t = np.array(data["translation"], dtype=np.float64).ravel()
        s = float(data.get("scale", 1.0))
        if isinstance(s, (list, np.ndarray)):
            s = float(np.mean(s))

    elif ext in {".npz", ".npy"}:
        data = np.load(pose_source, allow_pickle=True)
        if ext == ".npy":
            data = data.item()  # dict wrapped in npy
        rot_raw = np.array(data["rotation"], dtype=np.float64)
        if rot_raw.size == 6:
            R = rotation_6d_to_matrix(rot_raw.ravel())
        elif rot_raw.size == 9:
            R = rot_raw.reshape(3, 3)
        else:
            R = rot_raw.reshape(3, 3)
        t = np.array(data["translation"], dtype=np.float64).ravel()
        s = float(np.mean(data.get("scale", 1.0)))

    elif ext in {".pt", ".pth"}:
        import torch
        data = torch.load(pose_source, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            rot_raw = data.get("rotation", data.get("6drotation_normalized"))
            if hasattr(rot_raw, "numpy"):
                rot_raw = rot_raw.detach().numpy()
            rot_raw = np.array(rot_raw, dtype=np.float64).squeeze()
            if rot_raw.size == 6:
                R = rotation_6d_to_matrix(rot_raw.ravel())
            else:
                R = rot_raw.reshape(3, 3)

            t_raw = data.get("translation", np.zeros(3))
            if hasattr(t_raw, "numpy"):
                t_raw = t_raw.detach().numpy()
            t = np.array(t_raw, dtype=np.float64).ravel()

            s_raw = data.get("scale", data.get("translation_scale", 1.0))
            if hasattr(s_raw, "item"):
                s_raw = s_raw.item()
            s = float(np.mean(s_raw))
        else:
            raise ValueError(f"PT 파일 형식 인식 불가: {type(data)}")
    else:
        raise ValueError(f"지원하지 않는 포즈 파일 형식: {ext}")

    print(f"  SAM3D 포즈 로드: {pose_source}")
    print(f"    R = {R.ravel()[:6]}...")
    print(f"    t = {t}")
    print(f"    scale = {s:.6f}")
    return R, t, s


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """쿼터니언 [w,x,y,z] → 3x3 회전 행렬."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def load_sam3d_pointmap(
    pointmap_source: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """SAM3D pointmap 로드 (H×W×3 또는 N×3).

    pointmap: 각 픽셀의 3D 좌표 (SAM3D 정규화 공간)
    pointmap_colors: 각 픽셀의 RGB 색상

    Returns:
        points (N,3), colors (N,3) or None
    """
    ext = Path(pointmap_source).suffix.lower()

    if ext in {".npz", ".npy"}:
        data = np.load(pointmap_source, allow_pickle=True)
        if ext == ".npy":
            pointmap = data
            colors = None
        else:
            pointmap = data.get("pointmap", data.get("points", None))
            colors = data.get("pointmap_colors", data.get("colors", None))
            if pointmap is None:
                # dict-in-npy
                d = data.item() if hasattr(data, 'item') else data
                pointmap = d.get("pointmap", d.get("points"))
                colors = d.get("pointmap_colors", d.get("colors"))

    elif ext in {".pt", ".pth"}:
        import torch
        data = torch.load(pointmap_source, map_location="cpu", weights_only=False)
        pointmap = data.get("pointmap", data.get("points"))
        colors = data.get("pointmap_colors", data.get("colors"))
        if hasattr(pointmap, "numpy"):
            pointmap = pointmap.detach().numpy()
        if colors is not None and hasattr(colors, "numpy"):
            colors = colors.detach().numpy()
    else:
        raise ValueError(f"pointmap 형식 미지원: {ext}")

    pointmap = np.array(pointmap, dtype=np.float64)
    if pointmap.ndim == 3:
        # H×W×3 → N×3 (유효 점만)
        h, w, _ = pointmap.shape
        mask = np.linalg.norm(pointmap, axis=2) > 1e-8
        points = pointmap[mask]
        if colors is not None:
            colors = np.array(colors, dtype=np.float64)
            if colors.ndim == 3:
                colors = colors[mask]
            if colors.max() > 1.0:
                colors = colors / 255.0
    else:
        points = pointmap.reshape(-1, 3)
        if colors is not None:
            colors = np.array(colors, dtype=np.float64).reshape(-1, 3)
            if colors.max() > 1.0:
                colors = colors / 255.0

    print(f"  SAM3D pointmap: {len(points):,} pts from {pointmap_source}")
    return points, colors


def load_sam3d_splat_ply(ply_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """SAM3D Gaussian Splat PLY에서 점군 + (선택) 색상 추출.

    Gaussian Splat PLY는 xyz, scale, rot 등의 속성을 포함하지만
    여기서는 xyz 좌표만 사용하여 점군으로 변환.
    """
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None

    if len(points) == 0:
        raise RuntimeError(f"SAM3D PLY가 비어있음: {ply_path}")

    bbox = points.max(axis=0) - points.min(axis=0)
    print(f"  SAM3D splat PLY: {len(points):,} pts")
    print(f"    bbox: {bbox[0]:.4f} x {bbox[1]:.4f} x {bbox[2]:.4f}")
    print(f"    center: ({points.mean(0)[0]:.4f}, {points.mean(0)[1]:.4f}, {points.mean(0)[2]:.4f})")
    return points, colors


def sam3d_to_cam0(
    points_sam3d: np.ndarray,
    R_sam3d: Optional[np.ndarray],
    t_sam3d: Optional[np.ndarray],
    scale_sam3d: float,
    T_C0_Ci: np.ndarray,
    sam3d_is_opengl: bool = True,
) -> np.ndarray:
    """SAM3D 점군을 cam0 월드좌표계로 변환.

    변환 체인:
      1. SAM3D canonical → SAM3D output: scale * R @ P + t
      2. SAM3D output → Camera_i: OpenGL→OpenCV 변환 (Y,Z 반전)
      3. Camera_i → cam0: T_C0_Ci 적용

    Args:
        points_sam3d: (N,3) SAM3D 정규화 좌표
        R_sam3d: (3,3) SAM3D 회전 (None이면 항등)
        t_sam3d: (3,) SAM3D 이동 (None이면 0)
        scale_sam3d: SAM3D 스케일
        T_C0_Ci: (4,4) cam_i → cam0 변환 행렬
        sam3d_is_opengl: SAM3D 출력이 OpenGL 좌표계인지 여부

    Returns:
        (N,3) cam0 좌표계 점군
    """
    pts = points_sam3d.copy()

    # Step 1: SAM3D canonical → SAM3D output
    if R_sam3d is not None:
        pts = pts @ R_sam3d.T
    pts *= scale_sam3d
    if t_sam3d is not None:
        pts += t_sam3d

    # Step 2: OpenGL(Y-up) → OpenCV(Y-down): Y, Z 반전
    if sam3d_is_opengl:
        pts[:, 1] *= -1.0
        pts[:, 2] *= -1.0

    # Step 3: Camera_i → cam0
    pts_cam0 = pts @ T_C0_Ci[:3, :3].T + T_C0_Ci[:3, 3]

    return pts_cam0


def auto_detect_sam3d_files(sam3d_dir: str) -> dict:
    """SAM3D 출력 디렉토리에서 파일들을 자동 탐지.

    Returns:
        dict with keys: 'ply', 'pose', 'pointmap', 'mesh', 'gaussian'
    """
    d = Path(sam3d_dir)
    result = {}

    # Gaussian Splat PLY
    for pattern in ["*.ply", "splat*.ply", "gaussian*.ply", "*sam3d*.ply"]:
        files = sorted(d.glob(pattern))
        if files:
            result["ply"] = str(files[0])
            break

    # Mesh / GLB
    for ext in [".glb", ".gltf", ".obj"]:
        files = sorted(d.glob(f"*{ext}"))
        if files:
            result["mesh"] = str(files[0])
            break

    # Pose 파일 (rotation/translation/scale)
    for pattern in ["*pose*.json", "*transform*.json", "*.json"]:
        for f in sorted(d.glob(pattern)):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                if any(k in data for k in ["rotation", "translation", "scale",
                                            "6drotation_normalized"]):
                    result["pose"] = str(f)
                    break
            except (json.JSONDecodeError, Exception):
                continue
        if "pose" in result:
            break

    # Pointmap
    for pattern in ["*pointmap*", "*points*"]:
        for ext in [".npz", ".npy", ".pt", ".pth"]:
            files = sorted(d.glob(f"{pattern}{ext}"))
            if files:
                result["pointmap"] = str(files[0])
                break

    # PyTorch checkpoint (rotation/translation/scale 포함)
    for pattern in ["*.pt", "*.pth", "*checkpoint*", "*model*"]:
        files = sorted(d.glob(pattern))
        for f in files:
            if str(f) not in result.values():
                result.setdefault("checkpoint", str(f))
                break

    print(f"  SAM3D 파일 자동 탐지:")
    for k, v in result.items():
        print(f"    {k}: {Path(v).name}")
    return result


def run_sam3d_pipeline(
    args: argparse.Namespace,
    session: MultiViewSession,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, Optional[np.ndarray]]:
    """SAM3D 기반 포즈 추정 파이프라인.

    1. SAM3D 출력 로드 (PLY/pointmap + pose)
    2. SAM3D 좌표 → cam0 좌표 변환
    3. (선택) 관측 점군과 ICP 정밀 정합

    Returns:
        R (3x3), t (3,), scale, sam3d_points_cam0, sam3d_colors
    """
    sam3d_cam = args.sam3d_cam  # SAM3D가 어떤 카메라 시점에서 생성되었는지

    # --- SAM3D 파일 로드 ---
    if args.sam3d_dir:
        files = auto_detect_sam3d_files(args.sam3d_dir)
        ply_path = args.sam3d_ply or files.get("ply")
        pose_path = args.sam3d_pose or files.get("pose")
        pointmap_path = files.get("pointmap")
    else:
        ply_path = args.sam3d_ply
        pose_path = args.sam3d_pose
        pointmap_path = None

    # --- SAM3D 점군 로드 ---
    if pointmap_path:
        print("\n  [SAM3D] pointmap에서 점군 로드")
        sam3d_pts, sam3d_colors = load_sam3d_pointmap(pointmap_path)
    elif ply_path:
        print(f"\n  [SAM3D] Gaussian Splat PLY에서 점군 로드: {ply_path}")
        sam3d_pts, sam3d_colors = load_sam3d_splat_ply(ply_path)
    else:
        raise RuntimeError("SAM3D 점군 소스 없음 (--sam3d_ply 또는 --sam3d_dir 필요)")

    # --- SAM3D 포즈 로드 (있으면) ---
    R_sam3d, t_sam3d, scale_sam3d = None, None, 1.0

    if pose_path:
        print(f"\n  [SAM3D] 포즈 파라미터 로드: {pose_path}")
        R_sam3d, t_sam3d, scale_sam3d = load_sam3d_pose(pose_path)
    else:
        print("\n  [SAM3D] 포즈 파일 없음 - 항등 변환 + 자동 스케일 사용")

    # --- cam_i → cam0 변환 행렬 ---
    T_C0_Ci = session.calibrations[sam3d_cam].T_cam0_cam
    print(f"\n  [SAM3D] 좌표 변환: SAM3D → cam{sam3d_cam} → cam0")
    print(f"    T_C0_C{sam3d_cam} = {'identity' if sam3d_cam == 0 else 'loaded'}")

    # --- SAM3D → cam0 변환 ---
    sam3d_pts_cam0 = sam3d_to_cam0(
        sam3d_pts,
        R_sam3d, t_sam3d, scale_sam3d,
        T_C0_Ci,
        sam3d_is_opengl=not args.sam3d_opencv,  # 기본: OpenGL 좌표계
    )

    bbox_cam0 = (sam3d_pts_cam0.max(0) - sam3d_pts_cam0.min(0)) * 1000
    center_cam0 = sam3d_pts_cam0.mean(0) * 1000
    print(f"    cam0 변환 후: {len(sam3d_pts_cam0):,} pts")
    print(f"    bbox (mm): {bbox_cam0[0]:.1f} x {bbox_cam0[1]:.1f} x {bbox_cam0[2]:.1f}")
    print(f"    center (mm): ({center_cam0[0]:.1f}, {center_cam0[1]:.1f}, {center_cam0[2]:.1f})")

    # --- 관측 점군과 스케일 자동 조정 (포즈 없는 경우) ---
    if pose_path is None:
        print("\n  [SAM3D] 관측 점군 생성 (스케일 정합용)")
        obs_pts, obs_colors = _get_observation_points(args, session)
        obs_extent = obs_pts.max(0) - obs_pts.min(0)
        sam_extent = sam3d_pts_cam0.max(0) - sam3d_pts_cam0.min(0)
        auto_scale = float(np.median(obs_extent / (sam_extent + 1e-8)))

        print(f"    자동 스케일: {auto_scale:.4f}")
        sam3d_center = sam3d_pts_cam0.mean(0)
        sam3d_pts_cam0 = (sam3d_pts_cam0 - sam3d_center) * auto_scale
        # 관측 점군 중심으로 이동
        sam3d_pts_cam0 += obs_pts.mean(0)
        scale_sam3d *= auto_scale

    # --- (선택) ICP 정밀 정합 ---
    if args.sam3d_refine_icp:
        print("\n  [SAM3D] ICP 정밀 정합으로 cam0 포즈 보정")
        obs_pts, obs_colors = _get_observation_points(args, session)
        obs_pcd = build_pointcloud(obs_pts, obs_colors)
        sam_pcd = build_pointcloud(sam3d_pts_cam0, sam3d_colors)

        icp_dist_m = args.icp_dist / 1000.0
        T_refine, fitness, rmse = run_icp_multistage(
            sam_pcd, obs_pcd,
            init_transform=np.eye(4),
            max_corr_dist=icp_dist_m,
            fine_iters=args.icp_iters,
        )
        print(f"    ICP 보정: fitness={fitness:.4f}, RMSE={rmse*1000:.3f}mm")

        # 보정 적용
        sam3d_pts_cam0 = sam3d_pts_cam0 @ T_refine[:3, :3].T + T_refine[:3, 3]

        R_final = T_refine[:3, :3]
        t_final = T_refine[:3, 3]
    else:
        # ICP 없이 PCA 기반 포즈 추출
        R_final, t_final = _extract_pose_from_pointcloud(sam3d_pts_cam0)
        fitness, rmse = 1.0, 0.0

    return R_final, t_final, scale_sam3d, sam3d_pts_cam0, sam3d_colors


def _get_observation_points(
    args: argparse.Namespace,
    session: MultiViewSession,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """세그멘테이션된 관측 점군 반환 (캐시)."""
    if not hasattr(args, "_obs_cache"):
        if args.seg_mode == "hsv":
            pts, colors, _ = segment_by_hsv_anchor(
                session, args.z_min, args.z_max,
                h_range=args.hsv_h_range,
                s_min=args.hsv_s_min, v_min=args.hsv_v_min)
        else:
            pts, colors = segment_by_depth_roi(session, args.z_min, args.z_max)
        args._obs_cache = (pts, colors)
    return args._obs_cache


def _extract_pose_from_pointcloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """점군의 PCA 기반 포즈 (R, t) 추출."""
    center = points.mean(axis=0)
    centered = points - center
    cov = centered.T @ centered / len(centered)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    R = evecs[:, order]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R, center


# ---------------------------------------------------------------------------
# Step 5: 스케일 정합
# ---------------------------------------------------------------------------

def scale_reference_to_observation(
    ref_pcd: "open3d.geometry.PointCloud",
    obs_pcd: "open3d.geometry.PointCloud",
    ref_length_mm: Optional[float] = None,
    manual_scale: Optional[float] = None,
) -> Tuple["open3d.geometry.PointCloud", float]:
    """참조 모델의 스케일을 관측 점군에 맞춤.

    우선순위: manual_scale > ref_length_mm > 자동(bbox)
    """
    import open3d as o3d

    ref_pts = np.asarray(ref_pcd.points)
    ref_extent = ref_pts.max(axis=0) - ref_pts.min(axis=0)
    ref_max = float(ref_extent.max())
    if ref_max < 1e-8:
        raise RuntimeError("참조 모델 크기가 0")

    if manual_scale is not None:
        scale = float(manual_scale)
        print(f"  스케일: {scale:.6f} (수동 지정)")
    elif ref_length_mm is not None:
        scale = (float(ref_length_mm) / 1000.0) / ref_max
        print(f"  스케일: {scale:.6f} (실제 길이 {ref_length_mm:.1f}mm 기준)")
    else:
        obs_pts = np.asarray(obs_pcd.points)
        obs_extent = obs_pts.max(axis=0) - obs_pts.min(axis=0)
        obs_max = float(obs_extent.max())
        scale = obs_max / ref_max
        print(f"  스케일: {scale:.6f} (자동 bbox: ref_max={ref_max:.4f} -> obs_max={obs_max:.4f}m)")

    ref_scaled = o3d.geometry.PointCloud()
    ref_scaled.points = o3d.utility.Vector3dVector(ref_pts * scale)
    if ref_pcd.has_colors():
        ref_scaled.colors = ref_pcd.colors
    if ref_pcd.has_normals():
        ref_scaled.normals = ref_pcd.normals

    scaled_ext = np.asarray(ref_scaled.points).max(axis=0) - np.asarray(ref_scaled.points).min(axis=0)
    print(f"  스케일 후 크기: {scaled_ext[0]*1000:.1f} x {scaled_ext[1]*1000:.1f} x {scaled_ext[2]*1000:.1f} mm")
    return ref_scaled, scale


# ---------------------------------------------------------------------------
# Step 6: 초기 정합 - FPFH + RANSAC 전역 정합
# ---------------------------------------------------------------------------

def compute_fpfh(pcd: "open3d.geometry.PointCloud", voxel_size: float):
    """점군을 다운샘플링하고 FPFH 특징을 계산."""
    import open3d as o3d

    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh


def global_registration_fpfh(
    model_pcd: "open3d.geometry.PointCloud",
    scene_pcd: "open3d.geometry.PointCloud",
    voxel_size: float = 0.005,
) -> np.ndarray:
    """FPFH 특징 기반 RANSAC 전역 정합.

    Returns:
        4x4 초기 변환 행렬
    """
    import open3d as o3d

    print(f"    voxel_size: {voxel_size*1000:.1f}mm")

    model_down, model_fpfh = compute_fpfh(model_pcd, voxel_size)
    scene_down, scene_fpfh = compute_fpfh(scene_pcd, voxel_size)

    print(f"    model_down: {len(model_down.points):,} pts")
    print(f"    scene_down: {len(scene_down.points):,} pts")

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down, scene_down,
        model_fpfh, scene_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 3,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    print(f"    RANSAC fitness: {result.fitness:.4f}")
    print(f"    RANSAC RMSE: {result.inlier_rmse*1000:.3f}mm")
    print(f"    correspondences: {len(result.correspondence_set):,}")

    return result.transformation


def initial_alignment_pca(
    ref_points: np.ndarray,
    obs_points: np.ndarray,
) -> np.ndarray:
    """PCA 기반 초기 정렬 (fallback)."""
    def pca(pts):
        c = pts.mean(axis=0)
        centered = pts - c
        cov = centered.T @ centered / len(centered)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        return c, evecs[:, order], evals[order]

    ref_c, ref_axes, _ = pca(ref_points)
    obs_c, obs_axes, _ = pca(obs_points)

    R = obs_axes @ ref_axes.T
    if np.linalg.det(R) < 0:
        obs_axes[:, 2] *= -1
        R = obs_axes @ ref_axes.T

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = obs_c - R @ ref_c
    return T


# ---------------------------------------------------------------------------
# Step 7: 정밀 정합 - 다단계 ICP
# ---------------------------------------------------------------------------

def run_icp_multistage(
    model_pcd: "open3d.geometry.PointCloud",
    scene_pcd: "open3d.geometry.PointCloud",
    init_transform: np.ndarray,
    max_corr_dist: float = 0.005,
    fine_iters: int = 200,
) -> Tuple[np.ndarray, float, float]:
    """다단계 Point-to-Plane ICP.

    Coarse -> Medium -> Fine 순서로 correspondence distance를 줄여가며 정밀 정합.
    PCA 축 뒤집기를 통해 4가지 초기 자세를 시도하고 최적 결과를 반환.

    Returns:
        transform (4x4), fitness, inlier_rmse
    """
    import open3d as o3d

    model_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    scene_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    stages = [
        (max_corr_dist * 4, 30,         "coarse"),
        (max_corr_dist * 2, 30,         "medium"),
        (max_corr_dist,     fine_iters, "fine"),
    ]

    def run_once(start: np.ndarray, label: str = "") -> Tuple[np.ndarray, float, float]:
        current = start
        result = None
        for dist, iters, name in stages:
            result = o3d.pipelines.registration.registration_icp(
                model_pcd, scene_pcd,
                max_correspondence_distance=dist,
                init=current,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iters,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                ),
            )
            current = result.transformation
            print(f"    ICP [{label}/{name}] dist={dist*1000:.1f}mm: "
                  f"fitness={result.fitness:.4f}, RMSE={result.inlier_rmse*1000:.3f}mm")
        return current, float(result.fitness), float(result.inlier_rmse)

    # 원본 + 3가지 축 뒤집기 시도
    candidates = []
    T0, f0, r0 = run_once(init_transform, "original")
    candidates.append((T0, f0, r0, "original"))

    for i, signs in enumerate([
        np.diag([-1, -1, 1]),
        np.diag([1, -1, -1]),
        np.diag([-1, 1, -1]),
    ], start=1):
        flipped = init_transform.copy()
        flipped[:3, :3] = init_transform[:3, :3] @ signs
        Ti, fi, ri = run_once(flipped, f"flip{i}")
        candidates.append((Ti, fi, ri, f"flip{i}"))

    # fitness 기준 최적 선택 (동일 fitness면 RMSE가 작은 것)
    stable = [c for c in candidates if c[1] > 0.5]
    if stable:
        best = min(stable, key=lambda c: c[2])
    else:
        best = max(candidates, key=lambda c: (c[1], -c[2]))

    print(f"    -> '{best[3]}' 채택 (fitness={best[1]:.4f}, RMSE={best[2]*1000:.3f}mm)")
    return best[0], best[1], best[2]


# ---------------------------------------------------------------------------
# Step 8: 포즈 추출
# ---------------------------------------------------------------------------

def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    """회전 행렬 -> XYZ 오일러 각 (degrees)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """회전 행렬 -> 쿼터니언 [w, x, y, z]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, y = (R[0, 2] - R[2, 0]) / s, 0.25 * s
        x, z = (R[0, 1] + R[1, 0]) / s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, z = (R[1, 0] - R[0, 1]) / s, 0.25 * s
        x, y = (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


# ---------------------------------------------------------------------------
# 시각화 및 결과 저장
# ---------------------------------------------------------------------------

def visualize_alignment(
    obs_points: np.ndarray,
    obs_colors: Optional[np.ndarray],
    ref_points_aligned: np.ndarray,
    pose: dict,
    out_path: str,
    title: str,
) -> None:
    """정합 결과 3D 시각화 (3방향 뷰)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 6))
    obs_mm = obs_points * 1000.0
    ref_mm = ref_points_aligned * 1000.0
    R = pose["R"]
    center_mm = pose["position_mm"]

    views = [(25, -60, "Perspective"), (90, -90, "Top (XZ)"), (0, -90, "Front (XY)")]
    rng = np.random.default_rng(42)

    for idx, (elev, azim, subtitle) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.set_title(subtitle, fontsize=10)

        n_obs = min(len(obs_mm), 5000)
        sel_obs = rng.choice(len(obs_mm), n_obs, replace=False)
        if obs_colors is not None:
            ax.scatter(obs_mm[sel_obs, 0], obs_mm[sel_obs, 1], obs_mm[sel_obs, 2],
                       c=np.clip(obs_colors[sel_obs], 0, 1), s=0.5, alpha=0.4)
        else:
            ax.scatter(obs_mm[sel_obs, 0], obs_mm[sel_obs, 1], obs_mm[sel_obs, 2],
                       c="steelblue", s=0.5, alpha=0.4)

        n_ref = min(len(ref_mm), 3000)
        sel_ref = rng.choice(len(ref_mm), n_ref, replace=False)
        ax.scatter(ref_mm[sel_ref, 0], ref_mm[sel_ref, 1], ref_mm[sel_ref, 2],
                   c="#e74c3c", s=0.5, alpha=0.3)

        axis_len = 30.0
        colors = ["#e74c3c", "#27ae60", "#2980b9"]
        for a in range(3):
            vec = R[:, a] * axis_len
            ax.quiver(center_mm[0], center_mm[1], center_mm[2],
                      vec[0], vec[1], vec[2], color=colors[a], linewidth=2.5,
                      arrow_length_ratio=0.12)

        ax.set_xlabel("X mm", fontsize=7)
        ax.set_ylabel("Y mm", fontsize=7)
        ax.set_zlabel("Z mm", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)

        all_mm = np.vstack([obs_mm, ref_mm])
        mid = all_mm.mean(axis=0)
        rad = max((all_mm.max(axis=0) - all_mm.min(axis=0)).max() / 2 * 1.3, 40)
        ax.set_xlim(mid[0] - rad, mid[0] + rad)
        ax.set_ylim(mid[1] - rad, mid[1] + rad)
        ax.set_zlim(mid[2] - rad, mid[2] + rad)

    euler = pose["euler_xyz_deg"]
    fig.suptitle(title, fontsize=11, y=0.98)
    fig.text(0.5, 0.01,
             f"Position: ({center_mm[0]:.1f}, {center_mm[1]:.1f}, {center_mm[2]:.1f}) mm  |  "
             f"Euler: ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg  |  "
             f"fitness: {pose['icp_fitness']:.4f}, RMSE: {pose['icp_rmse_mm']:.2f}mm",
             ha="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ecf0f1", ec="#bdc3c7"))
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {out_path}")


def save_projection_overlay(
    obs_points: np.ndarray,
    ref_points_aligned: np.ndarray,
    R: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    rgb_path: str,
    out_path: str,
) -> None:
    """cam0 이미지 위에 관측/정합 점군 재투영 오버레이."""
    bgr = cv2.imread(rgb_path)
    if bgr is None:
        return

    h, w = bgr.shape[:2]
    D_arr = np.asarray(D, dtype=np.float64)

    def draw(pts, color, step=8):
        if len(pts) == 0:
            return
        uv, _ = cv2.projectPoints(
            pts[::step].astype(np.float64).reshape(-1, 1, 3),
            np.zeros(3), np.zeros(3), K, D_arr)
        uv = uv.reshape(-1, 2)
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        for u, v in uv[valid]:
            cv2.circle(bgr, (int(u), int(v)), 2, color, -1)

    draw(obs_points, (255, 80, 0))      # 관측 = 주황
    draw(ref_points_aligned, (0, 50, 255))  # 참조 = 빨강

    # 좌표축 표시
    center = ref_points_aligned.mean(axis=0)
    for a, color in enumerate([(0, 0, 200), (0, 200, 0), (200, 0, 0)]):
        end = center + R[:, a] * 0.030
        uv_s, _ = cv2.projectPoints(center.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, D_arr)
        uv_e, _ = cv2.projectPoints(end.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, D_arr)
        s = tuple(uv_s.reshape(2).astype(int))
        e = tuple(uv_e.reshape(2).astype(int))
        if 0 <= s[0] < w and 0 <= s[1] < h and 0 <= e[0] < w and 0 <= e[1] < h:
            cv2.arrowedLine(bgr, s, e, color, 2, tipLength=0.2)

    cv2.putText(bgr, "Orange=Observed  Red=Reference(aligned)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(out_path, bgr)
    print(f"  [저장] {out_path}")


def save_results(
    R: np.ndarray,
    t: np.ndarray,
    fitness: float,
    rmse: float,
    scale: float,
    obs_points: np.ndarray,
    obs_colors: Optional[np.ndarray],
    ref_points_aligned: np.ndarray,
    out_dir: Path,
    tag: str,
    elapsed_sec: float,
    ref_model_path: str,
    session: Optional[MultiViewSession] = None,
    frame_idx: int = 3,
) -> dict:
    """결과를 JSON, PLY, PNG로 저장."""
    import open3d as o3d

    out_dir.mkdir(parents=True, exist_ok=True)

    euler = rotation_to_euler(R).tolist()
    quat = rotation_to_quaternion(R).tolist()
    pos_mm = (t * 1000.0).tolist()

    print("\n" + "=" * 60)
    print("  RESULT - cam0 (X-right, Y-down, Z-forward)")
    print("=" * 60)
    print(f"  Position   (mm): ({pos_mm[0]:+.1f}, {pos_mm[1]:+.1f}, {pos_mm[2]:+.1f})")
    print(f"  Euler XYZ (deg): ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f})")
    print(f"  Quat wxyz      : ({quat[0]:.5f}, {quat[1]:.5f}, {quat[2]:.5f}, {quat[3]:.5f})")
    print(f"  fitness        : {fitness:.4f}")
    print(f"  RMSE      (mm) : {rmse*1000:.3f}")
    print(f"  Scale factor   : {scale:.6f}")
    print(f"  소요시간       : {elapsed_sec:.1f}s")
    print("=" * 60)

    result = {
        "coordinate_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
        "method": "FPFH_RANSAC + ICP",
        "reference_model": os.path.abspath(ref_model_path),
        "position_mm": pos_mm,
        "euler_xyz_deg": euler,
        "quaternion_wxyz": quat,
        "rotation_matrix": R.tolist(),
        "icp_fitness": float(fitness),
        "icp_rmse_mm": float(rmse * 1000),
        "scale_factor": float(scale),
        "elapsed_sec": round(float(elapsed_sec), 2),
    }

    # JSON
    json_path = out_dir / f"pose_{tag}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {json_path}")

    # 관측 점군 PLY
    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(obs_points)
    if obs_colors is not None:
        obs_pcd.colors = o3d.utility.Vector3dVector(np.clip(obs_colors, 0, 1))
    o3d.io.write_point_cloud(str(out_dir / f"observed_{tag}.ply"), obs_pcd)

    # 정합된 참조 점군 PLY
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_points_aligned)
    o3d.io.write_point_cloud(str(out_dir / f"aligned_ref_{tag}.ply"), ref_pcd)

    # 포즈 정보 dict (시각화용)
    pose_dict = {
        "R": R,
        "position_mm": np.array(pos_mm),
        "euler_xyz_deg": euler,
        "icp_fitness": fitness,
        "icp_rmse_mm": rmse * 1000,
    }

    # 3D 정합 시각화
    try:
        visualize_alignment(
            obs_points, obs_colors, ref_points_aligned, pose_dict,
            str(out_dir / f"alignment_{tag}.png"),
            f"FPFH+RANSAC+ICP Alignment - {tag}")
    except Exception as e:
        print(f"  [WARN] 정합 시각화 실패: {e}")

    # cam0 재투영 오버레이
    if session is not None:
        try:
            cam0_calib = session.calibrations[0]
            cam0_frame = session.frames[0]
            save_projection_overlay(
                obs_points, ref_points_aligned, R,
                cam0_calib.K, cam0_calib.D,
                str(cam0_frame.rgb_path),
                str(out_dir / f"overlay_cam0_{tag}.png"))
        except Exception as e:
            print(f"  [WARN] 오버레이 시각화 실패: {e}")

    return result


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------

def build_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None):
    """numpy -> Open3D PointCloud."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    return pcd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="멀티뷰 카메라 기반 물체 6-DOF 포즈 추정 (FPFH+RANSAC+ICP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ref_model", default=str(_DEFAULT_REF_MODEL),
                        help="참조 3D 모델 경로 (GLB/PLY/OBJ)")
    parser.add_argument("--capture_dir", default=str(_DEFAULT_CAP_DIR))
    parser.add_argument("--calib_dir", default=str(_DEFAULT_CAL_DIR))
    parser.add_argument("--intrinsics_dir", default=str(_DEFAULT_INT_DIR))
    parser.add_argument("--frame", type=int, default=3, help="프레임 번호")
    parser.add_argument("--z_min", type=float, default=0.1, help="최소 깊이 (m)")
    parser.add_argument("--z_max", type=float, default=1.5, help="최대 깊이 (m)")
    parser.add_argument("--out_dir", default=str(_DEFAULT_OUT_DIR))

    # 세그멘테이션
    parser.add_argument("--seg_mode", choices=["depth_roi", "hsv"], default="depth_roi",
                        help="세그멘테이션 방식: depth_roi(테이블 제거) / hsv(노란 손잡이)")
    parser.add_argument("--hsv_h_range", type=int, nargs=2, default=[15, 35])
    parser.add_argument("--hsv_s_min", type=int, default=80)
    parser.add_argument("--hsv_v_min", type=int, default=80)

    # 스케일
    parser.add_argument("--ref_length_mm", type=float, default=None,
                        help="물체 실제 최장 길이 (mm)")
    parser.add_argument("--scale", type=float, default=None,
                        help="참조 모델 스케일 직접 지정")

    # 정합 파라미터
    parser.add_argument("--voxel_size", type=float, default=5.0,
                        help="FPFH 전역 정합 voxel 크기 (mm)")
    parser.add_argument("--icp_dist", type=float, default=5.0,
                        help="ICP max correspondence distance (mm)")
    parser.add_argument("--icp_iters", type=int, default=200,
                        help="ICP fine 단계 최대 반복")
    parser.add_argument("--init_method", choices=["fpfh", "pca", "both"], default="both",
                        help="초기 정합 방법: fpfh, pca, both(둘 다 시도)")

    # 참조 모델 샘플 수
    parser.add_argument("--ref_samples", type=int, default=50000,
                        help="참조 모델 점군 샘플 수")

    # --- SAM3D 통합 ---
    sam3d = parser.add_argument_group("SAM3D 통합")
    sam3d.add_argument("--sam3d_dir", default=None,
                       help="SAM3D 출력 디렉토리 (자동 탐지)")
    sam3d.add_argument("--sam3d_ply", default=None,
                       help="SAM3D Gaussian Splat PLY 경로")
    sam3d.add_argument("--sam3d_pose", default=None,
                       help="SAM3D 포즈 파일 (rotation/translation/scale)")
    sam3d.add_argument("--sam3d_pointmap", default=None,
                       help="SAM3D pointmap 파일 (.npz/.pt)")
    sam3d.add_argument("--sam3d_cam", type=int, default=0,
                       help="SAM3D가 생성된 카메라 시점 (0/1/2)")
    sam3d.add_argument("--sam3d_refine_icp", action="store_true",
                       help="관측 점군과 ICP 정밀 정합 수행")
    sam3d.add_argument("--sam3d_opencv", action="store_true",
                       help="SAM3D 출력이 이미 OpenCV 좌표계인 경우")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    use_sam3d = bool(args.sam3d_dir or args.sam3d_ply or args.sam3d_pointmap)
    mode_str = "SAM3D 통합" if use_sam3d else "ICP 기반"

    print("=" * 60)
    print(f"  Obj_pose_0311.py - 멀티뷰 물체 포즈 추정 [{mode_str}]")
    print(f"  ref_model   : {args.ref_model}")
    print(f"  frame       : {args.frame}")
    if use_sam3d:
        print(f"  sam3d_dir   : {args.sam3d_dir or 'N/A'}")
        print(f"  sam3d_ply   : {args.sam3d_ply or 'auto'}")
        print(f"  sam3d_cam   : cam{args.sam3d_cam}")
        print(f"  refine_icp  : {args.sam3d_refine_icp}")
    else:
        print(f"  seg_mode    : {args.seg_mode}")
        print(f"  init_method : {args.init_method}")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: 데이터 로드 (공통)
    # ---------------------------------------------------------------
    print("\n[Step 1] 캘리브레이션 + 프레임 로드")
    session = load_session(args)
    print(f"  카메라: {list(session.camera_ids)}")
    for cid in session.camera_ids:
        cal = session.calibrations[cid]
        print(f"    cam{cid}: depth_scale={cal.depth_scale:.6f} m/unit")

    # ---------------------------------------------------------------
    # 분기: SAM3D 또는 ICP 파이프라인
    # ---------------------------------------------------------------
    if use_sam3d:
        R, t, scale, sam3d_pts_cam0, sam3d_colors, fitness, rmse = _main_sam3d(args, session)
        obs_points = sam3d_pts_cam0
        obs_colors = sam3d_colors
        ref_aligned = sam3d_pts_cam0  # SAM3D 모델 자체가 정합 결과
        method_tag = "sam3d"
    else:
        R, t, scale, obs_points, obs_colors, ref_aligned, fitness, rmse = _main_icp(args, session)
        method_tag = "icp"

    # ---------------------------------------------------------------
    # 포즈 출력 + 결과 저장
    # ---------------------------------------------------------------
    euler = rotation_to_euler(R)
    quat = rotation_to_quaternion(R)
    print(f"\n[포즈 추출]")
    print(f"  위치 (cam0, m): ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
    print(f"  오일러 XYZ (deg): ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f})")
    print(f"  쿼터니언 wxyz: ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")

    tag = f"frame{args.frame:06d}_{method_tag}"
    out_dir = Path(args.out_dir) / f"output_{tag}"

    elapsed = time.time() - start_time
    save_results(
        R, t, fitness, rmse, scale,
        obs_points, obs_colors, ref_aligned,
        out_dir, tag, elapsed,
        args.ref_model if not use_sam3d else (args.sam3d_ply or args.sam3d_dir or "sam3d"),
        session=session,
        frame_idx=args.frame,
    )

    print(f"\n완료! (총 {elapsed:.1f}s)")


# ---------------------------------------------------------------
# 파이프라인 A: ICP 기반
# ---------------------------------------------------------------
def _main_icp(args, session):
    """기존 ICP 파이프라인 (FPFH+RANSAC → 다단계 ICP)."""

    print(f"\n[Step 2-3] 세그멘테이션 ({args.seg_mode}) + 멀티뷰 점군 통합")
    if args.seg_mode == "hsv":
        obs_points, obs_colors, _ = segment_by_hsv_anchor(
            session, args.z_min, args.z_max,
            h_range=args.hsv_h_range,
            s_min=args.hsv_s_min, v_min=args.hsv_v_min)
    else:
        obs_points, obs_colors = segment_by_depth_roi(session, args.z_min, args.z_max)

    obs_bbox = (obs_points.max(axis=0) - obs_points.min(axis=0)) * 1000
    print(f"  관측 점군: {len(obs_points):,} pts")
    print(f"  bbox: {obs_bbox[0]:.1f} x {obs_bbox[1]:.1f} x {obs_bbox[2]:.1f} mm")
    obs_pcd = build_pointcloud(obs_points, obs_colors)

    print(f"\n[Step 4] 참조 모델 로드: {args.ref_model}")
    ref_pcd = load_reference_model(args.ref_model, num_samples=args.ref_samples)

    print("\n[Step 5] 스케일 정합")
    ref_scaled, scale = scale_reference_to_observation(
        ref_pcd, obs_pcd,
        ref_length_mm=args.ref_length_mm, manual_scale=args.scale)

    voxel_size_m = args.voxel_size / 1000.0
    icp_dist_m = args.icp_dist / 1000.0
    init_transforms = []

    if args.init_method in ("fpfh", "both"):
        print("\n[Step 6a] 전역 정합 (FPFH + RANSAC)")
        try:
            T_fpfh = global_registration_fpfh(ref_scaled, obs_pcd, voxel_size=voxel_size_m)
            init_transforms.append(("FPFH", T_fpfh))
        except Exception as e:
            print(f"  [WARN] FPFH 전역 정합 실패: {e}")

    if args.init_method in ("pca", "both"):
        print("\n[Step 6b] PCA 초기 정렬")
        T_pca = initial_alignment_pca(np.asarray(ref_scaled.points), obs_points)
        init_transforms.append(("PCA", T_pca))
        print("    PCA 초기 정렬 완료")

    if not init_transforms:
        raise RuntimeError("초기 정합 방법이 모두 실패")

    print(f"\n[Step 7] 정밀 정합 (Point-to-Plane ICP, max_dist={args.icp_dist:.1f}mm)")
    best_result = None
    for method_name, init_T in init_transforms:
        print(f"\n  --- ICP with {method_name} init ---")
        T, fitness, rmse = run_icp_multistage(
            ref_scaled, obs_pcd,
            init_transform=init_T, max_corr_dist=icp_dist_m, fine_iters=args.icp_iters)
        if best_result is None or fitness > best_result[1] or \
           (fitness == best_result[1] and rmse < best_result[2]):
            best_result = (T, fitness, rmse, method_name)

    T_final, fitness, rmse, best_method = best_result
    print(f"\n  최종 채택: {best_method} (fitness={fitness:.4f}, RMSE={rmse*1000:.3f}mm)")

    R = T_final[:3, :3]
    t_vec = T_final[:3, 3]
    ref_aligned = np.asarray(ref_scaled.points) @ R.T + t_vec

    return R, t_vec, scale, obs_points, obs_colors, ref_aligned, fitness, rmse


# ---------------------------------------------------------------
# 파이프라인 B: SAM3D 통합
# ---------------------------------------------------------------
def _main_sam3d(args, session):
    """SAM3D 좌표계 변환 + (선택) ICP 정밀 정합 파이프라인.

    SAM3D 좌표 변환 체인:
        P_canonical (SAM3D 정규화)
          ↓  scale * R_sam3d @ P + t_sam3d
        P_sam3d (SAM3D output, OpenGL Y-up)
          ↓  Y,Z 반전 (OpenGL → OpenCV)
        P_cam_i (카메라_i 좌표계)
          ↓  T_C0_Ci (외부 캘리브레이션)
        P_cam0 (월드좌표계, 최종 포즈)
    """
    sam3d_cam = args.sam3d_cam
    print(f"\n[SAM3D] 파이프라인 시작 (source camera: cam{sam3d_cam})")

    # --- 1. SAM3D 파일 탐지 ---
    if args.sam3d_dir:
        files = auto_detect_sam3d_files(args.sam3d_dir)
    else:
        files = {}

    ply_path = args.sam3d_ply or files.get("ply")
    pose_path = args.sam3d_pose or files.get("pose")
    pointmap_path = args.sam3d_pointmap or files.get("pointmap")
    checkpoint_path = files.get("checkpoint")

    # --- 2. SAM3D 점군 로드 ---
    print("\n[SAM3D Step 1] 3D 데이터 로드")
    sam3d_pts = sam3d_colors = None

    if pointmap_path:
        print(f"  pointmap에서 로드: {pointmap_path}")
        sam3d_pts, sam3d_colors = load_sam3d_pointmap(pointmap_path)
    elif ply_path:
        print(f"  Gaussian Splat PLY에서 로드: {ply_path}")
        sam3d_pts, sam3d_colors = load_sam3d_splat_ply(ply_path)
    else:
        raise RuntimeError("SAM3D 점군 소스 없음 (--sam3d_ply 또는 --sam3d_dir 필요)")

    # --- 3. SAM3D 포즈 로드 ---
    print("\n[SAM3D Step 2] 포즈 파라미터 (rotation/translation/scale)")
    R_sam3d = t_sam3d = None
    scale_sam3d = 1.0
    has_pose = False

    if pose_path:
        R_sam3d, t_sam3d, scale_sam3d = load_sam3d_pose(pose_path)
        has_pose = True
    elif checkpoint_path:
        try:
            R_sam3d, t_sam3d, scale_sam3d = load_sam3d_pose(checkpoint_path)
            has_pose = True
        except Exception as e:
            print(f"  checkpoint에서 포즈 로드 실패: {e}")

    if not has_pose:
        print("  포즈 파일 없음 → 항등 변환 사용 (스케일 자동 추정 예정)")

    # --- 4. SAM3D → cam0 좌표 변환 ---
    print(f"\n[SAM3D Step 3] 좌표 변환 체인")
    T_C0_Ci = session.calibrations[sam3d_cam].T_cam0_cam

    print(f"  [1] SAM3D canonical → SAM3D output")
    if R_sam3d is not None:
        print(f"      R_sam3d 적용 (det={np.linalg.det(R_sam3d):.4f})")
    print(f"      scale = {scale_sam3d:.6f}")
    if t_sam3d is not None:
        print(f"      t_sam3d = ({t_sam3d[0]:.4f}, {t_sam3d[1]:.4f}, {t_sam3d[2]:.4f})")

    gl_to_cv = not args.sam3d_opencv
    print(f"  [2] {'OpenGL(Y-up) → OpenCV(Y-down): Y,Z 반전' if gl_to_cv else 'OpenCV 좌표 (변환 없음)'}")
    print(f"  [3] cam{sam3d_cam} → cam0: T_C0_C{sam3d_cam}")

    pts_cam0 = sam3d_to_cam0(
        sam3d_pts, R_sam3d, t_sam3d, scale_sam3d,
        T_C0_Ci, sam3d_is_opengl=gl_to_cv)

    bbox_mm = (pts_cam0.max(0) - pts_cam0.min(0)) * 1000
    center_mm = pts_cam0.mean(0) * 1000
    print(f"\n  cam0 변환 결과:")
    print(f"    {len(pts_cam0):,} pts")
    print(f"    bbox: {bbox_mm[0]:.1f} x {bbox_mm[1]:.1f} x {bbox_mm[2]:.1f} mm")
    print(f"    center: ({center_mm[0]:.1f}, {center_mm[1]:.1f}, {center_mm[2]:.1f}) mm")

    # --- 5. 관측 점군과 스케일/위치 자동 조정 (포즈 없는 경우) ---
    if not has_pose:
        print("\n[SAM3D Step 4] 자동 스케일 + 위치 정합 (포즈 없음)")
        obs_pts, obs_colors = _get_observation_points(args, session)
        obs_extent = obs_pts.max(0) - obs_pts.min(0)
        sam_extent = pts_cam0.max(0) - pts_cam0.min(0)
        auto_scale = float(np.median(obs_extent / (sam_extent + 1e-8)))
        print(f"    관측 bbox: {obs_extent*1000}")
        print(f"    SAM3D bbox: {sam_extent*1000}")
        print(f"    자동 스케일: {auto_scale:.4f}")

        sam_center = pts_cam0.mean(0)
        pts_cam0 = (pts_cam0 - sam_center) * auto_scale + obs_pts.mean(0)
        scale_sam3d *= auto_scale

    # --- 6. (선택) ICP 정밀 정합 ---
    fitness, rmse = 1.0, 0.0

    if args.sam3d_refine_icp:
        print(f"\n[SAM3D Step 5] ICP 정밀 정합 (관측 점군 대비)")
        obs_pts, obs_colors = _get_observation_points(args, session)
        obs_pcd = build_pointcloud(obs_pts, obs_colors)
        sam_pcd = build_pointcloud(pts_cam0, sam3d_colors)

        icp_dist_m = args.icp_dist / 1000.0
        T_refine, fitness, rmse = run_icp_multistage(
            sam_pcd, obs_pcd,
            init_transform=np.eye(4),
            max_corr_dist=icp_dist_m,
            fine_iters=args.icp_iters)
        print(f"    ICP 보정 결과: fitness={fitness:.4f}, RMSE={rmse*1000:.3f}mm")
        pts_cam0 = pts_cam0 @ T_refine[:3, :3].T + T_refine[:3, 3]

    # --- 7. 포즈 추출 ---
    R, t_vec = _extract_pose_from_pointcloud(pts_cam0)

    return R, t_vec, scale_sam3d, pts_cam0, sam3d_colors, fitness, rmse


if __name__ == "__main__":
    main()
