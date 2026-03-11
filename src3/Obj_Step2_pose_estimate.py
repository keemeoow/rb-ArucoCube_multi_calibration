#!/usr/bin/env python3
"""
Obj_Step2_pose_estimate.py - GLB 참조 모델 기반 물체 6-DOF 포즈 추정
=====================================================================
포즈 추정 모드 (--pose_mode):
  icp        : Depth 점군 + ICP 정합 (기본)
  foundation : FoundationPose 딥러닝 모델 (실패 시 ICP fallback)

세그멘테이션 모드 (--seg_mode):
  hsv        : HSV 앵커 + depth 연결 object mask (기본)
  sam2       : GroundingDINO + SAM2
  depth_roi  : depth ROI + 테이블 제거 + DBSCAN

좌표계: cam0 (OpenCV: X-right, Y-down, Z-forward)

사용법:
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_knife.glb
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_knife.glb --seg_mode depth_roi
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_knife.glb --pose_mode foundation
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CAP_DIR = _THIS_DIR / "data/object_capture"
_DEFAULT_CAL_DIR = _THIS_DIR / "data/cube_session_01/calib_out_cube"
_DEFAULT_INT_DIR = _THIS_DIR / "data/_intrinsics"
_DEFAULT_OUT_DIR = _THIS_DIR / "Obj_Step2_output"


@dataclass(frozen=True)
class CameraCalibration:
    K: np.ndarray
    D: np.ndarray
    depth_scale_m_per_unit: float
    T_cam0_cam: np.ndarray


@dataclass
class FrameCapture:
    camera_id: int
    rgb_path: Path
    depth_path: Path
    bgr: Optional[np.ndarray]
    depth: Optional[np.ndarray]


@dataclass
class CaptureSession:
    camera_ids: Tuple[int, ...]
    pad: int
    calibrations: Dict[int, CameraCalibration]
    frames: Dict[int, FrameCapture]


@dataclass
class ObservationData:
    points: np.ndarray
    colors: Optional[np.ndarray]
    anchor_center: Optional[np.ndarray] = None
    blade_dir_hint: Optional[np.ndarray] = None
    masks: Optional[Dict[int, np.ndarray]] = None


def load_reference_model(path: str) -> "open3d.geometry.PointCloud":
    """GLB/PLY/OBJ 파일을 Open3D point cloud로 로드한다."""
    import open3d as o3d

    ref_path = Path(path)
    ext = ref_path.suffix.lower()

    if ext == ".ply":
        mesh = o3d.io.read_triangle_mesh(str(ref_path))
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=30000)
        else:
            pcd = o3d.io.read_point_cloud(str(ref_path))
    elif ext in {".glb", ".gltf", ".obj"}:
        import trimesh

        scene = trimesh.load(str(ref_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            mesh_tm = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
        else:
            mesh_tm = scene

        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_tm.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_tm.faces)
        if mesh_tm.visual is not None and hasattr(mesh_tm.visual, "vertex_colors"):
            vertex_colors = mesh_tm.visual.vertex_colors[:, :3] / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        mesh_o3d.compute_vertex_normals()
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=30000)
    else:
        raise ValueError(f"지원하지 않는 참조 모델 형식: {ext}")

    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise RuntimeError(f"참조 모델이 비어있음: {path}")

    if ext in {".glb", ".gltf"}:
        points[:, 1] *= -1.0
        points[:, 2] *= -1.0
        print("  [좌표계 변환] OpenGL(Y-up) -> OpenCV(Y-down): Y,Z 반전")

    centroid = points.mean(axis=0)
    points -= centroid
    pcd.points = o3d.utility.Vector3dVector(points)

    bbox_size = points.max(axis=0) - points.min(axis=0)
    print(f"  참조 모델: {len(points):,} pts")
    print(f"  원본 bbox: {bbox_size[0]:.4f} x {bbox_size[1]:.4f} x {bbox_size[2]:.4f}")
    return pcd


def scale_reference_to_observation(
    ref_pcd: "open3d.geometry.PointCloud",
    obs_pcd: "open3d.geometry.PointCloud",
    ref_length_mm: Optional[float] = None,
    manual_scale: Optional[float] = None,
) -> Tuple["open3d.geometry.PointCloud", float]:
    """참조 모델을 관측 점군 크기에 맞춘다."""
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

    scaled_extent = np.asarray(ref_scaled.points).max(axis=0) - np.asarray(ref_scaled.points).min(axis=0)
    print(
        "  스케일 후 크기: "
        f"{scaled_extent[0] * 1000:.1f} x {scaled_extent[1] * 1000:.1f} x {scaled_extent[2] * 1000:.1f} mm"
    )
    return ref_scaled, scale


def load_intrinsics(intrin_dir: str, camera_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
    data = np.load(Path(intrin_dir) / f"cam{camera_id}.npz", allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64) if "color_D" in data else np.zeros(5, dtype=np.float64)
    depth_scale = float(data.get("depth_scale_m_per_unit", 0.001))
    return K, D, depth_scale


def discover_cameras(capture_dir: str) -> Tuple[int, ...]:
    camera_ids = []
    for directory in sorted(glob.glob(str(Path(capture_dir) / "cam*"))):
        try:
            camera_id = int(Path(directory).name.replace("cam", ""))
        except ValueError:
            continue
        if glob.glob(str(Path(directory) / "rgb_*.jpg")):
            camera_ids.append(camera_id)
    if not camera_ids:
        raise RuntimeError(f"cam* 폴더를 찾지 못함: {capture_dir}")
    return tuple(camera_ids)


def frame_pad(capture_dir: str, camera_id: int) -> int:
    files = glob.glob(str(Path(capture_dir) / f"cam{camera_id}" / "rgb_*.jpg"))
    if not files:
        return 6
    stem = Path(files[0]).stem.replace("rgb_", "")
    return len(stem)


def load_extrinsics(calib_dir: str, camera_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    transforms: Dict[int, np.ndarray] = {}
    for camera_id in camera_ids:
        if camera_id == 0:
            transforms[camera_id] = np.eye(4, dtype=np.float64)
        else:
            transforms[camera_id] = np.load(Path(calib_dir) / f"T_C0_C{camera_id}.npy").astype(np.float64)
    return transforms


def frame_id(frame_idx: int, pad: int) -> str:
    return f"{frame_idx:0{pad}d}"


def load_capture_session(args: argparse.Namespace) -> CaptureSession:
    camera_ids = discover_cameras(args.capture_dir)
    pad = frame_pad(args.capture_dir, camera_ids[0])
    transforms = load_extrinsics(args.calib_dir, camera_ids)

    calibrations: Dict[int, CameraCalibration] = {}
    for camera_id in camera_ids:
        K, D, depth_scale = load_intrinsics(args.intrinsics_dir, camera_id)
        calibrations[camera_id] = CameraCalibration(
            K=K,
            D=D,
            depth_scale_m_per_unit=depth_scale,
            T_cam0_cam=transforms[camera_id],
        )

    fid = frame_id(args.frame, pad)
    frames: Dict[int, FrameCapture] = {}
    for camera_id in camera_ids:
        rgb_path = Path(args.capture_dir) / f"cam{camera_id}" / f"rgb_{fid}.jpg"
        depth_path = Path(args.capture_dir) / f"cam{camera_id}" / f"depth_{fid}.png"
        bgr = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED) if depth_path.exists() else None
        frames[camera_id] = FrameCapture(
            camera_id=camera_id,
            rgb_path=rgb_path,
            depth_path=depth_path,
            bgr=bgr,
            depth=depth,
        )

    return CaptureSession(
        camera_ids=camera_ids,
        pad=pad,
        calibrations=calibrations,
        frames=frames,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    return points @ transform[:3, :3].T + transform[:3, 3]


def depth_to_points(
    depth_u16: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    depth_scale_m_per_unit: float,
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """depth + mask를 3D 점군과 해당 픽셀 좌표로 변환한다."""
    height, width = depth_u16.shape[:2]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

    vg, ug = np.mgrid[0:height, 0:width]
    z = depth_u16.astype(np.float64) * depth_scale_m_per_unit
    valid = mask & (z > z_min) & (z < z_max)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    z = z[valid]
    u = ug[valid].astype(np.float64)
    v = vg[valid].astype(np.float64)
    undistorted = cv2.undistortPoints(
        np.column_stack([u, v]).reshape(-1, 1, 2).astype(np.float64),
        K,
        D,
    ).reshape(-1, 2)
    xyz = np.column_stack([undistorted[:, 0] * z, undistorted[:, 1] * z, z])
    return xyz, np.column_stack([u, v])


def sample_colors(bgr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    ui = uv[:, 0].astype(int)
    vi = uv[:, 1].astype(int)
    return bgr[vi, ui][:, ::-1] / 255.0


def masked_points_in_cam0(
    frame: FrameCapture,
    calibration: CameraCalibration,
    mask: np.ndarray,
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if frame.bgr is None or frame.depth is None:
        return np.empty((0, 3), dtype=np.float64), None

    points_cam, uv = depth_to_points(
        frame.depth,
        mask,
        calibration.K,
        calibration.D,
        calibration.depth_scale_m_per_unit,
        z_min,
        z_max,
    )
    if len(points_cam) == 0:
        return np.empty((0, 3), dtype=np.float64), None

    colors = sample_colors(frame.bgr, uv)
    points_cam0 = transform_points(points_cam, calibration.T_cam0_cam)
    return points_cam0, colors


def collect_full_depth_points(
    session: CaptureSession,
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    all_points = []
    all_colors = []

    for camera_id in session.camera_ids:
        frame = session.frames[camera_id]
        calibration = session.calibrations[camera_id]
        if frame.bgr is None or frame.depth is None:
            continue

        mask = np.ones(frame.depth.shape[:2], dtype=bool)
        points_cam0, colors = masked_points_in_cam0(frame, calibration, mask, z_min, z_max)
        if len(points_cam0) == 0 or colors is None:
            continue

        all_points.append(points_cam0)
        all_colors.append(colors)
        print(f"    cam{camera_id}: {len(points_cam0):,} pts -> cam0")

    if not all_points:
        return np.empty((0, 3), dtype=np.float64), None
    return np.concatenate(all_points), np.concatenate(all_colors)


def fuse_multicam_masks(
    session: CaptureSession,
    masks: Dict[int, np.ndarray],
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    all_points = []
    all_colors = []

    for camera_id in session.camera_ids:
        mask = masks.get(camera_id)
        if mask is None:
            continue

        frame = session.frames[camera_id]
        calibration = session.calibrations[camera_id]
        points_cam0, colors = masked_points_in_cam0(frame, calibration, mask, z_min, z_max)
        if len(points_cam0) == 0 or colors is None:
            continue

        all_points.append(points_cam0)
        all_colors.append(colors)
        print(f"    cam{camera_id}: {len(points_cam0):,} pts -> cam0")

    if not all_points:
        return np.empty((0, 3), dtype=np.float64), None
    return np.concatenate(all_points), np.concatenate(all_colors)


def sor(points: np.ndarray, colors: Optional[np.ndarray] = None, std_ratio: float = 1.5) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """IQR + 거리 기반 간단한 이상치 제거."""
    num_points = len(points)
    if num_points < 10:
        return points, colors

    mask = np.ones(num_points, dtype=bool)
    for axis in range(3):
        values = points[:, axis]
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        mask &= (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)

    center = points[mask].mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    mean_dist = distances[mask].mean()
    std_dist = distances[mask].std()
    mask &= distances < mean_dist + std_ratio * std_dist

    print(f"    SOR: {num_points:,} -> {mask.sum():,}")
    return points[mask], (colors[mask] if colors is not None else None)


def remove_plane(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    distance_threshold: float = 0.005,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    import open3d as o3d

    if len(points) < 50:
        raise RuntimeError("평면 제거를 수행하기에 점군이 너무 적음")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    _, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)

    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[inliers] = True
    filtered_points = points[~inlier_mask]
    filtered_colors = colors[~inlier_mask] if colors is not None else None
    print(f"    테이블 제거 후: {len(filtered_points):,} pts")
    return filtered_points, filtered_colors


def keep_largest_cluster(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    eps: float = 0.005,
    min_samples: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    from sklearn.cluster import DBSCAN

    if len(points) < min_samples:
        raise RuntimeError("클러스터링을 수행하기에 점군이 너무 적음")

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        raise RuntimeError("DBSCAN 클러스터 없음")

    unique, counts = np.unique(valid_labels, return_counts=True)
    best_label = unique[np.argmax(counts)]
    mask = labels == best_label
    print(f"    DBSCAN: {len(unique)} 클러스터, 최대={mask.sum():,} pts 채택")
    return points[mask], (colors[mask] if colors is not None else None)


def compute_hsv_mask(
    bgr: np.ndarray,
    h_range: Sequence[int],
    s_min: int,
    v_min: int,
) -> np.ndarray:
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


def largest_connected_component(mask: np.ndarray, min_area: int = 1) -> Optional[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    best_label = -1
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_label = label
            best_area = area
    if best_label < 0:
        return None
    return labels == best_label


def component_touching_seed(
    candidate_mask: np.ndarray,
    seed_mask: np.ndarray,
    min_area: int,
) -> Optional[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return None

    seed_dilated = cv2.dilate(
        seed_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    ).astype(bool)

    keep_mask = np.zeros_like(candidate_mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component_mask = labels == label
        if np.any(component_mask & seed_dilated):
            keep_mask |= component_mask

    if not keep_mask.any():
        return None
    return keep_mask


def oriented_anchor_roi(anchor_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(anchor_mask)
    anchor_pixels = np.column_stack([xs, ys]).astype(np.float64)
    center = anchor_pixels.mean(axis=0)
    centered = anchor_pixels - center
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    projection = centered @ axes

    long_half = max((projection[:, 0].max() - projection[:, 0].min()) * 0.5 * 5.5, 55.0)
    short_half = max((projection[:, 1].max() - projection[:, 1].min()) * 0.5 * 2.5, 22.0)

    height, width = anchor_mask.shape[:2]
    vg, ug = np.mgrid[0:height, 0:width]
    grid = np.column_stack([ug.reshape(-1), vg.reshape(-1)]).astype(np.float64)
    grid_projection = (grid - center) @ axes
    roi = (
        (grid_projection[:, 0] / (long_half + 16.0)) ** 2 +
        (grid_projection[:, 1] / (short_half + 16.0)) ** 2
    ) <= 1.0
    return roi.reshape(height, width)


def extract_hsv_object_mask(
    frame: FrameCapture,
    calibration: CameraCalibration,
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if frame.bgr is None or frame.depth is None:
        return None, None, None

    yellow_mask = compute_hsv_mask(frame.bgr, args.hsv_h_range, args.hsv_s_min, args.hsv_v_min)
    anchor_mask = largest_connected_component(yellow_mask, args.min_component_area)
    if anchor_mask is None:
        return None, None, None

    anchor_points_cam, _ = depth_to_points(
        frame.depth,
        anchor_mask,
        calibration.K,
        calibration.D,
        calibration.depth_scale_m_per_unit,
        args.z_min,
        args.z_max,
    )
    if len(anchor_points_cam) == 0:
        return anchor_mask, None, None

    depth_m = frame.depth.astype(np.float64) * calibration.depth_scale_m_per_unit
    anchor_depth_values = depth_m[anchor_mask & (depth_m > args.z_min) & (depth_m < args.z_max)]
    if len(anchor_depth_values) == 0:
        return anchor_mask, None, anchor_points_cam

    anchor_depth = float(np.median(anchor_depth_values))
    roi_mask = oriented_anchor_roi(anchor_mask)
    object_mask: Optional[np.ndarray] = None

    for front_tol, back_tol in [(0.015, 0.018), (0.020, 0.020), (0.020, 0.025)]:
        depth_ok = (
            (frame.depth > 0) &
            (depth_m > max(args.z_min, anchor_depth - front_tol)) &
            (depth_m < min(args.z_max, anchor_depth + back_tol))
        )
        candidate_mask = roi_mask & depth_ok
        object_mask = component_touching_seed(
            candidate_mask,
            anchor_mask,
            min_area=max(120, args.min_component_area // 4),
        )
        if object_mask is None:
            continue

        object_mask = cv2.morphologyEx(
            object_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        ).astype(bool)
        object_mask |= anchor_mask

        area_ratio = object_mask.sum() / max(anchor_mask.sum(), 1)
        if 1.05 <= area_ratio <= 5.0:
            break
    else:
        object_mask = anchor_mask.copy()

    return anchor_mask, object_mask, anchor_points_cam


def run_hsv_anchor_crop_segmentation(
    args: argparse.Namespace,
    session: CaptureSession,
    anchor_center: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    all_points = []
    all_colors = []
    sphere_radius = 0.12

    for camera_id in session.camera_ids:
        frame = session.frames[camera_id]
        calibration = session.calibrations[camera_id]
        if frame.bgr is None or frame.depth is None:
            continue

        full_mask = np.ones(frame.depth.shape[:2], dtype=bool)
        points_cam, uv = depth_to_points(
            frame.depth,
            full_mask,
            calibration.K,
            calibration.D,
            calibration.depth_scale_m_per_unit,
            args.z_min,
            args.z_max,
        )
        if len(points_cam) == 0:
            continue

        points_cam0 = transform_points(points_cam, calibration.T_cam0_cam)
        in_sphere = np.linalg.norm(points_cam0 - anchor_center, axis=1) < sphere_radius
        if not in_sphere.any():
            continue

        all_points.append(points_cam0[in_sphere])
        all_colors.append(sample_colors(frame.bgr, uv[in_sphere]))

    if not all_points:
        raise RuntimeError("HSV 세그멘테이션 실패")

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)

    centered = points - anchor_center
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    projection = centered @ eigenvectors[:, order]

    a_long, a_mid, a_short = 0.100, 0.018, 0.018
    ellipsoid_dist = (
        (projection[:, 0] / a_long) ** 2 +
        (projection[:, 1] / a_mid) ** 2 +
        (projection[:, 2] / a_short) ** 2
    )
    keep_mask = ellipsoid_dist < 1.0
    print(f"    [fallback] 타원체 크롭: {keep_mask.sum():,} pts")
    return points[keep_mask], colors[keep_mask]


def estimate_blade_dir_from_anchor_pts(anchor_pts_3d: np.ndarray) -> Optional[np.ndarray]:
    """앵커(노란 손잡이) 3D 점군의 PCA 주축으로부터 날 방향 힌트를 추정한다.
    부호: 칼날이 이미지 오른쪽(+X 방향)을 향하도록 정렬한다."""
    if len(anchor_pts_3d) < 10:
        return None
    c = anchor_pts_3d.mean(axis=0)
    cov = (anchor_pts_3d - c).T @ (anchor_pts_3d - c) / len(anchor_pts_3d)
    _, evecs = np.linalg.eigh(cov)
    blade_dir = evecs[:, -1].copy()  # 가장 큰 고유값 → 긴 축
    if blade_dir[0] < 0:             # +X (이미지 오른쪽) 방향으로 부호 통일
        blade_dir = -blade_dir
    return blade_dir


def run_hsv_segmentation(args: argparse.Namespace, session: CaptureSession) -> ObservationData:
    """노란 손잡이 앵커를 seed로 각 카메라 object mask를 만든 뒤 융합한다."""
    anchor_points = []
    masks: Dict[int, np.ndarray] = {}

    for camera_id in session.camera_ids:
        frame = session.frames[camera_id]
        calibration = session.calibrations[camera_id]
        if frame.bgr is None or frame.depth is None:
            continue

        anchor_mask, object_mask, anchor_points_cam = extract_hsv_object_mask(frame, calibration, args)
        if anchor_mask is None or anchor_points_cam is None:
            continue
        points_cam0 = transform_points(anchor_points_cam, calibration.T_cam0_cam)
        anchor_points.append(points_cam0)
        print(f"    cam{camera_id}: HSV 앵커 {len(points_cam0):,} pts")
        if object_mask is not None and object_mask.sum() > 0:
            masks[camera_id] = object_mask
            print(f"    cam{camera_id}: object mask {object_mask.sum():,} px")

    if not anchor_points:
        raise RuntimeError("노란 손잡이 앵커 검출 실패")

    anchor_center = np.concatenate(anchor_points).mean(axis=0)
    print(
        "    앵커 중심 (cam0): "
        f"({anchor_center[0] * 1000:.1f}, {anchor_center[1] * 1000:.1f}, {anchor_center[2] * 1000:.1f}) mm"
    )

    if masks:
        points, colors = fuse_multicam_masks(session, masks, args.z_min, args.z_max)
    else:
        points = np.empty((0, 3), dtype=np.float64)
        colors = None

    if len(points) < 50 or colors is None:
        print("    [fallback] 마스크 기반 추출 부족 -> 3D 앵커 크롭 사용")
        points, colors = run_hsv_anchor_crop_segmentation(args, session, anchor_center)

    if len(points) < 50:
        raise RuntimeError("세그멘테이션 결과 부족")

    points, colors = sor(points, colors)
    points, colors = keep_largest_cluster(points, colors, eps=0.005, min_samples=10)

    blade_dir_hint = estimate_blade_dir_from_anchor_pts(np.concatenate(anchor_points))
    if blade_dir_hint is not None:
        print(
            "    3D 앵커 PCA 날방향 힌트: "
            f"({blade_dir_hint[0]:.3f}, {blade_dir_hint[1]:.3f}, {blade_dir_hint[2]:.3f})"
        )

    return ObservationData(
        points=points,
        colors=colors,
        anchor_center=anchor_center,
        blade_dir_hint=blade_dir_hint,
        masks=(masks if masks else None),
    )


def run_depth_roi_segmentation(args: argparse.Namespace, session: CaptureSession) -> ObservationData:
    """depth ROI로 배경을 줄이고 가장 큰 클러스터를 선택한다."""
    points, colors = collect_full_depth_points(session, args.z_min, args.z_max)
    if len(points) == 0:
        raise RuntimeError("depth 점군 없음")

    points, colors = remove_plane(points, colors, distance_threshold=0.005)
    if len(points) < 50:
        raise RuntimeError("평면 제거 후 점군 부족")

    points, colors = sor(points, colors)
    points, colors = keep_largest_cluster(points, colors, eps=0.005, min_samples=10)
    return ObservationData(points=points, colors=colors)


def run_sam2_segmentation(args: argparse.Namespace, session: CaptureSession) -> ObservationData:
    """GroundingDINO + SAM2로 각 카메라의 object mask를 만든 뒤 융합한다."""
    sam_dir = _THIS_DIR / "Obj_Step2-(1)_pose_estimate_grounding_sam"
    if sam_dir.is_dir():
        sam_dir_str = str(sam_dir)
        if sam_dir_str not in sys.path:
            sys.path.insert(0, sam_dir_str)

    try:
        import torch
        from PIL import Image as PILImage
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("SAM2 또는 GroundingDINO 패키지가 설치되지 않음. --seg_mode hsv 를 사용하세요.") from exc

    print(f"  GroundingDINO + SAM2 로딩 (device={args.device})...")
    processor = AutoProcessor.from_pretrained(args.gdino_model)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model).to(args.device)
    predictor = SAM2ImagePredictor(build_sam2(args.sam2_config, args.sam2_checkpoint))
    predictor.model = predictor.model.to(args.device)

    masks: Dict[int, np.ndarray] = {}

    for camera_id in session.camera_ids:
        frame = session.frames[camera_id]
        if frame.bgr is None:
            continue

        rgb_image = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        inputs = processor(images=pil_image, text=args.text_prompt, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = detector(**inputs)

        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        if len(result["boxes"]) == 0:
            print(f"    cam{camera_id}: 검출 없음")
            continue

        best_idx = int(result["scores"].argmax().item())
        box = result["boxes"][best_idx].cpu().numpy()
        score = float(result["scores"][best_idx].item())
        print(f"    cam{camera_id}: score={score:.3f}")

        predictor.set_image(rgb_image)
        predicted_masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)
        masks[camera_id] = predicted_masks[0].astype(bool)

    if not masks:
        raise RuntimeError("모든 카메라에서 검출 실패")

    points, colors = fuse_multicam_masks(session, masks, args.z_min, args.z_max)
    points, colors = sor(points, colors)
    if len(points) < 50:
        raise RuntimeError("융합 점군 부족")

    return ObservationData(points=points, colors=colors, masks=masks)


def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    eigenvalues = eigenvalues[order]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    return center, axes, eigenvalues


def initial_alignment_pca(ref_points: np.ndarray, obs_points: np.ndarray) -> np.ndarray:
    ref_center, ref_axes, _ = pca_axes(ref_points)
    obs_center, obs_axes, _ = pca_axes(obs_points)

    rotation = obs_axes @ ref_axes.T
    if np.linalg.det(rotation) < 0:
        obs_axes[:, 2] *= -1
        rotation = obs_axes @ ref_axes.T

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = obs_center - rotation @ ref_center
    return transform


def initial_alignment_with_blade_dir(
    ref_points: np.ndarray,
    obs_points: np.ndarray,
    blade_dir: np.ndarray,
) -> np.ndarray:
    ref_center, ref_axes, _ = pca_axes(ref_points)
    obs_center = obs_points.mean(axis=0)

    z_new = blade_dir / np.linalg.norm(blade_dir)
    y_cam = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_new = y_cam - np.dot(y_cam, z_new) * z_new
    if np.linalg.norm(y_new) < 1e-6:
        y_new = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_new -= np.dot(y_new, z_new) * z_new
    y_new /= np.linalg.norm(y_new)
    x_new = np.cross(y_new, z_new)
    x_new /= np.linalg.norm(x_new)

    target_axes = np.column_stack([x_new, y_new, z_new])
    if np.linalg.det(target_axes) < 0:
        target_axes[:, 0] *= -1

    rotation = target_axes @ ref_axes.T
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = obs_center - rotation @ ref_center
    return transform


def blade_dir_score(
    transform: np.ndarray,
    ref_points: np.ndarray,
    obs_points: np.ndarray,
    anchor_center: np.ndarray,
) -> float:
    num_samples = max(len(obs_points) // 10, 1)

    obs_dist = np.linalg.norm(obs_points - anchor_center, axis=1)
    obs_blade = obs_points[np.argsort(obs_dist)[-num_samples:]].mean(axis=0) - anchor_center
    obs_norm = np.linalg.norm(obs_blade)
    if obs_norm < 1e-8:
        return 0.0
    obs_blade /= obs_norm

    ref_transformed = ref_points @ transform[:3, :3].T + transform[:3, 3]
    ref_dist = np.linalg.norm(ref_transformed - anchor_center, axis=1)
    ref_blade = ref_transformed[np.argsort(ref_dist)[-num_samples:]].mean(axis=0) - anchor_center
    ref_norm = np.linalg.norm(ref_blade)
    if ref_norm < 1e-8:
        return 0.0
    ref_blade /= ref_norm

    return float(np.dot(obs_blade, ref_blade))


def run_icp_stages(
    ref_pcd: "open3d.geometry.PointCloud",
    obs_pcd: "open3d.geometry.PointCloud",
    init_transform: np.ndarray,
    max_correspondence_dist: float,
    num_iterations: int,
    anchor_center: Optional[np.ndarray] = None,
    blade_dir_hint: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    import open3d as o3d

    ref_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    obs_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    stages = [
        (max_correspondence_dist * 4, 30, "coarse"),
        (max_correspondence_dist * 2, 30, "medium"),
        (max_correspondence_dist, num_iterations, "fine"),
    ]

    def run_once(start_transform: np.ndarray) -> Tuple[np.ndarray, float, float]:
        current = start_transform
        result = None
        for distance, iterations, name in stages:
            result = o3d.pipelines.registration.registration_icp(
                ref_pcd,
                obs_pcd,
                max_correspondence_distance=distance,
                init=current,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=iterations,
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                ),
            )
            current = result.transformation
            print(
                f"    ICP [{name}] dist={distance * 1000:.1f}mm: "
                f"fitness={result.fitness:.4f}, RMSE={result.inlier_rmse * 1000:.3f}mm"
            )
        assert result is not None
        return current, float(result.fitness), float(result.inlier_rmse)

    candidates = []
    base_transform, base_fitness, base_rmse = run_once(init_transform)
    candidates.append((base_transform, base_fitness, base_rmse, "원래"))

    for index, signs in enumerate(
        [
            np.diag([-1, -1, 1]),
            np.diag([1, -1, -1]),
            np.diag([-1, 1, -1]),
        ],
        start=1,
    ):
        flipped_init = init_transform.copy()
        flipped_init[:3, :3] = init_transform[:3, :3] @ signs
        candidate_transform, candidate_fitness, candidate_rmse = run_once(flipped_init)
        candidates.append((candidate_transform, candidate_fitness, candidate_rmse, f"flip{index}"))
        print(
            f"    ICP [flip{index}] fitness={candidate_fitness:.4f}, "
            f"RMSE={candidate_rmse * 1000:.3f}mm"
        )

    ref_points = np.asarray(ref_pcd.points)

    if anchor_center is not None:
        if blade_dir_hint is not None:
            def ref_blade_direction(transform: np.ndarray) -> np.ndarray:
                transformed = ref_points @ transform[:3, :3].T + transform[:3, 3]
                num_samples = max(len(transformed) // 10, 1)
                distances = np.linalg.norm(transformed - anchor_center, axis=1)
                blade_vec = transformed[np.argsort(distances)[-num_samples:]].mean(axis=0) - anchor_center
                norm = np.linalg.norm(blade_vec)
                return blade_vec / norm if norm > 1e-8 else blade_vec

            scores = [abs(float(np.dot(blade_dir_hint, ref_blade_direction(candidate[0])))) for candidate in candidates]
            label = "2D 이미지 날방향"
        else:
            obs_points = np.asarray(obs_pcd.points)
            scores = [blade_dir_score(candidate[0], ref_points, obs_points, anchor_center) for candidate in candidates]
            label = "날 방향 일치도"

        print(f"    {label}: " + "  ".join(f"{candidate[3]}={score:.3f}" for candidate, score in zip(candidates, scores)))
        best_score = max(scores)
        compatible = [
            candidate
            for candidate, score in zip(candidates, scores)
            if score >= best_score - 0.05
        ]
        best = max(compatible, key=lambda item: (item[1], -item[2]))
        print(f"    -> '{best[3]}' 채택 ({label} 기준)")
        return best[0], best[1], best[2]

    stable_candidates = [candidate for candidate in candidates if candidate[1] > 0.5]
    if stable_candidates:
        best = min(stable_candidates, key=lambda item: item[2])
        print(f"    -> '{best[3]}' 채택 (RMSE 기준)")
        return best[0], best[1], best[2]

    best = max(candidates, key=lambda item: (item[1], -item[2]))
    print(f"    -> '{best[3]}' 채택 (fitness 기준)")
    return best[0], best[1], best[2]


def estimate_pose_icp(
    ref_pcd: "open3d.geometry.PointCloud",
    obs_pcd: "open3d.geometry.PointCloud",
    max_correspondence_dist: float = 0.01,
    num_iterations: int = 100,
    anchor_center: Optional[np.ndarray] = None,
    blade_dir_hint: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    ref_points = np.asarray(ref_pcd.points)
    obs_points = np.asarray(obs_pcd.points)

    if blade_dir_hint is not None:
        init_transform = initial_alignment_with_blade_dir(ref_points, obs_points, blade_dir_hint)
        print("    2D 날방향 기반 초기 정렬 완료")
    else:
        init_transform = initial_alignment_pca(ref_points, obs_points)
        print("    PCA 초기 정렬 완료")

    return run_icp_stages(
        ref_pcd,
        obs_pcd,
        init_transform,
        max_correspondence_dist,
        num_iterations,
        anchor_center=anchor_center,
        blade_dir_hint=blade_dir_hint,
    )


def estimate_pose_foundation(
    ref_model_path: str,
    args: argparse.Namespace,
    session: CaptureSession,
    observation: ObservationData,
) -> Optional[Tuple[np.ndarray, float, float]]:
    """FoundationPose 추정. 실패하면 None을 반환한다."""
    try:
        from FoundationPose import FoundationPose  # type: ignore
    except ImportError:
        print("  [WARNING] FoundationPose 미설치.")
        print("            설치: https://github.com/NVlabs/FoundationPose")
        return None

    estimator = FoundationPose(mesh_file=ref_model_path, debug=0)
    transforms = []
    scores = []

    for camera_id in session.camera_ids:
        frame = session.frames[camera_id]
        calibration = session.calibrations[camera_id]
        if frame.bgr is None:
            continue

        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
        depth_m = None
        if frame.depth is not None:
            depth_m = frame.depth.astype(np.float32) * calibration.depth_scale_m_per_unit

        if observation.masks and observation.masks.get(camera_id) is not None:
            object_mask = observation.masks[camera_id].astype(np.uint8) * 255
        else:
            object_mask = np.ones(rgb.shape[:2], dtype=np.uint8) * 255

        pose_cam = estimator.register(
            K=calibration.K.astype(np.float64),
            rgb=rgb,
            depth=depth_m,
            ob_mask=object_mask,
        )
        if pose_cam is None:
            print(f"    cam{camera_id}: FoundationPose 검출 실패")
            continue

        pose_cam0 = calibration.T_cam0_cam @ pose_cam
        transforms.append(pose_cam0)
        scores.append(float(object_mask.sum()))
        print(f"    cam{camera_id}: FoundationPose 성공")

    if not transforms:
        print("    모든 카메라 FoundationPose 실패")
        return None

    weights = np.array(scores, dtype=np.float64)
    weights /= weights.sum()

    rotation_avg = sum(weight * transform[:3, :3] for weight, transform in zip(weights, transforms))
    U, _, Vt = np.linalg.svd(rotation_avg)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = U @ Vt
    translation = sum(weight * transform[:3, 3] for weight, transform in zip(weights, transforms))

    final_transform = np.eye(4, dtype=np.float64)
    final_transform[:3, :3] = rotation
    final_transform[:3, 3] = translation

    fitness = float(len(transforms) / len(session.camera_ids))
    rmse = 0.003
    return final_transform, fitness, rmse


def R_to_euler(rotation: np.ndarray) -> np.ndarray:
    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(rotation[2, 1], rotation[2, 2])
        y = np.arctan2(-rotation[2, 0], sy)
        z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        y = np.arctan2(-rotation[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def R_to_quat(rotation: np.ndarray) -> np.ndarray:
    trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rotation[2, 1] - rotation[1, 2]) * s
        y = (rotation[0, 2] - rotation[2, 0]) * s
        z = (rotation[1, 0] - rotation[0, 1]) * s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s

    quaternion = np.array([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0 else -quaternion


def visualize_result(
    obs_points: np.ndarray,
    obs_colors: Optional[np.ndarray],
    ref_points_transformed: np.ndarray,
    pose: dict,
    out_path: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 6))
    obs_mm = obs_points * 1000.0
    ref_mm = ref_points_transformed * 1000.0
    rotation = pose["R"]
    center_mm = pose["position_mm"]

    views = [(25, -60, "Perspective"), (90, -90, "Top (XZ)"), (0, -90, "Front (XY)")]
    for index, (elev, azim, subtitle) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.set_title(subtitle, fontsize=10)

        obs_sel = np.random.default_rng(42).choice(len(obs_mm), min(len(obs_mm), 5000), replace=False)
        if obs_colors is not None:
            ax.scatter(obs_mm[obs_sel, 0], obs_mm[obs_sel, 1], obs_mm[obs_sel, 2], c=np.clip(obs_colors[obs_sel], 0, 1), s=0.5, alpha=0.4)
        else:
            ax.scatter(obs_mm[obs_sel, 0], obs_mm[obs_sel, 1], obs_mm[obs_sel, 2], c="steelblue", s=0.5, alpha=0.4)

        ref_sel = np.random.default_rng(7).choice(len(ref_mm), min(len(ref_mm), 3000), replace=False)
        ax.scatter(ref_mm[ref_sel, 0], ref_mm[ref_sel, 1], ref_mm[ref_sel, 2], c="#e74c3c", s=0.5, alpha=0.3)

        axis_length = 30.0
        colors = ["#e74c3c", "#27ae60", "#2980b9"]
        for axis in range(3):
            vec = rotation[:, axis] * axis_length
            ax.quiver(center_mm[0], center_mm[1], center_mm[2], vec[0], vec[1], vec[2], color=colors[axis], linewidth=2.5, arrow_length_ratio=0.12)
            cam_vec = np.eye(3)[:, axis] * 50.0
            ax.quiver(0, 0, 0, cam_vec[0], cam_vec[1], cam_vec[2], color=colors[axis], linewidth=1.5, arrow_length_ratio=0.08, alpha=0.4)

        ax.set_xlabel("X mm", fontsize=7)
        ax.set_ylabel("Y mm", fontsize=7)
        ax.set_zlabel("Z mm", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)

        all_mm = np.vstack([obs_mm, ref_mm])
        middle = all_mm.mean(axis=0)
        radius = max((all_mm.max(axis=0) - all_mm.min(axis=0)).max() / 2 * 1.3, 40)
        ax.set_xlim(middle[0] - radius, middle[0] + radius)
        ax.set_ylim(middle[1] - radius, middle[1] + radius)
        ax.set_zlim(middle[2] - radius, middle[2] + radius)

    euler = pose["euler_xyz_deg"]
    fig.suptitle(title, fontsize=11, y=0.98)
    fig.text(
        0.5,
        0.01,
        f"Position: ({center_mm[0]:.1f}, {center_mm[1]:.1f}, {center_mm[2]:.1f}) mm  |  "
        f"Euler: ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg  |  "
        f"fitness: {pose['icp_fitness']:.4f}, RMSE: {pose['icp_rmse_mm']:.2f}mm",
        ha="center",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="#ecf0f1", ec="#bdc3c7"),
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {out_path}")


def save_projection_overlay(
    obs_points: np.ndarray,
    ref_points: np.ndarray,
    rotation: np.ndarray,
    K: Optional[np.ndarray],
    D: Optional[np.ndarray],
    capture_dir: Optional[str],
    frame_idx: Optional[int],
    pad: Optional[int],
    out_path: str,
) -> None:
    if K is None or D is None or capture_dir is None or frame_idx is None or pad is None:
        return

    rgb_path = Path(capture_dir) / "cam0" / f"rgb_{frame_id(frame_idx, pad)}.jpg"
    bgr = cv2.imread(str(rgb_path))
    if bgr is None:
        return

    height, width = bgr.shape[:2]
    distortion = np.asarray(D, dtype=np.float64)

    def draw_points(points: np.ndarray, color: Tuple[int, int, int], step: int = 8) -> None:
        if len(points) == 0:
            return
        uv, _ = cv2.projectPoints(
            points[::step].astype(np.float64).reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            K,
            distortion,
        )
        uv = uv.reshape(-1, 2)
        valid = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        for u, v in uv[valid]:
            cv2.circle(bgr, (int(u), int(v)), 2, color, -1)

    draw_points(obs_points, (255, 80, 0))
    draw_points(ref_points, (0, 50, 255))

    center = ref_points.mean(axis=0)
    axis_length = 0.030
    for axis, color in enumerate([(0, 0, 200), (0, 200, 0), (200, 0, 0)]):
        end = center + rotation[:, axis] * axis_length
        uv_start, _ = cv2.projectPoints(center.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, distortion)
        uv_end, _ = cv2.projectPoints(end.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, distortion)
        start = tuple(uv_start.reshape(2).astype(int))
        finish = tuple(uv_end.reshape(2).astype(int))
        if 0 <= start[0] < width and 0 <= start[1] < height and 0 <= finish[0] < width and 0 <= finish[1] < height:
            cv2.arrowedLine(bgr, start, finish, color, 2, tipLength=0.2)

    cv2.putText(bgr, "Blue=Observed  Red=Reference(aligned)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(out_path, bgr)
    print(f"  [저장] {out_path}")


def visualize_pose_cam0(pose: dict, out_path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rotation = pose["R"]
    position_mm = np.array(pose["position_mm"], dtype=np.float64)
    euler = pose["euler_xyz_deg"]
    quaternion = pose["quaternion_wxyz"]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    def draw_axes(rotation_matrix: np.ndarray, origin: np.ndarray, length: float, label: str, lw: float = 2.0, alpha: float = 1.0) -> None:
        colors = ["#e74c3c", "#27ae60", "#2980b9"]
        for axis, axis_name in enumerate(["X", "Y", "Z"]):
            vec = rotation_matrix[:, axis] * length
            ax.quiver(origin[0], origin[1], origin[2], vec[0], vec[1], vec[2], color=colors[axis], linewidth=lw, arrow_length_ratio=0.12, alpha=alpha)
            tip = origin + vec * 1.14
            ax.text(tip[0], tip[1], tip[2], axis_name, fontsize=7, color=colors[axis], fontweight="bold", alpha=alpha)
        if label:
            ax.text(origin[0], origin[1], origin[2] - length * 0.4, label, fontsize=9, fontweight="bold", ha="center")

    draw_axes(np.eye(3), np.zeros(3), 70.0, "cam0 (ref)", lw=3.0)
    draw_axes(rotation, position_mm, 55.0, "Object", lw=3.5)
    ax.plot3D([0, position_mm[0]], [0, position_mm[1]], [0, position_mm[2]], ":", color="#c0392b", lw=1.0, alpha=0.35)
    ax.text(position_mm[0], position_mm[1] - 45, position_mm[2] + 52, f"({position_mm[0]:.1f}, {position_mm[1]:.1f}, {position_mm[2]:.1f}) mm", fontsize=8, color="#2c3e50", ha="center")
    ax.text(position_mm[0], position_mm[1] - 45, position_mm[2] + 35, f"euler ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg", fontsize=8, color="#2c3e50", ha="center")

    points = np.array([[0, 0, 0], position_mm.tolist()], dtype=np.float64)
    center = points.mean(axis=0)
    radius = max((points.max(axis=0) - points.min(axis=0)).max() / 2 * 1.4, 120.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=24, azim=-55)

    fig.text(
        0.5,
        0.01,
        "Axis color: X=Red  Y=Green  Z=Blue\n"
        f"Quat wxyz: ({quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f})",
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ecf0f1", ec="#bdc3c7"),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {out_path}")


def save_results(
    pose: dict,
    observation: ObservationData,
    ref_points_transformed: np.ndarray,
    out_dir: str,
    tag: str,
    elapsed_sec: float,
    pose_mode: str,
) -> dict:
    import open3d as o3d

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rotation = np.asarray(pose["R"], dtype=np.float64)
    position_mm = np.asarray(pose["position_mm"], dtype=np.float64)
    euler_xyz_deg = np.asarray(pose["euler_xyz_deg"], dtype=np.float64).tolist()
    quaternion = np.asarray(pose["quaternion_wxyz"], dtype=np.float64).tolist()

    print("\n" + "=" * 60)
    print("  RESULT - cam0 (X-right, Y-down, Z-forward)")
    print(f"  method: GLB reference + {pose_mode.upper()}")
    print("=" * 60)
    print(f"  Position   (mm): ({position_mm[0]:+.1f}, {position_mm[1]:+.1f}, {position_mm[2]:+.1f})")
    print(f"  Euler XYZ (deg): ({euler_xyz_deg[0]:+.1f}, {euler_xyz_deg[1]:+.1f}, {euler_xyz_deg[2]:+.1f})")
    print(f"  Quat wxyz      : ({quaternion[0]:.5f}, {quaternion[1]:.5f}, {quaternion[2]:.5f}, {quaternion[3]:.5f})")
    print(f"  fitness        : {pose['icp_fitness']:.4f}")
    print(f"  RMSE      (mm) : {pose['icp_rmse_mm']:.3f}")
    print(f"  Scale factor   : {pose['scale_factor']:.6f}")
    print(f"  소요시간       : {elapsed_sec:.1f}s")
    print("=" * 60)

    result = {
        "coordinate_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
        "method": f"GLB_reference_{pose_mode.upper()}",
        "pose_mode": pose_mode,
        "reference_model": pose["ref_model_path"],
        "position_mm": position_mm.tolist(),
        "euler_xyz_deg": euler_xyz_deg,
        "quaternion_wxyz": quaternion,
        "rotation_matrix": rotation.tolist(),
        "icp_fitness": float(pose["icp_fitness"]),
        "icp_rmse_mm": float(pose["icp_rmse_mm"]),
        "scale_factor": float(pose["scale_factor"]),
        "elapsed_sec": round(float(elapsed_sec), 2),
    }

    json_path = out_path / f"pose_{tag}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(f"  [저장] {json_path}")

    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(np.asarray(observation.points))
    if observation.colors is not None:
        obs_pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(observation.colors), 0, 1))
    obs_ply = out_path / f"observed_pointcloud_{tag}.ply"
    o3d.io.write_point_cloud(str(obs_ply), obs_pcd)
    print(f"  [저장] {obs_ply}")

    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(np.asarray(ref_points_transformed))
    ref_ply = out_path / f"aligned_reference_{tag}.ply"
    o3d.io.write_point_cloud(str(ref_ply), ref_pcd)
    print(f"  [저장] {ref_ply}")

    try:
        visualize_result(
            observation.points,
            observation.colors,
            ref_points_transformed,
            pose,
            str(out_path / f"alignment_{tag}.png"),
            f"{pose_mode.upper()} Alignment - {tag}",
        )
    except Exception as exc:
        print(f"  [WARN] 정합 시각화 실패: {exc}")

    try:
        visualize_pose_cam0(
            pose,
            str(out_path / f"pose_cam0_{tag}.png"),
            f"GLB Reference + {pose_mode.upper()} Pose (cam0, mm)",
        )
    except Exception as exc:
        print(f"  [WARN] 포즈 시각화 실패: {exc}")

    try:
        save_projection_overlay(
            observation.points,
            ref_points_transformed,
            pose["R"],
            pose.get("_K_cam0"),
            pose.get("_D_cam0"),
            pose.get("_cap_dir"),
            pose.get("_frame_idx"),
            pose.get("_pad"),
            str(out_path / f"overlay_cam0_{tag}.png"),
        )
    except Exception as exc:
        print(f"  [WARN] 오버레이 시각화 실패: {exc}")

    return result


def build_point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> "open3d.geometry.PointCloud":
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    return point_cloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLB 참조 모델 기반 물체 6-DOF 포즈 추정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
포즈 추정 모드:
  icp        : depth 점군 -> PCA/방향 힌트 초기 정렬 -> Point-to-Plane ICP
  foundation : FoundationPose 딥러닝 (실패 시 ICP fallback)

세그멘테이션 모드:
  hsv        : 노란 손잡이 앵커 + depth 연결 object mask
  sam2       : GroundingDINO + SAM2
  depth_roi  : 깊이 기반 테이블 제거 + 최대 클러스터

예시:
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_knife.glb
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_knife.glb --seg_mode depth_roi
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_knife.glb --pose_mode foundation
""",
    )

    parser.add_argument("--ref_model", required=True, help="참조 3D 모델 경로 (GLB/PLY/OBJ)")
    parser.add_argument("--pose_mode", choices=["icp", "foundation"], default="icp", help="포즈 추정 방식")
    parser.add_argument("--capture_dir", default=str(_DEFAULT_CAP_DIR))
    parser.add_argument("--calib_dir", default=str(_DEFAULT_CAL_DIR))
    parser.add_argument("--intrinsics_dir", default=str(_DEFAULT_INT_DIR))
    parser.add_argument("--frame", type=int, default=3)
    parser.add_argument("--z_min", type=float, default=0.1)
    parser.add_argument("--z_max", type=float, default=1.5)

    parser.add_argument("--seg_mode", choices=["hsv", "sam2", "depth_roi"], default="hsv", help="세그멘테이션 방식")
    parser.add_argument("--hsv_h_range", type=int, nargs=2, default=[15, 35])
    parser.add_argument("--hsv_s_min", type=int, default=80)
    parser.add_argument("--hsv_v_min", type=int, default=80)
    parser.add_argument("--min_component_area", type=int, default=500)

    parser.add_argument("--text_prompt", default="utility knife.")
    parser.add_argument("--gdino_model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2_checkpoint", default=str(_THIS_DIR / "checkpoints/sam2.1_hiera_large.pt"))
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--box_threshold", type=float, default=0.15)
    parser.add_argument("--text_threshold", type=float, default=0.15)

    parser.add_argument("--ref_length_mm", type=float, default=None, help="물체 실제 길이 mm")
    parser.add_argument("--scale", type=float, default=None, help="참조 모델 스케일 직접 지정")
    parser.add_argument("--icp_dist", type=float, default=10.0, help="ICP max correspondence distance (mm)")
    parser.add_argument("--icp_iters", type=int, default=100, help="ICP fine 단계 최대 반복")
    parser.add_argument("--out_dir", default=str(_DEFAULT_OUT_DIR))
    return parser.parse_args()


def select_observation(args: argparse.Namespace, session: CaptureSession) -> ObservationData:
    segmenters = {
        "hsv": run_hsv_segmentation,
        "sam2": run_sam2_segmentation,
        "depth_roi": run_depth_roi_segmentation,
    }
    observation = segmenters[args.seg_mode](args, session)
    if len(observation.points) == 0:
        raise RuntimeError("세그멘테이션 결과가 비어 있음")
    return observation


def estimate_pose(
    args: argparse.Namespace,
    ref_scaled: "open3d.geometry.PointCloud",
    obs_pcd: "open3d.geometry.PointCloud",
    observation: ObservationData,
    session: CaptureSession,
) -> Tuple[np.ndarray, float, float]:
    icp_dist_m = args.icp_dist / 1000.0

    if args.pose_mode == "foundation":
        foundation_result = estimate_pose_foundation(args.ref_model, args, session, observation)
        if foundation_result is not None:
            return foundation_result
        print("  FoundationPose 실패 -> ICP fallback")

    return estimate_pose_icp(
        ref_scaled,
        obs_pcd,
        max_correspondence_dist=icp_dist_m,
        num_iterations=args.icp_iters,
        anchor_center=observation.anchor_center,
        blade_dir_hint=observation.blade_dir_hint,
    )


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 60)
    print("  Obj_Step2_pose_estimate.py")
    print(f"  pose_mode : {args.pose_mode.upper()}")
    print(f"  seg_mode  : {args.seg_mode}")
    print(f"  ref_model : {args.ref_model}")
    print(f"  frame     : {args.frame}")
    print("=" * 60)

    print(f"\n[Step 1] 참조 모델 로드: {args.ref_model}")
    ref_pcd = load_reference_model(args.ref_model)

    print("\n[Step 2] 캘리브레이션 + 프레임 로드")
    session = load_capture_session(args)
    print(f"  카메라: {list(session.camera_ids)}")

    print(f"\n[Step 3] 세그멘테이션 (seg={args.seg_mode}, pose={args.pose_mode})")
    observation = select_observation(args, session)
    if observation.anchor_center is not None:
        anchor = observation.anchor_center
        print(f"  핸들 앵커 힌트: ({anchor[0] * 1000:.1f}, {anchor[1] * 1000:.1f}, {anchor[2] * 1000:.1f}) mm")

    obs_bbox = (observation.points.max(axis=0) - observation.points.min(axis=0)) * 1000.0
    print(f"  관측 점군: {len(observation.points):,} pts")
    print(f"  bbox: {obs_bbox[0]:.1f} x {obs_bbox[1]:.1f} x {obs_bbox[2]:.1f} mm")

    obs_pcd = build_point_cloud(observation.points, observation.colors)

    print("\n[Step 4] 스케일 조정")
    ref_scaled, scale = scale_reference_to_observation(
        ref_pcd,
        obs_pcd,
        ref_length_mm=args.ref_length_mm,
        manual_scale=args.scale,
    )

    print(f"\n[Step 5] 포즈 추정 - {args.pose_mode.upper()}")
    transform, fitness, rmse = estimate_pose(args, ref_scaled, obs_pcd, observation, session)

    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    euler = R_to_euler(rotation)
    quaternion = R_to_quat(rotation)

    pose = {
        "R": rotation,
        "position_mm": translation * 1000.0,
        "euler_xyz_deg": np.asarray(euler, dtype=float).tolist(),
        "quaternion_wxyz": np.asarray(quaternion, dtype=float).tolist(),
        "icp_fitness": float(fitness),
        "icp_rmse_mm": float(rmse * 1000.0),
        "scale_factor": float(scale),
        "ref_model_path": os.path.abspath(args.ref_model),
        "_K_cam0": session.calibrations[session.camera_ids[0]].K,
        "_D_cam0": session.calibrations[session.camera_ids[0]].D,
        "_cap_dir": args.capture_dir,
        "_frame_idx": args.frame,
        "_pad": session.pad,
    }

    ref_points_transformed = np.asarray(ref_scaled.points) @ rotation.T + translation

    print("\n[Step 6] 결과 저장")
    tag = f"frame{args.frame:06d}_{args.pose_mode}"
    out_dir = Path(args.out_dir) / f"output_{tag}"
    save_results(
        pose,
        observation,
        ref_points_transformed,
        str(out_dir),
        tag,
        time.time() - start_time,
        args.pose_mode,
    )

    print("\n완료!")


if __name__ == "__main__":
    main()
