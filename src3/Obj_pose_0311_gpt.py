#!/usr/bin/env python3
"""
멀티뷰 RGB-D 기반 물체 포즈 추정.

파이프라인:
1. cam0/cam1/cam2 RGB-D 로드
2. depth 역투영으로 각 카메라 점군 생성
3. T_C0_Ci 로 cam0 좌표계에 통합
4. 평면 제거 + 클러스터링으로 물체 점군 추출
5. GLB 메쉬를 점군으로 샘플링
6. 스케일 정합
7. FPFH + RANSAC 전역 정합
8. ICP 정밀 정합
9. cam0 기준 pose 저장

예시:
  python src3/Obj_pose_0311_gpt.py --frame 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTURE_DIR = THIS_DIR / "data/object_capture"
DEFAULT_INTRINSICS_DIR = THIS_DIR / "data/_intrinsics"
DEFAULT_CALIB_DIR = THIS_DIR / "data/cube_session_01/calib_out_cube"
DEFAULT_REF_MODEL = THIS_DIR / "data/reference_knife.glb"
DEFAULT_OUTPUT_DIR = THIS_DIR / "Obj_pose_0311_gpt_output"


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: int
    K: np.ndarray
    D: np.ndarray
    depth_scale_m_per_unit: float
    T_cam0_cam: np.ndarray


@dataclass(frozen=True)
class FramePaths:
    rgb_path: Path
    depth_path: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def frame_token(frame_idx: int, pad: int) -> str:
    return f"{frame_idx:0{pad}d}"


def discover_camera_ids(capture_dir: Path) -> Tuple[int, ...]:
    camera_ids: List[int] = []
    for folder in sorted(capture_dir.glob("cam*")):
        try:
            camera_id = int(folder.name.replace("cam", ""))
        except ValueError:
            continue
        if list(folder.glob("rgb_*.jpg")):
            camera_ids.append(camera_id)
    if not camera_ids:
        raise RuntimeError(f"cam* 폴더를 찾지 못했습니다: {capture_dir}")
    return tuple(camera_ids)


def discover_frame_pad(capture_dir: Path, camera_id: int) -> int:
    rgb_files = sorted((capture_dir / f"cam{camera_id}").glob("rgb_*.jpg"))
    if not rgb_files:
        return 6
    return len(rgb_files[0].stem.replace("rgb_", ""))


def load_intrinsics(intrinsics_dir: Path, camera_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
    path = intrinsics_dir / f"cam{camera_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"intrinsics 파일이 없습니다: {path}")

    data = np.load(path, allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64).reshape(-1)
    depth_scale = float(data["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in data else 0.001
    return K, D, depth_scale


def load_extrinsics(calib_dir: Path, camera_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    transforms: Dict[int, np.ndarray] = {}
    for camera_id in camera_ids:
        if camera_id == 0:
            transforms[camera_id] = np.eye(4, dtype=np.float64)
            continue

        path = calib_dir / f"T_C0_C{camera_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"extrinsics 파일이 없습니다: {path}")
        transforms[camera_id] = np.load(path).astype(np.float64)
    return transforms


def build_calibration_map(
    intrinsics_dir: Path,
    calib_dir: Path,
    camera_ids: Sequence[int],
) -> Dict[int, CameraCalibration]:
    transforms = load_extrinsics(calib_dir, camera_ids)
    calibrations: Dict[int, CameraCalibration] = {}
    for camera_id in camera_ids:
        K, D, depth_scale = load_intrinsics(intrinsics_dir, camera_id)
        calibrations[camera_id] = CameraCalibration(
            camera_id=camera_id,
            K=K,
            D=D,
            depth_scale_m_per_unit=depth_scale,
            T_cam0_cam=transforms[camera_id],
        )
    return calibrations


def build_frame_paths(capture_dir: Path, camera_ids: Sequence[int], fid: str) -> Dict[int, FramePaths]:
    frames: Dict[int, FramePaths] = {}
    for camera_id in camera_ids:
        frames[camera_id] = FramePaths(
            rgb_path=capture_dir / f"cam{camera_id}" / f"rgb_{fid}.jpg",
            depth_path=capture_dir / f"cam{camera_id}" / f"depth_{fid}.png",
        )
    return frames


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    return points @ transform[:3, :3].T + transform[:3, 3]


def backproject_depth(
    depth_u16: np.ndarray,
    rgb_bgr: np.ndarray,
    calibration: CameraCalibration,
    stride: int,
    z_min: float,
    z_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = depth_u16.shape[:2]
    v_grid, u_grid = np.mgrid[0:height:stride, 0:width:stride]
    depth_sample = depth_u16[0:height:stride, 0:width:stride].astype(np.float64)
    z = depth_sample * calibration.depth_scale_m_per_unit

    valid = (z > z_min) & (z < z_max)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    z = z[valid]

    undistorted = cv2.undistortPoints(
        np.column_stack([u, v]).reshape(-1, 1, 2),
        calibration.K,
        calibration.D,
    ).reshape(-1, 2)

    points_cam = np.column_stack(
        [
            undistorted[:, 0] * z,
            undistorted[:, 1] * z,
            z,
        ]
    )

    ui = np.clip(np.round(u).astype(int), 0, rgb_bgr.shape[1] - 1)
    vi = np.clip(np.round(v).astype(int), 0, rgb_bgr.shape[0] - 1)
    colors_rgb = rgb_bgr[vi, ui][:, ::-1].astype(np.float64) / 255.0
    return points_cam, colors_rgb


def build_scene_cloud(
    frames: Dict[int, FramePaths],
    calibrations: Dict[int, CameraCalibration],
    stride: int,
    z_min: float,
    z_max: float,
) -> o3d.geometry.PointCloud:
    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []

    print("[1/6] RGB-D 역투영 및 cam0 통합")
    for camera_id, frame_paths in frames.items():
        if not frame_paths.rgb_path.exists():
            raise FileNotFoundError(f"RGB 파일이 없습니다: {frame_paths.rgb_path}")
        if not frame_paths.depth_path.exists():
            raise FileNotFoundError(f"Depth 파일이 없습니다: {frame_paths.depth_path}")

        rgb_bgr = cv2.imread(str(frame_paths.rgb_path), cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(str(frame_paths.depth_path), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None:
            raise RuntimeError(f"RGB 로드 실패: {frame_paths.rgb_path}")
        if depth_u16 is None:
            raise RuntimeError(f"Depth 로드 실패: {frame_paths.depth_path}")

        points_cam, colors = backproject_depth(
            depth_u16=depth_u16,
            rgb_bgr=rgb_bgr,
            calibration=calibrations[camera_id],
            stride=stride,
            z_min=z_min,
            z_max=z_max,
        )
        points_cam0 = transform_points(points_cam, calibrations[camera_id].T_cam0_cam)
        print(f"  cam{camera_id}: {len(points_cam0):,} pts")

        all_points.append(points_cam0)
        all_colors.append(colors)

    if not all_points:
        raise RuntimeError("유효한 depth 점군을 만들지 못했습니다.")

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    return pcd


def keep_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd

    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        print("  DBSCAN 유효 클러스터 없음 -> 전체 점군 유지")
        return pcd

    valid_labels = labels[labels >= 0]
    unique_labels, counts = np.unique(valid_labels, return_counts=True)
    best_label = int(unique_labels[np.argmax(counts)])
    keep_indices = np.where(labels == best_label)[0].tolist()
    print(f"  DBSCAN largest cluster: label={best_label}, points={len(keep_indices):,}")
    return pcd.select_by_index(keep_indices)


def extract_object_cloud(
    scene_pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    sor_nb_neighbors: int,
    sor_std_ratio: float,
    plane_distance_threshold: float,
    plane_min_inlier_ratio: float,
    cluster_eps: float,
    cluster_min_points: int,
    skip_plane_removal: bool,
) -> o3d.geometry.PointCloud:
    print("[2/6] 물체 점군 추출")
    pcd = scene_pcd.voxel_down_sample(voxel_size)
    print(f"  voxel downsample: {len(scene_pcd.points):,} -> {len(pcd.points):,}")
    if len(pcd.points) == 0:
        raise RuntimeError("downsample 결과 점군이 비었습니다.")

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=sor_nb_neighbors,
        std_ratio=sor_std_ratio,
    )
    print(f"  statistical outlier removal -> {len(pcd.points):,}")

    if not skip_plane_removal and len(pcd.points) >= 100:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=plane_distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        inlier_ratio = len(inliers) / max(len(pcd.points), 1)
        print(
            "  plane fit: "
            f"inliers={len(inliers):,} ({inlier_ratio * 100:.1f}%), "
            f"model=[{plane_model[0]:.3f}, {plane_model[1]:.3f}, {plane_model[2]:.3f}, {plane_model[3]:.3f}]"
        )
        if inlier_ratio >= plane_min_inlier_ratio:
            pcd = pcd.select_by_index(inliers, invert=True)
            print(f"  plane removed -> {len(pcd.points):,}")
        else:
            print("  plane 비율이 낮아 제거하지 않음")

    if len(pcd.points) == 0:
        raise RuntimeError("평면 제거 후 점군이 비었습니다.")

    pcd = keep_largest_cluster(
        pcd=pcd,
        eps=cluster_eps,
        min_points=cluster_min_points,
    )
    if len(pcd.points) == 0:
        raise RuntimeError("클러스터링 후 점군이 비었습니다.")
    return pcd


def load_reference_cloud(
    ref_model_path: Path,
    sample_count: int,
    flip_glb_yz: bool,
) -> o3d.geometry.PointCloud:
    print("[3/6] 참조 GLB 로드")
    if not ref_model_path.exists():
        raise FileNotFoundError(f"참조 모델이 없습니다: {ref_model_path}")

    loaded = trimesh.load(str(ref_model_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise RuntimeError(f"GLB 안에 triangle mesh가 없습니다: {ref_model_path}")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"지원하지 않는 참조 형식입니다: {type(loaded)}")

    sampled_points = mesh.sample(sample_count)
    if flip_glb_yz and ref_model_path.suffix.lower() in {".glb", ".gltf"}:
        sampled_points[:, 1] *= -1.0
        sampled_points[:, 2] *= -1.0
        print("  GLB 좌표계 보정: Y/Z 축 반전(OpenGL -> OpenCV)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points.astype(np.float64))
    print(f"  model sampled points: {len(pcd.points):,}")
    return pcd


def compute_scale_factor(
    model_pcd: o3d.geometry.PointCloud,
    scene_object_pcd: o3d.geometry.PointCloud,
    manual_scale: float | None,
    ref_length_mm: float | None,
) -> float:
    if manual_scale is not None:
        print(f"  manual scale 사용: {manual_scale:.6f}")
        return float(manual_scale)

    if ref_length_mm is not None:
        model_extent = np.sort(model_pcd.get_oriented_bounding_box().extent)
        model_max_extent = float(model_extent[-1])
        if model_max_extent <= 1e-9:
            raise RuntimeError("참조 모델 길이가 0에 가깝습니다.")
        scale = (ref_length_mm / 1000.0) / model_max_extent
        print(f"  실제 길이 기준 scale 사용: {scale:.6f}")
        return scale

    scene_extent = np.sort(scene_object_pcd.get_oriented_bounding_box().extent)
    model_extent = np.sort(model_pcd.get_oriented_bounding_box().extent)
    if np.any(model_extent <= 1e-9):
        raise RuntimeError("참조 모델 extent가 0에 가깝습니다.")

    scale_candidates = scene_extent / model_extent
    scale = float(np.median(scale_candidates))
    print(
        "  auto scale: "
        f"scene_extent={scene_extent.round(5)}, "
        f"model_extent={model_extent.round(5)}, "
        f"scale={scale:.6f}"
    )
    return scale


def scale_reference_cloud(
    model_pcd: o3d.geometry.PointCloud,
    scene_object_pcd: o3d.geometry.PointCloud,
    manual_scale: float | None,
    ref_length_mm: float | None,
) -> Tuple[o3d.geometry.PointCloud, float]:
    print("[4/6] 참조 모델 스케일 정합")
    scale = compute_scale_factor(
        model_pcd=model_pcd,
        scene_object_pcd=scene_object_pcd,
        manual_scale=manual_scale,
        ref_length_mm=ref_length_mm,
    )

    model_scaled = o3d.geometry.PointCloud(model_pcd)
    model_scaled.scale(scale, center=model_scaled.get_center())
    return model_scaled, scale


def prepare_fpfh(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) < 30:
        raise RuntimeError("FPFH 계산을 위한 점 개수가 부족합니다.")

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd_down, fpfh


def global_registration(
    model_pcd: o3d.geometry.PointCloud,
    scene_pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    ransac_iterations: int,
    ransac_confidence: float,
) -> o3d.pipelines.registration.RegistrationResult:
    print("[5/6] 전역 정합 (FPFH + RANSAC)")
    model_down, model_fpfh = prepare_fpfh(model_pcd, voxel_size)
    scene_down, scene_fpfh = prepare_fpfh(scene_pcd, voxel_size)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down,
        scene_down,
        model_fpfh,
        scene_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 3.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 3.0),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            ransac_iterations,
            ransac_confidence,
        ),
    )
    print(f"  RANSAC fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}")
    return result


def coarse_refine_alignment(
    model_pcd: o3d.geometry.PointCloud,
    scene_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    distance_threshold: float,
    max_iterations: int,
) -> o3d.pipelines.registration.RegistrationResult:
    model_down = model_pcd.voxel_down_sample(max(distance_threshold * 0.5, 0.002))
    scene_down = scene_pcd.voxel_down_sample(max(distance_threshold * 0.5, 0.002))
    return o3d.pipelines.registration.registration_icp(
        model_down,
        scene_down,
        max_correspondence_distance=distance_threshold,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )


def sorted_obb_axes(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obb = pcd.get_oriented_bounding_box()
    order = np.argsort(np.asarray(obb.extent))[::-1]
    axes = obb.R[:, order]
    extents = np.asarray(obb.extent)[order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return np.asarray(obb.center), axes, extents


def obb_axis_alignment(
    model_pcd: o3d.geometry.PointCloud,
    scene_pcd: o3d.geometry.PointCloud,
    distance_threshold: float,
    max_iterations: int,
) -> Tuple[str, o3d.pipelines.registration.RegistrationResult]:
    model_center, model_axes, _ = sorted_obb_axes(model_pcd)
    scene_center, scene_axes, _ = sorted_obb_axes(scene_pcd)

    best_name = ""
    best_result = None
    best_score = None

    for signs in itertools.product([-1.0, 1.0], repeat=3):
        sign_matrix = np.diag(signs)
        rotation = scene_axes @ sign_matrix @ model_axes.T
        if np.linalg.det(rotation) < 0:
            continue

        init_transform = np.eye(4, dtype=np.float64)
        init_transform[:3, :3] = rotation
        init_transform[:3, 3] = scene_center - rotation @ model_center

        coarse_result = coarse_refine_alignment(
            model_pcd=model_pcd,
            scene_pcd=scene_pcd,
            init_transform=init_transform,
            distance_threshold=distance_threshold,
            max_iterations=max_iterations,
        )
        score = (float(coarse_result.fitness), -float(coarse_result.inlier_rmse))
        if best_score is None or score > best_score:
            best_name = f"obb_axes_signs={tuple(int(s) for s in signs)}"
            best_result = coarse_result
            best_score = score

    if best_result is None:
        raise RuntimeError("OBB 축 정렬 초기화를 만들지 못했습니다.")
    return best_name, best_result


def select_initial_alignment(
    model_pcd: o3d.geometry.PointCloud,
    scene_pcd: o3d.geometry.PointCloud,
    ransac_result: o3d.pipelines.registration.RegistrationResult,
    coarse_distance_threshold: float,
    coarse_max_iterations: int,
) -> Tuple[str, o3d.pipelines.registration.RegistrationResult]:
    print("  초기값 후보 평가")

    candidates: List[Tuple[str, o3d.pipelines.registration.RegistrationResult]] = []
    if np.isfinite(ransac_result.fitness) and np.all(np.isfinite(ransac_result.transformation)):
        ransac_coarse = coarse_refine_alignment(
            model_pcd=model_pcd,
            scene_pcd=scene_pcd,
            init_transform=np.asarray(ransac_result.transformation),
            distance_threshold=coarse_distance_threshold,
            max_iterations=coarse_max_iterations,
        )
        print(
            f"    ransac_seed -> fitness={ransac_coarse.fitness:.4f}, "
            f"rmse={ransac_coarse.inlier_rmse:.6f}"
        )
        candidates.append(("ransac_seed", ransac_coarse))

    obb_name, obb_result = obb_axis_alignment(
        model_pcd=model_pcd,
        scene_pcd=scene_pcd,
        distance_threshold=coarse_distance_threshold,
        max_iterations=coarse_max_iterations,
    )
    print(f"    {obb_name} -> fitness={obb_result.fitness:.4f}, rmse={obb_result.inlier_rmse:.6f}")
    candidates.append((obb_name, obb_result))

    def candidate_score(item: Tuple[str, o3d.pipelines.registration.RegistrationResult]) -> Tuple[float, float]:
        result = item[1]
        return float(result.fitness), -float(result.inlier_rmse)

    best_name, best_result = max(candidates, key=candidate_score)
    print(f"  선택된 초기값: {best_name}")
    return best_name, best_result


def refine_icp(
    model_pcd: o3d.geometry.PointCloud,
    scene_pcd: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    icp_distance_threshold: float,
    icp_max_iterations: int,
) -> o3d.pipelines.registration.RegistrationResult:
    print("[6/6] 정밀 정합 (ICP)")
    model = o3d.geometry.PointCloud(model_pcd)
    scene = o3d.geometry.PointCloud(scene_pcd)
    model.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=icp_distance_threshold * 2.0, max_nn=30)
    )
    scene.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=icp_distance_threshold * 2.0, max_nn=30)
    )

    result = o3d.pipelines.registration.registration_icp(
        model,
        scene,
        max_correspondence_distance=icp_distance_threshold,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=icp_max_iterations),
    )
    print(f"  ICP fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}")
    return result


def save_pose_json(
    out_path: Path,
    transform: np.ndarray,
    scale: float,
    frame_idx: int,
    camera_ids: Sequence[int],
    initial_alignment_method: str,
    initial_alignment_result: o3d.pipelines.registration.RegistrationResult,
    ransac_result: o3d.pipelines.registration.RegistrationResult,
    icp_result: o3d.pipelines.registration.RegistrationResult,
) -> None:
    rotation_matrix = transform[:3, :3]
    translation_m = transform[:3, 3]
    rotation = Rotation.from_matrix(rotation_matrix)

    payload = {
        "frame": int(frame_idx),
        "coordinate_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
        "camera_ids": list(camera_ids),
        "T_model_to_cam0": transform.tolist(),
        "rotation_matrix": rotation_matrix.tolist(),
        "translation_m": translation_m.tolist(),
        "translation_mm": (translation_m * 1000.0).tolist(),
        "euler_xyz_deg": rotation.as_euler("xyz", degrees=True).tolist(),
        "quaternion_xyzw": rotation.as_quat().tolist(),
        "scale_applied_to_reference": float(scale),
        "initial_alignment": {
            "method": initial_alignment_method,
            "fitness": float(initial_alignment_result.fitness),
            "inlier_rmse": float(initial_alignment_result.inlier_rmse),
            "transformation": np.asarray(initial_alignment_result.transformation).tolist(),
        },
        "ransac": {
            "fitness": float(ransac_result.fitness),
            "inlier_rmse": float(ransac_result.inlier_rmse),
            "transformation": np.asarray(ransac_result.transformation).tolist(),
        },
        "icp": {
            "fitness": float(icp_result.fitness),
            "inlier_rmse": float(icp_result.inlier_rmse),
            "transformation": np.asarray(icp_result.transformation).tolist(),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_point_cloud(path: Path, pcd: o3d.geometry.PointCloud) -> None:
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def set_axes_equal(ax: "Axes3D", points: np.ndarray) -> None:
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float((maxs - mins).max()) * 0.6, 0.05)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_coordinate_axes(ax: "Axes3D", transform: np.ndarray, axis_length: float, label: str) -> None:
    origin = transform[:3, 3]
    rotation = transform[:3, :3]
    colors = ["r", "g", "b"]
    names = ["X", "Y", "Z"]
    for axis_idx in range(3):
        direction = rotation[:, axis_idx] * axis_length
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=colors[axis_idx],
            linewidth=2.0,
            arrow_length_ratio=0.12,
        )
        tip = origin + direction
        ax.text(tip[0], tip[1], tip[2], f"{label}-{names[axis_idx]}", color=colors[axis_idx], fontsize=9)


def save_visualization_image(
    out_path: Path,
    raw_scene_pcd: o3d.geometry.PointCloud,
    object_pcd: o3d.geometry.PointCloud,
    aligned_model_pcd: o3d.geometry.PointCloud,
    pose_transform: np.ndarray,
    frame_idx: int,
    max_points_scene: int = 20000,
    max_points_object: int = 10000,
    max_points_model: int = 10000,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_points = subsample_points(np.asarray(raw_scene_pcd.points), max_points_scene)
    object_points = subsample_points(np.asarray(object_pcd.points), max_points_object)
    model_points = subsample_points(np.asarray(aligned_model_pcd.points), max_points_model)
    combined_points = np.concatenate([scene_points, object_points, model_points], axis=0)

    fig = plt.figure(figsize=(14, 7), dpi=180)
    views = [
        (18, -62, "Perspective"),
        (90, -90, "Top View"),
    ]
    axis_length = max(float((combined_points.max(axis=0) - combined_points.min(axis=0)).max()) * 0.18, 0.04)

    for subplot_idx, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, subplot_idx, projection="3d")
        ax.scatter(
            scene_points[:, 0],
            scene_points[:, 1],
            scene_points[:, 2],
            s=0.5,
            c="#b7bec7",
            alpha=0.20,
            depthshade=False,
            label="Merged scene",
        )
        ax.scatter(
            object_points[:, 0],
            object_points[:, 1],
            object_points[:, 2],
            s=1.2,
            c="#1f77b4",
            alpha=0.70,
            depthshade=False,
            label="Object cloud",
        )
        ax.scatter(
            model_points[:, 0],
            model_points[:, 1],
            model_points[:, 2],
            s=1.2,
            c="#d62728",
            alpha=0.70,
            depthshade=False,
            label="Aligned reference",
        )
        draw_coordinate_axes(ax, np.eye(4), axis_length, "cam0")
        draw_coordinate_axes(ax, pose_transform, axis_length, "obj")
        set_axes_equal(ax, combined_points)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Frame {frame_idx:06d} - {title}")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Multiview Point Cloud and Estimated Object Pose in cam0", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def project_points_to_image(points_cam: np.ndarray, K: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(points_cam) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=bool)

    valid = points_cam[:, 2] > 1e-6
    points_valid = points_cam[valid]
    if len(points_valid) == 0:
        return np.empty((0, 2), dtype=np.float64), valid

    image_points, _ = cv2.projectPoints(
        points_valid.astype(np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K,
        D,
    )
    return image_points.reshape(-1, 2), valid


def draw_projected_axes(
    image_bgr: np.ndarray,
    transform: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    axis_length: float,
) -> np.ndarray:
    axis_points_obj = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    axis_points_cam = transform_points(axis_points_obj, transform)
    projected, valid = project_points_to_image(axis_points_cam, K, D)
    if len(projected) != 4:
        return image_bgr

    overlay = image_bgr.copy()
    p0 = tuple(np.round(projected[0]).astype(int))
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    labels = ["X", "Y", "Z"]
    for idx in range(3):
        p1 = tuple(np.round(projected[idx + 1]).astype(int))
        cv2.line(overlay, p0, p1, colors[idx], 3, cv2.LINE_AA)
        cv2.circle(overlay, p1, 4, colors[idx], -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"obj-{labels[idx]}",
            (p1[0] + 6, p1[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colors[idx],
            2,
            cv2.LINE_AA,
        )
    cv2.circle(overlay, p0, 5, (255, 255, 255), -1, cv2.LINE_AA)
    return overlay


def save_cam0_overlay_image(
    out_path: Path,
    cam0_rgb_path: Path,
    aligned_model_pcd: o3d.geometry.PointCloud,
    pose_transform: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    max_project_points: int = 15000,
) -> None:
    image_bgr = cv2.imread(str(cam0_rgb_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"cam0 RGB 로드 실패: {cam0_rgb_path}")

    model_points = subsample_points(np.asarray(aligned_model_pcd.points), max_project_points)
    projected, valid = project_points_to_image(model_points, K, D)

    overlay = image_bgr.copy()
    height, width = overlay.shape[:2]
    kept_count = 0
    for uv in projected:
        x = int(round(float(uv[0])))
        y = int(round(float(uv[1])))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1, cv2.LINE_AA)
            kept_count += 1

    bbox = aligned_model_pcd.get_axis_aligned_bounding_box()
    axis_length = max(float(np.max(bbox.get_extent())) * 0.25, 0.03)
    overlay = draw_projected_axes(
        image_bgr=overlay,
        transform=pose_transform,
        K=K,
        D=D,
        axis_length=axis_length,
    )

    blended = cv2.addWeighted(overlay, 0.72, image_bgr, 0.28, 0.0)
    cv2.putText(
        blended,
        f"Projected aligned reference on cam0 ({kept_count} pts)",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), blended)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="멀티뷰 RGB-D + GLB 정합 기반 물체 포즈 추정")
    parser.add_argument("--frame", type=int, default=3, help="처리할 프레임 번호")
    parser.add_argument("--capture_dir", type=Path, default=DEFAULT_CAPTURE_DIR, help="object_capture 폴더")
    parser.add_argument("--intrinsics_dir", type=Path, default=DEFAULT_INTRINSICS_DIR, help="카메라 intrinsics 폴더")
    parser.add_argument("--calib_dir", type=Path, default=DEFAULT_CALIB_DIR, help="cam0 기준 extrinsics 폴더")
    parser.add_argument("--ref_model", type=Path, default=DEFAULT_REF_MODEL, help="참조 GLB/mesh 파일")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="출력 루트 폴더")

    parser.add_argument("--depth_stride", type=int, default=2, help="depth 샘플 stride")
    parser.add_argument("--z_min", type=float, default=0.15, help="depth 최소 거리(m)")
    parser.add_argument("--z_max", type=float, default=1.20, help="depth 최대 거리(m)")
    parser.add_argument("--scene_voxel_size", type=float, default=0.002, help="scene downsample voxel(m)")
    parser.add_argument("--registration_voxel_size", type=float, default=0.005, help="FPFH voxel(m)")
    parser.add_argument("--icp_distance_threshold", type=float, default=0.005, help="ICP correspondence 거리(m)")

    parser.add_argument("--sor_nb_neighbors", type=int, default=30, help="statistical outlier neighbors")
    parser.add_argument("--sor_std_ratio", type=float, default=1.5, help="statistical outlier std ratio")
    parser.add_argument("--plane_distance_threshold", type=float, default=0.004, help="평면 제거 거리 임계값(m)")
    parser.add_argument("--plane_min_inlier_ratio", type=float, default=0.15, help="평면 제거 최소 inlier 비율")
    parser.add_argument("--skip_plane_removal", action="store_true", help="평면 제거 단계 건너뛰기")
    parser.add_argument("--cluster_eps", type=float, default=0.01, help="DBSCAN eps(m)")
    parser.add_argument("--cluster_min_points", type=int, default=80, help="DBSCAN 최소 점 수")

    parser.add_argument("--model_sample_points", type=int, default=50000, help="GLB 샘플링 점 개수")
    parser.add_argument("--manual_scale", type=float, default=None, help="참조 모델 수동 scale")
    parser.add_argument("--ref_length_mm", type=float, default=None, help="실제 물체 최대 길이(mm)")
    parser.add_argument("--skip_glb_axis_flip", action="store_true", help="GLB Y/Z 축 반전 보정 비활성화")

    parser.add_argument("--ransac_iterations", type=int, default=100000, help="RANSAC 최대 반복 수")
    parser.add_argument("--ransac_confidence", type=float, default=0.999, help="RANSAC convergence confidence")
    parser.add_argument("--coarse_init_distance_threshold", type=float, default=0.03, help="초기값 후보 평가 거리(m)")
    parser.add_argument("--coarse_init_iterations", type=int, default=80, help="초기값 후보 평가 ICP 반복 수")
    parser.add_argument("--icp_max_iterations", type=int, default=200, help="ICP 최대 반복 수")
    parser.add_argument("--no_save_visualization", action="store_true", help="3D 시각화 PNG 저장 비활성화")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_ids = discover_camera_ids(args.capture_dir)
    pad = discover_frame_pad(args.capture_dir, camera_ids[0])
    fid = frame_token(args.frame, pad)
    calibrations = build_calibration_map(args.intrinsics_dir, args.calib_dir, camera_ids)
    frames = build_frame_paths(args.capture_dir, camera_ids, fid)

    output_dir = ensure_dir(args.output_dir / f"output_frame{fid}_icp")
    print(f"출력 폴더: {output_dir}")
    print(f"frame={args.frame}, cameras={camera_ids}")

    scene_pcd = build_scene_cloud(
        frames=frames,
        calibrations=calibrations,
        stride=args.depth_stride,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    save_point_cloud(output_dir / f"merged_scene_frame{fid}_raw.ply", scene_pcd)

    object_pcd = extract_object_cloud(
        scene_pcd=scene_pcd,
        voxel_size=args.scene_voxel_size,
        sor_nb_neighbors=args.sor_nb_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        plane_distance_threshold=args.plane_distance_threshold,
        plane_min_inlier_ratio=args.plane_min_inlier_ratio,
        cluster_eps=args.cluster_eps,
        cluster_min_points=args.cluster_min_points,
        skip_plane_removal=args.skip_plane_removal,
    )
    save_point_cloud(output_dir / f"merged_object_frame{fid}.ply", object_pcd)

    model_pcd = load_reference_cloud(
        ref_model_path=args.ref_model,
        sample_count=args.model_sample_points,
        flip_glb_yz=not args.skip_glb_axis_flip,
    )
    model_scaled, scale = scale_reference_cloud(
        model_pcd=model_pcd,
        scene_object_pcd=object_pcd,
        manual_scale=args.manual_scale,
        ref_length_mm=args.ref_length_mm,
    )
    save_point_cloud(output_dir / f"reference_scaled_frame{fid}.ply", model_scaled)

    ransac_result = global_registration(
        model_pcd=model_scaled,
        scene_pcd=object_pcd,
        voxel_size=args.registration_voxel_size,
        ransac_iterations=args.ransac_iterations,
        ransac_confidence=args.ransac_confidence,
    )
    initial_method, initial_result = select_initial_alignment(
        model_pcd=model_scaled,
        scene_pcd=object_pcd,
        ransac_result=ransac_result,
        coarse_distance_threshold=args.coarse_init_distance_threshold,
        coarse_max_iterations=args.coarse_init_iterations,
    )
    icp_result = refine_icp(
        model_pcd=model_scaled,
        scene_pcd=object_pcd,
        init_transform=initial_result.transformation,
        icp_distance_threshold=args.icp_distance_threshold,
        icp_max_iterations=args.icp_max_iterations,
    )

    aligned_model = o3d.geometry.PointCloud(model_scaled)
    aligned_model.transform(icp_result.transformation)
    save_point_cloud(output_dir / f"aligned_reference_frame{fid}_icp.ply", aligned_model)

    visualization_path = output_dir / f"pointcloud_pose_frame{fid}_icp.png"
    overlay_cam0_path = output_dir / f"overlay_cam0_frame{fid}_icp.png"
    if not args.no_save_visualization:
        save_visualization_image(
            out_path=visualization_path,
            raw_scene_pcd=scene_pcd,
            object_pcd=object_pcd,
            aligned_model_pcd=aligned_model,
            pose_transform=np.asarray(icp_result.transformation),
            frame_idx=args.frame,
        )
        cam0_frame = frames[0]
        save_cam0_overlay_image(
            out_path=overlay_cam0_path,
            cam0_rgb_path=cam0_frame.rgb_path,
            aligned_model_pcd=aligned_model,
            pose_transform=np.asarray(icp_result.transformation),
            K=calibrations[0].K,
            D=calibrations[0].D,
        )

    pose_json_path = output_dir / f"pose_frame{fid}_icp.json"
    save_pose_json(
        out_path=pose_json_path,
        transform=np.asarray(icp_result.transformation),
        scale=scale,
        frame_idx=args.frame,
        camera_ids=camera_ids,
        initial_alignment_method=initial_method,
        initial_alignment_result=initial_result,
        ransac_result=ransac_result,
        icp_result=icp_result,
    )

    translation_m = np.asarray(icp_result.transformation)[:3, 3]
    euler_deg = Rotation.from_matrix(np.asarray(icp_result.transformation)[:3, :3]).as_euler("xyz", degrees=True)
    print("\n[RESULT]")
    print(f"  pose json: {pose_json_path}")
    print(f"  aligned ply: {output_dir / f'aligned_reference_frame{fid}_icp.ply'}")
    if not args.no_save_visualization:
        print(f"  visualization png: {visualization_path}")
        print(f"  overlay cam0 png: {overlay_cam0_path}")
    print(f"  translation (m): {np.array2string(translation_m, precision=6)}")
    print(f"  euler xyz (deg): {np.array2string(euler_deg, precision=3)}")


if __name__ == "__main__":
    main()
