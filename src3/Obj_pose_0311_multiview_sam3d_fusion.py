#!/usr/bin/env python3
"""
Multi-view SAM3D-style pose fusion for synchronized RGB-D frames.

Pipeline
1. Build per-camera object masks from RGB-D.
2. Estimate one object pose candidate per camera in that camera frame.
3. Convert each candidate to cam0 with T_C0_O^(i) = T_C0_Ci @ T_Ci_O.
4. Reproject every candidate to every camera and score:
   - mask IoU
   - silhouette overlap (Dice/F1)
   - depth consistency
5. Keep only validated candidates and robustly fuse:
   - translation: weighted Huber mean initialized by weighted median
   - rotation: weighted quaternion average
   - scale: weighted median
6. Refine with:
   - point-to-plane ICP on fused multi-view depth
   - local reprojection optimization using silhouette and depth residuals

Notes
- This repo does not contain the full external SAM3D inference runtime. The script
  therefore estimates per-view T_Ci_O from the provided SAM3D reference geometry
  plus the current RGB-D masks. The fusion, validation, and refinement stages are
  exactly the robust multi-view part you asked for.
- All outputs are in cam0 / OpenCV coordinates:
    X right, Y down, Z forward
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from Obj_pose_0311 import (  # noqa: E402
    build_pointcloud,
    compute_hsv_mask,
    depth_to_pointcloud,
    global_registration_fpfh,
    initial_alignment_pca,
    largest_connected_component,
    load_reference_model,
    load_session,
    rotation_to_euler,
    rotation_to_quaternion,
    run_icp_multistage,
    scale_reference_to_observation,
    transform_points,
)


_DEFAULT_CAPTURE_DIR = _THIS_DIR / "data/object_capture"
_DEFAULT_CALIB_DIR = _THIS_DIR / "data/cube_session_01/calib_out_cube"
_DEFAULT_INTRINSICS_DIR = _THIS_DIR / "data/_intrinsics"
_DEFAULT_REF_MODEL = _THIS_DIR / "Obj_Step2-(2)_pose_estimate_sam3d" / "sam3d_knife.ply"
_DEFAULT_OUT_DIR = _THIS_DIR / "Obj_pose_0311_multiview_sam3d_output"


@dataclass
class CameraObservation:
    camera_id: int
    rgb_path: Path
    depth_path: Path
    bgr: np.ndarray
    depth_u16: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    D: np.ndarray
    T_cam0_cam: np.ndarray
    mask: np.ndarray
    anchor_mask: np.ndarray
    points_cam: np.ndarray
    points_cam0: np.ndarray
    colors: np.ndarray
    dt_outside_mask: np.ndarray


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        Vt[-1, :] *= -1.0
        R_ortho = U @ Vt
    return R_ortho


def subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
    return points[indices]


def weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cdf = np.cumsum(weights_sorted)
    cutoff = 0.5 * float(weights_sorted.sum())
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = min(max(idx, 0), len(values_sorted) - 1)
    return float(values_sorted[idx])


def weighted_median_vec(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.array(
        [weighted_median_1d(values[:, axis], weights) for axis in range(values.shape[1])],
        dtype=np.float64,
    )


def huber_translation_mean(
    translations: np.ndarray,
    weights: np.ndarray,
    delta_m: float,
    num_iters: int = 12,
) -> np.ndarray:
    estimate = weighted_median_vec(translations, weights)
    for _ in range(num_iters):
        residuals = np.linalg.norm(translations - estimate[None, :], axis=1)
        huber = np.ones_like(residuals)
        mask = residuals > delta_m
        huber[mask] = delta_m / np.maximum(residuals[mask], 1e-9)
        eff = weights * huber
        eff_sum = float(eff.sum())
        if eff_sum <= 1e-12:
            break
        updated = np.sum(translations * eff[:, None], axis=0) / eff_sum
        if np.linalg.norm(updated - estimate) < 1e-7:
            estimate = updated
            break
        estimate = updated
    return estimate


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    return rotation_to_quaternion(orthonormalize_rotation(R))


def weighted_quaternion_average(rotations: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    quaternions = np.array([rotation_matrix_to_quaternion(R) for R in rotations], dtype=np.float64)
    ref = quaternions[0]
    for idx in range(len(quaternions)):
        if np.dot(quaternions[idx], ref) < 0:
            quaternions[idx] *= -1.0
    A = np.zeros((4, 4), dtype=np.float64)
    for q, w in zip(quaternions, weights):
        q_col = q.reshape(4, 1)
        A += float(w) * (q_col @ q_col.T)
    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, int(np.argmax(eigvals))]
    if q_avg[0] < 0:
        q_avg *= -1.0
    q_avg /= np.linalg.norm(q_avg) + 1e-12
    return q_avg


def quaternion_to_rotation_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_points(
    points_cam: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(points_cam) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    proj, _ = cv2.projectPoints(
        points_cam.astype(np.float64).reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K.astype(np.float64),
        D.astype(np.float64),
    )
    return proj.reshape(-1, 2), points_cam[:, 2].astype(np.float64)


def render_projected_mask(
    points_cam: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    image_shape: Tuple[int, int],
    dilation_radius_px: int,
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points_cam) == 0:
        return mask.astype(bool)

    uv, depth = project_points(points_cam, K, D)
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    valid = (depth > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(valid):
        return mask.astype(bool)
    mask[v[valid], u[valid]] = 255
    if dilation_radius_px > 0:
        k = 2 * dilation_radius_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask.astype(bool)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())
    return inter / union if union > 0 else 0.0


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a, mask_b).sum())
    denom = float(mask_a.sum() + mask_b.sum())
    return 2.0 * inter / denom if denom > 0 else 0.0


def evaluate_pose_in_camera(
    model_points_scaled: np.ndarray,
    T_cam_obj: np.ndarray,
    observation: CameraObservation,
    render_radius_px: int,
    depth_inlier_thresh_m: float,
) -> Dict[str, float]:
    model_in_cam = transform_points(model_points_scaled, T_cam_obj)
    predicted_mask = render_projected_mask(
        model_in_cam,
        observation.K,
        observation.D,
        observation.mask.shape,
        dilation_radius_px=render_radius_px,
    )

    iou = mask_iou(predicted_mask, observation.mask)
    dice = mask_dice(predicted_mask, observation.mask)

    uv, depth_pred = project_points(model_in_cam, observation.K, observation.D)
    if len(uv) == 0:
        return {
            "mask_iou": 0.0,
            "silhouette_dice": 0.0,
            "depth_inlier_ratio": 0.0,
            "depth_mae_mm": float("inf"),
            "depth_median_mm": float("inf"),
            "predicted_area_px": 0.0,
            "mask_area_px": float(observation.mask.sum()),
        }

    h, w = observation.mask.shape
    u = np.round(uv[:, 0]).astype(np.int32)
    v = np.round(uv[:, 1]).astype(np.int32)
    valid = (depth_pred > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(valid):
        return {
            "mask_iou": iou,
            "silhouette_dice": dice,
            "depth_inlier_ratio": 0.0,
            "depth_mae_mm": float("inf"),
            "depth_median_mm": float("inf"),
            "predicted_area_px": float(predicted_mask.sum()),
            "mask_area_px": float(observation.mask.sum()),
        }

    u = u[valid]
    v = v[valid]
    depth_pred = depth_pred[valid]
    on_object = observation.mask[v, u]
    depth_gt = observation.depth_m[v, u]
    depth_valid = on_object & (depth_gt > 1e-6)

    if not np.any(depth_valid):
        depth_mae_mm = float("inf")
        depth_median_mm = float("inf")
        depth_inlier_ratio = 0.0
    else:
        depth_err = np.abs(depth_pred[depth_valid] - depth_gt[depth_valid])
        depth_mae_mm = float(depth_err.mean() * 1000.0)
        depth_median_mm = float(np.median(depth_err) * 1000.0)
        depth_inlier_ratio = float(np.mean(depth_err < depth_inlier_thresh_m))

    return {
        "mask_iou": float(iou),
        "silhouette_dice": float(dice),
        "depth_inlier_ratio": float(depth_inlier_ratio),
        "depth_mae_mm": float(depth_mae_mm),
        "depth_median_mm": float(depth_median_mm),
        "predicted_area_px": float(predicted_mask.sum()),
        "mask_area_px": float(observation.mask.sum()),
    }


def build_object_mask(
    bgr: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    h_range: Sequence[int],
    s_min: int,
    v_min: int,
    min_area: int,
    depth_band_m: float,
    roi_expand_x: float,
    roi_expand_y: float,
    z_min: float,
    z_max: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    yellow_mask = compute_hsv_mask(bgr, h_range, s_min, v_min)
    anchor_mask = largest_connected_component(yellow_mask, min_area=min_area)
    if anchor_mask is None:
        return None, None

    depth_m = depth_u16.astype(np.float64) * float(depth_scale)
    anchor_depth_values = depth_m[anchor_mask & (depth_m > z_min) & (depth_m < z_max)]
    if len(anchor_depth_values) == 0:
        return None, None

    anchor_depth = float(np.median(anchor_depth_values))
    ys, xs = np.where(anchor_mask)
    if len(xs) == 0:
        return None, None

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    box_w = max(x2 - x1 + 1, 1)
    box_h = max(y2 - y1 + 1, 1)
    cx = int(round((x1 + x2) * 0.5))
    cy = int(round((y1 + y2) * 0.5))
    half_w = int(round(box_w * roi_expand_x))
    half_h = int(round(box_h * roi_expand_y))

    h, w = depth_u16.shape[:2]
    roi = np.zeros((h, w), dtype=bool)
    roi[max(0, cy - half_h):min(h, cy + half_h), max(0, cx - half_w):min(w, cx + half_w)] = True

    depth_ok = (
        (depth_u16 > 0)
        & (depth_m > max(z_min, anchor_depth - depth_band_m))
        & (depth_m < min(z_max, anchor_depth + depth_band_m))
    )

    object_mask = depth_ok & roi
    object_mask |= anchor_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    object_mask = cv2.morphologyEx(object_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
    component = largest_connected_component(object_mask, min_area=max(100, min_area // 2))
    if component is not None:
        object_mask = component
    return object_mask.astype(bool), anchor_mask.astype(bool)


def clean_observation(
    camera_id: int,
    points_cam: np.ndarray,
    points_cam0: np.ndarray,
    colors: np.ndarray,
    anchor_center_cam0: np.ndarray,
    sphere_radius_m: float,
    sor_nb_neighbors: int,
    sor_std_ratio: float,
    dbscan_eps: float,
    dbscan_min_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points_cam0) == 0:
        return points_cam, points_cam0, colors

    keep = np.linalg.norm(points_cam0 - anchor_center_cam0[None, :], axis=1) < sphere_radius_m
    if np.any(keep):
        points_cam = points_cam[keep]
        points_cam0 = points_cam0[keep]
        colors = colors[keep]

    if len(points_cam0) < dbscan_min_points:
        return points_cam, points_cam0, colors

    pcd = build_pointcloud(points_cam0, colors)
    pcd_filtered, indices = pcd.remove_statistical_outlier(
        nb_neighbors=sor_nb_neighbors,
        std_ratio=sor_std_ratio,
    )
    if len(indices) >= dbscan_min_points:
        indices = np.asarray(indices, dtype=np.int32)
        points_cam = points_cam[indices]
        points_cam0 = points_cam0[indices]
        colors = colors[indices]
        pcd = pcd_filtered

    if len(points_cam0) < dbscan_min_points:
        return points_cam, points_cam0, colors

    labels = np.asarray(pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=False))
    valid = labels[labels >= 0]
    if len(valid) == 0:
        return points_cam, points_cam0, colors
    unique, counts = np.unique(valid, return_counts=True)
    best_label = int(unique[np.argmax(counts)])
    keep = labels == best_label
    return points_cam[keep], points_cam0[keep], colors[keep]


def collect_observations(args, session) -> Tuple[Dict[int, CameraObservation], np.ndarray]:
    observations: Dict[int, CameraObservation] = {}
    anchor_points_cam0: List[np.ndarray] = []
    pending_raw: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for camera_id in session.camera_ids:
        frame = session.frames[camera_id]
        calib = session.calibrations[camera_id]
        if frame.bgr is None or frame.depth is None:
            continue

        obj_mask, anchor_mask = build_object_mask(
            bgr=frame.bgr,
            depth_u16=frame.depth,
            depth_scale=calib.depth_scale,
            h_range=args.hsv_h_range,
            s_min=args.hsv_s_min,
            v_min=args.hsv_v_min,
            min_area=args.min_area,
            depth_band_m=args.depth_band_m,
            roi_expand_x=args.roi_expand_x,
            roi_expand_y=args.roi_expand_y,
            z_min=args.z_min,
            z_max=args.z_max,
        )
        if obj_mask is None or anchor_mask is None:
            print(f"    cam{camera_id}: anchor/mask build failed")
            continue

        anchor_points_cam, _ = depth_to_pointcloud(
            frame.depth,
            frame.bgr,
            calib.K,
            calib.D,
            calib.depth_scale,
            mask=anchor_mask,
            z_min=args.z_min,
            z_max=args.z_max,
        )
        if len(anchor_points_cam) == 0:
            print(f"    cam{camera_id}: anchor depth points empty")
            continue
        anchor_points_cam0.append(transform_points(anchor_points_cam, calib.T_cam0_cam))

        points_cam, colors = depth_to_pointcloud(
            frame.depth,
            frame.bgr,
            calib.K,
            calib.D,
            calib.depth_scale,
            mask=obj_mask,
            z_min=args.z_min,
            z_max=args.z_max,
        )
        if len(points_cam) == 0:
            print(f"    cam{camera_id}: object point cloud empty")
            continue
        points_cam0 = transform_points(points_cam, calib.T_cam0_cam)
        depth_m = frame.depth.astype(np.float64) * float(calib.depth_scale)
        dt_outside = cv2.distanceTransform((~obj_mask).astype(np.uint8), cv2.DIST_L2, 3)
        pending_raw[camera_id] = (points_cam, points_cam0, colors, obj_mask, anchor_mask)
        print(
            f"    cam{camera_id}: mask={int(obj_mask.sum()):,} px, "
            f"raw_points={len(points_cam):,}"
        )

    if len(anchor_points_cam0) < 1:
        raise RuntimeError("No valid camera anchor points found.")

    anchor_center_cam0 = np.concatenate(anchor_points_cam0, axis=0).mean(axis=0)
    print(
        "    anchor_center_cam0(mm): "
        f"({anchor_center_cam0[0] * 1000:.1f}, {anchor_center_cam0[1] * 1000:.1f}, {anchor_center_cam0[2] * 1000:.1f})"
    )

    for camera_id, raw in pending_raw.items():
        frame = session.frames[camera_id]
        calib = session.calibrations[camera_id]
        points_cam, points_cam0, colors, obj_mask, anchor_mask = raw
        points_cam, points_cam0, colors = clean_observation(
            camera_id=camera_id,
            points_cam=points_cam,
            points_cam0=points_cam0,
            colors=colors,
            anchor_center_cam0=anchor_center_cam0,
            sphere_radius_m=args.anchor_sphere_radius_m,
            sor_nb_neighbors=args.sor_nb_neighbors,
            sor_std_ratio=args.sor_std_ratio,
            dbscan_eps=args.dbscan_eps_m,
            dbscan_min_points=args.dbscan_min_points,
        )
        if len(points_cam) < args.min_object_points:
            print(f"    cam{camera_id}: cleaned points too few ({len(points_cam)})")
            continue
        observations[camera_id] = CameraObservation(
            camera_id=camera_id,
            rgb_path=frame.rgb_path,
            depth_path=frame.depth_path,
            bgr=frame.bgr,
            depth_u16=frame.depth,
            depth_m=frame.depth.astype(np.float64) * float(calib.depth_scale),
            K=calib.K,
            D=calib.D,
            T_cam0_cam=calib.T_cam0_cam,
            mask=obj_mask,
            anchor_mask=anchor_mask,
            points_cam=points_cam,
            points_cam0=points_cam0,
            colors=colors,
            dt_outside_mask=cv2.distanceTransform((~obj_mask).astype(np.uint8), cv2.DIST_L2, 3),
        )
        extent_mm = (points_cam0.max(axis=0) - points_cam0.min(axis=0)) * 1000.0
        print(
            f"    cam{camera_id}: cleaned_points={len(points_cam0):,}, "
            f"bbox(mm)=({extent_mm[0]:.1f}, {extent_mm[1]:.1f}, {extent_mm[2]:.1f})"
        )

    if len(observations) < 2:
        raise RuntimeError("Need at least 2 valid camera observations for multi-view fusion.")

    return observations, anchor_center_cam0


def estimate_candidate_pose(
    observation: CameraObservation,
    reference_model: o3d.geometry.PointCloud,
    args,
) -> Dict[str, object]:
    obs_pcd = build_pointcloud(observation.points_cam, observation.colors)
    ref_scaled, scale = scale_reference_to_observation(
        reference_model,
        obs_pcd,
        ref_length_mm=args.ref_length_mm,
        manual_scale=args.scale,
    )

    init_candidates: List[Tuple[str, np.ndarray]] = []
    if args.init_method in ("fpfh", "both"):
        try:
            T_fpfh = global_registration_fpfh(ref_scaled, obs_pcd, voxel_size=args.voxel_size_mm / 1000.0)
            init_candidates.append(("fpfh", T_fpfh))
        except Exception as exc:
            print(f"      cam{observation.camera_id}: FPFH init failed: {exc}")
    if args.init_method in ("pca", "both") or not init_candidates:
        T_pca = initial_alignment_pca(np.asarray(ref_scaled.points), observation.points_cam)
        init_candidates.append(("pca", T_pca))

    best: Optional[Tuple[np.ndarray, float, float, str]] = None
    for init_name, init_T in init_candidates:
        try:
            T_cam_obj, fitness, rmse = run_icp_multistage(
                ref_scaled,
                obs_pcd,
                init_transform=init_T,
                max_corr_dist=args.icp_dist_mm / 1000.0,
                fine_iters=args.icp_iters,
            )
        except Exception as exc:
            print(f"      cam{observation.camera_id}: ICP failed for {init_name}: {exc}")
            continue
        if best is None or fitness > best[1] or (fitness == best[1] and rmse < best[2]):
            best = (T_cam_obj, float(fitness), float(rmse), init_name)

    if best is None:
        raise RuntimeError(f"cam{observation.camera_id}: no valid candidate pose found")

    T_cam_obj, fitness, rmse, init_name = best
    T_cam0_obj = observation.T_cam0_cam @ T_cam_obj
    model_points_scaled = np.asarray(ref_scaled.points, dtype=np.float64)

    return {
        "source_camera_id": int(observation.camera_id),
        "scale": float(scale),
        "init_method": init_name,
        "icp_fitness": float(fitness),
        "icp_rmse_mm": float(rmse * 1000.0),
        "T_cam_obj": T_cam_obj,
        "T_cam0_obj": T_cam0_obj,
        "model_points_scaled": model_points_scaled,
    }


def validate_candidates(
    candidates: List[Dict[str, object]],
    observations: Dict[int, CameraObservation],
    args,
) -> None:
    candidate_scores = []
    for candidate in candidates:
        model_points_scaled = candidate["model_points_scaled"]
        T_cam0_obj = candidate["T_cam0_obj"]
        per_camera_metrics: Dict[str, Dict[str, float]] = {}
        ious: List[float] = []
        dices: List[float] = []
        depth_inliers: List[float] = []
        depth_scores: List[float] = []

        for camera_id, observation in observations.items():
            T_cam_obj = invert_transform(observation.T_cam0_cam) @ T_cam0_obj
            metrics = evaluate_pose_in_camera(
                model_points_scaled=model_points_scaled,
                T_cam_obj=T_cam_obj,
                observation=observation,
                render_radius_px=args.render_radius_px,
                depth_inlier_thresh_m=args.depth_inlier_thresh_mm / 1000.0,
            )
            per_camera_metrics[str(camera_id)] = metrics
            ious.append(metrics["mask_iou"])
            dices.append(metrics["silhouette_dice"])
            depth_inliers.append(metrics["depth_inlier_ratio"])
            if np.isfinite(metrics["depth_median_mm"]):
                depth_scores.append(np.exp(-metrics["depth_median_mm"] / args.depth_score_scale_mm))
            else:
                depth_scores.append(0.0)

        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_dice = float(np.mean(dices)) if dices else 0.0
        mean_depth_inlier = float(np.mean(depth_inliers)) if depth_inliers else 0.0
        mean_depth_score = float(np.mean(depth_scores)) if depth_scores else 0.0
        composite = (
            args.score_iou_weight * mean_iou
            + args.score_silhouette_weight * mean_dice
            + args.score_depth_weight * mean_depth_score
        )
        candidate["validation"] = {
            "per_camera": per_camera_metrics,
            "mean_mask_iou": mean_iou,
            "mean_silhouette_dice": mean_dice,
            "mean_depth_inlier_ratio": mean_depth_inlier,
            "mean_depth_score": mean_depth_score,
            "composite_score": float(composite),
        }
        candidate_scores.append(composite)

    if not candidates:
        raise RuntimeError("No candidates to validate.")

    best_score = max(candidate_scores)
    accepted_any = False
    for candidate in candidates:
        validation = candidate["validation"]
        accepted = (
            validation["composite_score"] >= max(args.min_candidate_score, best_score * args.accept_ratio)
            and validation["mean_mask_iou"] >= args.min_mean_iou
            and validation["mean_depth_inlier_ratio"] >= args.min_mean_depth_inlier
        )
        candidate["accepted"] = bool(accepted)
        accepted_any = accepted_any or accepted

    if not accepted_any:
        best_idx = int(np.argmax(candidate_scores))
        candidates[best_idx]["accepted"] = True


def fuse_validated_candidates(candidates: List[Dict[str, object]], args) -> Tuple[np.ndarray, float, Dict[str, object]]:
    accepted = [candidate for candidate in candidates if candidate.get("accepted", False)]
    if not accepted:
        raise RuntimeError("No validated candidates to fuse.")

    weights = np.array(
        [max(1e-6, float(candidate["validation"]["composite_score"])) for candidate in accepted],
        dtype=np.float64,
    )
    weights /= weights.sum()

    translations = np.array([candidate["T_cam0_obj"][:3, 3] for candidate in accepted], dtype=np.float64)
    rotations = [candidate["T_cam0_obj"][:3, :3] for candidate in accepted]
    scales = np.array([float(candidate["scale"]) for candidate in accepted], dtype=np.float64)

    translation = huber_translation_mean(
        translations=translations,
        weights=weights,
        delta_m=args.translation_huber_delta_mm / 1000.0,
    )
    quaternion = weighted_quaternion_average(rotations, weights)
    rotation = quaternion_to_rotation_matrix(quaternion)
    scale = weighted_median_1d(scales, weights)

    fused_T = make_transform(rotation, translation)
    stats = {
        "accepted_source_cameras": [int(candidate["source_camera_id"]) for candidate in accepted],
        "weights": weights.tolist(),
        "translation_huber_delta_mm": float(args.translation_huber_delta_mm),
        "scale_method": "weighted_median",
        "rotation_method": "weighted_quaternion_average",
        "translation_method": "weighted_huber_mean_initialized_by_weighted_median",
    }
    return fused_T, float(scale), stats


def optimize_reprojection_pose(
    init_T_cam0_obj: np.ndarray,
    model_points_scaled: np.ndarray,
    observations: Dict[int, CameraObservation],
    args,
) -> Tuple[np.ndarray, Dict[str, object]]:
    sampled = subsample_points(model_points_scaled, args.reproj_opt_points)
    invalid_silhouette_res = args.invalid_projection_penalty_px / max(args.silhouette_sigma_px, 1e-6)

    def residual_fn(delta: np.ndarray) -> np.ndarray:
        rot_delta = Rotation.from_rotvec(delta[:3]).as_matrix()
        translation = init_T_cam0_obj[:3, 3] + delta[3:6]
        rotation = rot_delta @ init_T_cam0_obj[:3, :3]
        T_cam0_obj = make_transform(rotation, translation)

        residuals: List[np.ndarray] = []
        for observation in observations.values():
            T_cam_obj = invert_transform(observation.T_cam0_cam) @ T_cam0_obj
            points_cam = transform_points(sampled, T_cam_obj)
            uv, depth_pred = project_points(points_cam, observation.K, observation.D)
            silhouette_res = np.full((len(sampled),), invalid_silhouette_res, dtype=np.float64)
            depth_res = np.zeros((len(sampled),), dtype=np.float64)
            if len(uv) == 0:
                residuals.append(silhouette_res)
                residuals.append(depth_res)
                continue

            h, w = observation.mask.shape
            u = np.round(uv[:, 0]).astype(np.int32)
            v = np.round(uv[:, 1]).astype(np.int32)
            valid = (depth_pred > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if np.any(valid):
                valid_idx = np.flatnonzero(valid)
                silhouette_res[valid_idx] = (
                    observation.dt_outside_mask[v[valid], u[valid]] / max(args.silhouette_sigma_px, 1e-6)
                )
                depth_gt = observation.depth_m[v[valid], u[valid]]
                depth_valid = observation.mask[v[valid], u[valid]] & (depth_gt > 1e-6)
                if np.any(depth_valid):
                    depth_res[valid_idx[depth_valid]] = (
                        (depth_pred[valid][depth_valid] - depth_gt[depth_valid]) / max(args.depth_sigma_m, 1e-9)
                    )

            residuals.append(silhouette_res)
            residuals.append(depth_res)

        return np.concatenate(residuals)

    bounds_lo = np.array(
        [
            -np.deg2rad(args.reproj_rot_bound_deg),
            -np.deg2rad(args.reproj_rot_bound_deg),
            -np.deg2rad(args.reproj_rot_bound_deg),
            -args.reproj_trans_bound_mm / 1000.0,
            -args.reproj_trans_bound_mm / 1000.0,
            -args.reproj_trans_bound_mm / 1000.0,
        ],
        dtype=np.float64,
    )
    bounds_hi = -bounds_lo
    result = least_squares(
        residual_fn,
        x0=np.zeros((6,), dtype=np.float64),
        bounds=(bounds_lo, bounds_hi),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=args.reproj_max_nfev,
        verbose=0,
    )

    rot_delta = Rotation.from_rotvec(result.x[:3]).as_matrix()
    refined_rotation = orthonormalize_rotation(rot_delta @ init_T_cam0_obj[:3, :3])
    refined_translation = init_T_cam0_obj[:3, 3] + result.x[3:6]
    refined_T = make_transform(refined_rotation, refined_translation)
    info = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "delta_rotvec_deg": np.degrees(result.x[:3]).tolist(),
        "delta_translation_mm": (result.x[3:6] * 1000.0).tolist(),
    }
    return refined_T, info


def draw_axes_on_image(
    image_bgr: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    axis_length_m: float,
) -> None:
    origin = T_cam_obj[:3, 3]
    R = T_cam_obj[:3, :3]
    axes = np.stack(
        [
            origin,
            origin + R[:, 0] * axis_length_m,
            origin + R[:, 1] * axis_length_m,
            origin + R[:, 2] * axis_length_m,
        ],
        axis=0,
    )
    uv, _ = cv2.projectPoints(
        axes.astype(np.float64).reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        K.astype(np.float64),
        D.astype(np.float64),
    )
    uv = np.round(uv.reshape(-1, 2)).astype(np.int32)
    origin_uv = tuple(uv[0])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for idx in range(3):
        tip = tuple(uv[idx + 1])
        cv2.arrowedLine(image_bgr, origin_uv, tip, colors[idx], 2, tipLength=0.2)


def save_final_overlays(
    out_dir: Path,
    frame_idx: int,
    observations: Dict[int, CameraObservation],
    final_T_cam0_obj: np.ndarray,
    model_points_scaled: np.ndarray,
    args,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for camera_id, observation in observations.items():
        T_cam_obj = invert_transform(observation.T_cam0_cam) @ final_T_cam0_obj
        predicted_mask = render_projected_mask(
            transform_points(model_points_scaled, T_cam_obj),
            observation.K,
            observation.D,
            observation.mask.shape,
            dilation_radius_px=args.render_radius_px,
        )
        overlay = observation.bgr.copy()
        gt_uint8 = (observation.mask.astype(np.uint8) * 255)
        pred_uint8 = (predicted_mask.astype(np.uint8) * 255)
        gt_contours, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)
        cv2.drawContours(overlay, pred_contours, -1, (0, 0, 255), 2)
        draw_axes_on_image(overlay, T_cam_obj, observation.K, observation.D, axis_length_m=args.axis_length_m)
        score = evaluate_pose_in_camera(
            model_points_scaled=model_points_scaled,
            T_cam_obj=T_cam_obj,
            observation=observation,
            render_radius_px=args.render_radius_px,
            depth_inlier_thresh_m=args.depth_inlier_thresh_mm / 1000.0,
        )
        cv2.putText(
            overlay,
            f"cam{camera_id}  IoU={score['mask_iou']:.3f} Dice={score['silhouette_dice']:.3f} DepthInlier={score['depth_inlier_ratio']:.3f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            overlay,
            "Green=observed mask  Red=projected model  Axes=RGB xyz",
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(
            str(out_dir / f"overlay_cam{camera_id}_frame{frame_idx:06d}.png"),
            overlay,
        )
        cv2.imwrite(
            str(out_dir / f"mask_cam{camera_id}_frame{frame_idx:06d}.png"),
            gt_uint8,
        )


def save_point_cloud(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    pcd = build_pointcloud(points, colors)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def serialize_candidate(candidate: Dict[str, object]) -> Dict[str, object]:
    payload = {
        "source_camera_id": int(candidate["source_camera_id"]),
        "accepted": bool(candidate.get("accepted", False)),
        "scale": float(candidate["scale"]),
        "init_method": str(candidate["init_method"]),
        "icp_fitness": float(candidate["icp_fitness"]),
        "icp_rmse_mm": float(candidate["icp_rmse_mm"]),
        "T_cam_obj": np.asarray(candidate["T_cam_obj"], dtype=np.float64).tolist(),
        "T_cam0_obj": np.asarray(candidate["T_cam0_obj"], dtype=np.float64).tolist(),
        "validation": candidate["validation"],
    }
    return payload


def save_results(
    out_dir: Path,
    frame_idx: int,
    reference_model_path: str,
    candidates: List[Dict[str, object]],
    fusion_stats: Dict[str, object],
    fused_T: np.ndarray,
    fused_scale: float,
    icp_T: np.ndarray,
    icp_fitness: float,
    icp_rmse: float,
    final_T: np.ndarray,
    final_scale: float,
    reproj_info: Dict[str, object],
    observations: Dict[int, CameraObservation],
    model_points_scaled_final: np.ndarray,
    elapsed_sec: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    observed_points = np.concatenate([obs.points_cam0 for obs in observations.values()], axis=0)
    observed_colors = np.concatenate([obs.colors for obs in observations.values()], axis=0)
    aligned_points = transform_points(model_points_scaled_final, final_T)

    save_point_cloud(out_dir / f"observed_frame{frame_idx:06d}.ply", observed_points, observed_colors)
    save_point_cloud(out_dir / f"aligned_reference_frame{frame_idx:06d}.ply", aligned_points)

    position_m = final_T[:3, 3]
    position_mm = position_m * 1000.0
    rotation = final_T[:3, :3]
    euler = rotation_to_euler(rotation).tolist()
    quaternion = rotation_to_quaternion(rotation).tolist()

    result = {
        "frame": int(frame_idx),
        "coordinate_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
        "reference_model": str(Path(reference_model_path).resolve()),
        "candidate_pose_method": "per_view_masked_depth_alignment_to_reference",
        "candidates": [serialize_candidate(candidate) for candidate in candidates],
        "fusion": {
            "T_cam0_obj": fused_T.tolist(),
            "scale": float(fused_scale),
            **fusion_stats,
        },
        "refinement": {
            "point_to_plane_icp": {
                "T_cam0_obj": icp_T.tolist(),
                "fitness": float(icp_fitness),
                "rmse_mm": float(icp_rmse * 1000.0),
            },
            "silhouette_depth_reprojection": reproj_info,
        },
        "final_pose_cam0": {
            "position_m": position_m.tolist(),
            "position_mm": position_mm.tolist(),
            "rotation_matrix": rotation.tolist(),
            "euler_xyz_deg": euler,
            "quaternion_wxyz": quaternion,
            "scale": float(final_scale),
        },
        "elapsed_sec": round(float(elapsed_sec), 3),
    }

    json_path = out_dir / f"pose_frame{frame_idx:06d}_multiview_sam3d.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-view SAM3D-style pose fusion with mask validation and depth refinement.",
    )
    parser.add_argument("--capture_dir", default=str(_DEFAULT_CAPTURE_DIR))
    parser.add_argument("--calib_dir", default=str(_DEFAULT_CALIB_DIR))
    parser.add_argument("--intrinsics_dir", default=str(_DEFAULT_INTRINSICS_DIR))
    parser.add_argument("--reference_model", default=str(_DEFAULT_REF_MODEL))
    parser.add_argument("--out_dir", default=str(_DEFAULT_OUT_DIR))
    parser.add_argument("--frame", type=int, default=3)

    parser.add_argument("--z_min", type=float, default=0.10)
    parser.add_argument("--z_max", type=float, default=1.50)
    parser.add_argument("--hsv_h_range", type=int, nargs=2, default=[15, 35])
    parser.add_argument("--hsv_s_min", type=int, default=80)
    parser.add_argument("--hsv_v_min", type=int, default=80)
    parser.add_argument("--min_area", type=int, default=500)
    parser.add_argument("--depth_band_m", type=float, default=0.025)
    parser.add_argument("--roi_expand_x", type=float, default=3.5)
    parser.add_argument("--roi_expand_y", type=float, default=2.2)
    parser.add_argument("--anchor_sphere_radius_m", type=float, default=0.14)
    parser.add_argument("--min_object_points", type=int, default=300)
    parser.add_argument("--sor_nb_neighbors", type=int, default=20)
    parser.add_argument("--sor_std_ratio", type=float, default=1.5)
    parser.add_argument("--dbscan_eps_m", type=float, default=0.008)
    parser.add_argument("--dbscan_min_points", type=int, default=20)

    parser.add_argument("--ref_length_mm", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--voxel_size_mm", type=float, default=5.0)
    parser.add_argument("--icp_dist_mm", type=float, default=6.0)
    parser.add_argument("--icp_iters", type=int, default=120)
    parser.add_argument("--init_method", choices=["pca", "fpfh", "both"], default="both")

    parser.add_argument("--render_radius_px", type=int, default=3)
    parser.add_argument("--depth_inlier_thresh_mm", type=float, default=8.0)
    parser.add_argument("--depth_score_scale_mm", type=float, default=12.0)
    parser.add_argument("--score_iou_weight", type=float, default=0.40)
    parser.add_argument("--score_silhouette_weight", type=float, default=0.25)
    parser.add_argument("--score_depth_weight", type=float, default=0.35)
    parser.add_argument("--accept_ratio", type=float, default=0.85)
    parser.add_argument("--min_candidate_score", type=float, default=0.15)
    parser.add_argument("--min_mean_iou", type=float, default=0.05)
    parser.add_argument("--min_mean_depth_inlier", type=float, default=0.15)

    parser.add_argument("--translation_huber_delta_mm", type=float, default=25.0)
    parser.add_argument("--reproj_opt_points", type=int, default=3000)
    parser.add_argument("--reproj_points_per_camera", type=int, default=1400)
    parser.add_argument("--silhouette_sigma_px", type=float, default=3.0)
    parser.add_argument("--invalid_projection_penalty_px", type=float, default=8.0)
    parser.add_argument("--depth_sigma_m", type=float, default=0.005)
    parser.add_argument("--reproj_rot_bound_deg", type=float, default=12.0)
    parser.add_argument("--reproj_trans_bound_mm", type=float, default=25.0)
    parser.add_argument("--reproj_max_nfev", type=int, default=40)
    parser.add_argument("--axis_length_m", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    print("=" * 72)
    print("  Obj_pose_0311_multiview_sam3d_fusion.py")
    print(f"  frame           : {args.frame}")
    print(f"  reference_model : {args.reference_model}")
    print("=" * 72)

    print("\n[Step 1] Load session")
    session = load_session(args)

    print("\n[Step 2] Build per-camera masks and object observations")
    observations, anchor_center = collect_observations(args, session)
    print(f"  valid cameras: {sorted(observations.keys())}")

    print(f"\n[Step 3] Load reference model: {args.reference_model}")
    reference_model = load_reference_model(args.reference_model, num_samples=50000)
    raw_model_points = np.asarray(reference_model.points, dtype=np.float64)

    print("\n[Step 4] Per-view pose candidates")
    candidates: List[Dict[str, object]] = []
    for camera_id in sorted(observations.keys()):
        print(f"    source cam{camera_id}")
        candidate = estimate_candidate_pose(observations[camera_id], reference_model, args)
        candidates.append(candidate)
        center_mm = candidate["T_cam0_obj"][:3, 3] * 1000.0
        print(
            f"      scale={candidate['scale']:.6f} "
            f"fitness={candidate['icp_fitness']:.4f} "
            f"rmse={candidate['icp_rmse_mm']:.3f}mm "
            f"center(mm)=({center_mm[0]:.1f}, {center_mm[1]:.1f}, {center_mm[2]:.1f})"
        )

    print("\n[Step 5] Cross-view reprojection validation")
    validate_candidates(candidates, observations, args)
    for candidate in candidates:
        validation = candidate["validation"]
        print(
            f"    cam{candidate['source_camera_id']} -> "
            f"score={validation['composite_score']:.3f} "
            f"IoU={validation['mean_mask_iou']:.3f} "
            f"Dice={validation['mean_silhouette_dice']:.3f} "
            f"DepthInlier={validation['mean_depth_inlier_ratio']:.3f} "
            f"accepted={candidate['accepted']}"
        )

    print("\n[Step 6] Robust fusion in cam0")
    fused_T, fused_scale, fusion_stats = fuse_validated_candidates(candidates, args)
    fused_center_mm = fused_T[:3, 3] * 1000.0
    print(
        f"  fused center(mm)=({fused_center_mm[0]:.1f}, {fused_center_mm[1]:.1f}, {fused_center_mm[2]:.1f}) "
        f"scale={fused_scale:.6f}"
    )

    print("\n[Step 7] Point-to-plane ICP refinement on fused multi-view depth")
    model_points_scaled = raw_model_points * float(fused_scale)
    model_pcd_scaled = build_pointcloud(model_points_scaled)
    observed_points = np.concatenate([obs.points_cam0 for obs in observations.values()], axis=0)
    observed_colors = np.concatenate([obs.colors for obs in observations.values()], axis=0)
    observed_pcd = build_pointcloud(observed_points, observed_colors)
    icp_T, icp_fitness, icp_rmse = run_icp_multistage(
        model_pcd_scaled,
        observed_pcd,
        init_transform=fused_T,
        max_corr_dist=args.icp_dist_mm / 1000.0,
        fine_iters=args.icp_iters,
    )
    icp_center_mm = icp_T[:3, 3] * 1000.0
    print(
        f"  icp center(mm)=({icp_center_mm[0]:.1f}, {icp_center_mm[1]:.1f}, {icp_center_mm[2]:.1f}) "
        f"fitness={icp_fitness:.4f} rmse={icp_rmse * 1000.0:.3f}mm"
    )

    print("\n[Step 8] Silhouette + depth reprojection refinement")
    final_T, reproj_info = optimize_reprojection_pose(
        init_T_cam0_obj=icp_T,
        model_points_scaled=model_points_scaled,
        observations=observations,
        args=args,
    )
    final_center_mm = final_T[:3, 3] * 1000.0
    final_euler = rotation_to_euler(final_T[:3, :3])
    print(
        f"  final center(mm)=({final_center_mm[0]:.1f}, {final_center_mm[1]:.1f}, {final_center_mm[2]:.1f}) "
        f"euler(deg)=({final_euler[0]:.1f}, {final_euler[1]:.1f}, {final_euler[2]:.1f})"
    )
    print(
        f"  reproj success={reproj_info['success']} cost={reproj_info['cost']:.3f} nfev={reproj_info['nfev']}"
    )

    elapsed = time.time() - start_time
    out_dir = Path(args.out_dir) / f"output_frame{args.frame:06d}"

    print("\n[Step 9] Save outputs")
    save_final_overlays(
        out_dir=out_dir,
        frame_idx=args.frame,
        observations=observations,
        final_T_cam0_obj=final_T,
        model_points_scaled=model_points_scaled,
        args=args,
    )
    json_path = save_results(
        out_dir=out_dir,
        frame_idx=args.frame,
        reference_model_path=args.reference_model,
        candidates=candidates,
        fusion_stats=fusion_stats,
        fused_T=fused_T,
        fused_scale=fused_scale,
        icp_T=icp_T,
        icp_fitness=icp_fitness,
        icp_rmse=icp_rmse,
        final_T=final_T,
        final_scale=fused_scale,
        reproj_info=reproj_info,
        observations=observations,
        model_points_scaled_final=model_points_scaled,
        elapsed_sec=elapsed,
    )

    final_quat = rotation_to_quaternion(final_T[:3, :3])
    print("\n" + "=" * 72)
    print("  Final pose in cam0 / OpenCV")
    print("=" * 72)
    print(
        "  position_mm    : "
        f"({final_center_mm[0]:+.2f}, {final_center_mm[1]:+.2f}, {final_center_mm[2]:+.2f})"
    )
    print(
        "  euler_xyz_deg  : "
        f"({final_euler[0]:+.2f}, {final_euler[1]:+.2f}, {final_euler[2]:+.2f})"
    )
    print(
        "  quaternion_wxyz: "
        f"({final_quat[0]:+.5f}, {final_quat[1]:+.5f}, {final_quat[2]:+.5f}, {final_quat[3]:+.5f})"
    )
    print(f"  scale          : {fused_scale:.6f}")
    print(f"  output         : {json_path}")
    print(f"  elapsed_sec    : {elapsed:.3f}")


if __name__ == "__main__":
    main()
