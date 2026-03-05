# reconstruct_3d.py
# 멀티카메라 RGBD → 3D 포인트클라우드 복원
# ArUco 마커 없이, 기존 캘리브레이션 결과(T_C0_Ci.npy)만 사용
#
# 사용법 (capture_rgbd_3cam.py로 촬영한 데이터 기준):
#
"""
python reconstruct_3d.py \
  --capture_dir ./data/rgbd_capture \
  --intrinsics_dir ./intrinsics \
  --calib_dir ./data/cube_session_01/calib_out_cube \
  --each_frame --no_plot --remove_plane

# ICP로 캘리브레이션 보정 후 재구성 (찌그러짐 개선)
python reconstruct_3d.py \
  --capture_dir ./data/rgbd_capture \
  --intrinsics_dir ./intrinsics \
  --calib_dir ./data/cube_session_01/calib_out_cube \
  --each_frame --no_plot --remove_plane --icp
"""

"""
단일 프레임 복원
python reconstruct_3d.py \
    --capture_dir ./data/rgbd_capture \
    --intrinsics_dir ./intrinsics \
    --calib_dir ./data/cube_session_01/calib_out_cube \
    --frame 0
"""

"""
모든 프레임 각각 개별 PLY로 저장 (물체마다 다를 때)
python reconstruct_3d.py \
    --capture_dir ./data/rgbd_capture 
    --intrinsics_dir ./intrinsics \
    --calib_dir ./data/cube_session_01/calib_out_cube \
    --each_frame --no_plot
"""
#
#   # 전체 프레임 하나로 합치기 (같은 물체를 여러 번 찍었을 때)
#   python reconstruct_3d.py \
#     --capture_dir ./data/rgbd_capture \
#     --intrinsics_dir ./intrinsics \
#     --calib_dir ./data/cube_session_01/calib_out_cube \
#     --all_frames --voxel_mm 2
#
"""
Open3D 뷰어로 보기
python reconstruct_3d.py \
    --capture_dir ./data/rgbd_capture \
    --intrinsics_dir ./intrinsics \
    --calib_dir ./data/cube_session_01/calib_out_cube \
    --frame 0 --open3d
"""

import os
import glob
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


# ──────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────
def load_intrinsics(intrinsics_dir: str, cam_idx: int):
    p = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"intrinsics 없음: {p}")
    data = np.load(p, allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64)
    depth_scale = float(data["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in data else 0.001
    return K, D, depth_scale


def load_extrinsics(calib_dir: str, ref_idx: int, cam_indices: List[int]) -> Dict[int, np.ndarray]:
    T_ref = {}
    for ci in cam_indices:
        if ci == ref_idx:
            T_ref[ci] = np.eye(4, dtype=np.float64)
            continue
        p = os.path.join(calib_dir, f"T_C{ref_idx}_C{ci}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"extrinsics 없음: {p}  (Step3 캘리브레이션 먼저 수행 필요)")
        T_ref[ci] = np.load(p).astype(np.float64)
        print(f"[INFO] Loaded T_C{ref_idx}_C{ci}")
    return T_ref


def discover_cameras(capture_dir: str) -> List[int]:
    cam_dirs = sorted(glob.glob(os.path.join(capture_dir, "cam*")))
    indices = []
    for d in cam_dirs:
        name = os.path.basename(d)
        if not name.startswith("cam"):
            continue
        try:
            idx = int(name.replace("cam", ""))
        except ValueError:
            continue
        if glob.glob(os.path.join(d, "rgb_*.jpg")):
            indices.append(idx)
    if not indices:
        raise RuntimeError(f"cam*/rgb_*.jpg 파일을 찾을 수 없음: {capture_dir}")
    return indices


def discover_frames(capture_dir: str, cam_indices: List[int]) -> List[int]:
    frames = set()
    for ci in cam_indices:
        for p in glob.glob(os.path.join(capture_dir, f"cam{ci}", "rgb_*.jpg")):
            fname = os.path.basename(p)              # rgb_000012.jpg
            num = fname.replace("rgb_", "").replace(".jpg", "")
            try:
                frames.add(int(num))
            except ValueError:
                continue
    if not frames:
        raise RuntimeError(f"프레임을 찾을 수 없음: {capture_dir}")
    return sorted(frames)


def _detect_zero_padding(capture_dir: str, cam_idx: int) -> int:
    """실제 파일명에서 zero-padding 자릿수를 감지한다."""
    pattern = os.path.join(capture_dir, f"cam{cam_idx}", "rgb_*.jpg")
    files = glob.glob(pattern)
    if not files:
        return 6  # default
    sample = os.path.basename(files[0])  # rgb_000012.jpg
    num_str = sample.replace("rgb_", "").replace(".jpg", "")
    return len(num_str)


# ──────────────────────────────────────────────────
# 3D reconstruction core
# ──────────────────────────────────────────────────
def depth_inpaint(depth_u16: np.ndarray, max_hole_px: int = 5) -> np.ndarray:
    """
    Depth 구멍 채우기: depth=0인 소규모 구멍을 주변 값으로 보간.
    물체 표면의 빈 영역을 채워서 포인트클라우드 밀도를 높임.
    - max_hole_px: 채울 구멍 최대 반경 (px). 너무 크면 배경을 오염시킴.
    """
    mask = (depth_u16 == 0).astype(np.uint8)
    # 작은 구멍만 채우기 위해 morphological closing 먼저 적용
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_hole_px, max_hole_px))
    # 구멍 영역 중 주변에 유효 값이 있는 부분만 채움
    depth_f32 = depth_u16.astype(np.float32)
    inpainted = cv2.inpaint(depth_f32, mask, max_hole_px, cv2.INPAINT_NS)
    return inpainted.astype(np.uint16)


def bilateral_depth_filter(depth_u16: np.ndarray, d: int = 5,
                           sigma_color: float = 30.0,
                           sigma_space: float = 5.0) -> np.ndarray:
    """
    Bilateral filter: depth 노이즈를 줄이되 엣지(물체 경계)는 보존.
    - sigma_color: 값 차이 허용 범위 (작을수록 엣지 보존 강함)
    - sigma_space: 공간적 영향 범위 (작을수록 좁은 이웃만 참고)
    """
    depth_f32 = depth_u16.astype(np.float32)
    filtered = cv2.bilateralFilter(depth_f32, d, sigma_color, sigma_space)
    return filtered.astype(np.uint16)


def depth_to_points(depth_u16: np.ndarray, K: np.ndarray, D: np.ndarray,
                    depth_scale: float, z_min: float, z_max: float,
                    stride: int = 1, undistort: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    depth 이미지 → camera frame 3D 점 + 픽셀 좌표(v, u).
    - undistort=True: 렌즈 왜곡 보정 적용 (외곽부 3~5mm 오차 제거)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    h, w = depth_u16.shape[:2]

    v_grid, u_grid = np.mgrid[0:h:stride, 0:w:stride]
    d = depth_u16[v_grid, u_grid].astype(np.float64) * depth_scale

    valid = (d > z_min) & (d < z_max)
    z = d[valid]
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)

    if undistort and D is not None and np.any(D != 0):
        # 렌즈 왜곡 보정: 픽셀 좌표 → 보정된 normalized 좌표 → 3D
        pts_2d = np.column_stack([u, v]).reshape(-1, 1, 2).astype(np.float64)
        pts_undist = cv2.undistortPoints(pts_2d, K, D, P=None)  # normalized coords
        pts_undist = pts_undist.reshape(-1, 2)
        x = pts_undist[:, 0] * z
        y = pts_undist[:, 1] * z
    else:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

    points = np.column_stack([x, y, z])
    pixels = np.column_stack([v.astype(np.int32), u.astype(np.int32)])
    return points, pixels


def statistical_outlier_removal(points: np.ndarray, colors: np.ndarray,
                                nb_neighbors: int = 20,
                                std_ratio: float = 2.0
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Statistical Outlier Removal (SOR):
    각 점의 k-nearest neighbor 평균 거리를 구하고,
    전체 평균 + std_ratio * 표준편차를 초과하는 점을 제거.
    떠다니는 노이즈 점을 효과적으로 제거.
    """
    if points.shape[0] < nb_neighbors + 1:
        return points, colors

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=nb_neighbors + 1)  # +1: 자기 자신 포함
        mean_dists = dists[:, 1:].mean(axis=1)  # 자기 자신 제외

        global_mean = mean_dists.mean()
        global_std = mean_dists.std()
        threshold = global_mean + std_ratio * global_std

        inlier_mask = mean_dists < threshold
        n_removed = int((~inlier_mask).sum())
        if n_removed > 0:
            print(f"    SOR: {n_removed:,} outliers 제거 "
                  f"(threshold={threshold:.4f}m, {100*n_removed/len(points):.1f}%)")
        return points[inlier_mask], colors[inlier_mask]

    except ImportError:
        print("    [WARN] scipy 없음, SOR 건너뜀 (pip install scipy)")
        return points, colors


def remove_background_plane(points: np.ndarray, colors: np.ndarray,
                            distance_threshold: float = 0.005,
                            ransac_n: int = 3,
                            num_iterations: int = 1000,
                            min_plane_ratio: float = 0.15
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC 평면 검출로 배경(테이블/바닥) 제거.
    가장 큰 평면을 찾아서 제거하면 물체만 남음.
    - distance_threshold: 평면으로부터 이 거리(m) 이내의 점을 평면으로 간주
    - min_plane_ratio: 전체 대비 이 비율 이상이어야 평면으로 인정
    """
    if points.shape[0] < 100:
        return points, colors

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        n_plane = len(inliers)
        ratio = n_plane / len(points)

        if ratio < min_plane_ratio:
            print(f"    plane: {ratio*100:.1f}% (< {min_plane_ratio*100:.0f}% threshold) -> 제거 안 함")
            return points, colors

        # 평면이 아닌 점만 남기기 (= 물체)
        outlier_pcd = pcd.select_by_index(inliers, invert=True)
        pts_out = np.asarray(outlier_pcd.points)
        cols_out = np.asarray(outlier_pcd.colors)
        a, b, c, d = plane_model
        print(f"    plane 제거: {n_plane:,}점 ({ratio*100:.1f}%) "
              f"[{a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0]")
        return pts_out, cols_out

    except ImportError:
        # open3d 없으면 numpy RANSAC fallback
        best_inliers = None
        n = len(points)
        for _ in range(num_iterations):
            idx = np.random.choice(n, ransac_n, replace=False)
            p0, p1, p2 = points[idx[0]], points[idx[1]], points[idx[2]]
            normal = np.cross(p1 - p0, p2 - p0)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-10:
                continue
            normal /= norm_len
            d = -normal.dot(p0)
            dists = np.abs(points.dot(normal) + d)
            inliers = np.where(dists < distance_threshold)[0]
            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers

        if best_inliers is not None:
            ratio = len(best_inliers) / n
            if ratio >= min_plane_ratio:
                mask = np.ones(n, dtype=bool)
                mask[best_inliers] = False
                print(f"    plane 제거 (numpy): {len(best_inliers):,}점 ({ratio*100:.1f}%)")
                return points[mask], colors[mask]

        return points, colors


def _build_pcd_with_color(capture_dir, cam_idx, frame_ids, K, D, ds,
                           z_min, z_max, pad, stride=2):
    """프레임들을 모아 colored point cloud 생성 (ICP용)."""
    fmt = f"{{:0{pad}d}}"
    all_pts, all_cols = [], []
    for fid in frame_ids:
        fid_str = fmt.format(fid)
        rgb_path = os.path.join(capture_dir, f"cam{cam_idx}", f"rgb_{fid_str}.jpg")
        depth_path = os.path.join(capture_dir, f"cam{cam_idx}", f"depth_{fid_str}.png")
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            continue
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_u16 is None:
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        pts, pixels = depth_to_points(depth_u16, K, D, ds, z_min, z_max,
                                      stride=stride, undistort=True)
        if pts.shape[0] < 100:
            continue
        cols = rgb[pixels[:, 0], pixels[:, 1]]
        all_pts.append(pts)
        all_cols.append(cols)
    if not all_pts:
        return None, None
    return np.concatenate(all_pts), np.concatenate(all_cols)


def refine_extrinsics_icp(
    capture_dir: str, frame_ids: List[int], cam_indices: List[int],
    T_ref: Dict[int, np.ndarray], K_map: Dict[int, np.ndarray],
    D_map: Dict[int, np.ndarray], ds_map: Dict[int, float],
    ref_idx: int, z_min: float, z_max: float, pad: int,
    max_correspondence_dist: float = 0.02,
    n_sample_frames: int = 5,
) -> Dict[int, np.ndarray]:
    """
    Robust extrinsics 보정:
      1) FPFH feature 기반 RANSAC global registration (대략 정합)
      2) Multi-scale colored ICP (정밀 정합 — 기하 + 색상)
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[WARN] open3d 없음, ICP refinement 건너뜀")
        return T_ref

    print(f"\n[REG] Robust extrinsics refinement 시작")
    print(f"      FPFH global registration → Multi-scale Colored ICP")

    sample_ids = frame_ids[:min(n_sample_frames, len(frame_ids))]

    # ref 카메라 포인트 빌드
    ref_pts, ref_cols = _build_pcd_with_color(
        capture_dir, ref_idx, sample_ids,
        K_map[ref_idx], D_map.get(ref_idx), ds_map[ref_idx],
        z_min, z_max, pad, stride=2,
    )
    if ref_pts is None:
        print("[REG] ref 카메라 데이터 부족, 원본 유지")
        return T_ref

    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(ref_pts)
    pcd_ref.colors = o3d.utility.Vector3dVector(ref_cols)

    T_refined = {}
    T_refined[ref_idx] = np.eye(4, dtype=np.float64)

    for ci in cam_indices:
        if ci == ref_idx:
            continue

        print(f"\n[REG] cam{ci} 정합 중...")

        ci_pts, ci_cols = _build_pcd_with_color(
            capture_dir, ci, sample_ids,
            K_map[ci], D_map.get(ci), ds_map[ci],
            z_min, z_max, pad, stride=2,
        )
        if ci_pts is None:
            print(f"  cam{ci}: 데이터 부족, 원본 유지")
            T_refined[ci] = T_ref[ci].copy()
            continue

        # ci 포인트를 현재 캘리브레이션으로 ref 좌표계로 변환
        ci_pts_transformed = transform_points(ci_pts, T_ref[ci])
        pcd_ci = o3d.geometry.PointCloud()
        pcd_ci.points = o3d.utility.Vector3dVector(ci_pts_transformed)
        pcd_ci.colors = o3d.utility.Vector3dVector(ci_cols)

        # ---- Step 1: FPFH Global Registration ----
        voxel_global = 0.005  # 5mm voxel for FPFH
        pcd_ref_ds = pcd_ref.voxel_down_sample(voxel_global)
        pcd_ci_ds = pcd_ci.voxel_down_sample(voxel_global)

        radius_normal = voxel_global * 2
        pcd_ref_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pcd_ci_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_global * 5
        fpfh_ref = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_ref_ds,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        fpfh_ci = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_ci_ds,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        # RANSAC으로 대략적 초기 정합
        dist_threshold_global = voxel_global * 1.5
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_ci_ds, pcd_ref_ds,
            fpfh_ci, fpfh_ref,
            mutual_filter=True,
            max_correspondence_distance=dist_threshold_global,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_threshold_global),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )

        T_init = result_ransac.transformation
        R_init = T_init[:3, :3]
        t_init = T_init[:3, 3]
        angle_init = np.degrees(np.arccos(np.clip((np.trace(R_init) - 1) / 2, -1, 1)))
        shift_init = np.linalg.norm(t_init) * 1000

        print(f"  FPFH RANSAC: fitness={result_ransac.fitness:.4f}  "
              f"corr={len(result_ransac.correspondence_set)}  "
              f"rot={angle_init:.2f}°  trans={shift_init:.1f}mm")

        # RANSAC 결과가 너무 크면 (local minimum 의심) identity 사용
        if angle_init > 15 or shift_init > 100:
            print(f"  -> RANSAC 결과 과대, identity 초기값 사용")
            T_init = np.eye(4)

        # ---- Step 2: Multi-scale Colored ICP ----
        # 색상 + 기하 정보 모두 활용 → 평면 영역에서도 정확
        voxel_sizes = [0.005, 0.003, 0.001]  # coarse → fine
        max_iters = [100, 80, 50]

        T_current = T_init.copy()
        for scale_i, (vs, mi) in enumerate(zip(voxel_sizes, max_iters)):
            pcd_ref_s = pcd_ref.voxel_down_sample(vs)
            pcd_ci_s = pcd_ci.voxel_down_sample(vs)

            pcd_ref_s.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 2, max_nn=30))
            pcd_ci_s.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=vs * 2, max_nn=30))

            result_icp = o3d.pipelines.registration.registration_colored_icp(
                pcd_ci_s, pcd_ref_s,
                vs * 3,  # correspondence distance = 3x voxel
                T_current,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=mi,
                ),
            )
            T_current = result_icp.transformation
            print(f"  Scale {scale_i} (voxel={vs*1000:.0f}mm): "
                  f"fitness={result_icp.fitness:.4f}  "
                  f"RMSE={result_icp.inlier_rmse*1000:.3f}mm")

        # 최종 보정량 합성
        T_correction = T_current
        T_refined[ci] = T_correction @ T_ref[ci]

        R_corr = T_correction[:3, :3]
        t_corr = T_correction[:3, 3]
        angle_deg = np.degrees(np.arccos(np.clip((np.trace(R_corr) - 1) / 2, -1, 1)))
        shift_mm = np.linalg.norm(t_corr) * 1000

        print(f"  최종 보정량: rot={angle_deg:.3f}°  trans={shift_mm:.2f}mm")

    return T_refined


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t.reshape(1, 3)


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
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
    sum_cols = np.zeros((n, 3), dtype=np.float64)
    np.add.at(sum_pts, inverse, points)
    np.add.at(sum_cols, inverse, colors)
    return sum_pts / counts[:, None], sum_cols / counts[:, None]


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    n = len(points)
    rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                    f"{rgb_u8[i,0]} {rgb_u8[i,1]} {rgb_u8[i,2]}\n")
    print(f"[SAVE] {path}  ({n:,} points)")


# ──────────────────────────────────────────────────
# single frame fusion
# ──────────────────────────────────────────────────
def fuse_frame(capture_dir: str, frame_idx: int, cam_indices: List[int],
               T_ref: Dict[int, np.ndarray], K_map: Dict[int, np.ndarray],
               D_map: Dict[int, np.ndarray], ds_map: Dict[int, float],
               z_min: float, z_max: float, stride: int, pad: int,
               use_bilateral: bool = True, use_undistort: bool = True,
               use_sor: bool = True, sor_neighbors: int = 20, sor_std: float = 2.0,
               use_inpaint: bool = True,
               ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    all_pts, all_cols = [], []
    fmt = f"{{:0{pad}d}}"
    fid_str = fmt.format(frame_idx)

    for ci in cam_indices:
        if ci not in T_ref:
            continue
        rgb_path = os.path.join(capture_dir, f"cam{ci}", f"rgb_{fid_str}.jpg")
        depth_path = os.path.join(capture_dir, f"cam{ci}", f"depth_{fid_str}.png")
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            print(f"[WARN] cam{ci} frame {frame_idx}: RGB 또는 depth 파일 없음, skip")
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_u16 is None:
            continue

        # [필터 0] Depth inpaint — 소규모 구멍 채우기
        if use_inpaint:
            depth_u16 = depth_inpaint(depth_u16)

        # [필터 1] Bilateral depth filter — 엣지 보존 노이즈 제거
        if use_bilateral:
            depth_u16 = bilateral_depth_filter(depth_u16)

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        # [개선] 렌즈 왜곡 보정 적용
        pts_cam, pixels = depth_to_points(
            depth_u16, K_map[ci], D_map.get(ci), ds_map[ci],
            z_min, z_max, stride, undistort=use_undistort,
        )
        if pts_cam.shape[0] == 0:
            print(f"[WARN] cam{ci} frame {frame_idx}: 유효 depth 점 0개")
            continue

        pts_ref = transform_points(pts_cam, T_ref[ci])
        cols = rgb[pixels[:, 0], pixels[:, 1]]

        # [필터 2] Statistical Outlier Removal — 떠다니는 노이즈 제거
        if use_sor:
            pts_ref, cols = statistical_outlier_removal(
                pts_ref, cols, nb_neighbors=sor_neighbors, std_ratio=sor_std,
            )

        all_pts.append(pts_ref)
        all_cols.append(cols)
        print(f"  cam{ci}: {pts_ref.shape[0]:>8,} points (after filters)")

    return all_pts, all_cols


# ──────────────────────────────────────────────────
# visualization
# ──────────────────────────────────────────────────
def show_open3d(ply_path: str):
    try:
        import open3d as o3d
    except ImportError:
        print("[WARN] open3d 미설치. pip install open3d 후 다시 시도하세요.")
        return
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"[INFO] Open3D viewer: {len(pcd.points):,} points")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="3D Reconstruction",
        width=1280,
        height=720,
    )


def show_matplotlib(points: np.ndarray, colors: np.ndarray, T_ref: Dict[int, np.ndarray],
                    ref_idx: int, title: str, save_path: Optional[str] = None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    max_vis = 50000
    vis_stride = max(1, len(points) // max_vis)
    ax.scatter(
        points[::vis_stride, 0],
        points[::vis_stride, 1],
        points[::vis_stride, 2],
        c=colors[::vis_stride],
        s=0.3, alpha=0.6, linewidths=0,
    )

    cam_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for i, ci in enumerate(sorted(T_ref.keys())):
        c = cam_colors[i % len(cam_colors)]
        pos = np.zeros(3) if ci == ref_idx else T_ref[ci][:3, 3]
        ax.scatter(*pos, c=c, s=150, marker="^", edgecolors="k", zorder=8)
        ax.text(*pos, f"  cam{ci}", fontsize=9, color=c)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    plt.show()


# ──────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="멀티카메라 RGBD → 3D 포인트클라우드 복원")
    parser.add_argument("--capture_dir", required=True, help="capture_rgbd_3cam.py 저장 폴더")
    parser.add_argument("--intrinsics_dir", default="./intrinsics", help="intrinsics 폴더")
    parser.add_argument("--calib_dir", required=True, help="T_C0_Ci.npy 가 있는 캘리브레이션 폴더")
    parser.add_argument("--ref_cam", type=int, default=0, help="기준 카메라 인덱스 (default: 0)")

    parser.add_argument("--frame", type=int, default=None, help="특정 프레임만 복원 (기본: 첫 번째)")
    parser.add_argument("--each_frame", action="store_true", help="모든 프레임을 각각 개별 PLY로 저장 (다른 물체들)")
    parser.add_argument("--all_frames", action="store_true", help="모든 프레임을 하나로 합치기 (같은 물체)")
    parser.add_argument("--per_cam", action="store_true",
                        help="진단 모드: 카메라별 개별 PLY 저장 (정합 문제 진단용)")
    parser.add_argument("--frame_skip", type=int, default=1, help="each_frame/all_frames 시 N번째마다 사용")

    parser.add_argument("--z_min", type=float, default=0.1, help="depth 최소 거리 (m)")
    parser.add_argument("--z_max", type=float, default=0.5, help="depth 최대 거리 (m)")
    parser.add_argument("--stride", type=int, default=1, help="depth subsampling (1=dense, 4=sparse)")
    parser.add_argument("--voxel_mm", type=float, default=0.0, help="voxel downsample 크기 (mm), 0=OFF")

    # 정확도 필터
    parser.add_argument("--no_undistort", action="store_true", help="렌즈 왜곡 보정 끄기")
    parser.add_argument("--no_bilateral", action="store_true", help="bilateral depth filter 끄기")
    parser.add_argument("--no_inpaint", action="store_true", help="depth 구멍 채우기 끄기")
    parser.add_argument("--no_sor", action="store_true", help="statistical outlier removal 끄기")
    parser.add_argument("--sor_neighbors", type=int, default=20, help="SOR k-nearest neighbors 수")
    parser.add_argument("--sor_std", type=float, default=2.0, help="SOR std_ratio (낮을수록 공격적 제거)")

    # 배경 제거
    parser.add_argument("--remove_plane", action="store_true", help="RANSAC 배경 평면(테이블/바닥) 제거")
    parser.add_argument("--plane_dist", type=float, default=0.005, help="평면 distance threshold (m)")

    # ICP extrinsics refinement
    parser.add_argument("--icp", action="store_true", help="ICP로 캘리브레이션 extrinsics 미세 보정")
    parser.add_argument("--icp_dist", type=float, default=0.02, help="ICP max correspondence distance (m)")
    parser.add_argument("--icp_frames", type=int, default=5, help="ICP에 사용할 샘플 프레임 수")

    parser.add_argument("--out", type=str, default=None, help="PLY 출력 경로 (기본: capture_dir 내)")
    parser.add_argument("--open3d", action="store_true", help="Open3D 뷰어로 결과 표시")
    parser.add_argument("--no_plot", action="store_true", help="matplotlib 시각화 끄기")

    args = parser.parse_args()

    # --- discover ---
    cam_indices = discover_cameras(args.capture_dir)
    frame_ids = discover_frames(args.capture_dir, cam_indices)
    print(f"[INFO] 카메라: {cam_indices}  ({len(cam_indices)}대)")
    print(f"[INFO] 프레임: {len(frame_ids)}장  (range: {frame_ids[0]} ~ {frame_ids[-1]})")

    # --- load calibration ---
    K_map, D_map, ds_map = {}, {}, {}
    for ci in cam_indices:
        K, D, ds = load_intrinsics(args.intrinsics_dir, ci)
        K_map[ci] = K
        D_map[ci] = D
        ds_map[ci] = ds

    T_ref = load_extrinsics(args.calib_dir, args.ref_cam, cam_indices)

    # --- detect file naming ---
    pad = _detect_zero_padding(args.capture_dir, cam_indices[0])
    print(f"[INFO] 파일명 zero-padding: {pad}자리")

    # --- ICP extrinsics refinement ---
    if args.icp:
        T_ref = refine_extrinsics_icp(
            args.capture_dir, frame_ids, cam_indices, T_ref,
            K_map, D_map, ds_map, args.ref_cam,
            args.z_min, args.z_max, pad,
            max_correspondence_dist=args.icp_dist,
            n_sample_frames=args.icp_frames,
        )

    # --- filter settings ---
    use_undistort = not args.no_undistort
    use_bilateral = not args.no_bilateral
    use_inpaint = not args.no_inpaint
    use_sor = not args.no_sor
    filters_on = []
    if use_inpaint:
        filters_on.append("inpaint")
    if use_undistort:
        filters_on.append("undistort")
    if use_bilateral:
        filters_on.append("bilateral")
    if use_sor:
        filters_on.append(f"SOR(k={args.sor_neighbors},std={args.sor_std})")
    if args.remove_plane:
        filters_on.append(f"plane_removal(d={args.plane_dist}m)")
    if args.icp:
        filters_on.append(f"ICP(d={args.icp_dist*1000:.0f}mm, frames={args.icp_frames})")
    print(f"[INFO] 필터: {', '.join(filters_on) if filters_on else 'OFF'}")

    # --- output directory ---
    out_dir = os.path.join(args.capture_dir, "ply")
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    # --per_cam : 카메라별 개별 PLY 저장 (정합 진단용)
    # ============================================================
    if args.per_cam:
        fid = args.frame if args.frame is not None else frame_ids[0]
        print(f"\n[DIAG] 카메라별 개별 PLY 저장 (frame {fid})")
        diag_dir = os.path.join(args.capture_dir, "ply_per_cam")
        os.makedirs(diag_dir, exist_ok=True)

        fmt = f"{{:0{pad}d}}"
        fid_str = fmt.format(fid)

        for ci in cam_indices:
            rgb_path = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid_str}.jpg")
            depth_path = os.path.join(args.capture_dir, f"cam{ci}", f"depth_{fid_str}.png")
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                print(f"  cam{ci}: 파일 없음, skip")
                continue
            rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None or depth_u16 is None:
                continue

            if use_inpaint:
                depth_u16 = depth_inpaint(depth_u16)
            if use_bilateral:
                depth_u16 = bilateral_depth_filter(depth_u16)

            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            pts_cam, pixels = depth_to_points(
                depth_u16, K_map[ci], D_map.get(ci), ds_map[ci],
                args.z_min, args.z_max, args.stride, undistort=use_undistort,
            )
            if pts_cam.shape[0] == 0:
                print(f"  cam{ci}: 유효 점 0개")
                continue
            cols = rgb[pixels[:, 0], pixels[:, 1]]

            # 카메라 자체 좌표계 PLY (변환 없음)
            path_local = os.path.join(diag_dir, f"frame{fid:0{pad}d}_cam{ci}_local.ply")
            save_ply(path_local, pts_cam, cols)

            # ref 좌표계로 변환된 PLY
            pts_ref = transform_points(pts_cam, T_ref[ci])
            path_ref = os.path.join(diag_dir, f"frame{fid:0{pad}d}_cam{ci}_ref.ply")
            save_ply(path_ref, pts_ref, cols)

            # 정보 출력
            bb_min = pts_cam.min(axis=0) * 1000
            bb_max = pts_cam.max(axis=0) * 1000
            bb_size = bb_max - bb_min
            print(f"  cam{ci}: {pts_cam.shape[0]:,} pts  "
                  f"bbox: {bb_size[0]:.0f}x{bb_size[1]:.0f}x{bb_size[2]:.0f} mm")
            if ci != args.ref_cam:
                T = T_ref[ci]
                r = T[:3, :3]
                t = T[:3, 3]
                angle = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
                dist = np.linalg.norm(t) * 1000
                print(f"         T_C{args.ref_cam}_C{ci}: rot={angle:.2f}°  trans={dist:.1f}mm")

        print(f"\n[DIAG] 결과:")
        print(f"  {diag_dir}/")
        print(f"  *_local.ply = 각 카메라 자체 좌표계 (개별 형태 확인)")
        print(f"  *_ref.ply   = ref 좌표계로 변환 (정합 확인)")
        print(f"\n  확인 방법:")
        print(f"  1) _local.ply 가 각각 정상이면 → extrinsics(변환행렬) 문제")
        print(f"  2) _local.ply 자체가 찌그러져 있으면 → intrinsics(내부 파라미터) 문제")
        print(f"  3) _ref.ply 3개를 동시에 열어서 겹침 확인 → 정합 오차 확인")
        return

    # ============================================================
    # --each_frame : 프레임별 개별 PLY 저장 (각각 다른 물체)
    # ============================================================
    if args.each_frame:
        target_frames = frame_ids[::args.frame_skip]
        print(f"\n[BATCH] 프레임별 개별 PLY 저장: {len(target_frames)}장")
        print(f"[BATCH] 출력 폴더: {os.path.abspath(out_dir)}\n")

        success, fail = 0, 0
        for i, fid in enumerate(target_frames):
            fmt_str = f"{{:0{pad}d}}".format(fid)
            print(f"[{i+1}/{len(target_frames)}] frame {fid}:")
            f_pts, f_cols = fuse_frame(
                args.capture_dir, fid, cam_indices, T_ref, K_map, D_map, ds_map,
                args.z_min, args.z_max, args.stride, pad,
                use_bilateral=use_bilateral, use_undistort=use_undistort,
                use_sor=use_sor, sor_neighbors=args.sor_neighbors, sor_std=args.sor_std,
                use_inpaint=use_inpaint,
            )
            if not f_pts:
                print(f"  -> SKIP (유효 점 없음)\n")
                fail += 1
                continue

            P = np.concatenate(f_pts, axis=0)
            C = np.concatenate(f_cols, axis=0)

            # 배경 평면 제거
            if args.remove_plane:
                P, C = remove_background_plane(P, C, distance_threshold=args.plane_dist)

            if args.voxel_mm > 0:
                n_before = len(P)
                P, C = voxel_downsample(P, C, args.voxel_mm / 1000.0)
                print(f"  voxel {args.voxel_mm:.1f}mm: {n_before:,} -> {len(P):,}")

            ply_name = f"frame_{fid:0{pad}d}.ply"
            ply_path = os.path.join(out_dir, ply_name)
            save_ply(ply_path, P, C)
            success += 1
            print()

        print(f"[BATCH 완료] 성공: {success}  실패: {fail}  총: {len(target_frames)}")
        print(f"[BATCH 결과] {os.path.abspath(out_dir)}/")
        return

    # ============================================================
    # --all_frames : 전체 프레임을 하나로 합치기 (같은 물체)
    # ============================================================
    if args.all_frames:
        target_frames = frame_ids[::args.frame_skip]
        print(f"\n[FUSE] 전체 프레임 합치기: {len(target_frames)}장 (skip={args.frame_skip})")
    else:
        fid = args.frame if args.frame is not None else frame_ids[0]
        if fid not in frame_ids:
            raise RuntimeError(f"frame {fid} 없음. 가능한 프레임: {frame_ids}")
        target_frames = [fid]
        print(f"\n[FUSE] 단일 프레임: {fid}")

    # --- fuse ---
    all_pts, all_cols = [], []
    for fid in target_frames:
        print(f"[FUSE] frame {fid}:")
        f_pts, f_cols = fuse_frame(
            args.capture_dir, fid, cam_indices, T_ref, K_map, D_map, ds_map,
            args.z_min, args.z_max, args.stride, pad,
            use_bilateral=use_bilateral, use_undistort=use_undistort,
            use_sor=use_sor, sor_neighbors=args.sor_neighbors, sor_std=args.sor_std,
        )
        all_pts.extend(f_pts)
        all_cols.extend(f_cols)

    if not all_pts:
        print("[ERROR] 유효한 3D 점이 없습니다. depth 파일과 z_min/z_max를 확인하세요.")
        return

    P = np.concatenate(all_pts, axis=0)
    C = np.concatenate(all_cols, axis=0)
    print(f"\n[INFO] 총 포인트: {P.shape[0]:,}")

    # --- 배경 평면 제거 ---
    if args.remove_plane:
        P, C = remove_background_plane(P, C, distance_threshold=args.plane_dist)
        print(f"[INFO] 평면 제거 후: {P.shape[0]:,}")

    # --- voxel downsample ---
    if args.voxel_mm > 0:
        n_before = len(P)
        P, C = voxel_downsample(P, C, args.voxel_mm / 1000.0)
        print(f"[INFO] Voxel downsample ({args.voxel_mm:.1f}mm): {n_before:,} -> {len(P):,}")

    # --- save PLY ---
    if args.out:
        ply_path = args.out
    else:
        if args.all_frames:
            ply_path = os.path.join(out_dir, "reconstruction_all.ply")
        else:
            ply_path = os.path.join(out_dir, f"reconstruction_frame{target_frames[0]:06d}.ply")
    save_ply(ply_path, P, C)

    # --- visualize ---
    if args.open3d:
        show_open3d(ply_path)

    if not args.no_plot:
        title = f"3D Reconstruction – {len(cam_indices)} cams, {len(target_frames)} frames, {len(P):,} pts"
        png_path = ply_path.replace(".ply", ".png")
        show_matplotlib(P, C, T_ref, args.ref_cam, title, save_path=png_path)


if __name__ == "__main__":
    main()
