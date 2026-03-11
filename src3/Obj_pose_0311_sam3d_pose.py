#!/usr/bin/env python3
"""
Obj_pose_0311_reconstruct.py
============================
멀티뷰 RGB-D 카메라 3대 → 3D 복원 + 6-DOF 포즈 추정 → OpenCV/cam0 좌표계 통합 출력

시뮬레이션 프로그램 연동을 위해 모든 출력물을 **cam0 (OpenCV: X→right, Y→down, Z→forward)**
좌표계로 통합합니다.

출력물:
  ├─ pointcloud_cam0.ply      # 관측 점군 (cam0 좌표계, 미터 단위)
  ├─ mesh_cam0.ply            # 복원 메시 (cam0 좌표계)
  ├─ mesh_cam0.glb            # 복원 메시 GLB (시뮬레이션용, cam0 좌표계)
  ├─ pose_cam0.json           # 6-DOF 포즈 (R, t, euler, quat, OBB)
  ├─ pose_cam0.npz            # NumPy 포즈 (R, t, scale 등 바이너리)
  ├─ pose_visualization.png   # 3D 포즈 시각화
  ├─ overlay_cam0.png         # cam0 이미지 위 재투영 오버레이
  └─ pointcloud_shape.png     # 점군 + OBB 시각화

사용법:
  python Obj_pose_0311_reconstruct.py --frame 3
  python Obj_pose_0311_reconstruct.py --frame 3 --ref_length_mm 165
  python Obj_pose_0311_reconstruct.py --frame 3 --no_tsdf  # 점군만 (메시 없이)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_CAP_DIR = _THIS_DIR / "data/object_capture"
_DEFAULT_CAL_DIR = _THIS_DIR / "data/cube_session_01/calib_out_cube"
_DEFAULT_INT_DIR = _THIS_DIR / "data/_intrinsics"
_DEFAULT_OUT_DIR = _THIS_DIR / "Obj_pose_0311_reconstruct_output"


# ======================================================================
#  1. 캘리브레이션 로드
# ======================================================================

def load_intrinsics(intrin_dir: Path, cid: int):
    d = np.load(str(intrin_dir / f"cam{cid}.npz"), allow_pickle=True)
    K = d["color_K"].astype(np.float64)
    D = d["color_D"].astype(np.float64) if "color_D" in d else np.zeros(5)
    ds = float(d.get("depth_scale_m_per_unit", 0.001))
    return K, D, ds


def load_extrinsics(calib_dir: Path, cams: List[int]) -> Dict[int, np.ndarray]:
    T = {}
    for ci in cams:
        if ci == 0:
            T[ci] = np.eye(4, dtype=np.float64)
        else:
            T[ci] = np.load(str(calib_dir / f"T_C0_C{ci}.npy")).astype(np.float64)
    return T


def discover_cameras(cap_dir: Path) -> List[int]:
    ids = []
    for d in sorted(glob.glob(str(cap_dir / "cam*"))):
        try:
            ci = int(Path(d).name.replace("cam", ""))
        except ValueError:
            continue
        if glob.glob(str(Path(d) / "rgb_*.jpg")):
            ids.append(ci)
    if not ids:
        raise RuntimeError(f"cam* 폴더를 찾지 못함: {cap_dir}")
    return ids


def frame_pad(cap_dir: Path, ci: int) -> int:
    files = glob.glob(str(cap_dir / f"cam{ci}" / "rgb_*.jpg"))
    return len(Path(files[0]).stem.replace("rgb_", "")) if files else 6


# ======================================================================
#  2. Depth → 3D 점군
# ======================================================================

def depth_to_pts(depth_u16, mask, K, D, ds, z_min, z_max):
    """depth + mask → (points Nx3, uvs Nx2) in camera coordinate."""
    h, w = depth_u16.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    vg, ug = np.mgrid[0:h, 0:w]
    z = depth_u16.astype(np.float64) * ds
    ok = mask & (z > z_min) & (z < z_max)
    if not ok.any():
        return np.empty((0, 3)), np.empty((0, 2))
    z, u, v = z[ok], ug[ok].astype(np.float64), vg[ok].astype(np.float64)
    und = cv2.undistortPoints(
        np.column_stack([u, v]).reshape(-1, 1, 2).astype(np.float64), K, D
    ).reshape(-1, 2)
    return np.column_stack([und[:, 0]*z, und[:, 1]*z, z]), np.column_stack([u, v])


# ======================================================================
#  3. HSV 앵커 세그멘테이션 + 멀티뷰 점군 융합
# ======================================================================

def segment_and_fuse(args, cams, K_m, D_m, ds_m, T_m, pad):
    """
    3단계 세그멘테이션 + 멀티카메라 점군 융합 (cam0 좌표계).

    Stage 1: HSV → 노란 손잡이 3D 앵커 중심
    Stage 2: 앵커 주변 구 크롭
    Stage 3: PCA 타원체 크롭 + SOR + DBSCAN
    """
    h_lo, h_hi = args.hsv_h_range
    s_lo, v_lo = args.hsv_s_min, args.hsv_v_min
    fid = f"{args.frame:0{pad}d}"

    # --- Stage 1: HSV 앵커 ---
    anchor_pts_3d = []
    for ci in cams:
        rp = str(Path(args.capture_dir) / f"cam{ci}" / f"rgb_{fid}.jpg")
        dp = str(Path(args.capture_dir) / f"cam{ci}" / f"depth_{fid}.png")
        if not (os.path.exists(rp) and os.path.exists(dp)):
            continue
        bgr = cv2.imread(rp)
        dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            continue

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ym = ((hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
              (hsv[:, :, 1] >= s_lo) & (hsv[:, :, 2] >= v_lo))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ym = cv2.morphologyEx(ym.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        if ym.sum() < args.min_area:
            continue
        pts_ci, _ = depth_to_pts(dep, ym, K_m[ci], D_m[ci], ds_m[ci],
                                  args.z_min, args.z_max)
        if len(pts_ci) == 0:
            continue
        pts_cam0 = pts_ci @ T_m[ci][:3, :3].T + T_m[ci][:3, 3]
        anchor_pts_3d.append(pts_cam0)
        print(f"    cam{ci}: HSV 앵커 {len(pts_ci):,} pts")

    if not anchor_pts_3d:
        raise RuntimeError("노란 손잡이 앵커 검출 실패")

    anchor_center = np.concatenate(anchor_pts_3d).mean(0)
    print(f"    앵커 중심 (cam0): ({anchor_center[0]*1000:.1f}, "
          f"{anchor_center[1]*1000:.1f}, {anchor_center[2]*1000:.1f}) mm")

    # --- Stage 2: 구 크롭 ---
    sphere_r = 0.12
    all_pts, all_rgb = [], []
    for ci in cams:
        rp = str(Path(args.capture_dir) / f"cam{ci}" / f"rgb_{fid}.jpg")
        dp = str(Path(args.capture_dir) / f"cam{ci}" / f"depth_{fid}.png")
        if not (os.path.exists(rp) and os.path.exists(dp)):
            continue
        bgr = cv2.imread(rp)
        dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            continue
        full_mask = np.ones(dep.shape[:2], dtype=bool)
        pts_ci, uvs = depth_to_pts(dep, full_mask, K_m[ci], D_m[ci], ds_m[ci],
                                    args.z_min, args.z_max)
        if len(pts_ci) == 0:
            continue
        pts_cam0 = pts_ci @ T_m[ci][:3, :3].T + T_m[ci][:3, 3]
        dist = np.linalg.norm(pts_cam0 - anchor_center, axis=1)
        inside = dist < sphere_r
        pts_cam0 = pts_cam0[inside]
        uvs = uvs[inside]
        if len(pts_cam0) == 0:
            continue
        ui, vi = uvs[:, 0].astype(int), uvs[:, 1].astype(int)
        rgb_ci = bgr[vi, ui][:, ::-1] / 255.0
        all_pts.append(pts_cam0)
        all_rgb.append(rgb_ci)

    if not all_pts:
        raise RuntimeError("세그멘테이션 실패")
    pts = np.concatenate(all_pts)
    rgb = np.concatenate(all_rgb)
    print(f"    구 크롭 (r={sphere_r*1000:.0f}mm): {len(pts):,} pts")

    # --- Stage 3: PCA 타원체 크롭 ---
    centered = pts - anchor_center
    cov = centered.T @ centered / len(centered)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    proj = centered @ evecs
    a_long, a_short = 0.100, 0.018
    ellip = (proj[:, 0]/a_long)**2 + (proj[:, 1]/a_short)**2 + (proj[:, 2]/a_short)**2
    in_e = ellip < 1.0
    if in_e.sum() >= 50:
        pts, rgb = pts[in_e], rgb[in_e]
    print(f"    타원체 크롭: {in_e.sum():,} pts")

    # SOR
    pts, rgb = _sor(pts, rgb)

    # DBSCAN
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=0.005, min_samples=10).fit(pts).labels_
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) > 0:
        best = unique[np.argmax(counts)]
        mask = labels == best
        pts, rgb = pts[mask], rgb[mask]
        print(f"    DBSCAN: {len(unique)} 클러스터, 최대={mask.sum():,} pts 채택")

    return pts, rgb, anchor_center


def _sor(pts, rgb=None, std_ratio=1.5):
    n = len(pts)
    if n < 10:
        return pts, rgb
    mask = np.ones(n, bool)
    for ax in range(3):
        v = pts[:, ax]
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        mask &= (v >= q1 - 1.5*iqr) & (v <= q3 + 1.5*iqr)
    c = pts[mask].mean(0)
    d = np.linalg.norm(pts - c, axis=1)
    mu, sg = d[mask].mean(), d[mask].std()
    mask &= d < mu + std_ratio * sg
    print(f"    SOR: {n:,} -> {mask.sum():,}")
    return pts[mask], (rgb[mask] if rgb is not None else None)


# ======================================================================
#  4. TSDF 3D 복원 → 메시 (cam0 좌표계)
# ======================================================================

def reconstruct_tsdf(args, cams, K_m, D_m, ds_m, T_m, pad,
                     seg_pts: np.ndarray,
                     voxel_length: float = 0.0005,
                     sdf_trunc: float = 0.003):
    """
    멀티카메라 TSDF Volume Integration → 메시 + 점군 추출.
    모든 출력은 cam0 좌표계 (미터 단위).
    """
    import open3d as o3d

    fid = f"{args.frame:0{pad}d}"
    h_lo, h_hi = args.hsv_h_range
    s_lo, v_lo = args.hsv_s_min, args.hsv_v_min

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    n_integrated = 0
    for ci in cams:
        rp = str(Path(args.capture_dir) / f"cam{ci}" / f"rgb_{fid}.jpg")
        dp = str(Path(args.capture_dir) / f"cam{ci}" / f"depth_{fid}.png")
        if not (os.path.exists(rp) and os.path.exists(dp)):
            continue
        bgr = cv2.imread(rp)
        dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            continue

        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_img, w_img = dep.shape[:2]

        # HSV 앵커 depth 범위
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        ym = ((hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
              (hsv[:, :, 1] >= s_lo) & (hsv[:, :, 2] >= v_lo))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ym = cv2.morphologyEx(ym.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n_lab, lab, stats, centroids = cv2.connectedComponentsWithStats(ym)
        best_lb, best_a = -1, 0
        for lb in range(1, n_lab):
            a = stats[lb, cv2.CC_STAT_AREA]
            if a >= args.min_area and a > best_a:
                best_lb, best_a = lb, a
        if best_lb < 0:
            continue

        cx_2d = int(centroids[best_lb][0])
        cy_2d = int(centroids[best_lb][1])
        roi_half = int(max(stats[best_lb, cv2.CC_STAT_WIDTH],
                           stats[best_lb, cv2.CC_STAT_HEIGHT]) * 2.5)

        yellow_mask = (lab == best_lb)
        dep_vals = dep[yellow_mask].astype(np.float64) * ds_m[ci]
        dep_vals = dep_vals[dep_vals > 0.05]
        if len(dep_vals) == 0:
            continue
        anchor_d = float(np.median(dep_vals))

        dep_m = dep.astype(np.float64) * ds_m[ci]
        depth_ok = (dep_m > anchor_d - 0.020) & (dep_m < anchor_d + 0.020)
        spatial_ok = np.zeros((h_img, w_img), dtype=bool)
        spatial_ok[max(0, cy_2d-roi_half):min(h_img, cy_2d+roi_half),
                   max(0, cx_2d-roi_half):min(w_img, cx_2d+roi_half)] = True
        valid = spatial_ok & depth_ok & (dep > 0)

        dep_masked = dep.copy()
        dep_masked[~valid] = 0

        # cam_i → cam0 (T_C0_Ci), Open3D expects world→camera = inv(T_C0_Ci)
        T_cam0_to_ci = np.linalg.inv(T_m[ci])

        color_o3d = o3d.geometry.Image(rgb_img.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(dep_masked)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0 / ds_m[ci],
            depth_trunc=anchor_d + 0.030,
            convert_rgb_to_intensity=False,
        )
        K = K_m[ci]
        h_rgb, w_rgb = rgb_img.shape[:2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w_rgb, h_rgb, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        volume.integrate(rgbd, intrinsic, T_cam0_to_ci)
        n_integrated += 1

    if n_integrated == 0:
        raise RuntimeError("TSDF 통합 프레임 없음")
    print(f"    TSDF 통합: {n_integrated} 프레임 (voxel={voxel_length*1000:.1f}mm)")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_point_cloud()

    # PCA OBB 기반 크롭
    seg_center = seg_pts.mean(0)
    seg_cen = seg_pts - seg_center
    cov = seg_cen.T @ seg_cen / len(seg_cen)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    evec = evec[:, order]
    seg_proj = seg_cen @ evec
    seg_ext = seg_proj.max(0) - seg_proj.min(0)
    margin = 1.3

    # 메시 크롭
    verts = np.asarray(mesh.vertices)
    vproj = (verts - seg_center) @ evec
    in_obb = np.all(np.abs(vproj) < seg_ext * margin / 2, axis=1)
    if in_obb.sum() > 100:
        mesh = mesh.select_by_index(np.where(in_obb)[0])
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # 점군 크롭
    ppts = np.asarray(pcd.points)
    pproj = (ppts - seg_center) @ evec
    in_pcd = np.all(np.abs(pproj) < seg_ext * margin / 2, axis=1)
    if in_pcd.sum() > 100:
        pcd = pcd.select_by_index(np.where(in_pcd)[0])

    print(f"    메시: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces")
    print(f"    점군: {len(pcd.points):,} pts")
    print(f"    OBB: {seg_ext[0]*margin*1000:.1f} x {seg_ext[1]*margin*1000:.1f} x "
          f"{seg_ext[2]*margin*1000:.1f} mm")

    return mesh, pcd, {"n_integrated": n_integrated,
                       "voxel_mm": voxel_length*1000,
                       "sdf_trunc_mm": sdf_trunc*1000}


# ======================================================================
#  5. 포즈 추정 (PCA + 색상 기반 방향 판별)
# ======================================================================

def estimate_pose_cam0(pts, rgb=None, blade_dir="auto"):
    """PCA 기반 6-DOF 포즈 추정 (cam0 OpenCV 좌표계).

    칼 canonical frame:
      X = 날 방향 (blade), Y = 너비, Z = 법선 (테이블 위로)
    """
    centroid = pts.mean(0)
    p = pts - centroid
    cov = p.T @ p / len(p)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    if np.linalg.det(evec) < 0:
        evec[:, 2] *= -1

    length_ax = evec[:, 0].copy()
    normal_ax = evec[:, 2].copy()
    if normal_ax[1] > 0:
        normal_ax = -normal_ax

    # 날 방향 판별 (3가지 지표 다수결)
    proj = p @ length_ax
    if blade_dir == "auto" and rgb is not None:
        half_pos = proj > np.median(proj)
        half_neg = ~half_pos

        # 지표 1: HSV 채도 — 손잡이(노란색)=높은 채도, 칼날(금속)=낮은 채도
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        hsv_s = cv2.cvtColor(rgb_u8.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0, :, 1].astype(float)
        sat_pos = hsv_s[half_pos].mean()
        sat_neg = hsv_s[half_neg].mean()

        # 지표 2: 밝기 분산 — 금속 칼날=반사로 분산 높음, 손잡이=균일
        gray = rgb.mean(axis=1)
        var_pos = gray[half_pos].var()
        var_neg = gray[half_neg].var()

        # 지표 3: 폭 — 칼날 끝이 좁고, 손잡이 쪽이 넓음
        width_proj = p @ evec[:, 1]
        spread_pos = np.percentile(width_proj[half_pos], 95) - np.percentile(width_proj[half_pos], 5)
        spread_neg = np.percentile(width_proj[half_neg], 95) - np.percentile(width_proj[half_neg], 5)

        # 투표: 칼날(blade) 쪽 판별
        # 채도 낮은 쪽 = blade, 분산 높은 쪽 = blade, 폭 좁은 쪽 = blade
        votes_pos = 0
        votes_pos += (1 if sat_pos < sat_neg else 0)    # 채도 낮으면 blade
        votes_pos += (1 if var_pos > var_neg else 0)     # 분산 높으면 blade
        votes_pos += (1 if spread_pos < spread_neg else 0)  # 폭 좁으면 blade

        blade_is_pos = votes_pos >= 2
        blade_dir_used = (
            f"auto_vote({votes_pos}/3 pos) "
            f"sat={sat_pos:.0f}/{sat_neg:.0f} "
            f"var={var_pos:.4f}/{var_neg:.4f} "
            f"width={spread_pos*1000:.1f}/{spread_neg*1000:.1f}mm"
        )
    elif blade_dir == "neg":
        blade_is_pos = False
        blade_dir_used = "manual:neg"
    else:
        blade_is_pos = True
        blade_dir_used = "manual:pos" if blade_dir == "pos" else "default:pos"

    if not blade_is_pos:
        length_ax = -length_ax

    width_ax = np.cross(normal_ax, length_ax)
    width_ax /= np.linalg.norm(width_ax)

    R = np.column_stack([length_ax, width_ax, normal_ax])

    proj_R = (pts - centroid) @ R
    obb = proj_R.max(0) - proj_R.min(0)

    return {
        "R": R,
        "centroid_m": centroid,
        "centroid_mm": centroid * 1000,
        "blade_axis": length_ax,
        "width_axis": width_ax,
        "normal_axis": normal_ax,
        "obb_mm": obb * 1000,
        "pca_eigenvalues": ev,
        "length_width_ratio": float(ev[0] / (ev[1] + 1e-12)),
        "normal_verticality": float(abs(normal_ax[1])),
        "blade_dir_used": blade_dir_used,
    }


# ======================================================================
#  6. 회전 변환
# ======================================================================

def R_to_euler(R):
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


def R_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5/np.sqrt(tr+1)
        w, x, y, z = 0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
        w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1, 1] > R[2, 2]:
        s = 2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
        w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
        w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


# ======================================================================
#  7. 시각화
# ======================================================================

def visualize_pointcloud(pts, rgb, pose, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 6))
    pts_mm = pts * 1000
    R = pose["R"]
    c_mm = pose["centroid_mm"]
    obb = pose["obb_mm"]

    for idx, (elev, azim, sub) in enumerate([
        (90, -90, "Top (XZ)"), (0, -90, "Front (XY)"), (25, -55, "Perspective")
    ]):
        ax = fig.add_subplot(1, 3, idx+1, projection="3d")
        ax.set_title(sub, fontsize=10)
        if rgb is not None:
            ax.scatter(pts_mm[:, 0], pts_mm[:, 1], pts_mm[:, 2],
                       c=np.clip(rgb, 0, 1), s=0.3, alpha=0.6)
        else:
            ax.scatter(pts_mm[:, 0], pts_mm[:, 1], pts_mm[:, 2],
                       c="steelblue", s=0.3, alpha=0.6)

        # OBB
        h = obb / 2
        corners_l = np.array([
            [-h[0],-h[1],-h[2]], [h[0],-h[1],-h[2]], [h[0],h[1],-h[2]], [-h[0],h[1],-h[2]],
            [-h[0],-h[1],h[2]], [h[0],-h[1],h[2]], [h[0],h[1],h[2]], [-h[0],h[1],h[2]]])
        corners = (R @ corners_l.T).T + c_mm
        for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            ax.plot3D(*zip(corners[i], corners[j]), "r-", lw=0.8, alpha=0.7)

        for a, col in enumerate(["#e74c3c", "#27ae60", "#2980b9"]):
            v = R[:, a] * 30
            ax.quiver(c_mm[0], c_mm[1], c_mm[2], v[0], v[1], v[2],
                      color=col, linewidth=2, arrow_length_ratio=0.15)

        ax.set_xlabel("X mm", fontsize=7); ax.set_ylabel("Y mm", fontsize=7)
        ax.set_zlabel("Z mm", fontsize=7); ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)
        mid = pts_mm.mean(0)
        rng = max((pts_mm.max(0) - pts_mm.min(0)).max() / 2 * 1.3, 40)
        ax.set_xlim(mid[0]-rng, mid[0]+rng)
        ax.set_ylim(mid[1]-rng, mid[1]+rng)
        ax.set_zlim(mid[2]-rng, mid[2]+rng)

    eu = R_to_euler(R)
    fig.suptitle(title, fontsize=11, y=0.98)
    fig.text(0.5, 0.01,
             f"OBB: {obb[0]:.1f}x{obb[1]:.1f}x{obb[2]:.1f} mm  |  "
             f"Pos: ({c_mm[0]:.1f}, {c_mm[1]:.1f}, {c_mm[2]:.1f}) mm  |  "
             f"Euler: ({eu[0]:.1f}, {eu[1]:.1f}, {eu[2]:.1f}) deg",
             ha="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ecf0f1", ec="#bdc3c7"))
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_overlay(pts, R, K, D, rgb_path, out_path):
    bgr = cv2.imread(rgb_path)
    if bgr is None:
        return
    h, w = bgr.shape[:2]
    step = max(1, len(pts) // 3000)
    uv, _ = cv2.projectPoints(
        pts[::step].reshape(-1, 1, 3).astype(np.float64),
        np.zeros(3), np.zeros(3), K, D)
    uv = uv.reshape(-1, 2)
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    for u, v in uv[valid]:
        cv2.circle(bgr, (int(u), int(v)), 2, (0, 50, 255), -1)

    center = pts.mean(0)
    for a, color in enumerate([(0, 0, 200), (0, 200, 0), (200, 0, 0)]):
        end = center + R[:, a] * 0.030
        uv_s, _ = cv2.projectPoints(center.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        uv_e, _ = cv2.projectPoints(end.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        s = tuple(uv_s.reshape(2).astype(int))
        e = tuple(uv_e.reshape(2).astype(int))
        if 0 <= s[0] < w and 0 <= s[1] < h and 0 <= e[0] < w and 0 <= e[1] < h:
            cv2.arrowedLine(bgr, s, e, color, 2, tipLength=0.2)

    cv2.putText(bgr, "Red=Reconstructed  Axes:R/G/B=X/Y/Z",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(out_path, bgr)


# ======================================================================
#  8. 결과 저장 (시뮬레이션용 통합 출력)
# ======================================================================

def save_all(pts, rgb, pose, mesh, tsdf_info, out_dir, tag, elapsed,
             K_cam0, D_cam0, rgb_path_cam0):
    """모든 결과를 cam0 OpenCV 좌표계로 저장."""
    import open3d as o3d

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    R = pose["R"]
    t = pose["centroid_m"]
    eu = R_to_euler(R).tolist()
    q = R_to_quat(R).tolist()
    pos_mm = pose["centroid_mm"].tolist()
    obb = pose["obb_mm"].tolist()

    # ── JSON 포즈 ──
    result = {
        "coordinate_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
        "units": "meters (position), degrees (euler), unitless (quaternion)",
        "position_m": t.tolist(),
        "position_mm": pos_mm,
        "rotation_matrix": R.tolist(),
        "euler_xyz_deg": eu,
        "quaternion_wxyz": q,
        "obb_mm": obb,
        "blade_axis_cam0": pose["blade_axis"].tolist(),
        "normal_axis_cam0": pose["normal_axis"].tolist(),
        "width_axis_cam0": pose["width_axis"].tolist(),
        "pca_length_width_ratio": pose["length_width_ratio"],
        "normal_verticality_pct": pose["normal_verticality"] * 100,
        "blade_dir_used": pose["blade_dir_used"],
        "n_points": len(pts),
        "elapsed_sec": round(elapsed, 2),
    }
    if tsdf_info:
        result["reconstruction"] = tsdf_info

    json_path = out / f"pose_cam0_{tag}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {json_path}")

    # ── NumPy 포즈 (바이너리) ──
    npz_path = out / f"pose_cam0_{tag}.npz"
    np.savez(str(npz_path),
             rotation_matrix=R,
             translation=t,
             euler_xyz_deg=np.array(eu),
             quaternion_wxyz=np.array(q),
             obb_mm=np.array(obb),
             blade_axis=pose["blade_axis"],
             normal_axis=pose["normal_axis"])
    print(f"  [저장] {npz_path}")

    # ── 점군 PLY ──
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))
    ply_path = out / f"pointcloud_cam0_{tag}.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"  [저장] {ply_path}")

    # ── 메시 PLY + GLB ──
    if mesh is not None and len(mesh.vertices) > 0:
        mesh_ply = out / f"mesh_cam0_{tag}.ply"
        o3d.io.write_triangle_mesh(str(mesh_ply), mesh)
        print(f"  [저장] {mesh_ply}")

        try:
            import trimesh
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            vc = None
            if mesh.has_vertex_colors():
                vc_arr = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
                vc = np.column_stack([vc_arr, np.full(len(vc_arr), 255, np.uint8)])
            # GLB는 OpenGL 좌표계이므로 Y,Z 반전
            verts_gl = verts.copy()
            verts_gl[:, 1] *= -1.0
            verts_gl[:, 2] *= -1.0
            tm = trimesh.Trimesh(vertices=verts_gl, faces=faces, vertex_colors=vc)
            glb_path = out / f"mesh_cam0_{tag}.glb"
            tm.export(str(glb_path), file_type="glb")
            print(f"  [저장] {glb_path} (OpenGL 좌표계 변환 포함)")

            # cam0 좌표계 그대로의 OBJ도 저장 (시뮬레이션용)
            tm_cv = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vc)
            obj_path = out / f"mesh_cam0_{tag}.obj"
            tm_cv.export(str(obj_path), file_type="obj")
            print(f"  [저장] {obj_path} (cam0/OpenCV 좌표계 그대로)")
        except Exception as e:
            print(f"  [WARN] GLB/OBJ 저장 실패: {e}")

    # ── 시각화 ──
    try:
        visualize_pointcloud(pts, rgb, pose,
                             str(out / f"pointcloud_shape_{tag}.png"),
                             f"3D Reconstruction + Pose (cam0) - {tag}")
        print(f"  [저장] pointcloud_shape_{tag}.png")
    except Exception as e:
        print(f"  [WARN] 점군 시각화 실패: {e}")

    try:
        save_overlay(pts, R, K_cam0, D_cam0, rgb_path_cam0,
                     str(out / f"overlay_cam0_{tag}.png"))
        print(f"  [저장] overlay_cam0_{tag}.png")
    except Exception as e:
        print(f"  [WARN] 오버레이 시각화 실패: {e}")

    # ── 콘솔 출력 ──
    print(f"\n{'='*60}")
    print(f"  RESULT - cam0 (OpenCV: X-right, Y-down, Z-forward)")
    print(f"{'='*60}")
    print(f"  Position   (mm): ({pos_mm[0]:+.1f}, {pos_mm[1]:+.1f}, {pos_mm[2]:+.1f})")
    print(f"  Euler XYZ (deg): ({eu[0]:+.1f}, {eu[1]:+.1f}, {eu[2]:+.1f})")
    print(f"  Quat wxyz      : ({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f})")
    print(f"  OBB       (mm) : {obb[0]:.1f} x {obb[1]:.1f} x {obb[2]:.1f}")
    print(f"  Blade dir      : [{pose['blade_axis'][0]:+.3f}, {pose['blade_axis'][1]:+.3f}, {pose['blade_axis'][2]:+.3f}]")
    print(f"  Normal vert.   : {pose['normal_verticality']*100:.1f}%")
    print(f"  PCA L/W ratio  : {pose['length_width_ratio']:.2f}")
    print(f"  N points       : {len(pts):,}")
    if mesh is not None:
        print(f"  Mesh verts     : {len(mesh.vertices):,}")
        print(f"  Mesh faces     : {len(mesh.triangles):,}")
    print(f"  소요시간       : {elapsed:.1f}s")
    print(f"{'='*60}")

    return result


# ======================================================================
#  메인
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="멀티뷰 RGB-D → 3D 복원 + 포즈 추정 → OpenCV/cam0 통합 출력")
    p.add_argument("--capture_dir", default=str(_DEFAULT_CAP_DIR))
    p.add_argument("--calib_dir", default=str(_DEFAULT_CAL_DIR))
    p.add_argument("--intrinsics_dir", default=str(_DEFAULT_INT_DIR))
    p.add_argument("--out_dir", default=str(_DEFAULT_OUT_DIR))
    p.add_argument("--frame", type=int, default=3)
    p.add_argument("--z_min", type=float, default=0.1)
    p.add_argument("--z_max", type=float, default=1.5)
    p.add_argument("--hsv_h_range", type=int, nargs=2, default=[15, 35])
    p.add_argument("--hsv_s_min", type=int, default=80)
    p.add_argument("--hsv_v_min", type=int, default=80)
    p.add_argument("--min_area", type=int, default=500)
    p.add_argument("--blade_dir", choices=["auto", "pos", "neg"], default="auto")
    p.add_argument("--no_tsdf", action="store_true", help="메시 복원 건너뜀 (점군만)")
    p.add_argument("--ref_length_mm", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    start = time.time()

    print("=" * 60)
    print("  멀티뷰 RGB-D → 3D 복원 + 포즈 (cam0/OpenCV)")
    print(f"  frame     : {args.frame}")
    print(f"  TSDF mesh : {'OFF' if args.no_tsdf else 'ON'}")
    print("=" * 60)

    # --- 캘리브레이션 ---
    cap_dir = Path(args.capture_dir)
    cams = discover_cameras(cap_dir)
    pad = frame_pad(cap_dir, cams[0])

    K_m, D_m, ds_m = {}, {}, {}
    for ci in cams:
        K_m[ci], D_m[ci], ds_m[ci] = load_intrinsics(Path(args.intrinsics_dir), ci)
    T_m = load_extrinsics(Path(args.calib_dir), cams)

    print(f"\n  카메라: {cams}")
    for ci in cams:
        print(f"    cam{ci}: depth_scale={ds_m[ci]:.6f}, "
              f"T_C0_C{ci}={'identity' if ci == 0 else 'loaded'}")

    # --- 세그멘테이션 + 점군 융합 ---
    print(f"\n[Step 1] 세그멘테이션 + 멀티뷰 점군 융합 (cam0)")
    pts, rgb, anchor = segment_and_fuse(args, cams, K_m, D_m, ds_m, T_m, pad)
    bbox = (pts.max(0) - pts.min(0)) * 1000
    print(f"  결과: {len(pts):,} pts, bbox: {bbox[0]:.1f} x {bbox[1]:.1f} x {bbox[2]:.1f} mm")

    # --- TSDF 3D 복원 ---
    mesh, tsdf_info = None, None
    if not args.no_tsdf:
        print(f"\n[Step 2] TSDF 3D 복원 (메시 + GLB)")
        mesh, tsdf_pcd, tsdf_info = reconstruct_tsdf(
            args, cams, K_m, D_m, ds_m, T_m, pad, pts)

    # --- 포즈 추정 ---
    print(f"\n[Step 3] 포즈 추정 (PCA + 색상 판별)")
    pose = estimate_pose_cam0(pts, rgb, args.blade_dir)

    # --- 저장 ---
    fid = f"{args.frame:0{pad}d}"
    tag = f"frame{fid}"
    out_dir = Path(args.out_dir) / f"output_{tag}"

    rgb_path_cam0 = str(cap_dir / f"cam0" / f"rgb_{fid}.jpg")
    elapsed = time.time() - start

    print(f"\n[Step 4] 결과 저장 (cam0/OpenCV 좌표계)")
    save_all(pts, rgb, pose, mesh, tsdf_info, out_dir, tag, elapsed,
             K_m[0], D_m[0], rgb_path_cam0)

    print(f"\n완료! (총 {elapsed:.1f}s)")
    print(f"\n출력 디렉토리: {out_dir}")
    print(f"  pose_cam0_{tag}.json   ← 시뮬레이션용 포즈 (R, t, euler, quat)")
    print(f"  pose_cam0_{tag}.npz    ← NumPy 바이너리 포즈")
    print(f"  pointcloud_cam0_{tag}.ply  ← 점군 (cam0 좌표계, 미터)")
    if not args.no_tsdf:
        print(f"  mesh_cam0_{tag}.ply    ← 메시 (cam0 좌표계)")
        print(f"  mesh_cam0_{tag}.glb    ← GLB (OpenGL 변환 포함)")
        print(f"  mesh_cam0_{tag}.obj    ← OBJ (cam0/OpenCV 그대로)")


if __name__ == "__main__":
    main()
