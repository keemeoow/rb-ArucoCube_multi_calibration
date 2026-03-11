#!/usr/bin/env python3
"""
pose_from_sam3d.py  —  카메라(cam0) 좌표계 기준 물체 6-DOF 포즈 추정
======================================================================
방법: scene PLY의 PCA 직접 계산 + 색상 기반 날/손잡이 방향 판별
      (ICP·참조모델 불필요, 기하학적으로 직접 추정)

좌표계: cam0 (OpenCV: X-right, Y-down, Z-forward)

사용법:
  # Mode D — 3D 복원 (PLY + GLB 메시 자동 생성) + 포즈 추정
  python Obj_Step2-(2)_pose_sam3d.py --frame 5 --reconstruct

  # Mode C — HSV 색상 필터 점군 (SAM2 불필요)
  python Obj_Step2-(2)_pose_sam3d.py --frame 5 --depth_only

  # Mode B — 기존 추출 점군 PLY 직접 사용 (빠름)
  python Obj_Step2-(2)_pose_sam3d.py \\
    --scene_ply "../Obj_Step2-(1)_pose_estimate_grounding_sam/output_frame000005/object_utility_knife_frame000005.ply"

  # Mode A — GDino+SAM2 재검출 후 멀티카메라 depth 융합
  python Obj_Step2-(2)_pose_sam3d.py --frame 5 --device mps

  # 날 방향 수동 지정 (색상 판별이 불확실할 때)
  python Obj_Step2-(2)_pose_sam3d.py --scene_ply <path> --blade_dir neg
"""

import os, sys, glob, json, time, argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CAP_DIR = os.path.join(_THIS_DIR, "../data/object_capture")
_DEFAULT_CAL_DIR = os.path.join(_THIS_DIR, "../data/cube_session_01/calib_out_cube")
_DEFAULT_INT_DIR = os.path.join(_THIS_DIR, "../data/_intrinsics")
_DEFAULT_OUT_DIR = os.path.join(_THIS_DIR, "./output")
_DEFAULT_SAM2    = os.path.join(_THIS_DIR, "../checkpoints/sam2.1_hiera_large.pt")


# ──────────────────────────────────────────────────────────────────
#  1. PLY 로드 (xyz + rgb)
# ──────────────────────────────────────────────────────────────────
def load_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """PLY → (xyz [N,3] float64, rgb [N,3] float64 or None)."""
    pts, cols, has_rgb = [], [], False
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if "property uchar red" in line:
                has_rgb = True
            if line == "end_header":
                break
        for _ in range(n):
            tok = f.readline().decode("ascii", errors="ignore").split()
            pts.append([float(tok[0]), float(tok[1]), float(tok[2])])
            if has_rgb:
                cols.append([int(tok[3]), int(tok[4]), int(tok[5])])
    pts = np.array(pts, dtype=np.float64)
    rgb = np.array(cols, dtype=np.float64) / 255.0 if has_rgb else None
    print(f"  [PLY] {os.path.basename(path)}: {len(pts):,} pts  rgb={'yes' if has_rgb else 'no'}")
    return pts, rgb


# ──────────────────────────────────────────────────────────────────
#  2. 캘리브레이션 로드
# ──────────────────────────────────────────────────────────────────
def load_intrinsics(intrin_dir: str, ci: int):
    d = np.load(os.path.join(intrin_dir, f"cam{ci}.npz"), allow_pickle=True)
    K  = d["color_K"].astype(np.float64)
    D  = d["color_D"].astype(np.float64) if "color_D" in d else np.zeros(5)
    ds = float(d.get("depth_scale_m_per_unit", 0.001))
    return K, D, ds

def load_extrinsics(calib_dir: str, cams: List[int]) -> Dict[int, np.ndarray]:
    T = {}
    for ci in cams:
        T[ci] = np.eye(4) if ci == 0 else \
                np.load(os.path.join(calib_dir, f"T_C0_C{ci}.npy")).astype(np.float64)
    return T

def discover_cameras(cap_dir: str) -> List[int]:
    ids = []
    for d in sorted(glob.glob(os.path.join(cap_dir, "cam*"))):
        try:
            ci = int(os.path.basename(d).replace("cam", ""))
        except ValueError:
            continue
        if glob.glob(os.path.join(d, "rgb_*.jpg")):
            ids.append(ci)
    if not ids:
        raise RuntimeError(f"cam* 없음: {cap_dir}")
    return ids

def frame_pad(cap_dir: str, ci: int) -> int:
    files = glob.glob(os.path.join(cap_dir, f"cam{ci}", "rgb_*.jpg"))
    return len(os.path.basename(files[0]).replace("rgb_","").replace(".jpg","")) if files else 6


# ──────────────────────────────────────────────────────────────────
#  3. Depth → 3D 점군 (멀티카메라 융합)
# ──────────────────────────────────────────────────────────────────
def depth_to_pts(depth_u16, mask, K, D, ds, z_min, z_max):
    h, w = depth_u16.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    vg, ug = np.mgrid[0:h, 0:w]
    z = depth_u16.astype(np.float64) * ds
    ok = mask & (z > z_min) & (z < z_max)
    if not ok.any():
        return np.empty((0,3)), np.empty((0,2))
    z, u, v = z[ok], ug[ok].astype(np.float64), vg[ok].astype(np.float64)
    und = cv2.undistortPoints(
        np.column_stack([u, v]).reshape(-1,1,2).astype(np.float64), K, D
    ).reshape(-1, 2)
    xyz = np.column_stack([und[:,0]*z, und[:,1]*z, z])
    return xyz, np.column_stack([u, v])

def fuse_multicam(cap_dir, frame_idx, cams, masks, K_m, D_m, ds_m, T_m,
                  z_min, z_max, pad):
    fid = f"{frame_idx:0{pad}d}"
    all_pts, all_rgb = [], []
    for ci in cams:
        if ci not in masks:
            continue
        rp = os.path.join(cap_dir, f"cam{ci}", f"rgb_{fid}.jpg")
        dp = os.path.join(cap_dir, f"cam{ci}", f"depth_{fid}.png")
        if not (os.path.exists(rp) and os.path.exists(dp)):
            continue
        bgr = cv2.imread(rp); dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            continue
        pts_ci, uvs = depth_to_pts(dep, masks[ci], K_m[ci], D_m[ci],
                                    ds_m[ci], z_min, z_max)
        if len(pts_ci) == 0:
            continue
        ui, vi = uvs[:,0].astype(int), uvs[:,1].astype(int)
        rgb = bgr[vi, ui][:, ::-1] / 255.0
        T   = T_m[ci]
        pts_cam0 = pts_ci @ T[:3,:3].T + T[:3,3]
        all_pts.append(pts_cam0); all_rgb.append(rgb)
        print(f"    cam{ci}: {len(pts_ci):,} pts → cam0")
    if not all_pts:
        return np.empty((0,3)), None
    return np.concatenate(all_pts), np.concatenate(all_rgb)

def sor(pts, rgb=None, std_ratio=1.5):
    n = len(pts)
    if n < 10:
        return pts, rgb
    mask = np.ones(n, bool)
    for ax in range(3):
        v = pts[:, ax]
        q1, q3 = np.percentile(v, [25, 75]); iqr = q3 - q1
        mask &= (v >= q1 - 1.5*iqr) & (v <= q3 + 1.5*iqr)
    c = pts[mask].mean(0)
    d = np.linalg.norm(pts - c, axis=1)
    mu, sg = d[mask].mean(), d[mask].std()
    mask &= d < mu + std_ratio * sg
    print(f"    SOR: {n:,} → {mask.sum():,}")
    return pts[mask], (rgb[mask] if rgb is not None else None)


# ──────────────────────────────────────────────────────────────────
#  4a. 2단계 세그멘테이션 + 멀티프레임 (Mode C/D — SAM2 불필요)
# ──────────────────────────────────────────────────────────────────
def _find_hsv_anchor(bgr, dep, ds, h_lo, h_hi, s_lo, v_lo, min_area):
    """
    HSV 색상 필터로 노란 손잡이 감지 → (bbox, median_depth_m) 반환.
    bbox = (x1, y1, x2, y2) 가장 큰 노란 영역의 bounding box.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
        (hsv[:, :, 1] >= s_lo) & (hsv[:, :, 2] >= v_lo)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    n_lab, lab_img, stats, _ = cv2.connectedComponentsWithStats(mask)
    best_lb, best_area = -1, 0
    for lb in range(1, n_lab):
        a = stats[lb, cv2.CC_STAT_AREA]
        if a >= min_area and a > best_area:
            best_lb, best_area = lb, a
    if best_lb < 0:
        return None, None

    x1 = stats[best_lb, cv2.CC_STAT_LEFT]
    y1 = stats[best_lb, cv2.CC_STAT_TOP]
    x2 = x1 + stats[best_lb, cv2.CC_STAT_WIDTH]
    y2 = y1 + stats[best_lb, cv2.CC_STAT_HEIGHT]

    # 노란 영역의 median depth
    region_mask = (lab_img == best_lb)
    dep_vals = dep[region_mask].astype(np.float64) * ds
    dep_vals = dep_vals[dep_vals > 0.05]
    med_depth = float(np.median(dep_vals)) if len(dep_vals) > 0 else None
    return (x1, y1, x2, y2), med_depth


def _expand_bbox_for_knife(bbox, img_shape, expand_ratio=1.8):
    """
    HSV 손잡이 bbox를 칼 전체를 포함하도록 확장.
    칼이 손잡이보다 길므로 장축 방향으로 더 확장.
    """
    x1, y1, x2, y2 = bbox
    h, w = img_shape[:2]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # 장축(칼 길이) 방향으로 더 넓게
    long_side = max(bw, bh)
    new_half = long_side * expand_ratio
    nx1 = max(0, int(cx - new_half))
    ny1 = max(0, int(cy - new_half * 0.5))
    nx2 = min(w, int(cx + new_half))
    ny2 = min(h, int(cy + new_half * 0.5))
    return nx1, ny1, nx2, ny2


def run_color_segmentation(args, cams, K_m, D_m, ds_m, T_m, pad):
    """
    3단계 세그멘테이션 + 멀티프레임 누적:
    Stage 1) HSV → 노란 손잡이에서 cam0 3D 앵커 중심점 확보
    Stage 2) 앵커 주변 넓은 depth ROI → cam0 융합
    Stage 3) 3D 공간에서 점진적 크롭 (구 → PCA → 타원체) → 칼만 분리
    """
    h_lo, h_hi = args.hsv_h_range
    s_lo = args.hsv_s_min
    v_lo = args.hsv_v_min
    frames = args.frames if args.frames else [args.frame]

    # ── Stage 1: HSV 앵커 → cam0 3D 중심 ─────────────────────
    # 기준 프레임(args.frame)에서 노란 손잡이의 cam0 3D 중심점 계산
    anchor_pts_3d = []
    fid_ref = f"{args.frame:0{pad}d}"
    for ci in cams:
        rp = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid_ref}.jpg")
        dp = os.path.join(args.capture_dir, f"cam{ci}", f"depth_{fid_ref}.png")
        if not (os.path.exists(rp) and os.path.exists(dp)):
            continue
        bgr = cv2.imread(rp)
        dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            continue

        # HSV 마스크 (노란색만)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask_yellow = (
            (hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
            (hsv[:, :, 1] >= s_lo) & (hsv[:, :, 2] >= v_lo)
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_yellow = cv2.morphologyEx(mask_yellow.astype(np.uint8),
                                        cv2.MORPH_OPEN, kernel).astype(bool)
        if mask_yellow.sum() < args.min_component_area:
            continue

        pts_ci, _ = depth_to_pts(dep, mask_yellow, K_m[ci], D_m[ci],
                                  ds_m[ci], args.z_min, args.z_max)
        if len(pts_ci) == 0:
            continue
        T = T_m[ci]
        pts_cam0 = pts_ci @ T[:3, :3].T + T[:3, 3]
        anchor_pts_3d.append(pts_cam0)
        print(f"    cam{ci}: HSV 앵커 {len(pts_ci):,} pts")

    if not anchor_pts_3d:
        raise RuntimeError("노란 손잡이 앵커 검출 실패")

    anchor_all = np.concatenate(anchor_pts_3d)
    anchor_center = anchor_all.mean(axis=0)
    print(f"    앵커 중심 (cam0): ({anchor_center[0]*1000:.1f}, "
          f"{anchor_center[1]*1000:.1f}, {anchor_center[2]*1000:.1f}) mm")

    # ── Stage 2: 멀티프레임 넓은 depth ROI → cam0 점군 ──────
    all_pts, all_rgb = [], []
    sphere_r = 0.12  # 앵커 주변 120mm 구

    for fi in frames:
        fid = f"{fi:0{pad}d}"
        for ci in cams:
            rp = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid}.jpg")
            dp = os.path.join(args.capture_dir, f"cam{ci}", f"depth_{fid}.png")
            if not (os.path.exists(rp) and os.path.exists(dp)):
                continue
            bgr = cv2.imread(rp)
            dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
            if bgr is None or dep is None:
                continue

            # 전체 depth → 3D (z 범위 내)
            full_mask = np.ones(dep.shape[:2], dtype=bool)
            pts_ci, uvs = depth_to_pts(dep, full_mask, K_m[ci], D_m[ci],
                                        ds_m[ci], args.z_min, args.z_max)
            if len(pts_ci) == 0:
                continue

            T = T_m[ci]
            pts_cam0 = pts_ci @ T[:3, :3].T + T[:3, 3]

            # 구 크롭: 앵커 중심 주변만
            dist = np.linalg.norm(pts_cam0 - anchor_center, axis=1)
            in_sphere = dist < sphere_r
            pts_cam0 = pts_cam0[in_sphere]
            uvs = uvs[in_sphere]

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
    print(f"    구 크롭 (r={sphere_r*1000:.0f}mm): {len(pts):,} pts "
          f"({len(frames)} frames × {len(cams)} cams)")

    # ── Stage 3: PCA 기반 타원체 크롭 → 칼만 분리 ────────────
    # 3-1) 앵커 중심으로 1차 PCA
    centered = pts - anchor_center
    cov = centered.T @ centered / len(centered)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    # PCA 좌표 변환
    proj = centered @ evecs  # [N, 3] in PCA space

    # 3-2) 타원체 크롭: 장축 ±100mm, 중간축 ±18mm, 단축 ±18mm
    # (칼: ~150mm × ~20mm × ~10mm → 넉넉하게 설정)
    a_long  = getattr(args, 'crop_long',  0.100)   # 장축 반경 100mm
    a_mid   = getattr(args, 'crop_short', 0.018)   # 중간축 반경 18mm
    a_short = getattr(args, 'crop_short', 0.018)   # 단축 반경 18mm

    ellip_dist = (proj[:, 0] / a_long)**2 + \
                 (proj[:, 1] / a_mid)**2 + \
                 (proj[:, 2] / a_short)**2
    in_ellipsoid = ellip_dist < 1.0

    pts_crop = pts[in_ellipsoid]
    rgb_crop = rgb[in_ellipsoid] if rgb is not None else None
    print(f"    타원체 크롭: {in_ellipsoid.sum():,} / {len(pts):,} pts "
          f"(장축={a_long*1000:.0f}mm, 단축={a_mid*1000:.0f}mm)")

    if len(pts_crop) < 50:
        print("    [WARN] 타원체 크롭 결과 부족, 구 크롭 결과 사용")
        pts_crop, rgb_crop = pts, rgb

    # SOR
    pts_crop, rgb_crop = sor(pts_crop, rgb_crop)

    # DBSCAN 최종 정리
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=0.005, min_samples=10).fit(pts_crop)
    labels = db.labels_
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) > 0:
        best_label = unique[np.argmax(counts)]
        mask_cl = labels == best_label
        print(f"    DBSCAN: {len(unique)} 클러스터, 최대={mask_cl.sum():,} pts")
        pts_crop = pts_crop[mask_cl]
        rgb_crop = rgb_crop[mask_cl] if rgb_crop is not None else None

    return pts_crop, rgb_crop


# ──────────────────────────────────────────────────────────────────
#  4b. 3D 복원: 점군 → 메시 → PLY + GLB (Mode D)
# ──────────────────────────────────────────────────────────────────
def save_ply_ascii(path: str, pts: np.ndarray, rgb: Optional[np.ndarray]):
    """점군을 ASCII PLY로 저장."""
    n = len(pts)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            line = f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f}"
            if rgb is not None:
                r, g, b = (rgb[i] * 255).astype(np.uint8)
                line += f" {r} {g} {b}"
            f.write(line + "\n")
    print(f"    [저장] {path} ({n:,} pts)")


def reconstruct_tsdf(args, cams, K_m, D_m, ds_m, T_m, pad,
                     seg_pts: np.ndarray,
                     out_dir: str, tag: str,
                     voxel_length: float = 0.0005,
                     sdf_trunc: float = 0.003):
    """
    TSDF Volume Integration: 멀티카메라 depth를 볼륨에 직접 통합.
    depth 센서 전용 3D 복원 — 노이즈 평균화, marching cubes 메시 추출.

    기존 점군 합치기와 근본적으로 다름:
    - 각 depth 프레임을 TSDF 볼륨에 통합 (가중 평균)
    - 여러 뷰의 관측이 겹치면 노이즈가 줄어듦
    - Marching cubes로 깨끗한 watertight 메시 추출

    Returns: (mesh_pts, mesh_rgb, info_dict)
    """
    import open3d as o3d

    frames = args.frames if args.frames else [args.frame]
    h_lo, h_hi = args.hsv_h_range
    s_lo, v_lo = args.hsv_s_min, args.hsv_v_min

    # TSDF 볼륨 생성 (voxel=0.5mm, sdf_trunc=3mm)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    n_integrated = 0
    for fi in frames:
        fid = f"{fi:0{pad}d}"
        for ci in cams:
            rp = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid}.jpg")
            dp = os.path.join(args.capture_dir, f"cam{ci}", f"depth_{fid}.png")
            if not (os.path.exists(rp) and os.path.exists(dp)):
                continue
            bgr = cv2.imread(rp)
            dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
            if bgr is None or dep is None:
                continue

            rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h_img, w_img = dep.shape[:2]

            # 물체 주변만 TSDF에 넣기 위해 마스크 적용
            # (전체 씬을 넣으면 테이블 전체가 복원됨)
            # 1) HSV 앵커의 2D 위치 찾기
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask_yellow = (
                (hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
                (hsv[:, :, 1] >= s_lo) & (hsv[:, :, 2] >= v_lo)
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_yellow = cv2.morphologyEx(mask_yellow.astype(np.uint8),
                                            cv2.MORPH_OPEN, kernel)

            # 가장 큰 노란 영역 찾기
            n_lab, lab_img, stats, centroids = cv2.connectedComponentsWithStats(mask_yellow)
            best_lb, best_area = -1, 0
            for lb in range(1, n_lab):
                a = stats[lb, cv2.CC_STAT_AREA]
                if a >= args.min_component_area and a > best_area:
                    best_lb, best_area = lb, a
            if best_lb < 0:
                continue

            # 앵커 bbox 중심으로 ROI 확장 (칼 전체 포함)
            cx_2d = int(centroids[best_lb][0])
            cy_2d = int(centroids[best_lb][1])
            roi_half = int(max(stats[best_lb, cv2.CC_STAT_WIDTH],
                              stats[best_lb, cv2.CC_STAT_HEIGHT]) * 2.5)
            rx1 = max(0, cx_2d - roi_half)
            ry1 = max(0, cy_2d - roi_half)
            rx2 = min(w_img, cx_2d + roi_half)
            ry2 = min(h_img, cy_2d + roi_half)

            # 앵커 depth
            yellow_mask_bool = (lab_img == best_lb)
            dep_vals = dep[yellow_mask_bool].astype(np.float64) * ds_m[ci]
            dep_vals = dep_vals[dep_vals > 0.05]
            if len(dep_vals) == 0:
                continue
            anchor_d = float(np.median(dep_vals))

            # ROI 마스크: 2D bbox + depth 범위
            dep_m = dep.astype(np.float64) * ds_m[ci]
            depth_ok = (dep_m > anchor_d - 0.020) & (dep_m < anchor_d + 0.020)
            spatial_ok = np.zeros((h_img, w_img), dtype=bool)
            spatial_ok[ry1:ry2, rx1:rx2] = True
            valid = spatial_ok & depth_ok & (dep > 0)

            # 마스크 적용: 유효하지 않은 영역의 depth를 0으로
            dep_masked = dep.copy()
            dep_masked[~valid] = 0

            # cam→cam0 외참 행렬 (Open3D는 카메라→월드 변환 사용)
            T_c0_ci = T_m[ci]  # cam_i → cam0
            # Open3D integrate는 extrinsic = world→camera 변환을 요구
            # 우리 T_c0_ci는 cam_i → cam0이므로, 역변환 = cam0 → cam_i
            T_cam0_to_ci = np.linalg.inv(T_c0_ci)

            # Open3D RGBD 이미지 생성
            color_o3d = o3d.geometry.Image(rgb_img.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(dep_masked)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0 / ds_m[ci],
                depth_trunc=anchor_d + 0.030,
                convert_rgb_to_intensity=False,
            )

            # intrinsic
            K = K_m[ci]
            h_rgb, w_rgb = rgb_img.shape[:2]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                w_rgb, h_rgb, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            )

            # TSDF에 통합
            volume.integrate(rgbd, intrinsic, T_cam0_to_ci)
            n_integrated += 1

    if n_integrated == 0:
        raise RuntimeError("TSDF 통합 프레임 없음")

    print(f"    TSDF 통합: {n_integrated} 프레임 (voxel={voxel_length*1000:.1f}mm, "
          f"sdf_trunc={sdf_trunc*1000:.1f}mm)")

    # 메시 추출 (Marching Cubes)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    # 점군도 추출
    pcd = volume.extract_point_cloud()

    # PCA 기반 타원체 크롭 (세그멘테이션 점군의 PCA 사용)
    seg_center = seg_pts.mean(axis=0)
    seg_centered = seg_pts - seg_center
    seg_cov = seg_centered.T @ seg_centered / len(seg_centered)
    seg_evals, seg_evecs = np.linalg.eigh(seg_cov)
    seg_order = np.argsort(seg_evals)[::-1]
    seg_evecs = seg_evecs[:, seg_order]

    # 세그멘테이션 OBB 크기 + 여유
    seg_proj = seg_centered @ seg_evecs
    seg_extents = seg_proj.max(0) - seg_proj.min(0)
    margin = 1.3  # 30% 여유 (TSDF가 조금 넓을 수 있음)

    # 메시 정점 타원체 필터링
    mesh_verts = np.asarray(mesh.vertices)
    mesh_proj = (mesh_verts - seg_center) @ seg_evecs
    half_ext = seg_extents * margin / 2
    in_obb = np.all(np.abs(mesh_proj) < half_ext, axis=1)
    mesh_indices = np.where(in_obb)[0]
    if len(mesh_indices) > 100:
        mesh = mesh.select_by_index(mesh_indices)

    # 점군도 동일 크롭
    pcd_pts_arr = np.asarray(pcd.points)
    pcd_proj = (pcd_pts_arr - seg_center) @ seg_evecs
    in_obb_pcd = np.all(np.abs(pcd_proj) < half_ext, axis=1)
    pcd_indices = np.where(in_obb_pcd)[0]
    if len(pcd_indices) > 100:
        pcd = pcd.select_by_index(pcd_indices)

    print(f"    PCA OBB 크롭: 메시 {len(mesh_verts):,} → {len(mesh.vertices):,}, "
          f"점군 {len(pcd_pts_arr):,} → {len(pcd.points):,}")
    print(f"    OBB 크기: {seg_extents[0]*1000*margin:.1f} × "
          f"{seg_extents[1]*1000*margin:.1f} × {seg_extents[2]*1000*margin:.1f} mm")

    # 메시 정리
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    print(f"    메시: {n_verts:,} vertices, {n_faces:,} faces")
    print(f"    점군: {len(pcd.points):,} pts")

    os.makedirs(out_dir, exist_ok=True)

    # PLY 점군 저장
    ply_pts_path = os.path.join(out_dir, f"reconstructed_pointcloud_{tag}.ply")
    o3d.io.write_point_cloud(ply_pts_path, pcd)
    print(f"    [저장] {ply_pts_path}")

    # PLY 메시 저장
    ply_mesh_path = os.path.join(out_dir, f"reconstructed_mesh_{tag}.ply")
    o3d.io.write_triangle_mesh(ply_mesh_path, mesh)
    print(f"    [저장] {ply_mesh_path}")

    # GLB 메시 저장
    glb_path = os.path.join(out_dir, f"reconstructed_mesh_{tag}.glb")
    try:
        import trimesh
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        vertex_colors = None
        if mesh.has_vertex_colors():
            vc = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
            vertex_colors = np.column_stack([vc, np.full(len(vc), 255, dtype=np.uint8)])
        tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
        tm.export(glb_path, file_type="glb")
        print(f"    [저장] {glb_path}")
    except Exception as e:
        print(f"    [WARN] GLB 저장 실패: {e}")

    # 포즈 추정용 데이터
    pcd_pts = np.asarray(pcd.points)
    pcd_rgb = np.asarray(pcd.colors) if pcd.has_colors() else None
    return pcd_pts, pcd_rgb, {
        "ply_pointcloud": ply_pts_path,
        "ply_mesh": ply_mesh_path,
        "glb_mesh": glb_path,
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "n_integrated_frames": n_integrated,
        "tsdf_voxel_mm": voxel_length * 1000,
        "tsdf_sdf_trunc_mm": sdf_trunc * 1000,
    }


def visualize_pointcloud(pts: np.ndarray, rgb: Optional[np.ndarray],
                         pose: dict, out_path: str, title: str):
    """점군 + OBB + 축을 3D scatter로 시각화."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))

    # (a) 3D scatter — 3 views
    pts_mm = pts * 1000
    R = pose["R"]
    c_mm = pose["centroid_mm"]
    obb = pose["obb_mm"]

    for idx, (elev, azim, subtitle) in enumerate([
        (90, -90, "Top (XZ)"), (0, -90, "Front (XY)"), (0, 0, "Side (YZ)")
    ]):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        ax.set_title(subtitle, fontsize=10)

        # 점군 scatter (색상 포함)
        if rgb is not None:
            colors = rgb[:len(pts)]
            ax.scatter(pts_mm[:, 0], pts_mm[:, 1], pts_mm[:, 2],
                      c=colors, s=0.3, alpha=0.6)
        else:
            ax.scatter(pts_mm[:, 0], pts_mm[:, 1], pts_mm[:, 2],
                      s=0.3, alpha=0.6, c="steelblue")

        # OBB 와이어프레임
        h = obb / 2.0
        corners_l = np.array([
            [-h[0], -h[1], -h[2]], [h[0], -h[1], -h[2]],
            [h[0], h[1], -h[2]], [-h[0], h[1], -h[2]],
            [-h[0], -h[1], h[2]], [h[0], -h[1], h[2]],
            [h[0], h[1], h[2]], [-h[0], h[1], h[2]],
        ])
        corners = (R @ corners_l.T).T + c_mm
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                 (0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            ax.plot3D(*zip(corners[i], corners[j]), "r-", lw=0.8, alpha=0.7)

        # 축
        axis_len = 30
        for ai, col in enumerate(["#e74c3c", "#27ae60", "#2980b9"]):
            v = R[:, ai] * axis_len
            ax.quiver(c_mm[0], c_mm[1], c_mm[2], v[0], v[1], v[2],
                      color=col, linewidth=2, arrow_length_ratio=0.15)

        ax.set_xlabel("X mm", fontsize=7)
        ax.set_ylabel("Y mm", fontsize=7)
        ax.set_zlabel("Z mm", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)

        # 등축 비율
        mid = pts_mm.mean(axis=0)
        rng = max((pts_mm.max(0) - pts_mm.min(0)).max() / 2 * 1.3, 40)
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)

    fig.suptitle(title, fontsize=11, y=0.98)
    fig.text(0.5, 0.01,
             f"OBB: {obb[0]:.1f}×{obb[1]:.1f}×{obb[2]:.1f} mm  |  "
             f"PCA ratio: {pose['length_width_ratio']:.1f}  |  "
             f"{len(pts):,} pts",
             ha="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ecf0f1", ec="#bdc3c7"))
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {out_path}")


# ──────────────────────────────────────────────────────────────────
#  4c. GDino + SAM2 검출 (Mode A)
# ──────────────────────────────────────────────────────────────────
def run_detection(args, cams, K_m, D_m, ds_m, T_m, pad):
    sam_dir = os.path.join(_THIS_DIR, "../pose_estimate_grounding_sam(최종)")
    if os.path.isdir(sam_dir) and sam_dir not in sys.path:
        sys.path.insert(0, sam_dir)
    from estimate_object_pose import (
        load_grounding_dino, load_sam2, detect_and_segment_object)

    print(f"  모델 로딩 (device={args.device})...")
    gp, gm = load_grounding_dino(args.gdino_model, args.device)
    sp     = load_sam2(args.sam2_checkpoint, args.sam2_config, args.device)

    fid = f"{args.frame:0{pad}d}"
    masks: Dict[int, np.ndarray] = {}
    for ci in cams:
        rp = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid}.jpg")
        if not os.path.exists(rp):
            continue
        bgr = cv2.imread(rp)
        res = detect_and_segment_object(
            gp, gm, sp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            bbox_pad_ratio=args.bbox_pad,
            sam_refine_iters=args.sam_refine,
            device=args.device,
        )
        if res is None:
            print(f"  cam{ci}: 검출 없음")
        else:
            mask, det = res
            print(f"  cam{ci}: score={det['score']:.3f}")
            masks[ci] = mask
    if not masks:
        raise RuntimeError("모든 카메라에서 검출 실패")

    pts, rgb = fuse_multicam(
        args.capture_dir, args.frame, cams, masks,
        K_m, D_m, ds_m, T_m, args.z_min, args.z_max, pad)
    pts, rgb = sor(pts, rgb)
    if len(pts) < 50:
        raise RuntimeError("융합 점군 부족")
    return pts, rgb


# ──────────────────────────────────────────────────────────────────
#  5. 포즈 추정 (PCA 직접 + 색상 판별)
# ──────────────────────────────────────────────────────────────────
def estimate_pose_cam0(pts: np.ndarray, rgb: Optional[np.ndarray],
                       blade_dir: str = "auto") -> dict:
    """
    cam0 좌표계 기준 6-DOF 포즈 추정.

    칼 canonical frame:
      x = 날 방향 (blade)
      y = 너비 방향 (width)
      z = 법선 방향 (normal, 테이블 위로)

    blade_dir: 'auto' | 'pos' | 'neg'
      auto — 색상 분석으로 자동 판별 (황색=손잡이, 회색=날)
      pos  — PCA 주축 양(+)방향을 날로 강제
      neg  — PCA 주축 음(-)방향을 날로 강제
    """
    centroid = pts.mean(0)
    p = pts - centroid

    # PCA
    cov = p.T @ p / len(p)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    if np.linalg.det(evec) < 0:
        evec[:, 2] *= -1

    length_ax = evec[:, 0].copy()   # 최대 분산 → 칼 길이
    normal_ax = evec[:, 2].copy()   # 최소 분산 → 칼 법선

    # 법선 방향: 카메라(-Y 방향, 위쪽)로 향하게 고정
    if normal_ax[1] > 0:
        normal_ax = -normal_ax

    # ── 날 방향 판별 ────────────────────────────────────────────
    proj = p @ length_ax

    if blade_dir == "auto":
        if rgb is not None:
            half_pos = proj > np.median(proj)
            def yellowness(mask):
                c = rgb[mask]
                return ((c[:,0] + c[:,1]) / 2 - c[:,2]).mean()
            yp = yellowness(half_pos)
            yn = yellowness(~half_pos)
            # 황색도 낮은 쪽 = 날 (은색/어두움)
            blade_is_pos = yp < yn
            conf = abs(yp - yn)
            blade_dir_used = f"color_auto (yp={yp:.3f}, yn={yn:.3f}, conf={conf:.3f})"
        else:
            blade_is_pos = True
            blade_dir_used = "color_unavailable → pos 기본값"
    elif blade_dir == "pos":
        blade_is_pos = True
        blade_dir_used = "manual:pos"
    else:  # neg
        blade_is_pos = False
        blade_dir_used = "manual:neg"

    if not blade_is_pos:
        length_ax = -length_ax

    # 너비 축: 오른손 좌표계
    width_ax = np.cross(normal_ax, length_ax)
    width_ax /= np.linalg.norm(width_ax)

    # 회전행렬 (knife → cam0)
    R = np.column_stack([length_ax, width_ax, normal_ax])
    assert abs(np.linalg.det(R) - 1.0) < 1e-5

    # OBB extents
    proj_R = (pts - centroid) @ R
    extents = proj_R.max(0) - proj_R.min(0)

    return {
        "centroid_m":   centroid,
        "centroid_mm":  centroid * 1000,
        "R":            R,
        "blade_axis":   length_ax,
        "width_axis":   width_ax,
        "normal_axis":  normal_ax,
        "obb_mm":       extents * 1000,
        "pca_ev":       ev,
        "blade_dir_used": blade_dir_used,
        "length_width_ratio": float(ev[0] / (ev[1] + 1e-12)),
        "normal_verticality": float(abs(normal_ax[1])),
    }


# ──────────────────────────────────────────────────────────────────
#  6. 회전 변환 유틸
# ──────────────────────────────────────────────────────────────────
def R_to_euler(R: np.ndarray) -> np.ndarray:
    """Euler XYZ (deg)."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0.0
    return np.degrees([x, y, z])

def R_to_quat(R: np.ndarray) -> np.ndarray:
    """Quaternion wxyz (w > 0)."""
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s=0.5/np.sqrt(tr+1); w,x,y,z=0.25/s,(R[2,1]-R[1,2])*s,(R[0,2]-R[2,0])*s,(R[1,0]-R[0,1])*s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); w,x,y,z=(R[2,1]-R[1,2])/s,0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); w,x,y,z=(R[0,2]-R[2,0])/s,(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s
    else:
        s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); w,x,y,z=(R[1,0]-R[0,1])/s,(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s
    q = np.array([w,x,y,z]); q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


# ──────────────────────────────────────────────────────────────────
#  7. 포즈 시각화 (cam0 좌표계)
# ──────────────────────────────────────────────────────────────────
def visualize_pose_cam0(pose: dict, out_path: str, title: str):
    """
    cam0 + 객체 포즈를 3D 좌표 프레임으로 시각화 저장.
    축 색상: X=Red, Y=Green, Z=Blue
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    R = pose["R"]
    pos_mm = pose["centroid_mm"]
    obb_mm = pose["obb_mm"]
    blade = pose["blade_axis"]
    euler = R_to_euler(R)
    quat = R_to_quat(R)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    def draw_axes(Rm, t, length, label, lw=2.0, alpha=1.0):
        colors = ["#e74c3c", "#27ae60", "#2980b9"]
        names = ["X", "Y", "Z"]
        for i in range(3):
            v = Rm[:, i] * length
            ax.quiver(t[0], t[1], t[2], v[0], v[1], v[2],
                      color=colors[i], linewidth=lw, arrow_length_ratio=0.12, alpha=alpha)
            tip = t + v * 1.14
            ax.text(tip[0], tip[1], tip[2], names[i],
                    fontsize=7, color=colors[i], fontweight="bold", alpha=alpha)
        if label:
            ax.text(t[0], t[1], t[2] - length * 0.4, label,
                    fontsize=9, fontweight="bold", ha="center")

    def draw_obb(Rm, center, extents, color="#c0392b"):
        h = extents / 2.0
        corners_l = np.array([
            [-h[0], -h[1], -h[2]], [h[0], -h[1], -h[2]],
            [h[0], h[1], -h[2]], [-h[0], h[1], -h[2]],
            [-h[0], -h[1], h[2]], [h[0], -h[1], h[2]],
            [h[0], h[1], h[2]], [-h[0], h[1], h[2]],
        ])
        corners = (Rm @ corners_l.T).T + center
        faces = [[corners[j] for j in f] for f in
                 [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]]
        ax.add_collection3d(Poly3DCollection(
            faces, alpha=0.06, facecolor=color, edgecolor=color, linewidth=0.4))
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i, j in edges:
            ax.plot3D(*zip(corners[i], corners[j]), color=color, lw=1.0, alpha=0.55)

    draw_axes(np.eye(3), np.zeros(3), 70, "cam0 (ref)", lw=3.0)
    draw_axes(R, pos_mm, 55, "", lw=3.5)
    draw_obb(R, pos_mm, obb_mm)

    ax.plot3D([0, pos_mm[0]], [0, pos_mm[1]], [0, pos_mm[2]],
              ":", color="#c0392b", lw=1.0, alpha=0.35)

    ax.text(pos_mm[0], pos_mm[1] - 45, pos_mm[2] + 70,
            "Object", fontsize=11, fontweight="bold", color="#c0392b", ha="center")
    ax.text(pos_mm[0], pos_mm[1] - 45, pos_mm[2] + 52,
            f"({pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}) mm",
            fontsize=8, color="#2c3e50", ha="center")
    ax.text(pos_mm[0], pos_mm[1] - 45, pos_mm[2] + 35,
            f"euler ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg",
            fontsize=8, color="#2c3e50", ha="center")
    ax.text(pos_mm[0], pos_mm[1] - 45, pos_mm[2] + 18,
            f"blade axis [{blade[0]:+.2f}, {blade[1]:+.2f}, {blade[2]:+.2f}]",
            fontsize=7, color="#2c3e50", ha="center")

    pts = np.array([[0, 0, 0], pos_mm.tolist()], dtype=np.float64)
    c = pts.mean(axis=0)
    r = max((pts.max(axis=0) - pts.min(axis=0)).max() / 2 * 1.4, 120.0)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=24, azim=-55)

    fig.text(
        0.5, 0.01,
        "Axis color: X=Red  Y=Green  Z=Blue\n"
        f"Quat wxyz: ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})",
        fontsize=8, ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ecf0f1", ec="#bdc3c7"),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
#  8. 결과 출력 + 저장
# ──────────────────────────────────────────────────────────────────
def print_and_save(pose: dict, out_dir: str, tag: str, elapsed: float):
    R   = pose["R"]
    pos = pose["centroid_mm"]
    eu  = R_to_euler(R)
    q   = R_to_quat(R)
    obb = pose["obb_mm"]
    ba  = pose["blade_axis"]

    print(f"\n{'='*56}")
    print(f"  RESULT  —  cam0 (X-right, Y-down, Z-forward)")
    print(f"{'='*56}")
    print(f"  Position  (mm): ({pos[0]:+.1f}, {pos[1]:+.1f}, {pos[2]:+.1f})")
    print(f"  Euler XYZ (deg): ({eu[0]:+.1f}, {eu[1]:+.1f}, {eu[2]:+.1f})")
    print(f"  Quat wxyz      : ({q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}, {q[3]:.5f})")
    print(f"  OBB       (mm) : {obb[0]:.1f} x {obb[1]:.1f} x {obb[2]:.1f}")
    print(f"  날 방향  (cam0): [{ba[0]:+.4f}, {ba[1]:+.4f}, {ba[2]:+.4f}]")
    print(f"  법선 수직성    : {pose['normal_verticality']*100:.1f}%")
    print(f"  PCA 길이/폭    : {pose['length_width_ratio']:.2f}")
    print(f"  날방향 판별    : {pose['blade_dir_used']}")
    print(f"  소요시간       : {elapsed:.1f}s")
    print(f"{'='*56}")

    os.makedirs(out_dir, exist_ok=True)
    result = {
        "coordinate_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
        "method": "PCA_direct + color_blade_disambiguation",
        "position_mm":      pos.tolist(),
        "euler_xyz_deg":    eu.tolist(),
        "quaternion_wxyz":  q.tolist(),
        "rotation_matrix":  R.tolist(),
        "blade_axis_cam0":  ba.tolist(),
        "normal_axis_cam0": pose["normal_axis"].tolist(),
        "obb_mm":           obb.tolist(),
        "pca_length_width_ratio": pose["length_width_ratio"],
        "normal_verticality_pct": pose["normal_verticality"] * 100,
        "blade_dir_used":   pose["blade_dir_used"],
        "elapsed_sec":      round(elapsed, 2),
    }
    json_path = os.path.join(out_dir, f"pose_cam0_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {json_path}")

    vis_path = os.path.join(out_dir, f"pose_cam0_{tag}.png")
    try:
        visualize_pose_cam0(
            pose,
            vis_path,
            "pose_from_sam3d (cam0 frame, mm)",
        )
        print(f"  [저장] {vis_path}")
    except Exception as e:
        print(f"  [WARN] 시각화 저장 실패: {e}")

    return result


# ──────────────────────────────────────────────────────────────────
#  9. Main
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="cam0 좌표계 기준 물체 6-DOF 포즈 추정 (PCA 직접)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Mode D: 3D 복원 (PLY + GLB 메시 생성) + 포즈 추정
  python Obj_Step2-(2)_pose_sam3d.py --frame 5 --reconstruct
  python Obj_Step2-(2)_pose_sam3d.py --frame 5 --reconstruct --poisson_depth 9

  # Mode C: HSV 색상 필터로 물체 추출 (SAM2 불필요)
  python Obj_Step2-(2)_pose_sam3d.py --frame 5 --depth_only

  # Mode B: 기존 PLY 직접 사용
  python Obj_Step2-(2)_pose_sam3d.py \\
    --scene_ply "../Obj_Step2-(1)_pose_estimate_grounding_sam/output_frame000005/object_utility_knife_frame000005.ply"

  # Mode A: GDino+SAM2 재검출
  python Obj_Step2-(2)_pose_sam3d.py --frame 5 --device mps
""")

    # 입력
    ap.add_argument("--scene_ply",  default=None,
                    help="[Mode B] 물체 점군 PLY 경로")
    ap.add_argument("--capture_dir",    default=_DEFAULT_CAP_DIR)
    ap.add_argument("--calib_dir",      default=_DEFAULT_CAL_DIR)
    ap.add_argument("--intrinsics_dir", default=_DEFAULT_INT_DIR)
    ap.add_argument("--frame",          type=int,   default=5)
    ap.add_argument("--text_prompt",    default="utility knife.")
    ap.add_argument("--gdino_model",    default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam2_checkpoint",default=_DEFAULT_SAM2)
    ap.add_argument("--sam2_config",    default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--device",         default="mps")
    ap.add_argument("--box_threshold",  type=float, default=0.15)
    ap.add_argument("--text_threshold", type=float, default=0.15)
    ap.add_argument("--bbox_pad",       type=float, default=0.10)
    ap.add_argument("--sam_refine",     type=int,   default=1)
    ap.add_argument("--z_min",          type=float, default=0.1)
    ap.add_argument("--z_max",          type=float, default=1.5)

    # Mode C (색상 기반, SAM2 불필요)
    ap.add_argument("--depth_only", action="store_true",
                    help="[Mode C] SAM2 없이 HSV 색상 필터로 물체 추출")
    ap.add_argument("--hsv_h_range", type=int, nargs=2, default=[15, 35],
                    help="HSV Hue 범위 (default: 15 35 = 노란색)")
    ap.add_argument("--hsv_s_min", type=int, default=80,
                    help="HSV Saturation 최소 (default: 80)")
    ap.add_argument("--hsv_v_min", type=int, default=80,
                    help="HSV Value 최소 (default: 80)")
    ap.add_argument("--min_component_area", type=int, default=500,
                    help="최소 연결 영역 면적 (px, default: 500)")
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="멀티프레임 누적 (e.g. --frames 3 4 5 6)")
    ap.add_argument("--crop_long", type=float, default=0.100,
                    help="타원체 장축 반경 (m, default: 0.100 = 100mm)")
    ap.add_argument("--crop_short", type=float, default=0.018,
                    help="타원체 단축 반경 (m, default: 0.018 = 18mm)")

    # Mode D (3D TSDF 복원)
    ap.add_argument("--reconstruct", action="store_true",
                    help="[Mode D] TSDF Volume Integration → 메시 복원 → PLY + GLB")
    ap.add_argument("--tsdf_voxel", type=float, default=0.0005,
                    help="TSDF voxel 크기 (m, default: 0.0005 = 0.5mm)")
    ap.add_argument("--tsdf_trunc", type=float, default=0.003,
                    help="TSDF truncation 거리 (m, default: 0.003 = 3mm)")

    # 알고리즘
    ap.add_argument("--blade_dir", choices=["auto","pos","neg"], default="auto",
                    help="날 방향: auto=색상판별(기본) / pos=PCA+ / neg=PCA-")

    # 출력
    ap.add_argument("--out_dir", default=_DEFAULT_OUT_DIR)

    args = ap.parse_args()
    t0   = time.time()

    if args.scene_ply:
        mode = "B"
    elif args.reconstruct:
        mode = "D"
    elif args.depth_only:
        mode = "C"
    else:
        mode = "A"

    mode_desc = {
        "B": "B (scene_ply)",
        "D": "D (3D reconstruct → PLY+GLB)",
        "C": "C (HSV color, no SAM2)",
        "A": "A (GDino+SAM2)",
    }
    print(f"{'='*56}")
    print(f"  pose_from_sam3d.py  (cam0 좌표계)")
    print(f"  mode       : {mode_desc[mode]}")
    print(f"  blade_dir  : {args.blade_dir}")
    print(f"{'='*56}")

    # ── 씬 점군 취득 ─────────────────────────────────────────────
    reconstruct_info = None
    if mode == "B":
        print(f"\n[Step 1] PLY 로드")
        pts, rgb = load_ply(args.scene_ply)
        tag = os.path.splitext(os.path.basename(args.scene_ply))[0]
        if len(pts) < 50:
            raise RuntimeError("점군 부족 (<50)")
    elif mode in ("C", "D"):
        print(f"\n[Step 1] 캘리브레이션 로드")
        cams = discover_cameras(args.capture_dir)
        pad  = frame_pad(args.capture_dir, cams[0])
        K_m, D_m, ds_m = {}, {}, {}
        for ci in cams:
            K_m[ci], D_m[ci], ds_m[ci] = load_intrinsics(args.intrinsics_dir, ci)
        T_m = load_extrinsics(args.calib_dir, cams)
        print(f"\n[Step 2] HSV 색상 필터 물체 추출 (SAM2 없음)")
        pts, rgb = run_color_segmentation(args, cams, K_m, D_m, ds_m, T_m, pad)
        tag = f"frame{args.frame:06d}"

        if mode == "D":
            seg_pts = pts.copy()
            bbox_raw = (seg_pts.max(0) - seg_pts.min(0)) * 1000
            print(f"  세그멘테이션 점군: {len(seg_pts):,} pts  "
                  f"bbox: {bbox_raw[0]:.1f}x{bbox_raw[1]:.1f}x{bbox_raw[2]:.1f} mm")
            print(f"\n[Step 3] TSDF Volume Integration → PLY + GLB")
            pts, rgb, reconstruct_info = reconstruct_tsdf(
                args, cams, K_m, D_m, ds_m, T_m, pad,
                seg_pts, args.out_dir, tag,
                voxel_length=args.tsdf_voxel,
                sdf_trunc=args.tsdf_trunc,
            )
    else:
        print(f"\n[Step 1] 캘리브레이션 로드")
        cams = discover_cameras(args.capture_dir)
        pad  = frame_pad(args.capture_dir, cams[0])
        K_m, D_m, ds_m = {}, {}, {}
        for ci in cams:
            K_m[ci], D_m[ci], ds_m[ci] = load_intrinsics(args.intrinsics_dir, ci)
        T_m = load_extrinsics(args.calib_dir, cams)
        print(f"\n[Step 2] GDino+SAM2 검출 → 3D 융합")
        pts, rgb = run_detection(args, cams, K_m, D_m, ds_m, T_m, pad)
        tag = f"frame{args.frame:06d}"

    bbox = (pts.max(0) - pts.min(0)) * 1000
    print(f"  점군: {len(pts):,} pts  bbox: {bbox[0]:.1f}x{bbox[1]:.1f}x{bbox[2]:.1f} mm")

    # ── 포즈 추정 ─────────────────────────────────────────────────
    step_n = "4" if mode == "D" else ("2" if mode == "B" else "3")
    print(f"\n[Step {step_n}] PCA 포즈 추정")
    pose = estimate_pose_cam0(pts, rgb, blade_dir=args.blade_dir)

    # ── 출력 + 저장 ───────────────────────────────────────────────
    result = print_and_save(pose, args.out_dir, tag, time.time() - t0)

    # 점군 형태 시각화
    try:
        pc_vis_path = os.path.join(args.out_dir, f"pointcloud_shape_{tag}.png")
        visualize_pointcloud(pts, rgb, pose, pc_vis_path,
                           f"Point Cloud Shape ({len(pts):,} pts)")
    except Exception as e:
        print(f"  [WARN] 점군 시각화 실패: {e}")

    # 복원 정보 추가 저장
    if reconstruct_info:
        result["reconstruction"] = reconstruct_info
        json_path = os.path.join(args.out_dir, f"pose_cam0_{tag}.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  3D 복원 파일:")
        print(f"    PLY 점군 : {reconstruct_info['ply_pointcloud']}")
        print(f"    PLY 메시 : {reconstruct_info['ply_mesh']}")
        print(f"    GLB 메시 : {reconstruct_info['glb_mesh']}")
        print(f"    정점: {reconstruct_info['n_vertices']:,}  면: {reconstruct_info['n_faces']:,}")


if __name__ == "__main__":
    main()
