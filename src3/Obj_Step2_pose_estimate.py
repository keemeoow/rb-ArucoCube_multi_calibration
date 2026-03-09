#!/usr/bin/env python3
"""
Obj_Step2_pose_estimate.py — GLB 참조 모델 + ICP 기반 물체 6-DOF 포즈 추정
======================================================================
파이프라인:
  1. 참조 3D 모델 로드 (SAM 3D Objects 앱에서 다운로드한 GLB/PLY)
  2. 캘리브레이션 로드 (내부/외부 파라미터)
  3. 물체 세그멘테이션 (HSV 색상 필터 or SAM2)
  4. Depth → 3D 역투영 + 멀티뷰 점군 융합 (cam0 좌표계)
  5. ICP 정합: 참조 모델 → 관측 점군 → 6-DOF 포즈
  6. 결과 저장 (JSON + PLY + 시각화)

좌표계: cam0 (OpenCV: X-right, Y-down, Z-forward)

사용법:
  cd src3
  # GLB 참조 모델 + HSV 세그멘테이션 (기본)
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_model.glb

  # GLB 참조 모델 + SAM2 세그멘테이션
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_model.glb --seg_mode sam2

  # PLY 참조 모델도 가능
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_model.ply
"""

import os
import sys
import glob
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CAP_DIR = os.path.join(_THIS_DIR, "data/object_capture")
_DEFAULT_CAL_DIR = os.path.join(_THIS_DIR, "data/cube_session_01/calib_out_cube")
_DEFAULT_INT_DIR = os.path.join(_THIS_DIR, "data/_intrinsics")
_DEFAULT_OUT_DIR = os.path.join(_THIS_DIR, "Obj_Step2_output")


# ──────────────────────────────────────────────────────────────────
#  1. 참조 3D 모델 로드 (GLB / PLY)
# ──────────────────────────────────────────────────────────────────
def load_reference_model(path: str) -> "open3d.geometry.PointCloud":
    """GLB/PLY/OBJ 파일 → Open3D PointCloud (정규화: 중심=원점, 단위=미터)."""
    import open3d as o3d

    ext = os.path.splitext(path)[1].lower()

    if ext == ".ply":
        # PLY: 메시인지 점군인지 판별
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=30000)
            if mesh.has_vertex_colors():
                pcd.colors = o3d.utility.Vector3dVector(
                    np.asarray(mesh.vertex_colors)[:len(pcd.points)]
                    if len(mesh.vertex_colors) >= len(pcd.points)
                    else np.asarray(mesh.vertex_colors)
                )
        else:
            pcd = o3d.io.read_point_cloud(path)
    elif ext in (".glb", ".gltf", ".obj"):
        # trimesh로 GLB 로드 → Open3D 변환
        import trimesh
        scene = trimesh.load(path, force="scene")
        if isinstance(scene, trimesh.Scene):
            mesh_tm = scene.to_geometry() if hasattr(scene, 'to_geometry') else scene.dump(concatenate=True)
        else:
            mesh_tm = scene
        # trimesh → Open3D mesh → 점군 샘플링
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_tm.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_tm.faces)
        if mesh_tm.visual is not None and hasattr(mesh_tm.visual, 'vertex_colors'):
            vc = mesh_tm.visual.vertex_colors[:, :3] / 255.0
            mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(vc)
        mesh_o3d.compute_vertex_normals()
        pcd = mesh_o3d.sample_points_uniformly(number_of_points=30000)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise RuntimeError(f"참조 모델이 비어있음: {path}")

    # SAM 3D Objects는 Y-up (OpenGL) 좌표계 → cam0 Y-down (OpenCV)로 변환
    if ext in (".glb", ".gltf"):
        pts[:, 1] = -pts[:, 1]  # Y 반전
        pts[:, 2] = -pts[:, 2]  # Z 반전
        print(f"  [좌표계 변환] OpenGL(Y-up) → OpenCV(Y-down): Y,Z 반전")

    # 중심을 원점으로 이동
    centroid = pts.mean(axis=0)
    pts -= centroid
    pcd.points = o3d.utility.Vector3dVector(pts)

    # 스케일 확인 (SAM 3D Objects 출력은 보통 정규화 스케일)
    bbox_size = pts.max(axis=0) - pts.min(axis=0)
    print(f"  참조 모델: {len(pts):,} pts")
    print(f"  원본 bbox: {bbox_size[0]:.4f} x {bbox_size[1]:.4f} x {bbox_size[2]:.4f}")

    return pcd


def scale_reference_to_observation(ref_pcd, obs_pcd,
                                    ref_length_mm: float = None,
                                    manual_scale: float = None
                                    ) -> Tuple["open3d.geometry.PointCloud", float]:
    """참조 모델을 실제 크기에 맞게 스케일링.

    우선순위:
      1. manual_scale: 직접 지정한 스케일 값
      2. ref_length_mm: 실제 물체 길이(mm)로 역산
      3. 자동(bbox 비교): 관측 점군의 가장 긴 축 기준
    """
    import open3d as o3d

    ref_pts = np.asarray(ref_pcd.points)
    ref_extent = ref_pts.max(0) - ref_pts.min(0)
    ref_max = ref_extent.max()

    if ref_max < 1e-8:
        raise RuntimeError("참조 모델 크기가 0")

    if manual_scale is not None:
        scale = manual_scale
        print(f"  스케일: {scale:.6f} (수동 지정)")
    elif ref_length_mm is not None:
        # 참조 모델의 가장 긴 축 = ref_length_mm (m)
        scale = (ref_length_mm / 1000.0) / ref_max
        print(f"  스케일: {scale:.6f} (실제 길이 {ref_length_mm:.1f}mm 기준, "
              f"ref_max={ref_max:.4f})")
    else:
        obs_pts = np.asarray(obs_pcd.points)
        obs_extent = obs_pts.max(0) - obs_pts.min(0)
        obs_max = obs_extent.max()
        scale = obs_max / ref_max
        print(f"  스케일: {scale:.6f} (자동 bbox: ref_max={ref_max:.4f} → "
              f"obs_max={obs_max:.4f}m)")

    ref_pts_scaled = ref_pts * scale
    ref_scaled = o3d.geometry.PointCloud()
    ref_scaled.points = o3d.utility.Vector3dVector(ref_pts_scaled)
    if ref_pcd.has_colors():
        ref_scaled.colors = ref_pcd.colors
    if ref_pcd.has_normals():
        ref_scaled.normals = ref_pcd.normals

    scaled_extent = ref_pts_scaled.max(0) - ref_pts_scaled.min(0)
    print(f"  스케일 후 크기: {scaled_extent[0]*1000:.1f} x "
          f"{scaled_extent[1]*1000:.1f} x {scaled_extent[2]*1000:.1f} mm")
    return ref_scaled, scale


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
    return len(os.path.basename(files[0]).replace("rgb_", "").replace(".jpg", "")) if files else 6


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
        return np.empty((0, 3)), np.empty((0, 2))
    z, u, v = z[ok], ug[ok].astype(np.float64), vg[ok].astype(np.float64)
    und = cv2.undistortPoints(
        np.column_stack([u, v]).reshape(-1, 1, 2).astype(np.float64), K, D
    ).reshape(-1, 2)
    xyz = np.column_stack([und[:, 0] * z, und[:, 1] * z, z])
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
        bgr = cv2.imread(rp)
        dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            continue
        pts_ci, uvs = depth_to_pts(dep, masks[ci], K_m[ci], D_m[ci],
                                    ds_m[ci], z_min, z_max)
        if len(pts_ci) == 0:
            continue
        ui, vi = uvs[:, 0].astype(int), uvs[:, 1].astype(int)
        rgb = bgr[vi, ui][:, ::-1] / 255.0
        T = T_m[ci]
        pts_cam0 = pts_ci @ T[:3, :3].T + T[:3, 3]
        all_pts.append(pts_cam0)
        all_rgb.append(rgb)
        print(f"    cam{ci}: {len(pts_ci):,} pts → cam0")
    if not all_pts:
        return np.empty((0, 3)), None
    return np.concatenate(all_pts), np.concatenate(all_rgb)


def sor(pts, rgb=None, std_ratio=1.5):
    n = len(pts)
    if n < 10:
        return pts, rgb
    mask = np.ones(n, bool)
    for ax in range(3):
        v = pts[:, ax]
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        mask &= (v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)
    c = pts[mask].mean(0)
    d = np.linalg.norm(pts - c, axis=1)
    mu, sg = d[mask].mean(), d[mask].std()
    mask &= d < mu + std_ratio * sg
    print(f"    SOR: {n:,} → {mask.sum():,}")
    return pts[mask], (rgb[mask] if rgb is not None else None)


# ──────────────────────────────────────────────────────────────────
#  4. 세그멘테이션
# ──────────────────────────────────────────────────────────────────
def run_hsv_segmentation(args, cams, K_m, D_m, ds_m, T_m, pad):
    """HSV 색상 필터로 물체 세그멘테이션 (SAM2 불필요)."""
    h_lo, h_hi = args.hsv_h_range
    s_lo = args.hsv_s_min
    v_lo = args.hsv_v_min
    fid = f"{args.frame:0{pad}d}"

    # Stage 1: HSV 앵커 (노란 손잡이) → cam0 3D 중심
    anchor_pts_3d = []
    for ci in cams:
        rp = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid}.jpg")
        dp = os.path.join(args.capture_dir, f"cam{ci}", f"depth_{fid}.png")
        if not (os.path.exists(rp) and os.path.exists(dp)):
            continue
        bgr = cv2.imread(rp)
        dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            continue
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

    # Stage 2: 앵커 주변 구 크롭 + 전체 depth
    all_pts, all_rgb = [], []
    sphere_r = 0.12
    for ci in cams:
        rp = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid}.jpg")
        dp = os.path.join(args.capture_dir, f"cam{ci}", f"depth_{fid}.png")
        if not (os.path.exists(rp) and os.path.exists(dp)):
            continue
        bgr = cv2.imread(rp)
        dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            continue
        full_mask = np.ones(dep.shape[:2], dtype=bool)
        pts_ci, uvs = depth_to_pts(dep, full_mask, K_m[ci], D_m[ci],
                                    ds_m[ci], args.z_min, args.z_max)
        if len(pts_ci) == 0:
            continue
        T = T_m[ci]
        pts_cam0 = pts_ci @ T[:3, :3].T + T[:3, 3]
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
        raise RuntimeError("HSV 세그멘테이션 실패")

    pts = np.concatenate(all_pts)
    rgb = np.concatenate(all_rgb)

    # Stage 3: PCA 기반 타원체 크롭
    centered = pts - anchor_center
    cov = centered.T @ centered / len(centered)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]
    proj = centered @ evecs
    a_long = 0.100
    a_mid = 0.018
    a_short = 0.018
    ellip_dist = (proj[:, 0] / a_long) ** 2 + \
                 (proj[:, 1] / a_mid) ** 2 + \
                 (proj[:, 2] / a_short) ** 2
    in_ellipsoid = ellip_dist < 1.0
    pts = pts[in_ellipsoid]
    rgb = rgb[in_ellipsoid]
    print(f"    타원체 크롭: {in_ellipsoid.sum():,} pts")

    if len(pts) < 50:
        raise RuntimeError("세그멘테이션 결과 부족")

    # SOR + DBSCAN
    pts, rgb = sor(pts, rgb)
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=0.005, min_samples=10).fit(pts)
    labels = db.labels_
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) > 0:
        best_label = unique[np.argmax(counts)]
        mask_cl = labels == best_label
        print(f"    DBSCAN: {len(unique)} 클러스터, 최대={mask_cl.sum():,} pts")
        pts = pts[mask_cl]
        rgb = rgb[mask_cl] if rgb is not None else None

    return pts, rgb


def run_depth_roi_segmentation(args, cams, K_m, D_m, ds_m, T_m, pad):
    """
    딥러닝 모델 없이 depth 범위 + 테이블 평면 제거 + DBSCAN으로 물체 추출.

    원리:
      1. 전체 depth 이미지를 z_min~z_max 범위로 필터링
      2. 멀티뷰 융합 → cam0 좌표계 점군
      3. RANSAC으로 테이블 평면 제거
      4. 평면 위 점군에서 DBSCAN 가장 큰 클러스터 = 물체
    """
    import open3d as o3d
    from sklearn.cluster import DBSCAN

    fid = f"{args.frame:0{pad}d}"
    all_pts, all_rgb = [], []

    for ci in cams:
        rp = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid}.jpg")
        dp = os.path.join(args.capture_dir, f"cam{ci}", f"depth_{fid}.png")
        if not (os.path.exists(rp) and os.path.exists(dp)):
            continue
        bgr = cv2.imread(rp)
        dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if bgr is None or dep is None:
            continue

        full_mask = np.ones(dep.shape[:2], dtype=bool)
        pts_ci, uvs = depth_to_pts(dep, full_mask, K_m[ci], D_m[ci],
                                    ds_m[ci], args.z_min, args.z_max)
        if len(pts_ci) == 0:
            continue

        T = T_m[ci]
        pts_cam0 = pts_ci @ T[:3, :3].T + T[:3, 3]
        ui, vi = uvs[:, 0].astype(int), uvs[:, 1].astype(int)
        rgb_ci = bgr[vi, ui][:, ::-1] / 255.0
        all_pts.append(pts_cam0)
        all_rgb.append(rgb_ci)
        print(f"    cam{ci}: {len(pts_ci):,} pts → cam0")

    if not all_pts:
        raise RuntimeError("depth 점군 없음")

    pts = np.concatenate(all_pts)
    rgb = np.concatenate(all_rgb)
    print(f"    전체 점군: {len(pts):,} pts")

    # RANSAC 테이블 평면 제거
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd_o3d.segment_plane(
        distance_threshold=0.005,   # 5mm 이내 = 테이블
        ransac_n=3,
        num_iterations=1000,
    )
    inlier_mask = np.zeros(len(pts), dtype=bool)
    inlier_mask[inliers] = True
    above_plane = ~inlier_mask
    pts = pts[above_plane]
    rgb = rgb[above_plane]
    print(f"    테이블 제거 후: {len(pts):,} pts "
          f"(평면 법선: [{plane_model[0]:.2f},{plane_model[1]:.2f},{plane_model[2]:.2f}])")

    if len(pts) < 50:
        raise RuntimeError("평면 제거 후 점군 부족")

    # SOR
    pts, rgb = sor(pts, rgb)

    # DBSCAN — 가장 큰 클러스터 = 물체
    db = DBSCAN(eps=0.005, min_samples=10).fit(pts)
    labels = db.labels_
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        raise RuntimeError("DBSCAN 클러스터 없음")

    best_label = unique[np.argmax(counts)]
    mask_cl = labels == best_label
    print(f"    DBSCAN: {len(unique)} 클러스터, 최대={mask_cl.sum():,} pts 채택")
    pts = pts[mask_cl]
    rgb = rgb[mask_cl]

    return pts, rgb


def run_sam2_segmentation(args, cams, K_m, D_m, ds_m, T_m, pad):
    """GroundingDINO + SAM2로 물체 세그멘테이션."""
    sam_dir = os.path.join(_THIS_DIR,
                           "Obj_Step2-(1)_pose_estimate_grounding_sam")
    if os.path.isdir(sam_dir) and sam_dir not in sys.path:
        sys.path.insert(0, sam_dir)
    from_mod = "Obj_Step2-(1)_pose_grounding_sam"
    mod = __import__(from_mod.replace("-", "_").replace("(", "").replace(")", ""))
    load_grounding_dino = getattr(mod, "load_grounding_dino", None)
    load_sam2 = getattr(mod, "load_sam2", None)
    detect_and_segment = getattr(mod, "detect_and_segment_object", None)

    if load_grounding_dino is None:
        # fallback: 직접 import
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise RuntimeError("SAM2 패키지가 설치되지 않음. --seg_mode hsv 를 사용하세요.")

        print(f"  GroundingDINO + SAM2 로딩 (device={args.device})...")
        gp = AutoProcessor.from_pretrained(args.gdino_model)
        gm = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_model)
        import torch
        gm = gm.to(args.device)
        sp = SAM2ImagePredictor(build_sam2(args.sam2_config, args.sam2_checkpoint))
        sp.model = sp.model.to(args.device)
    else:
        print(f"  GroundingDINO + SAM2 로딩 (device={args.device})...")
        gp, gm = load_grounding_dino(args.gdino_model, args.device)
        sp = load_sam2(args.sam2_checkpoint, args.sam2_config, args.device)

    fid = f"{args.frame:0{pad}d}"
    masks: Dict[int, np.ndarray] = {}
    for ci in cams:
        rp = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid}.jpg")
        if not os.path.exists(rp):
            continue
        bgr = cv2.imread(rp)
        rgb_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image as PILImage
        import torch
        pil_img = PILImage.fromarray(rgb_img)
        inputs = gp(images=pil_img, text=args.text_prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = gm(**inputs)
        results = gp.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[pil_img.size[::-1]]
        )[0]
        if len(results["boxes"]) == 0:
            print(f"    cam{ci}: 검출 없음")
            continue
        best_idx = results["scores"].argmax().item()
        box = results["boxes"][best_idx].cpu().numpy()
        score = results["scores"][best_idx].item()
        print(f"    cam{ci}: score={score:.3f}")
        sp.set_image(rgb_img)
        box_torch = torch.tensor(box, dtype=torch.float32, device=args.device).unsqueeze(0)
        pred_masks, _, _ = sp.predict(box=box_torch.cpu().numpy(), multimask_output=False)
        masks[ci] = pred_masks[0].astype(bool)

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
#  5. ICP 정합 → 6-DOF 포즈 추정
# ──────────────────────────────────────────────────────────────────
def initial_alignment_pca(ref_pts, obs_pts):
    """PCA 기반 초기 정렬 (ICP 수렴을 위한 초기값)."""
    def pca_axes(pts):
        c = pts.mean(axis=0)
        p = pts - c
        cov = p.T @ p / len(p)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        return c, evecs[:, order], evals[order]

    c_ref, axes_ref, _ = pca_axes(ref_pts)
    c_obs, axes_obs, _ = pca_axes(obs_pts)

    # PCA 축 정렬: R * ref_axes = obs_axes
    # det 보정으로 오른손 좌표계 보장
    if np.linalg.det(axes_ref) < 0:
        axes_ref[:, 2] *= -1
    if np.linalg.det(axes_obs) < 0:
        axes_obs[:, 2] *= -1

    R = axes_obs @ axes_ref.T
    if np.linalg.det(R) < 0:
        axes_obs[:, 2] *= -1
        R = axes_obs @ axes_ref.T

    t = c_obs - R @ c_ref

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def estimate_pose_icp(ref_pcd, obs_pcd, max_correspondence_dist=0.01,
                      n_icp_iterations=100):
    """
    ICP 정합으로 6-DOF 포즈 추정.
    Returns: (T_4x4, fitness, rmse)
      T_4x4: 참조 모델 → cam0 좌표계 변환 (4x4 행렬)
    """
    import open3d as o3d

    ref_pts = np.asarray(ref_pcd.points)
    obs_pts = np.asarray(obs_pcd.points)

    # PCA 기반 초기 정렬
    T_init = initial_alignment_pca(ref_pts, obs_pts)
    print(f"    PCA 초기 정렬 완료")

    # 노말 추정 (Point-to-Plane ICP용)
    ref_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    obs_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    # 다단계 ICP: coarse → fine
    stages = [
        (max_correspondence_dist * 4, 30, "coarse"),
        (max_correspondence_dist * 2, 30, "medium"),
        (max_correspondence_dist, n_icp_iterations, "fine"),
    ]

    T_current = T_init
    for dist, iters, name in stages:
        result = o3d.pipelines.registration.registration_icp(
            ref_pcd, obs_pcd,
            max_correspondence_distance=dist,
            init=T_current,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iters,
                relative_fitness=1e-6,
                relative_rmse=1e-6,
            ),
        )
        T_current = result.transformation
        print(f"    ICP [{name}] dist={dist*1000:.1f}mm: "
              f"fitness={result.fitness:.4f}, RMSE={result.inlier_rmse*1000:.3f}mm")

    T_final = T_current
    fitness = result.fitness
    rmse = result.inlier_rmse

    # 180도 모호성 검증: PCA 주축 반전 후 ICP 재시도
    T_init_flip = T_init.copy()
    T_init_flip[:3, :3] = T_init[:3, :3] @ np.diag([-1, -1, 1])
    T_flip = T_init_flip
    for dist, iters, name in stages:
        result_flip = o3d.pipelines.registration.registration_icp(
            ref_pcd, obs_pcd,
            max_correspondence_distance=dist,
            init=T_flip,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=iters,
                relative_fitness=1e-6,
                relative_rmse=1e-6,
            ),
        )
        T_flip = result_flip.transformation

    print(f"    ICP [flip]  fitness={result_flip.fitness:.4f}, "
          f"RMSE={result_flip.inlier_rmse*1000:.3f}mm")

    if result_flip.fitness > fitness * 1.05 or \
       (result_flip.fitness >= fitness * 0.95 and result_flip.inlier_rmse < rmse):
        print(f"    → 180도 반전 결과 채택")
        T_final = T_flip
        fitness = result_flip.fitness
        rmse = result_flip.inlier_rmse

    return T_final, fitness, rmse


# ──────────────────────────────────────────────────────────────────
#  6. 회전 변환 유틸
# ──────────────────────────────────────────────────────────────────
def R_to_euler(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def R_to_quat(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


# ──────────────────────────────────────────────────────────────────
#  7. 시각화
# ──────────────────────────────────────────────────────────────────
def visualize_result(obs_pts, obs_rgb, ref_pts_transformed, pose, out_path, title):
    """관측 점군 + ICP 정합된 참조 모델 + 포즈 축 시각화."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(18, 6))
    obs_mm = obs_pts * 1000
    ref_mm = ref_pts_transformed * 1000

    R = pose["R"]
    c_mm = pose["position_mm"]

    views = [
        (25, -60, "Perspective"),
        (90, -90, "Top (XZ)"),
        (0, -90, "Front (XY)"),
    ]

    for idx, (elev, azim, subtitle) in enumerate(views):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        ax.set_title(subtitle, fontsize=10)

        # 관측 점군 (서브샘플링)
        n_obs = len(obs_mm)
        if n_obs > 5000:
            sel = np.random.default_rng(42).choice(n_obs, 5000, replace=False)
        else:
            sel = np.arange(n_obs)
        if obs_rgb is not None:
            rgb_clip = np.clip(obs_rgb[sel], 0.0, 1.0)
            ax.scatter(obs_mm[sel, 0], obs_mm[sel, 1], obs_mm[sel, 2],
                       c=rgb_clip, s=0.5, alpha=0.4, label="observed")
        else:
            ax.scatter(obs_mm[sel, 0], obs_mm[sel, 1], obs_mm[sel, 2],
                       c="steelblue", s=0.5, alpha=0.4, label="observed")

        # 참조 모델 (빨간색)
        n_ref = len(ref_mm)
        if n_ref > 3000:
            sel_r = np.random.default_rng(7).choice(n_ref, 3000, replace=False)
        else:
            sel_r = np.arange(n_ref)
        ax.scatter(ref_mm[sel_r, 0], ref_mm[sel_r, 1], ref_mm[sel_r, 2],
                   c="#e74c3c", s=0.5, alpha=0.3, label="reference (ICP)")

        # 포즈 축
        axis_len = 30.0
        colors = ["#e74c3c", "#27ae60", "#2980b9"]
        labels_ax = ["X", "Y", "Z"]
        for ai in range(3):
            v = R[:, ai] * axis_len
            ax.quiver(c_mm[0], c_mm[1], c_mm[2], v[0], v[1], v[2],
                      color=colors[ai], linewidth=2.5, arrow_length_ratio=0.12)

        # cam0 원점 축
        for ai in range(3):
            v = np.eye(3)[:, ai] * 50
            ax.quiver(0, 0, 0, v[0], v[1], v[2],
                      color=colors[ai], linewidth=1.5, arrow_length_ratio=0.08, alpha=0.4)

        ax.set_xlabel("X mm", fontsize=7)
        ax.set_ylabel("Y mm", fontsize=7)
        ax.set_zlabel("Z mm", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=elev, azim=azim)

        all_mm = np.vstack([obs_mm, ref_mm])
        mid = all_mm.mean(axis=0)
        rng = max((all_mm.max(0) - all_mm.min(0)).max() / 2 * 1.3, 40)
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)

    eu = pose["euler_xyz_deg"]
    fig.suptitle(title, fontsize=11, y=0.98)
    fig.text(0.5, 0.01,
             f"Position: ({c_mm[0]:.1f}, {c_mm[1]:.1f}, {c_mm[2]:.1f}) mm  |  "
             f"Euler: ({eu[0]:.1f}, {eu[1]:.1f}, {eu[2]:.1f}) deg  |  "
             f"ICP fitness: {pose['icp_fitness']:.4f}, RMSE: {pose['icp_rmse_mm']:.2f}mm",
             ha="center", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ecf0f1", ec="#bdc3c7"))
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [저장] {out_path}")


def visualize_pose_cam0(pose, out_path, title):
    """cam0 + 객체 포즈를 3D 좌표 프레임으로 시각화."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    R = pose["R"]
    pos_mm = np.array(pose["position_mm"])
    euler = pose["euler_xyz_deg"]
    quat = pose["quaternion_wxyz"]

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

    draw_axes(np.eye(3), np.zeros(3), 70, "cam0 (ref)", lw=3.0)
    draw_axes(R, pos_mm, 55, "Object", lw=3.5)

    ax.plot3D([0, pos_mm[0]], [0, pos_mm[1]], [0, pos_mm[2]],
              ":", color="#c0392b", lw=1.0, alpha=0.35)

    ax.text(pos_mm[0], pos_mm[1] - 45, pos_mm[2] + 52,
            f"({pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}) mm",
            fontsize=8, color="#2c3e50", ha="center")
    ax.text(pos_mm[0], pos_mm[1] - 45, pos_mm[2] + 35,
            f"euler ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg",
            fontsize=8, color="#2c3e50", ha="center")

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
#  8. 결과 저장
# ──────────────────────────────────────────────────────────────────
def save_results(pose, obs_pts, obs_rgb, ref_pts_transformed,
                 out_dir, tag, elapsed):
    import open3d as o3d

    os.makedirs(out_dir, exist_ok=True)

    # --- robust normalize (list/np/None 모두 처리) ---
    R = np.asarray(pose.get("R"), dtype=float) if pose.get("R") is not None else None
    pos_mm = np.asarray(pose.get("position_mm", [0, 0, 0]), dtype=float).reshape(-1)

    eu_raw = pose.get("euler_xyz_deg", None)
    q_raw  = pose.get("quaternion_wxyz", None)

    eu_list = None if eu_raw is None else np.asarray(eu_raw, dtype=float).reshape(-1).tolist()
    q_list  = None if q_raw  is None else np.asarray(q_raw,  dtype=float).reshape(-1).tolist()

    # 콘솔 출력
    print(f"\n{'='*56}")
    print(f"  RESULT — cam0 (X-right, Y-down, Z-forward)")
    print(f"  method: GLB reference + ICP")
    print(f"{'='*56}")
    print(f"  Position   (mm): ({pos_mm[0]:+.1f}, {pos_mm[1]:+.1f}, {pos_mm[2]:+.1f})")

    if eu_list is not None and len(eu_list) >= 3:
        print(f"  Euler XYZ (deg): ({eu_list[0]:+.1f}, {eu_list[1]:+.1f}, {eu_list[2]:+.1f})")
    else:
        print(f"  Euler XYZ (deg): N/A")

    if q_list is not None and len(q_list) >= 4:
        print(f"  Quat wxyz      : ({q_list[0]:.5f}, {q_list[1]:.5f}, {q_list[2]:.5f}, {q_list[3]:.5f})")
    else:
        print(f"  Quat wxyz      : N/A")

    print(f"  ICP fitness    : {pose['icp_fitness']:.4f}")
    print(f"  ICP RMSE  (mm) : {pose['icp_rmse_mm']:.3f}")
    print(f"  Scale factor   : {pose['scale_factor']:.6f}")
    print(f"  소요시간       : {elapsed:.1f}s")
    print(f"{'='*56}")

    # JSON 저장
    result = {
        "coordinate_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
        "method": "GLB_reference_ICP",
        "reference_model": pose["ref_model_path"],
        "position_mm": pos_mm.tolist(),
        "euler_xyz_deg": eu_list,
        "quaternion_wxyz": q_list,
        "rotation_matrix": (None if R is None else R.tolist()),
        "icp_fitness": float(pose["icp_fitness"]),
        "icp_rmse_mm": float(pose["icp_rmse_mm"]),
        "scale_factor": float(pose["scale_factor"]),
        "elapsed_sec": round(float(elapsed), 2),
    }

    json_path = os.path.join(out_dir, f"pose_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [저장] {json_path}")

    # 관측 점군 PLY
    obs_ply_path = os.path.join(out_dir, f"observed_pointcloud_{tag}.ply")
    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(np.asarray(obs_pts))
    if obs_rgb is not None:
        obs_pcd.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(obs_rgb), 0, 1))
    o3d.io.write_point_cloud(obs_ply_path, obs_pcd)
    print(f"  [저장] {obs_ply_path}")

    # ICP 정합 결과 PLY (참조 모델 변환 후)
    ref_ply_path = os.path.join(out_dir, f"aligned_reference_{tag}.ply")
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(np.asarray(ref_pts_transformed))
    o3d.io.write_point_cloud(ref_ply_path, ref_pcd)
    print(f"  [저장] {ref_ply_path}")

    # 시각화 1: 점군 + 정합 결과
    try:
        vis_path = os.path.join(out_dir, f"icp_alignment_{tag}.png")
        visualize_result(obs_pts, obs_rgb, ref_pts_transformed,
                        pose, vis_path,
                        f"ICP Alignment — {tag}")
    except Exception as e:
        print(f"  [WARN] 정합 시각화 실패: {e}")

    # 시각화 2: 포즈 축
    try:
        pose_vis_path = os.path.join(out_dir, f"pose_cam0_{tag}.png")
        visualize_pose_cam0(pose, pose_vis_path,
                           "GLB Reference + ICP Pose (cam0, mm)")
        print(f"  [저장] {pose_vis_path}")
    except Exception as e:
        print(f"  [WARN] 포즈 시각화 실패: {e}")

    return result


# ──────────────────────────────────────────────────────────────────
#  9. Main
# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="GLB 참조 모델 + ICP 기반 물체 6-DOF 포즈 추정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # GLB 참조 모델 + HSV 세그멘테이션 (기본)
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_model.glb

  # SAM2 세그멘테이션 사용
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_model.glb --seg_mode sam2

  # PLY 참조 모델
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_model.ply

  # ICP 파라미터 조정
  python Obj_Step2_pose_estimate.py --frame 3 --ref_model data/reference_model.glb --icp_dist 15
""")

    # 참조 모델
    ap.add_argument("--ref_model", required=True,
                    help="참조 3D 모델 경로 (GLB/PLY/OBJ)")

    # 입력
    ap.add_argument("--capture_dir", default=_DEFAULT_CAP_DIR)
    ap.add_argument("--calib_dir", default=_DEFAULT_CAL_DIR)
    ap.add_argument("--intrinsics_dir", default=_DEFAULT_INT_DIR)
    ap.add_argument("--frame", type=int, default=3)
    ap.add_argument("--z_min", type=float, default=0.1)
    ap.add_argument("--z_max", type=float, default=1.5)

    # 세그멘테이션 모드
    ap.add_argument("--seg_mode", choices=["hsv", "sam2", "depth_roi"], default="hsv",
                    help="세그멘테이션 방식: hsv(기본) / sam2 / depth_roi(모델 없음)")

    # HSV 파라미터
    ap.add_argument("--hsv_h_range", type=int, nargs=2, default=[15, 35])
    ap.add_argument("--hsv_s_min", type=int, default=80)
    ap.add_argument("--hsv_v_min", type=int, default=80)
    ap.add_argument("--min_component_area", type=int, default=500)

    # SAM2 파라미터
    ap.add_argument("--text_prompt", default="utility knife.")
    ap.add_argument("--gdino_model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam2_checkpoint",
                    default=os.path.join(_THIS_DIR, "checkpoints/sam2.1_hiera_large.pt"))
    ap.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--box_threshold", type=float, default=0.15)
    ap.add_argument("--text_threshold", type=float, default=0.15)

    # 스케일 지정 (우선순위: --scale > --ref_length_mm > 자동)
    ap.add_argument("--ref_length_mm", type=float, default=None,
                    help="물체 실제 길이 mm (예: --ref_length_mm 150). "
                         "지정 시 bbox 자동 스케일 대신 이 값으로 스케일 고정")
    ap.add_argument("--scale", type=float, default=None,
                    help="참조 모델 스케일 직접 지정 (예: --scale 0.1501). "
                         "지정 시 자동 스케일 무시")

    # ICP 파라미터
    ap.add_argument("--icp_dist", type=float, default=10.0,
                    help="ICP max correspondence distance (mm, default: 10)")
    ap.add_argument("--icp_iters", type=int, default=100,
                    help="ICP fine 단계 최대 반복 (default: 100)")

    # 출력
    ap.add_argument("--out_dir", default=_DEFAULT_OUT_DIR)

    args = ap.parse_args()
    t0 = time.time()

    print(f"{'='*56}")
    print(f"  Obj_Step2_pose_estimate.py")
    print(f"  method   : GLB Reference + ICP")
    print(f"  ref_model: {args.ref_model}")
    print(f"  seg_mode : {args.seg_mode}")
    print(f"  frame    : {args.frame}")
    print(f"{'='*56}")

    # ── Step 1: 참조 모델 로드 ─────────────────────────────────────
    print(f"\n[Step 1] 참조 모델 로드: {args.ref_model}")
    import open3d as o3d
    ref_pcd = load_reference_model(args.ref_model)

    # ── Step 2: 캘리브레이션 로드 ──────────────────────────────────
    print(f"\n[Step 2] 캘리브레이션 로드")
    cams = discover_cameras(args.capture_dir)
    pad = frame_pad(args.capture_dir, cams[0])
    K_m, D_m, ds_m = {}, {}, {}
    for ci in cams:
        K_m[ci], D_m[ci], ds_m[ci] = load_intrinsics(args.intrinsics_dir, ci)
    T_m = load_extrinsics(args.calib_dir, cams)
    print(f"  카메라: {cams}")

    # ── Step 3: 세그멘테이션 + 멀티뷰 융합 ─────────────────────────
    print(f"\n[Step 3] 세그멘테이션 ({args.seg_mode})")
    if args.seg_mode == "hsv":
        obs_pts, obs_rgb = run_hsv_segmentation(args, cams, K_m, D_m, ds_m, T_m, pad)
    elif args.seg_mode == "depth_roi":
        obs_pts, obs_rgb = run_depth_roi_segmentation(args, cams, K_m, D_m, ds_m, T_m, pad)
    else:
        obs_pts, obs_rgb = run_sam2_segmentation(args, cams, K_m, D_m, ds_m, T_m, pad)

    obs_bbox = (obs_pts.max(0) - obs_pts.min(0)) * 1000
    print(f"  관측 점군: {len(obs_pts):,} pts")
    print(f"  bbox: {obs_bbox[0]:.1f} x {obs_bbox[1]:.1f} x {obs_bbox[2]:.1f} mm")

    # Open3D 점군 객체 생성
    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(obs_pts)
    if obs_rgb is not None:
        obs_pcd.colors = o3d.utility.Vector3dVector(np.clip(obs_rgb, 0, 1))

    # ── Step 4: 스케일 맞추기 + ICP 정합 ───────────────────────────
    print(f"\n[Step 4] ICP 정합")
    ref_scaled, scale = scale_reference_to_observation(
        ref_pcd, obs_pcd,
        ref_length_mm=args.ref_length_mm,
        manual_scale=args.scale,
    )

    icp_dist_m = args.icp_dist / 1000.0
    T_icp, fitness, rmse = estimate_pose_icp(
        ref_scaled, obs_pcd,
        max_correspondence_dist=icp_dist_m,
        n_icp_iterations=args.icp_iters,
    )

    # 포즈 추출
    R = T_icp[:3, :3]
    t = T_icp[:3, 3]
    eu = R_to_euler(R)
    q = R_to_quat(R)

    pose = {
        "R": R,
        "position_mm": t * 1000,
        "euler_xyz_deg": (None if eu is None else np.asarray(eu, dtype=float).tolist()),
        "quaternion_wxyz": (None if q is None else np.asarray(q, dtype=float).tolist()),
        "icp_fitness": float(fitness),
        "icp_rmse_mm": float(rmse * 1000),
        "scale_factor": float(scale),
        "ref_model_path": os.path.abspath(args.ref_model),
    }

    # 참조 모델 변환 (시각화용)
    ref_pts_transformed = np.asarray(ref_scaled.points) @ R.T + t

    # ── Step 5: 결과 저장 ──────────────────────────────────────────
    tag = f"frame{args.frame:06d}"
    out_dir = os.path.join(args.out_dir, f"output_{tag}")
    result = save_results(pose, obs_pts, obs_rgb, ref_pts_transformed,
                         out_dir, tag, time.time() - t0)

    print(f"\n완료!")


if __name__ == "__main__":
    main()
