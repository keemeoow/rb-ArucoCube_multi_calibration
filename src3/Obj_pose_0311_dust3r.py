#!/usr/bin/env python3
"""
Obj_pose_0311_dust3r.py
========================
DUSt3R 기반 멀티뷰 3D 재구성 + 6-DOF 포즈 추정 (Object-agnostic)

색상 기반 세그멘테이션 없이 DUSt3R의 학습된 stereo matching으로
멀티뷰 RGB 이미지에서 직접 3D 점군을 복원하고, 기존 카메라 캘리브레이션
(intrinsics, T_C0_C1, T_C0_C2)을 활용하여 cam0 좌표계로 통합합니다.

파이프라인:
  1. DUSt3R로 이미지 쌍 → 3D pointmap 추론
  2. Global alignment (known intrinsics + known poses 활용)
  3. 깊이 맵 + confidence → 3D 점군 추출
  4. 관심 영역 크롭 (depth range + DBSCAN)
  5. PCA 6-DOF 포즈 추정 + 3-vote blade direction
  6. TSDF 메시 복원 (optional)
  7. 출력: pose JSON/NPZ, PLY, GLB, OBJ (모두 cam0/OpenCV 좌표계)

사용법:
  python Obj_pose_0311_dust3r.py --frame 3
  python Obj_pose_0311_dust3r.py --frame 3 --image_size 512
  python Obj_pose_0311_dust3r.py --frame 3 --no_known_poses  # DUSt3R가 포즈도 추정
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# DUSt3R 경로 추가
_DUST3R_ROOT = Path(__file__).resolve().parent.parent.parent / "dust3r"
if _DUST3R_ROOT.exists():
    sys.path.insert(0, str(_DUST3R_ROOT))

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent


def _pick_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


_DEFAULT_CAP_DIR = _pick_existing(
    _PROJECT_ROOT / "data/object_capture",
    _THIS_DIR / "data/object_capture",
    _THIS_DIR / "data/cube_session_01",
)
_DEFAULT_CAL_DIR = _pick_existing(
    _PROJECT_ROOT / "data/cube_session_01/calib_out_cube",
    _THIS_DIR / "data/cube_session_01/calib_out_cube",
)
_DEFAULT_INT_DIR = _pick_existing(
    _PROJECT_ROOT / "data/_intrinsics",
    _THIS_DIR / "data/_intrinsics",
)
_DEFAULT_OUT_DIR = _THIS_DIR / "Obj_pose_0311_dust3r_output"


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
#  2. DUSt3R 추론 + Global Alignment
# ======================================================================

def resolve_device(device: str) -> str:
    import torch

    d = device.lower()
    if d == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                _ = torch.empty(1, device="mps")
                return "mps"
            except Exception:
                pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if d == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                _ = torch.empty(1, device="mps")
                return "mps"
            except Exception as e:
                print(f"[DUSt3R] MPS 초기화 실패, CPU로 폴백: {e}")
        else:
            print("[DUSt3R] MPS 사용 불가, CPU로 폴백")
        return "cpu"

    if d == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("[DUSt3R] CUDA 사용 불가, CPU로 폴백")
        return "cpu"

    return "cpu"


def run_dust3r(image_paths: List[str],
               known_intrinsics: Optional[List[np.ndarray]] = None,
               known_poses: Optional[List[np.ndarray]] = None,
               image_size: int = 512,
               device: str = "auto",
               niter: int = 300) -> dict:
    """
    DUSt3R 추론 → global alignment → 3D 점군 + 깊이 맵 + 카메라 포즈.

    Args:
        image_paths: RGB 이미지 경로 리스트
        known_intrinsics: [K0, K1, ...] 3x3 카메라 내부 행렬 (옵션)
        known_poses: [T_w_c0, T_w_c1, ...] 4x4 cam-to-world 행렬 (옵션)
        image_size: DUSt3R 입력 해상도
        device: 'mps', 'cuda', 'cpu'
        niter: global alignment 반복 횟수

    Returns:
        dict with keys: pts3d, colors, confidence, im_poses, focals, imgs
    """
    import torch
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.utils.device import to_numpy

    device = resolve_device(device)
    print(f"\n[DUSt3R] 모델 로드 중 (device={device})...")
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")

    # MPS 호환성: float32 강제
    if device == "mps":
        model = model.float()
    model = model.to(device)
    model.eval()

    print(f"[DUSt3R] 이미지 {len(image_paths)}장 로드 (size={image_size})...")
    images = load_images(image_paths, size=image_size, verbose=False)

    # 모든 이미지 쌍 생성
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    print(f"[DUSt3R] {len(pairs)} 이미지 쌍 추론 중...")

    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=True)

    # Global alignment mode
    n_imgs = len(images)
    mode = GlobalAlignerMode.PointCloudOptimizer if n_imgs > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=True)

    # Known intrinsics 설정
    if known_intrinsics is not None:
        # DUSt3R는 리사이즈된 이미지 기준 focal 사용
        # 원본 K에서 focal 추출 후 리사이즈 비율 적용
        focals = []
        pps = []
        for i, K in enumerate(known_intrinsics):
            orig_h, orig_w = cv2.imread(image_paths[i]).shape[:2]
            # DUSt3R 리사이즈 비율 계산
            img_tensor = images[i]['img']
            dust3r_h, dust3r_w = img_tensor.shape[-2:]
            sx = dust3r_w / orig_w
            sy = dust3r_h / orig_h
            fx_scaled = K[0, 0] * sx
            fy_scaled = K[1, 1] * sy
            focal = (fx_scaled + fy_scaled) / 2.0
            pp_x = K[0, 2] * sx
            pp_y = K[1, 2] * sy
            focals.append(focal)
            pps.append([pp_x, pp_y])
        if hasattr(scene, 'preset_focal'):
            scene.preset_focal(focals)
        if hasattr(scene, 'preset_principal_point'):
            import torch as _t
            try:
                scene.preset_principal_point(
                    [_t.tensor(pp, dtype=_t.float32) for pp in pps]
                )
            except AssertionError as e:
                print(f"[DUSt3R] principal point preset 건너뜀: {e}")

    # Known poses 설정 (cam-to-world)
    if known_poses is not None and hasattr(scene, 'preset_pose'):
        import torch as _t
        scene.preset_pose(
            [_t.tensor(p, dtype=_t.float32) for p in known_poses]
        )
        init_mode = 'known_poses'
    else:
        init_mode = 'mst'

    print(f"[DUSt3R] Global alignment (init={init_mode}, niter={niter})...")
    loss = scene.compute_global_alignment(init=init_mode, niter=niter)
    print(f"[DUSt3R] Alignment 완료, loss={loss:.6f}")

    # 결과 추출
    pts3d_list = scene.get_pts3d()       # list of (H, W, 3) tensors, world frame
    confidence = scene.get_conf()         # list of (H, W) tensors
    im_poses = to_numpy(scene.get_im_poses())  # (N, 4, 4) cam-to-world
    focals = to_numpy(scene.get_focals())       # (N,)

    pts3d_np = [to_numpy(p) for p in pts3d_list]
    conf_np = [to_numpy(c) for c in confidence]

    # 이미지 색상 추출
    imgs_np = []
    for img_dict in images:
        # img shape: (1, 3, H, W) → (H, W, 3)
        img_t = img_dict['img']
        if isinstance(img_t, torch.Tensor):
            img_np = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            img_np = np.array(img_t).squeeze(0).transpose(1, 2, 0)
        # 정규화 복원 (DUSt3R는 ImageNet normalization 사용)
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        imgs_np.append(img_np)

    return {
        'pts3d': pts3d_np,       # list of (H, W, 3) in world frame
        'confidence': conf_np,    # list of (H, W)
        'im_poses': im_poses,     # (N, 4, 4) cam-to-world
        'focals': focals,          # (N,)
        'imgs': imgs_np,           # list of (H, W, 3) RGB [0,1]
        'loss': float(loss),
    }


# ======================================================================
#  3. DUSt3R 결과 → cam0 좌표계 점군 변환
# ======================================================================

def dust3r_to_cam0(dust3r_result: dict,
                   T_C0_Ci: Dict[int, np.ndarray],
                   use_known_poses: bool = True,
                   conf_threshold: float = 0.8,
                   z_min: float = 0.05,
                   z_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    DUSt3R 3D 점군을 cam0 좌표계로 변환.

    DUSt3R의 world frame은 global alignment 결과이므로:
    - known_poses 사용 시: world = cam0 좌표계
    - known_poses 미사용 시: DUSt3R가 추정한 arbitrary world frame

    Returns:
        pts_cam0 (Nx3), colors (Nx3)
    """
    pts3d_list = dust3r_result['pts3d']
    conf_list = dust3r_result['confidence']
    imgs_list = dust3r_result['imgs']
    im_poses = dust3r_result['im_poses']

    all_pts = []
    all_rgb = []

    for i, (pts, conf, img) in enumerate(zip(pts3d_list, conf_list, imgs_list)):
        H, W = pts.shape[:2]
        pts_flat = pts.reshape(-1, 3)
        conf_flat = conf.reshape(-1)
        rgb_flat = img.reshape(-1, 3)

        # confidence 필터
        mask = conf_flat > conf_threshold
        if not np.any(mask):
            finite_conf = conf_flat[np.isfinite(conf_flat)]
            if finite_conf.size == 0:
                continue
            auto_thr = float(np.percentile(finite_conf, 70))
            mask = conf_flat > auto_thr
            print(f"    cam{i}: conf_threshold={conf_threshold:.2f}로 0점 -> auto={auto_thr:.2f} 사용")

        # NaN/Inf 제거
        valid = np.isfinite(pts_flat).all(axis=1) & mask
        pts_sel = pts_flat[valid]
        rgb_sel = rgb_flat[valid]

        if len(pts_sel) == 0:
            continue

        if use_known_poses:
            # DUSt3R world = cam0 (known_poses가 T_C0_Ci이므로)
            pts_cam0 = pts_sel
        else:
            # DUSt3R world → cam_i → cam0 변환
            # im_poses[i] = cam_i_to_world (DUSt3R arbitrary world)
            # world → cam_i: inv(im_poses[i])
            T_w2ci = np.linalg.inv(im_poses[i])
            pts_ci = pts_sel @ T_w2ci[:3, :3].T + T_w2ci[:3, 3]
            # cam_i → cam0
            T = T_C0_Ci.get(i, np.eye(4))
            pts_cam0 = pts_ci @ T[:3, :3].T + T[:3, 3]

        # depth range 필터 (cam0 Z축 기준)
        z_mask = (pts_cam0[:, 2] > z_min) & (pts_cam0[:, 2] < z_max)
        pts_cam0 = pts_cam0[z_mask]
        rgb_sel = rgb_sel[z_mask]

        all_pts.append(pts_cam0)
        all_rgb.append(rgb_sel)
        print(f"    cam{i}: {len(pts_cam0):,} pts (conf>{conf_threshold:.1f})")

    if not all_pts:
        raise RuntimeError("유효한 3D 점 없음")

    return np.concatenate(all_pts), np.concatenate(all_rgb)


# ======================================================================
#  4. 관심 영역 추출 (DBSCAN)
# ======================================================================

def extract_object_region(pts: np.ndarray, rgb: np.ndarray,
                          anchor_center: Optional[np.ndarray] = None,
                          sphere_r: float = 0.12,
                          dbscan_eps: float = 0.005,
                          dbscan_min: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """DBSCAN 클러스터링으로 주요 물체 영역 추출."""

    # 앵커 기반 구 크롭
    if anchor_center is not None:
        dist = np.linalg.norm(pts - anchor_center, axis=1)
        inside = dist < sphere_r
        if inside.sum() > 50:
            pts, rgb = pts[inside], rgb[inside]
            print(f"    구 크롭 (r={sphere_r*1000:.0f}mm): {len(pts):,} pts")

    # SOR (Statistical Outlier Removal)
    pts, rgb = _sor(pts, rgb)

    # DBSCAN
    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min).fit(pts).labels_
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) > 0:
        best = unique[np.argmax(counts)]
        mask = labels == best
        pts, rgb = pts[mask], rgb[mask]
        print(f"    DBSCAN: {len(unique)} 클러스터, 최대={mask.sum():,} pts 채택")

    return pts, rgb


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
#  5. PCA 6-DOF 포즈 추정
# ======================================================================

def estimate_pose_cam0(pts, rgb=None, blade_dir="auto"):
    """PCA 기반 6-DOF 포즈 추정 (cam0 OpenCV 좌표계).

    물체 canonical frame:
      X = 길이 방향 (blade), Y = 너비, Z = 법선 (테이블 위로)
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

    # 방향 판별 (3가지 지표 다수결)
    proj = p @ length_ax
    if blade_dir == "auto" and rgb is not None:
        half_pos = proj > np.median(proj)
        half_neg = ~half_pos

        # 지표 1: HSV 채도
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        hsv_s = cv2.cvtColor(rgb_u8.reshape(1, -1, 3), cv2.COLOR_RGB2HSV)[0, :, 1].astype(float)
        sat_pos = hsv_s[half_pos].mean()
        sat_neg = hsv_s[half_neg].mean()

        # 지표 2: 밝기 분산
        gray = rgb.mean(axis=1)
        var_pos = gray[half_pos].var()
        var_neg = gray[half_neg].var()

        # 지표 3: 폭
        width_proj = p @ evec[:, 1]
        spread_pos = np.percentile(width_proj[half_pos], 95) - np.percentile(width_proj[half_pos], 5)
        spread_neg = np.percentile(width_proj[half_neg], 95) - np.percentile(width_proj[half_neg], 5)

        votes_pos = 0
        votes_pos += (1 if sat_pos < sat_neg else 0)
        votes_pos += (1 if var_pos > var_neg else 0)
        votes_pos += (1 if spread_pos < spread_neg else 0)

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
#  6. 회전 변환 유틸리티
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
    mpl_dir = Path(tempfile.gettempdir()) / "rb_dust3r_mpl_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 5))
    c = pose["centroid_m"]
    R = pose["R"]
    obb_m = pose["obb_mm"] / 1000

    # 좌측: 3D 점군
    ax1 = fig.add_subplot(121, projection='3d')
    sub = np.random.choice(len(pts), min(5000, len(pts)), replace=False)
    ax1.scatter(pts[sub, 0]*1000, pts[sub, 1]*1000, pts[sub, 2]*1000,
                c=rgb[sub], s=0.5, alpha=0.6)
    ax_len = max(obb_m) * 1000 * 0.8
    for ax_i, col, lbl in zip(range(3), ['r','g','b'], ['Blade','Width','Normal']):
        ax1.quiver(c[0]*1000, c[1]*1000, c[2]*1000,
                   R[0,ax_i]*ax_len, R[1,ax_i]*ax_len, R[2,ax_i]*ax_len,
                   color=col, linewidth=2, label=lbl)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.legend(fontsize=7)
    ax1.set_title("Point Cloud + Axes")

    # 우측: 정보 텍스트
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    euler = R_to_euler(R)
    info = (
        f"Position (mm): [{c[0]*1000:.1f}, {c[1]*1000:.1f}, {c[2]*1000:.1f}]\n"
        f"Euler XYZ (deg): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]\n"
        f"OBB (mm): [{obb_m[0]*1000:.1f} x {obb_m[1]*1000:.1f} x {obb_m[2]*1000:.1f}]\n"
        f"L/W ratio: {pose['length_width_ratio']:.1f}\n"
        f"Normal verticality: {pose['normal_verticality']*100:.1f}%\n"
        f"Blade dir: {pose['blade_dir_used']}\n"
        f"N points: {len(pts):,}"
    )
    ax2.text(0.05, 0.95, info, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', family='monospace')
    ax2.set_title(title)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"    시각화 저장: {out_path}")


def overlay_on_image(pts_cam0, pose, K, img_path, out_path):
    """cam0 이미지 위에 3D 점군 재투영 오버레이."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return

    h, w = bgr.shape[:2]
    z = pts_cam0[:, 2]
    ok = z > 0.01
    uv = pts_cam0[ok] @ K[:3, :3].T
    uv = uv[:, :2] / uv[:, 2:3]

    in_frame = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[in_frame].astype(int)

    overlay = bgr.copy()
    for u, v in uv[::max(1, len(uv)//3000)]:
        cv2.circle(overlay, (u, v), 1, (0, 255, 0), -1)

    # 포즈 축 그리기
    c3d = pose["centroid_m"]
    R = pose["R"]
    ax_len = 0.03
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    labels = ['X:Blade', 'Y:Width', 'Z:Normal']
    c_px = (K @ c3d).astype(int)
    c_px = (int(c_px[0]/c3d[2]), int(c_px[1]/c3d[2]))
    for i in range(3):
        tip = c3d + R[:, i] * ax_len
        t_px = (K @ tip).astype(int)
        t_px = (int(t_px[0]/tip[2]), int(t_px[1]/tip[2]))
        cv2.arrowedLine(overlay, c_px, t_px, colors[i], 2, tipLength=0.15)
        cv2.putText(overlay, labels[i], t_px, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)

    cv2.imwrite(str(out_path), overlay)
    print(f"    오버레이 저장: {out_path}")


# ======================================================================
#  8. 출력 저장
# ======================================================================

def save_all(pts, rgb, pose, out_dir, tag, elapsed, dust3r_info=None):
    """포즈 JSON/NPZ, PLY, GLB 저장 (모두 cam0/OpenCV 좌표계)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    R = pose["R"]
    t = pose["centroid_m"]
    euler = R_to_euler(R)
    quat = R_to_quat(R)

    # --- JSON ---
    pose_json = {
        "coordinate_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
        "method": "DUSt3R + PCA_direct + 3vote_blade_disambiguation",
        "position_mm": t.tolist() if isinstance(t, np.ndarray) else list(np.array(t)*1000),
        "euler_xyz_deg": euler.tolist(),
        "quaternion_wxyz": quat.tolist(),
        "rotation_matrix": R.tolist(),
        "blade_axis_cam0": pose["blade_axis"].tolist(),
        "normal_axis_cam0": pose["normal_axis"].tolist(),
        "width_axis_cam0": pose["width_axis"].tolist(),
        "obb_mm": pose["obb_mm"].tolist(),
        "pca_length_width_ratio": pose["length_width_ratio"],
        "normal_verticality_pct": pose["normal_verticality"] * 100,
        "blade_dir_used": pose["blade_dir_used"],
        "n_points": len(pts),
        "elapsed_sec": round(elapsed, 2),
    }
    # position_mm 수정 (centroid가 이미 m 단위)
    pose_json["position_mm"] = (t * 1000).tolist()

    if dust3r_info:
        pose_json["dust3r"] = dust3r_info

    json_path = out_dir / f"pose_cam0_{tag}.json"
    with open(str(json_path), 'w') as f:
        json.dump(pose_json, f, indent=2, ensure_ascii=False)
    print(f"    JSON 저장: {json_path}")

    # --- NPZ ---
    npz_path = out_dir / f"pose_cam0_{tag}.npz"
    np.savez(str(npz_path),
             R=R, t=t, euler_xyz_deg=euler, quaternion_wxyz=quat,
             obb_mm=pose["obb_mm"],
             blade_axis=pose["blade_axis"],
             normal_axis=pose["normal_axis"])
    print(f"    NPZ 저장: {npz_path}")

    # --- PLY (cam0 좌표계, 미터) ---
    ply_path = out_dir / f"pointcloud_cam0_{tag}.ply"
    _save_ply(pts, rgb, str(ply_path))
    print(f"    PLY 저장: {ply_path}")

    # --- GLB (OpenGL 좌표계 변환: Y,Z 반전) ---
    try:
        import trimesh
        glb_path = out_dir / f"mesh_cam0_{tag}.glb"
        # cam0 OpenCV → OpenGL: Y 반전, Z 반전
        pts_gl = pts.copy()
        pts_gl[:, 1] *= -1
        pts_gl[:, 2] *= -1
        colors_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        colors_rgba = np.column_stack([colors_u8, np.full(len(colors_u8), 255, dtype=np.uint8)])
        pc = trimesh.PointCloud(vertices=pts_gl, colors=colors_rgba)
        pc.export(str(glb_path))
        print(f"    GLB 저장: {glb_path}")
    except ImportError:
        print("    GLB 저장 건너뜀 (trimesh 없음)")

    return json_path


def _save_ply(pts, rgb, path):
    n = len(pts)
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    header = (
        f"ply\nformat ascii 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    )
    with open(path, 'w') as f:
        f.write(header)
        for i in range(n):
            f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                    f"{rgb_u8[i,0]} {rgb_u8[i,1]} {rgb_u8[i,2]}\n")


# ======================================================================
#  9. 메인
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="DUSt3R 기반 멀티뷰 3D 포즈 추정")
    parser.add_argument("--frame", type=int, default=3, help="프레임 번호")
    parser.add_argument("--capture_dir", type=str, default=str(_DEFAULT_CAP_DIR))
    parser.add_argument("--calib_dir", type=str, default=str(_DEFAULT_CAL_DIR))
    parser.add_argument("--intrinsics_dir", type=str, default=str(_DEFAULT_INT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(_DEFAULT_OUT_DIR))
    parser.add_argument("--image_size", type=int, default=512, help="DUSt3R 입력 해상도")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--niter", type=int, default=300, help="Global alignment 반복")
    parser.add_argument("--conf_threshold", type=float, default=0.8,
                        help="DUSt3R confidence 임계값")
    parser.add_argument("--no_known_poses", action="store_true",
                        help="카메라 포즈를 DUSt3R가 추정하도록 함")
    parser.add_argument("--blade_dir", type=str, default="auto",
                        choices=["auto", "pos", "neg"])
    parser.add_argument("--z_min", type=float, default=0.05)
    parser.add_argument("--z_max", type=float, default=1.0)
    parser.add_argument("--sphere_r", type=float, default=0.12,
                        help="관심 영역 반경 (m)")
    args = parser.parse_args()

    t0 = time.time()
    cap_dir = Path(args.capture_dir)
    cal_dir = Path(args.calib_dir)
    int_dir = Path(args.intrinsics_dir)

    # 카메라 탐색
    cams = discover_cameras(cap_dir)
    print(f"카메라: {cams}")

    pad = frame_pad(cap_dir, cams[0])
    fid = f"{args.frame:0{pad}d}"

    # 캘리브레이션 로드
    K_m, D_m, ds_m = {}, {}, {}
    for ci in cams:
        K_m[ci], D_m[ci], ds_m[ci] = load_intrinsics(int_dir, ci)
    T_m = load_extrinsics(cal_dir, cams)

    # 이미지 경로 수집
    image_paths = []
    cam_order = []
    for ci in cams:
        rp = str(cap_dir / f"cam{ci}" / f"rgb_{fid}.jpg")
        if os.path.exists(rp):
            image_paths.append(rp)
            cam_order.append(ci)
    print(f"이미지 {len(image_paths)}장: {image_paths}")

    # Known intrinsics / poses 준비
    known_K = [K_m[ci] for ci in cam_order]

    if not args.no_known_poses:
        # cam-to-world = T_C0_Ci (cam_i → cam0 world)
        known_poses = [T_m[ci] for ci in cam_order]
    else:
        known_poses = None

    # DUSt3R 추론
    result = run_dust3r(
        image_paths,
        known_intrinsics=known_K,
        known_poses=known_poses,
        image_size=args.image_size,
        device=args.device,
        niter=args.niter,
    )

    # cam0 좌표계 점군 변환
    print("\n[변환] DUSt3R → cam0 좌표계...")
    pts, rgb = dust3r_to_cam0(
        result, T_m,
        use_known_poses=(not args.no_known_poses),
        conf_threshold=args.conf_threshold,
        z_min=args.z_min,
        z_max=args.z_max,
    )
    print(f"    전체 점군: {len(pts):,} pts")

    # 관심 영역 추출
    print("\n[세그멘테이션] 물체 영역 추출...")
    center = pts.mean(0)
    pts, rgb = extract_object_region(pts, rgb,
                                     anchor_center=center,
                                     sphere_r=args.sphere_r)
    print(f"    최종 점군: {len(pts):,} pts")

    # PCA 포즈 추정
    print("\n[포즈] PCA 6-DOF 추정...")
    pose = estimate_pose_cam0(pts, rgb, blade_dir=args.blade_dir)
    euler = R_to_euler(pose["R"])
    print(f"    위치 (mm): [{pose['centroid_mm'][0]:.1f}, {pose['centroid_mm'][1]:.1f}, {pose['centroid_mm'][2]:.1f}]")
    print(f"    Euler (deg): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]")
    print(f"    OBB (mm): [{pose['obb_mm'][0]:.1f} x {pose['obb_mm'][1]:.1f} x {pose['obb_mm'][2]:.1f}]")
    print(f"    Blade dir: {pose['blade_dir_used']}")

    elapsed = time.time() - t0

    # 출력 저장
    tag = f"frame{fid}"
    out_dir = Path(args.output_dir) / f"output_{tag}"
    print(f"\n[저장] {out_dir}")

    dust3r_info = {
        "alignment_loss": result['loss'],
        "image_size": args.image_size,
        "niter": args.niter,
        "n_images": len(image_paths),
        "known_poses": not args.no_known_poses,
    }

    save_all(pts, rgb, pose, out_dir, tag, elapsed, dust3r_info)

    # 시각화
    try:
        visualize_pointcloud(pts, rgb, pose,
                             out_dir / f"pointcloud_{tag}.png",
                             f"DUSt3R Pose - frame {fid}")
    except Exception as e:
        print(f"    시각화 건너뜀: {e}")

    # cam0 오버레이
    cam0_img = str(cap_dir / f"cam0" / f"rgb_{fid}.jpg")
    if os.path.exists(cam0_img):
        overlay_on_image(pts, pose, K_m[0], cam0_img,
                         out_dir / f"overlay_cam0_{tag}.png")

    print(f"\n완료! ({elapsed:.1f}초)")


if __name__ == "__main__":
    main()
