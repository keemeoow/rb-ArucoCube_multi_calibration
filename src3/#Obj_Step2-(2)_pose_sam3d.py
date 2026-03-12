#Obj_Step2-(2)_pose_sam3d.py
"""
pose_from_sam3d.py  —  카메라(cam0) 좌표계 기준 물체 6-DOF 포즈 추정
======================================================================
방법: scene PLY의 PCA 직접 계산 + 색상 기반 날/손잡이 방향 판별
      (ICP·참조모델 불필요, 기하학적으로 직접 추정)

좌표계: cam0 (OpenCV: X-right, Y-down, Z-forward)

사용법:
  # Mode B — 기존 추출 점군 PLY 직접 사용 (빠름)
  python pose_from_sam3d.py \
    --scene_ply ../pose_estimate_grounding_sam/output_selfref/object_utility_knife_frame000005.ply

  # Mode A — GDino+SAM2 재검출 후 멀티카메라 depth 융합
  python pose_from_sam3d.py --frame 5 --device mps

  # 날 방향 수동 지정 (색상 판별이 불확실할 때)
  python pose_from_sam3d.py --scene_ply <path> --blade_dir neg
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
#  4. GDino + SAM2 검출 (Mode A)
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
  # Mode B: 기존 PLY 직접 사용
  python pose_from_sam3d.py \\
    --scene_ply ../pose_estimate_grounding_sam/output_selfref/object_utility_knife_frame000005.ply

  # Mode A: GDino+SAM2 재검출
  python pose_from_sam3d.py --frame 5 --device mps

  # 날 방향 수동 지정
  python pose_from_sam3d.py --scene_ply <path> --blade_dir neg
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

    # 알고리즘
    ap.add_argument("--blade_dir", choices=["auto","pos","neg"], default="auto",
                    help="날 방향: auto=색상판별(기본) / pos=PCA+ / neg=PCA-")

    # 출력
    ap.add_argument("--out_dir", default=_DEFAULT_OUT_DIR)

    args = ap.parse_args()
    t0   = time.time()
    mode = "B" if args.scene_ply else "A"

    print(f"{'='*56}")
    print(f"  pose_from_sam3d.py  (cam0 좌표계)")
    print(f"  mode       : {'B (scene_ply)' if mode=='B' else 'A (GDino+SAM2)'}")
    print(f"  blade_dir  : {args.blade_dir}")
    print(f"{'='*56}")

    # ── 씬 점군 취득 ─────────────────────────────────────────────
    if mode == "B":
        print(f"\n[Step 1] PLY 로드")
        pts, rgb = load_ply(args.scene_ply)
        tag = os.path.splitext(os.path.basename(args.scene_ply))[0]
        if len(pts) < 50:
            raise RuntimeError("점군 부족 (<50)")
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
    print(f"\n[Step {'2' if mode=='B' else '3'}] PCA 포즈 추정")
    pose = estimate_pose_cam0(pts, rgb, blade_dir=args.blade_dir)

    # ── 출력 + 저장 ───────────────────────────────────────────────
    print_and_save(pose, args.out_dir, tag, time.time() - t0)


if __name__ == "__main__":
    main()
