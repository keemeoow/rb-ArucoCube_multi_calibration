# reconstruct_3d.py
# 멀티카메라 RGBD → 3D 포인트클라우드 복원
# ArUco 마커 없이, 기존 캘리브레이션 결과(T_C0_Ci.npy)만 사용
#
# 사용법 (capture_rgbd_3cam.py로 촬영한 데이터 기준):
#
"""
단일 프레임 복원
python reconstruct_3d.py \
    --capture_dir ./data/rgbd_capture \
    --intrinsics_dir ./intrinsics \
    --calib_dir ./data/cube_session_01/calib_out_cube \
    --frame 0
"""
#
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
def depth_to_points(depth_u16: np.ndarray, K: np.ndarray, depth_scale: float,
                    z_min: float, z_max: float, stride: int = 1
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    depth 이미지 → camera frame 3D 점 + 픽셀 좌표(v, u).
    vectorized 구현으로 Step4보다 빠름.
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

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.column_stack([x, y, z])
    pixels = np.column_stack([v.astype(np.int32), u.astype(np.int32)])
    return points, pixels


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
               ds_map: Dict[int, float], z_min: float, z_max: float,
               stride: int, pad: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        pts_cam, pixels = depth_to_points(depth_u16, K_map[ci], ds_map[ci], z_min, z_max, stride)
        if pts_cam.shape[0] == 0:
            print(f"[WARN] cam{ci} frame {frame_idx}: 유효 depth 점 0개")
            continue

        pts_ref = transform_points(pts_cam, T_ref[ci])
        cols = rgb[pixels[:, 0], pixels[:, 1]]

        all_pts.append(pts_ref)
        all_cols.append(cols)
        print(f"  cam{ci}: {pts_cam.shape[0]:>8,} points")

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
    parser.add_argument("--ref_cam", type=int, default=1, help="기준 카메라 인덱스 (default: 0)")

    parser.add_argument("--frame", type=int, default=None, help="특정 프레임만 복원 (기본: 첫 번째)")
    parser.add_argument("--each_frame", action="store_true", help="모든 프레임을 각각 개별 PLY로 저장 (다른 물체들)")
    parser.add_argument("--all_frames", action="store_true", help="모든 프레임을 하나로 합치기 (같은 물체)")
    parser.add_argument("--frame_skip", type=int, default=1, help="each_frame/all_frames 시 N번째마다 사용")

    parser.add_argument("--z_min", type=float, default=0.1, help="depth 최소 거리 (m)")
    parser.add_argument("--z_max", type=float, default=0.5, help="depth 최대 거리 (m)")
    parser.add_argument("--stride", type=int, default=1, help="depth subsampling (1=dense, 4=sparse)")
    parser.add_argument("--voxel_mm", type=float, default=0.0, help="voxel downsample 크기 (mm), 0=OFF")

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
    K_map, ds_map = {}, {}
    for ci in cam_indices:
        K, D, ds = load_intrinsics(args.intrinsics_dir, ci)
        K_map[ci] = K
        ds_map[ci] = ds

    T_ref = load_extrinsics(args.calib_dir, args.ref_cam, cam_indices)

    # --- detect file naming ---
    pad = _detect_zero_padding(args.capture_dir, cam_indices[0])
    print(f"[INFO] 파일명 zero-padding: {pad}자리")

    # --- output directory ---
    out_dir = os.path.join(args.capture_dir, "ply")
    os.makedirs(out_dir, exist_ok=True)

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
                args.capture_dir, fid, cam_indices, T_ref, K_map, ds_map,
                args.z_min, args.z_max, args.stride, pad,
            )
            if not f_pts:
                print(f"  -> SKIP (유효 점 없음)\n")
                fail += 1
                continue

            P = np.concatenate(f_pts, axis=0)
            C = np.concatenate(f_cols, axis=0)

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
            args.capture_dir, fid, cam_indices, T_ref, K_map, ds_map,
            args.z_min, args.z_max, args.stride, pad,
        )
        all_pts.extend(f_pts)
        all_cols.extend(f_cols)

    if not all_pts:
        print("[ERROR] 유효한 3D 점이 없습니다. depth 파일과 z_min/z_max를 확인하세요.")
        return

    P = np.concatenate(all_pts, axis=0)
    C = np.concatenate(all_cols, axis=0)
    print(f"\n[INFO] 총 포인트: {P.shape[0]:,}")

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
