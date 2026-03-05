"""
Reconstruct position-accurate 3D instances from multi-camera RGBD + SAM masks.

Inputs:
- capture_dir/cam{idx}/rgb_XXXXXX.jpg
- capture_dir/cam{idx}/depth_XXXXXX.png
- intrinsics_dir/cam{idx}.npz
- calib_dir/T_C{ref}_C{idx}.npy  (from Step3 calibration)
- masks_dir/cam{idx}/... mask files

Mask formats:
1) Label map per frame (recommended):
   - cam0/mask_000123.png (0=background, 1..N=instance id)
   - cam0/seg_000123.png / label_000123.png / rgb_000123.png / 000123.png also accepted
2) Binary masks per instance:
   - cam0/chair_000123.png
   - cam0/table_000123.png
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_intrinsics(intrinsics_dir: str, cam_idx: int):
    path = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing intrinsics: {path}")
    data = np.load(path, allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    depth_scale = (
        float(data["depth_scale_m_per_unit"])
        if "depth_scale_m_per_unit" in data
        else 0.001
    )
    return K, depth_scale


def load_extrinsics(calib_dir: str, ref_idx: int, cam_indices: List[int]) -> Dict[int, np.ndarray]:
    T_ref = {}
    for ci in cam_indices:
        if ci == ref_idx:
            T_ref[ci] = np.eye(4, dtype=np.float64)
            continue
        path = os.path.join(calib_dir, f"T_C{ref_idx}_C{ci}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing extrinsics: {path}")
        T_ref[ci] = np.load(path).astype(np.float64)
    return T_ref


def discover_cameras(capture_dir: str) -> List[int]:
    indices = []
    for d in sorted(glob.glob(os.path.join(capture_dir, "cam*"))):
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
        raise RuntimeError(f"No camera folders found in {capture_dir}")
    return indices


def discover_frames(capture_dir: str, cam_indices: List[int]) -> List[int]:
    frames = set()
    for ci in cam_indices:
        for p in glob.glob(os.path.join(capture_dir, f"cam{ci}", "rgb_*.jpg")):
            stem = os.path.basename(p).replace("rgb_", "").replace(".jpg", "")
            try:
                frames.add(int(stem))
            except ValueError:
                continue
    if not frames:
        raise RuntimeError(f"No RGB frames found in {capture_dir}")
    return sorted(frames)


def detect_zero_padding(capture_dir: str, cam_idx: int) -> int:
    files = glob.glob(os.path.join(capture_dir, f"cam{cam_idx}", "rgb_*.jpg"))
    if not files:
        return 6
    sample = os.path.basename(files[0]).replace("rgb_", "").replace(".jpg", "")
    return len(sample)


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    n = int(points.shape[0])
    rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = rgb_u8[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t.reshape(1, 3)


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float):
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


def _load_mask_image(path: str):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def _extract_masks_from_label_map(mask: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    uniq = np.unique(mask)
    for v in uniq:
        if int(v) <= 0:
            continue
        out[str(int(v))] = (mask == v)
    return out


def load_instance_masks(masks_dir: str, cam_idx: int, frame_id_str: str) -> Dict[str, np.ndarray]:
    cam_dir = os.path.join(masks_dir, f"cam{cam_idx}")
    if not os.path.isdir(cam_dir):
        return {}

    # Preferred: one label map image with multiple ids
    label_candidates = [
        f"mask_{frame_id_str}.png",
        f"seg_{frame_id_str}.png",
        f"label_{frame_id_str}.png",
        f"rgb_{frame_id_str}.png",
        f"{frame_id_str}.png",
    ]
    for fname in label_candidates:
        p = os.path.join(cam_dir, fname)
        if not os.path.exists(p):
            continue
        mask = _load_mask_image(p)
        if mask is None:
            continue
        masks = _extract_masks_from_label_map(mask)
        if masks:
            return masks
        # binary mask fallback
        if np.any(mask > 0):
            return {"1": (mask > 0)}

    # Fallback: multiple binary masks (instanceName_000123.png)
    out = {}
    for p in sorted(glob.glob(os.path.join(cam_dir, f"*_{frame_id_str}.png"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        suffix = f"_{frame_id_str}"
        inst_name = stem[:-len(suffix)] if stem.endswith(suffix) else stem
        mask = _load_mask_image(p)
        if mask is None:
            continue
        bin_mask = (mask > 0)
        if np.any(bin_mask):
            out[inst_name] = bin_mask
    return out


def masked_depth_to_points(
    depth_u16: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    depth_scale: float,
    z_min: float,
    z_max: float,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = depth_u16.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    v_grid, u_grid = np.mgrid[0:h:stride, 0:w:stride]
    d = depth_u16[v_grid, u_grid].astype(np.float64) * depth_scale
    m = mask[v_grid, u_grid]
    valid = m & (d > z_min) & (d < z_max)

    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    z = d[valid]
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam = np.column_stack([x, y, z])

    u_i = u_grid[valid].astype(np.int32)
    v_i = v_grid[valid].astype(np.int32)
    colors = rgb[v_i, u_i]
    return points_cam, colors


def main():
    parser = argparse.ArgumentParser(
        description="Fuse SAM masks into metric 3D using multi-camera calibration."
    )
    parser.add_argument("--capture_dir", required=True)
    parser.add_argument("--intrinsics_dir", default="./intrinsics")
    parser.add_argument("--calib_dir", required=True, help="Folder containing T_Cref_Ci.npy")
    parser.add_argument("--masks_dir", required=True, help="Folder containing cam*/mask files")
    parser.add_argument("--ref_cam", type=int, default=0)
    parser.add_argument("--frame", type=int, default=None, help="Single frame id")
    parser.add_argument("--all_frames", action="store_true", help="Fuse all frames")
    parser.add_argument("--frame_skip", type=int, default=1)
    parser.add_argument("--instance_ids", type=str, default="", help="Comma list. Empty=all instances")
    parser.add_argument("--z_min", type=float, default=0.1)
    parser.add_argument("--z_max", type=float, default=2.0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--voxel_mm", type=float, default=0.0)
    parser.add_argument("--min_points", type=int, default=300, help="Drop tiny noisy instance clouds")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    cam_indices = discover_cameras(args.capture_dir)
    frame_ids = discover_frames(args.capture_dir, cam_indices)
    pad = detect_zero_padding(args.capture_dir, cam_indices[0])

    if args.all_frames:
        target_frames = frame_ids[::args.frame_skip]
    else:
        fid = args.frame if args.frame is not None else frame_ids[0]
        if fid not in frame_ids:
            raise RuntimeError(f"Frame {fid} not found in capture data.")
        target_frames = [fid]

    selected_ids = set(s.strip() for s in args.instance_ids.split(",") if s.strip())

    K_map, ds_map = {}, {}
    for ci in cam_indices:
        K, ds = load_intrinsics(args.intrinsics_dir, ci)
        K_map[ci] = K
        ds_map[ci] = ds
    T_ref = load_extrinsics(args.calib_dir, args.ref_cam, cam_indices)

    out_dir = args.out_dir or os.path.join(args.capture_dir, "sam3d")
    os.makedirs(out_dir, exist_ok=True)

    instance_points: Dict[str, List[np.ndarray]] = {}
    instance_colors: Dict[str, List[np.ndarray]] = {}
    per_instance_meta = {}

    print(f"[INFO] cameras={cam_indices}, frames={target_frames}, ref_cam={args.ref_cam}")
    print(f"[INFO] masks_dir={os.path.abspath(args.masks_dir)}")

    for fid in target_frames:
        fid_str = f"{fid:0{pad}d}"
        print(f"[FRAME] {fid}")
        for ci in cam_indices:
            rgb_path = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid_str}.jpg")
            depth_path = os.path.join(args.capture_dir, f"cam{ci}", f"depth_{fid_str}.png")
            if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
                continue

            rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if rgb_bgr is None or depth_u16 is None:
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

            masks = load_instance_masks(args.masks_dir, ci, fid_str)
            if not masks:
                continue

            for inst_id, mask in masks.items():
                if selected_ids and inst_id not in selected_ids:
                    continue
                pts_cam, cols = masked_depth_to_points(
                    depth_u16=depth_u16,
                    rgb=rgb,
                    mask=mask,
                    K=K_map[ci],
                    depth_scale=ds_map[ci],
                    z_min=args.z_min,
                    z_max=args.z_max,
                    stride=args.stride,
                )
                if pts_cam.shape[0] == 0:
                    continue

                pts_ref = transform_points(pts_cam, T_ref[ci])
                instance_points.setdefault(inst_id, []).append(pts_ref)
                instance_colors.setdefault(inst_id, []).append(cols)

                meta = per_instance_meta.setdefault(inst_id, {"frames": set(), "cams": set(), "raw_points": 0})
                meta["frames"].add(int(fid))
                meta["cams"].add(int(ci))
                meta["raw_points"] += int(pts_ref.shape[0])

    if not instance_points:
        print("[WARN] No masked points found. Check mask naming and depth range.")
        return

    summary = {
        "ref_cam": int(args.ref_cam),
        "capture_dir": os.path.abspath(args.capture_dir),
        "calib_dir": os.path.abspath(args.calib_dir),
        "masks_dir": os.path.abspath(args.masks_dir),
        "frames": [int(x) for x in target_frames],
        "instances": {},
    }

    for inst_id in sorted(instance_points.keys(), key=lambda x: str(x)):
        P = np.concatenate(instance_points[inst_id], axis=0)
        C = np.concatenate(instance_colors[inst_id], axis=0)

        if args.voxel_mm > 0:
            P, C = voxel_downsample(P, C, args.voxel_mm / 1000.0)

        if P.shape[0] < args.min_points:
            print(f"[SKIP] instance={inst_id}: too few points ({P.shape[0]})")
            continue

        centroid = P.mean(axis=0)
        bmin = P.min(axis=0)
        bmax = P.max(axis=0)

        if args.all_frames:
            ply_name = f"instance_{inst_id}_allframes.ply"
        else:
            only_fid = target_frames[0]
            ply_name = f"instance_{inst_id}_frame{only_fid:0{pad}d}.ply"
        ply_path = os.path.join(out_dir, ply_name)
        save_ply(ply_path, P, C)
        print(f"[SAVE] {ply_path} ({P.shape[0]:,} pts)")

        meta = per_instance_meta[inst_id]
        summary["instances"][str(inst_id)] = {
            "point_count": int(P.shape[0]),
            "raw_point_count": int(meta["raw_points"]),
            "frames_used": sorted(int(x) for x in meta["frames"]),
            "cams_used": sorted(int(x) for x in meta["cams"]),
            "centroid_m": centroid.astype(float).tolist(),
            "bbox_min_m": bmin.astype(float).tolist(),
            "bbox_max_m": bmax.astype(float).tolist(),
            "ply_path": os.path.abspath(ply_path),
        }

    out_json = os.path.join(out_dir, "instance_positions.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] {out_json}")


if __name__ == "__main__":
    main()
