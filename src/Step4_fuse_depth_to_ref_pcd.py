# Step4_fuse_depth_to_ref_pcd.py
# Depth를 ref 카메라 좌표로 fuse (포인트클라우드)

"""
python Step4_fuse_depth_to_ref_pcd.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 2 \
  --frame_idx 0 \
  --stride 4 \
  --z_min 0.2 --z_max 1.5 \
  --save_ply \
  --eval_icp
"""

import os
import json
import argparse
import numpy as np
import cv2
import open3d as o3d

def load_device_map(intrinsics_dir: str):
    map_path = os.path.join(intrinsics_dir, "device_map.json")
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"device_map.json not found: {map_path}")
    with open(map_path, "r") as f:
        m = json.load(f)
    serial_to_idx = m.get("serial_to_idx", {})
    if not serial_to_idx:
        raise RuntimeError("device_map.json has empty serial_to_idx.")
    return serial_to_idx, map_path

def load_intrinsics_cam_npz(intrinsics_dir, cam_idx: int):
    p = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing intrinsics npz: {p}")
    data = np.load(p, allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64)
    depth_scale = float(data["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in data else None
    return K, D, depth_scale

def discover_cam_indices_from_data(root_folder: str):
    cam_idxs = []
    for name in os.listdir(root_folder):
        if not name.startswith("cam"):
            continue
        try:
            idx = int(name.replace("cam", ""))
        except:
            continue
        # any rgb_*.jpg exists
        import glob
        if len(glob.glob(os.path.join(root_folder, name, "rgb_*.jpg"))) > 0:
            cam_idxs.append(idx)
    cam_idxs = sorted(cam_idxs)
    if len(cam_idxs) == 0:
        raise RuntimeError(f"No cam folders with rgb images found in {root_folder}")
    return cam_idxs

def depth_to_points_cam(depth_u16, K, depth_scale, z_min, z_max, stride=4):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    h, w = depth_u16.shape[:2]

    pts = []
    pix = []
    for v in range(0, h, stride):
        for u in range(0, w, stride):
            d = int(depth_u16[v, u])
            if d == 0:
                continue
            z = float(d) * float(depth_scale)
            if z < z_min or z > z_max:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts.append([x, y, z])
            pix.append((v, u))
    if len(pts) == 0:
        return np.empty((0,3), np.float64), []
    return np.asarray(pts, np.float64), pix

def transform_points_rowvec(points, T_ref_cam):
    R = T_ref_cam[:3,:3]
    t = T_ref_cam[:3,3]
    return points @ R.T + t.reshape(1,3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True)
    parser.add_argument("--intrinsics_dir", type=str, required=True)
    parser.add_argument("--calib_dir", type=str, default=None, help="default: <root>/calib_out_cube")
    parser.add_argument("--ref_cam_idx", type=int, default=0)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--z_min", type=float, default=0.2)
    parser.add_argument("--z_max", type=float, default=1.5)
    parser.add_argument("--depth_scale_override", type=float, default=None)
    parser.add_argument("--save_ply", action="store_true")
    parser.add_argument("--eval_icp", action="store_true", help="evaluate alignment with ICP per camera")
    parser.add_argument("--icp_dist", type=float, default=0.02, help="ICP max correspondence distance (m)")
    args = parser.parse_args()

    map_path = None
    try:
        _, map_path = load_device_map(args.intrinsics_dir)
    except Exception as e:
        print(f"[WARN] device_map.json unavailable. Continue without it: {e}")
    cam_indices = discover_cam_indices_from_data(args.root_folder)

    calib_dir = args.calib_dir or os.path.join(args.root_folder, "calib_out_cube")
    if not os.path.exists(calib_dir):
        raise FileNotFoundError(f"calib_dir not found: {calib_dir}")

    if args.ref_cam_idx not in cam_indices:
        raise RuntimeError(f"ref_cam_idx={args.ref_cam_idx} not found in data cams={cam_indices}")

    if map_path is not None:
        print(f"[INFO] Using device map: {map_path}")
    print(f"[INFO] Data cams found: {cam_indices}")
    print(f"[INFO] Calib dir: {calib_dir}")

    # Load transforms T_Cref_Ci
    T_ref = {}
    for ci in cam_indices:
        if ci == args.ref_cam_idx:
            T_ref[ci] = np.eye(4, dtype=np.float64)
            continue
        p = os.path.join(calib_dir, f"T_C{args.ref_cam_idx}_C{ci}.npy")
        if not os.path.exists(p):
            print(f"[WARN] Missing transform for cam{ci}: {p} (skip)")
            continue
        T_ref[ci] = np.load(p).astype(np.float64)

    per_cam_pcd = {}
    all_pts = []
    all_cols = []

    for ci in cam_indices:
        if ci not in T_ref:
            continue

        rgb_path = os.path.join(args.root_folder, f"cam{ci}", f"rgb_{args.frame_idx:05d}.jpg")
        depth_path = os.path.join(args.root_folder, f"cam{ci}", f"depth_{args.frame_idx:05d}.png")

        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            print(f"[WARN] Missing rgb/depth for cam{ci} frame {args.frame_idx} (skip)")
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_u16 is None:
            print(f"[WARN] Failed to read for cam{ci} (skip)")
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        K, D, depth_scale_npz = load_intrinsics_cam_npz(args.intrinsics_dir, ci)

        ds = float(args.depth_scale_override) if args.depth_scale_override is not None else depth_scale_npz
        if ds is None or np.isnan(ds):
            ds = 0.001

        pts_cam, pix = depth_to_points_cam(depth_u16, K, ds, args.z_min, args.z_max, stride=args.stride)
        if pts_cam.shape[0] == 0:
            print(f"[WARN] cam{ci}: no valid depth points in range (skip)")
            continue

        pts_ref = transform_points_rowvec(pts_cam, T_ref[ci])
        cols = np.array([rgb[v, u] for (v, u) in pix], dtype=np.float64)

        all_pts.append(pts_ref)
        all_cols.append(cols)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_ref)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        per_cam_pcd[ci] = pcd

        print(f"[INFO] cam{ci}: points={pts_ref.shape[0]} depth_scale={ds}")

    if len(all_pts) == 0:
        raise RuntimeError("No points collected. Check frame_idx, z range, depth files, transforms.")

    P = np.concatenate(all_pts, axis=0)
    C = np.concatenate(all_cols, axis=0)

    fused = o3d.geometry.PointCloud()
    fused.points = o3d.utility.Vector3dVector(P)
    fused.colors = o3d.utility.Vector3dVector(C)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    if args.save_ply:
        out_ply = os.path.join(args.root_folder, f"fused_ref{args.ref_cam_idx}_frame{args.frame_idx:05d}.ply")
        o3d.io.write_point_cloud(out_ply, fused)
        print(f"[INFO] Saved: {out_ply}")

    # -------- alignment evaluation (ICP) --------
    if args.eval_icp:
        if args.ref_cam_idx not in per_cam_pcd:
            print("[WARN] ref camera pointcloud not found; ICP eval skipped.")
        else:
            ref_pcd = per_cam_pcd[args.ref_cam_idx]
            ref_down = ref_pcd.voxel_down_sample(voxel_size=0.005)
            print(f"[EVAL] ICP max_corr_dist = {args.icp_dist} m")
            for ci, pcd in per_cam_pcd.items():
                if ci == args.ref_cam_idx:
                    continue
                src_down = pcd.voxel_down_sample(voxel_size=0.005)
                reg = o3d.pipelines.registration.registration_icp(
                    src_down, ref_down, args.icp_dist,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
                print(f"[EVAL] cam{ci} -> ref{args.ref_cam_idx}: fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.6f}")

    o3d.visualization.draw_geometries([axis, fused])


if __name__ == "__main__":
    main()
