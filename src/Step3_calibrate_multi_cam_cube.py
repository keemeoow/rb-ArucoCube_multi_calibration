# Step3_calibrate_multi_cam_cube.py
# 캘리브레이션(카메라 간 변환 계산)

"""
python Step3_calibrate_multi_cam_cube.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 2 \
  --min_markers 1 \
  --reproj_max_px 16 \
  --save_overlay \
  --overlay_max_per_cam 30
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import cv2

from aruco_cube import CubeConfig, ArucoCubeTarget, rodrigues_to_Rt, inv_T


# utils_pose.py가 있다고 가정. 없으면 fallback
try:
    from utils_pose import robust_se3_average, se3_distance
except Exception:
    robust_se3_average = None
    se3_distance = None


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def load_intrinsics(intr_dir: str, cam_idx: int):
    d = np.load(os.path.join(intr_dir, f"cam{cam_idx}.npz"))
    return d["color_K"].astype(np.float64), d["color_D"].astype(np.float64)


def parse_frame_id_from_rgb_path(rgb_path: str) -> int:
    return int(os.path.basename(rgb_path).split("_")[-1].split(".")[0])


def draw_overlay(img_bgr, corners_list, ids, img_pts, proj_pts, text_lines):
    vis = img_bgr.copy()

    if ids is not None and corners_list is not None and len(corners_list) > 0:
        try:
            cv2.aruco.drawDetectedMarkers(vis, corners_list, ids)
        except Exception:
            pass

    if img_pts is not None and proj_pts is not None:
        ip = img_pts.reshape(-1, 2)
        pp = proj_pts.reshape(-1, 2)
        for (x1, y1), (x2, y2) in zip(ip, pp):
            x1, y1, x2, y2 = map(int, map(round, (x1, y1, x2, y2)))
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.circle(vis, (x1, y1), 3, (0, 255, 0), -1)
            cv2.circle(vis, (x2, y2), 3, (0, 0, 255), -1)

    y = 30
    for t in text_lines:
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
        cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        y += 28
    return vis


@dataclass
class FrameRec:
    frame_id: int
    ts_ms: Optional[float]
    rgb_path: str


class CubeCam:
    def __init__(self, cam_idx: int):
        self.cam_idx = cam_idx
        self.frames: List[FrameRec] = []
        # per frame: T_C_O (Obj->Cam)
        self.T_C_O: Dict[int, np.ndarray] = {}
        # per frame: reproj dict
        self.reproj: Dict[int, dict] = {}

    def add_frame(self, fr: FrameRec):
        self.frames.append(fr)


def se3_avg_fallback(T_list: List[np.ndarray]) -> np.ndarray:
    # 매우 단순 평균 (레포용으론 robust_se3_average 추천)
    # translation: mean, rotation: SVD mean on R
    ts = np.array([T[:3, 3] for T in T_list], dtype=np.float64)
    t_mean = ts.mean(axis=0)

    Rs = np.array([T[:3, :3] for T in T_list], dtype=np.float64)
    R_mean = Rs.mean(axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t_mean
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--intrinsics_dir", required=True)

    parser.add_argument("--ref_cam_idx", type=int, default=0)
    parser.add_argument("--min_markers", type=int, default=1)
    parser.add_argument("--use_ransac", action="store_true")
    parser.add_argument("--reproj_max_px", type=float, default=10.0)

    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--overlay_max_per_cam", type=int, default=30)

    args = parser.parse_args()

    root = args.root_folder
    intr_dir = args.intrinsics_dir

    with open(os.path.join(root, "meta.json"), "r") as f:
        meta = json.load(f)

    cam_ids = {int(k) for cap in meta["captures"] for k, v in cap["cams"].items() if v.get("saved")}
    cams = {ci: CubeCam(ci) for ci in sorted(cam_ids)}

    for cap in meta["captures"]:
        for ci in cams:
            v = cap["cams"].get(str(ci))
            if v and v.get("saved"):
                fid = parse_frame_id_from_rgb_path(v["rgb_path"])
                cams[ci].add_frame(FrameRec(fid, v.get("ts_ms"), v["rgb_path"]))

    cfg = CubeConfig()
    cube = ArucoCubeTarget(cfg)

    K_map, D_map = {}, {}
    for ci in cams:
        K_map[ci], D_map[ci] = load_intrinsics(intr_dir, ci)
        print(f"[INFO] cam{ci}: loaded intrinsics.")

    out_root = ensure_dir(os.path.join(root, "calib_out_cube"))
    overlay_root = ensure_dir(os.path.join(out_root, "overlay")) if args.save_overlay else None
    transforms_dir = ensure_dir(os.path.join(out_root, "transforms"))
    calib_results_dir = ensure_dir(os.path.join(out_root, "calib_results"))

    # -----------------------------
    # 1) per-cam per-frame solvePnP (T_C_O 저장)
    # -----------------------------
    for ci, cam in cams.items():
        overlay_saved = 0

        for fr in cam.frames:
            img = cv2.imread(os.path.join(root, fr.rgb_path))
            if img is None:
                continue

            ok, rvec, tvec, used, reproj = cube.solve_pnp_cube(
                img, K_map[ci], D_map[ci],
                use_ransac=args.use_ransac,
                min_markers=args.min_markers,
                reproj_thr_mean_px=args.reproj_max_px,
                return_reproj=True
            )
            if not ok or reproj is None:
                continue

            T_C_O = rodrigues_to_Rt(rvec, tvec)
            cam.T_C_O[fr.frame_id] = T_C_O

            cam.reproj[fr.frame_id] = {
                "ok": True,
                "used_ids": [int(x) for x in used],
                "used_markers": int(len(used)),
                "n_points": int(reproj["n_points"]),
                "err_mean": float(reproj["err_mean"]),
                "err_median": float(reproj["err_median"]),
                "err_p90": float(reproj["err_p90"]),
                # 저장 (후반부 상대변환 계산에 필요)
                "rvec": np.asarray(rvec).reshape(3, 1).astype(float).tolist(),
                "tvec": np.asarray(tvec).reshape(3, 1).astype(float).tolist(),
            }

            # overlay 저장
            if args.save_overlay and overlay_saved < args.overlay_max_per_cam:
                corners_list, ids_flat = cube.detect(img)
                obj_pts = reproj["obj_pts"]
                img_pts = reproj["img_pts"]
                proj2 = reproj["proj2"].reshape(-1, 1, 2)

                text = [
                    f"cam{ci} frame={fr.frame_id:05d} used={used}",
                    f"reproj mean={reproj['err_mean']:.2f}px  p90={reproj['err_p90']:.2f}px  n={reproj['n_points']}",
                ]
                vis = draw_overlay(
                    img,
                    corners_list,
                    (None if ids_flat is None else ids_flat.reshape(-1, 1)),
                    img_pts,
                    proj2,
                    text
                )
                cam_dir = ensure_dir(os.path.join(overlay_root, f"cam{ci}"))
                out_img = os.path.join(cam_dir, f"overlay_{fr.frame_id:05d}.jpg")
                cv2.imwrite(out_img, vis)
                overlay_saved += 1

        print(f"[INFO] cam{ci}: solved {len(cam.T_C_O)}/{len(cam.frames)} frames")

    # -----------------------------
    # 2) cam-to-ref transforms: T_Cref_Ci(frame) = T_Cref_O @ inv(T_Ci_O)
    #    그리고 robust 평균으로 최종 T_Cref_Ci 저장
    # -----------------------------
    ref_ci = args.ref_cam_idx
    if ref_ci not in cams:
        raise RuntimeError(f"ref_cam_idx={ref_ci} not in cams={sorted(cams.keys())}")

    ref_cam = cams[ref_ci]
    common_frames = set(ref_cam.T_C_O.keys())
    if len(common_frames) == 0:
        raise RuntimeError("Reference camera has 0 valid PnP frames. (min_markers/reproj_max_px 확인)")

    # per camera final transform 저장
    final_T: Dict[int, np.ndarray] = {ref_ci: np.eye(4, dtype=np.float64)}

    for ci, cam in cams.items():
        if ci == ref_ci:
            continue

        common = sorted(list(common_frames.intersection(cam.T_C_O.keys())))
        if len(common) == 0:
            print(f"[WARN] cam{ci}: no common frames with ref cam{ref_ci}. skip transform.")
            continue

        T_list = []
        for fid in common:
            T_Cref_O = ref_cam.T_C_O[fid]
            T_Ci_O = cam.T_C_O[fid]
            T_Cref_Ci = T_Cref_O @ inv_T(T_Ci_O)  # ✅ 방향 고정
            T_list.append(T_Cref_Ci)

        if robust_se3_average is not None:
            T_avg = robust_se3_average(T_list)
        else:
            T_avg = se3_avg_fallback(T_list)

        final_T[ci] = T_avg

        npy_path = os.path.join(out_root, f"T_C{ref_ci}_C{ci}.npy")
        np.save(npy_path, T_avg)
        print(f"[SAVE] {npy_path}  (from {len(T_list)} frames)")

    # -----------------------------
    # 3) 결과 저장 (CSV + json)
    # -----------------------------
    # (a) per-cam reproj stats csv
    for ci, cam in cams.items():
        result_file = os.path.join(calib_results_dir, f"cam{ci}_reproj.csv")
        with open(result_file, "w") as f:
            f.write("frame_id,ok,used_markers,n_points,err_mean,err_median,err_p90,used_ids\n")
            for fr in cam.frames:
                r = cam.reproj.get(fr.frame_id)
                if not r:
                    continue
                f.write(
                    f"{fr.frame_id},{r.get('ok', True)},{r['used_markers']},{r['n_points']},"
                    f"{r['err_mean']:.6f},{r['err_median']:.6f},{r['err_p90']:.6f},\"{r['used_ids']}\"\n"
                )
        print(f"[SAVE] {result_file}")

    # (b) final transforms json (human readable)
    final_json = {}
    for ci, T in final_T.items():
        final_json[str(ci)] = T.reshape(-1).tolist()

    out_json = os.path.join(transforms_dir, f"T_C{ref_ci}_Ci_all.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "ref_cam_idx": int(ref_ci),
                "T_Cref_Ci": final_json
            },
            f,
            indent=2
        )
    print(f"[SAVE] {out_json}")

    print("[INFO] Step3 calibration finished.")


if __name__ == "__main__":
    main()
