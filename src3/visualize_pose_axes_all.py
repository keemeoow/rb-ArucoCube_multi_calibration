#!/usr/bin/env python3
"""
Visualize aligned pose axes from all pose-estimation methods in one figure.

Methods:
- Step5 localize (position-only)
- Step6 ply (multicam_pose)
- Step7 direct (pose_frame*.json)
- Final grounding_sam (cam0_pose)
- Final sam3d knife (pose_cam0_*.json)
"""

import argparse
import glob
import json
import os
from typing import Dict, Optional

import numpy as np


def _latest_file(patterns, exclude_substr=None) -> Optional[str]:
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if exclude_substr:
        files = [f for f in files if exclude_substr not in os.path.basename(f)]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _as_np(v):
    return np.array(v, dtype=np.float64)


def load_step5(path: str, obj_idx: int = 0) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    data = json.load(open(path, "r", encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        return None
    frame0 = results[0]
    objects = frame0.get("objects", [])
    if not objects:
        return None
    obj = objects[min(obj_idx, len(objects) - 1)]
    pos_m = obj.get("position_ref_m") or obj.get("depth_centroid_ref_m")
    if pos_m is None:
        return None
    return {
        "name": "Step5 localize",
        "position_mm": _as_np(pos_m) * 1000.0,
        "rotation_matrix": None,
        "obb_mm": None,
        "note": "position-only (no rotation)",
    }


def load_step6(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    data = json.load(open(path, "r", encoding="utf-8"))
    mp = data.get("multicam_pose", {})
    if "position_mm" not in mp or "rotation_matrix" not in mp:
        return None
    return {
        "name": "Step6 ply",
        "position_mm": _as_np(mp["position_mm"]),
        "rotation_matrix": _as_np(mp["rotation_matrix"]),
        "obb_mm": _as_np(mp.get("obb_extents_mm", [0, 0, 0])),
        "note": "PCA + Umeyama + ICP",
    }


def load_step7(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    data = json.load(open(path, "r", encoding="utf-8"))
    pose = data.get("pose", {})
    if "position_mm" not in pose or "rotation_matrix" not in pose:
        return None
    return {
        "name": "Step7 direct",
        "position_mm": _as_np(pose["position_mm"]),
        "rotation_matrix": _as_np(pose["rotation_matrix"]),
        "obb_mm": _as_np(pose.get("obb_extents_mm", [0, 0, 0])),
        "note": f"rotation={pose.get('rotation_method', 'unknown')}",
    }


def load_final_grounding(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    data = json.load(open(path, "r", encoding="utf-8"))
    cam0 = data.get("cam0_pose", {})
    if "position_mm" not in cam0 or "rotation_matrix" not in cam0:
        return None
    return {
        "name": "Final grounding_sam",
        "position_mm": _as_np(cam0["position_mm"]),
        "rotation_matrix": _as_np(cam0["rotation_matrix"]),
        "obb_mm": _as_np(cam0.get("obb_extents_mm", [0, 0, 0])),
        "note": data.get("rotation_method", "icp"),
    }


def load_final_sam3d(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    data = json.load(open(path, "r", encoding="utf-8"))
    if "position_mm" not in data or "rotation_matrix" not in data:
        return None
    return {
        "name": "Final sam3d_knife",
        "position_mm": _as_np(data["position_mm"]),
        "rotation_matrix": _as_np(data["rotation_matrix"]),
        "obb_mm": _as_np(data.get("obb_mm", [0, 0, 0])),
        "note": data.get("method", "pca_direct"),
    }


def _draw_axes(ax, R, t, length, lw=2.0, alpha=1.0):
    colors = ["#e74c3c", "#27ae60", "#2980b9"]  # X, Y, Z
    for i in range(3):
        v = R[:, i] * length
        ax.quiver(
            t[0], t[1], t[2], v[0], v[1], v[2],
            color=colors[i], linewidth=lw, arrow_length_ratio=0.12, alpha=alpha
        )


def _draw_obb(ax, R, center, extents, color="#c0392b"):
    if extents is None or np.max(np.abs(extents)) < 1e-9:
        return
    h = extents / 2.0
    corners_l = np.array([
        [-h[0], -h[1], -h[2]], [h[0], -h[1], -h[2]],
        [h[0], h[1], -h[2]], [-h[0], h[1], -h[2]],
        [-h[0], -h[1], h[2]], [h[0], -h[1], h[2]],
        [h[0], h[1], h[2]], [-h[0], h[1], h[2]],
    ])
    corners = (R @ corners_l.T).T + center
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), color=color, lw=1.0, alpha=0.55)


def _panel(ax, item, radius):
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(item["name"], fontsize=11, pad=10)

    # cam0 reference frame
    _draw_axes(ax, np.eye(3), np.zeros(3), 60, lw=2.8, alpha=0.85)
    ax.text(0, 0, -25, "cam0", fontsize=8, ha="center")

    pos = item["position_mm"]
    R = item["rotation_matrix"]

    if R is not None:
        _draw_axes(ax, R, pos, 50, lw=3.2, alpha=1.0)
        _draw_obb(ax, R, pos, item["obb_mm"])
        euler = np.degrees(np.array([
            np.arctan2(R[2, 1], R[2, 2]),
            np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)),
            np.arctan2(R[1, 0], R[0, 0]),
        ]))
        ax.text(
            pos[0], pos[1], pos[2] + 35,
            f"euler=({euler[0]:.1f},{euler[1]:.1f},{euler[2]:.1f})",
            fontsize=7, ha="center",
        )
    else:
        ax.scatter([pos[0]], [pos[1]], [pos[2]], c="#c0392b", s=35)
        ax.text(pos[0], pos[1], pos[2] + 25, "rotation N/A", fontsize=7, ha="center")

    ax.plot3D([0, pos[0]], [0, pos[1]], [0, pos[2]], ":", color="#7f8c8d", lw=1.0, alpha=0.5)
    ax.text(pos[0], pos[1], pos[2] - 25, item.get("note", ""), fontsize=7, ha="center")

    c = pos / 2.0
    ax.set_xlim(c[0] - radius, c[0] + radius)
    ax.set_ylim(c[1] - radius, c[1] + radius)
    ax.set_zlim(c[2] - radius, c[2] + radius)
    ax.view_init(elev=24, azim=-55)


def main():
    parser = argparse.ArgumentParser(description="Visualize aligned pose axes for all methods")
    parser.add_argument("--step5_json", default="pose_step5_localize/output/localization_results.json")
    parser.add_argument("--step5_obj_idx", type=int, default=0)
    parser.add_argument("--step6_json", default="pose_step6_ply/output/pose_estimation_results.json")
    parser.add_argument("--step7_json", default=None, help="default: latest pose_step7_direct/output/pose_frame*.json")
    parser.add_argument("--grounding_json", default=None, help="default: latest pose_estimate_grounding_sam(최종)/output*/pose_*.json")
    parser.add_argument("--sam3d_json", default=None, help="default: latest pose_estimate_sam3d_knife(최최종)/output/pose_cam0_*.json")
    parser.add_argument("--out", default="pose_axes_comparison.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    step7_json = args.step7_json or _latest_file(
        ["pose_step7_direct/output/pose_frame*.json"], exclude_substr="_isaac"
    )
    grounding_json = args.grounding_json or _latest_file(
        [
            "pose_estimate_grounding_sam(최종)/output/pose_*.json",
            "pose_estimate_grounding_sam(최종)/output_selfref/pose_*.json",
        ]
    )
    sam3d_json = args.sam3d_json or _latest_file(
        ["pose_estimate_sam3d_knife(최최종)/output/pose_cam0_*.json"]
    )

    items = [
        load_step5(args.step5_json, args.step5_obj_idx),
        load_step6(args.step6_json),
        load_step7(step7_json),
        load_final_grounding(grounding_json),
        load_final_sam3d(sam3d_json),
    ]
    items = [it for it in items if it is not None]

    if not items:
        raise RuntimeError("No valid pose result files found.")

    all_pos = np.array([it["position_mm"] for it in items], dtype=np.float64)
    all_ext = []
    for it in items:
        if it["obb_mm"] is not None:
            all_ext.append(np.max(np.abs(it["obb_mm"])))
    max_pos = float(np.max(np.linalg.norm(all_pos, axis=1)))
    max_ext = float(max(all_ext) if all_ext else 80.0)
    radius = max(120.0, max_pos * 0.75, max_ext * 2.0)

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(items)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(6.2 * cols, 5.5 * rows))
    fig.suptitle("Pose Axes Comparison (cam0-aligned)", fontsize=15, y=0.98)

    for i, item in enumerate(items, start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        _panel(ax, item, radius)

    fig.text(
        0.5, 0.01,
        "Axis color: X=Red, Y=Green, Z=Blue | cam0 frame: X-right, Y-down, Z-forward",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ecf0f1", ec="#bdc3c7"),
    )

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"[SAVE] {os.path.abspath(out_path)}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

