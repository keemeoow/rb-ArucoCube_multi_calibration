#!/usr/bin/env python3
"""
포즈 추정 결과 시각화 — 2가지 방법 비교 (cam0 OpenCV 좌표계)

Methods:
  1. Obj_Step2-(1): grounding_sam (GDino+SAM2+ICP)
  2. Obj_Step2-(2): sam3d (PCA+색상)

시각화 내용:
  - 물체 포인트 클라우드 (색상 포함)
  - OBB (Oriented Bounding Box) — 물체의 회전 방향을 직관적으로 표시
  - 좌표축 (X=Red, Y=Green, Z=Blue)
  - Euler 회전각도 표시
  - cam0 원점 기준 좌표계

사용법:
  cd src3
  python Obj_Step3_visualize_object_pose.py

  # 특정 JSON/PLY 직접 지정
  python Obj_Step3_visualize_object_pose.py \
    --grounding_json "Obj_Step2-(1)_pose_estimate_grounding_sam/output_frame000005/pose_utility_knife_frame000005.json" \
    --grounding_ply  "Obj_Step2-(1)_pose_estimate_grounding_sam/output_frame000005/object_utility_knife_frame000005.ply" \
    --sam3d_json     "Obj_Step2-(2)_pose_estimate_sam3d/output_frame000005/pose_cam0_object_utility_knife_frame000005.json" \
    --sam3d_ply      "Obj_Step2-(1)_pose_estimate_grounding_sam/output_frame000005/object_utility_knife_frame000005.ply"
"""

import argparse
import glob
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ #
#  PLY loader (ASCII, xyz + optional rgb)
# ------------------------------------------------------------------ #
def _load_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pts, cols, has_rgb = [], [], False
    with open(path, "rb") as f:
        n = 0
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if "property uchar red" in line or "property uint8 red" in line:
                has_rgb = True
            if line == "end_header":
                break
        for _ in range(n):
            tok = f.readline().decode("ascii", errors="ignore").split()
            pts.append([float(tok[0]), float(tok[1]), float(tok[2])])
            if has_rgb:
                cols.append([int(tok[3]), int(tok[4]), int(tok[5])])
    pts = np.array(pts, dtype=np.float64)
    rgb = np.array(cols, dtype=np.float64) / 255.0 if has_rgb and cols else None
    return pts, rgb


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


def _R_to_euler_xyz(R):
    """Rotation matrix -> Euler XYZ (degrees)."""
    R = np.asarray(R, dtype=np.float64)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z]))


# ------------------------------------------------------------------ #
#  Pose loaders
# ------------------------------------------------------------------ #
def load_grounding_sam(json_path: str, ply_path: Optional[str] = None) -> Optional[Dict]:
    if not json_path or not os.path.exists(json_path):
        return None
    data = json.load(open(json_path, "r", encoding="utf-8"))
    cam0 = data.get("cam0_pose", {})
    if "position_mm" not in cam0 or "rotation_matrix" not in cam0:
        return None

    result = {
        "name": "Obj_Step2-(1) grounding_sam",
        "short": "GDino+SAM2+ICP",
        "position_mm": _as_np(cam0["position_mm"]),
        "rotation_matrix": _as_np(cam0["rotation_matrix"]),
        "obb_mm": _as_np(cam0.get("obb_extents_mm", [0, 0, 0])),
        "method": data.get("rotation_method", "icp"),
        "points": None,
        "colors": None,
    }

    # PLY 로드
    if ply_path and os.path.exists(ply_path):
        pts, rgb = _load_ply(ply_path)
        result["points"] = pts * 1000.0  # m -> mm
        result["colors"] = rgb
    else:
        # JSON 옆에서 object_*.ply 자동 탐색
        json_dir = os.path.dirname(json_path)
        ply_candidates = glob.glob(os.path.join(json_dir, "object_*.ply"))
        if ply_candidates:
            pts, rgb = _load_ply(ply_candidates[0])
            result["points"] = pts * 1000.0
            result["colors"] = rgb

    return result


def load_sam3d(json_path: str, ply_path: Optional[str] = None) -> Optional[Dict]:
    if not json_path or not os.path.exists(json_path):
        return None
    data = json.load(open(json_path, "r", encoding="utf-8"))
    if "position_mm" not in data or "rotation_matrix" not in data:
        return None

    result = {
        "name": "Obj_Step2-(2) sam3d",
        "short": "PCA+color",
        "position_mm": _as_np(data["position_mm"]),
        "rotation_matrix": _as_np(data["rotation_matrix"]),
        "obb_mm": _as_np(data.get("obb_mm", [0, 0, 0])),
        "method": data.get("method", "pca_direct"),
        "points": None,
        "colors": None,
    }

    if ply_path and os.path.exists(ply_path):
        pts, rgb = _load_ply(ply_path)
        result["points"] = pts * 1000.0
        result["colors"] = rgb

    return result


# ------------------------------------------------------------------ #
#  Drawing helpers
# ------------------------------------------------------------------ #
def _draw_axes(ax, R, t, length, lw=2.0, alpha=1.0, show_labels=False):
    colors = ["#e74c3c", "#27ae60", "#2980b9"]  # X, Y, Z
    names = ["X", "Y", "Z"]
    for i in range(3):
        v = R[:, i] * length
        ax.quiver(
            t[0], t[1], t[2], v[0], v[1], v[2],
            color=colors[i], linewidth=lw, arrow_length_ratio=0.12, alpha=alpha,
        )
        if show_labels:
            tip = t + v * 1.15
            ax.text(tip[0], tip[1], tip[2], names[i],
                    color=colors[i], fontsize=9, fontweight="bold")


def _draw_obb(ax, R, center, extents, color="#2c3e50", lw=1.8, alpha=0.7):
    """Oriented Bounding Box 12개 엣지 그리기."""
    if extents is None or np.max(np.abs(extents)) < 1e-9:
        return
    h = extents / 2.0
    corners_l = np.array([
        [-h[0], -h[1], -h[2]], [ h[0], -h[1], -h[2]],
        [ h[0],  h[1], -h[2]], [-h[0],  h[1], -h[2]],
        [-h[0], -h[1],  h[2]], [ h[0], -h[1],  h[2]],
        [ h[0],  h[1],  h[2]], [-h[0],  h[1],  h[2]],
    ])
    corners = (R @ corners_l.T).T + center
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), color=color, lw=lw, alpha=alpha)


def _draw_point_cloud(ax, pts, colors=None, subsample=2000, size=1.5, alpha=0.4):
    """포인트 클라우드를 scatter로 그린다. subsample로 렌더링 속도 제어."""
    if pts is None or len(pts) == 0:
        return
    n = len(pts)
    if n > subsample:
        idx = np.random.default_rng(42).choice(n, subsample, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]
    if colors is not None:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=colors, s=size, alpha=alpha, zorder=1)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c="#7f8c8d", s=size, alpha=alpha, zorder=1)


def _format_euler(euler):
    return f"({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f})"


def _draw_panel(ax, item, radius):
    """하나의 포즈 추정 결과를 3D 패널에 그린다."""
    pos = item["position_mm"]
    R = item["rotation_matrix"]
    euler = _R_to_euler_xyz(R)

    # 포인트 클라우드
    _draw_point_cloud(ax, item.get("points"), item.get("colors"),
                      subsample=3000, size=1.2, alpha=0.5)

    # cam0 기준 좌표축 (연하게)
    _draw_axes(ax, np.eye(3), np.zeros(3), 55, lw=2.5, alpha=0.3, show_labels=True)
    ax.text(0, 0, -20, "cam0 origin", fontsize=7, ha="center", alpha=0.6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.5))

    # 물체 포즈 축
    _draw_axes(ax, R, pos, 50, lw=3.5, alpha=1.0, show_labels=True)

    # OBB
    _draw_obb(ax, R, pos, item["obb_mm"])

    # 물체 중심 마커
    ax.scatter([pos[0]], [pos[1]], [pos[2]], c="#e74c3c", s=40,
               marker="o", edgecolors="k", linewidths=0.5, zorder=10)

    # cam0 → 물체 연결선
    ax.plot3D([0, pos[0]], [0, pos[1]], [0, pos[2]],
              ":", color="#7f8c8d", lw=1.0, alpha=0.4)

    # 정보 표시
    dist = np.linalg.norm(pos)
    info_lines = [
        f"pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm",
        f"dist: {dist:.1f} mm",
        f"euler XYZ: {_format_euler(euler)} deg",
        f"method: {item.get('method', '')}",
    ]
    obb = item["obb_mm"]
    if obb is not None and np.max(np.abs(obb)) > 1e-3:
        info_lines.append(f"OBB: {obb[0]:.1f} x {obb[1]:.1f} x {obb[2]:.1f} mm")

    info_text = "\n".join(info_lines)
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
              fontsize=7, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.4", fc="#ecf0f1", ec="#bdc3c7", alpha=0.9))

    # 제목
    ax.set_title(f"{item['name']}\n({item['short']})", fontsize=11, pad=12)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # 뷰 범위
    c = pos / 2.0
    ax.set_xlim(c[0] - radius, c[0] + radius)
    ax.set_ylim(c[1] - radius, c[1] + radius)
    ax.set_zlim(c[2] - radius, c[2] + radius)
    ax.view_init(elev=24, azim=-55)


def _draw_panel_each_axis(ax, item, radius, axis_idx: Optional[int]):
    """개별 축 뷰: None=전체, 0=X, 1=Y, 2=Z."""
    pos = item["position_mm"]
    R = item["rotation_matrix"]

    # 포인트 클라우드 (연하게)
    _draw_point_cloud(ax, item.get("points"), item.get("colors"),
                      subsample=2000, size=0.8, alpha=0.3)

    # cam0 기준 축 (연하게)
    _draw_axes(ax, np.eye(3), np.zeros(3), 50, lw=1.8, alpha=0.2)

    if axis_idx is None:
        _draw_axes(ax, R, pos, 50, lw=3.0, alpha=1.0, show_labels=True)
        title = "All axes (X,Y,Z)"
    else:
        # 해당 축만 강조
        axis_names = ["X", "Y", "Z"]
        axis_colors = ["#e74c3c", "#27ae60", "#2980b9"]
        v = R[:, axis_idx] * 55
        ax.quiver(pos[0], pos[1], pos[2], v[0], v[1], v[2],
                  color=axis_colors[axis_idx], linewidth=4.5,
                  arrow_length_ratio=0.12, alpha=1.0)
        tip = pos + v * 1.15
        ax.text(tip[0], tip[1], tip[2], axis_names[axis_idx],
                color=axis_colors[axis_idx], fontsize=10, fontweight="bold")
        # 나머지 축 연하게
        for j in range(3):
            if j == axis_idx:
                continue
            vj = R[:, j] * 45
            ax.quiver(pos[0], pos[1], pos[2], vj[0], vj[1], vj[2],
                      color=axis_colors[j], linewidth=1.0,
                      arrow_length_ratio=0.1, alpha=0.2)
        title = f"{axis_names[axis_idx]}-axis only"

    _draw_obb(ax, R, pos, item["obb_mm"], lw=1.2, alpha=0.5)

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    c = pos / 2.0
    ax.set_xlim(c[0] - radius, c[0] + radius)
    ax.set_ylim(c[1] - radius, c[1] + radius)
    ax.set_zlim(c[2] - radius, c[2] + radius)
    ax.view_init(elev=24, azim=-55)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="포즈 추정 결과 시각화 (grounding_sam + sam3d, cam0 좌표계)"
    )
    parser.add_argument("--grounding_json", default=None,
                        help="grounding_sam pose JSON (auto-detect if omitted)")
    parser.add_argument("--grounding_ply", default=None,
                        help="grounding_sam 물체 PLY (auto-detect if omitted)")
    parser.add_argument("--sam3d_json", default=None,
                        help="sam3d pose JSON (auto-detect if omitted)")
    parser.add_argument("--sam3d_ply", default=None,
                        help="sam3d에 사용할 물체 PLY (auto-detect if omitted)")
    parser.add_argument("--out", default="data/visualize_pose_axes_all_result/pose_comparison.png")
    parser.add_argument("--out_each_dir", default=None,
                        help="개별 축 시각화 저장 폴더 (default: <out>_each_axes)")
    parser.add_argument("--show", action="store_true", help="matplotlib 창 표시")
    args = parser.parse_args()

    # Auto-detect JSON paths
    grounding_json = args.grounding_json or _latest_file([
        "Obj_Step2-(1)_pose_estimate_grounding_sam/output_frame*/pose_*.json",
        "Obj_Step2-(1)_pose_estimate_grounding_sam/output/pose_*.json",
        "Obj_Step2-(1)_pose_estimate_grounding_sam/output_selfref/pose_*.json",
    ])
    sam3d_json = args.sam3d_json or _latest_file([
        "Obj_Step2-(2)_pose_estimate_sam3d/output_frame*/pose_cam0_*.json",
        "Obj_Step2-(2)_pose_estimate_sam3d/output/pose_cam0_*.json",
    ])

    print(f"[INFO] grounding_json: {grounding_json}")
    print(f"[INFO] sam3d_json:     {sam3d_json}")

    items = []

    # 1) grounding_sam
    gs_item = load_grounding_sam(grounding_json, args.grounding_ply)
    if gs_item:
        n_pts = len(gs_item["points"]) if gs_item["points"] is not None else 0
        print(f"[OK]   grounding_sam loaded  (pts={n_pts})")
        items.append(gs_item)
    else:
        print("[WARN] grounding_sam: pose JSON not found or invalid")

    # 2) sam3d
    sam3d_ply = args.sam3d_ply
    if sam3d_ply is None and gs_item and gs_item["points"] is not None:
        # sam3d는 같은 PLY를 사용할 수 있음 — grounding_sam의 PLY 재사용
        pass
    s_item = load_sam3d(sam3d_json, sam3d_ply)
    if s_item:
        # sam3d PLY가 없으면 grounding_sam의 PLY를 공유
        if s_item["points"] is None and gs_item and gs_item["points"] is not None:
            s_item["points"] = gs_item["points"]
            s_item["colors"] = gs_item["colors"]
        n_pts = len(s_item["points"]) if s_item["points"] is not None else 0
        print(f"[OK]   sam3d loaded  (pts={n_pts})")
        items.append(s_item)
    else:
        print("[WARN] sam3d: pose JSON not found or invalid")

    if not items:
        raise RuntimeError("No valid pose results found. Check JSON paths.")

    # 시각화 범위 계산
    all_pts_for_range = []
    for it in items:
        all_pts_for_range.append(it["position_mm"])
        if it["obb_mm"] is not None:
            all_pts_for_range.append(it["position_mm"] + it["obb_mm"] / 2)
            all_pts_for_range.append(it["position_mm"] - it["obb_mm"] / 2)
    all_pts_for_range = np.array(all_pts_for_range)
    max_pos = float(np.max(np.linalg.norm(all_pts_for_range, axis=1)))
    max_obb = max(
        float(np.max(np.abs(it["obb_mm"])))
        for it in items if it["obb_mm"] is not None
    ) if any(it["obb_mm"] is not None for it in items) else 80.0
    radius = max(120.0, max_pos * 0.75, max_obb * 2.0)

    # ---- matplotlib import ----
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- 메인 비교 그림 (1행 N열) ----
    n = len(items)
    fig = plt.figure(figsize=(8.0 * n, 7.0))
    fig.suptitle(
        "Object Pose Comparison (cam0 OpenCV: X-right, Y-down, Z-forward)",
        fontsize=14, y=0.98,
    )

    for i, item in enumerate(items, start=1):
        ax = fig.add_subplot(1, n, i, projection="3d")
        _draw_panel(ax, item, radius)

    fig.text(
        0.5, 0.01,
        "Axis: X=Red, Y=Green, Z=Blue | Box=OBB | Dots=Point Cloud",
        ha="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="#ecf0f1", ec="#bdc3c7"),
    )

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[SAVE] {os.path.abspath(out_path)}")

    # ---- 개별 축 시각화 (방법별 2x2: ALL/X/Y/Z) ----
    each_dir = args.out_each_dir or (os.path.splitext(out_path)[0] + "_each_axes")
    os.makedirs(each_dir, exist_ok=True)

    for item in items:
        slug = item["name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        fig2 = plt.figure(figsize=(14, 11))
        fig2.suptitle(
            f"{item['name']} — Axis Views (cam0-aligned)",
            fontsize=13, y=0.97,
        )

        for idx, pos_idx in enumerate([None, 0, 1, 2]):
            ax2 = fig2.add_subplot(2, 2, idx + 1, projection="3d")
            _draw_panel_each_axis(ax2, item, radius, pos_idx)

        fig2.text(
            0.5, 0.01,
            "Axis: X=Red, Y=Green, Z=Blue | cam0: X-right, Y-down, Z-forward",
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="#ecf0f1", ec="#bdc3c7"),
        )
        fig2.tight_layout(rect=[0, 0.05, 1, 0.95])
        each_path = os.path.join(each_dir, f"{slug}_axes.png")
        fig2.savefig(each_path, dpi=200, bbox_inches="tight")
        print(f"[SAVE] {os.path.abspath(each_path)}")
        if not args.show:
            plt.close(fig2)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
