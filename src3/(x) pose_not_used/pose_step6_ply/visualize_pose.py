"""
Step6 포즈 추정 결과를 3D 좌표 프레임으로 시각화.

  (좌) cam0 좌표계 — 실세계 mm 기준
  (우) COLMAP 좌표계 — 재구성 단위

사용법:  python visualize_pose.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── 그리기 헬퍼 ─────────────────────────────────────────────

def draw_frame(ax, R, t, length, label="", lw=2.0, font=8, alpha=1.0,
               show_axis_labels=False, label_offset=None):
    """좌표 프레임 (X=Red, Y=Green, Z=Blue) 그리기."""
    colors = ["#e74c3c", "#27ae60", "#2980b9"]
    names = ["X", "Y", "Z"]
    for i in range(3):
        vec = R[:, i] * length
        ax.quiver(
            t[0], t[1], t[2], vec[0], vec[1], vec[2],
            color=colors[i], linewidth=lw, arrow_length_ratio=0.13, alpha=alpha,
        )
        if show_axis_labels:
            tip = t + vec * 1.12
            ax.text(tip[0], tip[1], tip[2], names[i],
                    fontsize=6, color=colors[i], fontweight="bold", alpha=alpha)
    if label:
        off = label_offset if label_offset is not None else np.array([0, 0, -length * 0.35])
        lp = t + off
        ax.text(lp[0], lp[1], lp[2], label,
                fontsize=font, fontweight="bold", ha="center", va="top")


def draw_obb_wireframe(ax, R, center, extents, color="gray", lw=0.8, alpha=0.5):
    """OBB 와이어프레임 + 반투명 면."""
    hx, hy, hz = extents / 2
    c = np.array([
        [-hx,-hy,-hz],[+hx,-hy,-hz],[+hx,+hy,-hz],[-hx,+hy,-hz],
        [-hx,-hy,+hz],[+hx,-hy,+hz],[+hx,+hy,+hz],[-hx,+hy,+hz],
    ])
    corners = (R @ c.T).T + center
    faces_idx = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
    faces = [[corners[j] for j in fi] for fi in faces_idx]
    ax.add_collection3d(Poly3DCollection(
        faces, alpha=0.06, facecolor=color, edgecolor=color, linewidth=0.4))
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), color=color, lw=lw, alpha=alpha)


def draw_camera_frustum(ax, R, t, size, color, label):
    """카메라를 작은 피라미드로 표시."""
    s = size
    local = np.array([
        [0,0,0], [-s,-s*0.7,s*1.5],[s,-s*0.7,s*1.5],[s,s*0.7,s*1.5],[-s,s*0.7,s*1.5]
    ])
    pts = (R @ local.T).T + t
    for i in range(1,5):
        j = (i % 4) + 1
        ax.add_collection3d(Poly3DCollection(
            [[pts[0], pts[i], pts[j]]], alpha=0.12, facecolor=color,
            edgecolor=color, linewidth=0.5))
    ax.add_collection3d(Poly3DCollection(
        [[pts[1],pts[2],pts[3],pts[4]]], alpha=0.08, facecolor=color,
        edgecolor=color, linewidth=0.5))
    ax.text(t[0], t[1], t[2] - size*1.2, label,
            fontsize=8, fontweight="bold", ha="center", color=color)


def set_equal_aspect(ax, points, pad=1.3):
    """3D 축 등비 스케일."""
    pts = np.array(points)
    c = pts.mean(axis=0)
    r = max((pts.max(axis=0) - pts.min(axis=0)).max() / 2 * pad, 1.0)
    ax.set_xlim(c[0]-r, c[0]+r)
    ax.set_ylim(c[1]-r, c[1]+r)
    ax.set_zlim(c[2]-r, c[2]+r)


def dashed_line(ax, p1, p2, color, lw=0.8, alpha=0.4, style="--"):
    ax.plot3D([p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
              style, color=color, lw=lw, alpha=alpha)


def midpoint_label(ax, p1, p2, text, color, fontsize=7, offset=None):
    mid = (np.array(p1) + np.array(p2)) / 2
    if offset is not None:
        mid = mid + np.array(offset)
    ax.text(mid[0], mid[1], mid[2], text, fontsize=fontsize, color=color, ha="center")


# ── 메인 ────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base, "output", "pose_estimation_results.json")) as f:
        data = json.load(f)

    calib_dir = os.path.join(base, "..", "data", "cube_session_01", "calib_out_cube")
    T_C0_C1 = np.load(os.path.join(calib_dir, "T_C0_C1.npy"))
    T_C0_C2 = np.load(os.path.join(calib_dir, "T_C0_C2.npy"))

    fig = plt.figure(figsize=(20, 9))
    fig.suptitle("Step6 Pose Estimation — Coordinate Frame Visualization",
                 fontsize=15, fontweight="bold", y=0.97)

    # ================================================================
    #  LEFT: cam0 좌표계 (실세계 mm)
    # ================================================================
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("cam0 Frame  (Real-world, mm)", fontsize=12, pad=20)
    ax1.set_xlabel("X (mm)", fontsize=9, labelpad=8)
    ax1.set_ylabel("Y (mm)", fontsize=9, labelpad=8)
    ax1.set_zlabel("Z (mm)", fontsize=9, labelpad=8)

    # cam0 원점
    draw_frame(ax1, np.eye(3), np.zeros(3), length=70,
               label="cam0 (ref)", lw=3, font=9, show_axis_labels=True,
               label_offset=np.array([0, 0, -30]))

    # cam1, cam2
    R1, t1_mm = T_C0_C1[:3,:3], T_C0_C1[:3,3]*1000
    R2, t2_mm = T_C0_C2[:3,:3], T_C0_C2[:3,3]*1000
    draw_camera_frustum(ax1, R1, t1_mm, 30, "#e67e22", "cam1")
    draw_frame(ax1, R1, t1_mm, 45, lw=1.5, alpha=0.5, show_axis_labels=False)
    draw_camera_frustum(ax1, R2, t2_mm, 30, "#8e44ad", "cam2")
    draw_frame(ax1, R2, t2_mm, 45, lw=1.5, alpha=0.5, show_axis_labels=False)

    # 객체 포즈
    mp = data["multicam_pose"]
    obj_pos = np.array(mp["position_mm"])
    obj_R = np.array(mp["rotation_matrix"])
    obj_obb = np.array(mp["obb_extents_mm"])
    euler = mp["euler_xyz_deg"]

    draw_frame(ax1, obj_R, obj_pos, length=55, label="", lw=3.5, font=10,
               show_axis_labels=True)
    draw_obb_wireframe(ax1, obj_R, obj_pos, obj_obb, color="#c0392b", lw=1.0, alpha=0.6)

    # 객체 라벨 (위쪽)
    ax1.text(obj_pos[0], obj_pos[1]-60, obj_pos[2]+80,
             "Object (tiger)",
             fontsize=10, fontweight="bold", color="#c0392b", ha="center")
    ax1.text(obj_pos[0], obj_pos[1]-60, obj_pos[2]+55,
             f"pos = ({obj_pos[0]:.1f}, {obj_pos[1]:.1f}, {obj_pos[2]:.1f}) mm",
             fontsize=7, color="#2c3e50", ha="center")
    ax1.text(obj_pos[0], obj_pos[1]-60, obj_pos[2]+35,
             f"euler = ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg",
             fontsize=7, color="#2c3e50", ha="center")
    ax1.text(obj_pos[0], obj_pos[1]-60, obj_pos[2]+15,
             f"OBB = {obj_obb[0]:.1f} x {obj_obb[1]:.1f} x {obj_obb[2]:.1f} mm",
             fontsize=7, color="#c0392b", ha="center")

    # 베이스라인 연결
    for t_cam, lbl, col in [(t1_mm,"cam1","#e67e22"),(t2_mm,"cam2","#8e44ad")]:
        dashed_line(ax1, [0,0,0], t_cam, col)
        d = np.linalg.norm(t_cam)
        midpoint_label(ax1, [0,0,0], t_cam, f"{d:.0f} mm", col, fontsize=6)

    # cam0→object
    dashed_line(ax1, [0,0,0], obj_pos, "#c0392b", style=":")
    midpoint_label(ax1, [0,0,0], obj_pos,
                   f"{np.linalg.norm(obj_pos):.0f} mm", "#c0392b", fontsize=6)

    pts_mm = [[0,0,0], t1_mm.tolist(), t2_mm.tolist(), obj_pos.tolist()]
    set_equal_aspect(ax1, pts_mm)
    ax1.view_init(elev=28, azim=-55)

    # ================================================================
    #  RIGHT: COLMAP 좌표계
    # ================================================================
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("COLMAP Frame  (reconstruction units)", fontsize=12, pad=20)
    ax2.set_xlabel("X", fontsize=9, labelpad=8)
    ax2.set_ylabel("Y", fontsize=9, labelpad=8)
    ax2.set_zlabel("Z", fontsize=9, labelpad=8)

    fl = 0.6  # frame length

    # SAM3D
    sp = data["sam3d_pose"]
    sam_pos, sam_R = np.array(sp["position_m"]), np.array(sp["rotation_matrix"])
    sam_obb = np.array(sp["obb_extents"])
    draw_frame(ax2, sam_R, sam_pos, fl, lw=2.5, show_axis_labels=True)
    draw_obb_wireframe(ax2, sam_R, sam_pos, sam_obb, "#3498db", lw=0.7, alpha=0.4)
    ax2.text(sam_pos[0], sam_pos[1], sam_pos[2]-0.6,
             "SAM3D\n(3DGS coord)", fontsize=9, fontweight="bold",
             color="#2980b9", ha="center")

    # gs2mesh
    gp = data["gs2mesh_pose"]
    gs_pos, gs_R = np.array(gp["position_m"]), np.array(gp["rotation_matrix"])
    gs_obb = np.array(gp["obb_extents"])
    draw_frame(ax2, gs_R, gs_pos, fl, lw=2.5, show_axis_labels=True)
    draw_obb_wireframe(ax2, gs_R, gs_pos, gs_obb, "#e74c3c", lw=0.7, alpha=0.4)
    ax2.text(gs_pos[0], gs_pos[1]+0.8, gs_pos[2]+0.8,
             "gs2mesh\n(COLMAP coord)", fontsize=9, fontweight="bold",
             color="#c0392b", ha="center")

    # COLMAP crop
    ccp = data["colmap_crop_pose"]
    col_pos, col_R = np.array(ccp["position_m"]), np.array(ccp["rotation_matrix"])
    col_obb = np.array(ccp["obb_extents"])
    draw_frame(ax2, col_R, col_pos, fl, lw=2, alpha=0.7, show_axis_labels=False)
    draw_obb_wireframe(ax2, col_R, col_pos, col_obb, "#27ae60", lw=0.6, alpha=0.3)
    ax2.text(col_pos[0]-1.0, col_pos[1], col_pos[2]+1.0,
             "COLMAP crop\n(scene context)", fontsize=8, fontweight="bold",
             color="#27ae60", ha="center")

    # SAM3D→gs2mesh 정합 연결
    reg = data["registration"]
    dashed_line(ax2, sam_pos, gs_pos, "#f39c12", lw=1.5, alpha=0.7, style="-.")
    midpoint_label(ax2, sam_pos, gs_pos,
                   f"PCA+ICP align\nscale={reg['scale_ratio']:.2f}\nRMSE={reg['rmse']:.3f}",
                   "#e67e22", fontsize=7, offset=[0, 0.3, 0.3])

    # gs2mesh↔COLMAP crop 교차검증
    cv_rmse = data["cross_validation"]["gs2mesh_vs_colmap_crop_rmse"]
    dashed_line(ax2, gs_pos, col_pos, "#27ae60", lw=1.0, alpha=0.4)
    midpoint_label(ax2, gs_pos, col_pos,
                   f"cross-val RMSE={cv_rmse:.3f}", "#27ae60", fontsize=6,
                   offset=[0, -0.3, -0.2])

    # COLMAP 원점
    draw_frame(ax2, np.eye(3), np.zeros(3), 0.4, label="origin",
               lw=1.0, font=7, alpha=0.25)

    pts_colmap = [[0,0,0], sam_pos.tolist(), gs_pos.tolist(), col_pos.tolist()]
    set_equal_aspect(ax2, pts_colmap)
    ax2.view_init(elev=22, azim=-40)

    # ── 하단 정보 ──
    bridge = data["bridge_colmap_to_cam0"]
    info = (
        f"T_cam0_colmap:  scale={bridge['scale']:.4f}  "
        f"ICP RMSE={bridge['rmse_m']*1000:.1f}mm  "
        f"inlier={bridge['icp_inlier_ratio']*100:.1f}%  "
        f"converged={bridge['icp_converged']}    |    "
        f"Axis color: X=Red  Y=Green  Z=Blue"
    )
    fig.text(0.5, 0.01, info, fontsize=8, ha="center",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ecf0f1", ec="#bdc3c7"))

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path = os.path.join(base, "output", "pose_visualization.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[SAVE] {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
