#!/usr/bin/env python3
"""
포즈 추정 결과 시각화 — 방법별 개별 이미지 생성
각 방법(GLB+ICP, PCA, PnP)마다 3개 뷰(3D, 탑뷰, 사이드뷰)를 한 장에 저장
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── 경로 설정 ──
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
DATA_DIR = SCRIPT_DIR / "data"
CALIB_DIR = DATA_DIR / "cube_session_01" / "calib_out_cube"

# ── 공통 데이터 로드 ──

def load_object_pcd():
    pcd = o3d.io.read_point_cloud(str(OUTPUT_DIR / "object_pointcloud.ply"))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    return pts, colors


def load_cameras():
    """cam0~2의 world(cam0) 좌표계 위치 및 방향 로드"""
    cams = []
    # cam0 = identity
    cams.append({"id": 0, "T_world": np.eye(4), "pos": np.zeros(3)})
    for i, name in enumerate(["T_C0_C1", "T_C0_C2"], start=1):
        path = CALIB_DIR / f"{name}.npy"
        if path.exists():
            T = np.load(str(path))
            T_inv = np.linalg.inv(T)
            cams.append({"id": i, "T_world": T_inv, "pos": T_inv[:3, 3]})
    return cams


def load_glb_points():
    glb_path = DATA_DIR / "reference_knife.glb"
    if not glb_path.exists():
        return None
    scene_or_mesh = trimesh.load(str(glb_path))
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene_or_mesh.dump())
    else:
        mesh = scene_or_mesh
    return mesh.sample(15000)


def load_pose(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    return {
        "translation": data["translation"],
        "rotation_matrix": data["rotation_matrix"],
        "euler_xyz_deg": data["euler_xyz_deg"],
        "transform_4x4": data["transform_4x4"],
        "fitness": float(data["fitness"]),
        "rmse": float(data["rmse"]),
        "method": str(data["method"]),
    }


# ── 그리기 함수 ──

def draw_cameras(ax, cams, cam_colors):
    """카메라 위치 + frustum 방향"""
    cam_labels = ["cam0", "cam1", "cam2"]
    for cam, color, label in zip(cams, cam_colors, cam_labels):
        pos = cam["pos"]
        ax.scatter(pos[0], pos[2], pos[1],
                   c=[color], s=100, marker='^', edgecolors='white',
                   linewidths=0.8, zorder=10)
        ax.text(pos[0], pos[2], pos[1] + 0.025, label,
                fontsize=7, ha='center', color=color, fontweight='bold')

        # 카메라 바라보는 방향 (z축)
        R = cam["T_world"][:3, :3]
        z_dir = R[:, 2] * 0.06
        ax.quiver(pos[0], pos[2], pos[1],
                  z_dir[0], z_dir[2], z_dir[1],
                  color=color, arrow_length_ratio=0.15,
                  linewidth=1.2, alpha=0.7)


def draw_pointcloud(ax, pts, colors, alpha=0.3, size=0.8):
    """점군을 다운샘플하여 그리기"""
    step = max(1, len(pts) // 4000)
    pts_ds = pts[::step]
    if colors is not None:
        col_ds = colors[::step]
        ax.scatter(pts_ds[:, 0], pts_ds[:, 2], pts_ds[:, 1],
                   c=col_ds, s=size, alpha=alpha)
    else:
        ax.scatter(pts_ds[:, 0], pts_ds[:, 2], pts_ds[:, 1],
                   c='silver', s=size, alpha=alpha)


def draw_pose_axes(ax, T, size=0.06, linewidth=2.5):
    """포즈의 XYZ축 화살표"""
    t = T[:3, 3]
    axis_colors = ['#FF3333', '#33CC33', '#3366FF']  # X=R, Y=G, Z=B
    axis_labels = ['X', 'Y', 'Z']
    for i in range(3):
        d = T[:3, i] * size
        ax.quiver(t[0], t[2], t[1],
                  d[0], d[2], d[1],
                  color=axis_colors[i], arrow_length_ratio=0.18,
                  linewidth=linewidth, zorder=15)


def draw_glb_model(ax, model_pts, T_pose, obj_pts, color='red', alpha=0.25):
    """GLB 모델을 스케일+포즈 변환해서 그리기"""
    obj_extent = obj_pts.max(axis=0) - obj_pts.min(axis=0)
    mod_extent = model_pts.max(axis=0) - model_pts.min(axis=0)
    scale = np.median(obj_extent / mod_extent)

    center = model_pts.mean(axis=0)
    scaled = (model_pts - center) * scale + center

    ones = np.ones((len(scaled), 1))
    hom = np.hstack([scaled, ones])
    transformed = (T_pose @ hom.T)[:3].T

    step = max(1, len(transformed) // 3000)
    pts_ds = transformed[::step]
    ax.scatter(pts_ds[:, 0], pts_ds[:, 2], pts_ds[:, 1],
               c=color, s=0.6, alpha=alpha)


def set_equal_aspect(ax, pts):
    """3D 축 비율 맞추기"""
    center = pts.mean(axis=0)
    max_range = (pts.max(axis=0) - pts.min(axis=0)).max() * 0.6
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[2] - max_range, center[2] + max_range)
    ax.set_zlim(center[1] - max_range, center[1] + max_range)


# ── 방법별 시각화 ──

def visualize_method(method_name, pose, obj_pts, obj_colors, cams,
                     model_pts=None, show_model=False):
    """하나의 방법에 대해 3뷰 이미지 생성"""

    cam_colors = [[0.2, 0.8, 0.2], [0.2, 0.2, 1.0], [1.0, 0.8, 0.0]]
    T = pose["transform_4x4"]
    t = pose["translation"]
    euler = pose["euler_xyz_deg"]

    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor('#1a1a2e')
    gs = GridSpec(1, 3, figure=fig, wspace=0.25)

    views = [
        (gs[0, 0], 25, -55, "Perspective View"),
        (gs[0, 1], 90, -90, "Top View (XZ plane)"),
        (gs[0, 2], 0, -90, "Side View (XY plane)"),
    ]

    # 방법별 강조 색상
    method_colors = {
        "GLB+ICP": "#FF6B35",
        "PCA": "#A855F7",
        "PnP": "#06B6D4",
    }
    accent = method_colors.get(method_name, "orange")

    for sub, (gs_pos, elev, azim, view_title) in enumerate(views):
        ax = fig.add_subplot(gs_pos, projection='3d')
        ax.set_facecolor('#16213e')

        # 점군
        draw_pointcloud(ax, obj_pts, obj_colors, alpha=0.35, size=0.6)

        # GLB 모델 (GLB+ICP 방법일 때만)
        if show_model and model_pts is not None:
            draw_glb_model(ax, model_pts, T, obj_pts, color='#FF4444', alpha=0.3)

        # 카메라
        draw_cameras(ax, cams, cam_colors)

        # 포즈 위치 마커
        ax.scatter(t[0], t[2], t[1], c=accent, s=150,
                   marker='*', edgecolors='white', linewidths=0.8, zorder=20)

        # 포즈 축
        draw_pose_axes(ax, T, size=0.07, linewidth=2.5)

        # 포즈 위치에서 물체 중심까지 점선
        obj_center = obj_pts.mean(axis=0)
        ax.plot([t[0], obj_center[0]], [t[2], obj_center[2]], [t[1], obj_center[1]],
                '--', color=accent, alpha=0.4, linewidth=1)

        set_equal_aspect(ax, obj_pts)
        ax.set_xlabel("X (m)", fontsize=8, color='white', labelpad=2)
        ax.set_ylabel("Z (m)", fontsize=8, color='white', labelpad=2)
        ax.set_zlabel("Y (m)", fontsize=8, color='white', labelpad=2)
        ax.tick_params(labelsize=6, colors='gray')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_title, fontsize=10, color='white', pad=5)

        # grid style
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')

    # 상단 타이틀
    title_text = f"{method_name} Pose Estimation"
    fig.suptitle(title_text, fontsize=16, fontweight='bold', color=accent, y=0.98)

    # 하단 정보 박스
    info_lines = [
        f"Position: ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}) m",
        f"Rotation (Euler XYZ): ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}) deg",
        f"Fitness: {pose['fitness']:.4f}   RMSE: {pose['rmse']:.6f} m",
    ]
    info_text = "   |   ".join(info_lines)
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=9, color='#cccccc',
             fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#0f3460', alpha=0.8,
                       edgecolor='#555555'))

    # 범례 (왼쪽 하단)
    legend_lines = [
        "Colored points = Object (RGB-D)",
        "Triangles = Cameras (green=cam0, blue=cam1, yellow=cam2)",
        f"Star = {method_name} pose position",
        "Axes: X=Red  Y=Green  Z=Blue",
    ]
    if show_model:
        legend_lines.insert(1, "Red points = GLB model (transformed)")
    legend_text = "\n".join(legend_lines)
    fig.text(0.01, 0.02, legend_text, fontsize=7, color='#aaaaaa',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.9,
                       edgecolor='#444444'))

    save_path = OUTPUT_DIR / f"pose_vis_{method_name.replace('+','_')}.png"
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"  -> {save_path.name}")
    return save_path


def make_o3d_axes(T, size=0.05):
    """4x4 변환행렬 위치에 XYZ 축(RGB) 선분 + 끝 구 생성"""
    origin = T[:3, 3]
    geoms = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for i in range(3):
        end = origin + T[:3, i] * size
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.array([origin, end]))
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([colors[i]])
        geoms.append(line)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.15)
        sphere.translate(end)
        sphere.paint_uniform_color(colors[i])
        geoms.append(sphere)
    return geoms


def make_o3d_frustum(T_world, color):
    """카메라 frustum (피라미드) 생성"""
    R = T_world[:3, :3]
    t = T_world[:3, 3]
    s, d = 0.03, 0.06
    local_pts = np.array([
        [0, 0, 0], [-s, -s, d], [s, -s, d], [s, s, d], [-s, s, d],
    ])
    world_pts = (R @ local_pts.T).T + t
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(world_pts)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector([color] * len(lines))
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
    sphere.translate(world_pts[0])
    sphere.paint_uniform_color(color)
    return [frustum, sphere]


def open_o3d_viewer(method_name, pose, obj_pcd_o3d, cams, model_pts_raw=None, show_model=False):
    """하나의 방법에 대해 Open3D 인터랙티브 뷰어 열기"""
    geoms = []

    # 물체 점군
    geoms.append(obj_pcd_o3d)

    T = pose["transform_4x4"]
    t = pose["translation"]

    # GLB 모델 (GLB+ICP만)
    if show_model and model_pts_raw is not None:
        obj_pts = np.asarray(obj_pcd_o3d.points)
        obj_extent = obj_pts.max(axis=0) - obj_pts.min(axis=0)
        mod_extent = model_pts_raw.max(axis=0) - model_pts_raw.min(axis=0)
        scale = np.median(obj_extent / mod_extent)
        center = model_pts_raw.mean(axis=0)
        scaled = (model_pts_raw - center) * scale + center
        model_pcd = o3d.geometry.PointCloud()
        model_pcd.points = o3d.utility.Vector3dVector(scaled)
        model_pcd.transform(T)
        model_pcd.paint_uniform_color([1.0, 0.2, 0.2])
        geoms.append(model_pcd)

    # 포즈 축
    geoms.extend(make_o3d_axes(T, size=0.08))

    # 포즈 위치 마커
    marker_colors = {"GLB+ICP": [1, 0.42, 0.2], "PCA": [0.66, 0.33, 0.97], "PnP": [0.02, 0.71, 0.83]}
    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    marker.translate(t)
    marker.paint_uniform_color(marker_colors.get(method_name, [1, 0.5, 0]))
    geoms.append(marker)

    # 카메라들
    cam_colors = [[0.2, 0.8, 0.2], [0.2, 0.2, 1.0], [1.0, 0.8, 0.0]]
    for cam, color in zip(cams, cam_colors):
        geoms.extend(make_o3d_frustum(cam["T_world"], color))
        geoms.extend(make_o3d_axes(cam["T_world"], size=0.04))

    # 월드 좌표축
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0]))

    # 뷰어 열기
    euler = pose["euler_xyz_deg"]
    print(f"\n  [Open3D] {method_name} 뷰어 열림 — 창을 닫으면 다음 방법으로 넘어갑니다")
    print(f"    Position: ({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")
    print(f"    Rotation: ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f})")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"[{method_name}] Pose Estimation", width=1200, height=800)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.08, 0.08, 0.12])
    opt.point_size = 2.0
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    vis.run()
    vis.destroy_window()


def main():
    print("=" * 55)
    print(" Method-wise Pose Visualization")
    print("=" * 55)

    obj_pts, obj_colors = load_object_pcd()
    cams = load_cameras()
    model_pts = load_glb_points()

    # Open3D용 점군 객체
    obj_pcd_o3d = o3d.io.read_point_cloud(str(OUTPUT_DIR / "object_pointcloud.ply"))

    print(f"Object: {len(obj_pts)} pts | Cameras: {len(cams)} | GLB: {'loaded' if model_pts is not None else 'N/A'}")

    methods = {
        "GLB+ICP": {"file": "pose_Reference_Matching.npz", "show_model": False},
        "PCA":     {"file": "pose_PCA.npz",        "show_model": False},
        "PnP":     {"file": "pose_MultiView_PnP.npz", "show_model": False},
    }

    saved = []
    for method_name, cfg in methods.items():
        pose = load_pose(cfg["file"])
        if pose is None:
            print(f"  [{method_name}] pose file not found, skipped")
            continue

        print(f"\n[{method_name}]")
        print(f"  Position: ({pose['translation'][0]:+.4f}, "
              f"{pose['translation'][1]:+.4f}, {pose['translation'][2]:+.4f}) m")
        print(f"  Rotation: ({pose['euler_xyz_deg'][0]:+.1f}, "
              f"{pose['euler_xyz_deg'][1]:+.1f}, {pose['euler_xyz_deg'][2]:+.1f}) deg")
        print(f"  Fitness={pose['fitness']:.4f}  RMSE={pose['rmse']:.6f} m")

        # matplotlib 정적 이미지
        path = visualize_method(
            method_name, pose, obj_pts, obj_colors, cams,
            model_pts=model_pts, show_model=cfg["show_model"],
        )
        saved.append(path)

    # ── 비교 요약 이미지 (3방법 나란히) ──
    print("\n[Comparison Summary]")
    make_comparison(obj_pts, obj_colors, cams, methods, model_pts)

    print(f"\nAll images saved to: {OUTPUT_DIR}/")

    # ── Open3D 인터랙티브 뷰어 (방법별 순차) ──
    print("\n" + "=" * 55)
    print(" Open3D Interactive Viewers (close window to proceed)")
    print("=" * 55)

    for method_name, cfg in methods.items():
        pose = load_pose(cfg["file"])
        if pose is None:
            continue
        open_o3d_viewer(
            method_name, pose, obj_pcd_o3d, cams,
            model_pts_raw=model_pts, show_model=cfg["show_model"],
        )

    print("\nAll viewers closed.")


def make_comparison(obj_pts, obj_colors, cams, methods, model_pts):
    """3가지 방법을 한 이미지에 나란히 비교"""
    cam_colors = [[0.2, 0.8, 0.2], [0.2, 0.2, 1.0], [1.0, 0.8, 0.0]]
    method_colors = {"GLB+ICP": "#FF6B35", "PCA": "#A855F7", "PnP": "#06B6D4"}

    fig = plt.figure(figsize=(21, 7))
    fig.patch.set_facecolor('#1a1a2e')

    for idx, (method_name, cfg) in enumerate(methods.items()):
        pose = load_pose(cfg["file"])
        if pose is None:
            continue

        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        ax.set_facecolor('#16213e')
        accent = method_colors[method_name]

        T = pose["transform_4x4"]
        t = pose["translation"]
        euler = pose["euler_xyz_deg"]

        draw_pointcloud(ax, obj_pts, obj_colors, alpha=0.3, size=0.5)

        if cfg["show_model"] and model_pts is not None:
            draw_glb_model(ax, model_pts, T, obj_pts, color='#FF4444', alpha=0.25)

        draw_cameras(ax, cams, cam_colors)

        ax.scatter(t[0], t[2], t[1], c=accent, s=200,
                   marker='*', edgecolors='white', linewidths=0.8, zorder=20)
        draw_pose_axes(ax, T, size=0.06, linewidth=2.5)

        set_equal_aspect(ax, obj_pts)
        ax.set_xlabel("X", fontsize=7, color='white')
        ax.set_ylabel("Z", fontsize=7, color='white')
        ax.set_zlabel("Y", fontsize=7, color='white')
        ax.tick_params(labelsize=5, colors='gray')
        ax.view_init(elev=25, azim=-55)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')

        info = (f"Pos: ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})\n"
                f"Rot: ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f})\n"
                f"Fit: {pose['fitness']:.4f}  RMSE: {pose['rmse']:.4f}")
        ax.set_title(f"{method_name}\n", fontsize=13, fontweight='bold', color=accent)
        ax.text2D(0.5, -0.02, info, transform=ax.transAxes, fontsize=8,
                  ha='center', color='#cccccc',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', alpha=0.8))

    fig.suptitle("Pose Estimation Comparison (3 Methods)",
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    save_path = OUTPUT_DIR / "pose_comparison.png"
    plt.savefig(str(save_path), dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"  -> {save_path.name}")


if __name__ == "__main__":
    main()
