# ply_to_mesh.py
# 포인트클라우드 PLY → 컬러 Mesh PLY 변환
#
# 사용법:
#   pip install open3d
#
#   # 단일 파일
#   python ply_to_mesh.py --input ./data/rgbd_capture/ply/frame_000000.ply
#
#   # 폴더 내 전체 PLY 일괄 변환
#   python ply_to_mesh.py --input_dir ./data/rgbd_capture/ply
#
#   # Poisson depth 조절 (높을수록 디테일, 느림)
#   python ply_to_mesh.py --input_dir ./data/rgbd_capture/ply --depth 9 --view

import os
import glob
import argparse
from typing import Optional

import numpy as np


def pointcloud_to_mesh(input_path: str, output_path: str, depth: int = 8,
                       density_threshold_quantile: float = 0.05,
                       view: bool = False):
    import open3d as o3d

    print(f"[LOAD] {os.path.basename(input_path)}")
    pcd = o3d.io.read_point_cloud(input_path)
    n_pts = len(pcd.points)
    if n_pts == 0:
        print(f"  -> SKIP (0 points)")
        return False
    print(f"  포인트: {n_pts:,}")

    # 1) normal 추정 (mesh 생성에 필수)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 0.0]))

    # 2) Poisson surface reconstruction
    print(f"  Poisson reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, linear_fit=True
    )

    # 3) 밀도 낮은 면 제거 (외곽 노이즈)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, density_threshold_quantile)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # 4) 원본 포인트 bounding box로 crop (Poisson이 바깥으로 늘어나는 것 방지)
    bbox = pcd.get_axis_aligned_bounding_box()
    margin = 0.005  # 5mm 여유
    bbox_min = np.asarray(bbox.min_bound) - margin
    bbox_max = np.asarray(bbox.max_bound) + margin
    crop_box = o3d.geometry.AxisAlignedBoundingBox(bbox_min, bbox_max)
    mesh = mesh.crop(crop_box)

    # 5) vertex color 매핑 (원본 포인트 색상에서)
    if not mesh.has_vertex_colors():
        print("  vertex color 매핑 중...")
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_vertices = np.asarray(mesh.vertices)
        pcd_colors = np.asarray(pcd.colors)
        mesh_colors = np.zeros_like(mesh_vertices)
        for i in range(len(mesh_vertices)):
            _, idx, _ = pcd_tree.search_knn_vector_3d(mesh_vertices[i], 1)
            mesh_colors[i] = pcd_colors[idx[0]]
        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    n_triangles = len(mesh.triangles)
    n_vertices = len(mesh.vertices)
    print(f"  결과: {n_vertices:,} vertices, {n_triangles:,} triangles")

    # 6) 저장
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"  [SAVE] {output_path}")

    # 7) 뷰어
    if view:
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f"Mesh - {os.path.basename(input_path)}",
            width=1280, height=720,
            mesh_show_back_face=True,
        )

    return True


def main():
    parser = argparse.ArgumentParser(description="PLY 포인트클라우드 → Mesh 변환")
    parser.add_argument("--input", type=str, default=None, help="단일 PLY 파일")
    parser.add_argument("--input_dir", type=str, default=None, help="PLY 폴더 (일괄 변환)")
    parser.add_argument("--out_dir", type=str, default=None, help="출력 폴더 (기본: input 옆에 mesh/ 폴더)")
    parser.add_argument("--depth", type=int, default=8, help="Poisson depth (8=보통, 9=고품질, 10=매우세밀)")
    parser.add_argument("--density_quantile", type=float, default=0.05, help="밀도 하위 N%% 면 제거 (0.01~0.1)")
    parser.add_argument("--view", action="store_true", help="변환 후 Open3D 뷰어로 표시")
    args = parser.parse_args()

    if args.input is None and args.input_dir is None:
        parser.error("--input 또는 --input_dir 중 하나를 지정하세요.")

    try:
        import open3d
    except ImportError:
        print("[ERROR] open3d가 필요합니다: pip install open3d")
        return

    # 단일 파일
    if args.input:
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"파일 없음: {args.input}")
        out_dir = args.out_dir or os.path.join(os.path.dirname(args.input), "mesh")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = os.path.join(out_dir, f"{base}_mesh.ply")
        pointcloud_to_mesh(args.input, out_path, args.depth, args.density_quantile, args.view)
        return

    # 폴더 일괄
    ply_files = sorted(glob.glob(os.path.join(args.input_dir, "frame_*.ply")))
    if not ply_files:
        ply_files = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")))
    if not ply_files:
        raise RuntimeError(f"PLY 파일 없음: {args.input_dir}")

    out_dir = args.out_dir or os.path.join(args.input_dir, "mesh")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[BATCH] {len(ply_files)}개 PLY → mesh 변환")
    print(f"[BATCH] 출력: {os.path.abspath(out_dir)}/\n")

    success, fail = 0, 0
    for i, ply_path in enumerate(ply_files):
        print(f"[{i+1}/{len(ply_files)}]", end=" ")
        base = os.path.splitext(os.path.basename(ply_path))[0]
        out_path = os.path.join(out_dir, f"{base}_mesh.ply")
        ok = pointcloud_to_mesh(ply_path, out_path, args.depth, args.density_quantile,
                                view=(args.view and i == 0))
        if ok:
            success += 1
        else:
            fail += 1
        print()

    print(f"[BATCH 완료] 성공: {success}  실패: {fail}")
    print(f"[BATCH 결과] {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
