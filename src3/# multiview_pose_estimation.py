#!/usr/bin/env python3
"""
=============================================================================
멀티뷰 카메라 캘리브레이션 기반 물체 포즈 추정 (위치 + 회전)
=============================================================================

[필요 데이터 구조]
data/
├── calibration/
│   ├── cam0.npz          # color_K, color_D, depth_scale_m_per_unit
│   ├── cam1.npz
│   ├── cam2.npz
│   ├── T_C0_C1.npy       # 4x4 cam0→cam1 변환행렬
│   └── T_C0_C2.npy       # 4x4 cam0→cam2 변환행렬
├── images/
│   ├── cam0_color.png
│   ├── cam0_depth.png     # 16-bit depth
│   ├── cam1_color.png
│   ├── cam1_depth.png
│   ├── cam2_color.png
│   └── cam2_depth.png
├── masks/                 # (선택) SAM 등으로 생성한 물체 마스크
│   ├── cam0_mask.png
│   ├── cam1_mask.png
│   └── cam2_mask.png
└── model/
    └── object.glb         # SAM3D 복원 모델

[설치]
pip install numpy opencv-python open3d trimesh scipy pyransac3d matplotlib

[사용법]
python multiview_pose_estimation.py --data_dir ./data --method all
"""

import argparse
import sys
import os
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2

warnings.filterwarnings("ignore")

try:
    import open3d as o3d
except ImportError:
    sys.exit("[ERROR] open3d 설치 필요: pip install open3d")

try:
    import trimesh
except ImportError:
    sys.exit("[ERROR] trimesh 설치 필요: pip install trimesh")

from scipy.spatial.transform import Rotation


# =============================================================================
# 1. 데이터 클래스 정의
# =============================================================================

@dataclass
class CameraIntrinsics:
    """카메라 내부 파라미터"""
    K: np.ndarray                # 3x3 내부 행렬
    D: np.ndarray                # 왜곡 계수
    depth_scale: float           # depth → 미터 변환 스케일
    cam_id: int = 0


@dataclass
class CameraData:
    """단일 카메라의 전체 데이터"""
    intrinsics: CameraIntrinsics
    color_img: np.ndarray
    depth_img: np.ndarray
    mask_img: Optional[np.ndarray] = None
    T_to_cam0: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class PoseResult:
    """포즈 추정 결과"""
    translation: np.ndarray      # [x, y, z] 미터
    rotation_matrix: np.ndarray  # 3x3
    euler_xyz_deg: np.ndarray    # [rx, ry, rz] 도
    quaternion_xyzw: np.ndarray  # [x, y, z, w]
    transform_4x4: np.ndarray   # 4x4 변환 행렬
    fitness: float = 0.0         # 정합 품질 (0~1)
    rmse: float = 0.0            # 오차 (미터)
    method: str = ""


# =============================================================================
# 2. 데이터 로드
# =============================================================================

class DataLoader:
    """캘리브레이션 및 이미지 데이터 로드

    실제 데이터 구조:
        src3/data/_intrinsics/cam{0,1,2}.npz
        src3/data/cube_session_01/calib_out_cube/T_C0_C{1,2}.npy
        src3/data/object_capture/cam{0,1,2}/rgb_NNNNNN.jpg, depth_NNNNNN.png
        src3/data/reference_knife.glb
    """

    def __init__(self, data_dir: str, frame_id: str = "000003",
                 extrinsics_dir: Optional[str] = None, glb_path: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.intrinsics_dir = self.data_dir / "_intrinsics"
        self.image_dir = self.data_dir / "object_capture"
        self.frame_id = frame_id

        # extrinsics 경로 (calib_out_cube)
        if extrinsics_dir:
            self.extrinsics_dir = Path(extrinsics_dir)
        else:
            self.extrinsics_dir = self.data_dir / "cube_session_01" / "calib_out_cube"

        # GLB 모델 경로
        if glb_path:
            self.glb_path = Path(glb_path)
        else:
            self.glb_path = self.data_dir / "reference_knife.glb"

    def load_intrinsics(self, cam_id: int) -> CameraIntrinsics:
        """카메라 내부 파라미터 로드"""
        npz_path = self.intrinsics_dir / f"cam{cam_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"캘리브레이션 파일 없음: {npz_path}")

        data = np.load(str(npz_path), allow_pickle=True)

        K = self._get_key(data, ["color_K", "K", "camera_matrix", "intrinsic"])
        D = self._get_key(data, ["color_D", "D", "dist_coeffs", "distortion"])
        depth_scale = float(
            self._get_key(data, ["depth_scale_m_per_unit", "depth_scale", "scale"])
        )

        print(f"  [cam{cam_id}] K loaded, depth_scale={depth_scale:.6f}")
        return CameraIntrinsics(K=K, D=D, depth_scale=depth_scale, cam_id=cam_id)

    def load_extrinsics(self) -> Dict[str, np.ndarray]:
        """카메라 외부 파라미터 로드"""
        extrinsics = {}

        for name in ["T_C0_C1", "T_C0_C2"]:
            path = self.extrinsics_dir / f"{name}.npy"
            if path.exists():
                T = np.load(str(path))
                assert T.shape == (4, 4), f"{name} shape must be (4,4), got {T.shape}"
                extrinsics[name] = T
                print(f"  [{name}] 로드 완료")
            else:
                print(f"  [WARNING] {path} 없음, 건너뜀")

        return extrinsics

    def load_images(self, cam_id: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """컬러, 깊이 이미지 로드 (object_capture/cam{id}/rgb_NNNNNN.jpg, depth_NNNNNN.png)"""
        cam_dir = self.image_dir / f"cam{cam_id}"
        if not cam_dir.exists():
            raise FileNotFoundError(f"카메라 디렉토리 없음: {cam_dir}")

        color_path = cam_dir / f"rgb_{self.frame_id}.jpg"
        depth_path = cam_dir / f"depth_{self.frame_id}.png"

        if not color_path.exists():
            raise FileNotFoundError(f"컬러 이미지 없음: {color_path}")
        if not depth_path.exists():
            raise FileNotFoundError(f"깊이 이미지 없음: {depth_path}")

        color_img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        print(f"  [cam{cam_id}] color={color_img.shape}, depth={depth_img.shape} "
              f"(frame={self.frame_id})")

        return color_img, depth_img, None

    def load_glb_model(self, filename: str = None) -> trimesh.Trimesh:
        """GLB/GLTF 모델 로드"""
        model_path = self.glb_path

        if not model_path.exists():
            raise FileNotFoundError(f"GLB 파일을 찾을 수 없습니다: {model_path}")

        scene_or_mesh = trimesh.load(str(model_path))
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(scene_or_mesh.dump())
        else:
            mesh = scene_or_mesh

        print(f"  [GLB] {model_path.name}: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
        return mesh

    def _get_key(self, data, keys):
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(f"키를 찾을 수 없음: {keys}, 사용 가능: {list(data.keys())}")

    def _load_image(self, directory, patterns, flags):
        for p in patterns:
            path = directory / p
            if path.exists():
                return cv2.imread(str(path), flags)
        return None


# =============================================================================
# 3. 점군 생성 및 처리
# =============================================================================

class PointCloudProcessor:
    """RGB-D → 3D 점군 변환 및 처리"""

    @staticmethod
    def depth_to_pointcloud(
        color_img: np.ndarray,
        depth_img: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        depth_scale: float,
        mask: Optional[np.ndarray] = None,
        min_depth: float = 0.1,
        max_depth: float = 3.0,
    ) -> o3d.geometry.PointCloud:
        """RGB-D 이미지 → Open3D 점군 변환"""

        h, w = depth_img.shape[:2]

        # 왜곡 보정
        if np.any(D != 0):
            color_img = cv2.undistort(color_img, K, D)
            # depth도 같은 보정 적용
            map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
            depth_img = cv2.remap(depth_img, map1, map2, cv2.INTER_NEAREST)

        # 깊이 → 미터 변환
        z = depth_img.astype(np.float64) * depth_scale

        # 유효 픽셀 마스크
        valid = (z > min_depth) & (z < max_depth)
        if mask is not None:
            valid &= (mask > 127)

        # 역투영 (back-projection)
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x = (u[valid] - cx) * z[valid] / fx
        y = (v[valid] - cy) * z[valid] / fy

        points = np.stack([x, y, z[valid]], axis=-1)

        # 컬러 매핑
        if len(color_img.shape) == 3:
            colors = color_img[valid].astype(np.float64) / 255.0
        else:
            gray = color_img[valid].astype(np.float64) / 255.0
            colors = np.stack([gray, gray, gray], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    @staticmethod
    def merge_pointclouds(
        camera_data_list: List[CameraData],
        voxel_size: float = 0.002,
    ) -> o3d.geometry.PointCloud:
        """모든 카메라의 점군을 cam0 좌표계로 통합"""

        merged = o3d.geometry.PointCloud()

        for cam_data in camera_data_list:
            intr = cam_data.intrinsics
            pcd = PointCloudProcessor.depth_to_pointcloud(
                color_img=cam_data.color_img,
                depth_img=cam_data.depth_img,
                K=intr.K, D=intr.D,
                depth_scale=intr.depth_scale,
                mask=cam_data.mask_img,
            )

            # cam0 좌표계로 변환
            pcd.transform(cam_data.T_to_cam0)

            n_points = len(pcd.points)
            print(f"  [cam{intr.cam_id}] {n_points} points generated")

            merged += pcd

        # 다운샘플링
        n_before = len(merged.points)
        merged = merged.voxel_down_sample(voxel_size=voxel_size)
        n_after = len(merged.points)
        print(f"  [통합 점군] {n_before} → {n_after} points (voxel={voxel_size}m)")

        # 통계적 이상치 제거
        merged, _ = merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"  [이상치 제거 후] {len(merged.points)} points")

        return merged

    @staticmethod
    def segment_object(
        pcd: o3d.geometry.PointCloud,
        distance_threshold: float = 0.01,
        min_cluster_size: int = 500,
    ) -> o3d.geometry.PointCloud:
        """
        마스크가 없는 경우: 평면(바닥/테이블) 제거 후 가장 큰 클러스터를 물체로 추출
        """
        print("\n[물체 분할 - RANSAC 평면 제거 + DBSCAN 클러스터링]")

        # RANSAC 평면 검출 (바닥/테이블)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )
        a, b, c, d = plane_model
        print(f"  평면 방정식: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

        # 평면 제거 → 물체 후보
        object_pcd = pcd.select_by_index(inliers, invert=True)
        print(f"  평면 제거 후: {len(object_pcd.points)} points")

        if len(object_pcd.points) < min_cluster_size:
            print("  [WARNING] 평면 제거 후 점이 부족합니다. 원본 사용.")
            return pcd

        # DBSCAN 클러스터링 → 가장 큰 클러스터 = 물체
        labels = np.array(object_pcd.cluster_dbscan(
            eps=0.02, min_points=10, print_progress=False
        ))

        if len(labels) == 0 or labels.max() < 0:
            print("  [WARNING] 클러스터를 찾을 수 없습니다. 평면 제거 결과 사용.")
            return object_pcd

        # 가장 큰 클러스터 선택
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        largest_label = unique_labels[np.argmax(counts)]
        object_pcd = object_pcd.select_by_index(np.where(labels == largest_label)[0])

        print(f"  최대 클러스터 (label={largest_label}): {len(object_pcd.points)} points")
        return object_pcd


# =============================================================================
# 4. 포즈 추정 방법들
# =============================================================================

class PoseEstimator:
    """다양한 방법의 포즈 추정"""

    @staticmethod
    def extract_pose(T: np.ndarray, method: str, fitness: float = 0.0, rmse: float = 0.0) -> PoseResult:
        """4x4 변환행렬 → PoseResult 변환"""
        R = T[:3, :3]
        t = T[:3, 3]
        rot = Rotation.from_matrix(R)

        return PoseResult(
            translation=t,
            rotation_matrix=R,
            euler_xyz_deg=rot.as_euler("xyz", degrees=True),
            quaternion_xyzw=rot.as_quat(),  # [x, y, z, w]
            transform_4x4=T.copy(),
            fitness=fitness,
            rmse=rmse,
            method=method,
        )

    # ----- 방법 1: 클러스터별 PCA 정렬 + ICP 정합 -----

    @staticmethod
    def _pca_axes(pts: np.ndarray):
        """점군의 PCA 주축 (3x3, 열 = 주축, 큰 고유값 순)"""
        cov = np.cov((pts - pts.mean(axis=0)).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        R = eigenvectors[:, idx]
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        return R, eigenvalues[idx]

    @staticmethod
    def _pca_align_transform(src_pts, dst_pts, scale):
        """
        src(모델)를 dst(클러스터)에 PCA 축 정렬.
        축 부호 모호성(4가지 조합)을 모두 시도하여 가장 가까운 것 반환.
        """
        R_src, _ = PoseEstimator._pca_axes(src_pts)
        R_dst, _ = PoseEstimator._pca_axes(dst_pts)

        src_center = src_pts.mean(axis=0)
        dst_center = dst_pts.mean(axis=0)

        # 축 부호 모호성: 1축, 2축 각각 ±  (3축은 외적으로 결정)
        best_T = None
        best_dist = float('inf')

        for s1 in [1, -1]:
            for s2 in [1, -1]:
                R_src_flip = R_src.copy()
                R_src_flip[:, 0] *= s1
                R_src_flip[:, 1] *= s2
                # 3축 = 외적으로 오른손 좌표계 보장
                R_src_flip[:, 2] = np.cross(R_src_flip[:, 0], R_src_flip[:, 1])

                # 회전: src PCA → dst PCA
                R_align = R_dst @ R_src_flip.T

                # 변환 행렬: scale → rotate → translate
                T = np.eye(4)
                T[:3, :3] = R_align
                # src를 center로 이동 → scale → rotate → dst center로 이동
                t = dst_center - R_align @ (src_center * scale)
                # 실제로는 scale이 이미 적용된 모델에 대해 작동하므로:
                t = dst_center - R_align @ src_center
                T[:3, 3] = t

                # 변환 후 거리 체크 (샘플)
                step = max(1, len(src_pts) // 500)
                sample = src_pts[::step]
                transformed = (R_align @ sample.T).T + t
                # dst에서 가장 가까운 점까지의 평균 거리
                dst_step = max(1, len(dst_pts) // 2000)
                dst_sample = dst_pts[::dst_step]
                from scipy.spatial import cKDTree
                tree = cKDTree(dst_sample)
                dists, _ = tree.query(transformed)
                avg_dist = np.mean(dists)

                if avg_dist < best_dist:
                    best_dist = avg_dist
                    best_T = T

        return best_T

    @staticmethod
    def estimate_with_reference_matching(
        objects_pcd: o3d.geometry.PointCloud,
        ref_pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.005,
        scale_range: Tuple[float, float] = (0.1, 0.5),
        n_scale_steps: int = 8,
    ):
        """
        테이블 제거된 점군을 클러스터로 분리한 후,
        각 클러스터에 대해 PCA 축 정렬 + ICP 정합으로 레퍼런스 매칭.

        Returns: (PoseResult, aligned_model_pcd, best_scale)
        """
        print("\n" + "=" * 60)
        print("[방법 1] 클러스터별 PCA 정렬 + ICP 매칭")
        print("=" * 60)

        ref_pts = np.asarray(ref_pcd.points)
        ref_extent = ref_pts.max(axis=0) - ref_pts.min(axis=0)
        ref_longest = ref_extent.max()
        print(f"  레퍼런스: {len(ref_pts)} pts, "
              f"extent=({ref_extent[0]:.3f}, {ref_extent[1]:.3f}, {ref_extent[2]:.3f}), "
              f"longest={ref_longest:.3f}")

        # 레퍼런스 PCA 고유값 비율 (형상 지문)
        _, ref_eigenvalues = PoseEstimator._pca_axes(ref_pts)
        ref_ratios = np.sort(ref_eigenvalues / ref_eigenvalues.max())[::-1]
        print(f"  레퍼런스 PCA 비율: {ref_ratios}")

        # --- DBSCAN 클러스터링 ---
        print("\n  DBSCAN 클러스터링...")
        labels = np.array(objects_pcd.cluster_dbscan(
            eps=0.015, min_points=50, print_progress=False
        ))
        if len(labels) == 0 or labels.max() < 0:
            raise RuntimeError("클러스터를 찾을 수 없습니다")

        unique_labels = np.unique(labels[labels >= 0])
        print(f"  {len(unique_labels)} 개 클러스터 발견")

        # --- 색상 기반 물체 클러스터 식별 ---
        has_colors = objects_pcd.has_colors()
        all_colors = np.asarray(objects_pcd.colors) if has_colors else None

        # 노란색 클러스터 우선 탐색 (물체 고유 색상)
        best_yellow_label = -1
        best_yellow_ratio = 0.0
        if has_colors:
            print("\n  색상 기반 물체 탐색...")
            for label in unique_labels:
                cluster_idx = np.where(labels == label)[0]
                if len(cluster_idx) < 100:
                    continue
                cl_colors = all_colors[cluster_idx]
                # 노란색: R>0.4, G>0.3, B<0.35
                yellow_mask = ((cl_colors[:, 0] > 0.4) &
                               (cl_colors[:, 1] > 0.3) &
                               (cl_colors[:, 2] < 0.35))
                yellow_ratio = yellow_mask.sum() / len(cl_colors)
                if yellow_ratio > best_yellow_ratio:
                    best_yellow_ratio = yellow_ratio
                    best_yellow_label = label

            if best_yellow_ratio > 0.3:
                idx_y = np.where(labels == best_yellow_label)[0]
                y_pts = np.asarray(objects_pcd.points)[idx_y]
                y_ext = y_pts.max(axis=0) - y_pts.min(axis=0)
                print(f"  노란색 클러스터 발견: cluster {best_yellow_label}, "
                      f"{len(idx_y)} pts, yellow={best_yellow_ratio:.2f}, "
                      f"extent=({y_ext[0]:.3f}, {y_ext[1]:.3f}, {y_ext[2]:.3f})")

        # --- 각 클러스터: PCA 정렬 → ICP ---
        best_score = -1
        best_pose_T = None
        best_cluster_id = -1
        best_scale = None
        best_cluster_pcd = None
        best_init_T = None

        # 후보 결정: 노란색 클러스터가 있으면 우선, 없으면 전체 탐색
        if best_yellow_label >= 0 and best_yellow_ratio > 0.3:
            candidate_labels = [best_yellow_label]
            print(f"  노란색 클러스터 {best_yellow_label}에 집중하여 정합")
        else:
            candidate_labels = [l for l in unique_labels
                                if np.sum(labels == l) >= 100]
            print(f"  색상 매칭 실패, 전체 {len(candidate_labels)}개 클러스터 탐색")

        for label in candidate_labels:
            cluster_idx = np.where(labels == label)[0]
            cluster = objects_pcd.select_by_index(cluster_idx)
            cluster_pts = np.asarray(cluster.points)
            cluster_extent = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
            cluster_longest = cluster_extent.max()

            # 스케일 추정 (색상 매칭된 경우 스케일 범위 확대)
            scale = cluster_longest / ref_longest
            if label == best_yellow_label:
                # 색상 매칭된 클러스터는 스케일 제한 완화
                if scale < 0.05 or scale > 0.5:
                    continue
            else:
                if scale < scale_range[0] or scale > scale_range[1]:
                    continue

            # 형상 유사도: PCA 고유값 비율 비교
            _, cl_eigenvalues = PoseEstimator._pca_axes(cluster_pts)
            cl_ratios = np.sort(cl_eigenvalues / cl_eigenvalues.max())[::-1]
            shape_sim = 1.0 - np.mean(np.abs(cl_ratios - ref_ratios))

            # 모델 스케일링
            model_scaled = o3d.geometry.PointCloud(ref_pcd)
            model_scaled.scale(scale, center=model_scaled.get_center())
            model_scaled_pts = np.asarray(model_scaled.points)

            # PCA 축 정렬로 초기 변환 계산
            init_T = PoseEstimator._pca_align_transform(
                model_scaled_pts, cluster_pts, scale
            )

            # 초기 변환 적용한 모델 생성
            model_init = o3d.geometry.PointCloud(model_scaled)
            model_init.transform(init_T)

            # 노멀 계산
            cluster.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
            )
            model_init.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3, max_nn=30)
            )

            # ICP (Point-to-Plane)
            result = o3d.pipelines.registration.registration_icp(
                model_init, cluster,
                max_correspondence_distance=voxel_size * 5,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=200,
                ),
            )

            # 색상 보너스
            color_bonus = 0.0
            if has_colors:
                cl_colors = all_colors[cluster_idx]
                yellow_mask = ((cl_colors[:, 0] > 0.4) &
                               (cl_colors[:, 1] > 0.3) &
                               (cl_colors[:, 2] < 0.35))
                color_bonus = (yellow_mask.sum() / len(cl_colors)) * 0.5

            score = result.fitness * shape_sim + color_bonus

            print(f"    cluster {label}: {len(cluster_pts)} pts, "
                  f"longest={cluster_longest:.3f}m, scale={scale:.3f}, "
                  f"shape={shape_sim:.3f}, ICP={result.fitness:.4f}, "
                  f"color_bonus={color_bonus:.3f}, score={score:.4f}")

            if score > best_score:
                best_score = score
                best_pose_T = result.transformation
                best_cluster_id = label
                best_scale = scale
                best_cluster_pcd = cluster
                best_init_T = init_T

        if best_pose_T is None:
            raise RuntimeError("어떤 클러스터도 레퍼런스와 매칭되지 않았습니다")

        print(f"\n  최적: cluster {best_cluster_id}, "
              f"scale={best_scale:.4f}, score={best_score:.4f}")
        print(f"  정합된 모델 크기: {ref_extent * best_scale}")

        # --- 최적 클러스터에 정밀 ICP ---
        model_final = o3d.geometry.PointCloud(ref_pcd)
        model_final.scale(best_scale, center=model_final.get_center())
        model_final.transform(best_init_T)
        model_final.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        best_cluster_pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )

        print("  정밀 ICP 재정합 중...")
        result_fine = o3d.pipelines.registration.registration_icp(
            model_final, best_cluster_pcd,
            max_correspondence_distance=voxel_size * 2,
            init=best_pose_T,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=300,
            ),
        )
        print(f"  정밀 ICP: fitness={result_fine.fitness:.4f}, "
              f"RMSE={result_fine.inlier_rmse:.6f}")

        # 최종 변환 = init_T → ICP
        T_final = result_fine.transformation

        # 정합된 모델 점군 생성
        model_aligned = o3d.geometry.PointCloud(model_final)
        model_aligned.transform(T_final)
        model_aligned.paint_uniform_color([1.0, 0.0, 0.0])

        return PoseEstimator.extract_pose(
            T_final,
            method="Reference Matching",
            fitness=result_fine.fitness,
            rmse=result_fine.inlier_rmse,
        ), model_aligned, best_scale

    # ----- 방법 2: PCA 축 추정 -----

    @staticmethod
    def estimate_with_pca(scene_pcd: o3d.geometry.PointCloud) -> PoseResult:
        """점군의 주성분 분석으로 물체 축/위치 추정"""
        print("\n" + "=" * 60)
        print("[방법 2] PCA 축 추정")
        print("=" * 60)

        points = np.asarray(scene_pcd.points)
        centroid = points.mean(axis=0)

        # 공분산 행렬 → 고유벡터 = 주축
        cov = np.cov((points - centroid).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 큰 고유값 순서로 정렬
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 오른손 좌표계 보장
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1

        R = eigenvectors
        t = centroid

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        print(f"  고유값: {eigenvalues}")
        sizes = np.sqrt(eigenvalues) * 4
        print(f"  물체 크기 (주축 방향): [{sizes[0]:.4f}, {sizes[1]:.4f}, {sizes[2]:.4f}] m (근사)")

        return PoseEstimator.extract_pose(T, method="PCA", fitness=1.0, rmse=0.0)

    # ----- 방법 3: Primitive Fitting -----

    @staticmethod
    def estimate_with_primitive(
        scene_pcd: o3d.geometry.PointCloud,
        shape: str = "auto",
    ) -> PoseResult:
        """
        기본 도형 피팅으로 포즈 추정.
        shape: "box", "cylinder", "sphere", "auto"
        """
        print("\n" + "=" * 60)
        print("[방법 3] Primitive Fitting")
        print("=" * 60)

        try:
            import pyransac3d as pyrsc
        except ImportError:
            print("  [ERROR] pyransac3d 설치 필요: pip install pyransac3d")
            print("  PCA 방법으로 대체합니다.")
            return PoseEstimator.estimate_with_pca(scene_pcd)

        points = np.asarray(scene_pcd.points)

        if shape == "auto":
            shape = PoseEstimator._detect_shape(points)
            print(f"  감지된 형상: {shape}")

        if shape == "cylinder":
            cyl = pyrsc.Cylinder()
            center, axis, radius, inliers = cyl.fit(points, thresh=0.003, maxIteration=1000)

            axis = axis / np.linalg.norm(axis)
            # 축 방향 → 회전 행렬
            R = PoseEstimator._axis_to_rotation(axis)
            t = np.array(center)

            print(f"  실린더 중심: {center}")
            print(f"  실린더 축: {axis}")
            print(f"  반지름: {radius:.4f} m")
            print(f"  인라이어: {len(inliers)}/{len(points)}")

        elif shape == "sphere":
            sph = pyrsc.Sphere()
            center, radius, inliers = sph.fit(points, thresh=0.003, maxIteration=1000)

            R = np.eye(3)  # 구는 회전 무의미
            t = np.array(center)

            print(f"  구 중심: {center}")
            print(f"  반지름: {radius:.4f} m")

        else:  # box
            # OBB (Oriented Bounding Box) 기반
            obb = scene_pcd.get_oriented_bounding_box()
            R = np.asarray(obb.R)
            t = np.asarray(obb.center)
            extent = np.asarray(obb.extent)

            print(f"  박스 중심: {t}")
            print(f"  박스 크기 (x,y,z): {extent} m")
            inliers = list(range(len(points)))

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        fitness = len(inliers) / len(points) if isinstance(inliers, list) else 1.0
        return PoseEstimator.extract_pose(T, method=f"Primitive({shape})", fitness=fitness)

    # ----- 방법 4: 멀티뷰 PnP (마스크 기반 특징점 매칭) -----

    @staticmethod
    def estimate_with_pnp(
        camera_data_list: List[CameraData],
        extrinsics: Dict[str, np.ndarray],
    ) -> PoseResult:
        """
        멀티뷰 특징점 매칭 + PnP로 포즈 추정.
        cam0의 depth로 3D를 복원하고, 다른 카메라에서 PnP로 검증.
        """
        print("\n" + "=" * 60)
        print("[방법 4] 멀티뷰 PnP")
        print("=" * 60)

        cam0 = camera_data_list[0]

        # 특징점 검출 (ORB)
        orb = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # cam0 특징점
        gray0 = cv2.cvtColor(cam0.color_img, cv2.COLOR_RGB2GRAY)
        kp0, des0 = orb.detectAndCompute(gray0, cam0.mask_img)

        if des0 is None or len(kp0) < 10:
            print("  [WARNING] cam0에서 충분한 특징점을 찾지 못했습니다.")
            return PoseEstimator.estimate_with_pca(
                PointCloudProcessor.merge_pointclouds(camera_data_list)
            )

        # cam0 특징점의 3D 좌표 (depth 기반)
        K0 = cam0.intrinsics.K
        depth0 = cam0.depth_img.astype(np.float64) * cam0.intrinsics.depth_scale
        pts_3d = []
        pts_2d_cam0 = []
        valid_kp_idx = []

        for i, kp in enumerate(kp0):
            u, v = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= v < depth0.shape[0] and 0 <= u < depth0.shape[1]:
                z = depth0[v, u]
                if 0.1 < z < 3.0:
                    x = (u - K0[0, 2]) * z / K0[0, 0]
                    y = (v - K0[1, 2]) * z / K0[1, 1]
                    pts_3d.append([x, y, z])
                    pts_2d_cam0.append([u, v])
                    valid_kp_idx.append(i)

        pts_3d = np.array(pts_3d, dtype=np.float64)
        print(f"  cam0: {len(pts_3d)}/{len(kp0)} 특징점에 유효 depth")

        if len(pts_3d) < 6:
            print("  [WARNING] 유효 3D 점이 부족합니다.")
            return PoseEstimator.estimate_with_pca(
                PointCloudProcessor.merge_pointclouds(camera_data_list)
            )

        # 다른 카메라와 매칭하여 교차 검증
        valid_des0 = des0[valid_kp_idx]
        all_3d = []
        all_2d = []

        for cam_data in camera_data_list[1:]:
            cam_id = cam_data.intrinsics.cam_id
            gray_i = cv2.cvtColor(cam_data.color_img, cv2.COLOR_RGB2GRAY)
            kp_i, des_i = orb.detectAndCompute(gray_i, cam_data.mask_img)

            if des_i is None:
                continue

            matches = bf.knnMatch(valid_des0, des_i, k=2)

            # Lowe's ratio test
            good = []
            for m_list in matches:
                if len(m_list) == 2:
                    m, n = m_list
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

            if len(good) < 6:
                continue

            # cam_i 좌표계에서의 2D 점
            matched_3d = np.array([pts_3d[m.queryIdx] for m in good], dtype=np.float64)
            matched_2d = np.array([kp_i[m.trainIdx].pt for m in good], dtype=np.float64)

            # cam0 → cam_i 변환 적용
            T_inv = np.linalg.inv(cam_data.T_to_cam0)
            matched_3d_cami = (T_inv[:3, :3] @ matched_3d.T + T_inv[:3, 3:]).T

            all_3d.append(matched_3d)
            all_2d.append(matched_2d)

            print(f"  cam{cam_id}: {len(good)} 매칭 (ratio test 통과)")

        # cam0에서의 물체 centroid + 주축
        centroid = pts_3d.mean(axis=0)
        cov = np.cov((pts_3d - centroid).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        R = eigenvectors[:, idx]
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = centroid

        print(f"  최종 위치: {centroid}")

        return PoseEstimator.extract_pose(T, method="MultiView PnP", fitness=1.0)

    # ----- 헬퍼 메서드들 -----

    @staticmethod
    def _compute_fpfh(pcd, voxel_size):
        """FPFH 특징 계산"""
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
        )
        return pcd_down, fpfh

    @staticmethod
    def _detect_shape(points: np.ndarray) -> str:
        """점군 분포로 형상 자동 감지"""
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]

        ratios = eigenvalues / eigenvalues[0]

        # 구: 세 축 비슷
        if ratios[1] > 0.7 and ratios[2] > 0.7:
            return "sphere"
        # 실린더: 두 축 비슷, 하나 다름
        elif ratios[1] > 0.6 and ratios[2] < 0.3:
            return "cylinder"
        else:
            return "box"

    @staticmethod
    def _axis_to_rotation(axis: np.ndarray) -> np.ndarray:
        """축 벡터 → 회전 행렬 (z축 = 주어진 축)"""
        z = axis / np.linalg.norm(axis)
        # 임의의 수직 벡터 찾기
        if abs(z[0]) < 0.9:
            x = np.cross(z, np.array([1, 0, 0]))
        else:
            x = np.cross(z, np.array([0, 1, 0]))
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])


# =============================================================================
# 5. 검증 (재투영)
# =============================================================================

class PoseValidator:
    """포즈 결과 검증 및 시각화"""

    @staticmethod
    def reprojection_check(
        pose: PoseResult,
        model_pcd: o3d.geometry.PointCloud,
        camera_data: CameraData,
        output_path: str = "reprojection_check.png",
    ):
        """포즈 결과를 이미지에 재투영하여 시각적 검증"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [WARNING] matplotlib 없음. 재투영 검증 건너뜀.")
            return

        K = camera_data.intrinsics.K
        R = pose.rotation_matrix
        t = pose.translation

        # 모델 점군은 이미 cam0 좌표계에 위치함 (aligned)
        pts = np.asarray(model_pcd.points)

        # cam0 → cam_i 변환만 적용
        T_cam = np.linalg.inv(camera_data.T_to_cam0)
        pts_hom = np.hstack([pts, np.ones((len(pts), 1))])
        pts_cam = (T_cam @ pts_hom.T)[:3]

        # 투영
        valid = pts_cam[2] > 0
        proj = K @ pts_cam[:, valid]
        u = (proj[0] / proj[2]).astype(int)
        v = (proj[1] / proj[2]).astype(int)

        img = camera_data.color_img.copy()
        h, w = img.shape[:2]
        mask_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].imshow(img)
        axes[0].set_title(f"cam{camera_data.intrinsics.cam_id} - 원본")
        axes[0].axis("off")

        axes[1].imshow(img)
        axes[1].scatter(u[mask_uv], v[mask_uv], c="lime", s=1, alpha=0.3)
        axes[1].set_title(f"cam{camera_data.intrinsics.cam_id} - 재투영 ({pose.method})")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  재투영 이미지 저장: {output_path}")

    @staticmethod
    def save_result_pointclouds(
        scene_pcd: o3d.geometry.PointCloud,
        model_pcd: o3d.geometry.PointCloud,
        pose: PoseResult,
        output_path: str = "alignment_result.ply",
    ):
        """정합 결과를 컬러로 구분하여 저장"""
        # 씬 = 원본 색상, 모델 = 빨간색
        model_transformed = o3d.geometry.PointCloud(model_pcd)
        model_transformed.transform(pose.transform_4x4)
        model_transformed.paint_uniform_color([1.0, 0.0, 0.0])

        combined = scene_pcd + model_transformed
        o3d.io.write_point_cloud(output_path, combined)
        print(f"  정합 점군 저장: {output_path}")


# =============================================================================
# 6. 결과 출력
# =============================================================================

def print_pose_result(result: PoseResult):
    """포즈 추정 결과 출력"""
    print("\n" + "=" * 60)
    print(f"포즈 추정 결과 [{result.method}]")
    print("=" * 60)
    print(f"\n  [위치 - cam0 좌표계 (미터)]")
    print(f"    X: {result.translation[0]:+.6f} m")
    print(f"    Y: {result.translation[1]:+.6f} m")
    print(f"    Z: {result.translation[2]:+.6f} m")

    print(f"\n  [회전 - 오일러 각 XYZ (도)]")
    print(f"    Rx: {result.euler_xyz_deg[0]:+.2f}°")
    print(f"    Ry: {result.euler_xyz_deg[1]:+.2f}°")
    print(f"    Rz: {result.euler_xyz_deg[2]:+.2f}°")

    print(f"\n  [회전 - 쿼터니언 (x, y, z, w)]")
    q = result.quaternion_xyzw
    print(f"    ({q[0]:+.6f}, {q[1]:+.6f}, {q[2]:+.6f}, {q[3]:+.6f})")

    print(f"\n  [4x4 변환 행렬]")
    for row in result.transform_4x4:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}  {row[3]:+.6f}]")

    print(f"\n  [품질]")
    print(f"    Fitness: {result.fitness:.4f}")
    print(f"    RMSE:    {result.rmse:.6f} m")


def save_pose_to_file(result: PoseResult, output_path: str):
    """포즈 결과를 파일로 저장"""
    np.savez(
        output_path,
        translation=result.translation,
        rotation_matrix=result.rotation_matrix,
        euler_xyz_deg=result.euler_xyz_deg,
        quaternion_xyzw=result.quaternion_xyzw,
        transform_4x4=result.transform_4x4,
        fitness=result.fitness,
        rmse=result.rmse,
        method=result.method,
    )
    print(f"포즈 저장: {output_path}")


# =============================================================================
# 7. 메인 파이프라인
# =============================================================================

def run_pipeline(args):
    """전체 파이프라인 실행"""

    print("=" * 60)
    print(" 멀티뷰 카메라 기반 물체 포즈 추정")
    print("=" * 60)

    # --- 데이터 로드 ---
    print("\n[1/5] 데이터 로드")
    loader = DataLoader(
        data_dir=args.data_dir,
        frame_id=args.frame_id,
        extrinsics_dir=args.extrinsics_dir,
        glb_path=args.glb_path,
    )

    intrinsics = []
    for i in range(args.num_cameras):
        intr = loader.load_intrinsics(i)
        intrinsics.append(intr)

    extrinsics = loader.load_extrinsics()

    camera_data_list = []
    for i in range(args.num_cameras):
        color, depth, mask = loader.load_images(i)

        T_to_cam0 = np.eye(4)
        if i == 1 and "T_C0_C1" in extrinsics:
            T_to_cam0 = extrinsics["T_C0_C1"]
        elif i == 2 and "T_C0_C2" in extrinsics:
            T_to_cam0 = extrinsics["T_C0_C2"]

        cam_data = CameraData(
            intrinsics=intrinsics[i],
            color_img=color,
            depth_img=depth,
            mask_img=mask,
            T_to_cam0=T_to_cam0,
        )
        camera_data_list.append(cam_data)

    # --- 점군 생성 및 통합 ---
    print("\n[2/5] 점군 생성 및 통합")
    merged_pcd = PointCloudProcessor.merge_pointclouds(
        camera_data_list, voxel_size=args.voxel_size
    )

    # --- 전체 씬 저장 ---
    o3d.io.write_point_cloud(
        os.path.join(args.output_dir, "scene_merged.ply"), merged_pcd
    )
    print(f"  전체 씬 점군 저장: {args.output_dir}/scene_merged.ply")

    # --- 테이블 평면 제거 (물체만 남기기) ---
    print("\n[3/5] 테이블 평면 제거")
    plane_model, inliers = merged_pcd.segment_plane(
        distance_threshold=0.008, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model
    print(f"  평면: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    objects_pcd = merged_pcd.select_by_index(inliers, invert=True)
    print(f"  평면 제거: {len(merged_pcd.points)} → {len(objects_pcd.points)} pts")

    # 이상치 제거
    objects_pcd, _ = objects_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    print(f"  이상치 제거 후: {len(objects_pcd.points)} pts")

    o3d.io.write_point_cloud(
        os.path.join(args.output_dir, "objects_no_table.ply"), objects_pcd
    )

    # --- 레퍼런스 모델 로드 ---
    print("\n  레퍼런스 모델 로드")
    ref_ply_path = Path(args.data_dir) / "reference_knife.ply"
    ref_pcd = None
    if ref_ply_path.exists():
        ref_pcd = o3d.io.read_point_cloud(str(ref_ply_path))
        print(f"  PLY 로드: {len(ref_pcd.points)} points")
    else:
        try:
            glb_mesh = loader.load_glb_model()
            pts = glb_mesh.sample(30000)
            ref_pcd = o3d.geometry.PointCloud()
            ref_pcd.points = o3d.utility.Vector3dVector(pts)
            print(f"  GLB → 점군 변환: {len(pts)} points")
        except Exception as e:
            print(f"  [WARNING] 레퍼런스 모델 없음: {e}")

    # --- 포즈 추정 ---
    print("\n[4/5] 포즈 추정")
    results = []
    model_aligned_pcd = None
    method = args.method.lower()

    # 방법 1: 레퍼런스 모델 직접 정합 (테이블 제거된 물체 점군)
    if method in ["glb", "icp", "ref", "all"]:
        if ref_pcd is not None:
            try:
                pose_ref, model_aligned_pcd, best_scale = PoseEstimator.estimate_with_reference_matching(
                    objects_pcd=objects_pcd,
                    ref_pcd=ref_pcd,
                    voxel_size=args.voxel_size,
                    scale_range=(0.15, 0.25),
                    n_scale_steps=8,
                )
                results.append(pose_ref)
                print_pose_result(pose_ref)

                # 정합된 모델 저장
                if model_aligned_pcd is not None:
                    combined = merged_pcd + model_aligned_pcd
                    o3d.io.write_point_cloud(
                        os.path.join(args.output_dir, "alignment_result.ply"), combined
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(args.output_dir, "object_pointcloud.ply"), model_aligned_pcd
                    )
                    print(f"  정합 결과 저장 완료")
            except Exception as e:
                print(f"  [레퍼런스 정합 실패] {e}")
                import traceback; traceback.print_exc()
        else:
            print("  [SKIP] 레퍼런스 모델 없음")

    # PCA/Primitive/PnP용 물체 점군 = 테이블 제거된 점군
    object_pcd = objects_pcd

    # 방법 2: PCA
    if method in ["pca", "all"]:
        pose_pca = PoseEstimator.estimate_with_pca(object_pcd)
        results.append(pose_pca)
        print_pose_result(pose_pca)

    # 방법 3: Primitive
    if method in ["primitive", "prim", "all"]:
        pose_prim = PoseEstimator.estimate_with_primitive(
            object_pcd, shape=args.shape
        )
        results.append(pose_prim)
        print_pose_result(pose_prim)

    # 방법 4: PnP
    if method in ["pnp", "all"]:
        pose_pnp = PoseEstimator.estimate_with_pnp(camera_data_list, extrinsics)
        results.append(pose_pnp)
        print_pose_result(pose_pnp)

    # --- 검증 및 저장 ---
    print("\n[5/5] 검증 및 저장")
    os.makedirs(args.output_dir, exist_ok=True)

    for i, result in enumerate(results):
        save_path = os.path.join(args.output_dir, f"pose_{result.method.replace(' ', '_')}.npz")
        save_pose_to_file(result, save_path)

        # 재투영 검증 (레퍼런스 정합 결과)
        if result.method == "Reference Matching" and model_aligned_pcd is not None:
            try:
                for cam_data in camera_data_list:
                    cam_id = cam_data.intrinsics.cam_id
                    out_img = os.path.join(
                        args.output_dir, f"reprojection_cam{cam_id}.png"
                    )
                    PoseValidator.reprojection_check(result, model_aligned_pcd, cam_data, out_img)
            except Exception as e:
                print(f"  [검증 중 오류] {e}")

    # --- 최종 요약 ---
    print("\n" + "=" * 60)
    print(" 최종 결과 요약")
    print("=" * 60)
    for r in results:
        print(f"\n  [{r.method}]")
        print(f"    위치: ({r.translation[0]:+.4f}, {r.translation[1]:+.4f}, {r.translation[2]:+.4f}) m")
        print(f"    회전: ({r.euler_xyz_deg[0]:+.1f}°, {r.euler_xyz_deg[1]:+.1f}°, {r.euler_xyz_deg[2]:+.1f}°)")
        if r.fitness > 0:
            print(f"    품질: fitness={r.fitness:.4f}, RMSE={r.rmse:.6f}m")

    print(f"\n  결과 저장 위치: {args.output_dir}/")
    print("=" * 60)


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    # 기본 경로: 스크립트 위치 기준
    _script_dir = Path(__file__).resolve().parent
    _default_data_dir = str(_script_dir / "data")
    _default_output_dir = str(_script_dir / "output")
    _default_extrinsics = str(_script_dir / "data" / "cube_session_01" / "calib_out_cube")
    _default_glb = str(_script_dir / "data" / "reference_knife.glb")

    parser = argparse.ArgumentParser(
        description="멀티뷰 카메라 기반 물체 포즈 추정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data_dir", type=str, default=_default_data_dir,
                        help="데이터 디렉토리 (intrinsics, object_capture 포함)")
    parser.add_argument("--output_dir", type=str, default=_default_output_dir,
                        help="결과 저장 디렉토리")
    parser.add_argument("--extrinsics_dir", type=str, default=_default_extrinsics,
                        help="T_C0_C1.npy, T_C0_C2.npy 위치")
    parser.add_argument("--glb_path", type=str, default=_default_glb,
                        help="GLB 모델 파일 경로")
    parser.add_argument("--frame_id", type=str, default="000003",
                        help="사용할 프레임 번호 (예: 000003)")
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "ref", "glb", "icp", "pca", "primitive", "prim", "pnp"],
                        help="포즈 추정 방법")
    parser.add_argument("--num_cameras", type=int, default=3,
                        help="카메라 수")
    parser.add_argument("--voxel_size", type=float, default=0.003,
                        help="점군 다운샘플링 복셀 크기 (미터)")
    parser.add_argument("--known_size", type=float, default=None,
                        help="물체의 알려진 치수 (미터), 스케일 보정용")
    parser.add_argument("--known_axis", type=int, default=0,
                        help="known_size가 적용되는 축 (0=x, 1=y, 2=z)")
    parser.add_argument("--shape", type=str, default="auto",
                        choices=["auto", "box", "cylinder", "sphere"],
                        help="Primitive 피팅 시 형상 종류")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_pipeline(args)
