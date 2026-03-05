# Step5_localize_object_3d.py
# ──────────────────────────────────────────────────────────────
# 접근법 A: GroundingDINO + SAM2 기반 멀티카메라 3D 객체 위치 추정
#
# PLY 파일 없이, RGB-D 프레임 + 캘리브레이션만으로 3D 위치 추정.
# Open-vocabulary 텍스트 프롬프트로 아무 객체든 검출 가능.
#
# 파이프라인:
#   각 카메라 RGB → GroundingDINO(검출) → SAM2(세그멘테이션)
#   → Depth × 마스크 → 3D 역투영 (camera frame)
#   → T_C0_Ci로 ref frame 변환 → 멀티뷰 가중 평균
#   → (보조) N-view DLT 삼각측량
#
# 입력:
#   - RGB-D 프레임: data/rgbd_capture/cam{0,1,2}/
#   - 캘리브레이션: T_C0_C1.npy, T_C0_C2.npy, cam{0,1,2}.npz
#   - 텍스트 프롬프트: "bottle." "tiger figure." 등
#
# 출력:
#   - localization_results.json (프레임별, 객체별 3D 좌표)
#   - annotated 이미지 (bbox + mask + depth 거리 표시)
#   - (선택) 객체별 PLY 포인트클라우드
#
# ──────────────────────────────────────────────────────────────
#
# 필요 패키지 설치:
#   pip install torch torchvision
#   pip install transformers accelerate
#   pip install sam2
#
# SAM2 체크포인트 다운로드:
#   mkdir -p checkpoints
#   wget -P checkpoints/ \
#     https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
#
# ──────────────────────────────────────────────────────────────
"""
사용법 (pose_step5_localize/ 폴더에서 실행):

# 기본 (단일 프레임 000000, CPU)
python Step5_localize_object_3d.py \
  --capture_dir ../data/rgbd_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "bottle." \
  --frame 0

# MPS (Apple Silicon) + PLY 저장
python Step5_localize_object_3d.py \
  --capture_dir ../data/rgbd_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "tiger figure." \
  --device mps \
  --save_pcd

# 여러 객체 + 전체 프레임
python Step5_localize_object_3d.py \
  --capture_dir ../data/rgbd_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "red cup. green box." \
  --device mps \
  --all_frames
"""

import os
import glob
import json
import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


# ================================================================
#  1. I/O 헬퍼 (캘리브레이션 + 카메라/프레임 탐색)
# ================================================================
def load_intrinsics(intrinsics_dir: str, cam_idx: int):
    """cam{i}.npz에서 K, D, depth_scale 로드."""
    p = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"intrinsics 없음: {p}")
    data = np.load(p, allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64) if "color_D" in data else None
    ds = float(data["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in data else 0.001
    return K, D, ds


def load_extrinsics(
    calib_dir: str, ref_idx: int, cam_indices: List[int]
) -> Dict[int, np.ndarray]:
    """T_C{ref}_C{i}.npy 로드. ref 카메라는 identity."""
    T_ref = {}
    for ci in cam_indices:
        if ci == ref_idx:
            T_ref[ci] = np.eye(4, dtype=np.float64)
            continue
        p = os.path.join(calib_dir, f"T_C{ref_idx}_C{ci}.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"extrinsics 없음: {p}  (Step3 먼저 수행)")
        T_ref[ci] = np.load(p).astype(np.float64)
        print(f"[INFO] Loaded T_C{ref_idx}_C{ci}")
    return T_ref


def discover_cameras(capture_dir: str) -> List[int]:
    """cam* 디렉토리에서 카메라 인덱스 탐색."""
    indices = []
    for d in sorted(glob.glob(os.path.join(capture_dir, "cam*"))):
        name = os.path.basename(d)
        try:
            idx = int(name.replace("cam", ""))
        except ValueError:
            continue
        if glob.glob(os.path.join(d, "rgb_*.jpg")):
            indices.append(idx)
    if not indices:
        raise RuntimeError(f"cam*/rgb_*.jpg 없음: {capture_dir}")
    return indices


def discover_frames(capture_dir: str, cam_indices: List[int]) -> List[int]:
    """프레임 인덱스 탐색 (모든 카메라 통합)."""
    frames = set()
    for ci in cam_indices:
        for p in glob.glob(os.path.join(capture_dir, f"cam{ci}", "rgb_*.jpg")):
            num = os.path.basename(p).replace("rgb_", "").replace(".jpg", "")
            try:
                frames.add(int(num))
            except ValueError:
                continue
    if not frames:
        raise RuntimeError(f"프레임 없음: {capture_dir}")
    return sorted(frames)


def _detect_zero_padding(capture_dir: str, cam_idx: int) -> int:
    """파일명 zero-padding 길이 감지."""
    files = glob.glob(os.path.join(capture_dir, f"cam{cam_idx}", "rgb_*.jpg"))
    if not files:
        return 6
    num_str = os.path.basename(files[0]).replace("rgb_", "").replace(".jpg", "")
    return len(num_str)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """SE3 변환: p' = R @ p + t (row-vector 형태)."""
    return points @ T[:3, :3].T + T[:3, 3].reshape(1, 3)


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """ASCII PLY 저장 (colors: 0~1 float 또는 0~255 uint8)."""
    n = len(points)
    if colors.dtype == np.float64 or colors.dtype == np.float32:
        rgb_u8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = colors.astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write(
                f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                f"{rgb_u8[i,0]} {rgb_u8[i,1]} {rgb_u8[i,2]}\n"
            )
    print(f"[SAVE] {path}  ({n:,} points)")


# ================================================================
#  2. Detection 데이터 구조
# ================================================================
@dataclass
class Detection:
    cam_idx: int
    frame_idx: int
    label: str
    score: float
    box_xyxy: np.ndarray           # [x1, y1, x2, y2] pixel
    mask: np.ndarray               # H×W boolean
    center_2d: np.ndarray          # [cx, cy] 마스크 중심 pixel
    centroid_3d_cam: Optional[np.ndarray] = None  # [x,y,z] camera frame (m)
    centroid_3d_ref: Optional[np.ndarray] = None  # [x,y,z] ref frame (m)
    num_valid_depth: int = 0


# ================================================================
#  3. 모델 로딩 (GroundingDINO + SAM2)
# ================================================================
def load_grounding_dino(
    model_id: str = "IDEA-Research/grounding-dino-tiny",
    device: str = "cpu",
):
    """HuggingFace GroundingDINO 로드 (첫 실행 시 자동 다운로드)."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    print(f"[INFO] GroundingDINO loaded: {model_id} → {device}")
    return processor, model


def load_sam2(
    checkpoint_path: str,
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cpu",
):
    """SAM2.1 predictor 로드."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"SAM2 checkpoint 없음: {checkpoint_path}\n"
            "다운로드:\n"
            "  mkdir -p checkpoints\n"
            "  wget -P checkpoints/ "
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
            "sam2.1_hiera_large.pt"
        )
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"[INFO] SAM2 loaded: {checkpoint_path} → {device}")
    return predictor


# ================================================================
#  4. 객체 검출 (GroundingDINO)
# ================================================================
def detect_objects(
    processor,
    model,
    image_rgb: np.ndarray,
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    device: str = "cpu",
) -> List[dict]:
    """
    GroundingDINO 텍스트 기반 객체 검출.
    text_prompt 예: "bottle." "red cup. green box."
    반환: [{"box": [x1,y1,x2,y2], "score": float, "label": str}, ...]
    """
    import torch
    from PIL import Image

    pil_image = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    inputs = processor(
        images=pil_image, text=text_prompt, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )

    detections = []
    if results:
        r = results[0]
        boxes = r["boxes"].cpu().numpy()
        scores = r["scores"].cpu().numpy()
        # transformers 버전별 키 이름 호환
        if "text_labels" in r:
            labels = r["text_labels"]
        elif "labels" in r and not hasattr(r["labels"], "cpu"):
            labels = r["labels"]
        else:
            labels = [text_prompt.strip().rstrip(".") for _ in boxes]

        for box, score, label in zip(boxes, scores, labels):
            detections.append({
                "box": box.astype(np.float32),
                "score": float(score),
                "label": str(label).strip(),
            })
    return detections


# ================================================================
#  5. 세그멘테이션 (SAM2)
# ================================================================
def segment_objects(
    sam_predictor,
    image_rgb: np.ndarray,
    boxes: List[np.ndarray],
) -> List[np.ndarray]:
    """SAM2 bbox 프롬프트 기반 세그멘테이션. 반환: [mask_HxW_bool, ...]"""
    import torch

    sam_predictor.set_image(image_rgb)
    masks = []
    for box in boxes:
        with torch.inference_mode():
            mask_preds, scores, _ = sam_predictor.predict(
                box=box.reshape(1, 4), multimask_output=False,
            )
        best = mask_preds[0]
        if hasattr(best, "cpu"):
            best = best.cpu().numpy()
        masks.append(best.astype(bool))
    return masks


# ================================================================
#  6. Depth 기반 3D 중심점 / 점군
# ================================================================
def masked_depth_to_centroid(
    depth_u16: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    D: Optional[np.ndarray],
    depth_scale: float,
    z_min: float = 0.1,
    z_max: float = 1.5,
) -> Tuple[Optional[np.ndarray], int]:
    """
    마스크 영역 depth → 3D 중심점 (camera frame, median 기반).
    반환: (centroid_3d | None, num_valid_pixels)
    """
    h, w = depth_u16.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ) > 0

    v_grid, u_grid = np.mgrid[0:h, 0:w]
    d = depth_u16.astype(np.float64) * depth_scale
    valid = mask & (d > z_min) & (d < z_max)
    num_valid = int(valid.sum())

    if num_valid < 10:
        return None, num_valid

    z = d[valid]
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)

    # 왜곡 보정
    if D is not None and np.any(D != 0):
        pts_2d = np.column_stack([u, v]).reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(pts_2d, K, D, P=None).reshape(-1, 2)
        x = undist[:, 0] * z
        y = undist[:, 1] * z
    else:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

    return np.array([np.median(x), np.median(y), np.median(z)]), num_valid


def masked_depth_to_points(
    depth_u16: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    D: Optional[np.ndarray],
    depth_scale: float,
    z_min: float = 0.1,
    z_max: float = 1.5,
    stride: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """마스크 영역 전체 depth → 3D 점군 + 색상 (PLY 저장용)."""
    h, w = depth_u16.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ) > 0

    v_grid, u_grid = np.mgrid[0:h:stride, 0:w:stride]
    d = depth_u16[v_grid, u_grid].astype(np.float64) * depth_scale
    m = mask[v_grid, u_grid]
    valid = m & (d > z_min) & (d < z_max)

    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 3))

    z = d[valid]
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)

    if D is not None and np.any(D != 0):
        pts_2d = np.column_stack([u, v]).reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(pts_2d, K, D, P=None).reshape(-1, 2)
        x = undist[:, 0] * z
        y = undist[:, 1] * z
    else:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

    points = np.column_stack([x, y, z])
    colors = rgb[v_grid[valid].astype(int), u_grid[valid].astype(int)]
    return points, colors


# ================================================================
#  7. Multi-view DLT 삼각측량
# ================================================================
def build_projection_matrices(
    K_map: Dict[int, np.ndarray],
    T_ref: Dict[int, np.ndarray],
    cam_indices: List[int],
) -> Dict[int, np.ndarray]:
    """
    각 카메라의 3x4 projection matrix.
    P_i = K_i @ inv(T_C0_Ci)[:3, :]
    World = ref cam (cam0) 좌표계.
    """
    P_map = {}
    for ci in cam_indices:
        T_Ci_ref = np.linalg.inv(T_ref[ci])  # world → cam_i
        P_map[ci] = K_map[ci] @ T_Ci_ref[:3, :]
    return P_map


def triangulate_nview_dlt(
    projections: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[np.ndarray]:
    """
    N-view DLT 삼각측량 (Hartley & Zisserman).
    projections: [(P_3x4, point_2d_xy), ...]
    반환: [x, y, z] 3D (ref frame) 또는 None.
    """
    n = len(projections)
    if n < 2:
        return None

    A = np.zeros((2 * n, 4), dtype=np.float64)
    for i, (P, pt2d) in enumerate(projections):
        u, v = pt2d[0], pt2d[1]
        A[2 * i] = u * P[2, :] - P[0, :]
        A[2 * i + 1] = v * P[2, :] - P[1, :]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1, :]
    if abs(X[3]) < 1e-10:
        return None
    return X[:3] / X[3]


# ================================================================
#  8. Cross-camera 매칭
# ================================================================
def match_detections_across_cameras(
    per_cam_detections: Dict[int, List[Detection]],
    T_ref: Dict[int, np.ndarray],
) -> List[List[Detection]]:
    """
    카메라 간 동일 객체 매칭.
    라벨 기반 그룹핑 → 단일 인스턴스: 직접, 다중: 3D 근접성.
    """
    label_groups: Dict[str, Dict[int, List[Detection]]] = {}
    for ci, dets in per_cam_detections.items():
        for det in dets:
            label_groups.setdefault(det.label, {}).setdefault(ci, []).append(det)

    matched = []
    for label, cam_dets in label_groups.items():
        max_per_cam = max(len(d) for d in cam_dets.values())
        if max_per_cam == 1:
            group = [dets[0] for dets in cam_dets.values()]
            matched.append(group)
        else:
            matched.extend(_match_multiple_instances(cam_dets))
    return matched


def _match_multiple_instances(
    cam_dets: Dict[int, List[Detection]],
) -> List[List[Detection]]:
    """다중 인스턴스: 3D centroid 근접성 기반 greedy 매칭."""
    anchor_ci = max(
        cam_dets.keys(),
        key=lambda ci: sum(1 for d in cam_dets[ci] if d.centroid_3d_ref is not None),
    )
    groups = [[d] for d in cam_dets[anchor_ci]]

    for ci, dets in cam_dets.items():
        if ci == anchor_ci:
            continue
        used = set()
        for det in dets:
            best_gi = _find_nearest_group(groups, det, used)
            if best_gi is not None:
                groups[best_gi].append(det)
                used.add(best_gi)
    return groups


def _find_nearest_group(groups, det, used):
    if det.centroid_3d_ref is None:
        # fallback: score 유사도
        best_gi, best_diff = None, float("inf")
        for gi, g in enumerate(groups):
            if gi in used:
                continue
            diff = abs(det.score - g[0].score)
            if diff < best_diff:
                best_diff, best_gi = diff, gi
        return best_gi

    best_gi, best_dist = None, float("inf")
    for gi, g in enumerate(groups):
        if gi in used:
            continue
        for gd in g:
            if gd.centroid_3d_ref is not None:
                dist = np.linalg.norm(det.centroid_3d_ref - gd.centroid_3d_ref)
                if dist < best_dist:
                    best_dist, best_gi = dist, gi
                break
    return best_gi


# ================================================================
#  9. 메인 localization 파이프라인 (단일 프레임)
# ================================================================
def localize_objects(
    capture_dir: str,
    frame_idx: int,
    cam_indices: List[int],
    K_map: Dict[int, np.ndarray],
    D_map: Dict[int, Optional[np.ndarray]],
    ds_map: Dict[int, float],
    T_ref: Dict[int, np.ndarray],
    gdino_processor,
    gdino_model,
    sam_predictor,
    text_prompt: str,
    ref_cam: int = 0,
    z_min: float = 0.1,
    z_max: float = 1.5,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    pad: int = 6,
    device: str = "cpu",
    T_base_C0: Optional[np.ndarray] = None,
    save_pcd: bool = False,
    out_dir: str = "",
) -> dict:
    """
    단일 프레임 전체 파이프라인:
    1. GroundingDINO 검출 → SAM2 세그멘테이션
    2. Depth × 마스크 → 3D centroid (per camera)
    3. 카메라 간 매칭 + 3D 위치 융합 (depth + triangulation)
    """
    fid_str = f"{frame_idx:0{pad}d}"
    per_cam_dets: Dict[int, List[Detection]] = {}
    per_cam_rgb: Dict[int, np.ndarray] = {}

    # ─── 검출 + 세그멘테이션 + depth centroid ───
    for ci in cam_indices:
        rgb_path = os.path.join(capture_dir, f"cam{ci}", f"rgb_{fid_str}.jpg")
        depth_path = os.path.join(capture_dir, f"cam{ci}", f"depth_{fid_str}.png")

        if not os.path.exists(rgb_path):
            print(f"  cam{ci}: RGB 없음, skip")
            per_cam_dets[ci] = []
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            per_cam_dets[ci] = []
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        per_cam_rgb[ci] = rgb_bgr

        # GroundingDINO
        raw_dets = detect_objects(
            gdino_processor, gdino_model, rgb, text_prompt,
            box_threshold, text_threshold, device,
        )
        if not raw_dets:
            print(f"  cam{ci}: 검출 없음")
            per_cam_dets[ci] = []
            continue

        # SAM2
        boxes = [d["box"] for d in raw_dets]
        masks = segment_objects(sam_predictor, rgb, boxes)

        # Depth 로드
        depth_u16 = None
        if os.path.exists(depth_path):
            depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # Detection 객체 생성
        dets = []
        for det_raw, mask in zip(raw_dets, masks):
            if mask.sum() > 0:
                ys, xs = np.where(mask)
                center_2d = np.array([np.median(xs), np.median(ys)])
            else:
                box = det_raw["box"]
                center_2d = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])

            d = Detection(
                cam_idx=ci, frame_idx=frame_idx,
                label=det_raw["label"], score=det_raw["score"],
                box_xyxy=det_raw["box"], mask=mask, center_2d=center_2d,
            )

            if depth_u16 is not None and mask.sum() > 0:
                centroid_cam, nv = masked_depth_to_centroid(
                    depth_u16, mask, K_map[ci], D_map.get(ci),
                    ds_map[ci], z_min, z_max,
                )
                d.num_valid_depth = nv
                if centroid_cam is not None:
                    d.centroid_3d_cam = centroid_cam
                    d.centroid_3d_ref = transform_points(
                        centroid_cam.reshape(1, 3), T_ref[ci]
                    ).flatten()

            dets.append(d)

        per_cam_dets[ci] = dets
        labels_str = ", ".join(f"{d.label}({d.score:.2f})" for d in dets)
        print(f"  cam{ci}: {len(dets)}개 [{labels_str}]")

    # ─── 카메라 간 매칭 ───
    matched = match_detections_across_cameras(per_cam_dets, T_ref)

    # ─── 3D 위치 융합 ───
    P_map = build_projection_matrices(K_map, T_ref, cam_indices)
    results = []

    for obj_idx, group in enumerate(matched):
        # Method 1: Depth 가중 평균
        valid_centroids = []
        weights = []
        for det in group:
            if det.centroid_3d_ref is not None:
                valid_centroids.append(det.centroid_3d_ref)
                weights.append(det.num_valid_depth)

        depth_centroid = None
        if valid_centroids:
            w = np.array(weights, dtype=np.float64)
            w /= w.sum() + 1e-12
            depth_centroid = sum(c * wi for c, wi in zip(valid_centroids, w))

        # Method 2: N-view DLT 삼각측량
        projections = [
            (P_map[det.cam_idx], det.center_2d)
            for det in group if det.cam_idx in P_map
        ]
        triang_centroid = triangulate_nview_dlt(projections)

        # 최종 위치 (depth 우선)
        position_ref = depth_centroid if depth_centroid is not None else triang_centroid

        # Robot base 변환
        position_base = None
        if position_ref is not None and T_base_C0 is not None:
            position_base = transform_points(
                position_ref.reshape(1, 3), T_base_C0
            ).flatten()

        # depth ↔ triangulation 불일치
        discrepancy_mm = None
        if depth_centroid is not None and triang_centroid is not None:
            discrepancy_mm = float(
                np.linalg.norm(depth_centroid - triang_centroid) * 1000
            )

        result_obj = {
            "object_idx": obj_idx,
            "label": group[0].label,
            "cameras": [d.cam_idx for d in group],
            "scores": [round(d.score, 4) for d in group],
            "depth_centroid_ref_m": (
                depth_centroid.tolist() if depth_centroid is not None else None
            ),
            "triang_centroid_ref_m": (
                triang_centroid.tolist() if triang_centroid is not None else None
            ),
            "position_ref_m": (
                position_ref.tolist() if position_ref is not None else None
            ),
            "position_base_m": (
                position_base.tolist() if position_base is not None else None
            ),
            "discrepancy_mm": discrepancy_mm,
            "num_cameras_with_depth": len(valid_centroids),
            "per_camera": [
                {
                    "cam": d.cam_idx,
                    "score": round(d.score, 4),
                    "box_xyxy": d.box_xyxy.tolist(),
                    "center_2d": d.center_2d.tolist(),
                    "centroid_3d_cam_m": (
                        d.centroid_3d_cam.tolist()
                        if d.centroid_3d_cam is not None else None
                    ),
                    "centroid_3d_ref_m": (
                        d.centroid_3d_ref.tolist()
                        if d.centroid_3d_ref is not None else None
                    ),
                    "num_valid_depth_px": d.num_valid_depth,
                }
                for d in group
            ],
        }
        results.append(result_obj)

        # PLY 저장
        if save_pcd and out_dir:
            _save_object_pointcloud(
                capture_dir, frame_idx, group, K_map, D_map, ds_map,
                T_ref, z_min, z_max, pad, out_dir, obj_idx,
            )

    return {"frame": frame_idx, "text_prompt": text_prompt, "objects": results}


def _save_object_pointcloud(
    capture_dir, frame_idx, group, K_map, D_map, ds_map,
    T_ref, z_min, z_max, pad, out_dir, obj_idx,
):
    """객체별 masked 3D PLY 저장."""
    fid_str = f"{frame_idx:0{pad}d}"
    all_pts, all_cols = [], []
    for det in group:
        ci = det.cam_idx
        rgb_path = os.path.join(capture_dir, f"cam{ci}", f"rgb_{fid_str}.jpg")
        depth_path = os.path.join(capture_dir, f"cam{ci}", f"depth_{fid_str}.png")
        if not os.path.exists(depth_path):
            continue
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_u16 is None:
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        pts_cam, cols = masked_depth_to_points(
            depth_u16, rgb, det.mask, K_map[ci], D_map.get(ci),
            ds_map[ci], z_min, z_max, stride=1,
        )
        if pts_cam.shape[0] == 0:
            continue
        all_pts.append(transform_points(pts_cam, T_ref[ci]))
        all_cols.append(cols)

    if all_pts:
        label = group[0].label.replace(" ", "_")
        ply_path = os.path.join(
            out_dir, f"object_{obj_idx}_{label}_frame{fid_str}.ply"
        )
        save_ply(ply_path, np.concatenate(all_pts), np.concatenate(all_cols))


# ================================================================
#  10. 시각화
# ================================================================
_VIS_COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
]


def draw_annotated_image(
    rgb_bgr: np.ndarray, detections: List[Detection]
) -> np.ndarray:
    """검출 결과 시각화 (bbox + mask + depth 거리)."""
    vis = rgb_bgr.copy()
    for i, det in enumerate(detections):
        color = _VIS_COLORS[i % len(_VIS_COLORS)]
        x1, y1, x2, y2 = det.box_xyxy.astype(int)

        # 반투명 마스크
        if det.mask.sum() > 0:
            overlay = vis.copy()
            m = det.mask
            if m.shape[:2] != vis.shape[:2]:
                m = cv2.resize(
                    m.astype(np.uint8), (vis.shape[1], vis.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            overlay[m] = np.array(color, dtype=np.uint8)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label_text = f"{det.label} {det.score:.2f}"
        if det.centroid_3d_cam is not None:
            dist_mm = np.linalg.norm(det.centroid_3d_cam) * 1000
            label_text += f" d={dist_mm:.0f}mm"

        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            vis, label_text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )
        cx2d, cy2d = det.center_2d.astype(int)
        cv2.drawMarker(vis, (cx2d, cy2d), color, cv2.MARKER_CROSS, 15, 2)

    return vis


def print_results_table(results: dict) -> None:
    """결과 콘솔 출력."""
    print(f"\n{'='*70}")
    print(f" 3D Object Localization Results (frame {results['frame']})")
    print(f" prompt: \"{results['text_prompt']}\"")
    print(f"{'='*70}")

    if not results["objects"]:
        print("  (검출된 객체 없음)")
        print(f"{'='*70}")
        return

    for obj in results["objects"]:
        print(f"\n  [{obj['object_idx']}] \"{obj['label']}\"")
        print(f"      Cameras: {obj['cameras']}  Scores: {obj['scores']}")

        if obj["depth_centroid_ref_m"]:
            p = obj["depth_centroid_ref_m"]
            print(f"      Depth 3D (ref):      "
                  f"({p[0]*1000:+.1f}, {p[1]*1000:+.1f}, {p[2]*1000:+.1f}) mm")

        if obj["triang_centroid_ref_m"]:
            p = obj["triang_centroid_ref_m"]
            print(f"      Triangulation (ref):  "
                  f"({p[0]*1000:+.1f}, {p[1]*1000:+.1f}, {p[2]*1000:+.1f}) mm")

        if obj["discrepancy_mm"] is not None:
            print(f"      Depth vs Triang:      {obj['discrepancy_mm']:.1f} mm")

        if obj["position_ref_m"]:
            p = obj["position_ref_m"]
            print(f"      >> 최종 위치 (ref):   "
                  f"({p[0]*1000:+.1f}, {p[1]*1000:+.1f}, {p[2]*1000:+.1f}) mm")

        if obj["position_base_m"]:
            p = obj["position_base_m"]
            print(f"      >> 최종 위치 (base):  "
                  f"({p[0]*1000:+.1f}, {p[1]*1000:+.1f}, {p[2]*1000:+.1f}) mm")

    print(f"\n{'='*70}")


# ================================================================
#  Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Step5: GroundingDINO+SAM2 멀티카메라 3D 객체 위치 추정"
    )

    # 필수
    parser.add_argument("--capture_dir", required=True,
                        help="RGB-D 프레임 폴더 (cam0/, cam1/, cam2/)")
    parser.add_argument("--calib_dir", required=True,
                        help="T_C0_Ci.npy 캘리브레이션 폴더")
    parser.add_argument("--intrinsics_dir", default="../intrinsics",
                        help="내부파라미터 폴더")
    parser.add_argument("--text_prompt", required=True,
                        help='검출 텍스트 (예: "bottle." "tiger figure.")')

    # 모델
    parser.add_argument("--gdino_model",
                        default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2_checkpoint",
                        default="../checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_config",
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cpu",
                        help="추론 디바이스 (cpu/cuda/mps)")

    # 프레임
    parser.add_argument("--frame", type=int, default=None,
                        help="특정 프레임 (기본: 첫 프레임)")
    parser.add_argument("--all_frames", action="store_true")
    parser.add_argument("--frame_skip", type=int, default=1)

    # 검출
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)

    # Depth
    parser.add_argument("--ref_cam", type=int, default=0)
    parser.add_argument("--z_min", type=float, default=0.1)
    parser.add_argument("--z_max", type=float, default=1.5)

    # 로봇
    parser.add_argument("--T_base_C0", type=str, default=None,
                        help="Robot base→cam0 변환 npy")

    # 출력
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--save_pcd", action="store_true",
                        help="객체별 PLY 저장")
    parser.add_argument("--no_vis", action="store_true",
                        help="시각화 이미지 끄기")

    args = parser.parse_args()

    # ─── 의존성 체크 ───
    try:
        import torch
    except ImportError:
        print("[ERROR] torch 미설치. 설치 명령:")
        print("  pip install torch torchvision")
        print("  pip install transformers accelerate")
        print("  pip install sam2")
        return

    # ─── 카메라/프레임 탐색 ───
    cam_indices = discover_cameras(args.capture_dir)
    frame_ids = discover_frames(args.capture_dir, cam_indices)
    pad = _detect_zero_padding(args.capture_dir, cam_indices[0])

    print(f"[INFO] 카메라: {cam_indices} ({len(cam_indices)}대)")
    print(f"[INFO] 프레임: {len(frame_ids)}장 ({frame_ids[0]} ~ {frame_ids[-1]})")
    print(f"[INFO] 텍스트: \"{args.text_prompt}\"")

    # ─── 캘리브레이션 로드 ───
    K_map, D_map, ds_map = {}, {}, {}
    for ci in cam_indices:
        K, D, ds = load_intrinsics(args.intrinsics_dir, ci)
        K_map[ci], D_map[ci], ds_map[ci] = K, D, ds

    T_ref = load_extrinsics(args.calib_dir, args.ref_cam, cam_indices)

    T_base_C0 = None
    if args.T_base_C0 and os.path.exists(args.T_base_C0):
        T_base_C0 = np.load(args.T_base_C0).astype(np.float64)
        print(f"[INFO] T_base_C0: {args.T_base_C0}")

    # ─── 디바이스 ───
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 불가 → CPU")
        device = "cpu"
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("[WARN] MPS 불가 → CPU")
        device = "cpu"

    # ─── 모델 로딩 ───
    print(f"\n[INFO] 모델 로딩 (device={device})...")
    t0 = time.time()
    gdino_proc, gdino_model = load_grounding_dino(args.gdino_model, device)
    sam_pred = load_sam2(args.sam2_checkpoint, args.sam2_config, device)
    print(f"[INFO] 모델 로딩 완료 ({time.time()-t0:.1f}s)\n")

    # ─── 프레임 선택 ───
    if args.all_frames:
        target_frames = frame_ids[:: args.frame_skip]
    else:
        fid = args.frame if args.frame is not None else frame_ids[0]
        if fid not in frame_ids:
            raise RuntimeError(f"frame {fid} 없음. 가능: {frame_ids}")
        target_frames = [fid]

    # ─── 출력 폴더 ───
    out_dir = args.out_dir or "./output"
    os.makedirs(out_dir, exist_ok=True)

    # ─── 처리 ───
    all_results = []
    for fi, fid in enumerate(target_frames):
        print(f"\n[FRAME {fid}] ({fi+1}/{len(target_frames)})")
        t_frame = time.time()

        result = localize_objects(
            args.capture_dir, fid, cam_indices,
            K_map, D_map, ds_map, T_ref,
            gdino_proc, gdino_model, sam_pred,
            args.text_prompt, args.ref_cam,
            args.z_min, args.z_max,
            args.box_threshold, args.text_threshold,
            pad, device, T_base_C0,
            save_pcd=args.save_pcd, out_dir=out_dir,
        )
        all_results.append(result)
        print_results_table(result)

        # 시각화 저장
        if not args.no_vis:
            fid_str = f"{fid:0{pad}d}"
            for ci in cam_indices:
                rgb_path = os.path.join(
                    args.capture_dir, f"cam{ci}", f"rgb_{fid_str}.jpg"
                )
                if not os.path.exists(rgb_path):
                    continue
                rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                if rgb_bgr is None:
                    continue

                cam_dets = []
                for obj in result["objects"]:
                    for pc in obj["per_camera"]:
                        if pc["cam"] == ci:
                            d = Detection(
                                cam_idx=ci, frame_idx=fid,
                                label=obj["label"], score=pc["score"],
                                box_xyxy=np.array(pc["box_xyxy"]),
                                mask=np.zeros(rgb_bgr.shape[:2], dtype=bool),
                                center_2d=np.array(pc["center_2d"]),
                            )
                            if pc["centroid_3d_cam_m"]:
                                d.centroid_3d_cam = np.array(pc["centroid_3d_cam_m"])
                            cam_dets.append(d)

                if cam_dets:
                    vis = draw_annotated_image(rgb_bgr, cam_dets)
                    vis_path = os.path.join(
                        out_dir, f"annotated_cam{ci}_frame{fid_str}.jpg"
                    )
                    cv2.imwrite(vis_path, vis)

        print(f"  처리 시간: {time.time()-t_frame:.1f}s")

    # ─── JSON 저장 ───
    out_json = os.path.join(out_dir, "localization_results.json")
    with open(out_json, "w") as f:
        json.dump({"results": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] {out_json}")

    total_obj = sum(len(r["objects"]) for r in all_results)
    print(f"\n[DONE] {len(target_frames)}프레임, {total_obj}개 객체")
    print(f"[DONE] 결과: {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
