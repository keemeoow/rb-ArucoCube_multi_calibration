# Obj_Step2-(1)_pose_gorunding_sam.py
# ──────────────────────────────────────────────────────────────
# GroundingDINO + SAM2 물체 검출 → 멀티카메라 RGB-D 6DoF 포즈 추정
# → Isaac Sim 좌표계 변환까지 단일 스크립트
#
# ★ 회전값 추정 방식: ICP (CAD 모델 기반)
#
#   SAM2 마스크 점군 ← 레퍼런스 CAD 모델(--ref_ply)과 ICP 정합
#   → 180° 모호성 없이 정확한 6DoF 회전 획득
#   원리: 측정 점군과 모델이 가장 잘 겹치는 회전 = 물체의 실제 방향
#
# 파이프라인:
#   1. GroundingDINO: 텍스트 프롬프트로 물체 검출 (bbox)
#   2. SAM2:          bbox → 정밀 세그멘테이션 마스크
#   3. 마스크 × Depth → 3D 역투영 (카메라별, cv2.undistortPoints)
#   4. T_C0_Ci:       각 카메라 점군 → cam0 좌표계 통합 (멀티뷰 융합)
#   5. SOR:           통계적 아웃라이어 제거
#   6. ICP: 4가지 PCA 부호 조합 초기화 → SVD ICP → 최소 RMSE 선택
#   7. cam0 → Isaac Lab(USD) 좌표 변환 후 저장
#
# 입력:
#   - RGB-D 프레임: capture_dir/cam{0,1,2}/rgb_NNNNNN.jpg, depth_NNNNNN.png
#   - 캘리브레이션: calib_dir/T_C0_C1.npy, T_C0_C2.npy
#   - 내부파라미터: intrinsics_dir/cam{0,1,2}.npz
#   - 텍스트 프롬프트: "bottle." "tiger figure." 등
#   - (선택) 레퍼런스 PLY: tiger.ply 등 CAD 모델 (ICP용)
#
# 출력:
#   - pose_result.json   (cam0 + Isaac Lab 포즈, rotation_method 포함)
#   - object_masked.ply  (물체 점군)
#   - annotated 이미지   (bbox + mask 오버레이)
#   - 3D 시각화 PNG
#
# 좌표계 규약:
#   cam0 (OpenCV) : X→오른쪽, Y→아래, Z→앞 (깊이 방향)
#   Isaac Lab (USD): X→오른쪽, Y→앞,   Z→위
#   변환: X_isaac=X_cam0, Y_isaac=Z_cam0, Z_isaac=-Y_cam0
#
# 필요 패키지:
#   pip install torch torchvision transformers accelerate sam2
#   pip install numpy opencv-python matplotlib
#
# SAM2 체크포인트:
#   mkdir -p checkpoints
#   wget -P checkpoints/ \
#     https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
#
# ──────────────────────────────────────────────────────────────
"""
사용법 (pose_estimate_grounding_sam/ 폴더에서 실행):

# 기본 (frame 0, MPS)
python estimate_object_pose.py \
  --capture_dir ../data/object_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "tiger figure." \
  --ref_ply ../data/3d_ply/tiger.ply \
  --device mps

# CUDA GPU
python estimate_object_pose.py \
  --capture_dir ../data/object_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "red cup." \
  --ref_ply ../data/3d_ply/cup.ply \
  --device cuda

# 특정 프레임 + 출력 디렉토리 지정
python estimate_object_pose.py \
  --capture_dir ../data/object_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "tiger figure." \
  --ref_ply ../data/3d_ply/tiger.ply \
  --device mps \
  --frame 5 \
  --out_dir ./output
"""

import os
import glob
import json
import argparse
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2


# ================================================================
#  1. 캘리브레이션 I/O
# ================================================================

def load_intrinsics(intrinsics_dir: str, cam_idx: int):
    """
    cam{i}.npz → K(3x3 내부파라미터), D(왜곡계수), depth_scale(m/unit).

    K 행렬 구조:
      [[fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]]
    depth_scale: depth 픽셀값 × depth_scale = 미터 단위 거리
    """
    p = os.path.join(intrinsics_dir, f"cam{cam_idx}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"intrinsics 없음: {p}")
    data = np.load(p, allow_pickle=True)
    K = data["color_K"].astype(np.float64)
    D = data["color_D"].astype(np.float64) if "color_D" in data else np.zeros(5)
    ds = float(data["depth_scale_m_per_unit"]) if "depth_scale_m_per_unit" in data else 0.001
    return K, D, ds


def load_extrinsics(calib_dir: str, cam_indices: List[int]) -> Dict[int, np.ndarray]:
    """
    T_C0_C{i}.npy → {cam_idx: 4x4 SE3 변환행렬}.

    T_C0_Ci 의미: cam_i 좌표의 점 p_i → cam0 좌표로 변환
      p_cam0 = R @ p_cami + t
    cam0은 항등 행렬(identity) 사용.
    Step3(캘리브레이션) 결과물을 그대로 로드.
    """
    T = {}
    for ci in cam_indices:
        if ci == 0:
            T[ci] = np.eye(4, dtype=np.float64)
        else:
            p = os.path.join(calib_dir, f"T_C0_C{ci}.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"extrinsics 없음: {p}  (Step3 먼저 수행)")
            T[ci] = np.load(p).astype(np.float64)
    return T


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
    """프레임 인덱스 탐색."""
    frames = set()
    for ci in cam_indices:
        for p in glob.glob(os.path.join(capture_dir, f"cam{ci}", "rgb_*.jpg")):
            num = os.path.basename(p).replace("rgb_", "").replace(".jpg", "")
            try:
                frames.add(int(num))
            except ValueError:
                continue
    return sorted(frames)


def detect_zero_padding(capture_dir: str, cam_idx: int) -> int:
    """파일명 zero-padding 길이 감지."""
    files = glob.glob(os.path.join(capture_dir, f"cam{cam_idx}", "rgb_*.jpg"))
    if not files:
        return 6
    num_str = os.path.basename(files[0]).replace("rgb_", "").replace(".jpg", "")
    return len(num_str)


# ================================================================
#  2. 모델 로딩 (GroundingDINO + SAM2)
# ================================================================

def load_grounding_dino(model_id: str = "IDEA-Research/grounding-dino-tiny", device: str = "cpu"):
    """HuggingFace GroundingDINO 로드."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    print(f"  GroundingDINO: {model_id} → {device}")
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
    print(f"  SAM2: {checkpoint_path} → {device}")
    return predictor


# ================================================================
#  3. 물체 검출 + 세그멘테이션 (개선판)
#
#  문제: GroundingDINO가 긴 물체(칼 등)를 부분만 검출하는 경우 많음.
#
#  해결:
#    1) 낮은 threshold + 다중 텍스트로 후보 bbox 최대한 확보
#    2) 동일 물체의 겹치는 bbox들을 하나로 병합 (union)
#    3) 병합 bbox에 padding을 줘서 SAM2에 여유 있게 전달
#    4) SAM2 multimask → 가장 큰 마스크 선택
#    5) 1차 마스크의 중심점으로 2차 SAM2 point prompt 정제
# ================================================================

def detect_objects(
    processor, model,
    image_rgb: np.ndarray,
    text_prompt: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    device: str = "cpu",
) -> List[dict]:
    """GroundingDINO 텍스트 기반 물체 검출."""
    import torch
    from PIL import Image

    pil_image = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)
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


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """두 bbox의 IoU 계산."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def expand_bbox(
    box: np.ndarray, img_h: int, img_w: int, ratio: float = 0.10,
) -> np.ndarray:
    """bbox를 ratio만큼 확장."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx, dy = bw * ratio, bh * ratio
    return np.array([
        max(0, x1 - dx), max(0, y1 - dy),
        min(img_w, x2 + dx), min(img_h, y2 + dy),
    ], dtype=np.float32)


def _keep_largest_connected(mask: np.ndarray) -> np.ndarray:
    """마스크에서 가장 큰 연결 영역만 유지 (노이즈 제거)."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask
    # label 0 = 배경, 1~ = 영역. 가장 큰 영역 선택
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    return (labels == best_label)


def segment_with_sam2_tight(
    sam_predictor,
    image_rgb: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """
    SAM2 세그멘테이션 (빽빽한 단일 물체 마스크).
      1) bbox → multimask=True
      2) SAM2 confidence score가 가장 높은 마스크 선택 (NOT 가장 큰!)
      3) 가장 큰 연결 영역만 유지 (떨어진 노이즈 제거)
      4) bbox 영역 밖 마스크 제거
    """
    import torch

    sam_predictor.set_image(image_rgb)
    h, w = image_rgb.shape[:2]

    with torch.inference_mode():
        mask_preds, scores, _ = sam_predictor.predict(
            box=box.reshape(1, 4), multimask_output=True,
        )

    if hasattr(mask_preds, "cpu"):
        mask_preds = mask_preds.cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()

    # SAM2 confidence score 최대 마스크 선택 (물체에 가장 잘 맞는 마스크)
    best_idx = int(np.argmax(scores))
    mask = mask_preds[best_idx].astype(bool)

    # bbox 영역 밖 마스크 제거 (확장 bbox 기준)
    x1, y1, x2, y2 = box.astype(int)
    bbox_mask = np.zeros((h, w), dtype=bool)
    bbox_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    mask = mask & bbox_mask

    # 가장 큰 연결 영역만 유지
    if mask.sum() > 0:
        mask = _keep_largest_connected(mask)

    return mask


def detect_and_segment_object(
    gdino_proc, gdino_model, sam_pred,
    image_rgb: np.ndarray,
    text_prompt: str,
    box_threshold: float = 0.15,
    text_threshold: float = 0.15,
    bbox_pad_ratio: float = 0.10,
    sam_refine_iters: int = 1,
    device: str = "cpu",
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    물체 검출+세그멘테이션 (v3 — 빽빽한 단일 물체):

    전략:
      1) 메인 프롬프트로 높은 threshold(0.25) → 앵커 bbox 확보
      2) 동의어로 낮은 threshold → 앵커와 겹치는 것만 수집
      3) 앵커 기반 union bbox (관련 없는 물체 제외)
      4) SAM2: score-best 마스크 + bbox clipping + 연결 영역 필터
    """
    h, w = image_rgb.shape[:2]

    # ── Phase 1: 앵커 검출 (메인 프롬프트, 높은 threshold) ──
    anchor_dets = detect_objects(
        gdino_proc, gdino_model, image_rgb, text_prompt,
        box_threshold=0.25, text_threshold=0.20, device=device,
    )

    # 앵커가 없으면 낮은 threshold로 재시도
    if not anchor_dets:
        anchor_dets = detect_objects(
            gdino_proc, gdino_model, image_rgb, text_prompt,
            box_threshold=box_threshold, text_threshold=text_threshold,
            device=device,
        )
    if not anchor_dets:
        return None

    # score 최대 앵커 선택
    anchor = max(anchor_dets, key=lambda d: d["score"])
    anchor_box = anchor["box"]

    # ── Phase 2: 동의어로 보조 검출 → 앵커와 겹치는 것만 수집 ──
    base = text_prompt.strip().rstrip(".")
    synonym_prompts = []
    for w_tok in base.split():
        p = w_tok.strip() + "."
        if p != text_prompt and len(w_tok) > 2:
            synonym_prompts.append(p)

    related_boxes = [anchor_box]

    for sp in synonym_prompts:
        dets = detect_objects(
            gdino_proc, gdino_model, image_rgb, sp,
            box_threshold=box_threshold, text_threshold=text_threshold,
            device=device,
        )
        for d in dets:
            # 앵커와 겹치는지 확인 (IoU > 0.05 이상이면 같은 물체)
            if compute_iou(d["box"], anchor_box) > 0.05:
                related_boxes.append(d["box"])

    # ── Phase 3: 관련 bbox union ──
    all_boxes = np.array(related_boxes)
    union_box = np.array([
        all_boxes[:, 0].min(), all_boxes[:, 1].min(),
        all_boxes[:, 2].max(), all_boxes[:, 3].max(),
    ], dtype=np.float32)

    # bbox 확장 (약간만)
    expanded_box = expand_bbox(union_box, h, w, ratio=bbox_pad_ratio)

    # ── Phase 4: SAM2 빽빽한 세그멘테이션 ──
    mask = segment_with_sam2_tight(sam_pred, image_rgb, expanded_box)

    if mask.sum() < 10:
        # fallback: 앵커 bbox만으로 재시도
        expanded_anchor = expand_bbox(anchor_box, h, w, ratio=bbox_pad_ratio)
        mask = segment_with_sam2_tight(sam_pred, image_rgb, expanded_anchor)
        expanded_box = expanded_anchor

    det_info = {
        "box": union_box,
        "expanded_box": expanded_box,
        "score": anchor["score"],
        "label": anchor["label"],
        "merged_count": len(related_boxes),
        "total_candidates": len(related_boxes),
        "prompts_used": [text_prompt] + synonym_prompts,
    }
    return mask, det_info


# ================================================================
#  4. Depth × Mask → 3D 점군
# ================================================================

def masked_depth_to_points(
    depth_u16: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    depth_scale: float,
    z_min: float = 0.1,
    z_max: float = 1.5,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SAM2 마스크 영역의 depth → 카메라 프레임 3D 점군.

    ★ 핵심 원리 (핀홀 카메라 역투영):
      픽셀 (u, v)에서 depth z를 읽어 3D 좌표로 변환:
        1) 렌즈 왜곡 보정: cv2.undistortPoints(K, D) → 정규화 좌표 (x_n, y_n)
           - 렌즈 왜곡이 없는 이상적 핀홀 카메라 좌표계로 변환
        2) 3D 역투영 (미터):
             X = x_n × z
             Y = y_n × z
             Z = z          (Z = 깊이 방향, 카메라 앞쪽)

    PCA / ICP 회전 추정에 핵심: centroid 1개가 아닌
    마스크 내 모든 픽셀의 3D 점군을 반환.
    (Step5의 centroid-only 방식과 달리, 형태 정보까지 포함)

    stride: 서브샘플링 간격 (stride=2 → 점 수 1/4, 속도 4배)
    """
    h, w = depth_u16.shape[:2]
    if mask.shape[:2] != (h, w):
        # depth 해상도가 RGB와 다를 경우 마스크 리사이즈
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    # stride로 픽셀 격자 생성
    v_grid, u_grid = np.mgrid[0:h:stride, 0:w:stride]
    d = depth_u16[v_grid, u_grid].astype(np.float64) * depth_scale  # uint16 → 미터
    m = mask[v_grid, u_grid]
    # 마스크 내부 + 유효 depth 범위 픽셀만 선택
    valid = m & (d > z_min) & (d < z_max)

    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 2))

    z = d[valid]
    u = u_grid[valid].astype(np.float64)
    v = v_grid[valid].astype(np.float64)

    # 렌즈 왜곡 보정 → 정규화 좌표 (K, D 역적용)
    pts_2d = np.column_stack([u, v]).reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts_2d, K, D).reshape(-1, 2)

    # 핀홀 역투영: (x_n, y_n) × z = (X, Y, Z) in camera frame
    xyz = np.column_stack([undist[:, 0] * z, undist[:, 1] * z, z])
    return xyz, np.column_stack([u, v])


def masked_depth_to_colored_points(
    depth_u16: np.ndarray,
    rgb_bgr: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    depth_scale: float,
    z_min: float = 0.1,
    z_max: float = 1.5,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """마스크 영역 depth → 3D 점 + RGB 색상."""
    pts, uv = masked_depth_to_points(depth_u16, mask, K, D, depth_scale, z_min, z_max, stride)
    if len(pts) == 0:
        return pts, np.empty((0, 3))
    u_idx = uv[:, 0].astype(int)
    v_idx = uv[:, 1].astype(int)
    colors = rgb_bgr[v_idx, u_idx][:, ::-1].astype(np.float64) / 255.0  # BGR→RGB
    return pts, colors


# ================================================================
#  5. 멀티카메라 융합 (마스크 기반)
# ================================================================

def fuse_masked_multicam(
    capture_dir: str,
    frame_idx: int,
    cam_indices: List[int],
    masks_per_cam: Dict[int, np.ndarray],
    K_map: dict, D_map: dict, ds_map: dict, T_map: dict,
    z_min: float = 0.1, z_max: float = 1.5,
    stride: int = 1, pad: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    각 카메라의 SAM2 마스크 × depth → cam0 좌표계 통합 점군.

    멀티카메라 융합의 이점:
      - 단일 카메라: 물체 한쪽 면만 보임 (occluded surface 없음)
      - 3대 카메라: 여러 각도 → 더 완전한 점군 → PCA/ICP 정확도 향상

    변환 공식: p_cam0 = R_C0_Ci @ p_cami + t_C0_Ci
      (T_C0_Ci 행렬의 상위 3×3이 R, 상위 3×1이 t)

    마스크가 없는 카메라(검출 실패)는 자동으로 건너뜀.
    """
    fid = f"{frame_idx:0{pad}d}"
    all_pts, all_cols = [], []

    for ci in cam_indices:
        if ci not in masks_per_cam:
            continue
        mask = masks_per_cam[ci]

        rgb_path = os.path.join(capture_dir, f"cam{ci}", f"rgb_{fid}.jpg")
        depth_path = os.path.join(capture_dir, f"cam{ci}", f"depth_{fid}.png")
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_u16 is None:
            continue

        pts_cam, cols = masked_depth_to_colored_points(
            depth_u16, rgb_bgr, mask, K_map[ci], D_map[ci], ds_map[ci],
            z_min, z_max, stride,
        )
        if len(pts_cam) == 0:
            print(f"    cam{ci}: 유효 점 없음 (마스크 내 depth 부족)")
            continue

        # cam_i → cam0
        T = T_map[ci]
        pts_cam0 = pts_cam @ T[:3, :3].T + T[:3, 3]

        all_pts.append(pts_cam0)
        all_cols.append(cols)
        print(f"    cam{ci}: {len(pts_cam):,} pts → cam0")

    if not all_pts:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.concatenate(all_pts), np.concatenate(all_cols)


# ================================================================
#  6. 통계적 아웃라이어 제거 (SOR)
# ================================================================

def statistical_outlier_removal(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    std_ratio: float = 1.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    통계적 아웃라이어 제거 (SOR).

    2단계 필터링:
      1단계 — 축별 IQR(사분위수 범위) 필터:
        X, Y, Z 각 축에 대해 [Q1 - 1.5×IQR, Q3 + 1.5×IQR] 범위 밖 제거
        → depth 노이즈로 인한 튀는 점(flying pixel) 제거

      2단계 — 중심 거리 필터:
        centroid까지 거리 분포에서 mean + std_ratio×std 초과 제거
        → 배경이 잘못 포함된 경우 추가 제거

    ICP 정확도에 직결: 아웃라이어가 포함되면 nearest-neighbor
    대응이 틀어져 SVD rigid transform 품질 하락.
    """
    n = len(points)
    if n < 10:
        return points, colors

    mask = np.ones(n, dtype=bool)
    # 1단계: 축별 IQR 필터
    for axis in range(3):
        vals = points[:, axis]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        mask &= (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)

    # 2단계: 중심 거리 필터
    centroid = points[mask].mean(axis=0) if mask.sum() > 0 else points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    mu_d = dists[mask].mean() if mask.sum() > 0 else dists.mean()
    std_d = dists[mask].std() if mask.sum() > 0 else dists.std()
    mask &= dists < (mu_d + std_ratio * std_d)

    removed = n - mask.sum()
    print(f"    SOR: {n:,} → {mask.sum():,} ({removed:,} removed)")
    return points[mask], (colors[mask] if colors is not None else None)


# ================================================================
#  7. PCA 포즈 추정
# ================================================================

def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    """R → Euler XYZ (degrees)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z]))


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """R → quaternion (w, x, y, z)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotation_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """R → (axis, angle_deg)."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.array([0.0, 0.0, 1.0]), 0.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= (np.linalg.norm(axis) + 1e-12)
    return axis, np.degrees(angle)


def estimate_pose_pca(points: np.ndarray) -> dict:
    """
    PCA(주성분 분석) 기반 6DoF 포즈 추정.

    ★ 위치: 점군 평균(centroid) = 물체 중심
    ★ 회전: 공분산 행렬의 고유벡터 3개 = 물체의 주축 3개

    PCA 원리:
      점군의 공분산 행렬 C = (1/N) × Σ (p_i - centroid)(p_i - centroid)^T
      고유분해: C × v_i = λ_i × v_i
        - v_i: 고유벡터 = 분산이 λ_i인 방향 (= 물체의 i번째 주축)
        - λ_i: 고유값   = 해당 방향의 분산 크기
      고유값 내림차순 정렬:
        - v_0: 분산 최대 방향 (물체의 가장 긴 축)
        - v_1: 분산 중간 방향 (두 번째 축)
        - v_2: 분산 최소 방향 (가장 짧은 축, 주로 높이)
      이 3개 벡터를 열벡터로 쌓으면 → 3×3 회전행렬 R

    한계: 고유벡터는 +/- 부호가 정해지지 않아 180° 모호성 있음.
          → 정확한 회전이 필요하면 estimate_pose_icp() 사용.

    OBB(Oriented Bounding Box): 주축 방향으로 투영 후 min/max → 크기
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    # 3×3 공분산 행렬 계산
    cov = (centered.T @ centered) / (len(points) - 1)
    # 고유분해 (np.linalg.eigh: 대칭행렬 전용, 수치적으로 안정적)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 고유값 내림차순 정렬 (분산 큰 주축부터)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 오른손 좌표계 보장 (det(R) = +1)
    # det < 0 이면 반사(reflection) → Z축 반전으로 수정
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    R = eigenvectors  # 3×3 회전행렬 (각 열 = 주축 방향)
    # 주축 방향으로 투영해 OBB 크기 계산
    projected = centered @ R
    extents = projected.max(axis=0) - projected.min(axis=0)

    euler = rotation_to_euler(R)
    quat = rotation_to_quaternion(R)
    axis, angle = rotation_to_axis_angle(R)

    return {
        "position_m": centroid,
        "position_mm": centroid * 1000,
        "rotation_matrix": R,
        "euler_xyz_deg": euler,
        "quaternion_wxyz": quat,
        "axis_angle": {"axis": axis, "angle_deg": angle},
        "obb_extents_m": extents,
        "obb_extents_mm": extents * 1000,
        "eigenvalues": eigenvalues,
    }


# ================================================================
#  8. Isaac Lab 좌표 변환
# ================================================================

# cam0 (OpenCV)  →  Isaac Lab (USD, Z-up) 좌표 변환행렬
#
#   cam0 OpenCV 규약: X→오른쪽, Y→아래(중력방향), Z→앞(깊이)
#   Isaac Lab USD 규약: X→오른쪽, Y→앞,         Z→위(중력 반대)
#
#   매핑:
#     X_isaac =  X_cam0   (오른쪽 = 오른쪽, 동일)
#     Y_isaac =  Z_cam0   (앞 = 깊이 방향, 동일)
#     Z_isaac = -Y_cam0   (위 = cam0 아래의 반대)
#
#   위치:    p_isaac = T_ISAAC_CAM0 @ p_cam0
#   회전:    R_isaac = T_ISAAC_CAM0 @ R_cam0 @ T_ISAAC_CAM0.T
T_ISAAC_CAM0 = np.array([
    [1, 0,  0],
    [0, 0,  1],
    [0, -1, 0],
], dtype=np.float64)


def convert_to_isaac(pose: dict) -> dict:
    """cam0 포즈 → Isaac Lab 포즈."""
    pos_cam0 = pose["position_m"]
    R_cam0 = pose["rotation_matrix"]

    pos_isaac = T_ISAAC_CAM0 @ pos_cam0
    R_isaac = T_ISAAC_CAM0 @ R_cam0 @ T_ISAAC_CAM0.T
    quat_isaac = rotation_to_quaternion(R_isaac)
    euler_isaac = rotation_to_euler(R_isaac)

    return {
        "position_m": pos_isaac.tolist(),
        "position_mm": (pos_isaac * 1000).tolist(),
        "quaternion_wxyz": quat_isaac.tolist(),
        "euler_xyz_deg": euler_isaac.tolist(),
        "rotation_matrix": R_isaac.tolist(),
        "obb_extents_m": pose["obb_extents_m"].tolist(),
    }


# ================================================================
#  9. PLY 저장
# ================================================================

def save_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
    """ASCII PLY 저장."""
    n = len(points)
    has_color = colors is not None and len(colors) == n
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if has_color:
                r, g, b = np.clip(colors[i] * 255, 0, 255).astype(int)
                line += f" {r} {g} {b}"
            f.write(line + "\n")
    print(f"  [PLY] {path}  ({n:,} pts)")


# ================================================================
#  9b. 레퍼런스 PLY 로드 + ICP 회전 추정
#
#  PCA만으로 회전을 구하면 180° 부호 모호성이 있음.
#  해결: 레퍼런스 CAD 모델(tiger.ply 등)과 ICP 정합으로
#  회전을 정확히 결정.
#
#  핵심 흐름:
#    ① 두 점군을 zero-centroid + 단위 스케일로 정규화
#       (레퍼런스 PLY 단위가 mm/m/임의 무관)
#    ② PCA로 초기 정렬값 생성
#    ③ 4가지 180° 부호 조합 → 각각 ICP → RMSE 최소 선택
#       → 180° 모호성 자동 해소
#    ④ 회전(R): ICP 결과 / 위치(t): 측정 점군 centroid
# ================================================================

def load_ply_points(path: str, max_pts: int = 5000) -> np.ndarray:
    """
    PLY 파일에서 xyz 점군 로드 (ICP 레퍼런스 모델용).

    ASCII / binary-little-endian 자동 감지.
    헤더를 파싱해 속성 타입과 순서를 확인 후 데이터 읽음.

    max_pts: 무작위 다운샘플 상한.
      ICP는 1200 source × 2000 target 서브샘플로 작동하므로
      5000점이면 충분 (수백만 점 PLY도 빠르게 처리).

    단위 불일치 걱정 없음: estimate_pose_icp()에서
    두 점군을 각각 단위 스케일로 정규화하므로 mm/m/임의 단위 무관.
    """
    PLY_DTYPE = {
        "float": "f4", "float32": "f4",
        "double": "f8", "float64": "f8",
        "uchar": "u1", "uint8": "u1",
        "char": "i1", "int8": "i1",
        "short": "i2", "int16": "i2",
        "ushort": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4",
        "uint": "u4", "uint32": "u4",
    }

    with open(path, "rb") as f:
        n_verts = 0
        is_binary = False
        props = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            elif "binary_little_endian" in line:
                is_binary = True
            elif line.startswith("property"):
                parts = line.split()
                if len(parts) >= 3:
                    props.append((parts[-1], parts[-2]))  # (name, ply_type)
            elif line == "end_header":
                break

        if n_verts == 0:
            raise ValueError(f"PLY vertex 없음: {path}")

        if is_binary:
            dt = np.dtype([(p[0], PLY_DTYPE.get(p[1], "f4")) for p in props])
            data = np.frombuffer(f.read(n_verts * dt.itemsize), dtype=dt)
            xyz = np.column_stack([
                data["x"].astype(np.float64),
                data["y"].astype(np.float64),
                data["z"].astype(np.float64),
            ])
        else:
            rows = []
            for _ in range(n_verts):
                parts = f.readline().decode("ascii", errors="ignore").split()
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            xyz = np.array(rows, dtype=np.float64)

    if len(xyz) > max_pts:
        idx = np.random.choice(len(xyz), max_pts, replace=False)
        xyz = xyz[idx]

    print(f"  [REF PLY] {os.path.basename(path)}: {n_verts:,} pts → {len(xyz):,} sampled")
    return xyz


def icp_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    init_R: np.ndarray = None,
    init_t: np.ndarray = None,
    max_iter: int = 60,
    tol: float = 1e-5,
    max_dist_frac: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Point-to-Point ICP (pure numpy, scipy 불필요).

    ★ ICP 알고리즘 원리:
      목표: source 점군을 R, t로 변환했을 때 target과 최대한 겹치게 하는 R, t 탐색

      반복 1회:
        ① source @ R_cur.T + t_cur  → 현재 변환 적용
        ② 각 source 점의 target 최근접 이웃(NN) 탐색
           (서브샘플로 속도 최적화: scipy 없이 numpy 행렬 연산으로 구현)
        ③ 유효 대응쌍(거리 임계 이내)으로 SVD rigid transform 계산:
             H = (src_corr - c_s)^T × (tgt_corr - c_t)  [교차공분산]
             U, S, Vt = SVD(H)
             R_delta = Vt^T × U^T                        [최적 회전]
             t_delta = c_t - R_delta × c_s               [최적 이동]
             → det(R)<0 이면 반사 → Vt[-1] 부호 반전으로 수정
        ④ 누적 변환 업데이트:
             R_cur = R_delta @ R_cur
             t_cur = R_delta @ t_cur + t_delta
        ⑤ RMSE 변화 < tol 이면 수렴 종료

    max_dist_frac: target 대각선 × 이 비율 이내 대응만 사용 (노이즈 방지)
    Returns: (R, t, rmse)  — source @ R.T + t ≈ target
    """
    # 초기 R, t 설정
    R_cur = np.eye(3) if init_R is None else init_R.copy()
    t_cur = np.zeros(3) if init_t is None else init_t.copy()

    # 최대 대응 거리: target 바운딩박스 대각선의 30%
    tgt_diag = np.linalg.norm(target.max(axis=0) - target.min(axis=0))
    max_d = max_dist_frac * tgt_diag

    # 서브샘플 크기 (속도 최적화)
    # 1200 × 2000 거리행렬 = 1200×2000×8B ≈ 19 MB (메모리 안전)
    N_SRC = min(1200, len(source))
    N_TGT = min(2000, len(target))

    prev_rmse = np.inf
    for _ in range(max_iter):
        # ① 현재 변환 적용 (전체 source)
        src_t = source @ R_cur.T + t_cur

        # 서브샘플 (매 iter 랜덤 → stochastic ICP, local minima 탈출 효과)
        idx_s = np.random.choice(len(src_t), N_SRC, replace=len(src_t) < N_SRC)
        idx_t = np.random.choice(len(target), N_TGT, replace=len(target) < N_TGT)
        src_sub = src_t[idx_s]
        tgt_sub = target[idx_t]

        # ② 최근접 이웃: (N_SRC × N_TGT) 제곱거리 행렬 → argmin
        D2 = np.sum((src_sub[:, None, :] - tgt_sub[None, :, :]) ** 2, axis=2)
        nn_idx = np.argmin(D2, axis=1)       # 각 src 점의 가장 가까운 tgt 인덱스
        nn_d = np.sqrt(D2[np.arange(N_SRC), nn_idx])  # 실제 거리

        # 거리 임계 초과 대응 제외 (배경 노이즈 방지)
        valid = nn_d < max_d
        if valid.sum() < 10:
            break  # 유효 대응 너무 적으면 종료

        src_corr = src_sub[valid]
        tgt_corr = tgt_sub[nn_idx[valid]]
        rmse = float(np.sqrt(np.mean(nn_d[valid] ** 2)))

        # ⑤ 수렴 판정
        if abs(prev_rmse - rmse) < tol:
            prev_rmse = rmse
            break
        prev_rmse = rmse

        # ③ SVD로 최적 rigid transform 계산 (src_corr → tgt_corr)
        c_s = src_corr.mean(axis=0)  # src 대응점 centroid
        c_t = tgt_corr.mean(axis=0)  # tgt 대응점 centroid
        H = (src_corr - c_s).T @ (tgt_corr - c_t)  # 교차공분산 행렬
        U, _, Vt = np.linalg.svd(H)
        R_d = Vt.T @ U.T  # 최적 회전
        if np.linalg.det(R_d) < 0:
            # 반사 행렬 수정 (det=-1 → det=+1): 마지막 특이벡터 부호 반전
            Vt[-1] *= -1
            R_d = Vt.T @ U.T
        t_d = c_t - R_d @ c_s  # 최적 이동

        # ④ 누적 변환 업데이트 (새 변환 = delta 위에 이전 변환 합성)
        R_cur = R_d @ R_cur
        t_cur = R_d @ t_cur + t_d

    return R_cur, t_cur, prev_rmse


def estimate_pose_icp(
    pts_object: np.ndarray,
    ref_pts: np.ndarray,
    max_iter: int = 60,
) -> dict:
    """
    ICP 기반 6DoF 포즈 추정 (PCA 초기화 + 4가지 부호 조합).

    ★ 핵심 아이디어:
      PCA만으로는 각 주축의 +/- 방향이 정해지지 않아
      최대 4가지 유효한 회전이 존재 (180° 모호성).
      → 4가지 초기값으로 각각 ICP → 가장 낮은 RMSE 선택
      → 물체 형태 자체(tiger.ply)가 정답을 알고 있음:
         뒤집힌 방향으로는 tiger.ply와 잘 안 맞아 RMSE 크게 나옴

    ★ 스케일 정규화:
      두 점군을 각각 (zero-centroid, 단위 RMS 반경)으로 정규화.
      → tiger.ply 단위가 mm/m/임의값이어도 상관없음.
      → 회전(R)은 스케일 불변이므로 정규화해도 동일한 R 획득.

    회전: ICP 결과 → 정확 (180° 모호성 없음)
    위치: 측정 점군 centroid → 카메라 좌표계 기준 물체 중심 (미터)
    """
    # ── 1. Centroid 이동 + 스케일 정규화
    #   zero-centroid: 두 점군 모두 원점 중심으로
    #   단위 스케일: RMS 반경 = 1 (두 점군이 같은 스케일)
    c_obj = pts_object.mean(axis=0)   # 측정 점군 centroid (카메라 좌표)
    c_ref = ref_pts.mean(axis=0)      # 레퍼런스 centroid
    obj_c = pts_object - c_obj        # zero-centroid 이동
    ref_c = ref_pts - c_ref

    # RMS 반경으로 스케일 정규화 (두 점군을 단위 구에 맞춤)
    s_obj = np.sqrt(np.mean(np.sum(obj_c ** 2, axis=1))) + 1e-12
    s_ref = np.sqrt(np.mean(np.sum(ref_c ** 2, axis=1))) + 1e-12
    obj_n = obj_c / s_obj  # 정규화된 측정 점군
    ref_n = ref_c / s_ref  # 정규화된 레퍼런스

    # ── 2. PCA로 초기 정렬 방향 계산
    #   두 점군의 PCA 주축을 각각 구한 뒤, obj 주축 → ref 주축으로의
    #   회전 R_pca를 ICP 초기값으로 사용 (수렴 속도 향상)
    def _pca_axes(pts: np.ndarray) -> np.ndarray:
        """공분산 고유분해 → 주축 행렬(3×3, 열=주축 방향)."""
        cov = pts.T @ pts / max(len(pts) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # 분산 큰 순
        if np.linalg.det(evecs) < 0:
            evecs[:, 2] *= -1  # 오른손 좌표계 보장
        return evecs

    E_obj = _pca_axes(obj_n)           # 측정 점군 PCA 주축
    E_ref = _pca_axes(ref_n)           # 레퍼런스 PCA 주축
    R_pca = E_ref @ E_obj.T            # obj 주축 → ref 주축 회전

    # ── 3. 4가지 180° 부호 조합 × ICP → RMSE 최소 선택
    #
    #   PCA 고유벡터는 +/- 방향이 수학적으로 정해지지 않음.
    #   오른손 좌표계(det=+1)를 유지하면서 부호를 바꾸는 경우의 수:
    #     [+,+,+], [-,-,+], [-,+,-], [+,-,-]  → 총 4가지
    #   각 조합으로 R_init을 만들어 ICP 실행 → RMSE 최소 조합 채택.
    #
    #   결과: 물체 형태(tiger.ply)와 가장 잘 맞는 방향 = 실제 방향
    sign_combos = [
        np.diag([1.0,   1.0,  1.0]),   # 기본 (부호 반전 없음)
        np.diag([-1.0, -1.0,  1.0]),   # X, Y 반전
        np.diag([-1.0,  1.0, -1.0]),   # X, Z 반전
        np.diag([1.0,  -1.0, -1.0]),   # Y, Z 반전
    ]

    best_rmse = np.inf
    best_R = np.eye(3)

    for i, S in enumerate(sign_combos):
        R_init = R_pca @ S             # PCA 초기값에 부호 조합 적용
        if np.linalg.det(R_init) < 0:
            R_init[:, 2] *= -1         # 혹시 det<0이면 보정
        R, t, rmse = icp_point_to_point(
            obj_n, ref_n,
            init_R=R_init, init_t=np.zeros(3),  # 둘 다 zero-centroid이므로 t_init=0
            max_iter=max_iter,
        )
        print(f"    sign_combo {i}: RMSE(norm)={rmse:.5f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_R = R.copy()          # 가장 낮은 RMSE의 R 저장

    # 오른손 좌표계 최종 보장
    if np.linalg.det(best_R) < 0:
        best_R[:, 2] *= -1

    # ── 4. 포즈 조립
    #   회전: ICP 결과 best_R (정규화 공간에서 구했지만 스케일 불변이므로 유효)
    #   위치: 측정 점군의 원래 centroid c_obj (카메라 좌표계, 미터)
    centroid = c_obj  # 미터 단위 (정규화 전 값 사용)
    # OBB 크기: 최적 회전축으로 투영 후 min-max 범위
    projected = (pts_object - centroid) @ best_R
    extents = projected.max(axis=0) - projected.min(axis=0)

    euler = rotation_to_euler(best_R)
    quat = rotation_to_quaternion(best_R)
    axis, angle = rotation_to_axis_angle(best_R)

    print(f"  ICP best RMSE (normalized): {best_rmse:.5f}")
    return {
        "position_m": centroid,
        "position_mm": centroid * 1000,
        "rotation_matrix": best_R,
        "euler_xyz_deg": euler,
        "quaternion_wxyz": quat,
        "axis_angle": {"axis": axis, "angle_deg": angle},
        "obb_extents_m": extents,
        "obb_extents_mm": extents * 1000,
        "eigenvalues": np.zeros(3),
        "icp_rmse_normalized": float(best_rmse),
    }


# ================================================================
#  9b. 멀티프레임 자체 레퍼런스 구축
# ================================================================

def build_reference_from_frames(
    frame_indices: List[int],
    text_prompt: str,
    capture_dir: str,
    cam_indices: List[int],
    pad: int,
    K_map: dict,
    D_map: dict,
    ds_map: dict,
    T_map: dict,
    gdino_proc, gdino_model,
    sam_pred,
    device: str,
    z_min: float = 0.1,
    z_max: float = 1.5,
    stride: int = 1,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    bbox_pad_ratio: float = 0.10,
    sam_refine_iters: int = 1,
    max_ref_pts: int = 20000,
) -> Optional[np.ndarray]:
    """
    여러 프레임에서 동일 물체를 검출·분리하여 자체 레퍼런스 점군 구축.

    원리:
      CAD 모델이 없을 때, 같은 물체가 찍힌 여러 프레임에서 SAM2 마스크로
      3D 점군을 추출한 뒤 cam0 좌표계로 통합한다.
      - 동일 물체가 고정되어 있으면 프레임마다 같은 형상이 중첩됨
      - 3대 카메라 × N 프레임 = 3N개 뷰 → 더 밀도 높은 점군
      - 노이즈 평균화 효과 (같은 표면을 여러 번 샘플링)

    주의:
      - 물체 위치가 고정된 경우에만 유효 (물체가 움직이면 형상이 흐려짐)
      - 단일 프레임보다 더 완전한 3D 형상 → ICP 수렴 안정성 향상
      - 180° 모호성: 물체 형상이 비대칭일수록 잘 해소됨
    """
    all_pts = []

    for fi in frame_indices:
        fid_str = f"{fi:0{pad}d}"
        masks_per_cam_ref: Dict[int, np.ndarray] = {}

        print(f"  Frame {fi}: ", end="", flush=True)

        for ci in cam_indices:
            rgb_path = os.path.join(capture_dir, f"cam{ci}", f"rgb_{fid_str}.jpg")
            if not os.path.exists(rgb_path):
                continue
            rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

            result = detect_and_segment_object(
                gdino_proc, gdino_model, sam_pred, rgb,
                text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                bbox_pad_ratio=bbox_pad_ratio,
                sam_refine_iters=sam_refine_iters,
                device=device,
            )
            if result is None:
                print(f"cam{ci}✗ ", end="", flush=True)
                continue

            mask, _ = result
            masks_per_cam_ref[ci] = mask
            print(f"cam{ci}✓ ", end="", flush=True)

        print()

        if not masks_per_cam_ref:
            print(f"    [SKIP] frame {fi}: 검출 없음")
            continue

        pts, _ = fuse_masked_multicam(
            capture_dir, fi, cam_indices,
            masks_per_cam_ref, K_map, D_map, ds_map, T_map,
            z_min, z_max, stride, pad,
        )

        if len(pts) < 10:
            continue

        pts_clean, _ = statistical_outlier_removal(pts, std_ratio=1.5)
        if len(pts_clean) > 0:
            all_pts.append(pts_clean)
            print(f"    → {len(pts_clean):,} pts")

    if not all_pts:
        return None

    ref_pts = np.concatenate(all_pts, axis=0)
    print(f"  합계: {len(ref_pts):,} pts (raw)")

    # 균일 다운샘플링
    if len(ref_pts) > max_ref_pts:
        idx = np.linspace(0, len(ref_pts) - 1, max_ref_pts, dtype=int)
        ref_pts = ref_pts[idx]
        print(f"  다운샘플링 → {len(ref_pts):,} pts")

    return ref_pts


# ================================================================
#  10. 시각화
# ================================================================

_VIS_COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
]


def draw_annotated_image(
    rgb_bgr: np.ndarray,
    mask: np.ndarray,
    det_info: dict,
    centroid_3d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """검출 결과 시각화 (bbox + expanded bbox + mask + depth 거리)."""
    vis = rgb_bgr.copy()
    color = _VIS_COLORS[0]
    box = det_info["box"]
    label = det_info["label"]
    score = det_info["score"]
    x1, y1, x2, y2 = box.astype(int)

    # 반투명 마스크
    if mask.sum() > 0:
        overlay = vis.copy()
        m = mask
        if m.shape[:2] != vis.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (vis.shape[1], vis.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
        overlay[m] = np.array(color, dtype=np.uint8)
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    # expanded bbox (점선 효과 - 밝은 색)
    if "expanded_box" in det_info:
        eb = det_info["expanded_box"].astype(int)
        cv2.rectangle(vis, (eb[0], eb[1]), (eb[2], eb[3]), (200, 200, 200), 1)

    # 원래 bbox (굵은 선)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    merged = det_info.get("merged_count", 1)
    label_text = f"{label} {score:.2f}"
    if merged > 1:
        label_text += f" [{merged} merged]"
    if centroid_3d is not None:
        dist_mm = np.linalg.norm(centroid_3d) * 1000
        label_text += f" d={dist_mm:.0f}mm"

    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(vis, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return vis


def visualize_pose_3d(T_map: dict, pose: dict, label: str, out_path: str,
                      isaac_pose: Optional[dict] = None, show: bool = False):
    """cam0/cam1/cam2 + 물체 포즈 3D 시각화."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Object Pose: \"{label}\"  (cam0 frame, mm)", fontsize=13, pad=15)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    def draw_axes(R, t, length, lbl, lw=2, alpha=1.0):
        colors_ax = ["#e74c3c", "#27ae60", "#2980b9"]
        names = ["X", "Y", "Z"]
        for i in range(3):
            v = R[:, i] * length
            ax.quiver(t[0], t[1], t[2], v[0], v[1], v[2],
                      color=colors_ax[i], linewidth=lw, arrow_length_ratio=0.12, alpha=alpha)
        if lbl:
            ax.text(t[0], t[1], t[2] - length * 0.4, lbl,
                    fontsize=9, fontweight="bold", ha="center")

    def draw_cam(R, t, size, color, lbl):
        s = size
        local = np.array([
            [0, 0, 0], [-s, -s * 0.7, s * 1.5], [s, -s * 0.7, s * 1.5],
            [s, s * 0.7, s * 1.5], [-s, s * 0.7, s * 1.5]
        ])
        pts = (R @ local.T).T + t
        for i in range(1, 5):
            j = (i % 4) + 1
            ax.add_collection3d(Poly3DCollection(
                [[pts[0], pts[i], pts[j]]], alpha=0.12, facecolor=color,
                edgecolor=color, linewidth=0.5))
        ax.text(t[0], t[1], t[2] - size * 1.5, lbl,
                fontsize=9, fontweight="bold", ha="center", color=color)

    def draw_obb(R, center, extents, color):
        h = extents / 2
        corners_local = np.array([
            [-h[0], -h[1], -h[2]], [h[0], -h[1], -h[2]],
            [h[0], h[1], -h[2]], [-h[0], h[1], -h[2]],
            [-h[0], -h[1], h[2]], [h[0], -h[1], h[2]],
            [h[0], h[1], h[2]], [-h[0], h[1], h[2]],
        ])
        corners = (R @ corners_local.T).T + center
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            ax.plot3D(*zip(corners[i], corners[j]), color=color, lw=1.0, alpha=0.5)

    # cam0 원점
    draw_axes(np.eye(3), np.zeros(3), 70, "cam0 (ref)", lw=3)

    # cam1, cam2
    cam_colors = {1: "#e67e22", 2: "#8e44ad"}
    for ci in T_map:
        if ci == 0:
            continue
        R_ci = T_map[ci][:3, :3]
        t_ci = T_map[ci][:3, 3] * 1000
        draw_cam(R_ci, t_ci, 30, cam_colors.get(ci, "#95a5a6"), f"cam{ci}")
        draw_axes(R_ci, t_ci, 40, "", lw=1.5, alpha=0.5)

    # 물체
    obj_pos = pose["position_mm"]
    obj_R = pose["rotation_matrix"]
    obj_obb = pose["obb_extents_mm"]
    euler = pose["euler_xyz_deg"]

    draw_axes(obj_R, obj_pos, 55, "", lw=3.5)
    draw_obb(obj_R, obj_pos, obj_obb, "#c0392b")

    ax.text(obj_pos[0], obj_pos[1] - 50, obj_pos[2] + 75,
            f"\"{label}\"", fontsize=11, fontweight="bold", color="#c0392b", ha="center")
    ax.text(obj_pos[0], obj_pos[1] - 50, obj_pos[2] + 55,
            f"({obj_pos[0]:.1f}, {obj_pos[1]:.1f}, {obj_pos[2]:.1f}) mm",
            fontsize=8, color="#2c3e50", ha="center")
    ax.text(obj_pos[0], obj_pos[1] - 50, obj_pos[2] + 38,
            f"euler ({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}) deg",
            fontsize=8, color="#2c3e50", ha="center")
    ax.text(obj_pos[0], obj_pos[1] - 50, obj_pos[2] + 21,
            f"OBB {obj_obb[0]:.1f} x {obj_obb[1]:.1f} x {obj_obb[2]:.1f} mm",
            fontsize=7, color="#c0392b", ha="center")

    # 등비 축
    all_pts_ax = [[0, 0, 0], obj_pos.tolist()]
    for ci in T_map:
        if ci != 0:
            all_pts_ax.append((T_map[ci][:3, 3] * 1000).tolist())
    pts_arr = np.array(all_pts_ax)
    c = pts_arr.mean(axis=0)
    r = max((pts_arr.max(axis=0) - pts_arr.min(axis=0)).max() / 2 * 1.3, 1.0)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=25, azim=-55)

    # ── 하단 정보 패널 ──
    quat = pose["quaternion_wxyz"]
    dist_mm = np.linalg.norm(obj_pos)

    info_lines = [
        f"Axis: X=Red  Y=Green  Z=Blue",
        "",
        f"── cam0 (OpenCV: X-right, Y-down, Z-forward) ──",
        f"Position:  ({obj_pos[0]:+.1f}, {obj_pos[1]:+.1f}, {obj_pos[2]:+.1f}) mm   "
        f"Distance: {dist_mm:.1f} mm",
        f"Euler XYZ: ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}) deg   "
        f"Quat wxyz: ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})",
        f"OBB size:  {obj_obb[0]:.1f} x {obj_obb[1]:.1f} x {obj_obb[2]:.1f} mm",
    ]

    if isaac_pose is not None:
        ip = isaac_pose
        info_lines += [
            "",
            f"── Isaac Lab (USD: X-right, Y-forward, Z-up) ──",
            f"Position:  ({ip['position_mm'][0]:+.1f}, {ip['position_mm'][1]:+.1f}, {ip['position_mm'][2]:+.1f}) mm",
            f"Quat wxyz: ({ip['quaternion_wxyz'][0]:.4f}, {ip['quaternion_wxyz'][1]:.4f}, "
            f"{ip['quaternion_wxyz'][2]:.4f}, {ip['quaternion_wxyz'][3]:.4f})   "
            f"Euler XYZ: ({ip['euler_xyz_deg'][0]:.1f}, {ip['euler_xyz_deg'][1]:.1f}, {ip['euler_xyz_deg'][2]:.1f}) deg",
        ]

    info_text = "\n".join(info_lines)

    fig.text(0.5, 0.01, info_text,
             fontsize=8, ha="center", va="bottom",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#ecf0f1", ec="#bdc3c7", alpha=0.9))

    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  [VIS] {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ================================================================
#  Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GroundingDINO+SAM2 물체 검출 → 멀티카메라 RGB-D 6DoF 포즈 → Isaac Sim"
    )

    # 필수
    parser.add_argument("--capture_dir", required=True,
                        help="RGB-D 프레임 폴더 (cam0/, cam1/, cam2/)")
    parser.add_argument("--calib_dir", required=True,
                        help="T_C0_Ci.npy 캘리브레이션 폴더")
    parser.add_argument("--intrinsics_dir", default="../data/_intrinsics",
                        help="내부파라미터 폴더")
    parser.add_argument("--text_prompt", required=True,
                        help='검출 텍스트 (예: "bottle." "tiger figure.")')

    # 모델
    parser.add_argument("--gdino_model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2_checkpoint", default="../checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cpu", help="추론 디바이스 (cpu/cuda/mps)")

    # 프레임
    parser.add_argument("--frame", type=int, default=0, help="프레임 인덱스")

    # 검출 (개선: 낮은 threshold로 후보 최대 확보 → 병합)
    parser.add_argument("--box_threshold", type=float, default=0.15,
                        help="GroundingDINO bbox threshold (낮을수록 후보 많음)")
    parser.add_argument("--text_threshold", type=float, default=0.15,
                        help="GroundingDINO text threshold")
    parser.add_argument("--bbox_pad", type=float, default=0.10,
                        help="SAM2에 전달할 bbox 확장 비율 (0.10=10%%)")
    parser.add_argument("--sam_refine", type=int, default=1,
                        help="SAM2 point prompt 반복 정제 횟수")

    # Depth
    parser.add_argument("--z_min", type=float, default=0.1, help="최소 depth (m)")
    parser.add_argument("--z_max", type=float, default=1.5, help="최대 depth (m)")
    parser.add_argument("--stride", type=int, default=1, help="Depth 서브샘플링 stride")

    # ICP 레퍼런스 모델 — 아래 둘 중 하나 필수
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument(
        "--ref_ply",
        help=(
            "레퍼런스 CAD/스캔 PLY 경로. 예: ../data/3d_ply/tiger.ply\n"
            "ICP로 180° 모호성 없이 정확한 회전을 추정합니다.\n"
            "단위(mm/m/임의)와 무관하게 내부에서 자동 정규화합니다."
        ),
    )
    ref_group.add_argument(
        "--ref_frames",
        help=(
            "자체 레퍼런스 구축용 프레임 번호 (쉼표 구분). 예: 3,4,5,6\n"
            "CAD 모델 없을 때 사용. 지정 프레임에서 동일 물체를 검출·fuse하여\n"
            "레퍼런스 점군을 자동 생성 후 ICP에 사용합니다.\n"
            "물체가 고정되어 있는 경우에만 유효합니다."
        ),
    )
    parser.add_argument(
        "--icp_iter", type=int, default=60,
        help="ICP 최대 반복 횟수 (기본 60, 늘리면 정확도↑ 속도↓)",
    )

    # 출력
    parser.add_argument("--out_dir", default=None, help="출력 폴더")

    args = parser.parse_args()
    t_start = time.time()

    # ──────────────────────────────────────────────────────
    #  Step 0: 의존성 체크
    # ──────────────────────────────────────────────────────
    try:
        import torch
    except ImportError:
        print("[ERROR] torch 미설치:")
        print("  pip install torch torchvision transformers accelerate sam2")
        return

    # ──────────────────────────────────────────────────────
    #  Step 1: 캘리브레이션 로드
    # ──────────────────────────────────────────────────────
    print("=" * 60)
    print(" Step 1: Load Calibration")
    print("=" * 60)

    cam_indices = discover_cameras(args.capture_dir)
    frame_ids = discover_frames(args.capture_dir, cam_indices)
    pad = detect_zero_padding(args.capture_dir, cam_indices[0])

    if args.frame not in frame_ids:
        raise RuntimeError(f"frame {args.frame} 없음. 가능: {frame_ids}")

    K_map, D_map, ds_map = {}, {}, {}
    for ci in cam_indices:
        K, D, ds = load_intrinsics(args.intrinsics_dir, ci)
        K_map[ci], D_map[ci], ds_map[ci] = K, D, ds
        print(f"  cam{ci}: fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} scale={ds:.6f}")

    T_map = load_extrinsics(args.calib_dir, cam_indices)
    for ci in cam_indices:
        if ci == 0:
            print(f"  T_C0_C0: Identity")
        else:
            t = T_map[ci][:3, 3] * 1000
            print(f"  T_C0_C{ci}: t=({t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}) mm")

    print(f"\n  Cameras: {cam_indices} | Frame: {args.frame} | Prompt: \"{args.text_prompt}\"")

    # ──────────────────────────────────────────────────────
    #  Step 2: 모델 로딩
    # ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" Step 2: Load Models (device={args.device})")
    print("=" * 60)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  [WARN] CUDA 불가 → CPU")
        device = "cpu"
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print("  [WARN] MPS 불가 → CPU")
        device = "cpu"

    t_model = time.time()
    gdino_proc, gdino_model = load_grounding_dino(args.gdino_model, device)
    sam_pred = load_sam2(args.sam2_checkpoint, args.sam2_config, device)
    print(f"  모델 로딩: {time.time() - t_model:.1f}s")

    # ──────────────────────────────────────────────────────
    #  Step 3: 각 카메라 물체 검출 + 세그멘테이션 (개선판)
    # ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" Step 3: Detect & Segment Object (frame {args.frame})")
    print("=" * 60)

    fid_str = f"{args.frame:0{pad}d}"
    masks_per_cam: Dict[int, np.ndarray] = {}
    detections_per_cam: Dict[int, dict] = {}

    for ci in cam_indices:
        rgb_path = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid_str}.jpg")
        if not os.path.exists(rgb_path):
            print(f"  cam{ci}: RGB 없음, skip")
            continue

        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        # 개선된 검출+세그멘테이션
        result = detect_and_segment_object(
            gdino_proc, gdino_model, sam_pred, rgb,
            args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            bbox_pad_ratio=args.bbox_pad,
            sam_refine_iters=args.sam_refine,
            device=device,
        )
        if result is None:
            print(f"  cam{ci}: 검출 없음")
            continue

        mask, det_info = result
        mask_area = mask.sum()
        total_px = rgb.shape[0] * rgb.shape[1]
        print(f"  cam{ci}: \"{det_info['label']}\" score={det_info['score']:.3f} "
              f"(후보 {det_info['total_candidates']}개 → {det_info['merged_count']}개 병합)")
        print(f"    bbox:  [{det_info['box'][0]:.0f},{det_info['box'][1]:.0f},"
              f"{det_info['box'][2]:.0f},{det_info['box'][3]:.0f}] "
              f"→ expanded [{det_info['expanded_box'][0]:.0f},{det_info['expanded_box'][1]:.0f},"
              f"{det_info['expanded_box'][2]:.0f},{det_info['expanded_box'][3]:.0f}]")
        print(f"    mask:  {mask_area:,} px ({mask_area / total_px * 100:.1f}%)")

        masks_per_cam[ci] = mask
        detections_per_cam[ci] = det_info

    if not masks_per_cam:
        print("\n[ERROR] 어떤 카메라에서도 물체를 검출하지 못했습니다.")
        return

    print(f"\n  검출 성공: {len(masks_per_cam)}대 카메라 → {list(masks_per_cam.keys())}")

    # ──────────────────────────────────────────────────────
    #  Step 4: 마스크 기반 멀티카메라 3D 융합
    # ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" Step 4: Masked Multi-camera 3D Fusion")
    print("=" * 60)

    pts_fused, cols_fused = fuse_masked_multicam(
        args.capture_dir, args.frame, cam_indices,
        masks_per_cam, K_map, D_map, ds_map, T_map,
        args.z_min, args.z_max, args.stride, pad,
    )

    if len(pts_fused) < 50:
        print("\n[ERROR] 융합된 점군이 너무 적습니다.")
        return

    print(f"    Fused: {len(pts_fused):,} pts")

    # SOR
    pts_clean, cols_clean = statistical_outlier_removal(pts_fused, cols_fused)

    if len(pts_clean) < 50:
        print("\n[ERROR] SOR 후 점군이 너무 적습니다.")
        return

    # ──────────────────────────────────────────────────────
    #  Step 5: 레퍼런스 준비 + ICP 포즈 추정
    # ──────────────────────────────────────────────────────
    if args.ref_ply:
        # ── CAD/스캔 PLY 직접 사용 ──
        print(f"\n{'=' * 60}")
        print(f" Step 5: ICP Pose Estimation  (ref: {os.path.basename(args.ref_ply)})")
        print("=" * 60)
        print(f"  레퍼런스 PLY 로드 중...")
        ref_pts = load_ply_points(args.ref_ply, max_pts=5000)
        rotation_method = "icp"

    else:
        # ── 멀티프레임 자체 레퍼런스 구축 ──
        ref_frame_list = [int(x.strip()) for x in args.ref_frames.split(",")]
        print(f"\n{'=' * 60}")
        print(f" Step 5a: Build Self-Reference  (frames {ref_frame_list})")
        print("=" * 60)
        print(f"  물체 '{args.text_prompt}' 를 프레임 {ref_frame_list}에서 추출·fuse합니다...")

        ref_pts = build_reference_from_frames(
            ref_frame_list, args.text_prompt,
            args.capture_dir, cam_indices, pad,
            K_map, D_map, ds_map, T_map,
            gdino_proc, gdino_model, sam_pred, device,
            z_min=args.z_min, z_max=args.z_max, stride=args.stride,
            box_threshold=args.box_threshold, text_threshold=args.text_threshold,
            bbox_pad_ratio=args.bbox_pad, sam_refine_iters=args.sam_refine,
        )

        if ref_pts is None or len(ref_pts) < 50:
            print("[ERROR] 레퍼런스 점군 구축 실패 — 검출된 프레임이 없거나 점이 너무 적습니다.")
            return

        # 자체 구축 레퍼런스 PLY 저장 (나중에 --ref_ply로 재사용 가능)
        out_dir_tmp = args.out_dir or "./output"
        os.makedirs(out_dir_tmp, exist_ok=True)
        obj_slug = args.text_prompt.strip().rstrip(".").replace(" ", "_")
        ref_save_path = os.path.join(out_dir_tmp, f"self_ref_{obj_slug}.ply")
        save_ply(ref_save_path, ref_pts, None)
        print(f"  [저장] 자체 레퍼런스 PLY → {ref_save_path}")
        print(f"  (다음 실행 시 --ref_ply {ref_save_path} 로 재사용 가능)")

        rotation_method = "icp_self_ref"

        print(f"\n{'=' * 60}")
        print(f" Step 5b: ICP Pose Estimation  (ref: self-built)")
        print("=" * 60)

    print(f"  레퍼런스 점군: {len(ref_pts):,} pts")
    print(f"  ICP 실행 중 (4 부호 조합 × {args.icp_iter} iter)...")
    pose = estimate_pose_icp(pts_clean, ref_pts, max_iter=args.icp_iter)

    pos_mm = pose["position_mm"]
    euler = pose["euler_xyz_deg"]
    obb_mm = pose["obb_extents_mm"]
    quat = pose["quaternion_wxyz"]

    print(f"\n  Method:       ICP")
    print(f"  Position:     ({pos_mm[0]:+.1f}, {pos_mm[1]:+.1f}, {pos_mm[2]:+.1f}) mm")
    print(f"  Euler XYZ:    ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}) deg")
    print(f"  Quaternion:   ({quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f})")
    print(f"  OBB size:     ({obb_mm[0]:.1f} x {obb_mm[1]:.1f} x {obb_mm[2]:.1f}) mm")
    print(f"  Distance:     {np.linalg.norm(pos_mm):.1f} mm from cam0")
    print(f"  ICP RMSE:     {pose.get('icp_rmse_normalized', 0):.5f} (normalized)")

    # ──────────────────────────────────────────────────────
    #  Step 6: Isaac Lab 좌표 변환
    # ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" Step 6: Convert to Isaac Lab Coordinates")
    print("=" * 60)

    isaac_pose = convert_to_isaac(pose)
    ip = isaac_pose

    print(f"  Position (m):     [{ip['position_m'][0]:.6f}, {ip['position_m'][1]:.6f}, {ip['position_m'][2]:.6f}]")
    print(f"  Position (mm):    [{ip['position_mm'][0]:.1f}, {ip['position_mm'][1]:.1f}, {ip['position_mm'][2]:.1f}]")
    print(f"  Quaternion wxyz:  [{ip['quaternion_wxyz'][0]:.6f}, {ip['quaternion_wxyz'][1]:.6f}, "
          f"{ip['quaternion_wxyz'][2]:.6f}, {ip['quaternion_wxyz'][3]:.6f}]")
    print(f"  Euler XYZ (deg):  [{ip['euler_xyz_deg'][0]:.2f}, {ip['euler_xyz_deg'][1]:.2f}, "
          f"{ip['euler_xyz_deg'][2]:.2f}]")

    # Isaac Lab 코드 스니펫
    obj_name = args.text_prompt.strip().rstrip(".")
    pos = ip["position_m"]
    q = ip["quaternion_wxyz"]

    print(f"\n  # ── Isaac Lab Python snippet ──")
    print(f"  import torch")
    print(f"  pos  = torch.tensor([[{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]])")
    print(f"  quat = torch.tensor([[{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]])")
    print(f"  object.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))")

    print(f"\n  # ── Isaac Sim (omni.isaac) snippet ──")
    print(f'  from pxr import Gf')
    print(f'  prim = stage.GetPrimAtPath("/World/{obj_name.replace(" ", "_")}")')
    print(f'  prim.GetAttribute("xformOp:translate").Set('
          f'Gf.Vec3d({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}))')
    print(f'  prim.GetAttribute("xformOp:orient").Set('
          f'Gf.Quatd({q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}))')

    # ──────────────────────────────────────────────────────
    #  Step 7: 결과 저장
    # ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f" Step 7: Save Results")
    print("=" * 60)

    out_dir = args.out_dir or "./output"
    os.makedirs(out_dir, exist_ok=True)

    # PLY
    save_ply(
        os.path.join(out_dir, f"object_{obj_name.replace(' ', '_')}_frame{fid_str}.ply"),
        pts_clean, cols_clean,
    )

    # annotated 이미지
    for ci in masks_per_cam:
        rgb_path = os.path.join(args.capture_dir, f"cam{ci}", f"rgb_{fid_str}.jpg")
        rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            continue
        vis = draw_annotated_image(rgb_bgr, masks_per_cam[ci], detections_per_cam[ci])
        vis_path = os.path.join(out_dir, f"detected_cam{ci}_frame{fid_str}.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"  [IMG] {vis_path}")

    # 3D 시각화
    vis_path = os.path.join(out_dir, f"pose_3d_frame{fid_str}.png")
    visualize_pose_3d(T_map, pose, obj_name, vis_path, isaac_pose=isaac_pose)

    # JSON 결과
    result = {
        "frame": args.frame,
        "text_prompt": args.text_prompt,
        "object_label": obj_name,
        "detection": {
            "cameras_detected": list(masks_per_cam.keys()),
            "num_cameras": len(masks_per_cam),
            "per_camera": {
                str(ci): {
                    "label": detections_per_cam[ci]["label"],
                    "score": round(detections_per_cam[ci]["score"], 4),
                    "box_xyxy": detections_per_cam[ci]["box"].tolist(),
                    "expanded_box_xyxy": detections_per_cam[ci].get("expanded_box", detections_per_cam[ci]["box"]).tolist(),
                    "merged_count": detections_per_cam[ci].get("merged_count", 1),
                    "total_candidates": detections_per_cam[ci].get("total_candidates", 1),
                }
                for ci in detections_per_cam
            },
        },
        "fused_points": len(pts_fused),
        "clean_points": len(pts_clean),
        "rotation_method": rotation_method,
        "ref_ply": args.ref_ply,
        "ref_frames": args.ref_frames,
        "icp_rmse_normalized": pose.get("icp_rmse_normalized"),
        "cam0_pose": {
            "coordinate_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
            "position_m": pose["position_m"].tolist(),
            "position_mm": pose["position_mm"].tolist(),
            "rotation_matrix": pose["rotation_matrix"].tolist(),
            "euler_xyz_deg": pose["euler_xyz_deg"].tolist(),
            "quaternion_wxyz": pose["quaternion_wxyz"].tolist(),
            "axis_angle": {
                "axis": pose["axis_angle"]["axis"].tolist(),
                "angle_deg": float(pose["axis_angle"]["angle_deg"]),
            },
            "obb_extents_m": pose["obb_extents_m"].tolist(),
            "obb_extents_mm": pose["obb_extents_mm"].tolist(),
        },
        "isaac_pose": {
            "coordinate_frame": "Isaac Lab (USD: X-right, Y-forward, Z-up)",
            "position_m": isaac_pose["position_m"],
            "position_mm": isaac_pose["position_mm"],
            "quaternion_wxyz": isaac_pose["quaternion_wxyz"],
            "euler_xyz_deg": isaac_pose["euler_xyz_deg"],
            "rotation_matrix": isaac_pose["rotation_matrix"],
            "obb_extents_m": isaac_pose["obb_extents_m"],
        },
        "elapsed_sec": round(time.time() - t_start, 2),
    }

    json_path = os.path.join(out_dir, f"pose_{obj_name.replace(' ', '_')}_frame{fid_str}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] {json_path}")

    # ──────────────────────────────────────────────────────
    #  최종 요약
    # ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f" RESULT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Object:       \"{obj_name}\"")
    print(f"  Detected in:  {list(masks_per_cam.keys())} ({len(masks_per_cam)} cameras)")
    print(f"  Points:       {len(pts_clean):,} (after SOR)")
    print(f"")
    print(f"  ── cam0 (OpenCV) ──")
    print(f"  Position:     ({pos_mm[0]:+.1f}, {pos_mm[1]:+.1f}, {pos_mm[2]:+.1f}) mm")
    print(f"  Euler:        ({euler[0]:+.1f}, {euler[1]:+.1f}, {euler[2]:+.1f}) deg")
    print(f"  OBB:          ({obb_mm[0]:.1f} x {obb_mm[1]:.1f} x {obb_mm[2]:.1f}) mm")
    print(f"")
    print(f"  ── Isaac Lab (USD) ──")
    print(f"  Position:     ({ip['position_mm'][0]:+.1f}, {ip['position_mm'][1]:+.1f}, {ip['position_mm'][2]:+.1f}) mm")
    print(f"  Quaternion:   [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    print(f"  Euler:        [{ip['euler_xyz_deg'][0]:.2f}, {ip['euler_xyz_deg'][1]:.2f}, {ip['euler_xyz_deg'][2]:.2f}]")
    print(f"")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  Output:       {os.path.abspath(out_dir)}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()