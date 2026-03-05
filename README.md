# ArucoCube Multi-Camera Calibration

RealSense 다중 카메라 + ArUco 큐브 기반 캘리브레이션과 객체 포즈 추정을 다룹니다.  
루트 README는 `src2`(기준 파이프라인)와 `src3`(개선판 + 포즈 추정 확장)를 분리해 설명합니다.

## 준비 사항

- RealSense 카메라 연결 (USB 충분히 확보 후 realsense-viewer 에서 카메라 연결 확인)
- `librealsense` / 드라이버 설치
- Python 환경 준비 (`pyrealsense2`, OpenCV, NumPy, matplotlib 등)

## 환경 생성 (Conda)

프로젝트 루트에서:

```bash
conda env create -f "conda env/environment.yml"
conda activate multicam_cube
```

참고:
- 환경에 따라 `cv2.aruco`가 없을 수 있습니다.
- 이 경우 `opencv-contrib-python` 계열 설치가 필요할 수 있습니다.

## src2: 기준 캘리브레이션 파이프라인 (Step1~4)

`src2`는 멀티카메라 캘리브레이션의 기준 워크플로우입니다.

- `Step1_dump_intrinsics.py`: 카메라별 intrinsics + `device_map` 저장
- `Step2_capture_multi_cam.py`: 동기 RGB-D 캡처 (`meta.json`, RGB/Depth 저장)
- `Step3_calibrate_multi_cam_cube.py`: 큐브 기반 외부파라미터 추정 (`T_Cref_Ci`)
- `Step4_fuse_depth_to_ref_pcd.py`: ref 카메라 좌표계로 depth point cloud 융합
- `diagnose_detection.py`, `visualize_*.py`: 검출/캘리브레이션 진단 및 시각화 도구

### src2 실행 예시

```bash
cd src2

# Step1
python Step1_dump_intrinsics.py

# Step2
python Step2_capture_multi_cam.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --fps 15 --width 640 --height 480 \
  --min_markers 2 \
  --auto_save --stable_frames 3 --cooldown_ms 700 \
  --save_depth \
  --show

# Step3 (예: ref=cam0)
python Step3_calibrate_multi_cam_cube.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 0 \
  --min_markers 1 \
  --reproj_max_px 16 \
  --save_overlay \
  --overlay_max_per_cam 30

# Step4
python Step4_fuse_depth_to_ref_pcd.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 2 \
  --frame_idx 0 \
  --stride 4 \
  --z_min 0.2 --z_max 1.5 \
  --save_ply \
  --eval_icp
```

## src3: src2 개선 + 객체 포즈 추정

### src2 대비 캘리브레이션 개선 내용 (Step1~4 인프라 유지)

`src3`는 `src2`의 파이프라인을 유지하면서 캘리브레이션 운용 안정성을 강화

- `Step2_capture_multi_cam.py`: depth align on/off, 카메라 timeout/error 로깅, 주기적 카메라 통계 출력 옵션 추가
- `camera.py`: 카메라 시작 재시도, 시작 실패 시 하드웨어 리셋 옵션, warmup/timeout 제어 및 스트림 health 카운터 추가
- `aruco_cube.py`: depth 역투영 기반 3D-3D 정합(`depth_svd`) + PnP fallback 지원
- `Step4_fuse_depth_to_ref_pcd.py`: best-frame 자동선택, ROI 기반 depth 융합, 멀티프레임 depth fusion, NumPy-only PLY 저장 보강

### 객체 포즈 추정 디렉터리 구조

```text
src3/
├── pose_step5_localize/              ← GroundingDINO+SAM2 3D 위치추정
│   ├── Step5_localize_object_3d.py
│   └── output/ (7개 파일)
├── pose_step6_ply/                   ← PLY 기반 ICP 포즈 추정
│   ├── Step6_estimate_pose_from_ply.py
│   ├── visualize_pose.py
│   └── output/ (11개 파일)
├── pose_step7_direct/                ← RGB-D 직접 포즈 추정
│   ├── Step7_direct_pose_from_rgbd.py
│   └── output/ (5개 파일)
├── pose_estimate_grounding_sam/      ← GroundingDINO+SAM2 6DoF 풀 포즈
│   ├── estimate_object_pose.py
│   └── output/ (6개 파일)
│
├── convert_pose_to_isaac.py          ← 공통 유틸 (유지)
├── Step1~4*.py                       ← 캘리브레이션 인프라
├── data/                             ← 입력 데이터만
├── intrinsics/
└── checkpoints/
```

### 포즈 추정 코드 설명

- `pose_step5_localize/Step5_localize_object_3d.py`
  - GroundingDINO + SAM2로 객체 검출/분할 후, 마스크 depth를 역투영해 객체 3D 위치를 추정합니다.
  - 멀티카메라 좌표를 ref 카메라로 정합해 프레임별 위치를 `localization_results.json`으로 저장합니다.

- `pose_step6_ply/Step6_estimate_pose_from_ply.py`
  - `SAM3D/COLMAP/GS2Mesh` 계열 PLY를 ICP 기반으로 cam0 좌표계에 정합합니다.
  - `T_cam0_colmap.npy`를 추정/재사용하며, 정합된 PLY와 `pose_estimation_results.json`을 출력합니다.

- `pose_step7_direct/Step7_direct_pose_from_rgbd.py` 
  - ML 모델 없이 RGB-D만으로 직접 포즈를 구합니다.
  - 멀티카메라 depth 융합 -> 평면 제거 -> 클러스터링 -> PCA 주축 추정으로 위치/회전을 계산합니다.

- `pose_estimate_grounding_sam/estimate_object_pose.py` (최종 포즈 추정 모델로 체택)
  - GroundingDINO + SAM2 기반으로 6DoF 포즈를 추정하고 Isaac Lab 좌표계로 변환합니다.
  - 결과를 cam0/OpenCV 좌표계와 Isaac(USD) 좌표계 모두 JSON으로 저장합니다.

### 포즈 추정 코드 원리 및 결과

각 코드의 위치/회전 추정 원리와 실측 기반 정확도를 정리합니다.
테스트 물체: RealSense D415 3대, 칼(utility knife), frame 5 기준.

#### 원리 요약표

| 코드 | 검출 방식 | 점군 획득 | 위치 추정 | 회전 추정 |
|------|-----------|-----------|-----------|-----------|
| Step5_localize | GDino+SAM2 | 마스크×Depth 역투영 | 멀티뷰 depth centroid + N-view DLT 삼각측량 | **없음** |
| Step6_ply | 없음 (PLY 직접 입력) | SAM3D/COLMAP/gs2mesh PLY | PLY centroid | PCA → Umeyama + ICP (scale 포함) |
| Step7_direct | 없음 (depth 전체) | RANSAC 평면 제거 + BFS 클러스터링 | 점군 centroid | PCA (180° 모호성) 또는 ICP + CAD 모델 |
| estimate_object_pose | GDino+SAM2 | 마스크×Depth 3카메라 융합 | 멀티캠 융합 centroid | PCA 4조합 × ICP + ref_ply → RMSE 최솟값 선택 |
| pose_from_sam3d (최최종) | GDino+SAM2 또는 scene PLY 직접 | scene PLY (기생성) | scene PLY centroid | PCA + 색상 기반 180° 방향 판별 |

#### 위치(Position) 정확도

| 코드 | 정확도 | 비고 |
|------|--------|------|
| estimate_object_pose | ±2~3 mm | SAM2 정밀 마스크 × 3카메라 depth 융합 |
| Step5_localize | ±2~3 mm | 동일 SAM2 마스크, DLT 삼각측량 병행 |
| pose_from_sam3d (최최종) | ±2~3 mm | scene PLY(이미 SAM2로 생성) centroid 사용 |
| Step7_direct | ±3~5 mm | ML 없이 depth 전체 사용 → 클러스터 경계 오차 |
| Step6_ply | ±5~10 mm | COLMAP/SAM3D PLY 좌표계 변환 누적 오차 |

#### 회전(Rotation) 정확도

| 코드 | 정확도 | 비고 |
|------|--------|------|
| estimate_object_pose (+ ref_ply) | ±3~8° | SAM2 정밀 마스크 + 4조합 ICP → 180° 모호성 해결, CAD 형태로 정답 판별. 단, 부분 뷰 vs 완전 모델 불일치 시 ICP 불안정 가능 |
| pose_from_sam3d (최최종) | ±3~8° | 긴 물체(칼 등)에서 PCA 주축이 명확(aspect ratio 5.3), 색상 분석으로 날/손잡이 방향 판별. CAD 모델 불필요 |
| Step7_direct (+ ref_ply) | ±5~10° | CAD ICP 지원하나, ML 없이 추출한 점군에 노이즈 혼입 가능 |
| Step6_ply | ±5~15° | Umeyama(scale 포함) + ICP 수행하나, PLY 좌표계 변환 오차 누적 |
| Step7_direct (PCA만) | ±5~15° | ref_ply 없이 동작하나 180° 모호성 미해결 |
| Step5_localize | 불가 | 위치 추정 전용, 회전값 미출력 |

#### 결론

**일반 물체 (tiger figure 등 복잡한 형태):**
- `estimate_object_pose` + `ref_ply` 사용이 가장 정확합니다.
- SAM2 정밀 마스크 기반 3카메라 점군 + 4조합 PCA ICP로 180° 모호성 없는 6DoF 포즈를 획득합니다.

**긴 단순 형태 물체 (칼, 봉 등):**
- `pose_from_sam3d (최최종)` 이 동급 정확도를 제공하며 실행 속도가 훨씬 빠릅니다(~0.1초).
- PCA 주축이 형태에서 명확히 도출되고, 색상 분석으로 방향 판별이 가능하므로 CAD 모델 없이도 신뢰할 수 있습니다.

**위치만 필요할 때:**
- `Step5_localize` 로 충분합니다. DLT 삼각측량과 depth centroid를 이중으로 검증해 가장 안정적인 3D 위치를 출력합니다.

### src3 변경 사항

- 4개 포즈 추정 방법별로 스크립트와 결과를 각각 하나의 폴더로 정리
- 각 스크립트 기본 경로를 `../` 기준으로 통일 (`intrinsics`, `checkpoints`, `out_dir`)
- `pose_step6_ply/visualize_pose.py` 하드코딩 경로를 `os.path` 기반 상대경로로 변경
- 포즈 추정 스크립트 Python 문법 검증 완료

### src3 실행 방법

- 각 포즈 추정 폴더로 이동해서 상단에 실행 명령어 복붙하여 사용
- 입력 데이터는 `../data/...` 경로를 참조