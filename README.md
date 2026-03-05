# ArucoCube Multi-Camera Calibration

RealSense 다중 카메라 + ArUco 큐브 기반 캘리브레이션과 객체 포즈 추정을 다룹니다.
루트 README는 `src2`(기준 파이프라인)와 `src3`(depth 포함 개선판 + 포즈 추정)를 분리해 설명합니다.

## 준비 사항

- RealSense 카메라 연결 (USB 충분히 확보 후 realsense-viewer 에서 카메라 연결 확인)
- `librealsense` / 드라이버 설치
- Python 환경 준비 (`pyrealsense2`, OpenCV, NumPy, matplotlib 등)

## 가상 환경 생성 (Conda)

프로젝트 루트에서:

```bash
conda env create -f "conda env/environment.yml"
conda activate multicam_cube
```

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

## src3: src2 depth 포함 개선판 + 포즈 추정

### src2 대비 캘리브레이션 개선 내용 (Step1~4 인프라 유지)

`src3`는 `src2`의 파이프라인을 유지하면서 캘리브레이션 운용 안정성을 강화

- `Step2_capture_multi_cam.py`: depth align on/off, 카메라 timeout/error 로깅, 주기적 카메라 통계 출력 옵션 추가
- `_camera.py`: 카메라 시작 재시도, 시작 실패 시 하드웨어 리셋 옵션, warmup/timeout 제어 및 스트림 health 카운터 추가
- `_aruco_cube.py`: depth 역투영 기반 3D-3D 정합(`depth_svd`) + PnP fallback 지원
- `Step4_fuse_depth_to_ref_pcd.py`: best-frame 자동선택, ROI 기반 depth 융합, 멀티프레임 depth fusion, NumPy-only PLY 저장 보강

### src3 디렉터리 구조

```text
src3/
│
│  ── 캘리브레이션 파이프라인 (Step1~4) ──
├── Step1_dump_intrinsics.py                            ← 카메라 내부파라미터 저장
├── Step2_capture_multi_cam.py                          ← ArUco 큐브 다중 캡처
├── Step3_calibrate_multi_cam_cube.py                   ← 큐브 기반 외부파라미터 추정
├── Step4_fuse_depth_to_ref_pcd.py                      ← ref 카메라 기준 depth 융합
│
│  ── 캘리브레이션 시각화 (Step5) ──
├── Step5-1_visualize_3d_cube.py                        ← 큐브+카메라 PnP 3D 시각화
├── Step5-2_visualize_3d_refcam.py                      ← 전체 프레임 PnP scatter 시각화
├── visualize_calibration.py                            ← T_C0_Ci 캘리브레이션 결과 시각화
│
│  ── 물체 포즈 추정 파이프라인 (Obj_Step1~3) ──
├── Obj_Step1_capture_rgbd_3cam.py                      ← 물체 촬영 (3카메라 RGBD)
├── Obj_Step2-(1)_pose_estimate_grounding_sam/           ← 포즈 추정 방법1: GDino+SAM2+ICP
│   └── Obj_Step2-(1)_pose_grounding_sam.py
├── Obj_Step2-(2)_pose_estimate_sam3d/                   ← 포즈 추정 방법2: PCA+색상
│   └── Obj_Step2-(2)_pose_sam3d.py
├── Obj_Step3_visualize_object_pose.py                  ← 포즈 추정 결과 비교 시각화
│
│  ── 내부 모듈 ──
├── _aruco_cube.py                                      ← ArUco 큐브 모델/타겟
├── _camera.py                                          ← RealSense 카메라 제어
├── _utils_pose.py                                      ← 포즈 유틸 (quaternion 등)
│
│  ── 참고용 / 미사용 ──
├── (x) pose_not_used/                                  ← 이전 포즈 추정 코드 (참고용)
├── (x) arucocube_test/                                 ← ArUco 큐브 테스트/진단
├── (x) object_to_3d/                                   ← 3D 재구성 유틸
│
├── data/
│   ├── _intrinsics/                                    ← 카메라 내부파라미터
│   ├── cube_session_01/                                ← 캘리브레이션 세션 데이터
│   │   └── calib_out_cube/                             ← T_C0_C{1,2}.npy, T_C0_Ci_all.json
│   ├── object_capture/                                 ← 물체 촬영 RGB-D 데이터
│   └── 3d_ply/                                         ← 3D 모델/점군 PLY
└── checkpoints/                                        ← 모델 체크포인트 (SAM2 등)
```

### 포즈 추정 파이프라인

#### 전체 흐름

```
Obj_Step1: 물체 촬영  (capture_rgbd_3cam.py)
         ↓
   data/object_capture/cam{0,1,2}/rgb,depth
         ↓
   + data/cube_session_01/calib_out_cube/T_C0_C{1,2}.npy  (캘리브레이션)
   + data/_intrinsics/cam{0,1,2}.npz                       (내부파라미터)
         ↓
Obj_Step2: 포즈 추정 (2가지 방법 중 선택)
   ┌────────────────────────────────────────────────────────┐
   │ (1) Obj_Step2-(1)_pose_grounding_sam.py               │  ← GDino+SAM2+ICP
   │ (2) Obj_Step2-(2)_pose_sam3d.py                       │  ← PCA+색상
   └────────────────────────────────────────────────────────┘
         ↓
   pose JSON (position_mm + rotation_matrix, cam0 OpenCV 좌표계)
         ↓
Obj_Step3: 결과 시각화  (Obj_Step3_visualize_object_pose.py)
   → 포인트 클라우드 + OBB + 좌표축 + Euler 회전각 비교
```

#### Obj_Step1. 물체 촬영

```bash
cd src3
python Obj_Step1_capture_rgbd_3cam.py --save_dir ./data/object_capture
```

- 3대 RealSense로 RGB+Depth 동시 촬영
- `SPACE`: 프레임 저장, `s`: 연속 저장 모드, `ESC/q`: 종료

#### Obj_Step2-(1). 포즈 추정: grounding_sam (6DoF, ICP 기반)

```bash
cd "Obj_Step2-(1)_pose_estimate_grounding_sam"
python "Obj_Step2-(1)_pose_grounding_sam.py" \
  --capture_dir ../data/object_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "utility knife." \
  --ref_ply ../data/3d_ply/tiger.ply \
  --device mps
```

- GDino+SAM2로 물체 검출 → 마스크×Depth 3카메라 융합 → PCA 4조합×ICP로 6DoF 포즈
- 출력: `pose_<object>_frameXXXXXX.json` (cam0 + Isaac 좌표계), `object_*.ply`

#### Obj_Step2-(2). 포즈 추정: sam3d (고속, PCA 기반)

```bash
cd "Obj_Step2-(2)_pose_estimate_sam3d"

# Mode B: 기존 추출 PLY 사용 (빠름)
python "Obj_Step2-(2)_pose_sam3d.py" \
  --scene_ply "../Obj_Step2-(1)_pose_estimate_grounding_sam/output_selfref/object_utility_knife_frame000005.ply"

# Mode A: GDino+SAM2 재검출
python "Obj_Step2-(2)_pose_sam3d.py" --frame 5 --device mps
```

- PCA 주축 + 색상 기반 180도 방향 판별 (CAD 모델 불필요)
- 출력: `pose_cam0_*.json`, `pose_cam0_*.png`

#### Obj_Step3. 포즈 결과 시각화

```bash
cd src3
python Obj_Step3_visualize_object_pose.py

# 또는 특정 파일 지정
python Obj_Step3_visualize_object_pose.py \
  --grounding_json "Obj_Step2-(1)_pose_estimate_grounding_sam/output_frame000005/pose_utility_knife_frame000005.json" \
  --sam3d_json "Obj_Step2-(2)_pose_estimate_sam3d/output_frame000005/pose_cam0_object_utility_knife_frame000005.json"
```

시각화 내용:
- 물체 포인트 클라우드 (색상 포함)
- OBB (Oriented Bounding Box) — 물체 회전 방향을 직관적으로 표시
- 좌표축 (X=Red, Y=Green, Z=Blue)
- Euler XYZ 회전각도, 위치(mm), 거리 표시
- cam0 원점 기준 좌표계
- 방법별 2x2 개별 축 시각화 (ALL/X/Y/Z) 자동 생성

### 포즈 추정 코드 상세

#### 1) Obj_Step2-(1) grounding_sam (최종 6DoF)

- 입력 데이터
  - `capture_dir` RGB-D, `calib_dir`, `intrinsics_dir`, `text_prompt`
  - 레퍼런스: `--ref_ply` 또는 `--ref_frames`(둘 중 하나 필수)
- 사용 모델
  - GroundingDINO + SAM2
  - 기하 최적화: SOR + PCA 초기화 + ICP
- 결과
  - `pose_<object>_frameXXXXXX.json` (cam0 + Isaac 좌표계 모두 저장)
  - `pose_3d_frameXXXXXX.png`, `detected_cam*_frame*.jpg`, `object_*.ply`
- 회전 원리
  - SAM2 마스크로 얻은 멀티카메라 객체 점군을 cam0으로 융합 후 정규화
  - PCA 축 부호 4조합 × ICP → normalized RMSE 최소 조합 채택
  - cam0 포즈 + Isaac(USD) 좌표계 변환 저장

#### 2) Obj_Step2-(2) sam3d (고속, 칼 특화)

- 입력 데이터
  - Mode B: `--scene_ply` (이미 추출된 물체 점군)
  - Mode A: RGB-D + 캘리브레이션 + `text_prompt`
- 사용 모델
  - Mode B: 모델 없음 (기하 기반), Mode A: GDino+SAM2
- 결과
  - `pose_cam0_*.json`, `pose_cam0_*.png`
- 회전 원리
  - PCA length/normal axis 추출 → 색상 기반 180도 방향 판별
  - `width = cross(normal, length)` 오른손 좌표계 구성

### 포즈 추정 원리 요약표

| 코드 | 검출 방식 | 위치 추정 | 회전 추정 |
|------|-----------|-----------|-----------|
| Obj_Step2-(1) grounding_sam | GDino+SAM2 | 멀티캠 fusion centroid | PCA 4조합 × ICP + ref_ply |
| Obj_Step2-(2) sam3d | GDino+SAM2 또는 PLY 직접 | PLY centroid | PCA + 색상 방향 판별 |

### 정확도

| 코드 | 위치 | 회전 | 비고 |
|------|------|------|------|
| Obj_Step2-(1) (+ ref_ply) | ±2~3 mm | ±3~8° | 복잡한 형태 물체에 적합 |
| Obj_Step2-(2) | ±2~3 mm | ±3~8° | 긴 단순 형태(칼 등)에 적합, CAD 불필요 |

### 캘리브레이션 시각화

```bash
cd src3

# PnP 기반 (큐브 원점, 카메라 위치 확인)
python Step5-1_visualize_3d_cube.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./data/_intrinsics \
  --save

# 전체 프레임 PnP scatter (캘리브레이션 일관성 확인)
python Step5-2_visualize_3d_refcam.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./data/_intrinsics \
  --save

# 캘리브레이션 결과 (cam0 원점, T_C0_Ci 확인)
python visualize_calibration.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./data/_intrinsics \
  --ref_cam_idx 0 \
  --save
```

### 이전 포즈 추정 코드 (`(x) pose_not_used/`)

아래 코드들은 개발 과정에서 사용했으나 현재는 위 2가지 최종 방법으로 대체되어 참고용으로 보관됩니다.

- `pose_step5_localize/Step5_localize_object_3d.py`: 위치 전용 (회전 미추정)
- `pose_step6_ply/Step6_estimate_pose_from_ply.py`: PLY 정합 기반 ICP
- `pose_step7_direct/Step7_direct_pose_from_rgbd.py`: ML 없이 RGB-D 직접 포즈
