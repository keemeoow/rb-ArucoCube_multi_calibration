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

### 객체 포즈 추정 디렉터리 구조 (최신)

```text
src3/
├── pose_step5_localize/                  ← GroundingDINO+SAM2 3D 위치 추정
│   └── Step5_localize_object_3d.py
├── pose_step6_ply/                       ← PLY 기반 ICP 포즈 추정
│   ├── Step6_estimate_pose_from_ply.py
│   └── visualize_pose.py
├── pose_step7_direct/                    ← RGB-D 직접 포즈 추정
│   └── Step7_direct_pose_from_rgbd.py
├── pose_estimate_grounding_sam(최종)/    ← GDino+SAM2 6DoF + Isaac 변환
│   └── estimate_object_pose.py
├── pose_estimate_sam3d_knife(최최종)/    ← SAM3D/scene PLY 기반 고속 포즈
│   └── pose_from_sam3d.py
│
├── capture_rgbd_3cam.py                  ← RGB-D 캡처 유틸
├── reconstruct_3d.py                     ← 프레임 단위 재구성 유틸
├── reconstruct_sam3d_multicam.py         ← SAM3D 멀티캠 재구성 유틸
├── convert_pose_to_isaac.py              ← 좌표계 변환 유틸
├── Step1~4*.py                           ← 캘리브레이션 인프라
├── data/
├── intrinsics/
└── checkpoints/
```

### 포즈 추정 코드 설명

- `pose_step5_localize/Step5_localize_object_3d.py`
  - GroundingDINO + SAM2 검출/분할 후, 마스크 depth를 역투영해 객체 3D 위치를 추정합니다.
  - 필수 인자: `--capture_dir`, `--calib_dir`, `--text_prompt`
  - 출력: `localization_results.json`, `annotated_cam*_frame*.jpg`, (옵션) 객체 PLY

- `pose_step6_ply/Step6_estimate_pose_from_ply.py`
  - `SAM3D/COLMAP/GS2Mesh` PLY를 ICP로 cam0 좌표계에 정합합니다.
  - `--calib_dir --intrinsics_dir --rgbd_dir` 조합(권장) 또는 `--T_multicam_colmap` 재사용을 지원합니다.
  - 출력: `pose_estimation_results.json`, `T_cam0_colmap.npy`, 정합된 cam0 기준 PLY

- `pose_step7_direct/Step7_direct_pose_from_rgbd.py`
  - ML 모델 없이 RGB-D만으로 직접 포즈를 추정합니다.
  - `--ref_ply`를 주면 ICP 회전(권장), 없으면 PCA 회전(180도 모호성 존재)으로 동작합니다.
  - 출력: `pose_frame*.json`, `pose_frame*.png`, `cam0_fused_frame*.ply`, `object_frame*.ply`

- `pose_estimate_grounding_sam(최종)/estimate_object_pose.py` (최종 6DoF 파이프라인)
  - GroundingDINO + SAM2 + 멀티카메라 depth 융합 + ICP로 6DoF 포즈를 추정합니다.
  - `--ref_ply` 또는 `--ref_frames` 중 하나를 필수로 받아 레퍼런스 점군을 구성합니다.
  - 출력: cam0/OpenCV 좌표계 + Isaac(USD) 좌표계 JSON, 검출 이미지, 3D 시각화

- `pose_estimate_sam3d_knife(최최종)/pose_from_sam3d.py`
  - Mode B: 기존 `--scene_ply`를 바로 사용해 PCA 직접 포즈 추정
  - Mode A: GDino+SAM2 재검출 + 멀티캠 융합 후 PCA 직접 포즈 추정
  - 색상 기반 blade 방향 판별(`--blade_dir auto/pos/neg`)을 지원합니다.

### 포즈 축/회전축 시각화 방법 (정합 좌표계 확인)

기본 좌표축 색상은 공통으로 `X=Red, Y=Green, Z=Blue`입니다.

| 코드 | 좌표계 | 시각화 생성 방법 | 생성 파일 |
|------|--------|------------------|-----------|
| Step5_localize | ref(cam0) 위치만 | `annotated` 이미지(2D bbox/mask/거리) 확인 | `pose_step5_localize/output/annotated_cam*_frame*.jpg` |
| Step6_ply | cam0 + COLMAP | Step6 실행 후 `python visualize_pose.py` 실행 | `pose_step6_ply/output/pose_visualization.png` |
| Step7_direct | cam0 | Step7 실행 시 자동 저장 | `pose_step7_direct/output/pose_frameXXXXXX.png` |
| estimate_object_pose(최종) | cam0 + Isaac | 실행 시 자동 저장 | `pose_estimate_grounding_sam(최종)/output*/pose_3d_frameXXXXXX.png` |
| pose_from_sam3d(최최종) | cam0 | 실행 시 자동 저장 | `pose_estimate_sam3d_knife(최최종)/output/pose_cam0_*.png` |

참고:
- `Step5_localize`는 **위치 추정 전용**이라 회전축(orientation axis)은 출력하지 않습니다.
- 축 시각화 이미지는 모두 캘리브레이션으로 정합된 cam0 기준 결과를 사용합니다(단, Step6은 COLMAP 패널도 함께 표시).
- 전체 방법 비교: `python visualize_pose_axes_all.py --out ./pose_axes_comparison.png`
- 축 각각(X/Y/Z) 보기: 위 명령 실행 시 `./pose_axes_comparison_each_axes/*.png` 자동 생성

빠른 실행 예시:

```bash
cd src3

# Step6 축 시각화
cd pose_step6_ply
python Step6_estimate_pose_from_ply.py \
  --object_ply ../data/3d_ply/tiger.ply \
  --scene_ply ../data/3d_ply/point_cloud_cleaned.ply \
  --mesh_ply "../data/3d_ply/tiger_figure_custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_20_cleaned_mesh.ply" \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --intrinsics_dir ../intrinsics \
  --rgbd_dir ../data/rgbd_capture
python visualize_pose.py

# Step7 축 시각화 (자동)
cd ../pose_step7_direct
python Step7_direct_pose_from_rgbd.py \
  --rgbd_dir ../data/rgbd_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --ref_ply ../data/3d_ply/tiger.ply \
  --frame 0

# 최종(grounding_sam) 축 시각화 (자동)
cd "../pose_estimate_grounding_sam(최종)"
python estimate_object_pose.py \
  --capture_dir ../data/rgbd_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "tiger figure." \
  --ref_ply ../data/3d_ply/tiger.ply \
  --device mps

# 최최종(sam3d_knife) 축 시각화 (자동)
cd "../pose_estimate_sam3d_knife(최최종)"
python pose_from_sam3d.py \
  --scene_ply "../pose_estimate_grounding_sam(최종)/output_selfref/object_utility_knife_frame000005.ply"

# 전체 방법 축 비교 (src3 루트에서 실행)
cd ..
python visualize_pose_axes_all.py --out ./pose_axes_comparison.png
# 생성:
#   ./pose_axes_comparison.png
#   ./pose_axes_comparison_each_axes/step6_ply_axes.png (방법별 2x2: ALL/X/Y/Z)
```

### 코드별 입력·모델·결과·회전값 계산 원리 (상세)

아래에서 `정합 좌표계`는 기본적으로 `cam0(OpenCV: X-right, Y-down, Z-forward)`를 의미합니다.

#### 1) Step5_localize (위치 전용)

- 입력 데이터
  - `capture_dir/cam{0,1,2}/rgb_*.jpg`, `depth_*.png`
  - `calib_dir/T_C0_Ci.npy`, `intrinsics_dir/cam{i}.npz`
  - `text_prompt`
- 사용 모델
  - GroundingDINO + SAM2
- 결과
  - `localization_results.json`
  - `annotated_cam*_frame*.jpg`
  - (옵션) `object_*_frame*.ply`
- 원리
  - 카메라별로 `mask × depth`를 3D 역투영해 centroid를 구합니다.
  - 카메라별 centroid를 `T_C0_Ci`로 cam0에 정합한 뒤, 유효 depth 픽셀 수로 가중 평균합니다.
  - 보조로 N-view DLT 삼각측량을 계산해 depth 기반 위치와 불일치(mm)를 함께 기록합니다.
  - 회전값은 계산하지 않습니다.

#### 2) Step6_ply (PLY 정합 기반)

- 입력 데이터
  - `object_ply`(SAM3D), `scene_ply`(COLMAP), `mesh_ply`(gs2mesh)
  - 권장: `calib_dir + intrinsics_dir + rgbd_dir`로 cam0 브릿지 추정
- 사용 모델
  - 딥러닝 모델 없음(기하 기반): PCA, Umeyama(similarity), ICP(point-to-point)
- 결과
  - `pose_estimation_results.json`, `T_cam0_colmap.npy`
  - `*_in_cam0.ply`, `sam3d_aligned_to_colmap.ply`
  - `pose_visualization.png`(별도 `visualize_pose.py`)
- 회전 원리
  - 각 점군의 PCA 주축으로 초기 회전축을 만들고, 축 부호 모호성(180도)을 flip 조합으로 탐색합니다.
  - Umeyama로 scale+R+t 초기 정합 후 ICP로 미세 정합해 RMSE 최소 해를 선택합니다.
  - 최종 회전행렬을 Euler/Quaternion/Axis-Angle로 변환해 저장합니다.

#### 3) Step7_direct (RGB-D 직접 포즈)

- 입력 데이터
  - `rgbd_dir/cam{0,1,2}/rgb,depth`
  - `calib_dir/T_C0_Ci.npy`, `intrinsics_dir/cam{i}.npz`
  - 선택: `ref_ply`(CAD)
- 사용 모델
  - 딥러닝 모델 없음(순수 NumPy/OpenCV)
- 결과
  - `pose_frameXXXXXX.json`, `pose_frameXXXXXX.png`
  - `cam0_fused_frameXXXXXX.ply`, `object_frameXXXXXX.ply`
- 회전 원리
  - 공통 전처리: depth 역투영 → cam0 융합 → 평면 제거(RANSAC) → 객체 클러스터 추출
  - `--ref_ply` 없음: 객체 점군 공분산 고유벡터(PCA)로 회전축 추정 (180도 모호성 존재)
  - `--ref_ply` 있음: PCA 초기화 후 4개 부호 조합 × ICP 수행, 최소 RMSE 회전을 선택해 모호성 해소

#### 4) estimate_object_pose (최종 6DoF)

- 입력 데이터
  - `capture_dir` RGB-D, `calib_dir`, `intrinsics_dir`, `text_prompt`
  - 레퍼런스: `--ref_ply` 또는 `--ref_frames`(둘 중 하나 필수)
- 사용 모델
  - GroundingDINO + SAM2
  - 기하 최적화: SOR + PCA 초기화 + ICP
- 결과
  - `pose_<object>_frameXXXXXX.json` (cam0 + Isaac 좌표계 모두 저장)
  - `pose_3d_frameXXXXXX.png`, `detected_cam*_frame*.jpg`, `object_*.ply`
  - `--ref_frames` 사용 시 `self_ref_*.ply` 자동 생성 가능
- 회전 원리(핵심)
  - SAM2 마스크로 얻은 멀티카메라 객체 점군을 cam0으로 융합 후 정규화합니다.
  - 레퍼런스 점군도 정규화하여 PCA 주축 정렬을 초기값으로 생성합니다.
  - PCA 축 부호 4조합 각각에 대해 ICP를 실행하고, normalized RMSE 최소 조합을 채택합니다.
  - 선택된 회전행렬을 cam0 포즈로 저장하고, 추가로 Isaac(USD) 좌표계로 변환 저장합니다.

#### 5) pose_from_sam3d (최최종, 칼 특화 고속)

- 입력 데이터
  - Mode B: `--scene_ply` (이미 추출된 물체 점군)
  - Mode A: RGB-D + 캘리브레이션 + `text_prompt` (내부적으로 GDino+SAM2 재검출)
- 사용 모델
  - Mode B: 모델 없음 (기하 기반)
  - Mode A: GroundingDINO + SAM2 (점군 생성 단계에서만 사용)
- 결과
  - `pose_cam0_*.json`, `pose_cam0_*.png`
- 회전 원리
  - 객체 점군 PCA에서 `length axis(최대 고유값)`, `normal axis(최소 고유값)`를 얻습니다.
  - normal은 카메라 기준 위쪽 방향으로 부호를 고정합니다.
  - blade 방향은 색상 기반 자동 판별(또는 `--blade_dir pos/neg` 수동)로 180도 모호성을 해소합니다.
  - `width axis = cross(normal, length)`로 오른손 좌표계를 완성해 회전행렬을 구성합니다.

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

- 5개 포즈 추정 방법(Step5/6/7 + `pose_estimate_grounding_sam(최종)` + `pose_estimate_sam3d_knife(최최종)`)으로 정리
- 각 스크립트 기본 경로를 `../` 기준으로 통일 (`intrinsics`, `checkpoints`, `out_dir`)
- `pose_step6_ply/visualize_pose.py` 하드코딩 경로를 `os.path` 기반 상대경로로 변경
- `estimate_object_pose.py`에 `--ref_ply` / `--ref_frames` 상호배타 입력 지원
- `pose_from_sam3d.py`에 `scene_ply` 직접 입력(Mode B) + 재검출(Mode A) 이중 모드 지원
- 포즈 추정 스크립트 Python 문법 검증 완료

### src3 실행 방법

- 공통: `cd src3` 후 각 폴더에서 실행
- 입력 데이터 기본 경로: `../data/...`, 내부파라미터: `../intrinsics`, 모델 체크포인트: `../checkpoints`

```bash
# Step5: 위치 추정
cd pose_step5_localize
python Step5_localize_object_3d.py \
  --capture_dir ../data/rgbd_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "tiger figure."

# Step6: PLY 기반 ICP
cd ../pose_step6_ply
python Step6_estimate_pose_from_ply.py \
  --object_ply ../data/3d_ply/tiger.ply \
  --scene_ply ../data/3d_ply/point_cloud_cleaned.ply \
  --mesh_ply "../data/3d_ply/tiger_figure_custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask0_occ1_scale1_0_voxel2_512_trunc4_20_cleaned_mesh.ply" \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --intrinsics_dir ../intrinsics \
  --rgbd_dir ../data/rgbd_capture

# Step7: RGB-D 직접 포즈 (ICP)
cd ../pose_step7_direct
python Step7_direct_pose_from_rgbd.py \
  --rgbd_dir ../data/rgbd_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --ref_ply ../data/3d_ply/tiger.ply \
  --frame 0

# 최종 6DoF: ref_ply 기반
cd ../pose_estimate_grounding_sam\(최종\)
python estimate_object_pose.py \
  --capture_dir ../data/rgbd_capture \
  --calib_dir ../data/cube_session_01/calib_out_cube \
  --text_prompt "tiger figure." \
  --ref_ply ../data/3d_ply/tiger.ply \
  --device mps

# 고속 최종(칼): scene PLY 직접
cd ../pose_estimate_sam3d_knife\(최최종\)
python pose_from_sam3d.py \
  --scene_ply ../pose_estimate_grounding_sam\(최종\)/output_selfref/object_utility_knife_frame000005.ply
```
