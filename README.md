# ArucoCube Multi-Camera Calibration

RealSense 다중 카메라로 ArUco 큐브를 사용해
1) intrinsics 저장,
2) 동기 캡처,
3) 카메라 간 외부파라미터(상대변환) 추정,
4) depth point cloud fusion
를 수행하는 파이프라인입니다.

## 폴더 구성

- `conda env/environment.yml`: conda 환경 파일
- `src/Step1_dump_intrinsics.py`: 카메라별 intrinsics + device_map 저장
- `src/Step2_capture_multi_cam.py`: 멀티카메라 캡처 (`meta.json`, RGB/Depth 저장)
- `src/Step3_calibrate_multi_cam_cube.py`: 큐브 기반 캘리브레이션 (`T_Cref_Ci`)
- `src/Step4_fuse_depth_to_ref_pcd.py`: ref 카메라 좌표계로 depth fusion

## 사전 준비

- RealSense 카메라 연결 (USB bandwidth 충분히 확보)
- `librealsense`/드라이버 설치
- Python 환경 준비 (`pyrealsense2`, OpenCV, Open3D)

## 환경 생성 (Conda)

프로젝트 루트(`ArucoCube_multi_calibration`)에서:

```bash
conda env create -f "conda env/environment.yml"
conda activate multicam_cube
```

참고:
- 환경에 따라 `opencv` 패키지에 `cv2.aruco`가 없을 수 있습니다.
- 그 경우 `opencv-contrib-python` 계열 설치가 필요할 수 있습니다.

## 표준 실행 순서 (Step1 → Step4)

아래 명령은 `ArucoCube_multi_calibration/src`에서 실행하는 기준입니다.

```bash
cd ArucoCube_multi_calibration/src
```

### Step1. Intrinsics / device_map 생성

```bash
python Step1_dump_intrinsics.py
```

생성물:
- `intrinsics/device_map.json`
- `intrinsics/cam0.npz`, `intrinsics/cam1.npz`, ...
- `intrinsics/intrinsics_by_serial/serial_<SERIAL>.npz`
- `intrinsics/depth_scales.json`

권장:
- 카메라 연결 구성이 바뀌면 Step1을 다시 실행해서 `device_map.json`을 갱신하세요.

### Step2. 멀티카메라 캡처 (Step4 예정이면 depth 저장 필수)

```bash
python Step2_capture_multi_cam.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --fps 15 --width 640 --height 480 \
  --min_markers 2 \
  --auto_save --stable_frames 3 --cooldown_ms 700 \
  --save_depth \
  --show
```

조작:
- `SPACE`: 수동 저장
- `ESC` / `q`: 종료

생성물:
- `data/cube_session_01/meta.json`
- `data/cube_session_01/camX/rgb_XXXXX.jpg`
- `data/cube_session_01/camX/depth_XXXXX.png` (`--save_depth` 사용 시)

주의:
- `device_map.json` 기준으로 매핑 가능한 카메라가 0대면 스크립트가 종료됩니다 (안전장치).

### Step3. 큐브 기반 멀티카메라 캘리브레이션

예시 (ref 카메라를 `cam2`로 설정):

```bash
python Step3_calibrate_multi_cam_cube.py \
  --root_folder ./data/cube_session_01 \
  --intrinsics_dir ./intrinsics \
  --ref_cam_idx 2 \
  --min_markers 1 \
  --reproj_max_px 16 \
  --save_overlay \
  --overlay_max_per_cam 30
```

핵심 출력:
- `data/cube_session_01/calib_out_cube/T_C2_C0.npy`, `T_C2_C1.npy`, ...
- `data/cube_session_01/calib_out_cube/transforms/T_C2_Ci_all.json`
- `data/cube_session_01/calib_out_cube/calib_results/cam0_reproj.csv`, ...
- `data/cube_session_01/calib_out_cube/overlay/...` (`--save_overlay` 사용 시)

변환 의미:
- `T_Cref_Ci`: `cam i` 좌표계의 점을 `ref cam` 좌표계로 변환하는 4x4 변환행렬

### Step4. Depth Point Cloud Fusion (ref 카메라 좌표계)

```bash
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

생성물:
- `data/cube_session_01/fused_ref2_frame00000.ply` (`--save_ply` 사용 시)

참고:
- `device_map.json`이 없어도 `cam*.npz`, `T_Cref_Ci.npy`, depth/rgb 데이터가 있으면 실행 가능하도록 되어 있습니다.

## 빠른 점검 체크리스트

- Step1 후 `intrinsics/cam*.npz` 파일이 카메라 수만큼 생성되었는지
- Step2 후 `meta.json`의 `captures`가 증가하는지
- Step2에서 Step4를 할 예정이면 `depth_*.png`가 저장되는지 (`--save_depth`)
- Step3 후 `calib_out_cube/T_C{ref}_C{i}.npy`가 생성되는지
- Step3 `calib_results/cam*_reproj.csv`에서 `err_mean` 값이 과도하게 크지 않은지
- Step4에서 ICP 평가(`--eval_icp`)의 `rmse`/`fitness`가 일관적인지

## 자주 발생하는 문제

- `No RealSense devices found`
  - 케이블/전원/USB 포트 확인
  - 다른 프로세스가 카메라 점유 중인지 확인

- `device_map.json` 관련 경고/종료
  - Step1 재실행 후 Step2 재시도
  - 카메라 serial 변경/누락 여부 확인

- `cv2.aruco` 없음
  - OpenCV 설치 구성 확인 (contrib 포함 여부)

- Step3에서 유효 프레임 부족
  - `--min_markers` 낮추기
  - `--reproj_max_px` 완화
  - 큐브 가시성/조명/초점 개선
