# 객체 포즈 추정

## 디렉터리 구조

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
├── Step1~4*.py                       ← 캘리브레이션 인프라 (유지)
├── data/                             ← 입력 데이터만
├── intrinsics/
└── checkpoints/
```

## 변경 사항

- 4개 포즈 추정 방법별로 스크립트와 결과를 각각 하나의 폴더로 정리
- 각 스크립트의 기본 경로를 `../` 기준으로 수정 (`intrinsics`, `checkpoints`, `out_dir`)
- `visualize_pose.py`의 하드코딩 경로를 `os.path` 기반 상대경로로 변경
- 모든 스크립트 Python 문법 검증 통과

## 실행 방법

- 각 포즈 추정 폴더로 이동해서 실행
- 입력 데이터는 `../data/...` 경로를 참조

예시:

```bash
cd pose_step5_localize/
python Step5_localize_object_3d.py
```
