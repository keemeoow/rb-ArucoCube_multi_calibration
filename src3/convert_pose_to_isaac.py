# convert_pose_to_isaac.py
# ──────────────────────────────────────────────────────────────
# Step7 포즈 결과 → Isaac Lab (Isaac Sim) 좌표계 변환
#
# ★ 좌표계 차이:
#
#   cam0 (OpenCV)         Isaac Lab (USD/PhysX)
#   X → right             X → right
#   Y → down              Y → forward
#   Z → forward           Z → up
#
#   변환 행렬 T_isaac_cam0:
#     X_isaac =  X_cam0
#     Y_isaac =  Z_cam0     (cam0의 앞 = isaac의 앞)
#     Z_isaac = -Y_cam0     (cam0의 아래 = isaac의 위 반전)
#
# ★ 단위:
#   cam0 출력: meters → Isaac Lab: meters (동일)
#
# ★ 회전:
#   Isaac Lab은 quaternion (w, x, y, z) 사용
#   R_isaac = T_conv @ R_cam0 @ T_conv^T
#
# 사용법:
#   python convert_pose_to_isaac.py --json ./data/rgbd_capture/direct_pose_out/pose_frame000000.json
#   python convert_pose_to_isaac.py --json ./data/3d_ply/pose_out/pose_estimation_results.json --key multicam_pose
# ──────────────────────────────────────────────────────────────

import json
import argparse
import numpy as np


# ================================================================
#  좌표계 변환 행렬
# ================================================================

# cam0 (OpenCV) → Isaac Lab (USD, Z-up)
#   X_isaac =  X_cam0
#   Y_isaac =  Z_cam0
#   Z_isaac = -Y_cam0
T_ISAAC_CAM0 = np.array([
    [ 1,  0,  0],
    [ 0,  0,  1],
    [ 0, -1,  0],
], dtype=np.float64)


def convert_position(pos_cam0: np.ndarray) -> np.ndarray:
    """cam0 위치 → Isaac Lab 위치 (meters)."""
    return T_ISAAC_CAM0 @ pos_cam0


def convert_rotation(R_cam0: np.ndarray) -> np.ndarray:
    """cam0 회전행렬 → Isaac Lab 회전행렬."""
    return T_ISAAC_CAM0 @ R_cam0 @ T_ISAAC_CAM0.T


def rotation_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """회전행렬 → quaternion (w, x, y, z)."""
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotation_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """회전행렬 → Euler XYZ (degrees)."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z]))


# ================================================================
#  변환 + 출력
# ================================================================

def convert_pose(pose_data: dict) -> dict:
    """Step7 또는 Step6 포즈 → Isaac Lab 포즈 변환."""
    # position
    pos_cam0 = np.array(pose_data["position_m"])
    pos_isaac = convert_position(pos_cam0)

    # rotation
    R_cam0 = np.array(pose_data["rotation_matrix"])
    R_isaac = convert_rotation(R_cam0)
    quat_isaac = rotation_to_quat_wxyz(R_isaac)
    euler_isaac = rotation_to_euler_xyz(R_isaac)

    # OBB (크기는 좌표계 불변, 순서만 재배열)
    obb_cam0 = np.array(pose_data.get("obb_extents_m", pose_data.get("obb_extents", [0,0,0])))
    # cam0의 PCA 주축 순서와 isaac 주축 순서가 다를 수 있으므로
    # 변환된 R의 열 방향으로 재계산하는 것이 정확하지만,
    # 크기 자체는 정렬 순서만 바뀜 (값은 동일)
    obb_isaac = obb_cam0  # 크기값 자체는 좌표계 무관

    return {
        "position": pos_isaac.tolist(),
        "quaternion_wxyz": quat_isaac.tolist(),
        "euler_xyz_deg": euler_isaac.tolist(),
        "rotation_matrix": R_isaac.tolist(),
        "obb_extents_m": obb_isaac.tolist(),
    }


def print_isaac_pose(isaac_pose: dict, label: str = "Object"):
    """Isaac Lab용 포즈를 보기 좋게 출력."""
    pos = isaac_pose["position"]
    quat = isaac_pose["quaternion_wxyz"]
    euler = isaac_pose["euler_xyz_deg"]
    obb = isaac_pose["obb_extents_m"]

    print(f"\n{'='*60}")
    print(f"  Isaac Lab Pose: {label}")
    print(f"{'='*60}")
    print(f"  Position (m):     [{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]")
    print(f"  Position (mm):    [{pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}]")
    print(f"  Quaternion wxyz:  [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]")
    print(f"  Euler XYZ (deg):  [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
    print(f"  OBB size (m):     [{obb[0]:.4f}, {obb[1]:.4f}, {obb[2]:.4f}]")
    print(f"{'='*60}")

    # Isaac Lab 코드 스니펫
    print(f"\n  # ── Isaac Lab Python snippet ──")
    print(f"  import torch")
    print(f"  pos  = torch.tensor([[{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}]])")
    print(f"  quat = torch.tensor([[{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]])")
    print(f"  object.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))")

    print(f"\n  # ── Isaac Sim (omni.isaac) snippet ──")
    print(f'  from pxr import Gf')
    print(f'  prim = stage.GetPrimAtPath("/World/tiger")')
    print(f'  prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}))')
    print(f'  prim.GetAttribute("xformOp:orient").Set(Gf.Quatd({quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}))')
    print()


def main():
    parser = argparse.ArgumentParser(description="Pose → Isaac Lab coordinate conversion")
    parser.add_argument("--json", required=True, help="Step7 or Step6 JSON result file")
    parser.add_argument("--key", default=None,
                        help="JSON key for pose (e.g. 'multicam_pose' for Step6, None for Step7)")
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    # Step7 format: data["pose"]
    # Step6 format: data["multicam_pose"]
    if args.key:
        pose_data = data[args.key]
    elif "pose" in data:
        pose_data = data["pose"]
    elif "multicam_pose" in data:
        pose_data = data["multicam_pose"]
    else:
        print("[ERROR] Cannot find pose data in JSON")
        return

    # cam0 원본 출력
    pos_cam0 = np.array(pose_data["position_m"])
    euler_cam0 = np.array(pose_data["euler_xyz_deg"])
    quat_cam0 = np.array(pose_data["quaternion_wxyz"])

    print(f"\n{'='*60}")
    print(f"  Original (cam0 frame)")
    print(f"{'='*60}")
    print(f"  Position (m):     [{pos_cam0[0]:.6f}, {pos_cam0[1]:.6f}, {pos_cam0[2]:.6f}]")
    print(f"  Euler XYZ (deg):  [{euler_cam0[0]:.2f}, {euler_cam0[1]:.2f}, {euler_cam0[2]:.2f}]")
    print(f"  Quat wxyz:        [{quat_cam0[0]:.6f}, {quat_cam0[1]:.6f}, {quat_cam0[2]:.6f}, {quat_cam0[3]:.6f}]")

    # 변환
    isaac_pose = convert_pose(pose_data)
    print_isaac_pose(isaac_pose, "tiger figure")

    # JSON 저장
    out_path = args.json.replace(".json", "_isaac.json")
    with open(out_path, "w") as f:
        json.dump({
            "source": args.json,
            "source_frame": "cam0 (OpenCV: X-right, Y-down, Z-forward)",
            "target_frame": "Isaac Lab (USD: X-right, Y-forward, Z-up)",
            "cam0_pose": {
                "position_m": pos_cam0.tolist(),
                "euler_xyz_deg": euler_cam0.tolist(),
                "quaternion_wxyz": quat_cam0.tolist(),
            },
            "isaac_pose": isaac_pose,
        }, f, indent=2)
    print(f"  [SAVE] {out_path}")


if __name__ == "__main__":
    main()
