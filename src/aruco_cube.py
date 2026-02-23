import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# -------------------------
# utils
# -------------------------
def rodrigues_to_Rt(rvec, tvec) -> np.ndarray:
    """OpenCV rvec,tvec (Obj->Cam) => 4x4 T_C_O"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of 4x4 rigid transform"""
    R = T[:3, :3]
    t = T[:3, 3:4]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t
    return Ti

def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues formula from axis-angle to R"""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

# -------------------------
# Cube config
# -------------------------
@dataclass
class CubeConfig:
    cube_side_m: float = 0.06       # cube size
    marker_size_m: float = 0.05     # marker size
    dictionary_name: str = "DICT_4X4_50"
    marker_ids: tuple = (0, 1, 2, 3, 4)

    id_to_face: dict = None
    face_roll_deg: dict = None

    def __post_init__(self):
        if self.id_to_face is None:
            self.id_to_face = {
                0: "+Y",
                1: "+Z",
                2: "+X",
                3: "-Z",
                4: "-X",
            }
        if self.face_roll_deg is None:
            self.face_roll_deg = {int(i): 0.0 for i in self.marker_ids}

# -------------------------
# Cube geometry
# -------------------------
class ArucoCubeModel:
    def __init__(self, cfg: CubeConfig):
        self.cfg = cfg
        d = cfg.cube_side_m / 2.0
        s = cfg.marker_size_m / 2.0

        # Each face: center c, axes u,v, normal n (in rig/object coord)
        self.face_defs = {
            "+Z": (np.array([0, 0,  d], np.float64),
                   np.array([1, 0, 0], np.float64),   # u = +X
                   np.array([0, 1, 0], np.float64),   # v = +Y
                   np.array([0, 0, 1], np.float64)),  # n = +Z

            "-Z": (np.array([0, 0, -d], np.float64),
                   np.array([1, 0, 0], np.float64),   # u = +X
                   np.array([0, 1, 0], np.float64),   # v = +Y
                   np.array([0, 0,-1], np.float64)),  # n = -Z

            "+X": (np.array([ d, 0, 0], np.float64),
                   np.array([0, 0,-1], np.float64),   # u = -Z
                   np.array([0, 1, 0], np.float64),   # v = +Y
                   np.array([1, 0, 0], np.float64)),  # n = +X

            "-X": (np.array([-d, 0, 0], np.float64),
                   np.array([0, 0, 1], np.float64),   # u = +Z
                   np.array([0, 1, 0], np.float64),   # v = +Y
                   np.array([-1,0, 0], np.float64)),  # n = -X

            "+Y": (np.array([0, d, 0], np.float64),
                   np.array([1, 0, 0], np.float64),   # u = +X
                   np.array([0, 0,-1], np.float64),   # v = -Z
                   np.array([0, 1, 0], np.float64)),  # n = +Y
        }

        # Marker local corners on marker plane (z=0 in local marker frame)
        self.local_corners = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0],
        ], dtype=np.float64)

    def marker_corners_in_rig(self, marker_id: int) -> np.ndarray:
        marker_id = int(marker_id)
        face = self.cfg.id_to_face[marker_id]
        c, u, v, n = self.face_defs[face]

        roll_deg = float(self.cfg.face_roll_deg.get(marker_id, 0.0))
        roll = np.deg2rad(roll_deg)
        Rr = rot_axis_angle(n, roll)

        u2 = (Rr @ u.reshape(3, 1)).reshape(3)
        v2 = (Rr @ v.reshape(3, 1)).reshape(3)

        pts = []
        for p in self.local_corners:
            pts.append(c + u2 * p[0] + v2 * p[1])
        return np.asarray(pts, dtype=np.float64)

# -------------------------
# Target
# -------------------------
class ArucoCubeTarget:
    def __init__(self, cfg: CubeConfig):
        self.cfg = cfg
        self.model = ArucoCubeModel(cfg)

        d = getattr(cv2.aruco, cfg.dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(d)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, bgr) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Return (corners_list, ids_flat or None)"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return [], None
        return corners, ids.flatten().astype(int)

    # 공개 wrapper (Step3 overlay, 디버그에서 쓰기 위함)
    def build_correspondences(
        self,
        corners_list,
        ids,
        min_markers: int,
        only_ids: Optional[List[int]] = None
    ):
        return self._build_correspondences(corners_list, ids, min_markers, only_ids=only_ids)

    def _build_correspondences(self, corners_list, ids, min_markers, only_ids=None):
        obj_pts, img_pts, used = [], [], []
        only_ids = set(only_ids) if only_ids is not None else None

        for c, mid in zip(corners_list, ids):
            mid = int(mid)
            if mid not in self.cfg.id_to_face:
                continue
            if only_ids is not None and mid not in only_ids:
                continue

            obj = self.model.marker_corners_in_rig(mid)  # (4,3)
            img = c.reshape(4, 2)                        # (4,2)

            # NOTE: 네가 넣어둔 id==3 코너 reorder 유지
            if mid == 3:
                img = img[[1, 2, 3, 0]]

            obj_pts.append(obj)
            img_pts.append(img)
            used.append(mid)

        if len(used) < int(min_markers):
            return None, None, used

        obj_pts = np.concatenate(obj_pts).reshape(-1, 1, 3).astype(np.float64)
        img_pts = np.concatenate(img_pts).reshape(-1, 1, 2).astype(np.float64)
        return obj_pts, img_pts, used

    def _solve_and_score(self, obj_pts, img_pts, K, D, use_ransac: bool):
        n = int(obj_pts.shape[0])

        # if enough points -> ITERATIVE / else (square) IPPE_SQUARE
        flags = cv2.SOLVEPNP_ITERATIVE if n >= 8 else cv2.SOLVEPNP_IPPE_SQUARE

        if use_ransac and n >= 8:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                obj_pts, img_pts, K, D,
                flags=flags,
                reprojectionError=5.0,
                iterationsCount=200,
                confidence=0.999
            )
        else:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flags)

        if not ok:
            return None

        proj2, _ = cv2.projectPoints(obj_pts.reshape(-1, 3), rvec, tvec, K, D)
        proj2 = proj2.reshape(-1, 2)
        err = np.linalg.norm(proj2 - img_pts.reshape(-1, 2), axis=1).astype(np.float64)

        return dict(rvec=rvec, tvec=tvec, proj2=proj2, err=err)

    def solve_pnp_cube(
        self,
        bgr,
        K,
        D,
        use_ransac: bool = True,
        min_markers: int = 1,
        reproj_thr_mean_px: float = 10.0,
        only_ids: Optional[List[int]] = None,
        mean_err_max_px: Optional[float] = None,
        return_reproj: bool = False,
    ):
        """
        Returns:
          - if return_reproj False:
              (ok, rvec, tvec, used)
          - if return_reproj True:
              (ok, rvec, tvec, used, reproj_dict)
            reproj_dict contains: obj_pts,img_pts,proj2,err,err_mean,err_median,err_p90,n_points,rvec,tvec
        """
        corners_list, ids = self.detect(bgr)
        if ids is None:
            return (False, None, None, [], None) if return_reproj else (False, None, None, [])

        obj_pts, img_pts, used = self._build_correspondences(corners_list, ids, min_markers, only_ids=only_ids)
        if obj_pts is None:
            return (False, None, None, used, None) if return_reproj else (False, None, None, used)

        sol = self._solve_and_score(obj_pts, img_pts, K, D, use_ransac)
        if sol is None:
            return (False, None, None, used, None) if return_reproj else (False, None, None, used)

        err = sol["err"]
        reproj = {
            "obj_pts": obj_pts,
            "img_pts": img_pts,
            "proj2": sol["proj2"],
            "err": err,
            "err_mean": float(np.mean(err)) if err.size else float("inf"),
            "err_median": float(np.median(err)) if err.size else float("inf"),
            "err_p90": float(np.percentile(err, 90)) if err.size else float("inf"),
            "n_points": int(err.size),
            "rvec": sol["rvec"],
            "tvec": sol["tvec"],
        }

        max_thr = float("inf") if mean_err_max_px is None else float(mean_err_max_px)
        ok_final = (reproj["err_mean"] <= float(reproj_thr_mean_px)) and (reproj["err_mean"] <= max_thr)

        if return_reproj:
            return ok_final, sol["rvec"], sol["tvec"], used, reproj
        return ok_final, sol["rvec"], sol["tvec"], used