# camera.py 
import threading
import time
from typing import Optional, Tuple, Dict

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        use_color: bool = True,
        use_depth: bool = False,
        align_depth_to_color: bool = True,
        warmup_frames: int = 10,
    ):
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.use_color = bool(use_color)
        self.use_depth = bool(use_depth)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.warmup_frames = int(warmup_frames)

        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial)

        if self.use_color:
            self.cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.use_depth:
            self.cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.align = rs.align(rs.stream.color) if (self.use_depth and self.align_depth_to_color and self.use_color) else None

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._color = None
        self._depth = None
        self._ts_ms = None

    @staticmethod
    def list_devices() -> Dict[str, str]:
        ctx = rs.context()
        out = {}
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            out[serial] = name
        return out

    def start(self):
        self.pipeline.start(self.cfg)
        time.sleep(5.0)  # depth+color 복합 스트림 초기화 여유

        arrived = 0
        for attempt in range(self.warmup_frames * 2):
            try:
                self.pipeline.wait_for_frames(timeout_ms=30000)
                arrived += 1
                if arrived >= self.warmup_frames:
                    break
            except Exception as e:
                print(f"[WARN] serial={self.serial} warmup {attempt+1}: {e}")

        if arrived == 0:
            raise RuntimeError(
                f"Camera serial={self.serial}: 프레임이 도착하지 않았습니다. "
                "카메라를 재연결 후 재시도하세요."
            )

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                if self.align is not None:
                    frames = self.align.process(frames)

                color = frames.get_color_frame() if self.use_color else None
                depth = frames.get_depth_frame() if self.use_depth else None

                if self.use_color and (color is None):
                    continue

                ts_ms = None
                if color is not None:
                    ts_ms = float(color.get_timestamp())
                elif depth is not None:
                    ts_ms = float(depth.get_timestamp())

                with self._lock:
                    if color is not None:
                        self._color = np.asanyarray(color.get_data()).copy()
                    if depth is not None:
                        self._depth = np.asanyarray(depth.get_data()).copy()
                    self._ts_ms = ts_ms

            except Exception:
                time.sleep(0.005)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        with self._lock:
            c = None if self._color is None else self._color.copy()
            d = None if self._depth is None else self._depth.copy()
            ts = self._ts_ms
        return c, d, ts
