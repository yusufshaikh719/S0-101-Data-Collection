"""Camera acceptance checker: driver warmup must not look like dropped frames."""

from __future__ import annotations

import time

import numpy as np

from so101_collector.core.camera_hub import CameraHubStats
from so101_collector.exceptions import CameraError
from so101_collector.scripts.check_cameras import _check_sync, _measure


class _SlowStartHub:
    """Camera hub double whose drivers need a while before frames arrive."""

    names = ("base", "wrist")

    def __init__(self, delay_s: float) -> None:
        self._ready_at = time.monotonic() + delay_s
        self._frames = {name: 0 for name in self.names}

    def _ready(self) -> bool:
        return time.monotonic() >= self._ready_at

    def get_frame(self, name: str) -> object:
        if not self._ready():
            raise CameraError(f"camera '{name}' has not produced a frame yet")
        return object()

    def get_synced_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        if not self._ready():
            raise CameraError("camera 'base' has not produced a frame yet")
        self._frames = {name: count + 1 for name, count in self._frames.items()}
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        return frame, frame, 0.004

    def stats(self) -> CameraHubStats:
        return CameraHubStats(
            cameras={
                name: {"frames": count, "fps": 60.0, "error": None, "alive": True}
                for name, count in self._frames.items()
            },
            sync_samples=0,
            sync_mean_ms=0.0,
            sync_max_ms=0.0,
            sync_warnings=0,
        )


def test_measure_excludes_driver_warmup() -> None:
    hub = _SlowStartHub(delay_s=0.3)

    windows, sync_samples = _measure(
        hub, duration_s=2.2, window_s=1.0, fps=60.0, snapshot_dir=None
    )

    assert sync_samples, "expected sync samples once the cameras stream"
    assert windows["base"], "expected at least one FPS window"
    # The startup stall must not be reported as a sub-30 FPS window.
    assert min(windows["base"]) >= 50.0
    assert min(windows["wrist"]) >= 50.0


def test_check_sync_can_judge_mean_or_worst_case() -> None:
    # One late pair among otherwise tight frames: sustained sync is fine, the
    # worst case is not.
    samples = [0.002] * 99 + [0.030]

    assert _check_sync(samples, 15.0, "mean") is True
    assert _check_sync(samples, 15.0, "max") is False
