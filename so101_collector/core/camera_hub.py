"""Multi-threaded dual-camera acquisition with software frame synchronization.

Each camera runs in its own daemon thread (``BaseCameraThread`` /
``WristCameraThread``) that continuously overwrites a double-buffered
:class:`FramePacket` under a lock, so the 30 Hz control loop never blocks on
USB/V4L2 I/O and a stalled control loop never causes frame drops.

The control loop pulls synchronized frames through
:meth:`CameraHub.get_synced_frames`, which returns the latest base + wrist
frame together with the timestamp delta ``dt_sync = |t_base - t_wrist|`` and
warns when it exceeds the configured tolerance (20 ms by default).

Device addressing: cameras are opened via their persistent ``/dev/v4l/by-id``
paths (validated at config load), never dynamic ``/dev/videoN`` nodes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np

from so101_collector.config import CameraConfig, VisionConfig
from so101_collector.exceptions import CameraError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramePacket:
    """One frame plus its capture timestamp (monotonic seconds)."""

    frame: np.ndarray
    captured_at: float
    sequence: int
    camera: str


class CameraSource(Protocol):
    """Structural type for a frame source (V4L2, simulated, ...)."""

    def open(self) -> None: ...

    def read(self) -> tuple[bool, np.ndarray | None, float]: ...

    def close(self) -> None: ...


def coerce_frame(
    frame: np.ndarray, *, camera: str, expected_shape: tuple[int, int]
) -> np.ndarray:
    """Normalize a driver frame to HxWx3 uint8, validating the resolution."""
    if frame is None:
        raise CameraError(f"camera '{camera}' returned an empty frame")
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise CameraError(f"camera '{camera}' returned unsupported frame shape {frame.shape}")
    if frame.shape[:2] != expected_shape:
        raise CameraError(
            f"camera '{camera}' returned {frame.shape[1]}x{frame.shape[0]}, "
            f"expected {expected_shape[1]}x{expected_shape[0]}"
        )
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


class V4L2Camera:
    """OpenCV/V4L2 camera configured for low-latency high-speed streaming."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture = None
        self._cv2 = None

    def open(self) -> None:
        if self._capture is not None:
            return
        path = self._config.device_path
        if path.startswith("/dev/") and not Path(path).exists():
            raise CameraError(
                f"camera '{self._config.name}': device {path} does not exist "
                "(is the camera plugged in? check ls -l /dev/v4l/by-id/)"
            )
        try:
            import cv2  # noqa: PLC0415 - lazy so simulation/tests need no OpenCV
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise CameraError(
                "opencv-python is required for camera capture (pip install opencv-python)"
            ) from exc
        self._cv2 = cv2

        backend = cv2.CAP_V4L2 if path.startswith("/dev/") else cv2.CAP_ANY
        capture = cv2.VideoCapture(path, backend)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"camera '{self._config.name}': failed to open {path}")

        # MJPG fourcc is essential: the default YUYV cannot sustain
        # 640x480 @ 30 FPS and would silently drop to ~15 FPS.
        fourcc = cv2.VideoWriter_fourcc(*self._config.fourcc)
        settings = (
            ("CAP_PROP_FOURCC", fourcc),
            ("CAP_PROP_FRAME_WIDTH", self._config.width),
            ("CAP_PROP_FRAME_HEIGHT", self._config.height),
            ("CAP_PROP_FPS", self._config.fps),
            # Keep the driver queue minimal so reads stay fresh.
            ("CAP_PROP_BUFFERSIZE", self._config.buffer_size),
        )
        for prop_name, value in settings:
            prop_id = getattr(cv2, prop_name)
            if not capture.set(prop_id, value):
                logger.warning(
                    "camera '%s': driver rejected %s=%s", self._config.name, prop_name, value
                )

        if self._config.exposure is not None:
            # Lock exposure to stop auto-exposure from causing frame-rate
            # jitter under changing ambient lighting.
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)  # 1 = manual (V4L2)
            if not capture.set(cv2.CAP_PROP_EXPOSURE, float(self._config.exposure)):
                logger.warning(
                    "camera '%s': driver rejected exposure=%s",
                    self._config.name,
                    self._config.exposure,
                )
            logger.info("camera '%s': exposure locked at %s", self._config.name, self._config.exposure)

        actual = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
        )
        logger.info(
            "camera '%s': opened %s (%dx%d @ %.1f FPS requested, %dx%d @ %.1f reported)",
            self._config.name,
            path,
            self._config.width,
            self._config.height,
            self._config.fps,
            actual[0],
            actual[1],
            actual[2],
        )
        self._capture = capture

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        capture = self._capture
        if capture is None:
            raise CameraError(f"camera '{self._config.name}' is not open")
        ok, frame = capture.read()
        if ok and frame is not None:
            # Pipeline-wide frame convention is RGB; the OpenCV driver hands
            # us BGR, so convert at this boundary (the only cv2 touchpoint).
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return bool(ok), frame if ok else None, time.monotonic()

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()


class CameraWorker(threading.Thread):
    """Background acquisition loop with double-buffered frame swap."""

    def __init__(
        self,
        config: CameraConfig,
        source: CameraSource,
        *,
        max_read_failures: int = 90,
        max_reopen_attempts: int = 3,
        idle_sleep_s: float = 0.0005,
    ) -> None:
        super().__init__(name=f"{config.name.capitalize()}CameraThread", daemon=True)
        self._config = config
        self._source = source
        self._max_read_failures = max_read_failures
        self._max_reopen_attempts = max_reopen_attempts
        self._idle_sleep_s = idle_sleep_s
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._sequence = 0
        self._frames = 0
        self._fps = 0.0
        self._last_capture_at: float | None = None
        self._error: CameraError | None = None

    # -- worker loop -------------------------------------------------------
    def run(self) -> None:
        failures = 0
        try:
            try:
                self._source.open()
            except CameraError as exc:
                # Opening can fail (unplugged device): go through the same
                # recovery path as read errors instead of dying silently.
                if not self._recover(exc):
                    return
            while not self._stop_event.is_set():
                try:
                    ok, frame, captured_at = self._source.read()
                except CameraError as exc:
                    if not self._recover(exc):
                        return
                    continue
                if not ok or frame is None:
                    failures += 1
                    if failures >= self._max_read_failures:
                        self._fail(
                            CameraError(
                                f"camera '{self._config.name}': {failures} consecutive read failures"
                            )
                        )
                        return
                    if failures % 30 == 1:
                        logger.warning("camera '%s': read failure %d", self._config.name, failures)
                    time.sleep(self._idle_sleep_s)
                    continue
                failures = 0
                try:
                    frame = coerce_frame(
                        frame, camera=self._config.name, expected_shape=self._config.frame_shape
                    )
                except CameraError as exc:
                    self._fail(exc)
                    return
                self._publish(frame, captured_at)
        finally:
            self._source.close()

    def _publish(self, frame: np.ndarray, captured_at: float) -> None:
        with self._lock:
            self._sequence += 1
            self._latest = FramePacket(
                frame=frame, captured_at=captured_at, sequence=self._sequence, camera=self._config.name
            )
            self._frames += 1
            if self._last_capture_at is not None:
                interval = captured_at - self._last_capture_at
                if interval > 0:
                    instant_fps = 1.0 / interval
                    self._fps = instant_fps if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instant_fps
            self._last_capture_at = captured_at

    def _recover(self, exc: CameraError) -> bool:
        """Try to reopen a failed source; return False when to give up."""
        logger.warning("camera '%s': source error: %s", self._config.name, exc)
        for attempt in range(1, self._max_reopen_attempts + 1):
            if self._stop_event.is_set():
                return False
            time.sleep(0.1)
            try:
                self._source.close()
                self._source.open()
                logger.info(
                    "camera '%s': reopened after %d attempt(s)", self._config.name, attempt
                )
                return True
            except CameraError as reopen_exc:
                logger.warning(
                    "camera '%s': reopen attempt %d/%d failed: %s",
                    self._config.name,
                    attempt,
                    self._max_reopen_attempts,
                    reopen_exc,
                )
        self._fail(exc)
        return False

    def _fail(self, exc: CameraError) -> None:
        logger.error("camera '%s': worker stopped: %s", self._config.name, exc)
        with self._lock:
            self._error = exc

    # -- control / inspection ----------------------------------------------
    @property
    def latest(self) -> FramePacket | None:
        with self._lock:
            return self._latest

    @property
    def error(self) -> CameraError | None:
        with self._lock:
            return self._error

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frames

    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop(self, timeout_s: float = 2.0) -> None:
        self.request_stop()
        if self.is_alive():
            self.join(timeout=timeout_s)
            if self.is_alive():
                logger.warning("camera '%s': thread did not stop within %.1fs",
                               self._config.name, timeout_s)


@dataclass(frozen=True)
class CameraHubStats:
    """Aggregated hub statistics for monitoring and acceptance checks."""

    cameras: dict[str, dict[str, float | int | str | None]]
    sync_samples: int
    sync_mean_ms: float
    sync_max_ms: float
    sync_warnings: int


class CameraHub:
    """Owns both camera workers and hands out software-synchronized frames."""

    def __init__(
        self,
        vision: VisionConfig | Iterable[CameraConfig],
        *,
        sync_tolerance_ms: float | None = None,
        source_factory: Callable[[CameraConfig], CameraSource] = V4L2Camera,
        on_sync_warning: Callable[[float], None] | None = None,
    ) -> None:
        if isinstance(vision, VisionConfig):
            configs = vision.cameras
            tolerance = vision.sync_tolerance_ms if sync_tolerance_ms is None else sync_tolerance_ms
        else:
            configs = tuple(vision)
            tolerance = 20.0 if sync_tolerance_ms is None else sync_tolerance_ms
        if tolerance <= 0:
            raise ValueError(f"sync_tolerance_ms must be > 0, got {tolerance}")
        self._configs = {c.name: c for c in configs}
        self._tolerance_s = tolerance / 1000.0
        self._source_factory = source_factory
        self._on_sync_warning = on_sync_warning
        self._workers: dict[str, CameraWorker] = {}
        self._sync_samples = 0
        self._sync_sum_s = 0.0
        self._sync_max_s = 0.0
        self._sync_warnings = 0

    # -- lifecycle ---------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._configs)

    @property
    def running(self) -> bool:
        return bool(self._workers) and any(w.is_alive() for w in self._workers.values())

    def start(self) -> CameraHub:
        if self._workers:
            return self
        for name, config in self._configs.items():
            worker = CameraWorker(config, self._source_factory(config))
            worker.start()
            self._workers[name] = worker
        logger.info("camera hub started: %s", ", ".join(self._configs))
        return self

    def stop(self) -> None:
        workers, self._workers = self._workers, {}
        for worker in workers.values():
            worker.stop()

    def __enter__(self) -> CameraHub:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- frame access ------------------------------------------------------
    def _worker(self, name: str) -> CameraWorker:
        worker = self._workers.get(name)
        if worker is None:
            raise CameraError(f"camera '{name}' is not managed by this hub (have: {list(self._workers)})")
        return worker

    def get_frame(self, name: str) -> FramePacket:
        worker = self._worker(name)
        if worker.error is not None:
            raise CameraError(f"camera '{name}' worker failed: {worker.error}")
        packet = worker.latest
        if packet is None:
            raise CameraError(f"camera '{name}' has not produced a frame yet")
        return packet

    def get_synced_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``(base_frame, wrist_frame, dt_sync)`` with ``dt_sync`` in seconds.

        Logs (and forwards to ``on_sync_warning``) whenever the timestamp
        delta exceeds the configured tolerance (default 20 ms).
        """
        base = self.get_frame("base")
        wrist = self.get_frame("wrist")
        dt_sync = abs(base.captured_at - wrist.captured_at)

        self._sync_samples += 1
        self._sync_sum_s += dt_sync
        self._sync_max_s = max(self._sync_max_s, dt_sync)
        if dt_sync > self._tolerance_s:
            self._sync_warnings += 1
            logger.warning(
                "frame synchronization drift %.2f ms exceeds %.2f ms tolerance "
                "(base seq=%d, wrist seq=%d)",
                dt_sync * 1000.0,
                self._tolerance_s * 1000.0,
                base.sequence,
                wrist.sequence,
            )
            if self._on_sync_warning is not None:
                self._on_sync_warning(dt_sync)
        return base.frame, wrist.frame, dt_sync

    # -- monitoring --------------------------------------------------------
    def stats(self) -> CameraHubStats:
        cameras: dict[str, dict[str, float | int | str | None]] = {}
        for name, worker in self._workers.items():
            cameras[name] = {
                "frames": worker.frame_count,
                "fps": round(worker.fps, 2),
                "error": str(worker.error) if worker.error else None,
                "alive": worker.is_alive(),
            }
        mean_ms = (self._sync_sum_s / self._sync_samples * 1000.0) if self._sync_samples else 0.0
        return CameraHubStats(
            cameras=cameras,
            sync_samples=self._sync_samples,
            sync_mean_ms=mean_ms,
            sync_max_ms=self._sync_max_s * 1000.0,
            sync_warnings=self._sync_warnings,
        )
