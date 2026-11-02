"""Synchronization tests: loop-timer drift compensation and camera sync.

The timer uses an injected fake clock/sleep pair so scheduling behaviour is
deterministic; the camera hub uses scripted sources with exact timestamps.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from so101_collector.core.camera_hub import CameraHub, coerce_frame
from so101_collector.exceptions import CameraError
from so101_collector.timing import LoopTimer
from tests.conftest import make_vision_config


def _messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


# ---------------------------------------------------------------------------
# LoopTimer
# ---------------------------------------------------------------------------
class FakeClock:
    """Deterministic monotonic clock driven entirely by sleep calls."""

    def __init__(self, start: float = 1000.0, overshoot_s: float = 0.0) -> None:
        self.now = start
        self.overshoot_s = overshoot_s
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0, "timer attempted to sleep a negative duration"
        self.sleeps.append(seconds)
        self.now += seconds + self.overshoot_s


def test_timer_runs_at_exact_target_rate():
    clock = FakeClock()
    timer = LoopTimer(30.0, clock=clock, sleep=clock.sleep).start()
    periods = [timer.wait().period_s for _ in range(90)]

    assert all(p == pytest.approx(1 / 30, abs=1e-9) for p in periods)
    summary = timer.summary()
    assert summary.ticks == 90
    assert summary.overruns == 0
    assert summary.mean_fps == pytest.approx(30.0, rel=1e-6)
    assert summary.max_jitter_s == pytest.approx(0.0, abs=1e-9)


def test_timer_reports_overshoot_as_overrun():
    # Every sleep overshoots by 2 ms -> every tick overruns by ~2 ms.
    clock = FakeClock(overshoot_s=0.002)
    timer = LoopTimer(30.0, clock=clock, sleep=clock.sleep).start()
    stats = [timer.wait() for _ in range(10)]

    assert all(s.overrun_s == pytest.approx(0.002, abs=1e-9) for s in stats)
    summary = timer.summary()
    assert summary.overruns == 10
    assert summary.max_overrun_s == pytest.approx(0.002, abs=1e-9)


def test_timer_resynchronizes_after_catastrophic_stall(caplog):
    clock = FakeClock()
    stall = {"armed": True}

    def sleepy(seconds: float) -> None:
        if stall["armed"]:
            stall["armed"] = False
            clock.now += 0.5  # 500 ms GC-pause style stall
        else:
            clock.sleep(seconds)

    timer = LoopTimer(30.0, clock=clock, sleep=sleepy).start()
    with caplog.at_level(logging.WARNING, logger="so101_collector.timing"):
        first = timer.wait()
        second = timer.wait()

    assert first.overrun_s > 0.4
    assert second.overrun_s == pytest.approx(0.0, abs=1e-9)  # schedule recovered
    assert second.period_s == pytest.approx(1 / 30, abs=1e-9)
    assert any("resynchronizing" in m for m in _messages(caplog))


def test_timer_warns_only_above_threshold(caplog):
    clock = FakeClock(overshoot_s=0.001)  # 1 ms noise, below 5 ms default
    timer = LoopTimer(30.0, clock=clock, sleep=clock.sleep).start()
    with caplog.at_level(logging.WARNING, logger="so101_collector.timing"):
        for _ in range(5):
            timer.wait()
    assert timer.summary().overruns == 5  # still counted
    assert not any("Frame overrun" in m for m in _messages(caplog))


def test_timer_requires_start_and_valid_fps():
    clock = FakeClock()
    timer = LoopTimer(30.0, clock=clock, sleep=clock.sleep)
    with pytest.raises(RuntimeError, match="before start"):
        timer.wait()
    with pytest.raises(ValueError, match="target_fps"):
        LoopTimer(0.0)


def test_timer_context_manager():
    clock = FakeClock()
    with LoopTimer(50.0, clock=clock, sleep=clock.sleep) as timer:
        assert timer.running
        timer.wait()
    assert not timer.running


# ---------------------------------------------------------------------------
# CameraHub
# ---------------------------------------------------------------------------
class FixedTimestampSource:
    """CameraSource that emits a fixed timestamp after a tiny delay."""

    def __init__(self, timestamp: float, *, shape=(32, 48), fail_open: bool = False) -> None:
        self.timestamp = timestamp
        self.shape = shape
        self.fail_open = fail_open
        self.opened = False

    def open(self) -> None:
        if self.fail_open:
            raise CameraError("simulated open failure")
        self.opened = True

    def read(self):
        time.sleep(0.001)
        frame = np.full((*self.shape, 3), 7, dtype=np.uint8)
        return True, frame, self.timestamp

    def close(self) -> None:
        self.opened = False


def _await_frames(hub: CameraHub, timeout_s: float = 2.0):
    """Poll until both workers have published their first frame."""
    deadline = time.monotonic() + timeout_s
    last_error: CameraError | None = None
    while time.monotonic() < deadline:
        try:
            return hub.get_synced_frames()
        except CameraError as exc:
            last_error = exc
            time.sleep(0.005)
    raise AssertionError(f"frames never became available: {last_error}")


def _hub_with_timestamps(base_ts: float, wrist_ts: float, **kwargs) -> CameraHub:
    vision = make_vision_config(width=48, height=32)
    return CameraHub(
        vision,
        source_factory=lambda cfg: FixedTimestampSource(
            base_ts if cfg.name == "base" else wrist_ts, shape=(32, 48)
        ),
        **kwargs,
    )


def test_synced_frames_returns_both_frames_and_small_dt():
    hub = _hub_with_timestamps(100.0, 100.005)
    with hub:
        base, wrist, dt_sync = _await_frames(hub)
        assert base.shape == (32, 48, 3)
        assert wrist.shape == (32, 48, 3)
        assert dt_sync == pytest.approx(0.005, abs=1e-6)
        workers = list(hub._workers.values())  # noqa: SLF001
    assert workers and all(not w.is_alive() for w in workers)


def test_sync_warning_fires_above_tolerance(caplog):
    warnings: list[float] = []
    hub = _hub_with_timestamps(100.0, 100.05, on_sync_warning=warnings.append)
    with hub, caplog.at_level(logging.DEBUG, logger="so101_collector.core.camera_hub"):
        _, _, dt_sync = _await_frames(hub)
        hub.get_synced_frames()
        hub.get_synced_frames()

    assert dt_sync == pytest.approx(0.05, abs=1e-6)
    assert warnings and all(w == pytest.approx(0.05, abs=1e-6) for w in warnings)
    assert any("drift" in m for m in _messages(caplog))
    stats = hub.stats()
    assert stats.sync_warnings == 3
    assert stats.sync_max_ms == pytest.approx(50.0, abs=1e-3)


def test_synced_frames_below_tolerance_does_not_warn(caplog):
    hub = _hub_with_timestamps(100.0, 100.01)
    with hub, caplog.at_level(logging.DEBUG, logger="so101_collector.core.camera_hub"):
        _await_frames(hub)
        hub.get_synced_frames()
    assert not any("drift" in m for m in _messages(caplog))
    assert hub.stats().sync_warnings == 0


def test_synced_frames_raises_before_first_frame():
    """A slow-starting camera must surface a clear error, not stale data."""

    class SlowSource(FixedTimestampSource):
        def read(self):
            time.sleep(0.5)
            return super().read()

    vision = make_vision_config(width=48, height=32)
    hub = CameraHub(vision, source_factory=lambda cfg: SlowSource(0.0, shape=(32, 48)))
    with hub:
        with pytest.raises(CameraError, match="has not produced a frame"):
            hub.get_synced_frames()


def test_get_frame_unknown_camera_raises():
    hub = _hub_with_timestamps(1.0, 1.0)
    with hub:
        with pytest.raises(CameraError, match="not managed"):
            hub.get_frame("thermal")
        _await_frames(hub)
        assert hub.get_frame("base").camera == "base"


def test_open_failure_surfaces_as_worker_error():
    vision = make_vision_config(width=48, height=32)
    hub = CameraHub(
        vision, source_factory=lambda cfg: FixedTimestampSource(0.0, fail_open=True)
    )
    with hub:
        deadline = time.monotonic() + 3.0
        surfaced = False
        while time.monotonic() < deadline:
            try:
                hub.get_frame("base")
            except CameraError as exc:
                if "worker failed" in str(exc):
                    surfaced = True
                    break
            time.sleep(0.02)
        assert surfaced, "worker open failure never surfaced through get_frame"


# ---------------------------------------------------------------------------
# Frame coercion
# ---------------------------------------------------------------------------
def test_coerce_frame_promotes_grayscale_and_alpha():
    gray = np.zeros((10, 20), dtype=np.uint8)
    out = coerce_frame(gray, camera="base", expected_shape=(10, 20))
    assert out.shape == (10, 20, 3)

    rgba = np.zeros((10, 20, 4), dtype=np.uint8)
    out = coerce_frame(rgba, camera="base", expected_shape=(10, 20))
    assert out.shape == (10, 20, 3)

    float_frame = np.zeros((10, 20, 3), dtype=np.float32)
    out = coerce_frame(float_frame, camera="base", expected_shape=(10, 20))
    assert out.dtype == np.uint8


def test_coerce_frame_rejects_resolution_mismatch():
    wrong = np.zeros((5, 5, 3), dtype=np.uint8)
    with pytest.raises(CameraError, match="expected"):
        coerce_frame(wrong, camera="wrist", expected_shape=(10, 20))
