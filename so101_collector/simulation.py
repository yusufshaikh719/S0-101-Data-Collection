"""Simulated hardware, cameras, and teleoperation for ``--simulate`` and tests.

Lets the entire pipeline run end-to-end on a machine with no robot
attached (CI smoke tests, demos, dataset-layout verification):

* :class:`SimulatedBus` - in-memory Feetech register model implementing the
  same transport API as
  :class:`~so101_collector.core.hardware_interface.FeetechBus`.
* :class:`SimulatedCamera` - paced synthetic RGB frames at the configured
  FPS with deterministic content (frame index + gradient).
* :class:`ScriptedTeleop` - smooth joint-space trajectory generator
  (sinusoids around the home pose) with optional "press-to-move" gating.

``SimulatedBus.fail_next(n)`` and ``SimulatedCamera.fail_next(n)`` let tests
and demos exercise the retry/recovery paths without hardware.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from typing import Sequence

import numpy as np

from so101_collector.config import CameraConfig, JOINT_NAMES, RobotConfig
from so101_collector.core.hardware_interface import (
    DEG_PER_TICK,
    TICKS_PER_REV,
    Instruction,
    StatusPacket,
    degrees_to_ticks,
    signed_magnitude,
)
from so101_collector.exceptions import BusCommunicationError, CameraError
from so101_collector.devices import CartCommand

# Registers mirrored by SimulatedBus (subset used by the interface).
_REG_TORQUE = 40
_REG_GOAL_POSITION = 42
_REG_PRESENT_POSITION = 56
_REG_PRESENT_SPEED = 58
_REG_PRESENT_LOAD = 60


class SimulatedBus:
    """In-memory Feetech transport: same API as ``FeetechBus``, no serial."""

    def __init__(self, config: RobotConfig, *, settle_speed_ticks_s: int = 4000) -> None:
        self._config = config
        self._settle_speed = settle_speed_ticks_s
        self._positions: dict[int, float] = {
            joint.servo_id: float(degrees_to_ticks(joint.home_deg, joint.offset_deg))
            for joint in config.joints
        }
        self._goals: dict[int, float] = dict(self._positions)
        self._open = False
        self._fail_count = 0
        self._lock = threading.Lock()
        self.write_count = 0
        self.read_count = 0

    # -- test/demo failure injection ---------------------------------------
    def fail_next(self, count: int = 1) -> None:
        """Make the next ``count`` operations raise (exercises retry logic)."""
        self._fail_count = int(count)

    @property
    def is_open(self) -> bool:
        return self._open

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def _maybe_fail(self, label: str) -> None:
        if not self._open:
            raise BusCommunicationError(f"simulated bus is not open ({label})")
        if self._fail_count > 0:
            self._fail_count -= 1
            raise BusCommunicationError(f"simulated transient failure during {label}")

    def _step_towards_goals(self) -> None:
        """Move present positions a bounded step towards their goals."""
        step = max(1.0, self._settle_speed * 0.033)  # one control tick worth
        for servo_id, goal in self._goals.items():
            current = self._positions[servo_id]
            delta = goal - current
            if delta == 0.0:
                continue
            self._positions[servo_id] = current + math.copysign(min(abs(delta), step), delta)

    # -- transport API ------------------------------------------------------
    def ping(self, device_id: int) -> StatusPacket:
        self._maybe_fail(f"ping({device_id})")
        self._check_id(device_id)
        return _ok_status(device_id)

    def read(self, device_id: int, address: int, length: int) -> bytes:
        self._maybe_fail(f"read({device_id})")
        self._check_id(device_id)
        self.read_count += 1
        with self._lock:
            self._step_towards_goals()
            if address == _REG_PRESENT_POSITION and length >= 6:
                position = int(round(self._positions[device_id])) & 0xFFFF
                speed = self._settle_speed & 0x3FF
                load = 128  # light idle load
                return struct.pack("<HHH", position, speed, load)
            if address == _REG_PRESENT_POSITION and length == 2:
                return struct.pack("<H", int(round(self._positions[device_id])) & 0xFFFF)
        raise BusCommunicationError(
            f"simulated bus: unsupported read at address {address} len {length}"
        )

    def write(self, device_id: int, address: int, data: bytes) -> StatusPacket:
        self._maybe_fail(f"write({device_id})")
        self._check_id(device_id)
        if address == _REG_GOAL_POSITION and len(data) == 2:
            with self._lock:
                self._goals[device_id] = float(struct.unpack("<H", data)[0])
        return _ok_status(device_id)

    def sync_write(self, address: int, items: Sequence[tuple[int, bytes]]) -> None:
        self._maybe_fail("sync_write")
        self.write_count += 1
        if address == _REG_GOAL_POSITION:
            with self._lock:
                for device_id, data in items:
                    self._check_id(device_id)
                    self._goals[device_id] = float(struct.unpack("<H", data)[0])

    # -- inspection ---------------------------------------------------------
    def goal_degrees(self) -> np.ndarray:
        """Commanded joint angles in degrees (config joint order)."""
        return np.array(
            [
                (self._goals[j.servo_id] * DEG_PER_TICK)
                for j in self._config.joints
            ],
            dtype=np.float32,
        )

    def _check_id(self, device_id: int) -> None:
        known = {j.servo_id for j in self._config.joints}
        if device_id not in known:
            raise BusCommunicationError(f"simulated bus: no servo with id {device_id}")


def _ok_status(device_id: int) -> StatusPacket:
    return StatusPacket(device_id=device_id, error=0, params=b"")


# ---------------------------------------------------------------------------
# Simulated camera pacing epoch: both cameras align their frame deadlines to
# this shared origin so their timestamps stay phase-locked like a real
# hardware-triggered pair (a per-open origin would fake sync drift).
# ---------------------------------------------------------------------------
_SHARED_EPOCH = time.monotonic()


class SimulatedCamera:
    """Deterministic paced frames (gradient + encoded frame counter)."""

    def __init__(self, config: CameraConfig, *, clock=time.monotonic) -> None:
        self._config = config
        self._clock = clock
        self._frame_index = 0
        self._next_frame_at: float | None = None
        self._open = False
        self._fail_count = 0
        # Pre-render the static background once: per-frame cost is then a
        # single array copy + two patches, so the worker publishes promptly
        # after waking (render latency would fake synchronization errors).
        height, width = config.frame_shape
        template = np.empty((height, width, 3), dtype=np.uint8)
        template[:, :, 0] = (np.arange(width, dtype=np.uint32) % 256).astype(np.uint8)[None, :]
        template[:, :, 1] = (np.arange(height, dtype=np.uint32) % 256).astype(np.uint8)[:, None]
        template[:, :, 2] = 64 if config.name == "wrist" else 192
        self._template = template

    def fail_next(self, count: int = 1) -> None:
        self._fail_count = int(count)

    def open(self) -> None:
        self._open = True
        self._next_frame_at = self._clock()
        self._frame_index = 0

    def close(self) -> None:
        self._open = False

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if not self._open:
            raise CameraError(f"simulated camera '{self._config.name}' is not open")
        if self._fail_count > 0:
            self._fail_count -= 1
            return False, None, self._clock()

        # Pace on absolute deadlines anchored to a shared epoch so two
        # cameras stay phase-aligned even when thread wakeups are jittery.
        period = 1.0 / self._config.fps
        now = self._clock()
        assert self._next_frame_at is not None
        # time.sleep() may wake early (Windows timer granularity), so loop
        # until the deadline is genuinely reached.
        while now < self._next_frame_at - 1e-4:
            # 1 ms floor prevents a busy spin if sleep wakes immediately.
            time.sleep(max(self._next_frame_at - now, 0.001))
            now = self._clock()
        # Next deadline = the next epoch-aligned period boundary after now.
        elapsed = max(now - _SHARED_EPOCH, 0.0)
        self._next_frame_at = _SHARED_EPOCH + (math.floor(elapsed / period) + 1) * period

        frame = self._render()
        self._frame_index += 1
        return True, frame, now

    def _render(self) -> np.ndarray:
        frame = self._template.copy()
        # Moving vertical bar + frame counter keep frames distinguishable in
        # saved videos without any per-pixel work.
        width = frame.shape[1]
        marker_x = (self._frame_index * 4) % width
        frame[:, marker_x : min(marker_x + 4, width), :] = 255
        frame[0:8, 0:48] = (self._frame_index * 3) % 256
        return frame


class ScriptedTeleop:
    """Smooth sinusoidal joint trajectory around the home pose.

    ``gated=True`` only moves while :meth:`press` is held (mimics a
    SpaceMouse button in demos); otherwise it always moves.
    """

    def __init__(
        self,
        config: RobotConfig,
        *,
        amplitude_deg: float = 20.0,
        period_s: float = 6.0,
        gated: bool = False,
        clock=time.monotonic,
    ) -> None:
        self._config = config
        self._amplitude = float(amplitude_deg)
        self._period = float(period_s)
        self._gated = gated
        self._clock = clock
        self._pressed = not gated
        self._t0: float | None = None

    def press(self) -> None:
        """Start moving (for gated demos keyed to the START command)."""
        self._pressed = True

    def release(self) -> None:
        self._pressed = False

    def open(self) -> None:
        self._t0 = self._clock()

    def close(self) -> None:
        return None

    def read(self, dt_s: float) -> np.ndarray | None:
        del dt_s
        if self._t0 is None:
            self._t0 = self._clock()
        elapsed = self._clock() - self._t0
        home = np.asarray(self._config.home_pose, dtype=np.float64)
        action = home.copy()
        if self._pressed:
            phase = 2.0 * math.pi * elapsed / self._period
            for index in range(len(JOINT_NAMES) - 1):  # gripper stays put
                scale = 0.4 + 0.15 * index
                action[index] += self._amplitude * scale * math.sin(
                    phase + index * math.pi / 5.0
                )
        return action.astype(np.float32)


class StaticTeleop:
    """Holds the home pose (useful for pure camera/latency checks)."""

    def __init__(self, config: RobotConfig) -> None:
        self._home = np.asarray(config.home_pose, dtype=np.float32)

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, dt_s: float) -> np.ndarray | None:
        del dt_s
        return self._home.copy()


class ZeroCartDevice:
    """A motionless 6-DOF device (always returns a zero CartCommand)."""

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def poll(self) -> CartCommand:
        return CartCommand()


# Re-exported for convenience in scripts/tests.
__all__ = [
    "ScriptedTeleop",
    "SimulatedBus",
    "SimulatedCamera",
    "StaticTeleop",
    "ZeroCartDevice",
    "signed_magnitude",
    "Instruction",
    "TICKS_PER_REV",
]
