"""Teleoperation sources that produce follower joint-angle targets.

``LeaderArmTeleop`` mirrors a passive leader arm. ``CartesianTeleop`` turns
SpaceMouse or gamepad rates into tool-tip motion through the analytical IK.
Both implement ``TeleopSource``: ``read(dt)`` returns the next action vector,
or ``None`` when the device has no fresh data and the runner should hold the
previous command.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from so101_collector.config import JOINT_NAMES, RobotConfig
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.ik import SO101Kinematics
from so101_collector.devices import CartCommand, InputDevice
from so101_collector.exceptions import IKError

logger = logging.getLogger(__name__)

GRIPPER_INDEX = JOINT_NAMES.index("gripper")
WRIST_ROLL_INDEX = JOINT_NAMES.index("wrist_roll")


class TeleopSource(Protocol):
    """Produces 6-DOF joint targets at control-loop rate."""

    def open(self) -> None: ...

    def read(self, dt_s: float) -> np.ndarray | None: ...

    def close(self) -> None: ...


class LeaderArmTeleop:
    """Mirror a passive 1:1 leader arm onto the follower."""

    def __init__(
        self,
        leader_interface: SO101HardwareInterface,
        offsets_deg: Sequence[float] | None = None,
    ) -> None:
        self._leader = leader_interface
        offsets = np.zeros(len(JOINT_NAMES)) if offsets_deg is None else np.asarray(
            offsets_deg, dtype=np.float64
        )
        if offsets.size != len(JOINT_NAMES):
            raise ValueError(
                f"offsets_deg must have {len(JOINT_NAMES)} entries, got {offsets.size}"
            )
        self._offsets = offsets

    @property
    def leader(self) -> SO101HardwareInterface:
        return self._leader

    def open(self) -> None:
        self._leader.open()

    def read(self, dt_s: float) -> np.ndarray | None:
        del dt_s  # leader is an absolute position source; dt is irrelevant
        state = self._leader.read_state()
        return (state.positions.astype(np.float64) + self._offsets).astype(np.float32)

    def close(self) -> None:
        self._leader.close()


class CartesianTeleop:
    """Rate control of the tool tip through the analytical IK.

    Velocities are integrated into a tool-tip target, clamped to the
    workspace and solved. An unreachable target keeps the previous command.
    """

    def __init__(
        self,
        kinematics: SO101Kinematics,
        device: InputDevice,
        home_pose: Sequence[float],
        gripper_limits: tuple[float, float],
        *,
        start_position_mm: Sequence[float] | None = None,
        elbow: str = "up",
        hold_timeout_s: float = 5.0,
    ) -> None:
        home = np.asarray(home_pose, dtype=np.float64).reshape(-1)
        if home.size != len(JOINT_NAMES):
            raise ValueError(f"home_pose must have {len(JOINT_NAMES)} entries")
        if elbow not in ("up", "down"):
            raise ValueError(f"elbow must be 'up' or 'down', got {elbow!r}")
        self._kin = kinematics
        self._device = device
        self._elbow = elbow
        self._gripper_min, self._gripper_max = gripper_limits
        self._hold_timeout_s = hold_timeout_s
        self._last_ik_error_at: float | None = None

        start = (
            kinematics.forward(home)
            if start_position_mm is None
            else np.asarray(start_position_mm, dtype=np.float64)
        )
        self._position = kinematics.clamp_workspace(start)
        self._roll_deg = float(home[WRIST_ROLL_INDEX])
        self._gripper_deg = float(np.clip(home[GRIPPER_INDEX], *gripper_limits))
        try:
            q5 = kinematics.inverse(self._position, wrist_roll_deg=self._roll_deg, elbow=elbow)
        except IKError:  # pragma: no cover - home pose is always reachable
            q5 = home[:5]
        self._action = np.concatenate([q5, [self._gripper_deg]]).astype(np.float32)

    @property
    def position_mm(self) -> np.ndarray:
        """Current integrated tool-tip target (before IK)."""
        return self._position.copy()

    @property
    def action(self) -> np.ndarray:
        """Most recent joint command (useful for logging/inspection)."""
        return self._action.copy()

    def open(self) -> None:
        self._device.open()

    def read(self, dt_s: float) -> np.ndarray | None:
        command: CartCommand = self._device.poll()
        dt = float(np.clip(dt_s, 0.0, 0.1))

        vx, vy, vz = command.linear_mm_s
        self._position += np.array([vx, vy, vz], dtype=np.float64) * dt
        self._position = self._kin.clamp_workspace(self._position)
        self._roll_deg += command.roll_deg_s * dt
        self._gripper_deg = float(
            np.clip(
                self._gripper_deg + command.gripper_deg_s * dt,
                self._gripper_min,
                self._gripper_max,
            )
        )

        try:
            q5 = self._kin.inverse(
                self._position, wrist_roll_deg=self._roll_deg, elbow=self._elbow
            )
        except IKError as exc:
            self._log_ik_hold(exc)
            return self._action
        self._action = np.concatenate([q5, [self._gripper_deg]]).astype(np.float32)
        return self._action

    def _log_ik_hold(self, exc: IKError) -> None:
        now = time.monotonic()
        if self._last_ik_error_at is None or now - self._last_ik_error_at >= self._hold_timeout_s:
            self._last_ik_error_at = now
            logger.warning("IK failed for target %s - holding previous command: %s",
                           np.round(self._position, 1), exc)

    def close(self) -> None:
        self._device.close()


def gripper_limits_from_config(config: RobotConfig) -> tuple[float, float]:
    """(closed, open) gripper limits from the joint spec."""
    spec = config.joint_by_name["gripper"]
    return (spec.min_deg, spec.max_deg)
