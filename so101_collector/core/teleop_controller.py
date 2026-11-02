"""Teleoperation sources that produce follower joint-angle targets.

``LeaderArmTeleop`` mirrors a passive leader arm. ``BackdriveTeleop`` releases
the follower's servo torque so the arm can be dragged by hand, for rigs with no
leader arm. ``CartesianTeleop`` turns SpaceMouse or gamepad rates into tool-tip
motion through the analytical IK. All implement ``TeleopSource``: ``read(dt)``
returns the next action vector, or ``None`` when the device has no fresh data
and the runner should hold the previous command.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from so101_collector.config import JOINT_NAMES, RobotConfig
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.ik import SO101Kinematics
from so101_collector.devices import CartCommand, InputDevice
from so101_collector.exceptions import IKError, SO101Error

logger = logging.getLogger(__name__)

GRIPPER_INDEX = JOINT_NAMES.index("gripper")
WRIST_ROLL_INDEX = JOINT_NAMES.index("wrist_roll")


class TeleopSource(Protocol):
    """Produces 6-DOF joint targets at control-loop rate."""

    def open(self) -> None: ...

    def read(self, dt_s: float) -> np.ndarray | None: ...

    def close(self) -> None: ...


def request_home(teleop: TeleopSource) -> bool:
    """Let ``teleop`` home the arm itself.

    Returns True when the source owns the homing sequence (it must re-enable
    torque first), False when the runner should simply command the home pose.
    """
    handler = getattr(teleop, "request_home", None)
    return bool(handler()) if callable(handler) else False


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


class BackdriveTeleop:
    """Hand-guide the follower arm with servo torque released.

    Every servo is switched to free-spin so the arm can be dragged through a
    demonstration. Each tick returns the measured pose extrapolated one control
    period ahead: that is the position the follower's position loop would have
    to be commanded to land on the next demonstrated sample. Recording the
    current pose instead would teach a policy to hold still.

    Homing is owned by this source because the arm runs unpowered:
    ``request_home()`` restores torque, ramps the target back to ``home_deg`` at
    ``home_speed_deg_s`` and releases torque again on arrival. ``close()``
    restores torque holding whatever pose the arm is in, so it does not drop.
    """

    def __init__(
        self,
        hardware: SO101HardwareInterface,
        *,
        home_speed_deg_s: float = 45.0,
        home_tolerance_deg: float = 5.0,
        release_delay_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not home_speed_deg_s > 0:
            raise ValueError(f"home_speed_deg_s must be > 0, got {home_speed_deg_s}")
        if not home_tolerance_deg > 0:
            raise ValueError(f"home_tolerance_deg must be > 0, got {home_tolerance_deg}")
        if release_delay_s < 0:
            raise ValueError(f"release_delay_s must be >= 0, got {release_delay_s}")
        self._hardware = hardware
        config = hardware.config
        self._home = np.asarray(config.home_pose, dtype=np.float64)
        self._min = np.array([j.min_deg for j in config.joints], dtype=np.float64)
        self._max = np.array([j.max_deg for j in config.joints], dtype=np.float64)
        self._home_speed = float(home_speed_deg_s)
        self._home_tolerance = float(home_tolerance_deg)
        self._release_delay = float(release_delay_s)
        self._sleep = sleep
        self._target = self._home.copy()
        self._previous: np.ndarray | None = None
        self._homing = False
        self._released = False
        self._torque_audit_interval_s = 5.0
        self._last_torque_check = 0.0

    @property
    def hardware(self) -> SO101HardwareInterface:
        return self._hardware

    @property
    def released(self) -> bool:
        """True while the servos are free-spinning."""
        return self._released

    @property
    def homing(self) -> bool:
        """True while ramping back to the tucked pose under power."""
        return self._homing

    @property
    def writes_targets(self) -> bool:
        """Whether the runner must write this source's target to the servos.

        Free-dragging must not: a goal-position write can re-engage the
        position loop on some firmware, which makes a released arm feel stiff
        again. Homing must, because the ramp target has to reach the servos.
        """
        return self._homing

    def open(self) -> None:
        self._hardware.open()
        logger.warning(
            "backdrive: releasing servo torque in %.1f s - the arm will drop, "
            "hold it before then",
            self._release_delay,
        )
        if self._release_delay > 0:
            self._sleep(self._release_delay)
        self._target = self._measured()
        self._release()
        self._verify_released()

    def read(self, dt_s: float) -> np.ndarray | None:
        measured = self._measured()
        previous, self._previous = self._previous, measured
        if self._homing:
            return self._step_home(measured, dt_s)
        self._audit_torque()
        if previous is None:
            self._target = measured
            return self._clamped(measured)
        # Extrapolate by one frame; no servo speed sign or unit assumptions.
        self._target = 2.0 * measured - previous
        return self._clamped(self._target)

    def request_home(self) -> bool:
        """Restore torque and ramp back to the tucked pose.

        Returns False before torque was ever released (nothing to do).
        """
        if not self._released:
            return False
        if self._homing:
            return True
        self._hold_pose()  # power up without commanding a jump
        self._homing = True
        logger.info(
            "backdrive: torque on, homing at %.0f deg/s (torque releases again at the "
            "tucked pose)",
            self._home_speed,
        )
        return True

    def close(self) -> None:
        if not self._released:
            return
        try:
            self._hold_pose()
            logger.info("backdrive: torque restored, holding the current pose")
        except SO101Error as exc:
            logger.error("backdrive: could not restore servo torque: %s", exc)
        finally:
            self._released = False
            self._homing = False

    # -- internals ---------------------------------------------------------
    def _measured(self) -> np.ndarray:
        return self._hardware.read_state().positions.astype(np.float64)

    def _clamped(self, target: np.ndarray) -> np.ndarray:
        return np.clip(target, self._min, self._max).astype(np.float32)

    def _hold_pose(self) -> None:
        """Command the measured pose, then power the servos up.

        The target is clamped first because encoder quantization can put a
        measured angle a few hundredths of a degree outside a joint limit.
        """
        self._target = self._clamped(self._measured()).astype(np.float64)
        self._hardware.write_positions(self._target)
        self._hardware.set_torque(True)

    def _release(self) -> None:
        # Leave the goal register on the measured pose so a later torque enable
        # does not command a jump. Clamp first: encoder quantization can put a
        # measured pose a few hundredths of a degree outside a joint limit.
        self._target = self._clamped(self._target).astype(np.float64)
        self._hardware.write_positions(self._target)
        self._hardware.set_torque(False)
        self._released = True
        self._homing = False

    def _audit_torque(self) -> None:
        """Periodically confirm the servos are still free-spinning."""
        now = time.monotonic()
        if now - self._last_torque_check < self._torque_audit_interval_s:
            return
        self._last_torque_check = now
        try:
            enabled = self._hardware.read_torque()
        except SO101Error as exc:
            logger.debug("backdrive: torque audit skipped: %s", exc)
            return
        stuck = [
            joint.name
            for joint, on in zip(self._hardware.config.joints, enabled, strict=True)
            if on
        ]
        if stuck:
            logger.warning(
                "backdrive: servo(s) %s re-enabled torque while hand-guiding; releasing again",
                ", ".join(stuck),
            )
            self._hardware.set_torque(False)

    def _verify_released(self) -> None:
        """Fail loudly if a servo kept its position loop enabled.

        A joint that still holds torque fights the operator's hand, which
        reads as "the arm is stiff" but is a bus/firmware problem, not gear
        friction. Attempting to drag it anyway risks stalling the servo.
        """
        try:
            enabled = self._hardware.read_torque()
        except SO101Error as exc:
            logger.warning("backdrive: could not read back torque state: %s", exc)
            return
        stuck = [
            joint.name
            for joint, on in zip(self._hardware.config.joints, enabled, strict=True)
            if on
        ]
        if stuck:
            raise SO101Error(
                "backdrive: torque is still enabled on "
                + ", ".join(stuck)
                + "; the arm would fight your hand. Check the servo power supply and "
                "that no other program is driving the bus, then retry"
            )
        logger.info("backdrive: torque released and confirmed on all %d joints", enabled.size)

    def _step_home(self, measured: np.ndarray, dt_s: float) -> np.ndarray:
        if np.all(np.abs(measured - self._home) <= self._home_tolerance):
            logger.info("backdrive: tucked pose reached; torque released again")
            self._target = measured
            self._release()
            return self._clamped(measured)
        step = self._home_speed * max(dt_s, 1e-3)
        self._target = self._target + np.clip(self._home - self._target, -step, step)
        return self._clamped(self._target)


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
