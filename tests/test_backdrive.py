"""Hand-guided follower teleoperation (``--teleop backdrive``).

The arm is dragged by hand with servo torque released, so the tests drive the
simulated bus through ``manual_move`` like a human would.
"""

from __future__ import annotations

import numpy as np
import pytest

from so101_collector.console import Command
from so101_collector.core.camera_hub import CameraHub
from so101_collector.core.continuous_runner import ContinuousDataCollectionRunner
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.lerobot_writer import LeRobotWriter
from so101_collector.core.teleop_controller import (
    BackdriveTeleop,
    LeaderArmTeleop,
    request_home,
)
from so101_collector.exceptions import SO101Error
from so101_collector.simulation import ScriptedTeleop, SimulatedBus, SimulatedCamera
from tests.conftest import make_robot_config, make_vision_config
from tests.test_pipeline import FakeDataset

FPS = 30.0
DT = 1.0 / FPS


def _rig(**kwargs):
    robot = make_robot_config()
    bus = SimulatedBus(robot)
    hardware = SO101HardwareInterface(robot, bus=bus)
    teleop = BackdriveTeleop(hardware, release_delay_s=0.0, **kwargs)
    return robot, bus, hardware, teleop


def test_open_releases_torque_on_every_servo():
    robot, bus, _hardware, teleop = _rig()
    teleop.open()

    assert teleop.released
    assert all(not bus.torque_enabled(joint.servo_id) for joint in robot.joints)
    teleop.close()


def test_open_refuses_to_hand_guide_when_torque_stays_enabled():
    robot = make_robot_config()

    class StubbornHardware(SO101HardwareInterface):
        """Servo ignores the torque write, as a stuck/firmware-limited joint does."""

        def set_torque(self, enabled: bool) -> None:
            del enabled

    hardware = StubbornHardware(robot, bus=SimulatedBus(robot))
    teleop = BackdriveTeleop(hardware, release_delay_s=0.0)

    # Dragging a powered joint would fight the operator, so refuse instead.
    with pytest.raises(SO101Error, match="torque is still enabled"):
        teleop.open()


def test_free_dragging_never_commands_goal_positions():
    _robot, bus, _hardware, teleop = _rig()
    teleop.open()
    writes = bus.write_count

    for _ in range(10):
        teleop.read(DT)

    # A goal write can re-engage the position loop on some servo firmware, so
    # free-dragging must not send any.
    assert bus.write_count == writes
    assert teleop.writes_targets is False

    assert request_home(teleop) is True
    assert teleop.writes_targets is True  # homing needs the runner to write
    teleop.close()


def test_read_follows_the_dragged_arm_and_leads_by_one_frame():
    robot, bus, _hardware, teleop = _rig()
    teleop.open()
    home = np.asarray(robot.home_pose, dtype=np.float64)

    bus.manual_move([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    first = teleop.read(DT)
    # No previous frame yet: the action is the measured pose (no extrapolation).
    assert first[0] == pytest.approx(home[0] + 10.0, abs=0.2)

    bus.manual_move([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    second = teleop.read(DT)
    # Dragged 10 deg again: the action is the next pose, not the current one,
    # so a position-controlled policy reproduces the motion instead of holding.
    assert second[0] == pytest.approx(home[0] + 30.0, abs=0.5)

    steady = teleop.read(DT)
    assert steady[0] == pytest.approx(home[0] + 20.0, abs=0.3)
    teleop.close()


def test_read_clamps_to_joint_limits():
    robot, bus, _hardware, teleop = _rig()
    teleop.open()
    bus.manual_move([150.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # past shoulder_pan's 135 deg limit
    action = teleop.read(DT)
    assert action[0] == pytest.approx(robot.joints[0].max_deg)
    teleop.close()


def test_request_home_ramps_under_power_then_releases():
    robot, bus, hardware, teleop = _rig()
    teleop.open()
    home = np.asarray(robot.home_pose, dtype=np.float64)

    bus.manual_move([0.0, -25.0, 0.0, 0.0, 0.0, 0.0])  # arm sagged away from home
    assert request_home(teleop) is True
    assert all(bus.torque_enabled(joint.servo_id) for joint in robot.joints)
    assert teleop.homing

    ticks = 0
    while teleop.homing and ticks < 400:
        hardware.write_positions(teleop.read(DT))  # what the runner does each tick
        ticks += 1

    assert ticks < 400, "homing never reached the tucked pose"
    state = hardware.read_state()
    assert np.all(np.abs(state.positions - home) <= 5.0 + 0.5)
    assert teleop.released
    assert not bus.torque_enabled(robot.joints[0].servo_id)
    teleop.close()


def test_close_restores_torque_holding_the_current_pose():
    robot, bus, hardware, teleop = _rig()
    teleop.open()
    bus.manual_move([15.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    teleop.read(DT)

    teleop.close()

    assert not teleop.released
    assert all(bus.torque_enabled(joint.servo_id) for joint in robot.joints)
    assert hardware.read_state().positions[0] == pytest.approx(15.0, abs=1.0)


def test_sources_without_homing_defer_to_the_runner():
    robot = make_robot_config()
    hardware = SO101HardwareInterface(robot, bus=SimulatedBus(robot))
    assert request_home(ScriptedTeleop(robot)) is False
    assert request_home(LeaderArmTeleop(hardware)) is False
    assert request_home(BackdriveTeleop(hardware, release_delay_s=0.0)) is False  # not open


def test_runner_splits_hand_guided_episodes_on_home_departure(tmp_path):
    robot = make_robot_config()
    vision = make_vision_config(width=64, height=48)
    bus = SimulatedBus(robot)
    hardware = SO101HardwareInterface(robot, bus=bus)
    hub = CameraHub(vision, source_factory=lambda cfg: SimulatedCamera(cfg))
    writer = LeRobotWriter(
        tmp_path,
        "backdrive",
        fps=30,
        image_shapes={"base": (48, 64), "wrist": (48, 64)},
        dataset_class=FakeDataset,
    )

    class DragThenQuit:
        """Drag the arm out and back again, then stop the session."""

        def __init__(self) -> None:
            self.tick = 0

        def read(self):
            self.tick += 1
            if self.tick == 4:
                bus.manual_move([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif self.tick == 12:
                bus.manual_move([-20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return Command.QUIT if self.tick > 20 else None

        def close(self) -> None:
            pass

    runner = ContinuousDataCollectionRunner(
        hardware=hardware,
        camera_hub=hub,
        teleop=BackdriveTeleop(hardware, release_delay_s=0.0),
        writer=writer,
        commands=DragThenQuit(),
        fps=30,
        warmup_s=0.1,
    )
    stats = runner.run(max_seconds=20)

    episodes = writer._dataset.episodes  # noqa: SLF001
    assert len(episodes) == stats.episodes_saved == 1
    home = np.asarray(robot.home_pose)
    first = episodes[0][0]["observation.state"]
    last = episodes[0][-1]["observation.state"]
    assert np.abs(first - home).max() > 8.0
    assert np.all(np.abs(last - home) <= 5.0 + 1e-6)
