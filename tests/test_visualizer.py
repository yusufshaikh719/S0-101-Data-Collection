"""Tests for Rerun visualizer integration and 3D kinematics visualization."""

from __future__ import annotations

import numpy as np
import pytest

from so101_collector.console import TimedCommandReader
from so101_collector.core.camera_hub import CameraHub
from so101_collector.core.dataset_writer import DatasetWriter, NpzTableStore, OpenCVVideoEncoder
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.ik import SO101Kinematics
from so101_collector.core.runner import DataCollectionRunner, RunnerState
from so101_collector.core.teleop_controller import LeaderArmTeleop
from so101_collector.core.visualizer import NullVisualizer, RerunVisualizer
from so101_collector.simulation import SimulatedBus, SimulatedCamera
from tests.conftest import make_robot_config, make_vision_config


def test_forward_chain_consistency():
    """Verify forward_chain computes all link positions and matches forward()."""
    kin = SO101Kinematics()
    angles = [15.0, 30.0, -45.0, 10.0, 0.0]
    chain = kin.forward_chain(angles)

    assert set(chain.keys()) == {"base", "shoulder", "elbow", "wrist", "tool"}
    np.testing.assert_allclose(chain["base"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(chain["shoulder"], [0.0, 0.0, kin.config.shoulder_height_mm])
    # The tool position must match kin.forward()
    np.testing.assert_allclose(chain["tool"], kin.forward(angles))


def test_null_visualizer_noop():
    """NullVisualizer should execute without errors and have zero side effects."""
    vis = NullVisualizer()
    vis.start()
    vis.log_step(
        session_time_s=1.0,
        step_index=1,
        state="recording",
        measured_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        applied_action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    vis.log_event("test_event", "hello")
    vis.log_metrics({"fps": 30.0})
    vis.close()


def test_rerun_visualizer_saves_rrd(tmp_path):
    """RerunVisualizer can record multimodal streams to an .rrd file."""
    rrd_file = tmp_path / "recording.rrd"
    vis = RerunVisualizer(
        application_id="test_visualizer",
        save_path=rrd_file,
        spawn=False,
    )
    vis.start()
    assert vis.is_active

    kin = SO101Kinematics()
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    depth = np.full((48, 64), 500, dtype=np.uint16)

    vis.log_event("session_start", "Starting test")
    vis.log_step(
        session_time_s=0.033,
        step_index=1,
        state="recording",
        measured_positions=np.array([10.0, 20.0, -30.0, 40.0, 0.0, 15.0], dtype=np.float32),
        applied_action=np.array([10.0, 20.0, -30.0, 40.0, 0.0, 15.0], dtype=np.float32),
        base_rgb=rgb,
        wrist_rgb=rgb,
        base_depth=depth,
        wrist_depth=depth,
        dt_sync_ms=1.5,
        loop_fps=29.9,
        episode_index=0,
        episode_frame_index=1,
        kinematics=kin,
    )
    vis.log_metrics({"loop_jitter_ms": 0.2})
    vis.close()

    assert rrd_file.is_file()
    assert rrd_file.stat().st_size > 0


def test_runner_integration_with_rerun(tmp_path):
    """DataCollectionRunner runs smoothly with RerunVisualizer enabled."""
    robot = make_robot_config()
    vision = make_vision_config(width=64, height=48)
    hardware = SO101HardwareInterface(robot, bus=SimulatedBus(robot))
    leader = SO101HardwareInterface(robot, bus=SimulatedBus(robot))
    hub = CameraHub(vision, source_factory=lambda cfg: SimulatedCamera(cfg))
    teleop = LeaderArmTeleop(leader)

    writer = DatasetWriter(
        tmp_path / "data",
        "rerun_test_repo",
        fps=30.0,
        image_shape=vision.by_name["base"].frame_shape,
        encoder=OpenCVVideoEncoder(),
        table_store=NpzTableStore(),
    )
    rrd_path = tmp_path / "telemetry.rrd"
    vis = RerunVisualizer(
        application_id="runner_integration",
        save_path=rrd_path,
        spawn=False,
    )
    kinematics = SO101Kinematics(robot.kinematics)

    runner = DataCollectionRunner(
        hardware=hardware,
        camera_hub=hub,
        teleop=teleop,
        writer=writer,
        commands=TimedCommandReader(record_s=0.4),
        fps=30.0,
        warmup_s=0.1,
        task="test rerun runner",
        visualizer=vis,
        kinematics=kinematics,
    )

    stats = runner.run(max_episodes=1, max_seconds=10.0)

    assert stats.episodes_saved == 1
    assert stats.final_state == RunnerState.IDLE.value
    assert rrd_path.is_file()
    assert rrd_path.stat().st_size > 0
