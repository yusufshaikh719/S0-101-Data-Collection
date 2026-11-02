"""Tests for the Rerun visualizer and the 3D kinematics helper."""

from __future__ import annotations

import numpy as np

from so101_collector.core.ik import SO101Kinematics
from so101_collector.core.visualizer import NullVisualizer, RerunVisualizer


def test_forward_chain_consistency():
    kin = SO101Kinematics()
    angles = [15.0, 30.0, -45.0, 10.0, 0.0]
    chain = kin.forward_chain(angles)

    assert set(chain) == {"base", "shoulder", "elbow", "wrist", "tool"}
    np.testing.assert_allclose(chain["base"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(chain["shoulder"], [0.0, 0.0, kin.config.shoulder_height_mm])
    np.testing.assert_allclose(chain["tool"], kin.forward(angles))


def test_null_visualizer_is_a_noop():
    vis = NullVisualizer()
    vis.start()
    vis.log_step(
        session_time_s=1.0,
        step_index=1,
        state="recording",
        measured_positions=[0.0] * 6,
        applied_action=[0.0] * 6,
    )
    vis.log_event("test_event", "hello")
    vis.log_metrics({"fps": 30.0})
    vis.close()


def test_rerun_visualizer_saves_rrd(tmp_path):
    rrd_file = tmp_path / "recording.rrd"
    vis = RerunVisualizer(application_id="test_visualizer", save_path=rrd_file, spawn=False)
    vis.start()
    assert vis.is_active

    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    vis.log_event("session_start", "Starting test")
    vis.log_step(
        session_time_s=0.033,
        step_index=1,
        state="recording",
        measured_positions=np.array([10.0, 20.0, -30.0, 40.0, 0.0, 15.0], dtype=np.float32),
        applied_action=np.array([10.0, 20.0, -30.0, 40.0, 0.0, 15.0], dtype=np.float32),
        base_rgb=rgb,
        wrist_rgb=rgb,
        dt_sync_ms=1.5,
        loop_fps=29.9,
        episode_index=0,
        episode_frame_index=1,
        kinematics=SO101Kinematics(),
    )
    vis.log_metrics({"loop_jitter_ms": 0.2})
    vis.close()

    assert rrd_file.is_file()
    assert rrd_file.stat().st_size > 0
