"""Tests for the on-site setup/pre-flight and tucked-pose helpers (no hardware)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so101_collector.core.camera_hub import capture_source
from so101_collector.exceptions import SO101Error
from so101_collector.scripts.calibrate_joints import (
    _apply_fields,
    summarize_home_samples,
    validate_home_pose,
)
from so101_collector.scripts.setup_rig import (
    discover_video_devices,
    replace_yaml_scalar,
    yaml_scalar,
)
from tests.conftest import make_robot_config


def test_replace_yaml_scalar_keeps_comments_and_other_keys():
    text = (
        "# header\n"
        "port: /dev/serial/by-id/old\n"
        "allow_dynamic_port: false\n"
        "leader_port: null\n"
    )
    updated = replace_yaml_scalar(text, "port", "/dev/serial/by-id/new")
    assert "port: /dev/serial/by-id/new" in updated
    assert "leader_port: null" in updated
    assert "allow_dynamic_port: false" in updated
    assert "# header" in updated


def test_replace_yaml_scalar_hits_the_requested_occurrence():
    text = (
        "cameras:\n"
        "  - name: base\n"
        "    device_path: /dev/v4l/by-id/a\n"
        "  - name: wrist\n"
        "    device_path: /dev/v4l/by-id/b\n"
    )
    updated = replace_yaml_scalar(text, "device_path", "/dev/v4l/by-id/c", occurrence=1)
    assert "    device_path: /dev/v4l/by-id/a" in updated
    assert "    device_path: /dev/v4l/by-id/c" in updated


def test_replace_yaml_scalar_missing_key_raises():
    with pytest.raises(ValueError):
        replace_yaml_scalar("a: 1\n", "port", "/dev/x")


def test_replace_yaml_scalar_keeps_inline_comment():
    updated = replace_yaml_scalar("port: /dev/old  # adapter\n", "port", "/dev/new")
    assert updated == "port: /dev/new # adapter\n"


def test_discover_video_devices_keeps_capture_nodes_only(tmp_path):
    (tmp_path / "usb-A-video-index0").write_text("")
    (tmp_path / "usb-A-video-index1").write_text("")
    devices = discover_video_devices(tmp_path)
    assert [device.name for device in devices] == ["usb-A-video-index0"]


def test_discover_video_devices_missing_directory_is_empty():
    assert discover_video_devices(Path("does-not-exist")) == []


def test_summarize_home_samples_mean_and_spread():
    samples = [
        np.array([1.0, 10.0]),
        np.array([3.0, 10.5]),
        np.array([2.0, 9.5]),
    ]
    mean, spread = summarize_home_samples(samples)
    assert mean == pytest.approx([2.0, 10.0])
    assert spread == pytest.approx([2.0, 1.0])


def test_summarize_home_samples_requires_a_sample():
    with pytest.raises(ValueError):
        summarize_home_samples([])


def test_home_pose_outside_limits_is_rejected():
    robot = make_robot_config()
    captured = np.array([0.0, 45.0, -60.0, 15.0, 0.0, 90.0])
    with pytest.raises(SO101Error) as excinfo:
        validate_home_pose(robot, captured)
    assert "gripper" in str(excinfo.value)


def test_home_pose_inside_limits_is_accepted():
    robot = make_robot_config()
    validate_home_pose(robot, np.asarray(robot.home_pose, dtype=np.float64))


def test_apply_fields_moves_only_the_named_joint():
    robot = make_robot_config()
    updated = _apply_fields(robot, "home_deg", {"gripper": 0.0})
    assert updated.joints[-1].home_deg == 0.0
    assert updated.joints[0].home_deg == robot.joints[0].home_deg
    assert updated.joints[-1].offset_deg == robot.joints[-1].offset_deg


def test_capture_source_converts_indices_only():
    assert capture_source("0") == 0
    path = "/dev/v4l/by-id/usb-A-video-index0"
    assert capture_source(path) == path


def test_yaml_scalar_quotes_numeric_camera_index():
    assert yaml_scalar("0") == '"0"'
    assert yaml_scalar("/dev/cu.usbmodem1") == "/dev/cu.usbmodem1"
