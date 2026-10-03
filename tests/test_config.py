"""Config loading tests: shipped YAMLs, strict validation, path policy."""

from __future__ import annotations

import pytest

from so101_collector.config import (
    JOINT_NAMES,
    CameraConfig,
    JointSpec,
    RobotConfig,
    VisionConfig,
    default_config_dir,
)
from so101_collector.exceptions import ConfigError
from tests.conftest import make_robot_config, make_vision_config


# ---------------------------------------------------------------------------
# Shipped configuration files
# ---------------------------------------------------------------------------
def test_shipped_robot_config_loads():
    config = RobotConfig.from_yaml(default_config_dir() / "robot_config.yaml")
    assert config.joint_names == JOINT_NAMES
    assert config.baudrate == 1_000_000
    assert config.timeout_s == pytest.approx(0.05)
    assert config.max_retries == 3
    assert config.port.startswith("/dev/serial/by-id/")
    assert config.kinematics.floor_clearance_mm == pytest.approx(2.0)
    assert len(config.home_pose) == 6


def test_shipped_camera_config_loads():
    vision = VisionConfig.from_yaml(default_config_dir() / "camera_config.yaml")
    assert set(vision.by_name) == {"base", "wrist"}
    assert vision.sync_tolerance_ms == pytest.approx(20.0)
    for camera in vision.cameras:
        assert camera.width == 640 and camera.height == 480
        assert camera.fps == pytest.approx(30.0)
        assert camera.fourcc == "MJPG"
        assert camera.device_path.startswith("/dev/v4l/by-id/")
        assert camera.frame_shape == (480, 640)


# ---------------------------------------------------------------------------
# Zero index-drift policy
# ---------------------------------------------------------------------------
def test_dynamic_serial_port_rejected():
    with pytest.raises(ConfigError, match="dynamic device path"):
        make_robot_config(port="/dev/ttyUSB0")
    # Explicit opt-out is allowed for bench testing.
    make_robot_config(port="/dev/ttyUSB0", allow_dynamic_port=True)


def test_dynamic_camera_path_rejected():
    with pytest.raises(ConfigError, match="dynamic device path"):
        CameraConfig(name="base", device_path="/dev/video0")
    CameraConfig(name="base", device_path="/dev/video0", allow_dynamic_path=True)


def test_persistent_paths_accepted():
    make_robot_config(port="/dev/serial/by-path/pci-usb-0:1:1.0-port0")
    CameraConfig(name="base", device_path="/dev/v4l/by-path/pci-usb-video0")


# ---------------------------------------------------------------------------
# Strict validation
# ---------------------------------------------------------------------------
def test_unknown_top_level_key_rejected():
    payload = make_robot_config().to_dict()
    payload["baudrates"] = 115200  # typo'd key
    with pytest.raises(ConfigError, match="unknown key"):
        RobotConfig.from_dict(payload)


def test_unknown_joint_key_rejected():
    payload = make_robot_config().to_dict()
    payload["joints"][0]["maxangle_deg"] = 90
    with pytest.raises(ConfigError, match="unknown key"):
        RobotConfig.from_dict(payload)


def test_joint_order_must_match_canonical_names():
    payload = make_robot_config().to_dict()
    payload["joints"][0], payload["joints"][1] = payload["joints"][1], payload["joints"][0]
    with pytest.raises(ConfigError, match="joint order"):
        RobotConfig.from_dict(payload)


def test_duplicate_servo_ids_rejected():
    payload = make_robot_config().to_dict()
    payload["joints"][1]["servo_id"] = payload["joints"][0]["servo_id"]
    with pytest.raises(ConfigError, match="duplicate servo ids"):
        RobotConfig.from_dict(payload)


def test_home_pose_outside_limits_rejected():
    with pytest.raises(ConfigError, match="home_deg"):
        JointSpec("gripper", 6, 0.0, 60.0, home_deg=90.0)


def test_min_max_must_be_ordered():
    with pytest.raises(ConfigError, match="min_deg"):
        JointSpec("gripper", 6, 60.0, 0.0, 30.0)


def test_missing_required_key_rejected():
    payload = make_robot_config().to_dict()
    del payload["port"]
    with pytest.raises(ConfigError, match="missing required key 'port'"):
        RobotConfig.from_dict(payload)


def test_missing_camera_rejected():
    vision = make_vision_config()
    with pytest.raises(ConfigError, match="wrist"):
        VisionConfig(cameras=(vision.cameras[0],))


def test_duplicate_camera_names_rejected():
    vision = make_vision_config()
    with pytest.raises(ConfigError, match="duplicate camera"):
        VisionConfig(cameras=(vision.cameras[0], vision.cameras[0]))


def test_camera_fourcc_must_be_four_chars():
    with pytest.raises(ConfigError, match="fourcc"):
        CameraConfig(name="base", device_path="/dev/v4l/by-id/x", fourcc="MJPEG")


def test_missing_config_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        RobotConfig.from_yaml(tmp_path / "nope.yaml")


def test_empty_config_file_raises_config_error(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        RobotConfig.from_yaml(path)


def test_wrong_type_for_numeric_rejected():
    payload = make_robot_config().to_dict()
    payload["timeout_s"] = "fast"
    with pytest.raises(ConfigError, match="expected a number"):
        RobotConfig.from_dict(payload)


def test_joint_clamp_respects_limits():
    spec = JointSpec("elbow_flex", 3, -90.0, 90.0, 0.0)
    assert spec.clamp(-500.0) == -90.0
    assert spec.clamp(500.0) == 90.0
    assert spec.clamp(12.5) == 12.5
