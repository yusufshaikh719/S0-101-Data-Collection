"""Shared fixtures: hardware/vision configs with tiny frames for fast tests."""

from __future__ import annotations

import numpy as np
import pytest

from so101_collector.config import (
    CameraConfig,
    JointSpec,
    RobotConfig,
    VisionConfig,
)


def make_robot_config(**overrides) -> RobotConfig:
    """Robot config with persistent-path-valid test values."""
    kwargs = dict(
        port="/dev/serial/by-id/usb-SO101_TEST_adapter-if00-port0",
        joints=(
            # -180 deg offsets model a servo horn centered on joint zero so
            # every joint angle maps inside the 0..4095 encoder window.
            JointSpec("shoulder_pan", 1, -135.0, 135.0, 0.0, offset_deg=-178.5),
            JointSpec("shoulder_lift", 2, -90.0, 120.0, 45.0, offset_deg=-180.0),
            JointSpec("elbow_flex", 3, -135.0, 135.0, -60.0, offset_deg=-180.0),
            JointSpec("wrist_flex", 4, -100.0, 100.0, 15.0, offset_deg=-180.0),
            JointSpec("wrist_roll", 5, -160.0, 160.0, 0.0, offset_deg=-180.0),
            JointSpec("gripper", 6, 0.0, 60.0, 60.0, offset_deg=-180.0),
        ),
        max_retries=3,
        retry_backoff_s=0.0,
    )
    kwargs.update(overrides)
    return RobotConfig(**kwargs)


def make_vision_config(
    *, width: int = 64, height: int = 48, sync_tolerance_ms: float = 20.0
) -> VisionConfig:
    return VisionConfig(
        cameras=(
            CameraConfig(
                name="base",
                device_path="/dev/v4l/by-id/usb-SO101_TEST_BASE-video-index0",
                width=width,
                height=height,
            ),
            CameraConfig(
                name="wrist",
                device_path="/dev/v4l/by-id/usb-SO101_TEST_WRIST-video-index0",
                width=width,
                height=height,
            ),
        ),
        sync_tolerance_ms=sync_tolerance_ms,
    )


@pytest.fixture
def robot_config() -> RobotConfig:
    return make_robot_config()


@pytest.fixture
def vision_config() -> VisionConfig:
    return make_vision_config()


@pytest.fixture
def frame_factory():
    """Build a deterministic uint8 RGB test frame of a given shape."""

    def _make(shape=(48, 64), fill: int = 128) -> np.ndarray:
        return np.full((*shape, 3), fill, dtype=np.uint8)

    return _make
