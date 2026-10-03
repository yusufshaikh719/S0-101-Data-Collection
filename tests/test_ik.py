"""Analytical IK tests: FK/IK round trips, workspace clamps, error paths."""

from __future__ import annotations

import math

import numpy as np
import pytest

from so101_collector.config import KinematicsConfig
from so101_collector.core.ik import SO101Kinematics
from so101_collector.exceptions import IKError


@pytest.fixture
def kinematics() -> SO101Kinematics:
    return SO101Kinematics(KinematicsConfig())


def _targets() -> list[np.ndarray]:
    grid = []
    for radius in range(60, 231, 30):
        for z in (2.0, 40.0, 120.0, 240.0):
            for angle_deg in (0, 45, -100, 160):
                grid.append(
                    np.array(
                        [
                            radius * math.cos(math.radians(angle_deg)),
                            radius * math.sin(math.radians(angle_deg)),
                            z,
                        ]
                    )
                )
    return grid


@pytest.mark.parametrize("elbow", ["up", "down"])
def test_fk_ik_roundtrip_within_workspace(kinematics, elbow):
    """IK followed by FK must reproduce the clamped target to <1 um."""
    worst = 0.0
    for target in _targets():
        clamped = kinematics.clamp_workspace(target)
        joints = kinematics.inverse(clamped, elbow=elbow)
        recovered = kinematics.forward(joints)
        worst = max(worst, float(np.linalg.norm(recovered - clamped)))
    assert worst < 1e-6


def test_inverse_defaults_to_downward_tool(kinematics):
    joints = kinematics.inverse([150.0, 0.0, 40.0])
    tip_angle = joints[1] + joints[2] + joints[3]
    assert tip_angle == pytest.approx(-90.0, abs=1e-9)  # tool points at floor


def test_forward_matches_known_geometry(kinematics):
    # Straight arm pointing up: tip = shoulder + L1 + L2 + tool along z.
    cfg = kinematics.config
    tip = kinematics.forward([0.0, 90.0, 0.0, 0.0, 0.0])
    expected_z = cfg.shoulder_height_mm + cfg.upper_arm_mm + cfg.forearm_mm + cfg.tool_length_mm
    assert np.linalg.norm(tip - np.array([0.0, 0.0, expected_z])) < 1e-9


def test_floor_clamp_enforces_software_floor_stop(kinematics):
    clamped = kinematics.clamp_workspace([100.0, 50.0, -30.0])
    assert clamped[2] == pytest.approx(kinematics.config.floor_clearance_mm)
    # The whole floor plane clamps to exactly the clearance height.
    for x, y in ((0.0, 0.0), (200.0, 0.0), (10.0, -10.0)):
        assert kinematics.clamp_workspace([x, y, -1e6])[2] == pytest.approx(2.0)


def test_radial_clamp_keeps_target_reachable(kinematics):
    clamped = kinematics.clamp_workspace([900.0, 0.0, 50.0])
    assert np.hypot(clamped[0], clamped[1]) < 400.0  # pulled well inside
    joints = kinematics.inverse(clamped)  # must not raise
    assert joints.shape == (5,)


def test_ceiling_clamp_keeps_target_reachable(kinematics):
    clamped = kinematics.clamp_workspace([0.0, 0.0, 5000.0])
    assert clamped[2] < kinematics.config.shoulder_height_mm + 1000.0
    kinematics.inverse(clamped)  # must not raise


def test_inverse_raises_beyond_reach_without_clamp(kinematics):
    with pytest.raises(IKError, match="out of reach"):
        kinematics.inverse([5000.0, 0.0, 0.0])


def test_inverse_raises_inside_dead_zone(kinematics):
    # Wrist distance below |L1 - L2| cannot be solved.
    cfg = kinematics.config
    min_reach = abs(cfg.upper_arm_mm - cfg.forearm_mm)
    # Place the wrist 10 mm from the shoulder: for a downward tool the
    # wrist sits tool_length above the tip, so z = h - tool + 10.
    z = cfg.shoulder_height_mm - cfg.tool_length_mm + 10.0
    assert 10.0 < min_reach
    with pytest.raises(IKError, match="too close"):
        kinematics.inverse([0.0, 0.0, z])


def test_inverse_validates_arguments(kinematics):
    with pytest.raises(ValueError, match="elbow"):
        kinematics.inverse([100.0, 0.0, 50.0], elbow="sideways")
    with pytest.raises(ValueError, match="3D target"):
        kinematics.inverse([1.0, 2.0])
    with pytest.raises(IKError, match="NaN"):
        kinematics.inverse([np.nan, 0.0, 50.0])


def test_forward_validates_input(kinematics):
    with pytest.raises(ValueError, match="at least 5"):
        kinematics.forward([0.0, 1.0])


def test_clamp_workspace_validates_input(kinematics):
    with pytest.raises(ValueError, match="finite"):
        kinematics.clamp_workspace([np.inf, 0.0, 1.0])
