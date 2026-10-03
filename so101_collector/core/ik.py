"""Closed-form forward/inverse kinematics for the 5-DOF SO-101 arm + gripper.

Conventions (base frame: x forward, y left, z up; floor plane at ``z = 0``):

* ``shoulder_pan``   - rotation of the arm plane about z (0 deg = +x).
* ``shoulder_lift``  - absolute angle of the upper arm above horizontal,
  measured inside the arm plane (positive = up).
* ``elbow_flex``     - relative rotation from upper arm to forearm
  (positive = counter-clockwise inside the plane).
* ``wrist_flex``     - relative rotation from forearm to tool
  (positive = counter-clockwise inside the plane).
* ``wrist_roll``     - rotation about the tool axis (does not move the tip).

The tool direction inside the plane is ``q_lift + q_elbow + q_wrist``; the
default grasp approach points the tool straight down
(``tool_pitch_deg = -90``), which is what floor pickup requires.

The solver is intentionally analytic (no iterative numerical IK): it runs in
constant time inside the 30 Hz control loop and is fully unit-testable via
FK/IK round trips.
"""

from __future__ import annotations

import math

import numpy as np

from so101_collector.config import KinematicsConfig
from so101_collector.exceptions import IKError

_EPS = 1e-9


class SO101Kinematics:
    """Analytical FK/IK plus the software workspace (floor) clamp."""

    def __init__(self, config: KinematicsConfig | None = None) -> None:
        self._cfg = config or KinematicsConfig()

    @property
    def config(self) -> KinematicsConfig:
        return self._cfg

    # -- forward kinematics ------------------------------------------------
    def forward(self, joints_deg: np.ndarray | list[float]) -> np.ndarray:
        """Tool-tip position ``[x, y, z]`` (mm) for the given joint angles."""
        q = np.asarray(joints_deg, dtype=np.float64).reshape(-1)
        if q.size < 5:
            raise ValueError(f"expected at least 5 joint angles, got {q.size}")
        cfg = self._cfg
        pan, lift, elbow, wrist, _roll = q[:5]

        plane_angle = math.radians(pan)
        a_lift = math.radians(lift)
        a_forearm = a_lift + math.radians(elbow)
        a_tool = a_forearm + math.radians(wrist)

        u = (
            cfg.upper_arm_mm * math.cos(a_lift)
            + cfg.forearm_mm * math.cos(a_forearm)
            + cfg.tool_length_mm * math.cos(a_tool)
        )
        w = (
            cfg.upper_arm_mm * math.sin(a_lift)
            + cfg.forearm_mm * math.sin(a_forearm)
            + cfg.tool_length_mm * math.sin(a_tool)
        )
        return np.array(
            [u * math.cos(plane_angle), u * math.sin(plane_angle), cfg.shoulder_height_mm + w],
            dtype=np.float64,
        )

    # -- inverse kinematics ------------------------------------------------
    def inverse(
        self,
        target_mm: np.ndarray | list[float],
        *,
        wrist_roll_deg: float = 0.0,
        tool_pitch_deg: float = -90.0,
        elbow: str = "up",
    ) -> np.ndarray:
        """Solve joint angles for a tool-tip target.

        Returns ``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
        wrist_roll]`` in degrees (the gripper is independent of pose and must
        be supplied by the caller).

        Raises :class:`IKError` when the target is outside the reachable
        annulus - run :meth:`clamp_workspace` first for a guaranteed solve.
        """
        if elbow not in ("up", "down"):
            raise ValueError(f"elbow must be 'up' or 'down', got {elbow!r}")
        p = np.asarray(target_mm, dtype=np.float64).reshape(-1)
        if p.size != 3:
            raise ValueError(f"expected a 3D target, got shape {p.shape}")
        if not np.all(np.isfinite(p)):
            raise IKError(f"target contains NaN/inf: {p}")

        cfg = self._cfg
        radius = math.hypot(p[0], p[1])
        pan = math.atan2(p[1], p[0]) if radius > _EPS else 0.0

        # The wrist sits tool_length_mm back along the tool axis from the
        # tip: wrist = tip - Lt * (cos(theta), sin(theta)) in the arm plane,
        # where theta = tool_pitch_deg (absolute angle from horizontal).
        # For a downward tool (theta = -90) this yields wrist_u = radius and
        # wrist_w = (z - h) + Lt.
        pitch = math.radians(tool_pitch_deg)
        wrist_u = radius - cfg.tool_length_mm * math.cos(pitch)
        wrist_w = (p[2] - cfg.shoulder_height_mm) - cfg.tool_length_mm * math.sin(pitch)

        l1, l2 = cfg.upper_arm_mm, cfg.forearm_mm
        distance = math.hypot(wrist_u, wrist_w)
        if distance > l1 + l2 + _EPS:
            raise IKError(
                f"target out of reach: wrist distance {distance:.1f} mm > max {l1 + l2:.1f} mm"
            )
        if distance < abs(l1 - l2) - _EPS:
            raise IKError(
                f"target too close to shoulder: {distance:.1f} mm < min {abs(l1 - l2):.1f} mm"
            )
        if distance < _EPS:
            raise IKError("target wrist position coincides with the shoulder")

        cos_elbow = (l1 * l1 + l2 * l2 - distance * distance) / (2.0 * l1 * l2)
        cos_shoulder = (l1 * l1 + distance * distance - l2 * l2) / (2.0 * l1 * distance)
        interior_elbow = math.degrees(math.acos(float(np.clip(cos_elbow, -1.0, 1.0))))
        shoulder_offset = math.degrees(math.acos(float(np.clip(cos_shoulder, -1.0, 1.0))))

        sight_angle = math.degrees(math.atan2(wrist_w, wrist_u))
        if elbow == "up":
            lift = sight_angle + shoulder_offset
            elbow_flex = -(180.0 - interior_elbow)
        else:
            lift = sight_angle - shoulder_offset
            elbow_flex = 180.0 - interior_elbow

        wrist_flex = tool_pitch_deg - (lift + elbow_flex)
        return np.array([math.degrees(pan), lift, elbow_flex, wrist_flex, wrist_roll_deg])

    # -- workspace enforcement ---------------------------------------------
    def clamp_workspace(self, position_mm: np.ndarray | list[float]) -> np.ndarray:
        """Clamp a tool-tip target to the safe software workspace.

        Enforces the specification's floor rule (the gripper tip stops
        ``floor_clearance_mm`` - 2 mm - above the floor instead of relying on
        a physical motor stall) plus reachability limits, so
        :meth:`inverse` always succeeds afterwards.
        """
        cfg = self._cfg
        p = np.array(position_mm, dtype=np.float64).reshape(-1).copy()
        if p.size != 3 or not np.all(np.isfinite(p)):
            raise ValueError(f"expected a finite 3D position, got {p}")

        # 1) Floor: tip never descends below the software floor stop.
        p[2] = max(p[2], cfg.floor_clearance_mm)

        # 2) Ceiling: with the tool pointing down the wrist must stay within
        #    the arm's sphere, so tip_z + tool <= shoulder + L1 + L2.
        max_wrist = cfg.upper_arm_mm + cfg.forearm_mm - cfg.reach_margin_mm
        p[2] = min(p[2], cfg.shoulder_height_mm + max_wrist - cfg.tool_length_mm)

        # 3) Radial: keep the wrist inside the reachable circle.
        wrist_w = (p[2] - cfg.shoulder_height_mm) - cfg.tool_length_mm * math.sin(
            math.radians(-90.0)
        )
        radius = math.hypot(p[0], p[1])
        max_radius_sq = max_wrist * max_wrist - wrist_w * wrist_w
        if max_radius_sq <= 0.0:
            p[0] = p[1] = 0.0
        elif radius * radius > max_radius_sq:
            scale = math.sqrt(max_radius_sq) / radius
            p[0] *= scale
            p[1] *= scale
        return p
