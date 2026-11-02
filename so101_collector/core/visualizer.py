"""Optional Rerun visualization for the control loop.

``NullVisualizer`` is a no-op used when visualization is disabled. The Rerun
backend publishes camera feeds, joint telemetry and a 3D arm sketch; logging
errors (for example a closed viewer) are caught so they cannot stop capture.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from so101_collector.config import JOINT_NAMES
from so101_collector.core.ik import SO101Kinematics

logger = logging.getLogger(__name__)


class Visualizer(Protocol):
    """Structural interface for data collection visualizers."""

    def start(self) -> None: ...

    def close(self) -> None: ...

    def log_step(
        self,
        *,
        session_time_s: float,
        step_index: int,
        state: str,
        measured_positions: np.ndarray | Sequence[float] | None = None,
        applied_action: np.ndarray | Sequence[float] | None = None,
        base_rgb: np.ndarray | None = None,
        wrist_rgb: np.ndarray | None = None,
        dt_sync_ms: float | None = None,
        loop_fps: float | None = None,
        episode_index: int | None = None,
        episode_frame_index: int | None = None,
        kinematics: SO101Kinematics | None = None,
    ) -> None: ...

    def log_event(self, name: str, message: str) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None: ...


class NullVisualizer:
    """No-op visualizer used when live visualization is disabled."""

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def log_step(
        self,
        *,
        session_time_s: float,
        step_index: int,
        state: str,
        measured_positions: np.ndarray | Sequence[float] | None = None,
        applied_action: np.ndarray | Sequence[float] | None = None,
        base_rgb: np.ndarray | None = None,
        wrist_rgb: np.ndarray | None = None,
        dt_sync_ms: float | None = None,
        loop_fps: float | None = None,
        episode_index: int | None = None,
        episode_frame_index: int | None = None,
        kinematics: SO101Kinematics | None = None,
    ) -> None:
        pass

    def log_event(self, name: str, message: str) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        pass


class RerunVisualizer:
    """Live Rerun viewer with camera feeds, joint plots and a 3D arm sketch."""

    def __init__(
        self,
        *,
        application_id: str = "so101_collector",
        recording_id: str | None = None,
        spawn: bool = True,
        connect_addr: str | None = None,
        save_path: str | Path | None = None,
        publish_interval_steps: int = 1,
        base_entity: str = "world/base_camera",
        wrist_entity: str = "world/robot/wrist/wrist_camera",
    ) -> None:
        self._application_id = application_id
        self._recording_id = recording_id
        self._spawn = spawn
        self._connect_addr = connect_addr
        self._save_path = Path(save_path) if save_path is not None else None
        self._publish_interval = max(1, int(publish_interval_steps))
        self._base_entity = base_entity
        self._wrist_entity = wrist_entity
        self._rr = None
        self._active = False
        self._logged_error_once = False

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self) -> None:
        try:
            import rerun as rr
        except ImportError as exc:
            raise ImportError(
                "rerun-sdk is required for visualization; "
                "install it with pip install 'so101-collector[visualize]'"
            ) from exc

        self._rr = rr
        try:
            rr.init(
                self._application_id,
                recording_id=self._recording_id,
                spawn=self._spawn and not self._connect_addr and not self._save_path,
            )
            if self._connect_addr:
                rr.connect(self._connect_addr)
                logger.info("connected to external Rerun viewer at %s", self._connect_addr)
            elif self._save_path:
                self._save_path.parent.mkdir(parents=True, exist_ok=True)
                rr.save(str(self._save_path))
                logger.info("saving Rerun recording stream to %s", self._save_path)
            elif self._spawn:
                logger.info("spawned local Rerun viewer")

            rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

            # Outline of the 400 x 400 mm pick area at floor level (metres).
            half = 0.25
            floor_quad = [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
                [-half, -half, 0.0],
            ]
            rr.log("world/workspace_boundary", rr.LineStrips3D([floor_quad]), static=True)

            self._active = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to initialize Rerun visualizer: %s", exc)
            self._active = False

    def close(self) -> None:
        self._active = False

    def log_step(
        self,
        *,
        session_time_s: float,
        step_index: int,
        state: str,
        measured_positions: np.ndarray | Sequence[float] | None = None,
        applied_action: np.ndarray | Sequence[float] | None = None,
        base_rgb: np.ndarray | None = None,
        wrist_rgb: np.ndarray | None = None,
        dt_sync_ms: float | None = None,
        loop_fps: float | None = None,
        episode_index: int | None = None,
        episode_frame_index: int | None = None,
        kinematics: SO101Kinematics | None = None,
    ) -> None:
        if not self._active or self._rr is None:
            return
        if step_index % self._publish_interval != 0:
            return

        rr = self._rr
        try:
            rr.set_time("session_time", duration=max(0.0, float(session_time_s)))
            rr.set_time("step", sequence=int(step_index))
            if episode_index is not None:
                rr.set_time("episode", sequence=int(episode_index))
            if episode_frame_index is not None:
                rr.set_time("episode_frame", sequence=int(episode_frame_index))

            # 0.0 idle, 0.5 warmup, 1.0 recording
            state_code = 1.0 if state == "recording" else (0.5 if state == "warmup" else 0.0)
            rr.log("episode/active", rr.Scalars(state_code))

            if base_rgb is not None:
                rr.log(f"{self._base_entity}/rgb", rr.Image(base_rgb))
            if wrist_rgb is not None:
                rr.log(f"{self._wrist_entity}/rgb", rr.Image(wrist_rgb))

            if measured_positions is not None:
                meas = np.asarray(measured_positions, dtype=np.float32).ravel()
                act = (
                    np.asarray(applied_action, dtype=np.float32).ravel()
                    if applied_action is not None
                    else meas
                )
                for i, name in enumerate(JOINT_NAMES[: len(meas)]):
                    m_val = float(meas[i])
                    a_val = float(act[i]) if i < len(act) else m_val
                    rr.log(f"robot/joints/{name}/state", rr.Scalars(m_val))
                    rr.log(f"robot/joints/{name}/action", rr.Scalars(a_val))
                    rr.log(f"robot/joints/{name}/error", rr.Scalars(a_val - m_val))

                if kinematics is not None and len(meas) >= 5:
                    chain = kinematics.forward_chain(meas[:5])
                    # Millimetres to metres for the viewport.
                    pts_m = [
                        chain[k] / 1000.0
                        for k in ("base", "shoulder", "elbow", "wrist", "tool")
                    ]
                    rr.log("world/robot/skeleton", rr.LineStrips3D([pts_m]))
                    rr.log(
                        "world/robot/joints_3d",
                        rr.Points3D(
                            pts_m,
                            radii=[0.012, 0.010, 0.009, 0.008, 0.006],
                            colors=[
                                [50, 150, 255],
                                [80, 200, 120],
                                [80, 200, 120],
                                [255, 180, 50],
                                [255, 80, 80],
                            ],
                        ),
                    )
                    rr.log(
                        "world/robot/wrist",
                        rr.Transform3D(translation=chain["wrist"] / 1000.0),
                    )

            if dt_sync_ms is not None:
                rr.log("telemetry/camera_sync_drift_ms", rr.Scalars(float(dt_sync_ms)))
            if loop_fps is not None:
                rr.log("telemetry/loop_fps", rr.Scalars(float(loop_fps)))

        except Exception as exc:  # noqa: BLE001
            if not self._logged_error_once:
                self._logged_error_once = True
                logger.warning("Rerun logging error (further warnings suppressed): %s", exc)

    def log_event(self, name: str, message: str) -> None:
        if not self._active or self._rr is None:
            return
        try:
            self._rr.log(f"events/{name}", self._rr.TextDocument(message))
        except Exception:  # noqa: BLE001
            pass

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        if not self._active or self._rr is None:
            return
        try:
            for key, value in metrics.items():
                self._rr.log(f"metrics/{key}", self._rr.Scalars(float(value)))
        except Exception:  # noqa: BLE001
            pass
