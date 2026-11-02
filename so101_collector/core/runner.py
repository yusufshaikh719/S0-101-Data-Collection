"""Main execution engine: episode lifecycle + precision 30 Hz control loop.

Wires the four pillars of the architecture together::

    DataCollectionRunner
      |- RobotBus      (SO101HardwareInterface: read state, write actions)
      |- VisionHub     (CameraHub: background camera threads)
      |- Teleop        (leader arm or SpaceMouse/gamepad + analytical IK)
      |- DatasetWriter (in-memory buffer -> atomic MP4/Parquet commit)

Loop timing uses the drift-compensating :class:`~so101_collector.timing.LoopTimer`
(absolute monotonic deadlines, never a naive ``sleep(0.033)``).

Episode state machine::

    IDLE --START--> WARMUP (2 s visual stabilization, frames not buffered)
    WARMUP --(timeout)--> RECORDING (buffered at 30 Hz)
    RECORDING --FINISH--> save -> IDLE
    RECORDING --ABORT--> discard -> IDLE      (RAM cleared, disk untouched)
    any --RESET--> arm drives to home pose (recording continues if active)
    any --QUIT--> save nothing, tear down
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from so101_collector.console import Command, CommandReader
from so101_collector.core.camera_hub import CameraHub
from so101_collector.core.dataset_writer import DatasetWriter, FrameSample
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.ik import SO101Kinematics
from so101_collector.core.teleop_controller import TeleopSource
from so101_collector.core.visualizer import NullVisualizer, Visualizer
from so101_collector.exceptions import CameraError, DatasetWriteError, EpisodeStateError
from so101_collector.timing import LoopTimer

logger = logging.getLogger(__name__)

DEFAULT_TASK = "pick up the object from the floor and hold it in the air"


class RunnerState(Enum):
    IDLE = "idle"
    WARMUP = "warmup"
    RECORDING = "recording"


@dataclass
class RunnerStats:
    """Session outcome, returned by :meth:`DataCollectionRunner.run`."""

    episodes_saved: int = 0
    episodes_discarded: int = 0
    frames_recorded: int = 0
    duration_s: float = 0.0
    final_state: str = RunnerState.IDLE.value
    interrupted: bool = False
    timer_ticks: int = 0
    timer_overruns: int = 0
    mean_loop_fps: float = 0.0
    max_sync_drift_ms: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def episodes_completed(self) -> int:
        return self.episodes_saved + self.episodes_discarded

    def summary(self) -> str:
        return (
            f"episodes saved: {self.episodes_saved}  "
            f"discarded: {self.episodes_discarded}  "
            f"frames: {self.frames_recorded}\n"
            f"duration: {self.duration_s:.1f} s  loop: {self.mean_loop_fps:.1f} fps  "
            f"overruns: {self.timer_overruns}/{self.timer_ticks}  "
            f"max sync drift: {self.max_sync_drift_ms:.1f} ms"
        )


class DataCollectionRunner:
    """Orchestrates hardware, vision, teleop, and persistence at 30 Hz."""

    def __init__(
        self,
        *,
        hardware: SO101HardwareInterface,
        camera_hub: CameraHub,
        teleop: TeleopSource,
        writer: DatasetWriter,
        commands: CommandReader,
        timer: LoopTimer | None = None,
        fps: float = 30.0,
        warmup_s: float = 2.0,
        task: str = DEFAULT_TASK,
        stats_log_interval_s: float = 5.0,
        visualizer: Visualizer | None = None,
        kinematics: SO101Kinematics | None = None,
    ) -> None:
        if warmup_s < 0:
            raise ValueError(f"warmup_s must be >= 0, got {warmup_s}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self._hardware = hardware
        self._camera_hub = camera_hub
        self._teleop = teleop
        self._writer = writer
        self._commands = commands
        self._timer = timer if timer is not None else LoopTimer(fps)
        self._fps = fps
        self._warmup_s = warmup_s
        self._task = task
        self._stats_log_interval_s = stats_log_interval_s
        self._visualizer = visualizer if visualizer is not None else NullVisualizer()
        self._kinematics = kinematics

    # -- public API --------------------------------------------------------
    def run(
        self, *, max_episodes: int | None = None, max_seconds: float | None = None
    ) -> RunnerStats:
        """Run the collection loop until QUIT/limits/Ctrl-C; returns stats."""
        stats = RunnerStats()
        started_at = time.monotonic()
        step_index = 0
        last_action: np.ndarray | None = None
        last_dt = self._timer.target_dt
        state = RunnerState.IDLE
        warmup_started_at = 0.0
        last_stats_log = started_at
        last_camera_error_log = 0.0

        self._hardware.open()
        self._camera_hub.start()
        self._teleop.open()
        self._timer.start()
        self._visualizer.start()
        self._visualizer.log_event("session_start", f"Task: {self._task}")
        logger.info(
            "session started (fps=%.0f, warmup=%.1fs, task=%r)",
            self._fps,
            self._warmup_s,
            self._task,
        )

        try:
            while True:
                # -- session limits -------------------------------------
                if max_seconds is not None and time.monotonic() - started_at >= max_seconds:
                    logger.info("max session duration (%.0f s) reached", max_seconds)
                    break
                if max_episodes is not None and stats.episodes_completed >= max_episodes:
                    logger.info("episode limit (%d) reached", max_episodes)
                    break

                # -- commands (edge-triggered) --------------------------
                command = self._commands.read()
                if command is Command.QUIT:
                    logger.info("quit requested")
                    self._visualizer.log_event("command_quit", "Operator requested quit")
                    break
                if command is Command.RESET:
                    self._hardware.go_home()
                    logger.info("reset: driving to home pose")
                    self._visualizer.log_event("command_reset", "Driving to home pose")
                elif command is Command.START:
                    if state is RunnerState.IDLE:
                        state = RunnerState.WARMUP
                        warmup_started_at = time.monotonic()
                        logger.info("episode warmup started (%.1f s)", self._warmup_s)
                        self._visualizer.log_event("command_start", f"Warmup started ({self._warmup_s:.1f} s)")
                elif command is Command.FINISH:
                    state = self._finish(state, stats)
                elif command is Command.ABORT:
                    state = self._abort(state, stats)

                # -- warmup -> recording transition ---------------------
                if state is RunnerState.WARMUP and (
                    time.monotonic() - warmup_started_at
                ) >= self._warmup_s:
                    # Gate on live cameras: never start buffering frames we
                    # cannot pair with video.
                    try:
                        self._camera_hub.get_synced_frames()
                    except CameraError as exc:
                        now = time.monotonic()
                        if now - last_camera_error_log >= 2.0:
                            last_camera_error_log = now
                            logger.warning("cannot start episode, cameras not ready: %s", exc)
                    else:
                        episode_index = self._writer.start_episode(self._task)
                        state = RunnerState.RECORDING
                        logger.info("recording episode %06d", episode_index)
                        self._visualizer.log_event("episode_started", f"Recording episode {episode_index:06d}")

                # -- acquisition tick (teleop drives the arm every state) -
                step_index += 1
                tick_started = time.monotonic()
                measured = self._hardware.read_state()
                action = self._teleop.read(last_dt)
                if action is None:
                    action = (
                        last_action
                        if last_action is not None
                        else np.asarray(self._hardware.config.home_pose, dtype=np.float32)
                    )
                applied = self._hardware.write_positions(action)
                last_action = applied

                base_frame: np.ndarray | None = None
                wrist_frame: np.ndarray | None = None
                dt_sync_ms: float | None = None

                if state is RunnerState.RECORDING:
                    state, base_frame, wrist_frame, dt_sync_s = self._capture(
                        state, stats, tick_started, measured.positions, applied
                    )
                    if dt_sync_s is not None:
                        dt_sync_ms = dt_sync_s * 1000.0
                else:
                    try:
                        base_frame = self._camera_hub.get_frame("base").frame
                        wrist_frame = self._camera_hub.get_frame("wrist").frame
                    except Exception:
                        pass

                self._visualizer.log_step(
                    session_time_s=tick_started - started_at,
                    step_index=step_index,
                    state=state.value,
                    measured_positions=measured.positions,
                    applied_action=applied,
                    base_rgb=base_frame,
                    wrist_rgb=wrist_frame,
                    dt_sync_ms=dt_sync_ms,
                    loop_fps=self._timer.summary().mean_fps,
                    episode_index=self._writer.active_episode_index,
                    episode_frame_index=self._writer.frame_count if self._writer.episode_active else None,
                    kinematics=self._kinematics,
                )

                # -- drift-compensated pacing ---------------------------
                tick = self._timer.wait()
                last_dt = tick.period_s

                if tick_started - last_stats_log >= self._stats_log_interval_s:
                    last_stats_log = tick_started
                    self._log_progress()

        except KeyboardInterrupt:
            stats.interrupted = True
            logger.info("interrupted by user (Ctrl-C)")
        finally:
            # Crash/quit safety: never leave a RAM-only episode dangling.
            if self._writer.episode_active:
                self._writer.discard_episode()
                stats.episodes_discarded += 1
            self._timer.stop()
            self._visualizer.close()
            timer_summary = self._timer.summary()
            stats.timer_ticks = timer_summary.ticks
            stats.timer_overruns = timer_summary.overruns
            stats.mean_loop_fps = timer_summary.mean_fps
            hub_stats = self._camera_hub.stats()
            stats.max_sync_drift_ms = hub_stats.sync_max_ms
            stats.final_state = state.value
            stats.duration_s = time.monotonic() - started_at
            self._camera_hub.stop()
            self._teleop.close()
            self._hardware.close()
            self._commands.close()
            logger.info("session finished\n%s", stats.summary())
        return stats

    # -- helpers -----------------------------------------------------------
    def _finish(self, state: RunnerState, stats: RunnerStats) -> RunnerState:
        if state is RunnerState.RECORDING:
            try:
                self._writer.save_episode(self._task, success=True)
            except (EpisodeStateError, DatasetWriteError) as exc:
                # Buffer is retained on failure: stay in RECORDING so the
                # operator can press FINISH again to retry.
                logger.error("save failed: %s", exc)
                return state
            stats.episodes_saved += 1
            self._visualizer.log_event("episode_saved", f"Episode {stats.episodes_saved} saved")
            return RunnerState.IDLE
        if state is RunnerState.WARMUP:
            logger.info("warmup cancelled")
            return RunnerState.IDLE
        logger.debug("FINISH ignored while idle")
        return state

    def _abort(self, state: RunnerState, stats: RunnerStats) -> RunnerState:
        if state is RunnerState.RECORDING:
            self._writer.discard_episode()
            stats.episodes_discarded += 1
            self._visualizer.log_event("episode_aborted", "Episode discarded")
        elif state is RunnerState.WARMUP:
            logger.info("warmup cancelled")
        else:
            logger.debug("ABORT ignored while idle")
        return RunnerState.IDLE

    def _capture(
        self,
        state: RunnerState,
        stats: RunnerStats,
        tick_started: float,
        measured_positions: np.ndarray,
        applied_action: np.ndarray,
    ) -> tuple[RunnerState, np.ndarray | None, np.ndarray | None, float | None]:
        try:
            base_frame, wrist_frame, dt_sync = self._camera_hub.get_synced_frames()
        except CameraError as exc:
            # Losing video mid-episode would desynchronize state from pixels;
            # drop the episode rather than write inconsistent training data.
            logger.error("camera failure mid-episode, discarding: %s", exc)
            self._writer.discard_episode()
            self._visualizer.log_event("camera_error", str(exc))
            stats.episodes_discarded += 1
            return RunnerState.IDLE, None, None, None
        self._writer.append(
            FrameSample(
                timestamp=tick_started,
                state=np.asarray(measured_positions, dtype=np.float32),
                action=np.asarray(applied_action, dtype=np.float32),
                base=base_frame,
                wrist=wrist_frame,
            )
        )
        stats.frames_recorded += 1
        return state, base_frame, wrist_frame, dt_sync

    def _log_progress(self) -> None:
        timer_summary = self._timer.summary()
        hub_stats = self._camera_hub.stats()
        camera_bits = ", ".join(
            f"{name}: {data['fps']} fps" for name, data in hub_stats.cameras.items()
        )
        logger.info(
            "loop %.1f fps | overruns %d/%d (max %.1f ms) | sync mean %.1f ms max %.1f ms | "
            "buffer %d frames | %s",
            timer_summary.mean_fps,
            timer_summary.overruns,
            timer_summary.ticks,
            timer_summary.max_overrun_s * 1000.0,
            hub_stats.sync_mean_ms,
            hub_stats.sync_max_ms,
            self._writer.frame_count,
            camera_bits,
        )
