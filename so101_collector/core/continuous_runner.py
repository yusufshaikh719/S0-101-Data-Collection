"""Continuous capture with home-pose episode boundaries."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from so101_collector.console import Command
from so101_collector.core.lerobot_writer import FrameSample
from so101_collector.core.teleop_controller import request_home
from so101_collector.core.visualizer import NullVisualizer
from so101_collector.exceptions import CameraError
from so101_collector.timing import LoopTimer

logger = logging.getLogger(__name__)

DEFAULT_TASK = "pick up the object from the floor and hold it in the air"


class RunnerState(Enum):
    IDLE = "idle"
    WARMUP = "warmup"
    RECORDING = "recording"


@dataclass
class RunnerStats:
    """Session outcome returned by the runner."""

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

    def summary(self) -> str:
        return (
            f"episodes saved: {self.episodes_saved}  "
            f"discarded: {self.episodes_discarded}  "
            f"frames: {self.frames_recorded}\n"
            f"duration: {self.duration_s:.1f} s  loop: {self.mean_loop_fps:.1f} fps  "
            f"overruns: {self.timer_overruns}/{self.timer_ticks}  "
            f"max sync drift: {self.max_sync_drift_ms:.1f} ms"
        )


class ContinuousDataCollectionRunner:
    """Captures every tick and saves an episode when the arm returns home."""

    def __init__(
        self,
        *,
        hardware,
        camera_hub,
        teleop,
        writer,
        commands,
        fps: int = 30,
        warmup_s: float = 0.0,
        task: str = DEFAULT_TASK,
        home_tolerance_deg: float = 5.0,
        home_exit_tolerance_deg: float = 8.0,
        visualizer=None,
        kinematics=None,
        timer=None,
    ) -> None:
        if (
            fps <= 0
            or warmup_s < 0
            or not np.isfinite(home_tolerance_deg)
            or not np.isfinite(home_exit_tolerance_deg)
            or home_tolerance_deg <= 0
            or home_exit_tolerance_deg <= home_tolerance_deg
        ):
            raise ValueError("home exit tolerance must exceed a positive home tolerance")
        self._hardware = hardware
        self._camera_hub = camera_hub
        self._teleop = teleop
        self._writer = writer
        self._commands = commands
        self._fps = fps
        self._warmup_s = warmup_s
        self._task = task
        self._enter = home_tolerance_deg
        self._exit = home_exit_tolerance_deg
        self._visualizer = visualizer or NullVisualizer()
        self._kinematics = kinematics
        self._timer = timer or LoopTimer(fps)

    def run(
        self, *, max_episodes: int | None = None, max_seconds: float | None = None
    ) -> RunnerStats:
        stats = RunnerStats()
        started = time.monotonic()
        home = np.asarray(self._hardware.config.home_pose, dtype=np.float32)
        if not np.isfinite(home).all():
            raise ValueError("configured home pose contains non-finite values")
        armed = False
        has_sample = False
        last_report_frames = 0
        last_action = None
        last_dt = self._timer.target_dt
        state = RunnerState.WARMUP
        last_camera_warning = 0.0
        home_hint_shown = False
        last_report = started
        reset_active = False
        failure = False
        try:
            self._hardware.open()
            self._camera_hub.start()
            self._teleop.open()
            self._visualizer.start()
            self._timer.start()
            self._visualizer.log_event("session_start", f"Task: {self._task}")
            logger.info(
                "continuous recording: home tolerance %.1f deg, re-arm %.1f deg",
                self._enter,
                self._exit,
            )
            while True:
                now = time.monotonic()
                if max_seconds is not None and now - started >= max_seconds:
                    break
                if max_episodes is not None and stats.episodes_saved >= max_episodes:
                    break
                command = self._commands.read()
                if command is Command.QUIT:
                    break
                if command is Command.RESET:
                    if request_home(self._teleop):
                        # The source owns homing: with torque released it has to
                        # power the arm up before anything can move.
                        self._visualizer.log_event("command_reset", "Homing under power")
                    else:
                        reset_active = True
                        self._visualizer.log_event("command_reset", "Driving to home pose")
                elif command is Command.ABORT and self._writer.episode_active:
                    self._writer.discard_episode()
                    stats.episodes_discarded += 1
                    armed = False
                    self._visualizer.log_event("episode_aborted", "Episode discarded")
                elif (
                    command is Command.FINISH
                    and self._writer.episode_active
                    and self._writer.frame_count
                ):
                    self._writer.save_episode(self._task, success=False)
                    stats.episodes_saved += 1
                    armed = False
                    self._visualizer.log_event("episode_saved", "Manual episode split")
                    if max_episodes is not None and stats.episodes_saved >= max_episodes:
                        break

                tick_started = time.monotonic()
                measured = self._hardware.read_state()
                action = self._teleop.read(last_dt)
                if reset_active:
                    action = home
                    if np.all(np.abs(measured.positions - home) <= self._enter):
                        reset_active = False
                if action is None:
                    action = last_action if last_action is not None else home
                if getattr(self._teleop, "writes_targets", True):
                    applied = self._hardware.write_positions(action)
                else:
                    # The source owns the servos (torque released): writing a
                    # goal here can re-engage the position loop on some
                    # firmware and make the arm fight the operator.
                    applied = np.asarray(action, dtype=np.float32)
                last_action = applied
                base = wrist = None
                dt_sync_ms = None

                if tick_started - started >= self._warmup_s:
                    try:
                        base, wrist, dt_sync = self._camera_hub.get_synced_frames()
                        for name in ("base", "wrist"):
                            packet = self._camera_hub.get_frame(name)
                            if time.monotonic() - packet.captured_at > max(0.2, 3 / self._fps):
                                raise CameraError(f"camera '{name}' frame is stale")
                    except CameraError as exc:
                        if has_sample:
                            raise CameraError(
                                f"camera failure during continuous recording: {exc}"
                            ) from exc
                        if tick_started - last_camera_warning >= 2:
                            logger.warning("waiting for cameras: %s", exc)
                            last_camera_warning = tick_started
                    else:
                        error = np.abs(measured.positions - home)
                        if np.any(error > self._exit):
                            armed = True
                        if not self._writer.episode_active and armed:
                            # Episodes open on departure, never while parked at
                            # home, so they cannot begin with idle frames.
                            index = self._writer.start_episode(self._task)
                            logger.info("arm left the home pose; recording episode %06d", index)
                            home_hint_shown = False
                            self._visualizer.log_event(
                                "episode_started", f"Recording episode {index:06d}"
                            )
                        if not self._writer.episode_active:
                            state = RunnerState.IDLE
                            if not home_hint_shown:
                                home_hint_shown = True
                                logger.info(
                                    "parked at the home pose (max joint error %.1f deg); "
                                    "recording starts once a joint leaves it by > %.1f deg",
                                    float(error.max()),
                                    self._exit,
                                )
                        else:
                            state = RunnerState.RECORDING
                            self._writer.append(
                                FrameSample(tick_started, measured.positions, applied, base, wrist)
                            )
                            stats.frames_recorded += 1
                            has_sample = True
                            dt_sync_ms = dt_sync * 1000.0
                            if armed and np.all(error <= self._enter):
                                self._writer.save_episode(self._task, success=False)
                                stats.episodes_saved += 1
                                armed = False
                                self._visualizer.log_event(
                                    "episode_saved", "Arm returned to home pose"
                                )

                self._visualizer.log_step(
                    session_time_s=tick_started - started,
                    step_index=stats.timer_ticks + 1,
                    state=state.value,
                    measured_positions=measured.positions,
                    applied_action=applied,
                    base_rgb=base,
                    wrist_rgb=wrist,
                    dt_sync_ms=dt_sync_ms,
                    loop_fps=self._timer.summary().mean_fps,
                    episode_index=self._writer.active_episode_index,
                    episode_frame_index=(
                        self._writer.frame_count if self._writer.episode_active else None
                    ),
                    kinematics=self._kinematics,
                )

                tick = self._timer.wait()
                stats.timer_ticks += 1
                last_dt = tick.period_s
                if tick_started - last_report >= 5.0:
                    elapsed_report = tick_started - last_report
                    # Interval rate: a cumulative average decays while the arm is
                    # parked at home and looks like a throughput collapse.
                    capture_fps = (
                        stats.frames_recorded - last_report_frames
                    ) / elapsed_report
                    last_report = tick_started
                    last_report_frames = stats.frames_recorded
                    summary = self._timer.summary()
                    logger.debug(
                        "capture %.1f fps | loop %.1f fps | overruns %d/%d | writer queue %d/%d",
                        capture_fps,
                        summary.mean_fps,
                        summary.overruns,
                        summary.ticks,
                        self._writer.queue_depth,
                        self._writer.queue_capacity,
                    )
        except KeyboardInterrupt:
            stats.interrupted = True
            logger.info("interrupted by user; saving the partial episode")
        except Exception:
            failure = True
            raise
        finally:
            # Save the last partial episode on normal exit or interruption.
            # Writer failures surface through drain/close.
            try:
                self._writer.drain()
                if failure and self._writer.episode_active:
                    self._writer.discard_episode()
                    stats.episodes_discarded += 1
                elif self._writer.episode_active and self._writer.frame_count:
                    self._writer.save_episode(self._task, success=False)
                    stats.episodes_saved += 1
                elif self._writer.episode_active:
                    self._writer.discard_episode()
            finally:
                try:
                    self._writer.close()
                finally:
                    self._timer.stop()
                    self._visualizer.close()
                    summary = self._timer.summary()
                    stats.timer_ticks = summary.ticks
                    stats.timer_overruns = summary.overruns
                    stats.mean_loop_fps = summary.mean_fps
                    stats.max_sync_drift_ms = self._camera_hub.stats().sync_max_ms
                    stats.duration_s = time.monotonic() - started
                    stats.final_state = state.value
                    self._camera_hub.stop()
                    self._teleop.close()
                    self._hardware.close()
                    self._commands.close()
            logger.info("session finished")
        return stats
