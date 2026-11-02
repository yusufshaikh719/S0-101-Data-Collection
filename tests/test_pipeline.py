"""End-to-end integration: full runner session on a simulated rig."""

from __future__ import annotations

import json
import time

import numpy as np

from so101_collector.console import Command, TimedCommandReader
from so101_collector.core.camera_hub import CameraHub
from so101_collector.core.dataset_writer import (
    DatasetWriter,
    NpzTableStore,
    OpenCVVideoEncoder,
)
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.runner import DataCollectionRunner, RunnerState
from so101_collector.core.teleop_controller import LeaderArmTeleop
from so101_collector.simulation import ScriptedTeleop, SimulatedBus, SimulatedCamera
from tests.conftest import make_robot_config, make_vision_config


def _build(tmp_path, teleop, commands, *, fps=30.0, warmup_s=0.1):
    robot = make_robot_config()
    vision = make_vision_config(width=64, height=48)
    hardware = SO101HardwareInterface(robot, bus=SimulatedBus(robot))
    hub = CameraHub(vision, source_factory=lambda cfg: SimulatedCamera(cfg))
    writer = DatasetWriter(
        tmp_path / "data",
        "integration",
        fps=fps,
        image_shape=vision.by_name["base"].frame_shape,
        encoder=OpenCVVideoEncoder(),
        table_store=NpzTableStore(),
    )
    runner = DataCollectionRunner(
        hardware=hardware,
        camera_hub=hub,
        teleop=teleop,
        writer=writer,
        commands=commands,
        fps=fps,
        warmup_s=warmup_s,
        task="integration: pick the block off the floor",
    )
    return runner, hardware, hub, writer


def test_full_session_saves_one_episode(tmp_path):
    """START -> warmup -> record -> FINISH on the simulated rig."""
    robot = make_robot_config()
    leader = SO101HardwareInterface(robot, bus=SimulatedBus(robot))
    teleop = LeaderArmTeleop(leader)
    runner, hardware, hub, writer = _build(
        tmp_path, teleop, TimedCommandReader(record_s=0.7)
    )

    stats = runner.run(max_episodes=1, max_seconds=30.0)

    assert stats.episodes_saved == 1
    assert stats.episodes_discarded == 0
    assert stats.frames_recorded >= 5
    assert stats.final_state == RunnerState.IDLE.value
    assert stats.duration_s > 0

    # Dataset artifacts.
    assert writer.episode_count == 1
    episodes = [
        json.loads(line)
        for line in (writer.directory / "meta" / "episodes.jsonl")
        .read_text()
        .splitlines()
        if line
    ]
    assert len(episodes) == 1
    assert episodes[0]["length"] == stats.frames_recorded
    assert episodes[0]["success"] is True
    assert (writer.directory / episodes[0]["data_file"]).is_file()
    for rel_path in episodes[0]["videos"].values():
        assert (writer.directory / rel_path).stat().st_size > 0

    # State/action columns follow the schema with real values.
    with np.load(writer.directory / episodes[0]["data_file"]) as data:
        assert data["observation.state"].shape == (stats.frames_recorded, 6)
        assert data["action"].shape == (stats.frames_recorded, 6)
        assert data["observation.state"].dtype == np.float32
        assert data["action"].dtype == np.float32

    # Clean teardown: everything closed, threads joined.
    assert not hardware.is_open
    assert not leader.is_open
    assert not hub.running
    assert not writer.episode_active


def test_abort_discards_episode_without_touching_disk(tmp_path):
    """START -> record -> ABORT clears RAM only; QUIT exits the session."""

    class StartAbortQuit:
        """START immediately, ABORT after ~0.4 s, then QUIT."""

        def __init__(self) -> None:
            self._started_at: float | None = None
            self._aborted = False

        def read(self) -> Command | None:
            if self._started_at is None:
                self._started_at = time.monotonic()
                return Command.START
            if not self._aborted and time.monotonic() - self._started_at >= 0.4:
                self._aborted = True
                return Command.ABORT
            if self._aborted:
                return Command.QUIT
            return None

        def close(self) -> None:
            return None

    teleop = ScriptedTeleop(make_robot_config(), amplitude_deg=5.0)
    runner, hardware, hub, writer = _build(tmp_path, teleop, StartAbortQuit())

    stats = runner.run(max_seconds=30.0)

    assert stats.episodes_saved == 0
    assert stats.episodes_discarded == 1
    assert stats.frames_recorded >= 3  # frames were buffered before abort
    assert writer.episode_count == 0
    assert writer.frame_count == 0
    # Nothing but empty scaffolding reached the disk.
    assert not list((writer.directory / "data").rglob("*.npz"))
    assert not list((writer.directory / "videos").rglob("*.mp4"))
    assert (writer.directory / "meta" / "episodes.jsonl").read_text().strip() == ""
    assert not hardware.is_open
    assert not hub.running


def test_continuous_recording_splits_only_on_home_reentry(tmp_path):
    from so101_collector.core.continuous_runner import ContinuousDataCollectionRunner
    from so101_collector.core.lerobot_writer import LeRobotWriter

    class FakeDataset:
        def __init__(self):
            self.meta = type("Meta", (), {"total_episodes": 0})()
            self.pending = []
            self.episodes = []

        @classmethod
        def create(cls, **kwargs):
            return cls()

        def add_frame(self, frame):
            self.pending.append(frame)

        def save_episode(self, parallel_encoding=True):
            self.episodes.append(self.pending)
            self.pending = []

        def clear_episode_buffer(self):
            self.pending = []

        def finalize(self):
            pass

    robot = make_robot_config()
    vision = make_vision_config(width=64, height=48)
    hardware = SO101HardwareInterface(robot, bus=SimulatedBus(robot))
    hub = CameraHub(vision, source_factory=lambda cfg: SimulatedCamera(cfg))
    writer = LeRobotWriter(tmp_path, "continuous", fps=30,
                           image_shapes={"base": (48, 64), "wrist": (48, 64)},
                           dataset_class=FakeDataset)

    class PoseTeleop:
        def __init__(self):
            self.tick = 0

        def open(self):
            pass

        def close(self):
            pass

        def read(self, dt):
            self.tick += 1
            action = np.array(robot.home_pose, dtype=np.float32)
            if 6 <= self.tick < 12 or 20 <= self.tick < 26:
                action[0] += 20
            return action

    class QuitAfterTicks:
        def __init__(self):
            self.tick = 0

        def read(self):
            self.tick += 1
            return Command.QUIT if self.tick > 34 else None

        def close(self):
            pass

    runner = ContinuousDataCollectionRunner(
        hardware=hardware, camera_hub=hub, teleop=PoseTeleop(), writer=writer,
        commands=QuitAfterTicks(), fps=30, warmup_s=0.1,
    )
    stats = runner.run(max_seconds=10)
    episodes = writer._dataset.episodes
    assert len(episodes) == stats.episodes_saved == 3
    assert sum(map(len, episodes)) == stats.frames_recorded
    assert all(len(ep) > 1 for ep in episodes)
    assert not hardware.is_open and not hub.running
