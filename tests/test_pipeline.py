"""End-to-end recording session on the simulated rig."""

from __future__ import annotations

import numpy as np

from so101_collector.console import Command
from so101_collector.core.camera_hub import CameraHub
from so101_collector.core.continuous_runner import ContinuousDataCollectionRunner
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.lerobot_writer import LeRobotWriter
from so101_collector.simulation import SimulatedBus, SimulatedCamera
from tests.conftest import make_robot_config, make_vision_config


class FakeDataset:
    def __init__(self) -> None:
        self.meta = type("Meta", (), {"total_episodes": 0})()
        self.pending: list[dict] = []
        self.episodes: list[list[dict]] = []

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


def test_continuous_recording_splits_only_on_home_reentry(tmp_path):
    robot = make_robot_config()
    vision = make_vision_config(width=64, height=48)
    hardware = SO101HardwareInterface(robot, bus=SimulatedBus(robot))
    hub = CameraHub(vision, source_factory=lambda cfg: SimulatedCamera(cfg))
    writer = LeRobotWriter(
        tmp_path,
        "continuous",
        fps=30,
        image_shapes={"base": (48, 64), "wrist": (48, 64)},
        dataset_class=FakeDataset,
    )

    class PoseTeleop:
        def __init__(self) -> None:
            self.tick = 0

        def open(self) -> None:
            pass

        def close(self) -> None:
            pass

        def read(self, dt):
            self.tick += 1
            action = np.array(robot.home_pose, dtype=np.float32)
            if 6 <= self.tick < 12 or 20 <= self.tick < 26:
                action[0] += 20
            return action

    class QuitAfterTicks:
        def __init__(self) -> None:
            self.tick = 0

        def read(self):
            self.tick += 1
            return Command.QUIT if self.tick > 34 else None

        def close(self) -> None:
            pass

    runner = ContinuousDataCollectionRunner(
        hardware=hardware,
        camera_hub=hub,
        teleop=PoseTeleop(),
        writer=writer,
        commands=QuitAfterTicks(),
        fps=30,
        warmup_s=0.1,
    )
    stats = runner.run(max_seconds=10)

    episodes = writer._dataset.episodes
    assert len(episodes) == stats.episodes_saved == 3
    assert sum(map(len, episodes)) == stats.frames_recorded
    assert all(len(episode) > 1 for episode in episodes)
    assert not hardware.is_open
    assert not hub.running
