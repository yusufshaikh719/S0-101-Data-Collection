"""LeRobot writer tests, including a round trip through LeRobotDataset."""

from __future__ import annotations

import numpy as np
import pytest

from so101_collector.core.lerobot_writer import FrameSample, LeRobotWriter
from so101_collector.exceptions import DatasetWriteError, EpisodeStateError

FRAME_SHAPE = (32, 48)


def make_sample(index: int = 0) -> FrameSample:
    return FrameSample(
        timestamp=1000.0 + index / 30.0,
        state=np.full(6, index, dtype=np.float32),
        action=np.full(6, index + 0.5, dtype=np.float32),
        base=np.full((*FRAME_SHAPE, 3), index % 256, dtype=np.uint8),
        wrist=np.full((*FRAME_SHAPE, 3), (index + 1) % 256, dtype=np.uint8),
    )


class FakeDataset:
    """Stands in for LeRobotDataset and records the calls it receives."""

    def __init__(self) -> None:
        self.meta = type("Meta", (), {"total_episodes": 0})()
        self.frames: list[dict] = []
        self.episodes: list[list[dict]] = []
        self.cleared = 0
        self.finalized = False
        self.fail_on_frame = False

    @classmethod
    def create(cls, **kwargs):
        return cls()

    def add_frame(self, frame):
        if self.fail_on_frame:
            raise OSError("disk failed")
        self.frames.append(frame)

    def save_episode(self, parallel_encoding=True):
        self.episodes.append(self.frames)
        self.frames = []

    def clear_episode_buffer(self):
        self.frames = []
        self.cleared += 1

    def finalize(self):
        self.finalized = True


def open_writer(tmp_path, repo_id="unit_test", **kwargs) -> LeRobotWriter:
    return LeRobotWriter(
        tmp_path,
        repo_id,
        fps=30,
        image_shapes={"base": FRAME_SHAPE, "wrist": FRAME_SHAPE},
        **kwargs,
    )


def test_episode_is_buffered_and_saved(tmp_path):
    writer = open_writer(tmp_path, dataset_class=FakeDataset)
    assert writer.start_episode("pick up the block") == 0
    for index in range(3):
        writer.append(make_sample(index))
    writer.save_episode("pick up the block")
    writer.drain()
    writer.close()

    dataset = writer._dataset
    assert len(dataset.episodes) == 1
    assert [frame["observation.state"][0] for frame in dataset.episodes[0]] == [0.0, 1.0, 2.0]
    assert dataset.finalized
    assert not writer.episode_active


def test_discard_clears_the_buffer_without_saving(tmp_path):
    writer = open_writer(tmp_path, dataset_class=FakeDataset)
    writer.start_episode("pick up the block")
    writer.append(make_sample(0))
    writer.discard_episode()
    writer.drain()
    writer.close()

    assert writer._dataset.cleared == 1
    assert writer._dataset.episodes == []
    assert not writer.episode_active


def test_task_cannot_change_mid_episode(tmp_path):
    writer = open_writer(tmp_path, dataset_class=FakeDataset)
    writer.start_episode("task a")
    writer.append(make_sample(0))
    with pytest.raises(EpisodeStateError, match="task cannot change"):
        writer.save_episode("task b")
    writer.close()


def test_save_requires_an_active_episode(tmp_path):
    writer = open_writer(tmp_path, dataset_class=FakeDataset)
    with pytest.raises(EpisodeStateError, match="empty or inactive"):
        writer.save_episode("task")
    writer.close()


def test_append_validates_joint_count(tmp_path):
    writer = open_writer(tmp_path, dataset_class=FakeDataset)
    writer.start_episode("task")
    bad = make_sample(0)
    with pytest.raises(DatasetWriteError, match="joint count"):
        writer.append(
            FrameSample(
                timestamp=bad.timestamp,
                state=np.zeros(5, dtype=np.float32),
                action=bad.action,
                base=bad.base,
                wrist=bad.wrist,
            )
        )
    writer.close()


@pytest.mark.parametrize("repo_id", ["nested/name/extra", "..", "a:b"])
def test_repo_id_must_be_a_dataset_name(tmp_path, repo_id):
    with pytest.raises(ValueError, match="repo_id"):
        open_writer(tmp_path, repo_id, dataset_class=FakeDataset)


def test_worker_failure_is_reported_and_finalized(tmp_path):
    class FailingDataset(FakeDataset):
        def __init__(self) -> None:
            super().__init__()
            self.fail_on_frame = True

    writer = open_writer(tmp_path, repo_id="failure", dataset_class=FailingDataset)
    writer.start_episode("task")
    writer.append(make_sample())
    with pytest.raises(DatasetWriteError, match="disk failed"):
        writer.drain()
    with pytest.raises(DatasetWriteError, match="disk failed"):
        writer.close()
    assert writer._dataset.finalized


def test_official_lerobot_dataset_roundtrip_and_resume(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    writer = open_writer(tmp_path, repo_id="roundtrip")
    for episode in range(2):
        writer.start_episode("pick up the block")
        for index in range(4):
            writer.append(make_sample(episode * 4 + index))
        writer.save_episode("pick up the block", success=False)
    writer.close()

    resumed = open_writer(tmp_path, repo_id="roundtrip")
    assert resumed.start_episode("pick up the block") == 2
    for index in range(4):
        resumed.append(make_sample(8 + index))
    resumed.save_episode("pick up the block", success=False)
    resumed.close()

    dataset = LeRobotDataset("roundtrip", root=tmp_path / "roundtrip", video_backend="pyav")
    assert dataset.num_episodes == 3
    assert dataset.num_frames == 12
    for index in (0, 3, 4, 8, 11):
        row = dataset[index]
        assert row["episode_index"].item() == index // 4
        assert row["frame_index"].item() == index % 4
        assert row["timestamp"].item() == pytest.approx((index % 4) / 30)
        np.testing.assert_allclose(row["observation.state"].numpy(), np.full(6, index))
        np.testing.assert_allclose(row["action"].numpy(), np.full(6, index + 0.5))
        assert row["observation.images.base"].shape == (3, *FRAME_SHAPE)
        assert row["observation.images.wrist"].shape == (3, *FRAME_SHAPE)
