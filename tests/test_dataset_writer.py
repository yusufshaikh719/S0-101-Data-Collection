"""Dataset writer tests: buffer rules, atomic saves, manifest, discards."""

from __future__ import annotations

import json

import numpy as np
import pytest

from so101_collector.core.dataset_writer import (
    DatasetWriter,
    EpisodeBuffer,
    FrameSample,
    NpzTableStore,
    OpenCVVideoEncoder,
)
from so101_collector.exceptions import DatasetWriteError, EpisodeStateError

FRAME_SHAPE = (32, 48)  # (H, W) - small for fast encodes


def make_sample(index: int = 0) -> FrameSample:
    return FrameSample(
        timestamp=1000.0 + index / 30.0,
        state=np.full(6, index, dtype=np.float32),
        action=np.full(6, index + 0.5, dtype=np.float32),
        base=np.full((*FRAME_SHAPE, 3), index % 256, dtype=np.uint8),
        wrist=np.full((*FRAME_SHAPE, 3), (index + 1) % 256, dtype=np.uint8),
    )


class FlakyEncoder:
    """Delegates to a real encoder but fails the first N encodes."""

    def __init__(self, failures: int = 1) -> None:
        self.inner = OpenCVVideoEncoder()
        self.failures = failures

    def encode(self, frames, path, fps) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise DatasetWriteError("simulated encoder crash")
        self.inner.encode(frames, path, fps)


@pytest.fixture
def writer(tmp_path) -> DatasetWriter:
    return DatasetWriter(
        tmp_path / "data",
        "unit_test",
        fps=30.0,
        image_shape=FRAME_SHAPE,
        encoder=OpenCVVideoEncoder(),
        table_store=NpzTableStore(),
    )


def _columns(writer: DatasetWriter, manifest: dict) -> dict:
    data_path = writer.directory / manifest["data_file"]
    with np.load(data_path) as archive:
        return {key: archive[key] for key in archive.files}


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------
def test_buffer_append_and_clear(frame_factory):
    buffer = EpisodeBuffer()
    assert buffer.is_empty
    buffer.append(make_sample(0))
    buffer.append(make_sample(1))
    assert len(buffer) == 2
    assert buffer.image_shape == FRAME_SHAPE
    assert buffer.states.shape == (2, 6)
    buffer.clear()
    assert buffer.is_empty and buffer.image_shape is None


def test_buffer_rejects_bad_shapes():
    buffer = EpisodeBuffer()
    buffer.append(make_sample(0))
    with pytest.raises(ValueError, match="state/action"):
        buffer.append(
            FrameSample(
                timestamp=1.0,
                state=np.zeros(5, dtype=np.float32),
                action=np.zeros(6, dtype=np.float32),
                base=np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8),
                wrist=np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8),
            )
        )
    with pytest.raises(ValueError, match="uint8"):
        buffer.append(
            FrameSample(
                timestamp=1.0,
                state=np.zeros(6, dtype=np.float32),
                action=np.zeros(6, dtype=np.float32),
                base=np.zeros((*FRAME_SHAPE, 3), dtype=np.float32),
                wrist=np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8),
            )
        )


# ---------------------------------------------------------------------------
# Lifecycle rules
# ---------------------------------------------------------------------------
def test_append_without_start_raises(writer):
    with pytest.raises(EpisodeStateError, match="no episode in progress"):
        writer.append(make_sample(0))


def test_double_start_raises(writer):
    writer.start_episode("task a")
    with pytest.raises(EpisodeStateError, match="already in progress"):
        writer.start_episode("task b")


def test_save_without_start_raises(writer):
    with pytest.raises(EpisodeStateError, match="nothing to save"):
        writer.save_episode("task")


def test_save_empty_episode_raises(writer):
    writer.start_episode("task")
    with pytest.raises(EpisodeStateError, match="empty"):
        writer.save_episode("task")


def test_discard_without_episode_is_noop(writer, tmp_path):
    writer.discard_episode()  # must not raise
    assert writer.episode_count == 0


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def test_save_writes_videos_table_and_manifest(writer):
    writer.start_episode("pick up the block")
    for i in range(10):
        writer.append(make_sample(i))
    manifest = writer.save_episode("pick up the block", success=True)

    assert manifest["episode_index"] == 0
    assert manifest["length"] == 10
    assert manifest["success"] is True
    assert manifest["duration"] == pytest.approx(10 / 30.0)

    # Files exist, are non-empty, and use POSIX separators in the manifest.
    data_file = writer.directory / manifest["data_file"]
    assert data_file.is_file() and data_file.stat().st_size > 0
    assert "\\" not in manifest["data_file"]
    for key, rel_path in manifest["videos"].items():
        assert key.startswith("observation.images.")
        assert "\\" not in rel_path
        video = writer.directory / rel_path
        assert video.is_file() and video.stat().st_size > 0

    # Columns follow the LeRobot schema.
    columns = _columns(writer, manifest)
    assert columns["observation.state"].shape == (10, 6)
    assert columns["observation.state"].dtype == np.float32
    assert columns["action"].dtype == np.float32
    assert columns["timestamp"].dtype == np.float64
    np.testing.assert_allclose(columns["timestamp"], np.arange(10) / 30.0)
    assert columns["episode_index"].tolist() == [0] * 10
    assert columns["index"].tolist() == list(range(10))

    # Meta files.
    info = json.loads((writer.directory / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 10
    assert info["fps"] == pytest.approx(30.0)
    tasks = (writer.directory / "meta" / "tasks.jsonl").read_text().splitlines()
    assert json.loads(tasks[0])["task"] == "pick up the block"

    # Buffer was released and the next index advanced.
    assert not writer.episode_active
    assert writer.frame_count == 0
    assert writer.next_episode_index == 1


def test_second_episode_gets_contiguous_global_index(writer):
    writer.start_episode("first")
    for i in range(5):
        writer.append(make_sample(i))
    first = writer.save_episode("first")

    writer.start_episode("second")
    for i in range(7):
        writer.append(make_sample(i))
    second = writer.save_episode("second", success=False)

    first_cols = _columns(writer, first)
    second_cols = _columns(writer, second)
    assert second_cols["index"][0] == 5  # continues after episode 0
    assert second_cols["episode_index"].tolist() == [1] * 7
    assert second["success"] is False

    info = json.loads((writer.directory / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 12
    assert info["total_tasks"] == 2  # different task strings


def test_discard_clears_ram_without_touching_disk(writer):
    writer.start_episode("doomed")
    for i in range(4):
        writer.append(make_sample(i))

    writer.discard_episode()

    assert not writer.episode_active
    assert writer.frame_count == 0
    assert writer.episode_count == 0
    assert not list((writer.directory / "data").rglob("*.npz"))
    assert not list((writer.directory / "videos").rglob("*.mp4"))
    assert not (writer.directory / "meta" / "episodes.jsonl").read_text().strip()
    # Staging is cleaned too.
    assert not list((writer.directory / ".staging").iterdir())

    # A new episode can start immediately and gets the same index.
    assert writer.start_episode("fresh") == 0


def test_failed_save_keeps_buffer_for_retry(tmp_path):
    flaky = FlakyEncoder(failures=1)
    writer = DatasetWriter(
        tmp_path / "data",
        "retry_test",
        fps=30.0,
        image_shape=FRAME_SHAPE,
        encoder=flaky,
        table_store=NpzTableStore(),
    )
    writer.start_episode("retry me")
    for i in range(6):
        writer.append(make_sample(i))

    with pytest.raises(DatasetWriteError, match="encoder crash"):
        writer.save_episode("retry me")

    # Episode stays active with all frames intact; nothing hit the dataset.
    assert writer.episode_active
    assert writer.frame_count == 6
    assert writer.episode_count == 0
    assert not list((writer.directory / "data").rglob("*.npz"))
    assert not list((writer.directory / ".staging").iterdir())

    # Pressing FINISH again succeeds with the same episode index.
    manifest = writer.save_episode("retry me")
    assert manifest["episode_index"] == 0
    assert manifest["length"] == 6
    assert writer.episode_count == 1


def test_mid_save_failure_leaves_no_partial_episode(tmp_path):
    flaky = FlakyEncoder(failures=1)  # base.mp4 encodes, wrist.mp4 crashes
    writer = DatasetWriter(
        tmp_path / "data",
        "atomic_test",
        fps=30.0,
        image_shape=FRAME_SHAPE,
        encoder=flaky,
        table_store=NpzTableStore(),
    )
    writer.start_episode("atomic")
    for i in range(4):
        writer.append(make_sample(i))

    with pytest.raises(DatasetWriteError):
        writer.save_episode("atomic")

    # No file was committed: staging was cleaned before any os.replace.
    assert not list((writer.directory / "data").rglob("*.npz"))
    assert not list((writer.directory / "videos").rglob("*.mp4"))
    assert (writer.directory / "meta" / "episodes.jsonl").read_text().strip() == ""


def test_dataset_resumes_from_existing_episodes(tmp_path):
    def populate():
        w = DatasetWriter(
            tmp_path / "data", "resume", fps=30.0, image_shape=FRAME_SHAPE,
            encoder=OpenCVVideoEncoder(), table_store=NpzTableStore(),
        )
        return w

    writer = populate()
    writer.start_episode("one")
    writer.append(make_sample(0))
    writer.save_episode("one")

    reopened = populate()  # simulates restarting the CLI on the same dataset
    assert reopened.next_episode_index == 1
    assert reopened.episode_count == 1
    assert reopened.total_frames == 1


def test_repo_id_must_be_plain_name(tmp_path):
    with pytest.raises(ValueError, match="repo_id"):
        DatasetWriter(tmp_path, "nested/name", encoder=OpenCVVideoEncoder(),
                      table_store=NpzTableStore())


def test_official_lerobot_dataset_roundtrip_and_resume(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from so101_collector.core.lerobot_writer import LeRobotWriter

    def open_writer():
        return LeRobotWriter(tmp_path, "roundtrip", fps=30,
                             image_shapes={"base": FRAME_SHAPE, "wrist": FRAME_SHAPE})

    writer = open_writer()
    for episode in range(2):
        writer.start_episode("pick up the block")
        for i in range(4):
            writer.append(make_sample(episode * 4 + i))
        writer.save_episode("pick up the block", success=False)
    writer.close()

    resumed = open_writer()
    assert resumed.start_episode("pick up the block") == 2
    for i in range(4):
        resumed.append(make_sample(8 + i))
    resumed.save_episode("pick up the block", success=False)
    resumed.close()

    dataset = LeRobotDataset("roundtrip", root=tmp_path / "roundtrip", video_backend="pyav")
    assert dataset.num_episodes == 3
    assert dataset.num_frames == 12
    for i in (0, 3, 4, 8, 11):
        row = dataset[i]
        assert row["episode_index"].item() == i // 4
        assert row["frame_index"].item() == i % 4
        assert row["timestamp"].item() == pytest.approx((i % 4) / 30)
        np.testing.assert_allclose(row["observation.state"].numpy(), np.full(6, i))
        np.testing.assert_allclose(row["action"].numpy(), np.full(6, i + 0.5))
        assert row["observation.images.base"].shape == (3, *FRAME_SHAPE)
        assert row["observation.images.wrist"].shape == (3, *FRAME_SHAPE)


def test_lerobot_worker_failure_is_reported_and_finalized(tmp_path):
    from so101_collector.core.lerobot_writer import LeRobotWriter

    class FailingDataset:
        @classmethod
        def create(cls, **kwargs):
            instance = cls()
            instance.meta = type("Meta", (), {"total_episodes": 0})()
            instance.finalized = False
            return instance

        def add_frame(self, frame):
            raise OSError("disk failed")

        def finalize(self):
            self.finalized = True

    writer = LeRobotWriter(tmp_path, "failure", fps=30,
                           image_shapes={"base": FRAME_SHAPE, "wrist": FRAME_SHAPE},
                           dataset_class=FailingDataset)
    writer.start_episode("task")
    writer.append(make_sample())
    with pytest.raises(DatasetWriteError, match="disk failed"):
        writer.drain()
    with pytest.raises(DatasetWriteError, match="disk failed"):
        writer.close()
    assert writer._dataset.finalized
