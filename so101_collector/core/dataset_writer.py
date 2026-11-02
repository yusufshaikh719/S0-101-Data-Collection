"""LeRobot-compatible episode writer with atomic disk commits.

In-memory episode buffer -> on save, encode two H.264/MP4 camera streams and
one Parquet table with ``observation.state`` / ``action`` / ``timestamp``
columns, then commit them into a Hugging Face ``LeRobotDataset``-style
layout::

    <root>/<repo_id>/
      meta/info.json                 # fps, features schema, totals
      meta/tasks.jsonl               # task string -> task_index
      meta/episodes.jsonl            # one manifest line per episode
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/episode_000000/observation.images.base.mp4
      videos/chunk-000/episode_000000/observation.images.wrist.mp4

Data-safety guarantees from the specification:

* **Atomic episodes** - everything is first written into a hidden staging
  directory and moved into place with ``os.replace``; the ``episodes.jsonl``
  append is the commit point.  A crash can leave unreferenced files (ignored
  by loaders) but never a manifest entry pointing at missing data.
* **Discard never touches disk** - :meth:`DatasetWriter.discard_episode`
  only clears RAM buffers.
* **Failed saves keep the buffer** - if encoding or the commit fails, the
  episode stays active so the operator can retry FINISH without losing data.

Backends are pluggable and imported lazily: PyAV (H.264) with an OpenCV
fallback for video, Parquet (pyarrow) with an ``.npz`` fallback for tables.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from so101_collector.config import JOINT_NAMES
from so101_collector.exceptions import DatasetWriteError, EpisodeStateError

logger = logging.getLogger(__name__)

CHUNKS_SIZE = 1000
VIDEO_KEYS = ("observation.images.base", "observation.images.wrist")


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameSample:
    """One control-loop tick of synchronized multimodal data."""

    timestamp: float
    state: np.ndarray  # (6,) measured joint angles, deg
    action: np.ndarray  # (6,) commanded joint angles, deg
    base: np.ndarray  # (H, W, 3) uint8 RGB
    wrist: np.ndarray  # (H, W, 3) uint8 RGB


class EpisodeBuffer:
    """RAM-only storage for one episode (cleared on save/discard)."""

    def __init__(self, n_joints: int = len(JOINT_NAMES)) -> None:
        self._n_joints = n_joints
        self._timestamps: list[float] = []
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._base: list[np.ndarray] = []
        self._wrist: list[np.ndarray] = []
        self._image_shape: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self._timestamps)

    @property
    def is_empty(self) -> bool:
        return not self._timestamps

    @property
    def image_shape(self) -> tuple[int, int] | None:
        return self._image_shape

    def append(self, sample: FrameSample) -> None:
        state = np.asarray(sample.state, dtype=np.float32).reshape(-1)
        action = np.asarray(sample.action, dtype=np.float32).reshape(-1)
        if state.size != self._n_joints or action.size != self._n_joints:
            raise ValueError(
                f"state/action must have {self._n_joints} joints, "
                f"got {state.size}/{action.size}"
            )
        base = self._validate_frame(sample.base, "base")
        wrist = self._validate_frame(sample.wrist, "wrist")
        if self._image_shape is None:
            self._image_shape = (base.shape[0], base.shape[1])
        self._timestamps.append(float(sample.timestamp))
        self._states.append(state)
        self._actions.append(action)
        self._base.append(base)
        self._wrist.append(wrist)

    def _validate_frame(self, frame: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"{name} frame must be HxWx3 uint8, got {arr.dtype} {arr.shape}"
            )
        if self._image_shape is not None and arr.shape[:2] != self._image_shape:
            raise ValueError(
                f"{name} frame shape {arr.shape[:2]} != episode shape {self._image_shape}"
            )
        return arr

    @property
    def timestamps(self) -> list[float]:
        return list(self._timestamps)

    @property
    def states(self) -> np.ndarray:
        return np.stack(self._states)

    @property
    def actions(self) -> np.ndarray:
        return np.stack(self._actions)

    @property
    def base_frames(self) -> list[np.ndarray]:
        return list(self._base)

    @property
    def wrist_frames(self) -> list[np.ndarray]:
        return list(self._wrist)

    def clear(self) -> None:
        self._timestamps.clear()
        self._states.clear()
        self._actions.clear()
        self._base.clear()
        self._wrist.clear()
        self._image_shape = None


# ---------------------------------------------------------------------------
# Video encoders
# ---------------------------------------------------------------------------
class VideoEncoder(Protocol):
    def encode(self, frames: Sequence[np.ndarray], path: Path, fps: float) -> None: ...


class PyAVVideoEncoder:
    """H.264 MP4 via PyAV (preferred: standard, small, LeRobot-loadable)."""

    def __init__(self, codec: str = "libx264") -> None:
        try:
            import av
        except ImportError as exc:
            raise DatasetWriteError(
                "H.264 encoding requires PyAV (pip install 'so101-collector[video]') "
                "or select the opencv fallback"
            ) from exc
        self._av = av
        self._codec = codec

    def encode(self, frames: Sequence[np.ndarray], path: Path, fps: float) -> None:
        if not frames:
            raise DatasetWriteError(f"refusing to encode empty frame list -> {path}")
        height, width = frames[0].shape[:2]
        width -= width % 2  # yuv420p requires even dimensions
        height -= height % 2
        try:
            with self._av.open(str(path), mode="w") as container:
                stream = container.add_stream(
                    self._codec, rate=Fraction(fps).limit_denominator(1001)
                )
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                for array in frames:
                    video_frame = self._av.VideoFrame.from_ndarray(
                        array[:height, :width], format="rgb24"
                    )
                    for packet in stream.encode(video_frame):
                        container.mux(packet)
                for packet in stream.encode():  # flush
                    container.mux(packet)
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise DatasetWriteError(f"PyAV encoding failed for {path.name}: {exc}") from exc
        if not path.exists() or path.stat().st_size == 0:
            raise DatasetWriteError(f"PyAV produced no output for {path}")


class OpenCVVideoEncoder:
    """MP4 fallback via OpenCV ``mp4v`` (used when PyAV is unavailable)."""

    def __init__(self, codec: str = "mp4v") -> None:
        try:
            import cv2
        except ImportError as exc:
            raise DatasetWriteError(
                "OpenCV video fallback requires opencv-python (pip install opencv-python)"
            ) from exc
        self._cv2 = cv2
        self._codec = codec

    def encode(self, frames: Sequence[np.ndarray], path: Path, fps: float) -> None:
        if not frames:
            raise DatasetWriteError(f"refusing to encode empty frame list -> {path}")
        height, width = frames[0].shape[:2]
        fourcc = self._cv2.VideoWriter_fourcc(*self._codec)
        writer = self._cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            writer.release()
            raise DatasetWriteError(f"OpenCV could not open video writer for {path}")
        try:
            for array in frames:
                if array.shape[:2] != (height, width):
                    raise DatasetWriteError(
                        f"inconsistent frame shape {array.shape[:2]} in {path.name}"
                    )
                writer.write(self._cv2.cvtColor(array, self._cv2.COLOR_RGB2BGR))
        except Exception:
            writer.release()
            path.unlink(missing_ok=True)
            raise
        writer.release()
        if not path.exists() or path.stat().st_size == 0:
            raise DatasetWriteError(f"OpenCV produced no output for {path}")


def make_video_encoder(preference: str = "auto") -> VideoEncoder:
    """``auto`` picks PyAV when importable, otherwise the OpenCV fallback."""
    if preference == "auto":
        if importlib.util.find_spec("av") is not None:
            logger.info("video backend: PyAV (H.264)")
            return PyAVVideoEncoder()
        logger.warning("video backend: OpenCV mp4v fallback (install av for H.264)")
        return OpenCVVideoEncoder()
    if preference in ("av", "pyav", "h264"):
        return PyAVVideoEncoder()
    if preference in ("opencv", "cv2", "mp4v"):
        return OpenCVVideoEncoder()
    raise ValueError(f"unknown video backend {preference!r} (auto|av|opencv)")


# ---------------------------------------------------------------------------
# Tabular stores
# ---------------------------------------------------------------------------
class TableStore(Protocol):
    suffix: str

    def write(self, columns: Mapping[str, np.ndarray], stem: Path) -> Path: ...


class ParquetTableStore:
    """LeRobot Parquet tables via pandas + pyarrow."""

    suffix = ".parquet"

    def write(self, columns: Mapping[str, np.ndarray], stem: Path) -> Path:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - pandas is a hard dep
            raise DatasetWriteError("pandas is required for Parquet output") from exc
        path = Path(f"{stem}{self.suffix}")
        table_dict = {
            name: (list(arr) if getattr(arr, "ndim", 1) == 2 else arr)
            for name, arr in columns.items()
        }
        frame = pd.DataFrame(table_dict)
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:
            raise DatasetWriteError(
                "Parquet output requires pyarrow (pip install 'so101-collector[parquet]'); "
                "select the npz backend otherwise"
            ) from exc
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise DatasetWriteError(f"failed to write {path.name}: {exc}") from exc
        if not path.exists() or path.stat().st_size == 0:
            raise DatasetWriteError(f"Parquet writer produced no output for {path}")
        return path


class NpzTableStore:
    """Dev/CI fallback: compressed numpy archive with identical columns."""

    suffix = ".npz"

    def __init__(self) -> None:
        logger.warning(
            "table backend: .npz fallback (pyarrow not installed) - "
            "install pyarrow for LeRobot-compatible Parquet"
        )

    def write(self, columns: Mapping[str, np.ndarray], stem: Path) -> Path:
        path = Path(f"{stem}{self.suffix}")
        try:
            np.savez_compressed(path, **dict(columns))
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise DatasetWriteError(f"failed to write {path.name}: {exc}") from exc
        return path


def make_table_store(preference: str = "auto") -> TableStore:
    """``auto`` picks Parquet when pyarrow is importable, else ``.npz``."""
    if preference == "auto":
        if importlib.util.find_spec("pyarrow") is not None:
            logger.info("table backend: Parquet (pyarrow)")
            return ParquetTableStore()
        return NpzTableStore()
    if preference == "parquet":
        return ParquetTableStore()
    if preference == "npz":
        return NpzTableStore()
    raise ValueError(f"unknown table backend {preference!r} (auto|parquet|npz)")


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("skipping corrupt %s line %d: %s", path.name, line_no, exc)
    return records


def _append_jsonl(path: Path, record: Mapping) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
class DatasetWriter:
    """Episode lifecycle + LeRobot-style persistence (single-threaded)."""

    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        *,
        fps: float = 30.0,
        image_shape: tuple[int, int] = (480, 640),
        joint_names: Sequence[str] = JOINT_NAMES,
        encoder: VideoEncoder | None = None,
        table_store: TableStore | None = None,
        video_preference: str = "auto",
        table_preference: str = "auto",
        chunks_size: int = CHUNKS_SIZE,
        robot_type: str = "so101",
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if chunks_size <= 0:
            raise ValueError(f"chunks_size must be > 0, got {chunks_size}")
        if not repo_id or "/" in repo_id or "\\" in repo_id:
            raise ValueError(f"repo_id must be a plain folder name, got {repo_id!r}")

        self._dir = Path(root) / repo_id
        self._meta = self._dir / "meta"
        self._fps = float(fps)
        self._image_shape = tuple(int(v) for v in image_shape)
        self._joint_names = tuple(joint_names)
        self._chunks_size = chunks_size
        self._robot_type = robot_type
        self._encoder = encoder if encoder is not None else make_video_encoder(video_preference)
        self._store = table_store if table_store is not None else make_table_store(table_preference)

        for sub in ("meta", "data/chunk-000", "videos/chunk-000", ".staging"):
            (self._dir / sub).mkdir(parents=True, exist_ok=True)
        # Meta files exist from the start so tooling can rely on their paths
        # (they stay empty until the first committed episode).
        (self._meta / "episodes.jsonl").touch(exist_ok=True)
        (self._meta / "tasks.jsonl").touch(exist_ok=True)

        self._episodes: list[dict] = _read_jsonl(self._meta / "episodes.jsonl")
        self._tasks: list[str] = [
            record.get("task", "")
            for record in _read_jsonl(self._meta / "tasks.jsonl")
            if record.get("task")
        ]
        self._next_index = (max((e.get("episode_index", -1) for e in self._episodes), default=-1) + 1)
        self._total_frames = sum(int(e.get("length", 0)) for e in self._episodes)

        self._buffer = EpisodeBuffer(n_joints=len(self._joint_names))
        self._episode_index: int | None = None
        self._task: str | None = None
        self._ensure_info_json()

    # -- properties --------------------------------------------------------
    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def episode_active(self) -> bool:
        return self._episode_index is not None

    @property
    def active_episode_index(self) -> int | None:
        return self._episode_index

    @property
    def next_episode_index(self) -> int:
        return self._next_index

    @property
    def frame_count(self) -> int:
        return len(self._buffer)

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        return self._fps

    # -- lifecycle ---------------------------------------------------------
    def start_episode(self, task: str) -> int:
        """Open a new episode buffer and return its episode index."""
        if self._episode_index is not None:
            raise EpisodeStateError(
                f"episode {self._episode_index} is already in progress; "
                "call save_episode() or discard_episode() first"
            )
        if not task or not task.strip():
            raise ValueError("task must be a non-empty description")
        self._buffer.clear()
        self._episode_index = self._next_index
        self._task = task
        logger.info("episode %06d started (task: %s)", self._episode_index, task)
        return self._episode_index

    def append(self, sample: FrameSample) -> None:
        """Append one synchronized tick to the active episode."""
        if self._episode_index is None:
            raise EpisodeStateError("no episode in progress; call start_episode() first")
        self._buffer.append(sample)

    def save_episode(self, task_name: str, success: bool = True) -> dict:
        """Encode + atomically commit the active episode; returns the manifest.

        On failure the buffer and episode stay active so FINISH can be
        pressed again to retry without data loss.
        """
        if self._episode_index is None:
            raise EpisodeStateError("no episode in progress; nothing to save")
        if self._buffer.is_empty:
            raise EpisodeStateError("episode buffer is empty; refusing to save 0 frames")

        index = self._episode_index
        task = task_name.strip()
        if not task:
            raise ValueError("task_name must be a non-empty description")
        task_index = self._resolve_task_index(task)
        n_frames = len(self._buffer)
        staging = self._dir / ".staging" / f"episode_{index:06d}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)

        try:
            # 1) Encode video + table into staging.
            self._encoder.encode(self._buffer.base_frames, staging / "base.mp4", self._fps)
            self._encoder.encode(self._buffer.wrist_frames, staging / "wrist.mp4", self._fps)
            table_path = self._store.write(self._build_columns(index, task_index), staging / f"episode_{index:06d}")

            # 2) Move into the final layout (os.replace is atomic per file).
            chunk = index // self._chunks_size
            data_dir = self._dir / f"data/chunk-{chunk:03d}"
            video_dir = self._dir / f"videos/chunk-{chunk:03d}" / f"episode_{index:06d}"
            data_dir.mkdir(parents=True, exist_ok=True)
            video_dir.mkdir(parents=True, exist_ok=True)

            final_data = data_dir / table_path.name
            os.replace(table_path, final_data)
            final_videos: dict[str, str] = {}
            for key, staged_name in zip(VIDEO_KEYS, ("base.mp4", "wrist.mp4"), strict=True):
                target = video_dir / f"{key}.mp4"
                os.replace(staging / staged_name, target)
                # Manifests must be portable: always POSIX separators.
                final_videos[key] = target.relative_to(self._dir).as_posix()

            # 3) Commit: manifest append is the point of no return.
            manifest = {
                "episode_index": index,
                "tasks": [task],
                "task_index": task_index,
                "length": n_frames,
                "fps": self._fps,
                "duration": n_frames / self._fps,
                "success": bool(success),
                "data_file": final_data.relative_to(self._dir).as_posix(),
                "videos": final_videos,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if task not in self._tasks:
                _append_jsonl(self._meta / "tasks.jsonl", {"task": task, "task_index": task_index})
                self._tasks.append(task)
            _append_jsonl(self._meta / "episodes.jsonl", manifest)
            self._episodes.append(manifest)
            self._next_index += 1
            self._total_frames += n_frames
            self._write_info_json()
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            logger.error(
                "episode %06d save failed (%s) - buffer retained, press FINISH to retry",
                index,
                exc,
            )
            if isinstance(exc, DatasetWriteError):
                raise
            raise DatasetWriteError(f"failed to save episode {index}: {exc}") from exc

        shutil.rmtree(staging, ignore_errors=True)
        logger.info(
            "episode %06d saved (%d frames, %.1f s, success=%s)",
            index,
            n_frames,
            n_frames / self._fps,
            success,
        )
        self._buffer.clear()
        self._episode_index = None
        self._task = None
        return manifest

    def discard_episode(self) -> None:
        """Clear the active buffer immediately without touching the dataset."""
        index = self._episode_index
        if index is None:
            logger.debug("discard_episode() called with no episode in progress")
            return
        frames = len(self._buffer)
        shutil.rmtree(self._dir / ".staging" / f"episode_{index:06d}", ignore_errors=True)
        self._buffer.clear()
        self._episode_index = None
        self._task = None
        logger.info("episode %06d discarded (%d frames dropped, nothing written)", index, frames)

    # -- internals ---------------------------------------------------------
    def _resolve_task_index(self, task: str) -> int:
        if task in self._tasks:
            return self._tasks.index(task)
        return len(self._tasks)

    def _build_columns(self, episode_index: int, task_index: int) -> dict[str, np.ndarray]:
        n_frames = len(self._buffer)
        return {
            # Uniform 30 Hz grid - the control loop is drift-compensated, so
            # nominal timestamps are the correct dataset timeline.
            "timestamp": np.arange(n_frames, dtype=np.float64) / self._fps,
            "frame_index": np.arange(n_frames, dtype=np.int64),
            "episode_index": np.full(n_frames, episode_index, dtype=np.int64),
            "index": self._total_frames + np.arange(n_frames, dtype=np.int64),
            "task_index": np.full(n_frames, task_index, dtype=np.int64),
            "observation.state": self._buffer.states.astype(np.float32),
            "action": self._buffer.actions.astype(np.float32),
        }

    def _features(self) -> dict:
        joint_feature = {
            "dtype": "float32",
            "shape": [len(self._joint_names)],
            "names": list(self._joint_names),
        }
        height, width = self._image_shape
        video_feature = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
        }
        scalar = {"shape": [1], "names": None}
        return {
            "observation.state": dict(joint_feature),
            "action": dict(joint_feature),
            "observation.images.base": dict(video_feature),
            "observation.images.wrist": dict(video_feature),
            "timestamp": {"dtype": "float64", **scalar},
            "frame_index": {"dtype": "int64", **scalar},
            "episode_index": {"dtype": "int64", **scalar},
            "index": {"dtype": "int64", **scalar},
            "task_index": {"dtype": "int64", **scalar},
        }

    def _info_payload(self) -> dict:
        return {
            "robot_type": self._robot_type,
            "fps": self._fps,
            "chunks_size": self._chunks_size,
            "total_episodes": len(self._episodes),
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": 2 * len(self._episodes),
            "data_file_pattern": "data/chunk-{chunk:03d}/episode_{episode_index:06d}"
            + self._store.suffix,
            "video_file_patterns": [
                "videos/chunk-{chunk:03d}/episode_{episode_index:06d}/{video_key}.mp4"
            ],
            "features": self._features(),
        }

    def _ensure_info_json(self) -> None:
        info_path = self._meta / "info.json"
        if not info_path.exists():
            _atomic_write_json(info_path, self._info_payload())

    def _write_info_json(self) -> None:
        _atomic_write_json(self._meta / "info.json", self._info_payload())


def iter_episode_files(dataset_dir: str | Path, manifest: Mapping) -> Iterable[Path]:
    """Yield every file referenced by an episode manifest (verification aid)."""
    root = Path(dataset_dir)
    yield root / str(manifest["data_file"])
    for rel_path in manifest.get("videos", {}).values():
        yield root / rel_path
