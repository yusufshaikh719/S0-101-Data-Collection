"""Asynchronous adapter for the official LeRobot v3 dataset writer."""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Sequence

import numpy as np

from so101_collector.config import JOINT_NAMES
from so101_collector.core.dataset_writer import FrameSample
from so101_collector.exceptions import DatasetWriteError, EpisodeStateError

logger = logging.getLogger(__name__)


class LeRobotWriter:
    """Serialize samples on a worker so episode commits do not stop acquisition.

    The bounded queue fails closed if disk or encoding cannot sustain the
    capture rate; it never drops samples while pretending to maintain the fps.
    """

    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        *,
        fps: int = 30,
        image_shapes: dict[str, tuple[int, int]],
        joint_names: Sequence[str] = JOINT_NAMES,
        queue_size: int = 120,
        dataset_class=None,
    ) -> None:
        if not repo_id or any(part in ("", ".", "..") for part in repo_id.split("/")) or "\\" in repo_id:
            raise ValueError("repo_id must be a dataset name or namespace/name")
        if len(repo_id.split("/")) > 2 or Path(repo_id).is_absolute() or ":" in repo_id:
            raise ValueError("repo_id must be a dataset name or namespace/name")
        if not isinstance(fps, int) or fps <= 0 or queue_size <= 0:
            raise ValueError("fps and queue_size must be positive integers")
        if set(image_shapes) != {"base", "wrist"}:
            raise ValueError("image_shapes must contain base and wrist")
        self._dir = Path(root).joinpath(*repo_id.split("/"))
        self._fps = fps
        self._image_shapes = {name: tuple(shape) for name, shape in image_shapes.items()}
        self._joint_names = tuple(joint_names)
        self._task: str | None = None
        self._frame_count = 0
        self._active = False
        self._error: Exception | None = None
        self._closed = False

        if dataset_class is None:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset
            except ImportError as exc:
                raise DatasetWriteError(
                    "LeRobot dataset dependencies are required; install the project with pip install -e ."
                ) from exc
            dataset_class = LeRobotDataset

        features = {
            key: {"dtype": "float32", "shape": (len(self._joint_names),), "names": list(self._joint_names)}
            for key in ("observation.state", "action")
        }
        features.update({
            f"observation.images.{name}": {
                "dtype": "video", "shape": (*shape, 3), "names": ["height", "width", "channels"]
            }
            for name, shape in self._image_shapes.items()
        })
        info_path = self._dir / "meta" / "info.json"
        try:
            if self._dir.exists():
                if not info_path.is_file():
                    raise DatasetWriteError(f"existing directory is not a LeRobot dataset: {self._dir}")
                info = json.loads(info_path.read_text(encoding="utf-8"))
                if info.get("codebase_version") != "v3.0" or info.get("fps") != fps:
                    raise DatasetWriteError("existing dataset is not LeRobot v3 or has a different fps")
                for key, expected in features.items():
                    actual = info.get("features", {}).get(key, {})
                    if (actual.get("dtype") != expected["dtype"]
                            or tuple(actual.get("shape", ())) != expected["shape"]
                            or actual.get("names") != expected["names"]):
                        raise DatasetWriteError(f"existing dataset feature does not match: {key}")
                self._dataset = dataset_class.resume(repo_id=repo_id, root=self._dir)
            else:
                self._dir.parent.mkdir(parents=True, exist_ok=True)
                self._dataset = dataset_class.create(
                    repo_id=repo_id, root=self._dir, fps=fps, features=features,
                    robot_type="so101_follower", use_videos=True,
                    metadata_buffer_size=1,
                )
        except DatasetWriteError:
            raise
        except Exception as exc:
            raise DatasetWriteError(f"cannot open LeRobot dataset {self._dir}: {exc}") from exc

        self._next_index = self._dataset.meta.total_episodes
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="LeRobotDatasetWriter", daemon=True)
        self._thread.start()

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def episode_active(self) -> bool:
        return self._active

    @property
    def active_episode_index(self) -> int | None:
        return self._next_index if self._active else None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def queue_capacity(self) -> int:
        return self._queue.maxsize

    def _check_error(self) -> None:
        if self._error is not None:
            raise DatasetWriteError(f"LeRobot writer failed: {self._error}") from self._error
        if self._closed:
            raise DatasetWriteError("LeRobot writer is closed")

    def _enqueue(self, operation: str, payload=None) -> None:
        self._check_error()
        try:
            self._queue.put_nowait((operation, payload))
        except queue.Full as exc:
            raise DatasetWriteError("dataset writer cannot keep up; capture stopped without dropping frames") from exc

    def start_episode(self, task: str) -> int:
        self._check_error()
        if self._active:
            raise EpisodeStateError("episode already in progress")
        if not task.strip():
            raise ValueError("task must be a non-empty description")
        self._task = task
        self._active = True
        self._frame_count = 0
        return self._next_index

    def append(self, sample: FrameSample) -> None:
        if not self._active:
            raise EpisodeStateError("no episode in progress")
        state = np.asarray(sample.state, dtype=np.float32).reshape(-1)
        action = np.asarray(sample.action, dtype=np.float32).reshape(-1)
        if state.shape != (len(self._joint_names),) or action.shape != state.shape:
            raise DatasetWriteError("state/action joint count does not match dataset schema")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise DatasetWriteError("state/action contains non-finite values")
        frame = {"observation.state": state.copy(), "action": action.copy(), "task": self._task}
        for name in ("base", "wrist"):
            image = np.asarray(getattr(sample, name))
            if image.dtype != np.uint8 or image.shape != (*self._image_shapes[name], 3):
                raise DatasetWriteError(f"{name} camera frame does not match its configured RGB shape")
            frame[f"observation.images.{name}"] = image.copy()
        self._enqueue("frame", frame)
        self._frame_count += 1

    def save_episode(self, task_name: str, success: bool = True) -> None:
        del success  # Returning home does not imply success.
        if not self._active or self._frame_count == 0:
            raise EpisodeStateError("cannot save an empty or inactive episode")
        if task_name != self._task:
            raise EpisodeStateError("episode task cannot change while recording")
        self._enqueue("save")
        self._active = False
        self._frame_count = 0
        self._next_index += 1

    def discard_episode(self) -> None:
        if self._active:
            self._enqueue("discard")
            self._active = False
            self._frame_count = 0

    def _run(self) -> None:
        while True:
            operation, payload = self._queue.get()
            try:
                if operation == "close":
                    try:
                        self._dataset.finalize()
                    except Exception as exc:
                        if self._error is None:
                            self._error = exc
                    return
                if self._error is not None:
                    continue
                if operation == "frame":
                    self._dataset.add_frame(payload)
                elif operation == "save":
                    self._dataset.save_episode(parallel_encoding=False)
                elif operation == "discard":
                    self._dataset.clear_episode_buffer()
            except Exception as exc:
                self._error = exc
                logger.exception("LeRobot dataset writer failed")
            finally:
                self._queue.task_done()

    def drain(self) -> None:
        """Wait for queued work and surface any asynchronous writer error."""
        self._queue.join()
        self._check_error()

    def close(self) -> None:
        if self._closed:
            if self._error is not None:
                raise DatasetWriteError(f"LeRobot writer failed: {self._error}") from self._error
            return
        self._queue.put(("close", None))
        self._thread.join()
        self._closed = True
        if self._error is not None:
            raise DatasetWriteError(f"LeRobot writer failed: {self._error}") from self._error
