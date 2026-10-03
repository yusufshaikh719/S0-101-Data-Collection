"""Configuration models and YAML loaders.

The pipeline addresses hardware through *persistent* device identifiers only
(``/dev/serial/by-id/*``, ``/dev/v4l/by-id/*``) so a reboot or a re-plugged
USB hub can never silently swap the follower arm with the leader arm, or the
base camera with the wrist camera (zero index-drift requirement).  Loading a
config that contains a dynamic path such as ``/dev/video0`` raises
:class:`~so101_collector.exceptions.ConfigError` unless the config explicitly
opts out with ``allow_dynamic_*: true``.

The module intentionally depends only on the standard library and PyYAML so
it can be imported without cameras, serial adapters, or OpenCV present.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from so101_collector.exceptions import ConfigError

#: Canonical joint order for the SO-101 6-DOF arm (5 joints + gripper).
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

_PERSISTENT_SERIAL_PREFIXES = ("/dev/serial/by-id/", "/dev/serial/by-path/")
_DYNAMIC_SERIAL_PATTERNS = (re.compile(r"/dev/ttyUSB\d+"), re.compile(r"/dev/ttyACM\d+"))
_PERSISTENT_VIDEO_PREFIXES = ("/dev/v4l/by-id/", "/dev/v4l/by-path/")
_DYNAMIC_VIDEO_PATTERN = re.compile(r"/dev/video\d+")

_MISSING = object()


def _validate_persistent_path(
    path: str,
    *,
    kind: str,
    persistent_prefixes: tuple[str, ...],
    dynamic_patterns: tuple[re.Pattern[str], ...],
    allow_dynamic: bool,
    bypass_flag: str,
) -> None:
    """Reject volatile device paths (``/dev/video0``, ``/dev/ttyUSB0``, ...)."""
    if not isinstance(path, str) or not path.strip():
        raise ConfigError(f"{kind} must be a non-empty string, got {path!r}")
    if allow_dynamic:
        return
    is_dynamic = any(p.match(path) for p in dynamic_patterns)
    is_unspecified_dev = path.startswith("/dev/") and not path.startswith(persistent_prefixes)
    if is_dynamic or is_unspecified_dev:
        raise ConfigError(
            f"{kind} '{path}' is a dynamic device path; use a stable persistent identifier "
            f"({' or '.join(persistent_prefixes)}) or set {bypass_flag}: true to override"
        )


def _as_mapping(data: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{context}: expected a mapping, got {type(data).__name__}")
    return data


def _reject_unknown_keys(data: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"{context}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}")


def _get(data: Mapping[str, Any], key: str, context: str, default: Any = _MISSING) -> Any:
    if key in data:
        return data[key]
    if default is not _MISSING:
        return default
    raise ConfigError(f"{context}: missing required key '{key}'")


def _as_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{context}: expected a non-empty string, got {value!r}")
    return value


def _as_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context}: expected a number, got {value!r}")
    return float(value)


def _as_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context}: expected an integer, got {value!r}")
    return int(value)


def _as_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context}: expected a boolean, got {value!r}")
    return value


@dataclass(frozen=True)
class JointSpec:
    """Static limits and calibration for one Feetech-driven joint."""

    name: str
    servo_id: int
    min_deg: float
    max_deg: float
    home_deg: float
    offset_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.min_deg >= self.max_deg:
            raise ConfigError(f"joint '{self.name}': min_deg ({self.min_deg}) must be < max_deg ({self.max_deg})")
        if not self.min_deg <= self.home_deg <= self.max_deg:
            raise ConfigError(
                f"joint '{self.name}': home_deg ({self.home_deg}) outside [{self.min_deg}, {self.max_deg}]"
            )
        if not 1 <= self.servo_id <= 253:
            raise ConfigError(f"joint '{self.name}': servo_id {self.servo_id} outside 1..253")

    def clamp(self, value: float) -> float:
        """Safety clamp against floor collisions and motor stall."""
        return min(max(value, self.min_deg), self.max_deg)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, context: str) -> JointSpec:
        data = _as_mapping(data, context)
        allowed = {"name", "servo_id", "min_deg", "max_deg", "home_deg", "offset_deg"}
        _reject_unknown_keys(data, allowed, context)
        return cls(
            name=_as_str(_get(data, "name", context), f"{context}.name"),
            servo_id=_as_int(_get(data, "servo_id", context), f"{context}.servo_id"),
            min_deg=_as_float(_get(data, "min_deg", context), f"{context}.min_deg"),
            max_deg=_as_float(_get(data, "max_deg", context), f"{context}.max_deg"),
            home_deg=_as_float(_get(data, "home_deg", context), f"{context}.home_deg"),
            offset_deg=_as_float(_get(data, "offset_deg", 0.0), f"{context}.offset_deg"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "servo_id": self.servo_id,
            "min_deg": self.min_deg,
            "max_deg": self.max_deg,
            "home_deg": self.home_deg,
            "offset_deg": self.offset_deg,
        }


@dataclass(frozen=True)
class KinematicsConfig:
    """Link geometry used by the analytical IK (millimetres, floor at z = 0)."""

    shoulder_height_mm: float = 118.0
    upper_arm_mm: float = 117.0
    forearm_mm: float = 135.0
    tool_length_mm: float = 65.0
    floor_clearance_mm: float = 2.0
    reach_margin_mm: float = 1.0

    def __post_init__(self) -> None:
        for name in ("shoulder_height_mm", "upper_arm_mm", "forearm_mm", "tool_length_mm"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"kinematics.{name} must be > 0, got {getattr(self, name)}")
        if self.floor_clearance_mm < 0:
            raise ConfigError(f"kinematics.floor_clearance_mm must be >= 0, got {self.floor_clearance_mm}")
        if not 0 <= self.reach_margin_mm < min(self.upper_arm_mm, self.forearm_mm):
            raise ConfigError("kinematics.reach_margin_mm must be in [0, min(upper_arm_mm, forearm_mm))")

    @property
    def max_reach_mm(self) -> float:
        return self.upper_arm_mm + self.forearm_mm + self.tool_length_mm

    @classmethod
    def from_dict(cls, data: Any) -> KinematicsConfig:
        if data is None:
            return cls()
        data = _as_mapping(data, "kinematics")
        allowed = {
            "shoulder_height_mm",
            "upper_arm_mm",
            "forearm_mm",
            "tool_length_mm",
            "floor_clearance_mm",
            "reach_margin_mm",
        }
        _reject_unknown_keys(data, allowed, "kinematics")
        kwargs = {k: _as_float(v, f"kinematics.{k}") for k, v in data.items()}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, float]:
        return {
            "shoulder_height_mm": self.shoulder_height_mm,
            "upper_arm_mm": self.upper_arm_mm,
            "forearm_mm": self.forearm_mm,
            "tool_length_mm": self.tool_length_mm,
            "floor_clearance_mm": self.floor_clearance_mm,
            "reach_margin_mm": self.reach_margin_mm,
        }


@dataclass(frozen=True)
class RobotConfig:
    """Follower (or leader) arm bus + joint configuration."""

    port: str
    joints: tuple[JointSpec, ...]
    baudrate: int = 1_000_000
    timeout_s: float = 0.05
    max_retries: int = 3
    retry_backoff_s: float = 0.002
    allow_dynamic_port: bool = False
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    leader_port: str | None = None
    leader_offset_deg: tuple[float, ...] = (0.0,) * 6

    def __post_init__(self) -> None:
        _validate_persistent_path(
            self.port,
            kind="serial port",
            persistent_prefixes=_PERSISTENT_SERIAL_PREFIXES,
            dynamic_patterns=_DYNAMIC_SERIAL_PATTERNS,
            allow_dynamic=self.allow_dynamic_port,
            bypass_flag="allow_dynamic_port",
        )
        if self.leader_port is not None:
            _validate_persistent_path(
                self.leader_port,
                kind="leader serial port",
                persistent_prefixes=_PERSISTENT_SERIAL_PREFIXES,
                dynamic_patterns=_DYNAMIC_SERIAL_PATTERNS,
                allow_dynamic=self.allow_dynamic_port,
                bypass_flag="allow_dynamic_port",
            )
        names = tuple(j.name for j in self.joints)
        if names != JOINT_NAMES:
            raise ConfigError(
                f"joint order/names must be exactly {list(JOINT_NAMES)}, got {list(names)}"
            )
        ids = [j.servo_id for j in self.joints]
        if len(set(ids)) != len(ids):
            raise ConfigError(f"duplicate servo ids in joints: {ids}")
        if self.baudrate <= 0:
            raise ConfigError(f"baudrate must be > 0, got {self.baudrate}")
        if self.timeout_s <= 0:
            raise ConfigError(f"timeout_s must be > 0, got {self.timeout_s}")
        if self.max_retries < 0:
            raise ConfigError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.retry_backoff_s < 0:
            raise ConfigError(f"retry_backoff_s must be >= 0, got {self.retry_backoff_s}")
        if len(self.leader_offset_deg) != len(JOINT_NAMES):
            raise ConfigError(f"leader_offset_deg must have {len(JOINT_NAMES)} entries")

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def home_pose(self) -> tuple[float, ...]:
        return tuple(j.home_deg for j in self.joints)

    @property
    def joint_by_name(self) -> dict[str, JointSpec]:
        return {j.name: j for j in self.joints}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RobotConfig:
        data = _as_mapping(data, "robot config")
        allowed = {
            "port",
            "baudrate",
            "timeout_s",
            "max_retries",
            "retry_backoff_s",
            "allow_dynamic_port",
            "kinematics",
            "leader_port",
            "leader_offset_deg",
            "joints",
        }
        _reject_unknown_keys(data, allowed, "robot config")
        joints_raw = _get(data, "joints", "robot config")
        if not isinstance(joints_raw, (list, tuple)) or len(joints_raw) != len(JOINT_NAMES):
            raise ConfigError(
                f"robot config: 'joints' must be a list of {len(JOINT_NAMES)} joint mappings"
            )
        joints = tuple(
            JointSpec.from_dict(entry, context=f"joints[{i}]") for i, entry in enumerate(joints_raw)
        )
        offsets_raw = data.get("leader_offset_deg", [0.0] * len(JOINT_NAMES))
        if not isinstance(offsets_raw, (list, tuple)):
            raise ConfigError("robot config: leader_offset_deg must be a list of numbers")
        leader_port = data.get("leader_port")
        return cls(
            port=_as_str(_get(data, "port", "robot config"), "robot config.port"),
            joints=joints,
            baudrate=_as_int(data.get("baudrate", 1_000_000), "robot config.baudrate"),
            timeout_s=_as_float(data.get("timeout_s", 0.05), "robot config.timeout_s"),
            max_retries=_as_int(data.get("max_retries", 3), "robot config.max_retries"),
            retry_backoff_s=_as_float(data.get("retry_backoff_s", 0.002), "robot config.retry_backoff_s"),
            allow_dynamic_port=_as_bool(
                data.get("allow_dynamic_port", False), "robot config.allow_dynamic_port"
            ),
            kinematics=KinematicsConfig.from_dict(data.get("kinematics")),
            leader_port=None if leader_port is None else _as_str(leader_port, "robot config.leader_port"),
            leader_offset_deg=tuple(
                _as_float(v, "leader_offset_deg[]") for v in offsets_raw
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotConfig:
        return cls.from_dict(_load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "retry_backoff_s": self.retry_backoff_s,
            "allow_dynamic_port": self.allow_dynamic_port,
            "leader_port": self.leader_port,
            "leader_offset_deg": list(self.leader_offset_deg),
            "kinematics": self.kinematics.to_dict(),
            "joints": [j.to_dict() for j in self.joints],
        }


@dataclass(frozen=True)
class CameraConfig:
    """One V4L2 camera addressed by its persistent ``/dev/v4l/by-id`` path."""

    name: str
    device_path: str
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fourcc: str = "MJPG"
    exposure: float | None = None
    buffer_size: int = 1
    allow_dynamic_path: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ConfigError(f"camera name '{self.name}' must match [a-z][a-z0-9_]*")
        _validate_persistent_path(
            self.device_path,
            kind=f"camera '{self.name}' device path",
            persistent_prefixes=_PERSISTENT_VIDEO_PREFIXES,
            dynamic_patterns=(_DYNAMIC_VIDEO_PATTERN,),
            allow_dynamic=self.allow_dynamic_path,
            bypass_flag="allow_dynamic_path",
        )
        if self.width <= 0 or self.height <= 0:
            raise ConfigError(f"camera '{self.name}': width/height must be > 0")
        if self.fps <= 0:
            raise ConfigError(f"camera '{self.name}': fps must be > 0, got {self.fps}")
        if len(self.fourcc) != 4:
            raise ConfigError(f"camera '{self.name}': fourcc must be 4 characters, got '{self.fourcc}'")
        if self.buffer_size < 1:
            raise ConfigError(f"camera '{self.name}': buffer_size must be >= 1")

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) as reported to drivers."""
        return (self.width, self.height)

    @property
    def frame_shape(self) -> tuple[int, int]:
        """(height, width) - the numpy shape of incoming frames."""
        return (self.height, self.width)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, context: str) -> CameraConfig:
        data = _as_mapping(data, context)
        allowed = {
            "name",
            "device_path",
            "width",
            "height",
            "fps",
            "fourcc",
            "exposure",
            "buffer_size",
            "allow_dynamic_path",
        }
        _reject_unknown_keys(data, allowed, context)
        exposure = data.get("exposure")
        return cls(
            name=_as_str(_get(data, "name", context), f"{context}.name"),
            device_path=_as_str(_get(data, "device_path", context), f"{context}.device_path"),
            width=_as_int(data.get("width", 640), f"{context}.width"),
            height=_as_int(data.get("height", 480), f"{context}.height"),
            fps=_as_float(data.get("fps", 30.0), f"{context}.fps"),
            fourcc=_as_str(data.get("fourcc", "MJPG"), f"{context}.fourcc"),
            exposure=None if exposure is None else _as_float(exposure, f"{context}.exposure"),
            buffer_size=_as_int(data.get("buffer_size", 1), f"{context}.buffer_size"),
            allow_dynamic_path=_as_bool(
                data.get("allow_dynamic_path", False), f"{context}.allow_dynamic_path"
            ),
        )


@dataclass(frozen=True)
class VisionConfig:
    """Both cameras plus the software frame-synchronization tolerance."""

    cameras: tuple[CameraConfig, ...]
    sync_tolerance_ms: float = 20.0

    def __post_init__(self) -> None:
        if self.sync_tolerance_ms <= 0:
            raise ConfigError(f"sync_tolerance_ms must be > 0, got {self.sync_tolerance_ms}")
        names = [c.name for c in self.cameras]
        if len(set(names)) != len(names):
            raise ConfigError(f"duplicate camera names: {names}")
        missing = {"base", "wrist"} - set(names)
        if missing:
            raise ConfigError(f"vision config must define the {sorted(missing)} camera(s); found {names}")

    @property
    def by_name(self) -> dict[str, CameraConfig]:
        return {c.name: c for c in self.cameras}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VisionConfig:
        data = _as_mapping(data, "vision config")
        _reject_unknown_keys(data, {"cameras", "sync_tolerance_ms"}, "vision config")
        cameras_raw = _get(data, "cameras", "vision config")
        if not isinstance(cameras_raw, (list, tuple)) or not cameras_raw:
            raise ConfigError("vision config: 'cameras' must be a non-empty list")
        cameras = tuple(
            CameraConfig.from_dict(entry, context=f"cameras[{i}]")
            for i, entry in enumerate(cameras_raw)
        )
        return cls(
            cameras=cameras,
            sync_tolerance_ms=_as_float(
                data.get("sync_tolerance_ms", 20.0), "vision config.sync_tolerance_ms"
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> VisionConfig:
        return cls.from_dict(_load_yaml(path))


def _load_yaml(path: str | Path) -> Mapping[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        raise ConfigError(f"config file is empty: {path}")
    return _as_mapping(data, str(path))


def default_config_dir() -> Path:
    """Locate the shipped ``configs/`` directory.

    Search order: ``$SO101_CONFIG_DIR``, ``./configs`` (working directory),
    then the repository checkout that contains this package.
    """
    candidates: list[Path] = []
    env_dir = os.environ.get("SO101_CONFIG_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path.cwd() / "configs")
    candidates.append(Path(__file__).resolve().parent.parent / "configs")
    for candidate in candidates:
        if (candidate / "robot_config.yaml").is_file() and (candidate / "camera_config.yaml").is_file():
            return candidate
    searched = ", ".join(str(c) for c in candidates)
    raise ConfigError(
        f"could not locate configs/ directory (searched: {searched}); "
        "point --robot-config/--camera-config at the files or set SO101_CONFIG_DIR"
    )
