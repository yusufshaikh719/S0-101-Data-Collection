"""Interactive rig configuration and pre-flight verification.

Examples::

    # Discover devices, pick follower/leader/cameras, rewrite the YAMLs, verify
    so101-setup

    # Print candidate devices only (no prompts, no writes)
    so101-setup --list

    # Verify the current configuration (no prompts)
    so101-setup --check

    # Same, saving one frame per camera for a field-of-view check
    so101-setup --check --snapshot-dir ./camera_snapshots

    # Refuse to pass unless the arm is currently in its tucked-in pose
    so101-setup --check --require-home
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from so101_collector.config import RobotConfig, VisionConfig, default_config_dir
from so101_collector.core.camera_hub import CameraHub, V4L2Camera
from so101_collector.core.hardware_interface import JointState, SO101HardwareInterface
from so101_collector.exceptions import SO101Error

logger = logging.getLogger("so101.setup")

SERIAL_DIR = Path("/dev/serial/by-id")
VIDEO_DIR = Path("/dev/v4l/by-id")
MIN_PYTHON = (3, 12)

#: Import name -> install hint. LeRobot is required by so101-record.
REQUIRED_MODULES: dict[str, str] = {
    "serial": "pip install pyserial",
    "cv2": "pip install opencv-python",
    "numpy": "pip install numpy",
    "yaml": "pip install PyYAML",
    "lerobot": "pip install -e .  (installs lerobot[dataset])",
}
OPTIONAL_MODULES: dict[str, str] = {
    "av": "PyAV encoding (pip install av)",
    "pyarrow": "Parquet tables (pip install pyarrow)",
}


@dataclass(frozen=True)
class Device:
    """One ``/dev/*/by-id`` entry."""

    name: str
    path: str
    target: str


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # PASS / WARN / FAIL / SKIP
    detail: str


# ---------------------------------------------------------------------------
# Device discovery and YAML editing
# ---------------------------------------------------------------------------
def scan_by_id(
    directory: Path, *, include: Callable[[str], bool] | None = None
) -> list[Device]:
    """List ``/dev/{serial,v4l}/by-id`` entries sorted by path."""
    try:
        entries = sorted(directory.iterdir(), key=lambda entry: entry.name)
    except OSError:
        return []
    devices: list[Device] = []
    for entry in entries:
        if include is not None and not include(entry.name):
            continue
        try:
            target = str(entry.resolve())
        except OSError:  # pragma: no cover - broken symlink on hardware hosts
            target = str(entry)
        devices.append(Device(name=entry.name, path=str(entry), target=target))
    return devices


def discover_serial_devices(directory: Path = SERIAL_DIR) -> list[Device]:
    return scan_by_id(directory)


def discover_video_devices(directory: Path = VIDEO_DIR) -> list[Device]:
    # Keep only the capture node; -video-index1 is the metadata interface.
    return scan_by_id(directory, include=lambda name: name.endswith("-video-index0"))


_YAML_KEY = re.compile(
    r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<rest>[^\n]*)$", re.MULTILINE
)


def replace_yaml_scalar(text: str, key: str, value: str, *, occurrence: int = 0) -> str:
    """Replace the ``occurrence``-th ``key: ...`` line, keeping indentation/comments."""
    matches = [
        match for match in _YAML_KEY.finditer(text) if match.group("key") == key
    ]
    if len(matches) <= occurrence:
        raise ValueError(f"key '{key}' occurrence {occurrence} not found in config")
    match = matches[occurrence]
    rest = match.group("rest")
    comment = f" {rest[rest.index('#'):].strip()}" if "#" in rest else ""
    replacement = f"{match.group('indent')}{key}: {value}{comment}"
    return text[: match.start()] + replacement + text[match.end() :]


def update_config_file(path: Path, updates: Sequence[tuple[str, str, int]]) -> None:
    """Apply ``(key, value, occurrence)`` edits atomically."""
    text = path.read_text(encoding="utf-8")
    for key, value, occurrence in updates:
        text = replace_yaml_scalar(text, key, value, occurrence=occurrence)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def print_devices() -> None:
    for label, directory, devices in (
        ("serial (arms)", SERIAL_DIR, discover_serial_devices()),
        ("video (cameras)", VIDEO_DIR, discover_video_devices()),
    ):
        print(f"{label} in {directory}:")
        if not devices:
            print("  (none found)")
            continue
        for index, device in enumerate(devices, start=1):
            target = f" -> {device.target}" if device.target != device.path else ""
            print(f"  {index}) {device.name}{target}")
    if not SERIAL_DIR.exists() and not VIDEO_DIR.exists():
        print(
            "\nNo by-id device directories on this host: run the rig on Linux, or point\n"
            "--robot-config/--camera-config at a file with your own device paths."
        )


def _prompt_choice(
    title: str, devices: Sequence[Device], current: str | None, *, allow_none: bool = False
) -> str | None:
    if not devices:
        print(f"{title}: no candidates found; keeping {current or 'unset'}")
        return current
    print(f"{title}:")
    current_index = None
    for index, device in enumerate(devices, start=1):
        marker = "   <- current" if current and device.path == current else ""
        print(f"  {index}) {device.name} -> {device.target}{marker}")
        if current and device.path == current:
            current_index = index
    if allow_none:
        print("  0) none")
    hint = f" [{current_index}]" if current_index else " [Enter = keep]"
    while True:
        raw = input(f"  choose 1-{len(devices)}{hint}: ").strip()
        if not raw:
            return devices[current_index - 1].path if current_index else current
        if raw == "0" and allow_none:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(devices):
            return devices[int(raw) - 1].path
        print("  invalid choice, try again")


def _configure(
    robot_path: Path, camera_path: Path, *, dry_run: bool
) -> tuple[RobotConfig, VisionConfig]:
    """Prompt for every device path, validate, then write both YAMLs."""
    if not sys.stdin.isatty():
        raise SO101Error("interactive setup needs a TTY; use --list or --check instead")
    robot = RobotConfig.from_yaml(robot_path)
    vision = VisionConfig.from_yaml(camera_path)

    print_devices()
    print()
    serials = discover_serial_devices()
    videos = discover_video_devices()

    follower = _prompt_choice("Follower arm serial port", serials, robot.port)
    remaining = [device for device in serials if device.path != follower]
    leader = _prompt_choice(
        "Leader arm serial port (optional)", remaining, robot.leader_port, allow_none=True
    )

    chosen: set[str] = set()
    camera_paths: dict[str, str] = {}
    for camera in vision.cameras:
        candidates = [device for device in videos if device.path not in chosen]
        selected = _prompt_choice(
            f"{camera.name} camera", candidates, camera.device_path
        )
        camera_paths[camera.name] = selected or camera.device_path
        chosen.add(camera_paths[camera.name])

    # Validate before touching the files.
    updated_robot = replace(robot, port=follower or robot.port, leader_port=leader)
    updated_vision = VisionConfig(
        cameras=tuple(
            replace(camera, device_path=camera_paths[camera.name]) for camera in vision.cameras
        ),
        sync_tolerance_ms=vision.sync_tolerance_ms,
    )

    print("\nSelected:")
    print(f"  follower: {updated_robot.port}")
    print(f"  leader:   {updated_robot.leader_port or '(none)'}")
    for camera in updated_vision.cameras:
        print(f"  {camera.name:<8}: {camera.device_path}")

    if dry_run:
        print("\n(dry run - nothing written)")
        return updated_robot, updated_vision

    update_config_file(
        robot_path,
        [
            ("port", updated_robot.port, 0),
            ("leader_port", updated_robot.leader_port or "null", 0),
        ],
    )
    camera_updates = [
        ("device_path", camera.device_path, index)
        for index, camera in enumerate(updated_vision.cameras)
    ]
    update_config_file(camera_path, camera_updates)
    # Prove the written files still parse.
    RobotConfig.from_yaml(robot_path)
    VisionConfig.from_yaml(camera_path)
    print(f"\nWrote {robot_path} and {camera_path}")
    return updated_robot, updated_vision


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installation
        return False


def check_dependencies() -> list[CheckResult]:
    results: list[CheckResult] = []
    version = ".".join(str(part) for part in sys.version_info[:3])
    required = f">= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    if sys.version_info >= MIN_PYTHON:
        results.append(CheckResult("python", "PASS", f"{version} ({required})"))
    else:
        results.append(CheckResult("python", "FAIL", f"{version}; requires {required}"))
    for module, hint in REQUIRED_MODULES.items():
        if _module_available(module):
            results.append(CheckResult(f"import {module}", "PASS", "available"))
        else:
            results.append(CheckResult(f"import {module}", "FAIL", f"missing: {hint}"))
    for module, hint in OPTIONAL_MODULES.items():
        if _module_available(module):
            results.append(CheckResult(f"import {module}", "PASS", "available"))
        else:
            results.append(CheckResult(f"import {module}", "WARN", f"missing: {hint}"))
    return results


def check_disk(root: Path, min_free_gb: float) -> CheckResult:
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free_gb = shutil.disk_usage(probe).free / (1024**3)
    except OSError as exc:
        return CheckResult("disk space", "WARN", f"cannot inspect {probe}: {exc}")
    detail = f"{free_gb:.1f} GiB free at {probe} (need >= {min_free_gb:.1f})"
    return CheckResult("disk space", "PASS" if free_gb >= min_free_gb else "FAIL", detail)


def check_device_paths(robot: RobotConfig, vision: VisionConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    targets = [("follower", robot.port)]
    if robot.leader_port:
        targets.append(("leader", robot.leader_port))
    targets.extend((camera.name, camera.device_path) for camera in vision.cameras)

    for name, path in targets:
        if not path.startswith("/dev/"):
            results.append(
                CheckResult(f"device {name}", "SKIP", f"{path} (not a /dev path; bench setup)")
            )
        elif Path(path).exists():
            results.append(CheckResult(f"device {name}", "PASS", path))
        else:
            results.append(CheckResult(f"device {name}", "FAIL", f"{path} does not exist"))

    paths = [camera.device_path for camera in vision.cameras]
    if len(set(paths)) != len(paths):
        results.append(
            CheckResult("camera identity", "FAIL", "both cameras point at the same device")
        )
    return results


def check_serial(robot: RobotConfig) -> tuple[CheckResult, JointState | None]:
    interface = SO101HardwareInterface(robot)
    try:
        interface.open()
        state = interface.read_state()
    except SO101Error as exc:
        return CheckResult("serial bus", "FAIL", str(exc)), None
    finally:
        interface.close()
    return (
        CheckResult(
            "serial bus",
            "PASS",
            f"{len(robot.joints)} servos responded on {robot.port}",
        ),
        state,
    )


def check_leader(robot: RobotConfig) -> CheckResult:
    if robot.leader_port is None:
        return CheckResult(
            "leader arm",
            "WARN",
            "leader_port is unset: --teleop leader will refuse to start "
            "(set it, or use --teleop spacenav)",
        )
    interface = SO101HardwareInterface(replace(robot, port=robot.leader_port))
    try:
        interface.open()
        state = interface.read_state()
    except SO101Error as exc:
        return CheckResult("leader arm", "FAIL", str(exc))
    finally:
        interface.close()
    return CheckResult(
        "leader arm",
        "PASS",
        f"6 servos responded on {robot.leader_port} "
        f"(pan {state.positions[0]:.1f} deg)",
    )


def check_home_pose(
    robot: RobotConfig, state: JointState | None, tolerance: float, *, require_home: bool
) -> CheckResult:
    if state is None:
        return CheckResult("home pose", "SKIP", "no joint state (serial check did not run)")
    error = np.abs(state.positions - np.asarray(robot.home_pose, dtype=np.float64))
    worst = int(np.argmax(error))
    detail = (
        f"max joint error {error[worst]:.1f} deg at {robot.joints[worst].name} "
        f"(home detection needs <= {tolerance:.1f} deg on all six joints)"
    )
    if np.all(error <= tolerance):
        return CheckResult("home pose", "PASS", f"in the tucked pose; {detail}")
    if require_home:
        return CheckResult("home pose", "FAIL", f"not in the tucked pose; {detail}")
    return CheckResult("home pose", "WARN", f"not in the tucked pose now; {detail}")


def _save_snapshot(name: str, frame: np.ndarray, directory: Path) -> None:
    try:
        import cv2  # noqa: PLC0415 - optional at import time
    except ImportError:  # pragma: no cover - cv2 is a hard runtime dependency
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_preflight.png"
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"  [----] saved {path}")


def check_cameras(vision: VisionConfig, snapshot_dir: Path | None) -> list[CheckResult]:
    results: list[CheckResult] = []
    hub = CameraHub(vision, source_factory=V4L2Camera)
    frames: dict[str, object] = {}
    try:
        hub.start()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and len(frames) < len(hub.names):
            for name in hub.names:
                if name in frames:
                    continue
                try:
                    frames[name] = hub.get_frame(name)
                except SO101Error:
                    continue
            if len(frames) < len(hub.names):
                time.sleep(0.05)

        stats = hub.stats().cameras
        for name in hub.names:
            packet = frames.get(name)
            if packet is None:
                results.append(
                    CheckResult(
                        f"camera {name}",
                        "FAIL",
                        str(stats.get(name, {}).get("error") or "no frame within 8 s"),
                    )
                )
                continue
            frame = packet.frame
            mean = float(frame.astype(np.float32).mean())
            detail = f"{frame.shape[1]}x{frame.shape[0]} RGB, mean brightness {mean:.0f}"
            if mean < 8.0:
                results.append(
                    CheckResult(
                        f"camera {name}",
                        "WARN",
                        f"{detail} (lens covered or no light?)",
                    )
                )
            else:
                results.append(CheckResult(f"camera {name}", "PASS", detail))
            if snapshot_dir is not None:
                _save_snapshot(name, frame, snapshot_dir)
    except SO101Error as exc:
        results.append(CheckResult("cameras", "FAIL", str(exc)))
    finally:
        hub.stop()
    return results


def _print_results(results: Sequence[CheckResult]) -> None:
    for result in results:
        print(f"  [{result.status:<4}] {result.name}: {result.detail}")


def run_checks(
    robot_path: Path,
    camera_path: Path,
    *,
    output_dir: Path,
    min_free_gb: float,
    home_tolerance: float,
    require_home: bool,
    snapshot_dir: Path | None,
    skip_serial: bool,
    skip_cameras: bool,
) -> bool:
    print_config = f"{robot_path.name}, {camera_path.name}"
    print("Pre-flight checks:")
    results = check_dependencies()
    try:
        robot: RobotConfig | None = RobotConfig.from_yaml(robot_path)
        vision: VisionConfig | None = VisionConfig.from_yaml(camera_path)
    except SO101Error as exc:
        results.append(CheckResult("config", "FAIL", str(exc)))
        robot = vision = None
    else:
        results.append(CheckResult("config", "PASS", print_config))
    results.append(check_disk(output_dir, min_free_gb))

    if robot is not None and vision is not None:
        results.extend(check_device_paths(robot, vision))
        state: JointState | None = None
        if skip_serial:
            results.append(CheckResult("serial bus", "SKIP", "--skip-serial"))
            results.append(CheckResult("leader arm", "SKIP", "--skip-serial"))
        else:
            serial_result, state = check_serial(robot)
            results.append(serial_result)
            results.append(check_leader(robot))
        results.append(check_home_pose(robot, state, home_tolerance, require_home=require_home))
        if skip_cameras:
            results.append(CheckResult("cameras", "SKIP", "--skip-cameras"))
        else:
            results.extend(check_cameras(vision, snapshot_dir))

    _print_results(results)
    failures = [result for result in results if result.status == "FAIL"]
    warnings = [result for result in results if result.status == "WARN"]
    if failures:
        print(f"\n{len(failures)} check(s) FAILED, {len(warnings)} warning(s)")
        return False
    print(f"\nALL CHECKS PASSED ({len(warnings)} warning(s))")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="so101-setup",
        description="Configure SO-101 device paths and verify the rig before recording.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--list", action="store_true", help="print discovered devices and exit")
    parser.add_argument(
        "--check", action="store_true", help="verify the current config without prompting"
    )
    parser.add_argument("--dry-run", action="store_true", help="show selections without writing")
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the pre-flight checks after writing"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--home-tolerance", type=float, default=5.0, metavar="DEGREES")
    parser.add_argument(
        "--require-home",
        action="store_true",
        help="fail unless the arm is currently within --home-tolerance of home_deg",
    )
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--skip-serial", action="store_true")
    parser.add_argument("--skip-cameras", action="store_true")
    parser.add_argument(
        "--log-level", default="WARNING", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        config_dir = args.config_dir or default_config_dir()
    except SO101Error as exc:
        print(f"[FAIL] {exc}")
        return 1
    robot_path = args.robot_config or config_dir / "robot_config.yaml"
    camera_path = args.camera_config or config_dir / "camera_config.yaml"

    if args.list or (not args.check and not sys.stdin.isatty()):
        print_devices()
        return 0
    if args.check:
        return (
            0
            if run_checks(
                robot_path,
                camera_path,
                output_dir=args.output_dir,
                min_free_gb=args.min_free_gb,
                home_tolerance=args.home_tolerance,
                require_home=args.require_home,
                snapshot_dir=args.snapshot_dir,
                skip_serial=args.skip_serial,
                skip_cameras=args.skip_cameras,
            )
            else 1
        )

    try:
        _configure(robot_path, camera_path, dry_run=args.dry_run)
    except (SO101Error, ValueError, OSError) as exc:
        print(f"[FAIL] {exc}")
        return 1
    if args.dry_run or args.no_verify:
        return 0
    return (
        0
        if run_checks(
            robot_path,
            camera_path,
            output_dir=args.output_dir,
            min_free_gb=args.min_free_gb,
            home_tolerance=args.home_tolerance,
            require_home=args.require_home,
            snapshot_dir=args.snapshot_dir,
            skip_serial=args.skip_serial,
            skip_cameras=args.skip_cameras,
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
