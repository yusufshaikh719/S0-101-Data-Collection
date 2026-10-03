"""Main CLI entry point: record demonstration episodes.

Examples::

    # Real hardware, leader-arm teleoperation, keyboard commands
    so101-record --teleop leader --repo-id floor_pick_session_01

    # SpaceMouse + gamepad trigger commands
    so101-record --teleop spacenav \\
        --spacenouse-device /dev/input/by-id/usb-3Dconnexion_SpaceMouse-event-if00 \\
        --gamepad-device /dev/input/by-id/usb-Logitech_F310-event-if00

    # Full pipeline on a machine with no robot attached (smoke test)
    so101-record --simulate --max-episodes 2 --duration 60

Episode protocol (see the specification, section 4.4)::

    Space / trigger : start (2 s warmup, then record at 30 Hz)
    Enter / green   : finish - atomically save MP4 + Parquet + manifest
    Esc   / red     : abort - clear RAM buffers, nothing written
    R               : reset arm to home pose
    Q               : quit the session
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from so101_collector.config import (
    RobotConfig,
    VisionConfig,
    default_config_dir,
)
from so101_collector.console import (
    CommandReader,
    CompositeCommandReader,
    GamepadCommandReader,
    KeyboardCommandReader,
    NullCommandReader,
    TimedCommandReader,
)
from so101_collector.core.camera_hub import CameraHub, V4L2Camera
from so101_collector.core.dataset_writer import DatasetWriter
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.ik import SO101Kinematics
from so101_collector.core.runner import DEFAULT_TASK, DataCollectionRunner
from so101_collector.core.teleop_controller import (
    CartesianTeleop,
    LeaderArmTeleop,
    TeleopSource,
    gripper_limits_from_config,
)
from so101_collector.devices import GamepadButtonSource, SpaceMouseSource
from so101_collector.exceptions import ConfigError, SO101Error
from so101_collector.simulation import ScriptedTeleop, SimulatedBus, SimulatedCamera, StaticTeleop

logger = logging.getLogger("so101.record")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="so101-record",
        description="Record synchronized SO-101 demonstration episodes in LeRobotDataset format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    configs = parser.add_argument_group("configuration")
    configs.add_argument("--config-dir", type=Path, default=None,
                         help="directory containing robot_config.yaml and camera_config.yaml")
    configs.add_argument("--robot-config", type=Path, default=None, help="path to robot_config.yaml")
    configs.add_argument("--camera-config", type=Path, default=None, help="path to camera_config.yaml")
    configs.add_argument("--output-dir", type=Path, default=Path("data"), help="dataset root directory")
    configs.add_argument("--repo-id", default=None,
                         help="dataset folder name (default: so101_floor_pick_<timestamp>)")
    configs.add_argument("--task", default=DEFAULT_TASK, help="task description stored per episode")

    rig = parser.add_argument_group("rig")
    rig.add_argument("--teleop", choices=("leader", "spacenav", "simulated", "static"),
                     default=None, help="teleoperation backend (default: leader, or simulated with --simulate)")
    rig.add_argument("--leader-config", type=Path, default=None,
                     help="separate YAML for the leader arm (otherwise robot config's leader_port is used)")
    rig.add_argument("--spacenouse-device", default=None,
                     help="persistent evdev path of the SpaceMouse (/dev/input/by-id/...)")
    rig.add_argument("--gamepad-device", default=None,
                     help="persistent evdev path for gamepad trigger commands")
    rig.add_argument("--simulate", action="store_true",
                     help="run with simulated hardware/cameras (no robot attached)")

    loop = parser.add_argument_group("loop")
    loop.add_argument("--fps", type=float, default=30.0, help="control loop frequency")
    loop.add_argument("--warmup", type=float, default=2.0, help="warmup seconds before buffering")
    loop.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    loop.add_argument("--max-episodes", type=int, default=None, help="stop after N completed episodes")
    loop.add_argument("--auto", type=float, default=None, metavar="SECONDS",
                      help="headless auto-pilot: save each episode after SECONDS of "
                           "recording (for CI/smoke runs; ignores the keyboard)")

    output = parser.add_argument_group("storage")
    output.add_argument("--video-backend", choices=("auto", "av", "opencv"), default="auto")
    output.add_argument("--table-backend", choices=("auto", "parquet", "npz"), default="auto")
    output.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _load_configs(args: argparse.Namespace) -> tuple[RobotConfig, VisionConfig]:
    config_dir = args.config_dir or default_config_dir()
    robot_path = args.robot_config or config_dir / "robot_config.yaml"
    camera_path = args.camera_config or config_dir / "camera_config.yaml"
    robot = RobotConfig.from_yaml(robot_path)
    vision = VisionConfig.from_yaml(camera_path)
    logger.info("loaded configs: %s, %s", robot_path, camera_path)
    return robot, vision


def _build_hardware(robot: RobotConfig, *, simulate: bool) -> SO101HardwareInterface:
    """Follower interface (leader is built inside _build_teleop)."""
    if simulate:
        return SO101HardwareInterface(robot, bus=SimulatedBus(robot))
    return SO101HardwareInterface(robot)


def _build_teleop(
    args: argparse.Namespace, robot: RobotConfig, follower: SO101HardwareInterface
) -> TeleopSource:
    mode = args.teleop or ("simulated" if args.simulate else "leader")
    logger.info("teleop backend: %s", mode)

    if mode == "static":
        return StaticTeleop(robot)

    if mode == "simulated":
        return ScriptedTeleop(robot)

    if mode == "leader":
        if args.leader_config:
            leader_robot = RobotConfig.from_yaml(args.leader_config)
        elif robot.leader_port:
            leader_robot = replace(robot, port=robot.leader_port)
        else:
            raise ConfigError(
                "leader teleop requested but no leader configured: set 'leader_port' in "
                "robot_config.yaml or pass --leader-config"
            )
        if args.simulate:
            leader_bus = SimulatedBus(leader_robot)
            leader_interface = SO101HardwareInterface(leader_robot, bus=leader_bus)
        else:
            leader_interface = SO101HardwareInterface(leader_robot)
        return LeaderArmTeleop(leader_interface, leader_robot.leader_offset_deg)

    if mode == "spacenav":
        if not args.spacenouse_device:
            raise ConfigError(
                "--teleop spacenav requires --spacenouse-device "
                "(a persistent /dev/input/by-id path)"
            )
        device = SpaceMouseSource(args.spacenouse_device)
        kinematics = SO101Kinematics(robot.kinematics)
        return CartesianTeleop(
            kinematics,
            device,
            robot.home_pose,
            gripper_limits_from_config(robot),
        )

    raise ConfigError(f"unknown teleop mode {mode!r}")  # pragma: no cover - argparse guards


def _build_commands(args: argparse.Namespace) -> CommandReader:
    if args.auto is not None:
        if args.auto <= 0:
            raise ConfigError(f"--auto must be > 0, got {args.auto}")
        logger.info("auto-pilot: finishing each episode after %.1f s of recording", args.auto)
        # START is pressed first, then warmup elapses, then recording runs;
        # shift the timer so --auto counts pure recording time.
        return TimedCommandReader(args.auto + args.warmup)
    readers: list[CommandReader] = [KeyboardCommandReader()]
    if args.gamepad_device:
        readers.append(GamepadCommandReader(GamepadButtonSource(args.gamepad_device)))
    if len(readers) == 1:
        reader = readers[0]
        if not reader.enabled:  # type: ignore[attr-defined]
            logger.warning("stdin is not a TTY and no gamepad configured - running headless "
                           "(episodes limited by --duration/--max-episodes)")
            return NullCommandReader()
        return reader
    return CompositeCommandReader(*readers)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        robot, vision = _load_configs(args)
        follower = _build_hardware(robot, simulate=args.simulate)

        camera_hub = CameraHub(
            vision,
            source_factory=(lambda cfg: SimulatedCamera(cfg)) if args.simulate else V4L2Camera,
        )
        teleop = _build_teleop(args, robot, follower)
        commands = _build_commands(args)

        repo_id = args.repo_id or f"so101_floor_pick_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = DatasetWriter(
            args.output_dir,
            repo_id,
            fps=args.fps,
            image_shape=vision.by_name["base"].frame_shape,
            video_preference=args.video_backend,
            table_preference=args.table_backend,
        )
        logger.info("dataset directory: %s", writer.directory)

        runner = DataCollectionRunner(
            hardware=follower,
            camera_hub=camera_hub,
            teleop=teleop,
            writer=writer,
            commands=commands,
            fps=args.fps,
            warmup_s=args.warmup,
            task=args.task,
        )
        stats = runner.run(max_episodes=args.max_episodes, max_seconds=args.duration)
    except SO101Error as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    print(stats.summary())
    return 130 if stats.interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
