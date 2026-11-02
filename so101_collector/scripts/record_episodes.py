"""Record demonstration episodes.

    so101-record --teleop leader --repo-id floor_pick_session_01

    # SpaceMouse and gamepad commands
    so101-record --teleop spacenav \\
        --spacenav-device /dev/input/by-id/usb-3Dconnexion_SpaceMouse-event-if00 \\
        --gamepad-device /dev/input/by-id/usb-Logitech_F310-event-if00

    # No robot attached
    so101-record --simulate --max-episodes 2 --duration 60

Recording starts once the cameras are ready and an episode ends when the arm
returns to its home pose. Enter splits manually, Escape discards, Q saves the
partial episode and quits.
"""

from __future__ import annotations

import argparse
import logging
import math
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
from so101_collector.core.continuous_runner import DEFAULT_TASK, ContinuousDataCollectionRunner
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.ik import SO101Kinematics
from so101_collector.core.lerobot_writer import LeRobotWriter
from so101_collector.core.teleop_controller import (
    CartesianTeleop,
    LeaderArmTeleop,
    TeleopSource,
    gripper_limits_from_config,
)
from so101_collector.core.visualizer import NullVisualizer, RerunVisualizer
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
    configs.add_argument(
        "--robot-config", type=Path, default=None, help="path to robot_config.yaml"
    )
    configs.add_argument(
        "--camera-config", type=Path, default=None, help="path to camera_config.yaml"
    )
    configs.add_argument(
        "--output-dir", type=Path, default=Path("data"), help="dataset root directory"
    )
    configs.add_argument("--repo-id", default=None,
                         help="dataset folder name (default: so101_floor_pick_<timestamp>)")
    configs.add_argument("--task", default=DEFAULT_TASK, help="task description stored per episode")

    rig = parser.add_argument_group("rig")
    rig.add_argument(
        "--teleop",
        choices=("leader", "spacenav", "simulated", "static"),
        default=None,
        help="teleoperation backend (default: leader, or simulated with --simulate)",
    )
    rig.add_argument(
        "--leader-config",
        type=Path,
        default=None,
        help="separate YAML for the leader arm (otherwise robot config's leader_port is used)",
    )
    rig.add_argument("--spacenav-device", default=None,
                     help="persistent evdev path of the SpaceMouse (/dev/input/by-id/...)")
    rig.add_argument("--gamepad-device", default=None,
                     help="persistent evdev path for gamepad trigger commands")
    rig.add_argument("--simulate", action="store_true",
                     help="run with simulated hardware/cameras (no robot attached)")

    loop = parser.add_argument_group("loop")
    loop.add_argument("--fps", type=int, default=30, help="control loop frequency")
    loop.add_argument(
        "--warmup", type=float, default=0.0, help="optional camera warmup before recording"
    )
    loop.add_argument("--home-tolerance", type=float, default=5.0, metavar="DEGREES",
                      help="maximum per-joint error for detecting home")
    loop.add_argument("--home-exit-tolerance", type=float, default=8.0, metavar="DEGREES",
                      help="joint error needed to re-arm home detection")
    loop.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    loop.add_argument(
        "--max-episodes", type=int, default=None, help="stop after N completed episodes"
    )
    loop.add_argument("--auto", type=float, default=None, metavar="SECONDS",
                      help="headless auto-pilot: save each episode after SECONDS of "
                           "recording (for CI/smoke runs; ignores the keyboard)")

    vis = parser.add_argument_group("visualization")
    vis.add_argument("--rerun", action="store_true",
                     help="enable live Rerun visualization (spawns Rerun viewer window)")
    vis.add_argument("--rerun-connect", default=None, metavar="ADDR",
                     help="connect to an external running Rerun viewer (e.g. 127.0.0.1:9876)")
    vis.add_argument("--rerun-save", type=Path, default=None, metavar="PATH",
                     help="save the Rerun telemetry stream directly to an .rrd file")

    output = parser.add_argument_group("storage")
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
        if not args.spacenav_device:
            raise ConfigError(
                "--teleop spacenav requires --spacenav-device "
                "(a persistent /dev/input/by-id path)"
            )
        device = SpaceMouseSource(args.spacenav_device)
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
        # Warmup happens before the first frame, so add it to keep --auto
        # counting only recorded time.
        return TimedCommandReader(args.auto + args.warmup)
    readers: list[CommandReader] = [KeyboardCommandReader()]
    if args.gamepad_device:
        readers.append(GamepadCommandReader(GamepadButtonSource(args.gamepad_device)))
    if len(readers) == 1:
        reader = readers[0]
        if not reader.enabled:  # type: ignore[attr-defined]
            logger.warning("stdin is not a TTY and no gamepad configured - running headless "
                           "(stop with --duration/--max-episodes or Ctrl-C)")
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
        if (args.fps <= 0 or args.warmup < 0 or not math.isfinite(args.warmup)
                or not math.isfinite(args.home_tolerance) or args.home_tolerance <= 0
                or not math.isfinite(args.home_exit_tolerance)
                or args.home_exit_tolerance <= args.home_tolerance):
            raise ConfigError("fps must be positive, warmup nonnegative, and home exit tolerance "
                              "must exceed a positive home tolerance")
        for name in ("base", "wrist"):
            if vision.by_name[name].fps != args.fps:
                raise ConfigError(f"camera {name} fps must match the recording fps ({args.fps})")
        follower = _build_hardware(robot, simulate=args.simulate)

        camera_hub = CameraHub(
            vision,
            source_factory=(lambda cfg: SimulatedCamera(cfg)) if args.simulate else V4L2Camera,
        )
        teleop = _build_teleop(args, robot, follower)
        repo_id = args.repo_id or f"so101_floor_pick_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = LeRobotWriter(
            args.output_dir,
            repo_id,
            fps=args.fps,
            image_shapes={name: vision.by_name[name].frame_shape for name in ("base", "wrist")},
        )
        logger.info("dataset directory: %s", writer.directory)

        visualizer = NullVisualizer()
        if args.rerun or args.rerun_connect or args.rerun_save:
            visualizer = RerunVisualizer(
                application_id="so101_collector",
                recording_id=repo_id,
                spawn=bool(args.rerun and not args.rerun_connect and not args.rerun_save),
                connect_addr=args.rerun_connect,
                save_path=args.rerun_save,
            )

        commands = _build_commands(args)
        kinematics = SO101Kinematics(robot.kinematics)
        runner = ContinuousDataCollectionRunner(
            hardware=follower,
            camera_hub=camera_hub,
            teleop=teleop,
            writer=writer,
            commands=commands,
            fps=args.fps,
            warmup_s=args.warmup,
            task=args.task,
            home_tolerance_deg=args.home_tolerance,
            home_exit_tolerance_deg=args.home_exit_tolerance,
            visualizer=visualizer,
            kinematics=kinematics,
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
