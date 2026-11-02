"""Joint offset and range calibration.

Feetech servos report raw encoder ticks (0..4095 for one revolution); the
pipeline converts them to degrees using a per-joint ``offset_deg``.

    so101-calibrate --show                          # current ticks and degrees
    so101-calibrate --capture min home max          # guided capture
    so101-calibrate --capture min --write            # capture and save
    so101-calibrate --set shoulder_lift=-3.5        # direct value

Torque is disabled during capture so the joints move freely, then re-enabled
so the arm holds its position.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from so101_collector.config import JOINT_NAMES, RobotConfig, default_config_dir
from so101_collector.core.hardware_interface import DEG_PER_TICK, SO101HardwareInterface
from so101_collector.exceptions import SO101Error

logger = logging.getLogger("so101.calibrate")

REFERENCE_POINTS = ("min", "home", "max")


def compute_offset(target_deg: float, raw_ticks: int) -> float:
    """Offset that maps ``raw_ticks`` onto ``target_deg`` (deg = ticks*s + offset)."""
    return target_deg - raw_ticks * DEG_PER_TICK


def parse_set_points(pairs: list[str]) -> dict[str, float]:
    """Parse ``--set joint=offset`` arguments."""
    values: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects JOINT=DEG, got {pair!r}")
        name, _, raw = pair.partition("=")
        name = name.strip()
        if name not in JOINT_NAMES:
            raise ValueError(f"unknown joint {name!r}; valid: {', '.join(JOINT_NAMES)}")
        values[name] = float(raw)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="so101-calibrate",
        description="Calibrate SO-101 joint offsets and verify joint ranges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--joint", action="append", default=None, metavar="NAME",
                        help="restrict to a joint (repeatable; default: all six)")
    parser.add_argument("--show", action="store_true",
                        help="print raw ticks and derived degrees, then exit")
    parser.add_argument("--capture", nargs="+", choices=REFERENCE_POINTS, default=None,
                        metavar="{min,home,max}",
                        help="interactively capture reference positions")
    parser.add_argument("--set", action="append", default=None, metavar="JOINT=DEG",
                        help="set an offset directly (repeatable)")
    parser.add_argument("--write", action="store_true",
                        help="persist updated offsets to the YAML (default: dry run)")
    parser.add_argument("--no-torque", action="store_true",
                        help="skip re-enabling torque after capture")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _selected_joints(args: argparse.Namespace) -> list[int]:
    if not args.joint:
        return list(range(len(JOINT_NAMES)))
    indices = []
    for name in args.joint:
        if name not in JOINT_NAMES:
            raise ValueError(f"unknown joint {name!r}; valid: {', '.join(JOINT_NAMES)}")
        indices.append(JOINT_NAMES.index(name))
    return indices


def _show_state(interface: SO101HardwareInterface, config: RobotConfig, indices: list[int]) -> None:
    ticks = interface.read_raw_ticks()
    state = interface.read_state()
    print(f"{'joint':<14} {'servo':>5} {'ticks':>6} {'deg':>8} {'limits':>16} {'offset':>8}")
    for index in indices:
        spec = config.joints[index]
        print(
            f"{spec.name:<14} {spec.servo_id:>5} {ticks[index]:>6} "
            f"{state.positions[index]:>8.2f} "
            f"[{spec.min_deg:>6.1f},{spec.max_deg:>6.1f}] {spec.offset_deg:>8.2f}"
        )


def _capture(
    interface: SO101HardwareInterface,
    config: RobotConfig,
    indices: list[int],
    references: list[str],
) -> dict[str, float]:
    if not sys.stdin.isatty():
        raise SO101Error("--capture is interactive and requires a TTY")
    new_offsets: dict[str, float] = {}
    print("Torque DISABLED - move each joint by hand when prompted.")
    interface.set_torque(False)
    try:
        for index in indices:
            spec = config.joints[index]
            print(f"\n--- {spec.name} (servo {spec.servo_id}) ---")
            captured_offset: float | None = None
            for reference in references:
                target = {"min": spec.min_deg, "home": spec.home_deg, "max": spec.max_deg}[
                    reference
                ]
                input(
                    f"  move {spec.name} to its {reference} stop ({target:.1f} deg) "
                    "and press Enter..."
                )
                ticks = int(interface.read_raw_ticks()[index])
                if captured_offset is None:
                    # First captured reference defines the offset.
                    captured_offset = compute_offset(target, ticks)
                    print(f"    raw {ticks} ticks -> offset {captured_offset:+.2f} deg")
                else:
                    # Remaining references only verify the configured range.
                    predicted = ticks * DEG_PER_TICK + captured_offset
                    delta = predicted - target
                    status = "ok" if abs(delta) <= 2.0 else "MISMATCH"
                    print(
                        f"    raw {ticks} ticks -> {predicted:.1f} deg "
                        f"(expected {target:.1f}, delta {delta:+.1f}) [{status}]"
                    )
            new_offsets[spec.name] = float(captured_offset)
    finally:
        try:
            interface.set_torque(True)
            print("Torque re-enabled (arm holds its position).")
        except SO101Error as exc:
            logger.warning("could not re-enable torque: %s", exc)
    return new_offsets


def _apply_offsets(config: RobotConfig, offsets: dict[str, float]) -> RobotConfig:
    """Return a copy of ``config`` with the given per-joint offsets applied."""
    payload = config.to_dict()
    for joint in payload["joints"]:
        if joint["name"] in offsets:
            joint["offset_deg"] = offsets[joint["name"]]
    return RobotConfig.from_dict(payload)


def _write_config(path: Path, config: RobotConfig) -> None:
    header = (
        "# SO-101 follower arm: Feetech STS3215 serial bus + joint calibration\n"
        "# Written by so101-calibrate (hand-maintained comments are not preserved).\n"
    )
    payload = yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=None)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(header + payload, encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if not (args.show or args.capture or args.set):
        args.show = True  # default action

    config_dir = args.config_dir or default_config_dir()
    config_path = args.robot_config or config_dir / "robot_config.yaml"
    config = RobotConfig.from_yaml(config_path)
    indices = _selected_joints(args)

    try:
        interface = SO101HardwareInterface(config)
        interface.open()
    except SO101Error as exc:
        print(f"[FAIL] cannot open robot bus: {exc}")
        return 2

    exit_code = 0
    try:
        if args.show:
            _show_state(interface, config, indices)

        offsets: dict[str, float] = {}
        if args.set:
            offsets.update(parse_set_points(args.set))
        if args.capture:
            offsets.update(_capture(interface, config, indices, list(args.capture)))

        if offsets:
            updated = _apply_offsets(config, offsets)
            print("\nProposed offsets:")
            for name, value in sorted(offsets.items()):
                print(f"  {name}: {value:+.2f} deg")
            if args.write:
                _write_config(config_path, updated)
                print(f"Written to {config_path}")
            else:
                print("(dry run - pass --write to persist)")
    except (SO101Error, ValueError) as exc:
        print(f"[FAIL] {exc}")
        exit_code = 1
    finally:
        interface.close()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
