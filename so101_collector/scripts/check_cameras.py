"""Verify camera frame rate, frame sync and serial latency.

Runs both cameras for ``--duration`` seconds (5 minutes by default) and checks
30 FPS +/- 1, sync drift under 15 ms and serial round trips under 8 ms. Exits
non-zero if an automated check fails.

Examples::

    so101-check-cameras                       # full run
    so101-check-cameras --duration 30         # quick check
    so101-check-cameras --simulate            # no hardware
    so101-check-cameras --skip-serial \\
        --snapshot-dir ./camera_snapshots     # vision-only + FOV evidence
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from so101_collector.config import VisionConfig, default_config_dir
from so101_collector.core.camera_hub import CameraHub, V4L2Camera
from so101_collector.core.hardware_interface import SO101HardwareInterface
from so101_collector.core.visualizer import NullVisualizer, RerunVisualizer, Visualizer
from so101_collector.exceptions import CameraError, SO101Error
from so101_collector.simulation import SimulatedCamera
from so101_collector.timing import LoopTimer

logger = logging.getLogger("so101.check")

MANUAL_CHECKS = (
    "Visual field coverage: the base camera sees the pick area and arm base, and the "
    "wrist camera keeps the object in frame during descent (inspect the snapshots).",
    "Storage integrity: recorded episodes load with "
    "lerobot.datasets.lerobot_dataset.LeRobotDataset.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="so101-check-cameras",
        description="Verify camera FPS, frame synchronization and serial latency "
                    "against the acceptance criteria.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--robot-config", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=300.0,
                        help="measurement window in seconds (spec: 5 minutes)")
    parser.add_argument("--window", type=float, default=1.0,
                        help="FPS sampling window in seconds")
    parser.add_argument("--fps-tolerance", type=float, default=1.0,
                        help="allowed deviation from the nominal FPS")
    parser.add_argument("--sync-threshold-ms", type=float, default=15.0,
                        help="max allowed sync drift (spec: strictly below 15 ms)")
    parser.add_argument("--serial-threshold-ms", type=float, default=8.0,
                        help="max allowed serial roundtrip (spec: < 8 ms)")
    parser.add_argument("--serial-samples", type=int, default=100)
    parser.add_argument("--skip-serial", action="store_true",
                        help="skip the Feetech bus latency check")
    parser.add_argument("--snapshot-dir", type=Path, default=None,
                        help="save one PNG per camera for manual FOV inspection")
    parser.add_argument("--rerun", action="store_true",
                        help="stream live camera feeds and sync metrics to Rerun viewer")
    parser.add_argument("--rerun-connect", default=None, metavar="ADDR",
                        help="connect to an external running Rerun viewer (e.g. 127.0.0.1:9876)")
    parser.add_argument("--simulate", action="store_true",
                        help="use simulated cameras (no hardware)")
    parser.add_argument("--log-level", default="WARNING",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _save_snapshot(name: str, frame, directory: Path, counter: int) -> None:
    try:
        import cv2
    except ImportError:  # pragma: no cover - opencv is a hard dependency
        logger.warning("opencv missing; cannot save snapshots")
        return
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{counter:03d}.png"
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _measure(
    hub: CameraHub,
    *,
    duration_s: float,
    window_s: float,
    fps: float,
    snapshot_dir: Path | None,
    visualizer: Visualizer | None = None,
) -> tuple[dict[str, list[float]], list[float]]:
    """Return per-camera windowed FPS series and sync-dt samples (seconds)."""
    windows: dict[str, list[float]] = {name: [] for name in hub.names}
    sync_samples: list[float] = []
    last_counts: dict[str, int] = {name: 0 for name in hub.names}
    last_window_at = time.monotonic()
    snapshots = 0
    step_idx = 0
    started = time.monotonic()

    timer = LoopTimer(fps)
    timer.start()
    try:
        while time.monotonic() - started < duration_s:
            step_idx += 1
            now = time.monotonic()
            base_frame = None
            wrist_frame = None
            dt_sync_ms = None
            try:
                base, wrist, dt_sync = hub.get_synced_frames()
                sync_samples.append(dt_sync)
                base_frame = base
                wrist_frame = wrist
                dt_sync_ms = dt_sync * 1000.0
            except CameraError as exc:
                logger.warning("frame unavailable during check: %s", exc)

            if visualizer is not None:
                visualizer.log_step(
                    session_time_s=now - started,
                    step_index=step_idx,
                    state="check",
                    base_rgb=base_frame,
                    wrist_rgb=wrist_frame,
                    dt_sync_ms=dt_sync_ms,
                )

            if snapshot_dir is not None and snapshots < 3:
                try:
                    base, wrist, _ = hub.get_synced_frames()
                except CameraError:
                    pass
                else:
                    _save_snapshot("base", base, snapshot_dir, snapshots)
                    _save_snapshot("wrist", wrist, snapshot_dir, snapshots)
                    snapshots += 1

            if now - last_window_at >= window_s:
                stats = hub.stats()
                elapsed = now - last_window_at
                for name in hub.names:
                    count = int(stats.cameras[name]["frames"])
                    delta = count - last_counts[name]
                    windows[name].append(delta / elapsed)
                    last_counts[name] = count
                last_window_at = now
            timer.wait()
    finally:
        timer.stop()
    return windows, sync_samples


def _check_fps(windows: dict[str, list[float]], nominal: float, tolerance: float) -> bool:
    ok = True
    for name, series in windows.items():
        if not series:
            print(f"  [FAIL] {name} camera: no FPS samples collected")
            ok = False
            continue
        worst_low, worst_high = min(series), max(series)
        passed = worst_low >= nominal - tolerance and worst_high <= nominal + tolerance
        ok &= passed
        verdict = "PASS" if passed else "FAIL"
        print(
            f"  [{verdict}] {name} camera: {len(series)} windows, "
            f"min {worst_low:.2f} / max {worst_high:.2f} FPS "
            f"(required {nominal:.1f} +/- {tolerance:.1f})"
        )
    return ok


def _check_sync(sync_samples: list[float], threshold_ms: float) -> bool:
    if not sync_samples:
        print("  [FAIL] frame sync: no samples collected")
        return False
    max_ms = max(sync_samples) * 1000.0
    mean_ms = sum(sync_samples) / len(sync_samples) * 1000.0
    passed = max_ms < threshold_ms
    verdict = "PASS" if passed else "FAIL"
    print(
        f"  [{verdict}] frame sync: mean {mean_ms:.2f} ms / max {max_ms:.2f} ms "
        f"(required strictly < {threshold_ms:.0f} ms, n={len(sync_samples)})"
    )
    return passed


def _check_serial(config_path: Path, threshold_ms: float, samples: int,
                  simulate: bool) -> bool:
    if simulate:
        print("  [SKIP] serial latency: --simulate mode")
        return True
    from so101_collector.config import RobotConfig

    robot = RobotConfig.from_yaml(config_path)
    interface = SO101HardwareInterface(robot)
    try:
        interface.open()
        stats = interface.measure_latency(samples)
    except SO101Error as exc:
        print(f"  [FAIL] serial latency: {exc}")
        return False
    finally:
        interface.close()
    passed = stats.max_ms < threshold_ms
    verdict = "PASS" if passed else "FAIL"
    print(
        f"  [{verdict}] serial latency: read mean {stats.read_mean_ms:.2f} / "
        f"max {stats.read_max_ms:.2f} ms, write mean {stats.write_mean_ms:.2f} / "
        f"max {stats.write_max_ms:.2f} ms (required < {threshold_ms:.0f} ms)"
    )
    return passed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        config_dir = args.config_dir or default_config_dir()
        camera_path = args.camera_config or config_dir / "camera_config.yaml"
        robot_path = args.robot_config or config_dir / "robot_config.yaml"
        vision = VisionConfig.from_yaml(camera_path)
        nominal_fps = min(camera.fps for camera in vision.cameras)

        visualizer: Visualizer = NullVisualizer()
        if args.rerun or args.rerun_connect:
            visualizer = RerunVisualizer(
                application_id="so101_check_cameras",
                spawn=bool(args.rerun and not args.rerun_connect),
                connect_addr=args.rerun_connect,
            )
            visualizer.start()

        hub = CameraHub(
            vision,
            sync_tolerance_ms=args.sync_threshold_ms,
            source_factory=(lambda cfg: SimulatedCamera(cfg)) if args.simulate else V4L2Camera,
        )
        print(f"Measuring for {args.duration:.0f}s "
              f"({', '.join(c.name for c in vision.cameras)})...")
        with hub:
            try:
                windows, sync_samples = _measure(
                    hub,
                    duration_s=args.duration,
                    window_s=args.window,
                    fps=nominal_fps,
                    snapshot_dir=args.snapshot_dir,
                    visualizer=visualizer,
                )
                hub_final = hub.stats()
            finally:
                visualizer.close()
    except SO101Error as exc:
        print(f"[FAIL] setup error: {exc}")
        return 1

    print("\nAcceptance checklist:")
    fps_ok = _check_fps(windows, nominal_fps, args.fps_tolerance)
    sync_ok = _check_sync(sync_samples, args.sync_threshold_ms)
    if args.skip_serial:
        print("  [SKIP] serial latency: disabled with --skip-serial")
        serial_ok = True
    else:
        serial_ok = _check_serial(robot_path, args.serial_threshold_ms,
                                  args.serial_samples, args.simulate)
    if args.snapshot_dir:
        print(f"  [----] FOV snapshots written to {args.snapshot_dir} (inspect manually)")
    for index, check in enumerate(MANUAL_CHECKS, start=1):
        print(f"  [  ? ] manual check {index}: {check}")

    all_ok = fps_ok and sync_ok and serial_ok
    if hub_final.sync_warnings:
        print(f"  [note] {hub_final.sync_warnings} sync warnings logged during the run")
    print(f"\n{'ALL AUTOMATED CHECKS PASSED' if all_ok else 'CHECKS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
