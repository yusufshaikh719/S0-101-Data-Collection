# SO-101 Multimodal Data Collection Pipeline

Production-grade data collection for an **SO-101 6-DOF robotic arm** performing
ground/floor pick-and-place tasks. Records synchronized, time-aligned
state-action pairs and dual-camera video for offline RL and imitation learning
(ACT / Diffusion Policy), written in Hugging Face **`LeRobotDataset`** format.

Implements the full specification in
[so_101_data_collection_pipeline_specification.md](so_101_data_collection_pipeline_specification.md).

```
┌────────────────────────────────────────────────┐
│              DataCollectionRunner              │
│  episode lifecycle (warmup → record → save)    │
│  precision 30 Hz drift-compensating loop       │
└──────────┬─────────────────────────┬───────────┘
           ▼                         ▼
┌─────────────────────┐   ┌──────────────────────┐
│      CameraHub      │   │  SO101HardwareInterface│
│ BaseCameraThread    │   │  Feetech STS bus       │
│ WristCameraThread   │   │  retries + clamping    │
│ double-buffered     │   │  6-joint sync write    │
└──────────┬──────────┘   └──────────┬────────────┘
           └───────────┬─────────────┘
                       ▼
            ┌─────────────────────┐
            │    DatasetWriter    │
            │ MP4 + Parquet +     │
            │ episodes.jsonl      │
            │ (atomic commits)    │
            └─────────────────────┘
```

## Layout

```
configs/                      # hardware + vision YAML (persistent device IDs)
so101_collector/
├── config.py                 # strict dataclass/YAML config loaders
├── timing.py                 # drift-compensating 30 Hz LoopTimer
├── console.py                # keyboard/gamepad command readers
├── devices.py                # evdev SpaceMouse/gamepad (stdlib only)
├── simulation.py             # simulated bus/cameras/teleop (--simulate, tests)
├── core/
│   ├── hardware_interface.py # Feetech STS transport + joint API (retries, clamps)
│   ├── camera_hub.py         # threaded acquisition + software frame sync
│   ├── ik.py                 # closed-form FK/IK + floor/workspace clamp
│   ├── teleop_controller.py  # leader arm / Cartesian SpaceMouse teleop
│   ├── dataset_writer.py     # LeRobot-style atomic episode writer
│   └── runner.py             # DataCollectionRunner (episode state machine)
├── scripts/
│   ├── record_episodes.py    # so101-record      (main CLI)
│   ├── check_cameras.py      # so101-check-cameras (acceptance checks)
│   └── calibrate_joints.py   # so101-calibrate    (offset capture)
tests/                        # 78 unit/integration tests, no hardware needed
```

## Install

```bash
pip install -e .            # runtime deps (numpy, pyserial, opencv, pandas, yaml)
pip install -e ".[dev]"     # + pytest, ruff, av (H.264), pyarrow (Parquet)
```

Without the extras the pipeline still runs: video falls back to OpenCV
`mp4v` and tables fall back to `.npz` (with warnings). Install `av` and
`pyarrow` before recording real training datasets.

## Quick start

```bash
# 1. Verify the rig against the acceptance criteria (5-minute run)
so101-check-cameras --snapshot-dir ./camera_snapshots

# 2. Calibrate joint offsets (torque off, move joints by hand)
so101-calibrate --capture min home max --write

# 3. Record demonstrations with a 1:1 leader arm
so101-record --teleop leader --repo-id floor_pick_session_01

# Everything without a robot attached (CI / first-run smoke test)
so101-record --simulate --auto 3 --max-episodes 2 --duration 60
```

### Recording protocol

| Key / button        | Action                                             |
|---------------------|----------------------------------------------------|
| `Space` / trigger   | Start episode — 2 s warmup, then buffer at 30 Hz   |
| `Enter` / green     | Finish — atomically save MP4 + table + manifest    |
| `Esc` / red         | Abort — clear RAM buffers, nothing written         |
| `R`                 | Reset arm to home pose                             |
| `Q`                 | Quit session                                       |

Three-stage demonstration structure for floor pickup (see spec §5): pre-grasp
alignment (0–40 %), descent + grasp with 200 ms contact settle (40–70 %),
lift and hold 15 cm (70–100 %). Vary object position across the 30×30 cm
grid, lighting, and record recovery demonstrations for diversity.

## Key design decisions

- **Zero index-drift** — devices are addressed exclusively through
  `/dev/serial/by-id/*`, `/dev/v4l/by-id/*`, `/dev/input/by-id/*`. Configs
  containing `/dev/video0`-style paths fail validation
  (`allow_dynamic_*: true` overrides for bench testing).
- **Drift-compensating loop** — absolute monotonic deadlines
  (`next_tick += target_dt`), never `sleep(0.033)`. Overruns are counted,
  warned above 5 ms, and the schedule resynchronizes after stalls > 3 periods.
  Injected clock/sleep make timing fully unit-testable.
- **Decoupled video acquisition** — each camera runs in its own daemon thread
  with a lock-guarded latest-frame swap, so a stalled control loop never drops
  frames. `get_synced_frames()` returns `(base, wrist, dt_sync)` and warns
  when `dt_sync > 20 ms`.
- **Data safety** — episodes stage into `.staging/` and commit via
  `os.replace`; the `episodes.jsonl` append is the commit point. Discard
  clears RAM only. A failed save keeps the buffer so FINISH can retry without
  loss. A camera failure mid-episode discards the episode instead of writing
  state/video that no longer line up.
- **Safety clamping** — every write clamps to per-joint limits and the
  workspace clamps the gripper tip to `floor_clearance_mm` (2 mm) above the
  floor, so the arm never relies on motor stall. A calibration guard warns
  when a joint maps outside the 0–4095 encoder window.
- **Analytical IK** — closed-form solver (constant time, FK/IK round trips
  verified to 1e-6 mm in tests) instead of iterative numerical IK inside the
  control loop.

## Dataset layout

```
data/<repo_id>/
├── meta/
│   ├── info.json          # fps, features schema, totals (LeRobot v2.1-style)
│   ├── tasks.jsonl        # task string → task_index
│   └── episodes.jsonl     # per-episode manifest (length, success, files)
├── data/chunk-000/episode_000000.parquet    # timestamp, state, action, indices
└── videos/chunk-000/episode_000000/
    ├── observation.images.base.mp4          # H.264 (PyAV) or mp4v fallback
    └── observation.images.wrist.mp4
```

Parquet columns: `timestamp` (uniform 1/fps grid), `frame_index`,
`episode_index`, `index`, `task_index`, `observation.state (N,6) float32`,
`action (N,6) float32`.

## Acceptance checklist (spec §6)

`so101-check-cameras` verifies automatically:

- [x] Frame rate stability: both cameras `30.0 ± 1.0 FPS` in every window
- [x] Camera sync drift: strictly `< 15 ms`
- [x] Serial latency: read/write roundtrip `< 8 ms` (`--skip-serial` to opt out)
- [ ] Visual field coverage — inspect `--snapshot-dir` PNGs manually
- [ ] Storage integrity — load an episode with `LeRobotDataset` manually

Exit code `0` only when every automated check passes.

## Testing

```bash
python -m pytest            # 78 tests, no hardware required
ruff check .                # lint
```

Tests cover packet framing/checksums, retry-exhaustion behaviour, safety
clamping, FK/IK round trips, loop-timer determinism (fake clock), camera
synchronization warnings, atomic save/discard semantics, config validation
policy, and a full end-to-end recording session on the simulated rig.

## Platform notes

- The real rig targets **Linux** (V4L2 + evdev + `/dev/*/by-id` paths).
- `--simulate`, leader-arm teleop, recording, and all tests also run on
  Windows/macOS — SpaceMouse/gamepad evdev input is Linux-only.
