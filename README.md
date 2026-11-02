# SO-101 Multimodal Data Collection Pipeline

Data collection for an **SO-101 6-DOF robotic arm** performing
ground/floor pick-and-place tasks. Records synchronized, time-aligned
state-action pairs and dual-camera video for offline RL and imitation learning
(ACT / Diffusion Policy), written in Hugging Face **`LeRobotDataset`** format.

The recording command uses LeRobot's official dataset writer. The older
[pipeline specification](so_101_data_collection_pipeline_specification.md) describes the
previous manual recording protocol and custom storage layout.

```
┌────────────────────────────────────────────────┐
│              DataCollectionRunner              │
│  continuous capture → home-pose split          │
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
            │   LeRobotDataset    │
            │ MP4 + Parquet +     │
            │ dataset metadata    │
            │ (v3.0)              │
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
│   ├── runner.py             # legacy manual runner
│   ├── lerobot_writer.py     # official LeRobot writer on background thread
│   └── continuous_runner.py  # continuous recording and home detection
├── scripts/
│   ├── record_episodes.py    # so101-record      (main CLI)
│   ├── check_cameras.py      # so101-check-cameras (acceptance checks)
│   └── calibrate_joints.py   # so101-calibrate    (offset capture)
tests/                        # unit/integration tests, no hardware needed
```

## Install

```bash
pip install -e .                 # Python 3.12+, includes LeRobot dataset dependencies
pip install -e ".[dev]"          # + pytest, ruff, rerun-sdk
pip install -e ".[visualize]"    # + rerun-sdk
```

The recording command requires LeRobot and does not fall back to a non-LeRobot
`.npz` dataset. Install `rerun-sdk` for live 3D visualization.

## Quick start

```bash
# 1. Verify the rig against the acceptance criteria (5-minute run)
so101-check-cameras --snapshot-dir ./camera_snapshots

# 1b. Verify cameras with live Rerun 2D/3D stream and sync telemetry
so101-check-cameras --rerun --duration 30

# 2. Calibrate joint offsets (torque off, move joints by hand)
so101-calibrate --capture min home max --write

# 3. Record demonstrations with a 1:1 leader arm + live Rerun viewer
so101-record --teleop leader --repo-id floor_pick_session_01 --rerun

# Stream telemetry to an external Rerun viewer or save directly to an .rrd file
so101-record --teleop leader --rerun-connect 127.0.0.1:9876
so101-record --teleop leader --rerun-save session_01.rrd

# Everything without a robot attached (CI / first-run smoke test)
so101-record --simulate --duration 10
```

### Recording protocol

| Key / button        | Action                                             |
|---------------------|----------------------------------------------------|
| Automatic           | Start when cameras are ready; split on return home |
| `Enter` / green     | Split the current episode manually                 |
| `Esc` / red         | Discard the current episode and continue           |
| `R`                 | Send a home command                                |
| `Q` / Ctrl-C        | Save the final partial episode and quit            |

Home means **all six measured joints**, including the gripper, are within
`--home-tolerance` (default 5°) of their configured `home_deg`. A new split is
armed only after at least one joint moves beyond `--home-exit-tolerance`
(default 8°), preventing repeated splits while stationary. The sample that
reaches home ends the previous episode; the next sample starts the next one.
Set each `home_deg` to the measured joint angle at the actual tucked-in pose
in `configs/robot_config.yaml`, calibrate offsets, and verify the values on the
real arm before collecting data: the shipped values and offsets are
placeholders. The calibration command treats configured angles as reference
values; it does not discover the tucked-in angles. `--warmup` optionally delays capture;
the default is zero. `--auto` adds timed manual splits for smoke tests.

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
  with a lock-guarded latest-frame swap. `get_synced_frames()` returns
  `(base, wrist, dt_sync)` and warns
  when `dt_sync > 20 ms`.
- **Writer backpressure** — a bounded background queue keeps episode encoding
  off the control loop. If encoding cannot keep up or a camera fails, the
  session stops with an error rather than silently dropping samples. On failure
  the partial episode is discarded where possible; normal shutdown saves it
  and finalizes the dataset. Abrupt process
  termination or an encoding failure can leave an incomplete dataset.
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
├── meta/                 # info, statistics, tasks and episode boundaries
├── data/chunk-*/         # LeRobot v3 Parquet shards
└── videos/observation.images.{base,wrist}/chunk-*/  # MP4 shards
```

Parquet columns: `timestamp` (uniform 1/fps grid), `frame_index`,
`episode_index`, `index`, `task_index`, `observation.state (N,6) float32`,
`action (N,6) float32`. Timestamps are generated at the configured nominal
fps and reset per episode, so monitor loop overrun/fps logs and validate the
rig's actual throughput before training. Episodes are not labeled successful
merely because the arm returned home. Existing legacy custom datasets are not
automatically migrated; record to a new repo ID instead.

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
python -m pytest            # no hardware required
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
