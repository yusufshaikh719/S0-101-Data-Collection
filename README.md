# SO-101 data collection

Records synchronized joint states, actions and dual-camera video from an
SO-101 6-DOF arm performing floor pick-and-place, written as a Hugging Face
`LeRobotDataset` (v3) for imitation learning.

## Install

```bash
pip install -e .                 # Python 3.12+ and LeRobot dataset dependencies
pip install -e ".[dev]"          # + pytest, ruff
pip install -e ".[visualize]"    # + rerun-sdk
```

## Usage

```bash
# Discover devices, write both configs, then pre-flight the rig
so101-setup
so101-setup --check          # repeat checks any time, no prompts

# Check the rig against the timing requirements
so101-check-cameras --snapshot-dir ./camera_snapshots

# Calibrate joint offsets with torque off, then save them
so101-calibrate --capture min home max --write

# Capture the tucked-in pose that ends an episode
so101-calibrate --capture-home --write

# Record with a 1:1 leader arm and a live Rerun viewer
so101-record --teleop leader --repo-id floor_pick_session_01 --rerun

# Run without a robot attached
so101-record --simulate --duration 10
```

`so101-check-cameras` measures both cameras at 30 FPS ±1, frame sync drift
below 15 ms and serial round trips below 8 ms, and writes PNG snapshots for
field-of-view inspection. `--skip-serial` runs the vision checks only.

Rerun can also stream to an external viewer or a file:

```bash
so101-record --teleop leader --rerun-connect 127.0.0.1:9876
so101-record --teleop leader --rerun-save session_01.rrd
```

### Recording protocol

| Key / button   | Action                                          |
|----------------|-------------------------------------------------|
| automatic      | Starts on departure from home; splits on return |
| Enter / green  | Split the current episode                       |
| Esc / red      | Discard the current episode                     |
| R              | Drive the arm to its home pose                  |
| Q / Ctrl-C     | Save the partial episode and quit               |

An episode starts when any measured joint leaves the tucked pose by more than
`--home-exit-tolerance` (8° by default), so time spent parked at home is not
recorded. It ends when all six joints, gripper included, are within
`--home-tolerance` (5°) of `home_deg`; the sample that reaches home is the last
frame of the episode and the following departure opens the next one.

Calibrate `offset_deg` first (`so101-calibrate --capture min`), then run
`so101-calibrate --capture-home --write` to sample the tucked pose three times
and write the mean into `home_deg`. The shipped values are placeholders; a pose
outside the configured joint limits is rejected. `so101-setup --check` prints
the per-joint distance from home so you can confirm detection fires.

For floor pickup, split a demonstration into pre-grasp alignment, descent and
grasp with a short contact pause, then lift and hold. Vary object position
across the 30×30 cm grid and include recovery demonstrations.

## Configuration

Both YAML files in `configs/` are validated strictly on load:

- Device paths must be persistent: `/dev/.../by-id/...` symlinks on Linux,
  `/dev/cu.*` adapter names on macOS, or an OpenCV camera index. Dynamic paths
  such as `/dev/video0` are rejected unless `allow_dynamic_path` or
  `allow_dynamic_port` is set.
- `robot_config.yaml` holds the bus settings, joint limits, home pose, leader
  port and IK link lengths.
- `camera_config.yaml` holds both camera paths, resolution, frame rate and the
  frame-sync tolerance.

## Dataset

```
data/<repo_id>/
├── meta/            # info, statistics, tasks, episode boundaries
├── data/chunk-*/    # LeRobot v3 Parquet shards
└── videos/observation.images.{base,wrist}/chunk-*/  # MP4 shards
```

Episodes are written on a worker thread and committed with `LeRobotDataset`.
If encoding cannot keep up, recording stops with an error rather than dropping
frames. Timestamps use the nominal frame rate, so check the loop overrun and
FPS logs before training. Returning home does not mark an episode successful.

## Tests

```bash
python -m pytest
ruff check .
```

The suite needs no hardware. It covers packet framing and retries, joint
clamping, FK/IK round trips, loop-timer drift, camera sync warnings, config
validation, home-pose splitting and an end-to-end session on the simulated rig.

## macOS

The recording pipeline runs on macOS. Only SpaceMouse and gamepad input are
Linux-only, so use the leader arm for teleoperation there.

- Serial adapters appear as `/dev/cu.usbmodem*` or `/dev/cu.usbserial-*`.
  Those names embed the adapter's USB serial number and are accepted like a
  Linux by-id link. `so101-setup` lists them through pyserial when the Linux
  by-id directories are absent.
- Cameras are OpenCV indices (`"0"`, `"1"`). `so101-setup` probes which
  indices return a frame and writes the value quoted so YAML keeps it a
  string. Grant the terminal camera access under System Settings → Privacy &
  Security → Camera, or capture opens but never returns a frame.
- MJPG, manual exposure and buffer-size controls are V4L2-specific; the
  AVFoundation driver rejects them with a warning and falls back to the
  requested resolution. Measure with `so101-check-cameras --duration 30`
  before recording. If two cameras cannot both hold 30 FPS, lower `fps` in
  `camera_config.yaml` and pass the same value to `so101-record --fps`, so
  timestamps stay consistent with the delivered frames.

## Platform

The Linux rig adds V4L2 device paths and evdev input; recording, leader
teleoperation and the tests also run on macOS and Windows.
