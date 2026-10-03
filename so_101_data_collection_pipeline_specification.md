# SO-101 Multimodal Data Collection Pipeline Specification

## 1. System Overview & Objective
This specification defines the production-grade data collection pipeline for an **SO-101 6-DOF robotic arm** performing ground/floor pick-and-place tasks. The pipeline collects synchronized, time-aligned state-action pairs and dual-camera video feeds for Offline Reinforcement Learning (RL) and Imitation Learning (ACT / Diffusion Policy).

### Primary Hardware Architecture
* **Follower Robot:** SO-101 arm driven by Feetech STS3215 serial bus servos over USB-TTL (`/dev/serial/by-id/...` at 1,000,000 baud).
* **Teleoperation Interface:** 1:1 Leader Arm (passive follower twin) or 6-DOF Gamepad/SpaceMouse with real-time analytical Inverse Kinematics (IK).
* **Vision System:**
  * Base Camera (fixed third-person anchor observing workspace context and floor).
  * Wrist Camera (eye-in-hand viewing gripper alignment, pre-contact, and grasp engagement).
* **Output Format:** Hugging Face `LeRobotDataset` format (standardized zarr/parquet state data + synchronized H.264/MP4 camera streams).

---

## 2. Architecture & Design Principles

```
┌────────────────────────────────────────────────────────┐
│                   DataCollectionRunner                 │
│  - Episode lifecycle management (warmup, record, save) │
│  - Precision 30 Hz loop timer                          │
└───────────┬────────────────────────────────┬───────────┘
            │                                │
            ▼                                ▼
┌───────────────────────┐        ┌───────────────────────┐
│     VisionHub         │        │       RobotBus        │
│ - Base Camera Thread  │        │ - Feetech STS Bus     │
│ - Wrist Camera Thread │        │ - Joint State Polling │
│ - Software Frame-Sync │        │ - Teleop Action Ingest│
└───────────┬───────────┘        └───────────┬───────────┘
            │                                │
            └────────────────┬───────────────┘
                             ▼
                 ┌───────────────────────┐
                 │    DatasetWriter      │
                 │ - In-memory buffer    │
                 │ - Disk commit (MP4)   │
                 │ - Episode metadata    │
                 └───────────────────────┘
```

### Critical Non-Functional Requirements
1. **Zero Index-Drift:** Cameras and serial buses must be addressed strictly by stable persistent device identifiers (`/dev/v4l/by-id/*` and `/dev/serial/by-id/*`), never dynamic IDs (`/dev/video0`).
2. **Deterministic Sampling:** Fixed 30 Hz loop frequency ($T = 33.33\text{ ms}$). State readings and camera frames must maintain $<5\text{ ms}$ jitter.
3. **Decoupled Video Acquisition:** Camera acquisition must run in dedicated background threads with double-buffered frame swaps to prevent frame drops when the control loop stalls.
4. **Data Safety:** Atomic episode writes. Discard aborted/failed demonstrations without polluting disk datasets.

---

## 3. Directory Layout & Module Structure

```
so101_collector/
├── configs/
│   ├── robot_config.yaml       # Hardware serial ports, baud rate, joint limits
│   └── camera_config.yaml      # V4L2 persistent IDs, resolutions, FPS targets
├── core/
│   ├── __init__.py
│   ├── camera_hub.py           # Multi-threaded camera acquisition & synchronization
│   ├── hardware_interface.py   # Feetech STS bus wrapper (read/write joint states)
│   ├── teleop_controller.py    # Leader arm or SpaceMouse/Gamepad input mapper
│   └── dataset_writer.py       # LeRobot-compatible disk writer (HDF5/Parquet/MP4)
├── scripts/
│   ├── check_cameras.py        # Verification tool for camera latency and FOV
│   ├── calibrate_joints.py     # Joint offset and range calibration helper
│   └── record_episodes.py      # Main CLI entry point
├── tests/
│   ├── test_hardware.py
│   └── test_synchronization.py
├── pyproject.toml
└── README.md
```

---

## 4. Component Implementation Directives

### 4.1. Hardware Interface (`core/hardware_interface.py`)
* Implement a robust class `SO101HardwareInterface`:
  * Initialize serial connection via `pyserial` or `feetech-servo-sdk` with `baudrate=1000000` and `timeout=0.05`.
  * Define joint names: `['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']`.
  * **Read Function:** Returns current positions ($q \in \mathbb{R}^6$), velocities ($\dot{q} \in \mathbb{R}^6$), and applied currents/torques ($\tau \in \mathbb{R}^6$).
  * **Write Function:** Accepts target positions ($q_{target} \in \mathbb{R}^6$) with safety clamping against min/max joint angles to avoid floor collisions and motor burnout.
  * Implement error recovery: automatically retry on dropped serial packets up to 3 times before raising a critical `BusCommunicationError`.

### 4.2. Dual-Camera Hub (`core/camera_hub.py`)
* Class `CameraHub`:
  * Spawns two independent daemon threads: `BaseCameraThread` and `WristCameraThread`.
  * OpenCV backend initialized with `cv2.CAP_V4L2`.
  * Enforce hardware settings via V4L2 properties:
    * `cv2.CAP_PROP_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')` (vital for low-latency high-speed streaming).
    * `cv2.CAP_PROP_FRAME_WIDTH = 640`, `cv2.CAP_PROP_FRAME_HEIGHT = 480`.
    * `cv2.CAP_PROP_FPS = 30`.
    * Disable auto-exposure jitter: lock exposure if dynamic ambient lighting causes frame rate drop.
  * Method `get_synced_frames() -> Tuple[np.ndarray, np.ndarray, float]`:
    * Returns the latest available frame from both cameras along with the maximum timestamp delta $\Delta t_{sync} = |t_{base} - t_{wrist}|$.
    * If $\Delta t_{sync} > 20\text{ ms}$, log a synchronization warning.

### 4.3. Dataset Writer (`core/dataset_writer.py`)
* Class `DatasetWriter`:
  * Maintains an in-memory episode buffer for:
    * `observation.state`: Shape `(N, 6)` float32 (measured joint angles).
    * `action`: Shape `(N, 6)` float32 (commanded joint angles / leader arm positions).
    * `observation.images.base`: Shape `(N, 480, 640, 3)` uint8.
    * `observation.images.wrist`: Shape `(N, 480, 640, 3)` uint8.
    * `timestamp`: Shape `(N,)` float64.
  * On `save_episode(task_name: str, success: bool)`:
    * Encode video frames to H.264 MP4 using PyAV or OpenCV `avc1`/`mp4v`.
    * Persist tabular states into Apache Parquet or HDF5 compliant with the `LeRobotDataset` schema.
    * Write an episode manifest file (`episodes.jsonl`) recording total duration, frames, and success flag.
  * On `discard_episode()`: Clear RAM buffers immediately without touching disk.

### 4.4. Main Execution Engine (`scripts/record_episodes.py`)
* Use a keyboard or gamepad trigger loop:
  * Press **Space / Trigger**: Start episode (warmup phase of 2 seconds for visual stabilization).
  * Press **Enter / Green**: Successfully finish and flush episode to disk.
  * Press **Esc / Red**: Abort episode and clear memory buffers.
  * Press **R**: Reset robot back to home pose.
* Loop Timing Implementation:
  * Do **not** use naive `time.sleep(0.033)`.
  * Use a drift-compensating monotonic sleep:
    ```python
    target_dt = 1.0 / target_fps
    next_tick = time.monotonic() + target_dt
    while recording:
        # Step acquisition and teleop
        ...
        sleep_duration = next_tick - time.monotonic()
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        else:
            logger.warning(f"Frame overrun by {-sleep_duration*1000:.2f} ms")
        next_tick += target_dt
    ```

---

## 5. Floor Pickup Task Protocol & Data Strategy

To train a robust policy for picking up items from the floor:

1. **Workspace Boundary Enforcement:**
   * Clamp the maximum downward extension of the arm in software so the gripper stops $2\text{ mm}$ above the floor level. Do not rely on physical motor stall.
2. **Demonstration Phase Consistency:**
   Every episode must follow a three-stage temporal structure:
   * **Stage 1: Pre-Grasp Alignment (0% - 40% duration)** — Arm moves from home pose to a hovering position 5–10 cm above the object on the floor. Wrist camera aligns with the object's primary axis.
   * **Stage 2: Descent & Grasp (40% - 70% duration)** — Linear vertical descent, gripper close command issued, wait 200 ms for contact stabilization.
   * **Stage 3: Lift & Hold (70% - 100% duration)** — Lift object 15 cm above ground to demonstrate grasp stability.
3. **Data Diversity Strategy:**
   * Vary object location across a $30\text{ cm} \times 30\text{ cm}$ floor grid.
   * Introduce lighting variations (ambient room light vs. directional desk lamp).
   * Record negative/recovery demonstrations (e.g., misaligned grasp, retreat, and re-attempt).

---

## 6. Verification and Acceptance Criteria

Before recording training datasets, run the verification script `scripts/check_cameras.py` and pass the following checklist:

- [ ] **Frame Rate Stability:** Both cameras maintain $30.0 \pm 1.0\text{ FPS}$ continuously for 5 minutes.
- [ ] **Camera Sync Drift:** Temporal discrepancy between base and wrist frames stays strictly below $15\text{ ms}$.
- [ ] **Serial Latency:** Read/write roundtrip on the Feetech STS bus executes in $< 8\text{ ms}$.
- [ ] **Visual Field Coverage:** Base camera shows both the ground pick area and the arm base; wrist camera keeps object in frame during the descent phase.
- [ ] **Storage Integrity:** Recorded episodes load cleanly with `lerobot.common.datasets.lerobot_dataset.LeRobotDataset`.

---

# 7. Implementation Status & Agent Handoff (2026-10-03)

> **Read this section first.** The specification above has been implemented in
> this repository (~25 source/test files). This section records what is done,
> what is verified, and exactly where work stopped.

## 7.1 What exists (all spec sections covered)

| Spec | Status | Where |
|---|---|---|
| §3 layout | done | `pyproject.toml`, `configs/*.yaml`, `so101_collector/` (package), `tests/`, `README.md`. Note: Python packages are namespaced under `so101_collector/` (spec's flat `core/`, `scripts/` became `so101_collector/core/`, `so101_collector/scripts/`) to avoid top-level `core` module collision. |
| §4.1 hardware | done | `core/hardware_interface.py` — `FeetechBus` (Dynamixel-1.0-style framing `FF FF …`, checksum, STS3215 registers 40/42/56/58/60, lazy pyserial) + `SO101HardwareInterface` (retry 3x → `BusCommunicationError`, per-joint safety clamping, `read_state()` returns positions/velocities/efforts, broadcast sync-write, `measure_latency()`, encoder-range calibration guard). Known ping vector `FF FF 01 02 01 FB` unit-tested. |
| §4.2 cameras | done | `core/camera_hub.py` — `V4L2Camera` (CAP_V4L2, MJPG, 640×480@30, exposure lock, buffersize 1, BGR→RGB at the cv2 boundary) + daemon workers named `BaseCameraThread`/`WristCameraThread` with lock-guarded latest-frame double buffer + reopen-recovery; `get_synced_frames() -> (base, wrist, dt_sync)` warns >20 ms (logger + `on_sync_warning` hook). |
| §4.3 writer | done | `core/dataset_writer.py` — `EpisodeBuffer` (`observation.state`/`action` (N,6) float32, RGB uint8 frames), `save_episode(task_name, success)` stages → `os.replace` → `episodes.jsonl` append = commit point; `discard_episode()` RAM-only; failed save **keeps the buffer** for FINISH-retry; Parquet (pyarrow) with `.npz` fallback; PyAV H.264 with OpenCV `mp4v` fallback; LeRobot-style `meta/info.json` + `tasks.jsonl`; POSIX separators in manifests. |
| §4.4 loop/CLI | done | `timing.py` (`LoopTimer`: absolute deadlines, resync after >3 periods, overrun stats, hybrid sleep+spin — see §7.3) and `scripts/record_episodes.py` — state machine IDLE→WARMUP(2 s)→RECORDING; Space/Enter/Esc/R/Q + gamepad trigger/green/red via `console.py`; `--auto` headless mode (`TimedCommandReader`); `--simulate`. |
| §4 IK/teleop | done | `core/ik.py` closed-form FK/IK (elbow up/down), `clamp_workspace()` enforcing the 2 mm software floor stop — FK/IK round trips verified to 1e-6 mm in tests; `core/teleop_controller.py` (leader-arm mirror + Cartesian SpaceMouse through IK); `devices.py` stdlib evdev readers. |
| §5 protocol | documented | Three-stage demo structure + diversity strategy in `README.md`; floor clamp automated in IK. |
| §6 acceptance | done | `scripts/check_cameras.py` prints PASS/FAIL checklist (FPS windows, sync <15 ms, serial <8 ms), snapshots for FOV, exit code 0/1; manual items listed as `[ ? ]`. |
| tests | done | **78 passing**: `test_hardware`, `test_synchronization`, `test_ik`, `test_config`, `test_dataset_writer`, `test_pipeline` (end-to-end on simulated rig). Run: `python -m pytest tests/ -q`. |

Also present: `simulation.py` (`SimulatedBus`, `SimulatedCamera` epoch-aligned pacing, `ScriptedTeleop`) used by `--simulate` and the test suite; `README.md` with usage/architecture/decisions.

## 7.2 Verified runs (this machine: Windows, Python 3.12.10)

- `python -m so101_collector.scripts.record_episodes --simulate --auto 2 --max-episodes 1 …` → episode saved: `episode_000000` (mp4×2 + table + manifest + info.json), clean teardown, exit 0.
- `python -m so101_collector.scripts.check_cameras --simulate --duration 6` → FPS PASS (30.00 in every window), sync PASS, rc=0. *(Caveat: sync read 0.00 ms — likely an artifact of the clock issue below; re-verify after §7.3.)*
- `python -m so101_collector.scripts.calibrate_joints --show` → graceful `[FAIL]` + exit 2 without hardware (pyserial not installed here).
- All three `--help`s OK; shipped YAMLs load and pass strict validation.

## 7.3 IN PROGRESS — loop-timing clock investigation (start here)

**Goal:** meet spec §2.2 “state readings and camera frames must maintain <5 ms jitter”.

Measured facts in this sandbox:

1. `time.sleep` is **accurate**: witnessed with `time.perf_counter`, `sleep(33.3ms)` → 33.4–33.9 ms, `sleep(1ms)` → 1.24–1.55 ms, `sleep(20ms)` → 20.3–20.8 ms.
2. **`time.monotonic()` is quantized here**: only ~2 distinct readings during a 20 ms busy-spin; over the same interval perf_counter read 50.00 ms while monotonic read 47.00 ms. (Both are QPC on normal CPython/Windows, so this smells like sandbox clock virtualization/hooking — but treat it as given.)
3. Therefore all observed `LoopTimer` “Frame overrun by 5–14 ms” warnings (pattern 13.67 / 11.33 / 9.00 / 7.67 / 5.33 …) are an **artifact of measuring with the quantized monotonic clock**, not real lateness. `timeBeginPeriod(1)` is requested and returns success; the hybrid wait was tuned against a phantom.
4. Already added and tested (78 pass): `LoopTimer._wait_until()` (coarse sleep to `deadline − spin_margin`, then bounded busy-spin with a stall-detector fallback for frozen/fake clocks) and `_adapt_spin_margin()` (jump up on overshoot, decay 1 %/tick), plus `request_high_resolution_timer()` (Windows `timeBeginPeriod(1)` + `atexit` release). Keep these — they genuinely protect real Windows hosts where sleep quantizes to 15.6 ms — but they cannot help while the *clock* feeding them is coarse.

**Next steps, in order:**

1. Switch the default precision clock `time.monotonic` → `time.perf_counter` in:
   - `so101_collector/timing.py` (`LoopTimer.__init__` default `clock=`)
   - `so101_collector/core/camera_hub.py` (`V4L2Camera.read` timestamp)
   - `so101_collector/simulation.py` (`SimulatedCamera` default `clock=`)
   - `so101_collector/core/runner.py` (tick/warmup/stats timestamps)
   - `so101_collector/core/hardware_interface.py` (`read_state` timestamp)
   - `so101_collector/console.py` (`TimedCommandReader` default `clock=`)
   `perf_counter` is monotonic by contract, so deadline semantics are unchanged; keep “monotonic deadline” wording in docstrings.
2. `python -m pytest tests/ -q` — fake-clock tests should be unaffected; fix any that assert wall-clock specifics.
3. Jitter probe: run `LoopTimer(30)` for 3 s → expect overruns ≈ 0, `max_jitter` well under 5 ms; watch the adaptive spin margin converge to its ~0.5 ms floor.
4. Re-run `check_cameras --simulate --duration 6` and confirm sync is small but **non-zero** (both cameras previously read the same quantized tick → suspicious 0.00 ms).
5. Final sweep: full test suite, all `--help`s, unused-import/lint pass (ruff is configured in `pyproject.toml` but **not installed** in this env — ask before installing), then summarize for the user.

## 7.4 Other caveats / nice-to-haves

- `pyserial`, `pyarrow`, `av` are **not installed** here: serial/H.264/Parquet code paths are lazy-import-guarded and untested on real hardware; add `pytest.importorskip` tests when deps are available.
- evdev SpaceMouse/gamepad backends are Linux-only (documented in README “Platform notes”); leader-arm teleop, `--simulate`, and recording work on Windows/macOS.
- Spec §5 diversity strategy and §6’s two manual checklist items are documented, not automated.
- Shipped `configs/robot_config.yaml` offsets are placeholders (`-180.0`, servo horn centered); the calibration guard in `write_positions` warns when a joint maps outside 0..4095 ticks.
- Nothing has been committed to git; all changes are unstaged in the working tree.
