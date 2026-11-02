# Getting started on a new machine

Everything needed to take a second person from a fresh clone to a verified
recording, on Windows, macOS or Linux.

## What you need

- SO-101 follower arm (6 servos) with its 12 V power supply
- USB serial adapter for the servo bus
- Two USB cameras (base and wrist)
- Python 3.12+ and git

A leader arm is optional: `--teleop backdrive` (hand-guiding) needs only the
follower.

## 1. Install

```bash
git clone <repo-url> S0-101-Data-Collection
cd S0-101-Data-Collection

python3.12 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
                                   # Windows cmd:        .venv\Scripts\activate.bat
pip install -e ".[visualize]"
```

On Windows, create the environment with `py -3.12 -m venv .venv`. The install
pulls PyTorch and LeRobot and takes a few minutes; `.[visualize]` adds the
Rerun viewer and can be dropped if you will not pass `--rerun`.

## 2. Find the devices

```bash
so101-setup --list
```

| OS | Serial port looks like | Camera identifier |
|---|---|---|
| Linux | `/dev/serial/by-id/usb-...` | `/dev/v4l/by-id/usb-...` |
| macOS | `/dev/cu.usbmodem*`, `/dev/cu.usbserial-*` | OpenCV index, quoted: `"0"` |
| Windows | `COM3`, `COM4`, ... | OpenCV index, quoted: `"0"` |

Ignore ports that are clearly not the arm (`cu.Bluetooth-Incoming-Port`,
`cu.debug-console`). Then write the configs:

```bash
so101-setup
```

Pick the follower's port, choose `0) none` for the leader, and choose the two
camera identifiers. This writes `configs/robot_config.yaml` and
`configs/camera_config.yaml`.

**Keeping the host out of git.** Both files are tracked and will now contain
this machine's serial number and camera indices. To keep them separate:

```bash
export SO101_CONFIG_DIR="$HOME/so101-configs"     # Windows: set SO101_CONFIG_DIR=...
so101-setup --config-dir "$SO101_CONFIG_DIR"
```

Every other command reads `SO101_CONFIG_DIR` automatically, so a shared
checkout keeps the shipped placeholders.

## 3. Calibrate the joints

```bash
so101-calibrate --show                       # raw ticks and derived degrees
so101-calibrate --capture min --write        # offsets from each joint's min stop
so101-calibrate --capture-home --write       # tucked pose -> home_deg
```

Torque is switched off during capture, so the arm is free — support it. The
first captured reference defines the offset; `home` and `max` are then only
checked against the configured range:

```bash
so101-calibrate --capture min home max       # dry run, prints MISMATCH lines
```

A `MISMATCH` means `min_deg` / `max_deg` in `robot_config.yaml` do not match the
real stops. Widen them until the deltas are small: anything the arm can reach
but the config excludes gets clamped in the recorded action.

## 4. Check the cameras and the rig

```bash
so101-check-cameras --duration 30 --snapshot-dir ./camera_snapshots
so101-setup --check
```

Open the PNGs: `base` should see the pick area and the arm's base, `wrist`
should keep the gripper and object in frame. If they are swapped, exchange the
two identifiers in `camera_config.yaml`.

Acceptance: sustained 30 FPS on both cameras, mean sync drift below the
threshold, serial round trips under 8 ms. `--sync-metric max` restores the
strict worst-case reading. If two cameras cannot both hold 30 FPS, lower `fps`
in `camera_config.yaml` **and** pass the same value to `so101-record --fps`;
timestamps are written from that number.

## 5. Record

```bash
so101-record --teleop backdrive --repo-id session_01 --duration 300
```

`--duration` is wall-clock seconds for the whole session, including startup.
Omit it to run until you press Q. `--max-episodes N` stops after N saved
episodes.

| key | action |
|---|---|
| automatic | an episode starts when a joint leaves home by > `--home-exit-tolerance` (8°) and ends on return within `--home-tolerance` (5°) |
| Enter | save this episode and continue |
| Esc | discard this episode |
| R | power up, ramp to the home pose, release torque again |
| Q / Ctrl-C | save the partial episode and quit |

With `--teleop backdrive`, torque is released about a second after the cameras
start — hold the wrist before then. Each session prints
`backdrive: torque released and confirmed on all N joints`; if it instead names
joints, the arm is still powered and must not be dragged.

## 6. Verify the recording

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

ds = LeRobotDataset("session_01", root="data/session_01")
print(ds.meta.total_episodes, ds.meta.total_frames, ds.meta.fps)

hf = ds.hf_dataset
state = np.stack([np.asarray(s, dtype=float) for s in hf["observation.state"]])
action = np.stack([np.asarray(a, dtype=float) for a in hf["action"]])
episode = np.asarray(hf["episode_index"])

for e in np.unique(episode):
    s, a = state[episode == e], action[episode == e]
    print(f"episode {e}: {len(s)} frames")
    print("  joint range:", np.round(s.max(0) - s.min(0), 1))
    print("  action lead:", round(float(np.abs(a[:-1] - s[1:]).mean()), 2),
          "same-frame:", round(float(np.abs(a - s).mean()), 2))
```

What to look for:

- every joint the task uses travels a meaningful range (tens of degrees)
- `action lead` is clearly smaller than `same-frame`
- no `state` outside the configured joint limits
- frame counts match what the session printed

A LeRobot v3 dataset stores all episodes in one `chunk-000` until it holds 1000
files; one chunk does not mean one episode. Episode boundaries live in
`meta/episodes/`.

## Platform notes

| | Linux | macOS | Windows |
|---|---|---|---|
| Serial port | `/dev/serial/by-id/...` | `/dev/cu.usbmodem*` | `COM3` |
| Camera | `/dev/v4l/by-id/...` | index `"0"` | index `"0"` |
| Keyboard | cbreak | cbreak | `msvcrt` |
| SpaceMouse / gamepad | yes (evdev) | no | no |
| Timer resolution | 1 ms | 1 ms | raised to 1 ms on start |

**Linux:** add your user to the `dialout` group (`sudo usermod -aG dialout $USER`)
if the port is not writable, and prefer the `by-id` symlinks so the arm keeps
its name across reboots.

**macOS:** grant camera access under System Settings → Privacy & Security →
Camera, or captures open but never deliver a frame. MJPG, exposure and buffer
size are V4L2-only and are rejected by AVFoundation; capture still works at the
requested resolution.

**Windows:** use the COM port the adapter reports. Only one program can hold a
serial port, so close vendor tools before recording.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `torque is still enabled on ...` | servos kept their position loop: check the 12 V supply and that no other process holds the port |
| arm confirmed released but still stiff | mechanical — gearbox friction and arm weight; grab close to the joint axis and move one joint at a time |
| `camera ... has not produced a frame yet` | wrong index, another app holding the camera, or macOS camera permission |
| `driver rejected CAP_PROP_FOURCC` | V4L2-only property on macOS/Windows; harmless |
| `Could not load libtorchcodec ... Falling back to pyav` | decoder fallback; recording is unaffected |
| `objc[...]: Class AVFFrameReceiver is implemented in both ...` | two FFmpeg copies in one process (cv2 and pyav); harmless |
| `Svt[info]: SVT-AV1 Encoder Lib ...` | encoder banner; `so101-record` sets `SVT_LOG=0` to mute it |
| `frame synchronization drift ...` | steady offset between two cameras with independent clocks; reported as `max sync drift` |
| `/dev/video0 is a dynamic device path` | config rejected a volatile path; use the `by-id` link or set `allow_dynamic_path: true` |
| `serial ... could not open port` | wrong port, unplugged adapter, or another process holding it |

## Known limits

- SpaceMouse and gamepad teleoperation are Linux-only; use the leader arm or
  `--teleop backdrive` elsewhere.
- This pipeline has been exercised on a physical rig under macOS. The Windows
  and Linux code paths (serial enumeration, `msvcrt` keys, V4L2 properties,
  high-resolution timer) are covered by tests but have not been verified on
  hardware here.
- A 30 FPS loop on a busy laptop can miss deadlines; check `overruns` and the
  `Frame overrun` warnings before trusting a long session.
