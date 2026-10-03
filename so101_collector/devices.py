"""Input-device layer: Linux evdev readers for SpaceMouse / gamepad.

Stdlib only (no ``evdev``/``hidapi`` dependency): events are read directly
from ``/dev/input/event*`` with ``os.O_NONBLOCK`` and unpacked with the
kernel's ``struct input_event`` layout.

Device paths must be persistent (``/dev/input/by-id/*``) so replugging can
never swap the SpaceMouse with the gamepad - same zero index-drift rule as
cameras and serial ports.

Shared value types live here (``CartCommand``, the ``InputDevice`` /
``ButtonSource`` protocols) because this module sits at the bottom of the
import graph: ``devices`` -> ``teleop_controller`` / ``console`` -> ``runner``.
"""

from __future__ import annotations

import logging
import math
import os
import struct
import sys
from dataclasses import dataclass
from typing import Mapping, Protocol

from so101_collector.exceptions import TeleopError

logger = logging.getLogger(__name__)

# --- Linux input event codes (from linux/input-event-codes.h) --------------
EV_KEY = 0x01
EV_ABS = 0x03

ABS_X, ABS_Y, ABS_Z = 0x00, 0x01, 0x02
ABS_RX, ABS_RY, ABS_RZ = 0x03, 0x04, 0x05

BTN_0 = 0x100
BTN_1 = 0x101
BTN_SOUTH = 0x130  # A / cross  (green)
BTN_EAST = 0x131  # B / circle (red)
BTN_NORTH = 0x133  # Y / triangle
BTN_WEST = 0x134  # X / square
BTN_TL = 0x136  # left bumper
BTN_TR = 0x137
BTN_TL2 = 0x138  # left trigger (when reported as a key)
BTN_TR2 = 0x139
BTN_SELECT = 0x13A
BTN_START = 0x13B
BTN_MODE = 0x13C

#: Semantic gamepad button names used by the command reader.
DEFAULT_GAMEPAD_BUTTONS: dict[int, str] = {
    BTN_SOUTH: "green",  # finish episode
    BTN_EAST: "red",  # abort episode
    BTN_NORTH: "reset",  # home pose
    BTN_TL: "start",  # "trigger" starts an episode
    BTN_TL2: "start",
}

#: SpaceMouse buttons mapped to gripper rates.
SPACEMOUSE_GRIPPER_CLOSE = BTN_0
SPACEMOUSE_GRIPPER_OPEN = BTN_1

_EVENT_STRUCT = struct.Struct("@llHHi")


@dataclass(frozen=True)
class InputEvent:
    type: int
    code: int
    value: int


@dataclass(frozen=True)
class CartCommand:
    """Rate command produced by a 6-DOF input device.

    ``linear_mm_s`` is the desired tool-tip velocity in the base frame
    (x forward, y left, z up); ``roll_deg_s`` the wrist roll rate and
    ``gripper_deg_s`` the gripper rate (positive = open).
    """

    linear_mm_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    roll_deg_s: float = 0.0
    gripper_deg_s: float = 0.0


class InputDevice(Protocol):
    """A polled 6-DOF rate-control device."""

    def open(self) -> None: ...

    def poll(self) -> CartCommand: ...

    def close(self) -> None: ...


class ButtonSource(Protocol):
    """A polled source of named buttons (gamepad command input)."""

    def open(self) -> None: ...

    def active_buttons(self) -> set[str]: ...

    def close(self) -> None: ...


class EvdevDevice:
    """Non-blocking reader for a Linux ``/dev/input/event*`` node."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._fd: int | None = None
        self._pending = b""

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def open(self) -> None:
        if self._fd is not None:
            return
        if not sys.platform.startswith("linux"):
            raise TeleopError(
                f"evdev input ({self._path}) requires Linux; use --teleop leader or --simulate "
                "on this platform"
            )
        try:
            self._fd = os.open(self._path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            raise TeleopError(
                f"cannot open input device {self._path}: {exc} "
                "(list devices with: ls -l /dev/input/by-id/)"
            ) from exc

    def close(self) -> None:
        fd, self._fd = self._fd, None
        self._pending = b""
        if fd is not None:
            try:
                os.close(fd)
            except OSError:  # pragma: no cover - best effort
                pass

    def poll(self) -> list[InputEvent]:
        """Drain and return all currently queued events."""
        if self._fd is None:
            raise TeleopError(f"input device {self._path} is not open")
        chunks: list[bytes] = [self._pending]
        while True:
            try:
                chunks.append(os.read(self._fd, 4096))
            except BlockingIOError:
                break
            except OSError as exc:
                raise TeleopError(f"read error on {self._path}: {exc}") from exc
        data = b"".join(chunks)
        usable = len(data) - (len(data) % _EVENT_STRUCT.size)
        self._pending = data[usable:]
        events = []
        for offset in range(0, usable, _EVENT_STRUCT.size):
            _secs, _usecs, ev_type, code, value = _EVENT_STRUCT.unpack_from(data, offset)
            events.append(InputEvent(type=ev_type, code=code, value=value))
        return events


class SpaceMouseSource:
    """6-DOF SpaceMouse rate control (axes normalized to +/-1, deadzone applied)."""

    def __init__(
        self,
        device_path: str,
        *,
        translation_speed_mm_s: float = 60.0,
        rotation_speed_deg_s: float = 90.0,
        gripper_speed_deg_s: float = 60.0,
        deadzone: float = 0.08,
        device: EvdevDevice | None = None,
    ) -> None:
        if not 0.0 <= deadzone < 1.0:
            raise ValueError(f"deadzone must be in [0, 1), got {deadzone}")
        self._path = device_path
        self._translation = float(translation_speed_mm_s)
        self._rotation = float(rotation_speed_deg_s)
        self._gripper = float(gripper_speed_deg_s)
        self._deadzone = float(deadzone)
        self._device = device or EvdevDevice(device_path)
        self._axes: dict[int, int] = {ABS_X: 0, ABS_Y: 0, ABS_Z: 0, ABS_RZ: 0}
        self._buttons: set[int] = set()

    def open(self) -> None:
        self._device.open()
        logger.info("SpaceMouse input open: %s", self._path)

    def close(self) -> None:
        self._device.close()

    def _normalize(self, raw: int) -> float:
        value = max(-1.0, min(1.0, raw / 32767.0))
        magnitude = abs(value)
        if magnitude < self._deadzone:
            return 0.0
        scaled = (magnitude - self._deadzone) / (1.0 - self._deadzone)
        return math.copysign(scaled, value)

    def poll(self) -> CartCommand:
        for event in self._device.poll():
            if event.type == EV_ABS and event.code in self._axes:
                self._axes[event.code] = event.value
            elif event.type == EV_KEY:
                if event.value:
                    self._buttons.add(event.code)
                else:
                    self._buttons.discard(event.code)

        vx = self._normalize(self._axes[ABS_X]) * self._translation
        vy = self._normalize(self._axes[ABS_Y]) * self._translation
        vz = self._normalize(self._axes[ABS_Z]) * self._translation
        roll = self._normalize(self._axes[ABS_RZ]) * self._rotation
        grip = 0.0
        if SPACEMOUSE_GRIPPER_OPEN in self._buttons:
            grip += self._gripper
        if SPACEMOUSE_GRIPPER_CLOSE in self._buttons:
            grip -= self._gripper
        return CartCommand(linear_mm_s=(vx, vy, vz), roll_deg_s=roll, gripper_deg_s=grip)


class GamepadButtonSource:
    """Named buttons from a gamepad evdev node (edge detection happens upstream)."""

    def __init__(
        self,
        device_path: str,
        *,
        buttons: Mapping[int, str] | None = None,
        device: EvdevDevice | None = None,
    ) -> None:
        self._path = device_path
        self._buttons = dict(buttons) if buttons is not None else dict(DEFAULT_GAMEPAD_BUTTONS)
        self._device = device or EvdevDevice(device_path)
        self._active: set[str] = set()

    def open(self) -> None:
        self._device.open()
        logger.info("gamepad input open: %s", self._path)

    def close(self) -> None:
        self._device.close()

    def active_buttons(self) -> set[str]:
        for event in self._device.poll():
            if event.type != EV_KEY:
                continue
            name = self._buttons.get(event.code)
            if name is None:
                continue
            if event.value:
                self._active.add(name)
            else:
                self._active.discard(name)
        return set(self._active)
