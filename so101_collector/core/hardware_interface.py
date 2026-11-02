"""Feetech STS serial bus for the SO-101 follower arm.

``FeetechBus`` handles framing, checksums and serial I/O and raises on the
first failure. ``SO101HardwareInterface`` adds tick/degree conversion, joint
clamping, state reads and retries.

Wire format::

    0xFF 0xFF | ID | LENGTH | INSTRUCTION | PARAMS... | CHECKSUM
    CHECKSUM  = ~(ID + LENGTH + INSTRUCTION + PARAMS...) & 0xFF
"""

from __future__ import annotations

import logging
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import numpy as np

from so101_collector.config import RobotConfig
from so101_collector.exceptions import BusCommunicationError, ConfigError

logger = logging.getLogger(__name__)

#: STS3215 encoder range: 0..4095 ticks span a full 360 degree revolution.
TICKS_PER_REV = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_REV
BROADCAST_ID = 0xFE


class Instruction(IntEnum):
    PING = 0x01
    READ = 0x02
    WRITE = 0x03
    REG_WRITE = 0x04
    ACTION = 0x05
    RESET = 0x06
    SYNC_WRITE = 0x83


class Register(IntEnum):
    """STS3215 control-table addresses (subset used by this pipeline)."""

    MODEL_NUMBER = 3  # u16
    DEVICE_ID = 5  # u8
    BAUDRATE = 6  # u8
    MIN_ANGLE = 9  # u16
    MAX_ANGLE = 11  # u16
    TORQUE_ENABLE = 40  # u8
    ACCELERATION = 41  # u8
    GOAL_POSITION = 42  # u16
    GOAL_SPEED = 46  # u16
    PRESENT_POSITION = 56  # u16
    PRESENT_SPEED = 58  # u16
    PRESENT_LOAD = 60  # u16
    PRESENT_VOLTAGE = 62  # u8
    PRESENT_TEMPERATURE = 63  # u8
    MOVING = 66  # u8


def build_instruction_packet(device_id: int, instruction: int, params: bytes = b"") -> bytes:
    """Serialize an instruction packet (pure function, unit-tested)."""
    if not 0 <= device_id <= 0xFE:
        raise ValueError(f"device_id {device_id} out of range 0..254")
    if len(params) > 250:
        raise ValueError(f"params too long: {len(params)} bytes")
    body = bytes((device_id, len(params) + 2, instruction)) + params
    checksum = (~sum(body)) & 0xFF
    return b"\xff\xff" + body + bytes((checksum,))


def parse_status_packet(raw: bytes) -> StatusPacket:
    """Parse a status packet, validating header length and checksum."""
    if len(raw) < 6:
        raise BusCommunicationError(f"status packet truncated ({len(raw)} bytes): {raw.hex()}")
    if raw[0:2] != b"\xff\xff":
        raise BusCommunicationError(f"bad status packet header: {raw[:2].hex()}")
    device_id, length = raw[2], raw[3]
    if length < 2 or len(raw) != 4 + length:
        raise BusCommunicationError(
            f"status packet length mismatch: header says {length}, got {len(raw) - 4} payload bytes"
        )
    body = raw[2 : 4 + length - 1]  # ID, LENGTH, ERROR, PARAMS... (checksum excluded)
    expected = raw[-1]
    if (~sum(body)) & 0xFF != expected:
        raise BusCommunicationError(
            f"status packet checksum mismatch (got 0x{expected:02x}, "
            f"want 0x{(~sum(body)) & 0xFF:02x})"
        )
    error = raw[4]
    if error != 0x00:
        raise BusCommunicationError(f"servo {device_id} returned error flags 0x{error:02x}")
    return StatusPacket(device_id=device_id, error=error, params=raw[5:-1])


@dataclass(frozen=True)
class StatusPacket:
    device_id: int
    error: int
    params: bytes


def signed_magnitude(raw: int) -> int:
    """Decode Feetech's direction-flagged 11-bit magnitude (speed/load)."""
    raw &= 0xFFFF
    return -(raw & 0x3FF) if raw & 0x400 else (raw & 0x3FF)


def ticks_to_degrees(ticks: int, offset_deg: float = 0.0) -> float:
    return ticks * DEG_PER_TICK + offset_deg


def degrees_to_ticks(deg: float, offset_deg: float = 0.0) -> int:
    ticks = int(round((deg - offset_deg) / DEG_PER_TICK))
    return min(max(ticks, 0), TICKS_PER_REV - 1)


class PacketTransport(Protocol):
    """Structural type for the low-level bus used by the interface."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def ping(self, device_id: int) -> StatusPacket: ...

    def read(self, device_id: int, address: int, length: int) -> bytes: ...

    def write(self, device_id: int, address: int, data: bytes) -> StatusPacket: ...

    def sync_write(self, address: int, items: Sequence[tuple[int, bytes]]) -> None: ...


class FeetechBus:
    """Raw Feetech STS packet transport over USB-TTL (no retries)."""

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        timeout_s: float = 0.05,
        *,
        serial_factory: Callable[..., object] | None = None,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout_s = timeout_s
        self._serial_factory = serial_factory
        self._serial: object | None = None

    @property
    def is_open(self) -> bool:
        return self._serial is not None

    def open(self) -> None:
        if self._serial is not None:
            return
        factory = self._serial_factory
        if factory is None:
            try:
                import serial  # noqa: PLC0415 - lazy: pyserial only needed on hardware hosts
            except ImportError as exc:  # pragma: no cover - depends on environment
                # Permanent environment problem - not retryable.
                raise ConfigError(
                    "pyserial is required for hardware access (pip install pyserial)"
                ) from exc
            factory = serial.Serial
        try:
            self._serial = factory(
                port=self._port,
                baudrate=self._baudrate,
                timeout=self._timeout_s,
                write_timeout=self._timeout_s,
            )
        except Exception as exc:  # serial.SerialException and OS errors
            raise BusCommunicationError(f"failed to open serial port {self._port}: {exc}") from exc

    def close(self) -> None:
        serial_port, self._serial = self._serial, None
        if serial_port is not None:
            try:
                serial_port.close()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover - hardware path
                logger.warning("error closing serial port %s: %s", self._port, exc)

    def ping(self, device_id: int) -> StatusPacket:
        return self._exchange(build_instruction_packet(device_id, Instruction.PING))

    def read(self, device_id: int, address: int, length: int) -> bytes:
        status = self._exchange(
            build_instruction_packet(
                device_id, Instruction.READ, bytes((address & 0xFF, length & 0xFF))
            )
        )
        if len(status.params) != length:
            raise BusCommunicationError(
                f"servo {device_id}: read address {address} expected {length} bytes, "
                f"got {len(status.params)}"
            )
        return status.params

    def write(self, device_id: int, address: int, data: bytes) -> StatusPacket:
        return self._exchange(
            build_instruction_packet(device_id, Instruction.WRITE, bytes((address & 0xFF,)) + data)
        )

    def sync_write(self, address: int, items: Sequence[tuple[int, bytes]]) -> None:
        if not items:
            raise ValueError("sync_write requires at least one (device_id, data) item")
        data_len = len(items[0][1])
        if any(len(d) != data_len for _, d in items):
            raise ValueError("sync_write items must all have equal data length")
        params = bytes((address & 0xFF, data_len & 0xFF))
        for device_id, data in items:
            params += bytes((device_id & 0xFF,)) + data
        # Broadcast command: no status response is ever sent.
        self._exchange(build_instruction_packet(BROADCAST_ID, Instruction.SYNC_WRITE, params),
                       expect_response=False)

    # -- internals ---------------------------------------------------------
    def _exchange(self, packet: bytes, *, expect_response: bool = True) -> StatusPacket | None:
        serial_port = self._serial
        if serial_port is None:
            raise BusCommunicationError(f"serial port {self._port} is not open")
        try:
            reset = getattr(serial_port, "reset_input_buffer", None)
            if reset is not None:
                reset()  # drop stale half-duplex bytes before a fresh command
            serial_port.write(packet)  # type: ignore[attr-defined]
            if not expect_response:
                return None
            header = serial_port.read(4)  # type: ignore[attr-defined]
            if len(header) != 4:
                raise BusCommunicationError(
                    f"timeout waiting for status header on {self._port} "
                    f"({len(header)}/4 bytes)"
                )
            if header[0:2] != b"\xff\xff":
                raise BusCommunicationError(f"bad status header on {self._port}: {header.hex()}")
            length = header[3]
            payload = serial_port.read(length)  # type: ignore[attr-defined]
            if len(payload) != length:
                raise BusCommunicationError(
                    f"timeout waiting for status payload on {self._port} "
                    f"({len(payload)}/{length} bytes)"
                )
        except BusCommunicationError:
            raise
        except Exception as exc:  # serial.SerialException, OSError, ...
            raise BusCommunicationError(f"serial I/O failure on {self._port}: {exc}") from exc
        return parse_status_packet(header + payload)


@dataclass(frozen=True)
class JointState:
    """Measured arm state at one point in time.

    positions: joint angles in degrees, velocities: deg/s, efforts: applied
    load as a signed fraction of the servo's 10-bit load range [-1, 1].
    """

    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray
    timestamp: float

    def __post_init__(self) -> None:
        for name in ("positions", "velocities", "efforts"):
            value = getattr(self, name)
            if value.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {value.shape}")


@dataclass(frozen=True)
class LatencyStats:
    """Round-trip timing of the serial bus (acceptance: < 8 ms)."""

    samples: int
    read_mean_ms: float
    read_max_ms: float
    write_mean_ms: float
    write_max_ms: float

    @property
    def max_ms(self) -> float:
        return max(self.read_max_ms, self.write_max_ms)


class SO101HardwareInterface:
    """Joint-aware, retrying interface over the Feetech STS bus."""

    def __init__(
        self,
        config: RobotConfig,
        *,
        bus: PacketTransport | None = None,
        retry_backoff_s: float | None = None,
    ) -> None:
        self._config = config
        self._bus: PacketTransport = bus or FeetechBus(
            config.port, config.baudrate, config.timeout_s
        )
        self._backoff = config.retry_backoff_s if retry_backoff_s is None else retry_backoff_s
        self._clamped_warned: set[str] = set()
        self._range_warned: set[str] = set()
        self._opened = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def config(self) -> RobotConfig:
        return self._config

    @property
    def is_open(self) -> bool:
        return self._opened

    def open(self, *, verify_joints: bool = True) -> SO101HardwareInterface:
        if self._opened:
            return self
        self._robust("open", self._bus.open)
        self._opened = True
        if verify_joints:
            present = self.ping_all()
            missing = [j.name for j in self._config.joints if j.servo_id not in present]
            if missing:
                self.close()
                raise BusCommunicationError(
                    f"joint servo(s) not responding on {self._config.port}: {', '.join(missing)}"
                )
        logger.debug(
            "SO-101 bus open on %s @ %d baud (%d joints)",
            self._config.port,
            self._config.baudrate,
            len(self._config.joints),
        )
        return self

    def close(self) -> None:
        if not self._opened:
            return
        self._opened = False
        self._bus.close()

    def __enter__(self) -> SO101HardwareInterface:
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- operations --------------------------------------------------------
    def ping_all(self) -> set[int]:
        """Return the set of servo IDs that answer a ping."""
        responding: set[int] = set()
        for joint in self._config.joints:
            try:
                self._robust(f"ping({joint.servo_id})", self._bus.ping, joint.servo_id)
            except BusCommunicationError:
                logger.warning("servo id %d (%s) did not respond", joint.servo_id, joint.name)
                continue
            responding.add(joint.servo_id)
        return responding

    def read_state(self) -> JointState:
        """Read positions/velocities/efforts for all six joints."""
        n = len(self._config.joints)
        positions = np.empty(n, dtype=np.float32)
        velocities = np.empty(n, dtype=np.float32)
        efforts = np.empty(n, dtype=np.float32)
        for index, joint in enumerate(self._config.joints):
            raw = self._robust(
                f"read({joint.name})",
                self._bus.read,
                joint.servo_id,
                Register.PRESENT_POSITION,
                6,
            )
            ticks, speed_raw, load_raw = struct.unpack("<HHH", raw)
            positions[index] = ticks_to_degrees(ticks, joint.offset_deg)
            # Speed/load use the direction-flagged magnitude encoding; the
            # speed unit is encoder steps/s, i.e. deg/s after scaling.
            velocities[index] = signed_magnitude(speed_raw) * DEG_PER_TICK
            efforts[index] = signed_magnitude(load_raw) / 1023.0
        return JointState(
            positions=positions,
            velocities=velocities,
            efforts=efforts,
            timestamp=time.monotonic(),
        )

    def write_positions(self, targets: Sequence[float]) -> np.ndarray:
        """Clamp and write target joint angles in degrees.

        Returns the clamped targets that were sent. Clamping keeps each servo
        inside its configured range.
        """
        desired = np.asarray(targets, dtype=np.float64).reshape(-1)
        if desired.size != len(self._config.joints):
            raise ValueError(
                f"expected {len(self._config.joints)} joint targets, got {desired.size}"
            )
        if not np.all(np.isfinite(desired)):
            raise ValueError("joint targets contain NaN/inf; refusing to write to hardware")

        clamped = np.empty_like(desired)
        for index, joint in enumerate(self._config.joints):
            clamped[index] = joint.clamp(float(desired[index]))
            if clamped[index] != desired[index] and joint.name not in self._clamped_warned:
                self._clamped_warned.add(joint.name)
                logger.warning(
                    "joint '%s' target %.2f deg clamped to [%.2f, %.2f] (safety limit)",
                    joint.name,
                    desired[index],
                    joint.min_deg,
                    joint.max_deg,
                )
            # Calibration guard: a well-configured offset maps the whole
            # joint stroke into the servo's 0..4095 encoder window.
            raw_ticks = (clamped[index] - joint.offset_deg) / DEG_PER_TICK
            if not 0.0 <= raw_ticks <= TICKS_PER_REV - 1 and joint.name not in self._range_warned:
                self._range_warned.add(joint.name)
                logger.warning(
                    "joint '%s' maps to %.0f encoder ticks (outside 0..4095); "
                    "offset_deg is likely wrong - run so101-calibrate",
                    joint.name,
                    raw_ticks,
                )

        items = [
            (joint.servo_id, struct.pack("<H", degrees_to_ticks(clamped[index], joint.offset_deg)))
            for index, joint in enumerate(self._config.joints)
        ]
        self._robust(
            "sync_write(goal_position)",
            self._bus.sync_write,
            Register.GOAL_POSITION,
            items,
        )
        return clamped.astype(np.float32)

    def go_home(self) -> np.ndarray:
        """Drive every joint to its configured home pose."""
        return self.write_positions(self._config.home_pose)

    def set_torque(self, enabled: bool) -> None:
        for joint in self._config.joints:
            self._robust(
                f"write(torque, {joint.name})",
                self._bus.write,
                joint.servo_id,
                Register.TORQUE_ENABLE,
                bytes((1 if enabled else 0,)),
            )

    def read_torque(self) -> np.ndarray:
        """Per-joint torque-enable state read back from the servos.

        Used to confirm a torque release actually took effect: a joint that
        still reports enabled will fight a hand-guided drag.
        """
        states = np.empty(len(self._config.joints), dtype=bool)
        for index, joint in enumerate(self._config.joints):
            raw = self._robust(
                f"read(torque, {joint.name})",
                self._bus.read,
                joint.servo_id,
                Register.TORQUE_ENABLE,
                1,
            )
            states[index] = bool(raw[0])
        return states

    def read_raw_ticks(self) -> np.ndarray:
        """Raw encoder ticks for all joints (calibration helper)."""
        ticks = np.empty(len(self._config.joints), dtype=np.int32)
        for index, joint in enumerate(self._config.joints):
            raw = self._robust(
                f"read({joint.name})",
                self._bus.read,
                joint.servo_id,
                Register.PRESENT_POSITION,
                2,
            )
            ticks[index] = struct.unpack("<H", raw)[0]
        return ticks

    def measure_latency(self, samples: int = 50) -> LatencyStats:
        """Measure read/write round-trip latency (acceptance: < 8 ms)."""
        if samples < 1:
            raise ValueError("samples must be >= 1")
        read_ms: list[float] = []
        write_ms: list[float] = []
        for _ in range(samples):
            start = time.perf_counter()
            state = self.read_state()
            read_ms.append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            self.write_positions(state.positions)
            write_ms.append((time.perf_counter() - start) * 1000.0)
        return LatencyStats(
            samples=samples,
            read_mean_ms=float(np.mean(read_ms)),
            read_max_ms=float(np.max(read_ms)),
            write_mean_ms=float(np.mean(write_ms)),
            write_max_ms=float(np.max(write_ms)),
        )

    # -- error recovery ----------------------------------------------------
    def _robust(self, label: str, operation: Callable[..., object], *args: object) -> object:
        """Run ``operation`` with retries.

        Transient packet loss is retried up to ``max_retries`` times with a
        short backoff, then raises ``BusCommunicationError``.
        """
        attempts = self._config.max_retries + 1
        last_error: BusCommunicationError | None = None
        for attempt in range(attempts):
            try:
                return operation(*args)
            except BusCommunicationError as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                logger.warning(
                    "bus operation '%s' failed (attempt %d/%d): %s - retrying",
                    label,
                    attempt + 1,
                    attempts,
                    exc,
                )
                time.sleep(min(self._backoff * (attempt + 1), 0.05))
        raise BusCommunicationError(
            f"bus operation '{label}' failed after {attempts} attempts: {last_error}"
        ) from last_error
