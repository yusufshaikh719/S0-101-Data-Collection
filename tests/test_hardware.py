"""Hardware interface tests: packet framing, retries, clamping, decoding.

Everything runs against an in-memory fake bus - no serial port required.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from so101_collector.core.hardware_interface import (
    DEG_PER_TICK,
    Instruction,
    Register,
    SO101HardwareInterface,
    StatusPacket,
    build_instruction_packet,
    degrees_to_ticks,
    parse_status_packet,
    signed_magnitude,
    ticks_to_degrees,
)
from so101_collector.exceptions import BusCommunicationError


class FakeBus:
    """Minimal PacketTransport double with scripted failures."""

    def __init__(self, *, known_ids=range(1, 7)) -> None:
        self.opened = False
        self.fail_reads = 0
        self.fail_writes = 0
        self.fail_pings = 0
        self.known_ids = set(known_ids)
        self.position_ticks = {i: 2048 for i in self.known_ids}
        self.sync_writes: list[tuple[int, list[tuple[int, bytes]]]] = []
        self.read_count = 0
        self.attempts: list[str] = []

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def ping(self, device_id: int) -> StatusPacket:
        self.attempts.append(f"ping:{device_id}")
        if self.fail_pings > 0 or device_id not in self.known_ids:
            self.fail_pings -= 1 if self.fail_pings > 0 else 0
            raise BusCommunicationError(f"no response from servo {device_id}")
        return StatusPacket(device_id, 0, b"")

    def read(self, device_id: int, address: int, length: int) -> bytes:
        self.attempts.append(f"read:{device_id}:{address}")
        self.read_count += 1
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise BusCommunicationError("simulated packet loss")
        if address == Register.PRESENT_POSITION and length == 6:
            # ticks=2048 (180 deg), speed=1024|100 (dir flag -> -100), load=512
            return struct.pack("<HHH", self.position_ticks[device_id], 1024 | 100, 512)
        if address == Register.PRESENT_POSITION and length == 2:
            return struct.pack("<H", self.position_ticks[device_id])
        raise AssertionError(f"unexpected read {address}/{length}")

    def write(self, device_id: int, address: int, data: bytes) -> StatusPacket:
        self.attempts.append(f"write:{device_id}:{address}")
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise BusCommunicationError("simulated write failure")
        return StatusPacket(device_id, 0, b"")

    def sync_write(self, address: int, items) -> None:
        self.attempts.append(f"sync_write:{address}")
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise BusCommunicationError("simulated sync_write failure")
        self.sync_writes.append((int(address), list(items)))


# ---------------------------------------------------------------------------
# Packet framing
# ---------------------------------------------------------------------------
def test_ping_packet_known_vector():
    """Servo 1 ping must be FF FF 01 02 01 FB (verified against the STS spec)."""
    packet = build_instruction_packet(1, Instruction.PING)
    assert packet == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])


def test_instruction_packet_checksum_covers_body():
    packet = build_instruction_packet(6, Instruction.WRITE, bytes([42, 0x00, 0x08]))
    body, checksum = packet[2:-1], packet[-1]
    assert (~sum(body)) & 0xFF == checksum
    assert len(packet) == 2 + len(body) + 1


def test_parse_status_packet_roundtrip():
    params = bytes([0x00, 0x08])
    body = bytes([3, len(params) + 2, 0x00]) + params  # id, length, error
    raw = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])
    status = parse_status_packet(raw)
    assert status.device_id == 3
    assert status.error == 0
    assert status.params == params


def test_parse_status_packet_rejects_bad_checksum():
    body = bytes([3, 2, 0x00])  # id=3, length=2 (error + checksum), no params
    raw = b"\xff\xff" + body + bytes([0xEE])  # wrong checksum
    with pytest.raises(BusCommunicationError, match="checksum"):
        parse_status_packet(raw)


def test_parse_status_packet_rejects_error_flags():
    body = bytes([3, 2, 0x10])  # instruction error bit set
    raw = b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])
    with pytest.raises(BusCommunicationError, match="error flags"):
        parse_status_packet(raw)


def test_parse_status_packet_rejects_truncated():
    with pytest.raises(BusCommunicationError, match="truncated"):
        parse_status_packet(b"\xff\xff\x01\x02")


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
def test_ticks_degrees_roundtrip():
    # The raw servo domain is [0, 4095] ticks == [0, 360) deg; calibration
    # offsets shift joint angles into that window.
    for ticks in (0, 1, 512, 1536, 2048, 4095):
        degrees = ticks_to_degrees(ticks)
        assert degrees == pytest.approx(ticks * 360.0 / 4096)
        assert degrees_to_ticks(degrees) == ticks


def test_offset_applied_on_both_directions():
    # -180 deg offset centers the servo on the joint's zero.
    assert ticks_to_degrees(512, offset_deg=-180.0) == pytest.approx(-135.0)
    assert degrees_to_ticks(-135.0, offset_deg=-180.0) == 512
    assert ticks_to_degrees(0, offset_deg=1.5) == pytest.approx(1.5)
    assert degrees_to_ticks(1.5, offset_deg=1.5) == 0


def test_signed_magnitude_direction_bit():
    assert signed_magnitude(512) == 512
    assert signed_magnitude(1024 | 100) == -100
    assert signed_magnitude(0) == 0


# ---------------------------------------------------------------------------
# SO101HardwareInterface
# ---------------------------------------------------------------------------
def test_read_state_decodes_positions_velocities_efforts(robot_config):
    bus = FakeBus()
    interface = SO101HardwareInterface(robot_config, bus=bus)
    interface.open()
    state = interface.read_state()
    assert state.positions.shape == (6,)
    # 2048 ticks = 180 deg, plus shoulder_pan's -178.5 deg calibration offset.
    assert state.positions[0] == pytest.approx(180.0 - 178.5, abs=0.1)
    assert state.positions[1] == pytest.approx(180.0 - 180.0, abs=0.1)
    # speed raw = -100 steps/s -> deg/s; load 512/1023 -> ~0.5 effort.
    assert state.velocities[0] == pytest.approx(-100 * DEG_PER_TICK, abs=1e-3)
    assert state.efforts[0] == pytest.approx(512 / 1023.0, abs=1e-3)
    assert state.timestamp > 0
    interface.close()


def test_write_positions_clamps_to_joint_limits(robot_config):
    bus = FakeBus()
    interface = SO101HardwareInterface(robot_config, bus=bus)
    interface.open()

    targets = [999.0, -999.0, 0.0, 0.0, 0.0, 30.0]
    applied = interface.write_positions(targets)

    assert applied[0] == pytest.approx(135.0)  # shoulder_pan max
    assert applied[1] == pytest.approx(-90.0)  # shoulder_lift min
    assert applied[5] == pytest.approx(30.0)  # inside limits -> untouched

    address, items = bus.sync_writes[-1]
    assert address == Register.GOAL_POSITION
    for index, (servo_id, payload) in enumerate(items):
        joint = robot_config.joints[index]
        assert servo_id == joint.servo_id
        ticks = struct.unpack("<H", payload)[0]
        assert joint.min_deg <= applied[index] <= joint.max_deg
        assert 0 <= ticks <= 4095


def test_write_positions_rejects_nan_and_wrong_size(robot_config):
    interface = SO101HardwareInterface(robot_config, bus=FakeBus())
    interface.open()
    with pytest.raises(ValueError, match="NaN"):
        interface.write_positions([0.0, np.nan, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="6 joint targets"):
        interface.write_positions([0.0, 1.0])
    interface.close()


def test_retry_recovers_from_transient_packet_loss(robot_config):
    bus = FakeBus()
    interface = SO101HardwareInterface(robot_config, bus=bus)
    interface.open()

    bus.fail_reads = 2  # first two attempts drop, third succeeds
    state = interface.read_state()
    assert state.positions.shape == (6,)
    joint1_reads = [a for a in bus.attempts if a.startswith("read:1:")]
    assert len(joint1_reads) == 3  # 1 + 2 retries for the failing joint
    # Remaining joints each read exactly once.
    for servo_id in range(2, 7):
        assert len([a for a in bus.attempts if a.startswith(f"read:{servo_id}:")]) == 1


def test_retry_exhaustion_raises_after_configured_attempts(robot_config):
    bus = FakeBus()
    interface = SO101HardwareInterface(robot_config, bus=bus)
    interface.open()

    bus.fail_reads = 99
    with pytest.raises(BusCommunicationError, match="after 4 attempts"):
        interface.read_state()
    # 4 attempts = initial + max_retries (3).
    read_attempts = [a for a in bus.attempts if a.startswith("read:")]
    assert len(read_attempts) == 4


def test_open_detects_missing_servo(robot_config):
    bus = FakeBus(known_ids=range(1, 6))  # gripper (id 6) absent
    interface = SO101HardwareInterface(robot_config, bus=bus)
    with pytest.raises(BusCommunicationError, match="gripper"):
        interface.open()
    assert not interface.is_open
    assert not bus.opened


def test_open_and_close_lifecycle(robot_config):
    bus = FakeBus()
    interface = SO101HardwareInterface(robot_config, bus=bus)
    assert not interface.is_open
    with interface as active:
        assert active.is_open and bus.opened
    assert not interface.is_open and not bus.opened


def test_go_home_commands_home_pose(robot_config):
    bus = FakeBus()
    interface = SO101HardwareInterface(robot_config, bus=bus)
    interface.open()
    applied = interface.go_home()
    assert applied.tolist() == pytest.approx(list(robot_config.home_pose))
