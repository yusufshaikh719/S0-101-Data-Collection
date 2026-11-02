"""Keyboard and gamepad command input.

The recording loop polls the reader once per tick:

    Enter / green button   FINISH  save the current episode
    Esc   / red button     ABORT   discard the current episode
    R                      RESET   drive the arm to its home pose
    Q                      QUIT    save any partial episode and exit

Keyboard input uses cbreak mode on POSIX and ``msvcrt`` on Windows. When stdin
is not a TTY the reader disables itself so headless runs still work.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from enum import Enum
from typing import Protocol

from so101_collector.devices import ButtonSource

logger = logging.getLogger(__name__)


class Command(Enum):
    FINISH = "finish"
    ABORT = "abort"
    RESET = "reset"
    QUIT = "quit"


DEFAULT_KEY_MAP: dict[str, Command] = {
    "\r": Command.FINISH,
    "\n": Command.FINISH,
    "\x1b": Command.ABORT,
    "r": Command.RESET,
    "R": Command.RESET,
    "q": Command.QUIT,
    "Q": Command.QUIT,
}

DEFAULT_GAMEPAD_MAP: dict[str, Command] = {
    "green": Command.FINISH,
    "red": Command.ABORT,
    "reset": Command.RESET,
}


class CommandReader(Protocol):
    """Pollable source of edge-triggered commands."""

    def read(self) -> Command | None: ...

    def close(self) -> None: ...


class NullCommandReader:
    """Headless reader that always returns ``None``."""

    def read(self) -> Command | None:
        return None

    def close(self) -> None:
        return None


class TimedCommandReader:
    """Headless auto-pilot: FINISH every ``record_s`` seconds.

    Used by ``so101-record --auto`` for CI smoke tests.
    """

    def __init__(self, record_s: float, *, clock=time.monotonic) -> None:
        if record_s <= 0:
            raise ValueError(f"record_s must be > 0, got {record_s}")
        self._record_s = float(record_s)
        self._clock = clock
        self._recording_since: float | None = None

    def read(self) -> Command | None:
        now = self._clock()
        if self._recording_since is None:
            self._recording_since = now
            return None
        if now - self._recording_since >= self._record_s:
            self._recording_since = now
            return Command.FINISH
        return None

    def close(self) -> None:
        return None


class KeyboardCommandReader:
    """Non-blocking single-key reader with raw-mode lifecycle management."""

    def __init__(
        self, *, key_map: dict[str, Command] | None = None, enabled: bool | None = None
    ) -> None:
        self._key_map = dict(key_map) if key_map is not None else dict(DEFAULT_KEY_MAP)
        self._saved_attrs = None
        self._fd: int | None = None
        self._enabled = bool(sys.stdin.isatty()) if enabled is None else enabled
        if self._enabled:
            self._enter_raw_mode()

    def _enter_raw_mode(self) -> None:
        if os.name == "posix":
            try:
                import termios
                import tty

                self._fd = sys.stdin.fileno()
                self._saved_attrs = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
            except Exception as exc:  # pragma: no cover - exotic terminals
                logger.warning("keyboard input disabled (cannot enter cbreak mode): %s", exc)
                self._enabled = False
        elif os.name != "nt":  # pragma: no cover
            logger.warning("keyboard input disabled on platform %s", sys.platform)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def read(self) -> Command | None:
        if not self._enabled:
            return None
        if os.name == "posix":
            return self._read_posix()
        return self._read_windows()  # pragma: no cover - Windows only

    def _read_posix(self) -> Command | None:
        import select

        assert self._fd is not None
        ready, _, _ = select.select([self._fd], [], [], 0)
        if not ready:
            return None
        data = os.read(self._fd, 8)
        if not data:
            return None
        if data == b"\x1b":
            return self._key_map.get("\x1b")
        if data.startswith(b"\x1b["):
            return None  # arrow/function key sequence
        return self._key_map.get(data[:1].decode("utf-8", errors="ignore"))

    def _read_windows(self) -> Command | None:  # pragma: no cover - Windows only
        import msvcrt

        if not msvcrt.kbhit():
            return None
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            msvcrt.getwch()  # scan-code half of the chord
            return None
        return self._key_map.get(char)

    def close(self) -> None:
        if self._saved_attrs is not None and self._fd is not None:
            try:
                import termios

                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
            except Exception:  # pragma: no cover - best effort restore
                pass
            self._saved_attrs = None
        self._enabled = False

    def __enter__(self) -> KeyboardCommandReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class GamepadCommandReader:
    """Edge-detects named gamepad buttons into commands."""

    def __init__(
        self,
        source: ButtonSource,
        *,
        mapping: dict[str, Command] | None = None,
    ) -> None:
        self._source = source
        self._mapping = dict(mapping) if mapping is not None else dict(DEFAULT_GAMEPAD_MAP)
        self._previous: set[str] = set()

    def read(self) -> Command | None:
        active = self._source.active_buttons()
        pressed = active - self._previous
        self._previous = active
        for name, command in self._mapping.items():
            if name in pressed:
                return command
        return None

    def close(self) -> None:
        self._source.close()


class CompositeCommandReader:
    """Fan-in over several readers, for example keyboard plus gamepad."""

    def __init__(self, *readers: CommandReader) -> None:
        self._readers = readers

    def read(self) -> Command | None:
        for reader in self._readers:
            command = reader.read()
            if command is not None:
                return command
        return None

    def close(self) -> None:
        for reader in self._readers:
            reader.close()
