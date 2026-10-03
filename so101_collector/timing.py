"""Drift-compensating fixed-rate loop timer.

A naive ``time.sleep(1 / fps)`` loop drifts without bound: every sleep
overshoots by a small scheduling error and the accumulated phase error makes
the effective frame rate wander.  :class:`LoopTimer` instead schedules each
tick against an *absolute* monotonic deadline (``next_tick += target_dt``),
so per-tick error never accumulates.

The timer is fully injectable (clock + sleep callables) which makes the
scheduling behaviour deterministic under test.
"""

from __future__ import annotations

import atexit
import logging
import sys
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TIMER_RESOLUTION_REQUESTED = False


def request_high_resolution_timer() -> None:
    """Ask Windows for 1 ms sleep granularity (no-op elsewhere).

    Without this, ``time.sleep`` wakes on ~15.6 ms quantum boundaries and a
    30 Hz loop on Windows is only achievable by luck.  Paired with
    ``timeEndPeriod(1)`` at interpreter exit.
    """
    global _TIMER_RESOLUTION_REQUESTED
    if _TIMER_RESOLUTION_REQUESTED or sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.winmm.timeBeginPeriod(1)
        _TIMER_RESOLUTION_REQUESTED = True
        atexit.register(_release_high_resolution_timer)
    except Exception:  # pragma: no cover - non-critical optimization
        logger.debug("could not request high-resolution timer", exc_info=True)


def _release_high_resolution_timer() -> None:
    global _TIMER_RESOLUTION_REQUESTED
    if not _TIMER_RESOLUTION_REQUESTED:
        return
    try:
        import ctypes

        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:  # pragma: no cover
        pass
    _TIMER_RESOLUTION_REQUESTED = False


@dataclass(frozen=True)
class TickStats:
    """Outcome of a single :meth:`LoopTimer.wait` call."""

    period_s: float  # wall time since the previous tick
    overrun_s: float  # 0.0 when on schedule, lateness otherwise


@dataclass(frozen=True)
class TimerSummary:
    """Aggregated timing statistics for the session."""

    ticks: int
    overruns: int
    mean_period_s: float
    max_overrun_s: float
    max_jitter_s: float  # max |period - target_dt|

    @property
    def mean_fps(self) -> float:
        return 1.0 / self.mean_period_s if self.mean_period_s > 0 else 0.0


class LoopTimer:
    """Precision 30 Hz (default) loop timer with overrun accounting.

    Overruns are always counted; ones exceeding 5 ms (by default) are also
    logged as warnings.  Pass ``overrun_warn_threshold_s=0`` to log every
    single overrun, as in the specification's reference snippet.
    """

    def __init__(
        self,
        target_fps: float = 30.0,
        *,
        clock=time.monotonic,
        sleep=time.sleep,
        max_catchup_periods: float = 3.0,
        overrun_warn_threshold_s: float | None = None,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")
        if max_catchup_periods <= 0:
            raise ValueError(f"max_catchup_periods must be > 0, got {max_catchup_periods}")
        self._target_dt = 1.0 / float(target_fps)
        self._clock = clock
        self._sleep = sleep
        self._max_catchup = max_catchup_periods * self._target_dt
        # Sub-threshold overruns are normal OS scheduling noise and are still
        # counted, but only meaningful lateness (default 5 ms) is warned about
        # so a 30 Hz log stays readable.
        self._warn_threshold = (
            0.005 if overrun_warn_threshold_s is None else overrun_warn_threshold_s
        )
        # Hybrid wait strategy: sleep until (deadline - spin_margin), then
        # busy-spin the remainder.  OS sleep quantization (15.6 ms Windows
        # ticks, sandboxed sleep hooks) otherwise overshoots the deadline by
        # up to one quantum, far beyond the <5 ms jitter requirement.  The
        # margin adapts: it jumps up when an overshoot is observed and decays
        # slowly while the host proves it can sleep accurately.
        self._spin_margin_s = 0.003
        self._min_spin_margin_s = 0.0005
        self._max_spin_margin_s = min(0.012, self._target_dt * 0.5)
        self._running = False
        self._next_tick = 0.0
        self._last_tick_end = 0.0
        self._ticks = 0
        self._overruns = 0
        self._max_overrun = 0.0
        self._max_jitter = 0.0
        self._sum_period = 0.0

    @property
    def target_dt(self) -> float:
        return self._target_dt

    @property
    def target_fps(self) -> float:
        return 1.0 / self._target_dt

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> LoopTimer:
        """Arm the schedule (first tick one period from now) and reset stats."""
        request_high_resolution_timer()
        now = self._clock()
        self._next_tick = now + self._target_dt
        self._last_tick_end = now
        self._ticks = 0
        self._overruns = 0
        self._max_overrun = 0.0
        self._max_jitter = 0.0
        self._sum_period = 0.0
        self._running = True
        return self

    def __enter__(self) -> LoopTimer:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self._running = False

    def stop(self) -> None:
        self._running = False

    def wait(self) -> TickStats:
        """Block until the next scheduled tick and return timing statistics.

        Uses a drift-compensating absolute deadline with a hybrid
        coarse-sleep + bounded spin (see ``__init__``), so per-tick error
        never accumulates and jitter stays within the <5 ms requirement
        regardless of the host's sleep granularity.

        Should be called *after* the work of each iteration, mirroring the
        pattern from the specification::

            timer.start()
            while recording:
                ...step acquisition and teleop...
                stats = timer.wait()
        """
        if not self._running:
            raise RuntimeError("LoopTimer.wait() called before start()")

        scheduled = self._next_tick
        end = self._wait_until(scheduled)
        period = end - self._last_tick_end
        overrun = max(0.0, end - scheduled)
        if overrun < 1e-7:  # float noise is not an overrun
            overrun = 0.0
        self._adapt_spin_margin(overrun)

        if overrun >= self._warn_threshold:
            logger.warning("Frame overrun by %.2f ms", overrun * 1000.0)

        self._ticks += 1
        self._sum_period += period
        if overrun > 0.0:
            self._overruns += 1
        self._max_overrun = max(self._max_overrun, overrun)
        self._max_jitter = max(self._max_jitter, abs(period - self._target_dt))

        self._next_tick = scheduled + self._target_dt
        if overrun > self._max_catchup:
            # We are hopelessly behind (e.g. a long GC pause or disk stall);
            # drop the missed slots instead of spiralling.
            logger.warning(
                "Loop fell %.2f ms behind; resynchronizing schedule",
                overrun * 1000.0,
            )
            self._next_tick = end + self._target_dt
        self._last_tick_end = end
        return TickStats(period_s=period, overrun_s=overrun)

    def _wait_until(self, scheduled: float) -> float:
        """Sleep coarsely to ``scheduled - spin_margin``, then spin precisely.

        A stall detector falls back to sleeping when the clock stops
        advancing (fake clocks in tests, pathological hosts), which keeps the
        spin loop bounded.
        """
        now = self._clock()
        coarse_deadline = scheduled - self._spin_margin_s
        if scheduled - now > self._spin_margin_s and now < coarse_deadline - 1e-6:
            self._sleep(max(coarse_deadline - now, 0.001))
            now = self._clock()

        last_value: float | None = None
        stall_streak = 0
        while scheduled - now > 1e-5:
            now = self._clock()
            if now == last_value:
                stall_streak += 1
                if stall_streak >= 2000:
                    # Clock is frozen (injected fake clock): sleep the rest.
                    self._sleep(max(scheduled - now, 0.001))
                    now = self._clock()
                    stall_streak = 0
                    last_value = None
            else:
                stall_streak = 0
                last_value = now
        return now

    def _adapt_spin_margin(self, overrun: float) -> None:
        if overrun > 0.0:
            # Overslept: need a bigger head start before spinning.
            self._spin_margin_s = min(
                self._spin_margin_s + overrun, self._max_spin_margin_s
            )
        else:
            # On schedule: give back spin budget slowly (hysteresis).
            self._spin_margin_s = max(
                self._spin_margin_s * 0.99, self._min_spin_margin_s
            )

    def summary(self) -> TimerSummary:
        mean_period = self._sum_period / self._ticks if self._ticks else 0.0
        return TimerSummary(
            ticks=self._ticks,
            overruns=self._overruns,
            mean_period_s=mean_period,
            max_overrun_s=self._max_overrun,
            max_jitter_s=self._max_jitter,
        )
