"""Exception hierarchy for the data collection pipeline.

Every pipeline error derives from ``SO101Error`` so callers can catch one
base class at the session boundary.
"""

from __future__ import annotations


class SO101Error(Exception):
    """Base class for every error raised by this package."""


class ConfigError(SO101Error):
    """Invalid or missing configuration value."""


class BusCommunicationError(SO101Error):
    """Feetech STS serial bus failure after all retries were exhausted."""


class CameraError(SO101Error):
    """Camera acquisition failure (open, read, or synchronization)."""


class DatasetWriteError(SO101Error):
    """Failure while persisting an episode to disk."""


class EpisodeStateError(SO101Error):
    """Illegal episode lifecycle operation (e.g. saving an empty episode)."""


class IKError(SO101Error):
    """Inverse kinematics could not produce a solution for the target."""


class TeleopError(SO101Error):
    """Teleoperation input device failure or misconfiguration."""
