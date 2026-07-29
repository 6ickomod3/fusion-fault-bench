"""Qualified frame identifiers for geometry-safe transform composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self, cast

type FrameKind = Literal["global", "ego", "camera", "lidar"]

_QUALIFIER_COUNTS: dict[str, int] = {
    "global": 1,
    "ego": 2,
    "camera": 3,
    "lidar": 3,
}


def _validate_qualifier(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("frame qualifiers must be strings")
    if not value or ":" in value or any(ord(character) < 0x20 for character in value):
        raise ValueError("frame qualifiers must be nonempty, colon-free printable strings")


@dataclass(frozen=True, slots=True)
class FrameId:
    """A frame kind plus the runtime qualifiers needed to prevent unsafe reuse."""

    kind: FrameKind
    qualifiers: tuple[str, ...]

    def __post_init__(self) -> None:
        qualifiers = tuple(self.qualifiers)
        expected_count = _QUALIFIER_COUNTS.get(self.kind)
        if expected_count is None:
            raise ValueError("unsupported frame kind")
        if len(qualifiers) != expected_count:
            raise ValueError(f"{self.kind} frames require exactly {expected_count} qualifiers")
        for qualifier in qualifiers:
            _validate_qualifier(qualifier)
        object.__setattr__(self, "qualifiers", qualifiers)

    @classmethod
    def global_frame(cls, *, log_namespace: str) -> Self:
        """Construct a log-qualified global frame."""

        return cls(kind="global", qualifiers=(log_namespace,))

    @classmethod
    def ego(cls, *, log_namespace: str, timestamp_qualifier: str) -> Self:
        """Construct an ego frame at one sensor timestamp."""

        return cls(kind="ego", qualifiers=(log_namespace, timestamp_qualifier))

    @classmethod
    def camera(
        cls,
        *,
        channel: str,
        calibration_instance: str,
        timestamp_qualifier: str,
    ) -> Self:
        """Construct a timestamped calibrated camera frame."""

        return cls(
            kind="camera",
            qualifiers=(channel, calibration_instance, timestamp_qualifier),
        )

    @classmethod
    def lidar(
        cls,
        *,
        channel: str,
        calibration_instance: str,
        timestamp_qualifier: str,
    ) -> Self:
        """Construct a timestamped calibrated LiDAR frame."""

        return cls(
            kind="lidar",
            qualifiers=(channel, calibration_instance, timestamp_qualifier),
        )

    @classmethod
    def parse(cls, value: str) -> Self:
        """Parse the frozen colon-separated synthetic-fixture representation."""

        parts = value.split(":")
        kind = parts[0]
        if kind not in _QUALIFIER_COUNTS:
            raise ValueError("unsupported frame kind")
        return cls(
            kind=cast(FrameKind, kind),
            qualifiers=tuple(parts[1:]),
        )

    def qualified_name(self) -> str:
        """Return the local-only qualified representation."""

        return ":".join((self.kind, *self.qualifiers))
