"""Persistent-scene procedural sequences and transient M4 event intent.

This module adapts the frozen M3 procedural populations without changing their
RNG or profile contracts.  Fault metadata is represented only by
``HealthFaultSpec`` and is deliberately absent from ``HealthBaseSequence`` so
that downstream observable-frame construction cannot accidentally expose it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from fusion_fault_bench.contracts.procedural_profile_v1 import (
    EdgeProceduralProfile,
    MainProceduralProfile,
    ProceduralProfileV1,
    SplitId,
)
from fusion_fault_bench.geometry._arrays import immutable_float64_copy
from fusion_fault_bench.geometry.roi import CommonRoi, procedural_center_eligible
from fusion_fault_bench.geometry.se3 import quaternion_wxyz_to_rotation
from fusion_fault_bench.scenarios.procedural import (
    ProceduralSequence,
    generate_procedural_sequences,
)

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]
type HealthScheduleId = Literal["standard", "cold_start"]
type HealthFaultFamily = Literal[
    "identity",
    "additive-position-bias",
    "increased-noise-underreported",
    "increased-noise-correctly-reported",
    "timestamp-offset",
    "dropout",
    "calibration-translation",
    "calibration-yaw",
    "common-mode-position-bias",
    "clean-predictor-mismatch",
]
type HealthFaultTarget = Literal["none", "camera", "lidar", "both"]
type HealthFaultAxis = Literal[
    "none",
    "x",
    "y",
    "xy",
    "time",
    "availability",
    "yaw",
    "motion",
]
type HealthFaultUnit = Literal[
    "identity",
    "m",
    "std-scale",
    "s",
    "probability",
    "rad",
    "m/s^2",
]
type HealthMotionModel = Literal[
    "profile-constant-velocity",
    "bounded-acceleration-control",
]

M4_HEALTH_INTENT_SHA256 = "c19573b72a6ef58ad5efdd7039110da79beacd3b398007a508eb7ecfc4c81357"
_HEALTH_ELIGIBILITY_DOMAIN = b"fusion-fault-bench/health-eligibility/v1\x00"
_STANDARD_FRAME_COUNT = 48
_FRAME_PERIOD_S = 0.1
_ACCELERATION_MPS2 = 8.0


def _immutable_int64(value: npt.ArrayLike) -> IntArray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    return np.frombuffer(array.tobytes(order="C"), dtype=np.int64).reshape(array.shape)


def _immutable_bool(value: npt.ArrayLike) -> BoolArray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.bool_))
    return np.frombuffer(array.tobytes(order="C"), dtype=np.bool_).reshape(array.shape)


def _finite_scalar(value: float, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _immutable_finite(
    value: npt.ArrayLike,
    *,
    shape: tuple[int, ...],
    field_name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return immutable_float64_copy(array)


def _field(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(4, "big") + encoded


def _eligibility_sha256(
    sequence_id: str,
    object_ids: tuple[str, ...],
    eligibility_mask: BoolArray,
) -> str:
    payload = bytearray(_HEALTH_ELIGIBILITY_DOMAIN)
    payload.extend(_field(sequence_id))
    frame_indices, object_indices = np.nonzero(eligibility_mask)
    payload.extend(len(frame_indices).to_bytes(8, "big"))
    for frame_index, object_index in zip(frame_indices, object_indices, strict=True):
        payload.extend(_field(f"{int(frame_index):06d}:{object_ids[int(object_index)]}"))
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class HealthEventSchedule:
    """One exact zero-based half-open M4 transient event schedule."""

    schedule_id: HealthScheduleId
    frame_count: int
    score_frames: tuple[int, int]
    fault_active_frames: tuple[int, int]
    recovery_frames: tuple[int, int]
    clean_prefix_frames: tuple[int, int] | None
    predictor_initialization_frames: tuple[int, int] | None

    def __post_init__(self) -> None:
        if self.schedule_id not in {"standard", "cold_start"}:
            raise ValueError("unknown M4 event schedule")
        if self.frame_count != _STANDARD_FRAME_COUNT:
            raise ValueError("M4 event schedules require exactly 48 frames")
        for field_name in (
            "score_frames",
            "fault_active_frames",
            "recovery_frames",
        ):
            start, end = getattr(self, field_name)
            if not 0 <= start < end <= self.frame_count:
                raise ValueError(f"{field_name} must be an ordered half-open interval")
        for field_name in (
            "clean_prefix_frames",
            "predictor_initialization_frames",
        ):
            interval = getattr(self, field_name)
            if interval is not None:
                start, end = interval
                if not 0 <= start < end <= self.frame_count:
                    raise ValueError(f"{field_name} must be an ordered half-open interval")

    def active_mask(self) -> BoolArray:
        """Return the immutable frame mask on which the fault operator is active."""

        start, end = self.fault_active_frames
        result = np.zeros(self.frame_count, dtype=np.bool_)
        result[start:end] = True
        return _immutable_bool(result)


_STANDARD_SCHEDULE = HealthEventSchedule(
    schedule_id="standard",
    frame_count=48,
    score_frames=(2, 48),
    fault_active_frames=(12, 36),
    recovery_frames=(36, 48),
    clean_prefix_frames=(0, 12),
    predictor_initialization_frames=(0, 2),
)
_COLD_START_SCHEDULE = HealthEventSchedule(
    schedule_id="cold_start",
    frame_count=48,
    score_frames=(0, 48),
    fault_active_frames=(0, 24),
    recovery_frames=(24, 48),
    clean_prefix_frames=None,
    predictor_initialization_frames=None,
)


def health_event_schedule(schedule_id: HealthScheduleId) -> HealthEventSchedule:
    """Return the immutable preregistered schedule selected by its exact ID."""

    if schedule_id == "standard":
        return _STANDARD_SCHEDULE
    if schedule_id == "cold_start":
        return _COLD_START_SCHEDULE
    raise ValueError("unknown M4 event schedule")


@dataclass(frozen=True, slots=True)
class HealthFaultSpec:
    """Generator-only fault coordinate, never an observable scorer input."""

    family: HealthFaultFamily
    target: HealthFaultTarget
    axis: HealthFaultAxis
    unit: HealthFaultUnit
    value: float
    schedule: HealthScheduleId = "standard"

    def __post_init__(self) -> None:
        value = _finite_scalar(self.value, field_name="value")
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise ValueError("value must use canonical positive zero")
        object.__setattr__(self, "value", value)
        health_event_schedule(self.schedule)

        coordinate = (self.target, self.axis, self.unit)
        expected: dict[
            HealthFaultFamily, tuple[HealthFaultTarget, HealthFaultAxis, HealthFaultUnit]
        ] = {
            "identity": ("none", "none", "identity"),
            "additive-position-bias": ("camera", "y", "m"),
            "increased-noise-underreported": ("camera", "xy", "std-scale"),
            "increased-noise-correctly-reported": ("camera", "xy", "std-scale"),
            "timestamp-offset": ("camera", "time", "s"),
            "dropout": ("camera", "availability", "probability"),
            "calibration-translation": ("camera", "x", "m"),
            "calibration-yaw": ("camera", "yaw", "rad"),
            "common-mode-position-bias": ("both", "x", "m"),
            "clean-predictor-mismatch": ("none", "motion", "m/s^2"),
        }
        if self.family not in expected:
            raise ValueError("unknown M4 fault family")
        if self.family in {
            "additive-position-bias",
            "increased-noise-underreported",
            "increased-noise-correctly-reported",
            "timestamp-offset",
            "dropout",
        }:
            _, axis, unit = expected[self.family]
            if self.target not in {"camera", "lidar"} or (self.axis, self.unit) != (
                axis,
                unit,
            ):
                raise ValueError(f"{self.family} has an invalid M4 coordinate")
        elif coordinate != expected[self.family]:
            raise ValueError(f"{self.family} has an invalid M4 coordinate")

        if self.family == "identity" and value != 0.0:
            raise ValueError("identity requires value zero")
        if self.family == "clean-predictor-mismatch" and value != _ACCELERATION_MPS2:
            raise ValueError("bounded-acceleration control requires exactly 8 m/s^2")
        if (
            self.family
            in {
                "increased-noise-underreported",
                "increased-noise-correctly-reported",
            }
            and value <= 1.0
        ):
            raise ValueError("noise scale must exceed identity scale one")
        if self.family == "dropout" and not 0.0 < value <= 1.0:
            raise ValueError("dropout probability must lie in (0, 1]")
        if (
            self.family
            not in {
                "identity",
                "clean-predictor-mismatch",
                "increased-noise-underreported",
                "increased-noise-correctly-reported",
                "dropout",
            }
            and value == 0.0
        ):
            raise ValueError("nonidentity signed faults require nonzero values")
        if self.schedule == "cold_start" and (
            self.family,
            self.target,
            self.axis,
            value,
        ) not in {
            ("calibration-translation", "camera", "x", 3.0),
            ("additive-position-bias", "lidar", "y", 3.0),
        }:
            raise ValueError("cold-start is frozen to its two exact M4 controls")


@dataclass(frozen=True, slots=True)
class HealthBaseSequence:
    """One immutable M4 base sequence in a persistent stationary scene frame."""

    sequence_id: str
    profile_id: Literal[
        "constant-velocity-front-roi-v1",
        "constant-velocity-fov-edge-v1",
    ]
    split: SplitId
    sequence_index: int
    object_ids: tuple[str, ...]
    frame_indices: IntArray
    reference_times_s: FloatArray
    truth_xy_m: FloatArray
    velocity_xy_mps: FloatArray
    eligibility_mask: BoolArray
    camera_standard_normal_xy: FloatArray
    lidar_standard_normal_xy: FloatArray
    dropout_uniform_by_frame: FloatArray
    camera_true_translation_m: FloatArray
    camera_true_rotation: FloatArray
    roi_x_min_m: float
    roi_x_max_m: float
    roi_abs_y_max_m: float
    camera_half_fov_rad: float
    eligibility_sha256: str
    motion_model: HealthMotionModel
    health_frame: Literal["persistent-scene-bev"] = "persistent-scene-bev"

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id must be nonempty")
        if self.profile_id not in {
            "constant-velocity-front-roi-v1",
            "constant-velocity-fov-edge-v1",
        }:
            raise ValueError("M4 supports only the main and edge procedural profiles")
        if self.motion_model not in {
            "profile-constant-velocity",
            "bounded-acceleration-control",
        }:
            raise ValueError("unknown M4 motion model")
        if self.health_frame != "persistent-scene-bev":
            raise ValueError("M4 health inputs require persistent scene BEV")
        frame_count = len(self.frame_indices)
        object_count = len(self.object_ids)
        if frame_count != _STANDARD_FRAME_COUNT or object_count == 0:
            raise ValueError("M4 base sequence has an invalid shape")
        if len(set(self.object_ids)) != object_count or self.object_ids != tuple(
            sorted(self.object_ids, key=lambda item: item.encode("utf-8"))
        ):
            raise ValueError("object IDs must be unique and ordered by UTF-8 bytes")
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be nonnegative")

        frame_indices = np.asarray(self.frame_indices)
        if frame_indices.shape != (frame_count,) or frame_indices.dtype.kind not in {"i", "u"}:
            raise ValueError("frame_indices must be an integer vector")
        canonical_indices = np.arange(frame_count, dtype=np.int64)
        if not np.array_equal(frame_indices, canonical_indices):
            raise ValueError("M4 frame indices must be zero-based and contiguous")
        object.__setattr__(self, "frame_indices", _immutable_int64(canonical_indices))

        object.__setattr__(
            self,
            "reference_times_s",
            _immutable_finite(
                self.reference_times_s,
                shape=(frame_count,),
                field_name="reference_times_s",
            ),
        )
        if not np.array_equal(
            self.reference_times_s,
            np.arange(frame_count, dtype=np.float64) * _FRAME_PERIOD_S,
        ):
            raise ValueError("M4 reference times must use exact 0.1 s frame periods")
        for field_name in (
            "truth_xy_m",
            "velocity_xy_mps",
            "camera_standard_normal_xy",
            "lidar_standard_normal_xy",
        ):
            object.__setattr__(
                self,
                field_name,
                _immutable_finite(
                    getattr(self, field_name),
                    shape=(frame_count, object_count, 2),
                    field_name=field_name,
                ),
            )

        eligibility = np.asarray(self.eligibility_mask, dtype=np.bool_)
        if eligibility.shape != (frame_count, object_count):
            raise ValueError("eligibility_mask has an invalid shape")
        if np.any(np.count_nonzero(eligibility, axis=1) == 0):
            raise ValueError("every M4 frame requires at least one eligible object")
        object.__setattr__(self, "eligibility_mask", _immutable_bool(eligibility))

        object.__setattr__(
            self,
            "dropout_uniform_by_frame",
            _immutable_finite(
                self.dropout_uniform_by_frame,
                shape=(frame_count,),
                field_name="dropout_uniform_by_frame",
            ),
        )
        if np.any(self.dropout_uniform_by_frame < 0.0) or np.any(
            self.dropout_uniform_by_frame >= 1.0
        ):
            raise ValueError("dropout uniforms must lie in [0, 1)")

        object.__setattr__(
            self,
            "camera_true_translation_m",
            _immutable_finite(
                self.camera_true_translation_m,
                shape=(3,),
                field_name="camera_true_translation_m",
            ),
        )
        rotation = _immutable_finite(
            self.camera_true_rotation,
            shape=(3, 3),
            field_name="camera_true_rotation",
        )
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12):
            raise ValueError("camera_true_rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("camera_true_rotation must be right-handed")
        object.__setattr__(self, "camera_true_rotation", rotation)

        for field_name in (
            "roi_x_min_m",
            "roi_x_max_m",
            "roi_abs_y_max_m",
            "camera_half_fov_rad",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_scalar(getattr(self, field_name), field_name=field_name),
            )
        if (
            self.roi_x_min_m < 0.0
            or self.roi_x_max_m <= self.roi_x_min_m
            or self.roi_abs_y_max_m <= 0.0
            or not 0.0 < self.camera_half_fov_rad < math.pi / 2.0
        ):
            raise ValueError("M4 ROI is invalid")
        expected_eligibility_sha256 = _eligibility_sha256(
            self.sequence_id,
            self.object_ids,
            self.eligibility_mask,
        )
        if self.eligibility_sha256 != expected_eligibility_sha256:
            raise ValueError("eligibility_sha256 does not match the ordered mask")

    @property
    def frame_count(self) -> int:
        return int(self.frame_indices.size)

    @property
    def object_count(self) -> int:
        return len(self.object_ids)

    @property
    def eligible_object_frame_count(self) -> int:
        return int(np.count_nonzero(self.eligibility_mask))


def _full_error_grid(
    sequence: ProceduralSequence,
    flat_values: FloatArray,
) -> FloatArray:
    result = np.zeros((sequence.frame_count, sequence.object_count, 2), dtype=np.float64)
    if flat_values.shape != (sequence.eligible_object_frame_count, 2):
        raise ValueError("procedural error rows disagree with eligibility")
    result[sequence.eligible_frame_indices, sequence.eligible_object_indices] = flat_values
    return immutable_float64_copy(result)


def adapt_procedural_sequence(
    sequence: ProceduralSequence,
    *,
    profile: ProceduralProfileV1,
) -> HealthBaseSequence:
    """Adapt one M3 procedural draw bundle into the stationary M4 scene frame."""

    if isinstance(profile, MainProceduralProfile):
        expected_profile_id = "constant-velocity-front-roi-v1"
    elif isinstance(profile, EdgeProceduralProfile):
        expected_profile_id = "constant-velocity-fov-edge-v1"
    else:
        raise ValueError("M4 excludes the procedural CI-smoke profile")
    if sequence.profile_id != expected_profile_id or profile.profile_id != expected_profile_id:
        raise ValueError("sequence and profile identities disagree")
    if profile.rig.ego_motion != "stationary":
        raise ValueError("procedural M4 requires a stationary scene frame")
    if sequence.frame_count != profile.source.frame_count:
        raise ValueError("sequence frame count disagrees with the profile")
    if sequence.object_count != profile.source.object_count:
        raise ValueError("sequence object count disagrees with the profile")

    velocity = np.broadcast_to(
        sequence.velocity_xy_mps[np.newaxis, :, :],
        (sequence.frame_count, sequence.object_count, 2),
    )
    extrinsic = profile.rig.camera_true_extrinsic
    eligibility = _immutable_bool(sequence.eligibility_mask)
    return HealthBaseSequence(
        sequence_id=sequence.sequence_id,
        profile_id=expected_profile_id,
        split=sequence.split,
        sequence_index=sequence.sequence_index,
        object_ids=sequence.object_ids,
        frame_indices=sequence.frame_indices,
        reference_times_s=sequence.frame_times_s,
        truth_xy_m=sequence.truth_xy_m,
        velocity_xy_mps=velocity,
        eligibility_mask=eligibility,
        camera_standard_normal_xy=_full_error_grid(
            sequence,
            sequence.camera_standard_normal_xy,
        ),
        lidar_standard_normal_xy=_full_error_grid(
            sequence,
            sequence.lidar_standard_normal_xy,
        ),
        dropout_uniform_by_frame=sequence.fault_uniform_by_frame,
        camera_true_translation_m=np.asarray(extrinsic.translation_m, dtype=np.float64),
        camera_true_rotation=quaternion_wxyz_to_rotation(extrinsic.quaternion_wxyz),
        roi_x_min_m=profile.eligibility.x_min_m,
        roi_x_max_m=profile.eligibility.x_max_m,
        roi_abs_y_max_m=profile.eligibility.abs_y_max_m,
        camera_half_fov_rad=profile.eligibility.camera_half_fov_rad,
        eligibility_sha256=_eligibility_sha256(
            sequence.sequence_id,
            sequence.object_ids,
            eligibility,
        ),
        motion_model="profile-constant-velocity",
    )


def generate_health_base_sequences(
    profile: ProceduralProfileV1,
    *,
    split: SplitId,
    sequence_count: int,
    data_master_seed: int,
) -> tuple[HealthBaseSequence, ...]:
    """Reuse the exact M3 latent/error/dropout streams for an M4 population."""

    sequences = generate_procedural_sequences(
        profile,
        split=split,
        sequence_count=sequence_count,
        data_master_seed=data_master_seed,
    )
    return tuple(adapt_procedural_sequence(item, profile=profile) for item in sequences)


def build_bounded_acceleration_control(base: HealthBaseSequence) -> HealthBaseSequence:
    """Apply the exact clean 8 m/s² test maneuver and recompute eligibility."""

    if (
        base.profile_id != "constant-velocity-front-roi-v1"
        or base.split != "test"
        or base.object_count != 6
        or base.frame_count != 48
    ):
        raise ValueError("bounded acceleration requires the frozen main test population")
    if base.motion_model != "profile-constant-velocity":
        raise ValueError("bounded acceleration cannot be applied recursively")
    if not np.all(base.eligibility_mask):
        raise ValueError("frozen maneuver requires full base draw support")

    positions = np.empty_like(base.truth_xy_m)
    velocities = np.empty_like(base.velocity_xy_mps)
    positions[0] = base.truth_xy_m[0]
    velocities[0] = base.velocity_xy_mps[0]
    side = np.asarray((-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), dtype=np.float64)
    for transition_index in range(47):
        acceleration = np.zeros((6, 2), dtype=np.float64)
        if 18 <= transition_index < 24:
            acceleration[:, 1] = side * _ACCELERATION_MPS2
        elif 24 <= transition_index < 30:
            acceleration[:, 1] = -side * _ACCELERATION_MPS2
        positions[transition_index + 1] = (
            positions[transition_index]
            + velocities[transition_index] * _FRAME_PERIOD_S
            + 0.5 * acceleration * (_FRAME_PERIOD_S**2)
        )
        velocities[transition_index + 1] = (
            velocities[transition_index] + acceleration * _FRAME_PERIOD_S
        )

    roi = CommonRoi(
        x_min_m=base.roi_x_min_m,
        x_max_m=base.roi_x_max_m,
        abs_y_max_m=base.roi_abs_y_max_m,
    )
    eligibility = np.zeros((base.frame_count, base.object_count), dtype=np.bool_)
    for frame_index in range(base.frame_count):
        for object_index in range(base.object_count):
            eligibility[frame_index, object_index] = procedural_center_eligible(
                center_ego_m=positions[frame_index, object_index],
                roi=roi,
                camera_half_fov_rad=base.camera_half_fov_rad,
                lidar_support_available=True,
            )
    immutable_eligibility = _immutable_bool(eligibility)
    return HealthBaseSequence(
        sequence_id=base.sequence_id,
        profile_id=base.profile_id,
        split=base.split,
        sequence_index=base.sequence_index,
        object_ids=base.object_ids,
        frame_indices=base.frame_indices,
        reference_times_s=base.reference_times_s,
        truth_xy_m=positions,
        velocity_xy_mps=velocities,
        eligibility_mask=immutable_eligibility,
        camera_standard_normal_xy=base.camera_standard_normal_xy,
        lidar_standard_normal_xy=base.lidar_standard_normal_xy,
        dropout_uniform_by_frame=base.dropout_uniform_by_frame,
        camera_true_translation_m=base.camera_true_translation_m,
        camera_true_rotation=base.camera_true_rotation,
        roi_x_min_m=base.roi_x_min_m,
        roi_x_max_m=base.roi_x_max_m,
        roi_abs_y_max_m=base.roi_abs_y_max_m,
        camera_half_fov_rad=base.camera_half_fov_rad,
        eligibility_sha256=_eligibility_sha256(
            base.sequence_id,
            base.object_ids,
            immutable_eligibility,
        ),
        motion_model="bounded-acceleration-control",
    )
