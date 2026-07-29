"""Independent scalar and affine-Gaussian references for M3 validation.

This module intentionally does not import the production scenario, RNG, fault,
fusion, or evaluation implementations. It duplicates the frozen equations so a
shared implementation mistake cannot satisfy its own release gate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type ProfileId = Literal[
    "constant-velocity-front-roi-v1",
    "constant-velocity-fov-edge-v1",
    "constant-velocity-ci-smoke-v1",
]
type SplitId = Literal["train", "validation", "test"]

_RNG_DOMAIN = b"fusion-fault-bench/rng/v1"
_UINT32_LIMIT = 2**32
_UINT128_LIMIT = 2**128


@dataclass(frozen=True, slots=True)
class ReferenceLatentState:
    """Independent initial positions and velocities in canonical object order."""

    initial_xy_m: FloatArray
    velocity_xy_mps: FloatArray


@dataclass(frozen=True, slots=True)
class AffineLossMoments:
    """Expectation and variance of one Gaussian squared loss or contrast."""

    expected_m2: float
    variance_m4: float


@dataclass(frozen=True, slots=True)
class PopulationLossMoments:
    """Equal-sequence population moments from independent object-frame terms."""

    expected_m2: float
    standard_error_m2: float
    sequence_count: int
    object_frame_count: int


def _immutable_float64(value: npt.ArrayLike) -> FloatArray:
    source = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError("reference array must contain only finite values")
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(source.shape)


def _immutable_bool(value: npt.ArrayLike) -> BoolArray:
    source = np.asarray(value, dtype=np.bool_)
    return np.frombuffer(source.tobytes(order="C"), dtype=np.bool_).reshape(source.shape)


def _uniform_rows(uniforms: npt.ArrayLike, *, object_count: int) -> FloatArray:
    rows = np.asarray(uniforms, dtype=np.float64)
    if rows.shape != (object_count, 4):
        raise ValueError(f"uniforms must have shape ({object_count}, 4)")
    if not np.all(np.isfinite(rows)) or np.any(rows < 0.0) or np.any(rows >= 1.0):
        raise ValueError("uniforms must be finite values in [0, 1)")
    return rows


def reference_latent_state(
    profile_id: ProfileId,
    split: SplitId,
    uniforms: npt.ArrayLike,
) -> ReferenceLatentState:
    """Apply the preregistered profile mapping without production profile code."""

    if profile_id == "constant-velocity-front-roi-v1":
        rows = _uniform_rows(uniforms, object_count=6)
        object_indices = np.arange(6, dtype=np.int64)
        if split == "train":
            initial_x = 10.0 + 18.0 * rows[:, 0]
            lane = np.where(object_indices % 2 == 0, -3.5, 3.5)
            initial_y = lane + 0.25 * (2.0 * rows[:, 1] - 1.0)
            velocity_x = -1.0 + 2.0 * rows[:, 2]
            velocity_y = 0.1 * (2.0 * rows[:, 3] - 1.0)
        elif split == "validation":
            side = np.where(object_indices % 2 == 0, -1.0, 1.0)
            initial_x = 30.0 + 10.0 * rows[:, 0]
            initial_y = side * (5.0 + 3.0 * rows[:, 1])
            velocity_x = -1.0 + 2.0 * rows[:, 2]
            velocity_y = -side * (1.5 + 1.5 * rows[:, 3])
        elif split == "test":
            lateral_centers = np.asarray([-7.0, -4.0, -1.0, 1.0, 4.0, 7.0])
            initial_x = 44.0 + 12.0 * rows[:, 0]
            initial_y = lateral_centers + 0.25 * (2.0 * rows[:, 1] - 1.0)
            velocity_x = -(3.0 + 2.0 * rows[:, 2])
            velocity_y = 0.2 * (2.0 * rows[:, 3] - 1.0)
        else:
            raise ValueError("unknown split")
    elif profile_id == "constant-velocity-fov-edge-v1":
        if split != "test":
            raise ValueError("the FOV-edge profile defines only the test split")
        rows = _uniform_rows(uniforms, object_count=4)
        object_indices = np.arange(4, dtype=np.int64)
        side = np.where(object_indices % 2 == 0, 1.0, -1.0)
        radius = 20.0 + 20.0 * rows[:, 0]
        bearing = side * (0.7 - (0.005 + 0.015 * rows[:, 1]))
        cosine = np.cos(bearing)
        sine = np.sin(bearing)
        initial_x = radius * cosine
        initial_y = radius * sine
        radial_speed = -0.5 + rows[:, 2]
        velocity_x = radial_speed * cosine
        velocity_y = radial_speed * sine
    elif profile_id == "constant-velocity-ci-smoke-v1":
        if split != "test":
            raise ValueError("the CI smoke profile defines only the test split")
        rows = _uniform_rows(uniforms, object_count=3)
        lateral_centers = np.asarray([-2.0, 0.0, 2.0])
        initial_x = 10.0 + 10.0 * rows[:, 0]
        initial_y = lateral_centers + 0.1 * (2.0 * rows[:, 1] - 1.0)
        velocity_x = -0.5 + rows[:, 2]
        velocity_y = 0.05 * (2.0 * rows[:, 3] - 1.0)
    else:
        raise ValueError("unknown profile_id")

    return ReferenceLatentState(
        initial_xy_m=_immutable_float64(np.column_stack((initial_x, initial_y))),
        velocity_xy_mps=_immutable_float64(np.column_stack((velocity_x, velocity_y))),
    )


def reference_truth(
    state: ReferenceLatentState,
    *,
    frame_count: int,
    frame_period_s: float,
) -> FloatArray:
    """Return truth in canonical ``(frame, object, xy)`` order."""

    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if not math.isfinite(frame_period_s) or frame_period_s <= 0.0:
        raise ValueError("frame_period_s must be finite and positive")
    if (
        state.initial_xy_m.ndim != 2
        or state.initial_xy_m.shape[1] != 2
        or state.velocity_xy_mps.shape != state.initial_xy_m.shape
    ):
        raise ValueError("latent state arrays must have aligned (object, 2) shape")
    times = np.arange(frame_count, dtype=np.float64) * frame_period_s
    truth = (
        state.initial_xy_m[np.newaxis, :, :]
        + times[:, np.newaxis, np.newaxis] * state.velocity_xy_mps[np.newaxis, :, :]
    )
    return _immutable_float64(truth)


def reference_eligibility_mask(
    truth_xy_m: npt.ArrayLike,
    *,
    x_min_m: float,
    x_max_m: float,
    abs_y_max_m: float,
    camera_half_fov_rad: float,
) -> BoolArray:
    """Apply the frozen inclusive pre-fault common-ROI predicate."""

    truth = np.asarray(truth_xy_m, dtype=np.float64)
    if truth.ndim != 3 or truth.shape[2] != 2 or not np.all(np.isfinite(truth)):
        raise ValueError("truth_xy_m must be a finite (frame, object, 2) array")
    limits = (x_min_m, x_max_m, abs_y_max_m, camera_half_fov_rad)
    if not all(math.isfinite(value) for value in limits):
        raise ValueError("eligibility limits must be finite")
    if x_min_m < 0.0 or x_max_m <= x_min_m or abs_y_max_m <= 0.0:
        raise ValueError("eligibility spatial limits are invalid")
    if not 0.0 < camera_half_fov_rad < math.pi / 2.0:
        raise ValueError("camera_half_fov_rad must be in (0, pi/2)")
    x = truth[:, :, 0]
    y = truth[:, :, 1]
    bearing = np.arctan2(y, x)
    mask = (
        (x > 0.0)
        & (x >= x_min_m)
        & (x <= x_max_m)
        & (np.abs(y) <= abs_y_max_m)
        & (np.abs(bearing) <= camera_half_fov_rad)
    )
    return _immutable_bool(mask)


def _utf8_field(value: str, *, field_name: str) -> bytes:
    if not value:
        raise ValueError(f"{field_name} must be nonempty")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    if len(encoded) >= _UINT32_LIMIT:
        raise ValueError(f"{field_name} is too long")
    return encoded


def independent_fault_stream_seed(*, data_master_seed: int, sequence_id: str) -> int:
    """Reimplement the frozen framed seed derivation without production helpers."""

    if type(data_master_seed) is not int or not 0 <= data_master_seed < _UINT128_LIMIT:
        raise ValueError("data_master_seed must be an unsigned 128-bit integer")
    stream = _utf8_field("fault", field_name="stream_name")
    sequence = _utf8_field(sequence_id, field_name="sequence_id")
    payload = b"".join(
        (
            _RNG_DOMAIN,
            b"\x00",
            data_master_seed.to_bytes(16, byteorder="big", signed=False),
            len(stream).to_bytes(4, byteorder="big", signed=False),
            stream,
            len(sequence).to_bytes(4, byteorder="big", signed=False),
            sequence,
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], byteorder="big")


def independent_fault_uniforms(
    *,
    data_master_seed: int,
    sequence_id: str,
    frame_count: int,
) -> FloatArray:
    """Draw exactly one independent-reference float64 uniform per frame."""

    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    seed = independent_fault_stream_seed(
        data_master_seed=data_master_seed,
        sequence_id=sequence_id,
    )
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    return _immutable_float64(generator.random(frame_count, dtype=np.float64))


def independent_dropout_mask(uniforms: npt.ArrayLike, probability: float) -> BoolArray:
    """Return the frozen frame mask, where true means the target is dropped."""

    values = np.asarray(uniforms, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("uniforms must be a finite vector")
    if np.any(values < 0.0) or np.any(values >= 1.0):
        raise ValueError("uniforms must lie in [0, 1)")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return _immutable_bool(values < probability)


def _affine_inputs(matrix: npt.ArrayLike, bias: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    transform = np.asarray(matrix, dtype=np.float64)
    offset = np.asarray(bias, dtype=np.float64)
    if transform.ndim != 2 or transform.shape[0] != 2:
        raise ValueError("matrix must have shape (2, latent_dimension)")
    if offset.shape != (2,):
        raise ValueError("bias must have shape (2,)")
    if transform.shape[1] == 0 or not np.all(np.isfinite(transform)):
        raise ValueError("matrix must have a positive finite latent dimension")
    if not np.all(np.isfinite(offset)):
        raise ValueError("bias must be finite")
    return transform, offset


def affine_squared_loss_moments(
    matrix: npt.ArrayLike,
    bias: npt.ArrayLike,
) -> AffineLossMoments:
    """Return exact moments of ``||A u + b||^2`` for standard-normal ``u``."""

    transform, offset = _affine_inputs(matrix, bias)
    quadratic = transform.T @ transform
    expected = float(np.trace(quadratic) + offset @ offset)
    variance = float(
        2.0 * np.trace(quadratic @ quadratic) + 4.0 * offset @ transform @ transform.T @ offset
    )
    return AffineLossMoments(expected_m2=expected, variance_m4=variance)


def affine_signed_contrast_moments(
    first_matrix: npt.ArrayLike,
    first_bias: npt.ArrayLike,
    second_matrix: npt.ArrayLike,
    second_bias: npt.ArrayLike,
) -> AffineLossMoments:
    """Return exact moments of one paired squared-loss difference."""

    first_transform, first_offset = _affine_inputs(first_matrix, first_bias)
    second_transform, second_offset = _affine_inputs(second_matrix, second_bias)
    if first_transform.shape != second_transform.shape:
        raise ValueError("contrast matrices must have the same shape")
    quadratic = first_transform.T @ first_transform - second_transform.T @ second_transform
    linear = first_transform.T @ first_offset - second_transform.T @ second_offset
    constant = float(first_offset @ first_offset - second_offset @ second_offset)
    expected = float(np.trace(quadratic) + constant)
    variance = float(2.0 * np.trace(quadratic @ quadratic) + 4.0 * linear @ linear)
    return AffineLossMoments(expected_m2=expected, variance_m4=variance)


def equal_sequence_population_moments(
    rows_by_sequence: Sequence[Sequence[AffineLossMoments]],
) -> PopulationLossMoments:
    """Combine independent row moments using object-frame then sequence means."""

    if not rows_by_sequence:
        raise ValueError("at least one sequence is required")
    population_expected = 0.0
    population_variance = 0.0
    object_frame_count = 0
    for rows in rows_by_sequence:
        if not rows:
            raise ValueError("every sequence must contain an eligible row")
        count = len(rows)
        object_frame_count += count
        population_expected += math.fsum(row.expected_m2 for row in rows) / count
        population_variance += math.fsum(row.variance_m4 for row in rows) / (count * count)
    sequence_count = len(rows_by_sequence)
    population_expected /= sequence_count
    population_variance /= sequence_count * sequence_count
    return PopulationLossMoments(
        expected_m2=population_expected,
        standard_error_m2=math.sqrt(population_variance),
        sequence_count=sequence_count,
        object_frame_count=object_frame_count,
    )


def mean_six_se_bound(*, standard_deviation: float, sample_count: int) -> float:
    """Return the preregistered inclusive absolute sample-mean bound."""

    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be finite and positive")
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return 6.0 * standard_deviation / math.sqrt(sample_count)


def variance_six_se_bound(*, variance: float, sample_count: int) -> float:
    """Return the preregistered `ddof=1` sample-variance bound."""

    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("variance must be finite and positive")
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be at least two")
    return 6.0 * variance * math.sqrt(2.0 / (sample_count - 1))


def covariance_six_se_bound(
    *,
    first_standard_deviation: float,
    second_standard_deviation: float,
    sample_count: int,
) -> float:
    """Return the preregistered `ddof=1` zero-covariance bound."""

    values = (first_standard_deviation, second_standard_deviation)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("standard deviations must be finite and positive")
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be at least two")
    return 6.0 * first_standard_deviation * second_standard_deviation / math.sqrt(sample_count - 1)


def yaw_displacement_xy(point_xy_m: Sequence[float], yaw_rad: float) -> tuple[float, float]:
    """Return independent ``(R(yaw)-I)p`` scalar arithmetic."""

    if len(point_xy_m) != 2:
        raise ValueError("point_xy_m must contain two coordinates")
    x_m, y_m = (float(value) for value in point_xy_m)
    if not all(math.isfinite(value) for value in (x_m, y_m, yaw_rad)):
        raise ValueError("yaw oracle inputs must be finite")
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        cosine * x_m - sine * y_m - x_m,
        sine * x_m + cosine * y_m - y_m,
    )


def timestamp_displacement_xy(
    velocity_xy_mps: Sequence[float],
    offset_s: float,
) -> tuple[float, float]:
    """Return the frozen oracle-alignment error ``-v * offset``."""

    if len(velocity_xy_mps) != 2:
        raise ValueError("velocity_xy_mps must contain two coordinates")
    velocity_x, velocity_y = (float(value) for value in velocity_xy_mps)
    if not all(math.isfinite(value) for value in (velocity_x, velocity_y, offset_s)):
        raise ValueError("timestamp oracle inputs must be finite")
    return -velocity_x * offset_s, -velocity_y * offset_s
