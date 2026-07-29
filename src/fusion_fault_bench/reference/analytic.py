"""Independent scalar closed forms for the M1 diagonal-Gaussian experiments.

This module deliberately depends only on the immutable manifest contract and the
standard library. It must not share RNG, fusion, fault, experiment, evaluation,
or inference implementation with the production path it validates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AdditivePositionBiasFault,
    AnalyticCrossoverManifest,
    CorrectlyReportedNoiseFault,
)

type AnalyticFaultFamily = Literal[
    "additive-position-bias",
    "increased-noise-correctly-reported",
    "increased-noise-underreported",
]
type AnalyticFaultAxis = Literal["x", "y", "xy"]
type AnalyticSeverityDirection = Literal["identity", "negative", "positive", "increase"]
type AnalyticCrossoverDirection = Literal["negative", "positive", "increase"]
type AnalyticSeverityUnit = Literal["m", "std-scale"]
type GaussianMethodId = Literal["camera-only", "lidar-only", "fixed-fusion"]
type GridStatus = Literal["crossed", "not-crossed"]
type GridCensoring = Literal["none", "right-above-tested-maximum"]
type ContinuousStatus = Literal["finite", "no-finite-root"]
type XY = tuple[float, float]

ANALYTIC_GAUSSIAN_METHODS: tuple[GaussianMethodId, ...] = (
    "camera-only",
    "lidar-only",
    "fixed-fusion",
)


@dataclass(frozen=True, slots=True)
class AnalyticCondition:
    """One manifest fault coordinate in deterministic evaluation order."""

    fault_family: AnalyticFaultFamily
    fault_axis: AnalyticFaultAxis
    index: int
    magnitude: float
    direction: AnalyticSeverityDirection
    unit: AnalyticSeverityUnit


@dataclass(frozen=True, slots=True)
class GaussianPopulation:
    """Closed-form error distribution and squared-error moments for one method."""

    method_id: GaussianMethodId
    mean_xy_m: XY
    actual_variance_xy_m2: XY
    reported_variance_xy_m2: XY
    mse_m2: float
    mse_variance_m4: float

    def mse_standard_error_m2(self, sequence_count: int) -> float:
        """Return the standard error of the sequence mean MSE."""

        if sequence_count <= 0:
            raise ValueError("sequence_count must be positive")
        return math.sqrt(self.mse_variance_m4 / sequence_count)


@dataclass(frozen=True, slots=True)
class AnalyticPopulationPoint:
    """A condition paired with one preregistered Gaussian method population."""

    condition: AnalyticCondition
    population: GaussianPopulation


@dataclass(frozen=True, slots=True)
class CrossoverReference:
    """Population grid and continuous crossover truth for one direction."""

    direction: AnalyticCrossoverDirection
    severity_unit: AnalyticSeverityUnit
    tested_maximum: float
    grid_status: GridStatus
    grid_point_estimate: float | None
    grid_censoring: GridCensoring
    continuous_status: ContinuousStatus
    continuous_point_estimate: float | None


@dataclass(frozen=True, slots=True)
class _SensorPopulation:
    mean_xy_m: XY
    actual_variance_xy_m2: XY
    reported_variance_xy_m2: XY


def _fault_grid(manifest: AnalyticCrossoverManifest) -> tuple[float, ...]:
    fault = manifest.fault_sweep
    if isinstance(fault, AdditivePositionBiasFault):
        return fault.magnitude_values_m
    return fault.std_scale_values


def _crossover_directions(
    manifest: AnalyticCrossoverManifest,
) -> tuple[AnalyticCrossoverDirection, ...]:
    if isinstance(manifest.fault_sweep, AdditivePositionBiasFault):
        return ("negative", "positive")
    return ("increase",)


def expand_conditions(manifest: AnalyticCrossoverManifest) -> tuple[AnalyticCondition, ...]:
    """Expand identity and nonidentity coordinates without production fault code."""

    fault = manifest.fault_sweep
    grid = _fault_grid(manifest)
    fault_family: AnalyticFaultFamily = fault.kind
    fault_axis: AnalyticFaultAxis = fault.axis
    unit: AnalyticSeverityUnit = fault.unit
    conditions = [
        AnalyticCondition(
            fault_family=fault_family,
            fault_axis=fault_axis,
            index=0,
            magnitude=grid[0],
            direction="identity",
            unit=unit,
        )
    ]
    for index, magnitude in enumerate(grid[1:], start=1):
        conditions.extend(
            AnalyticCondition(
                fault_family=fault_family,
                fault_axis=fault_axis,
                index=index,
                magnitude=magnitude,
                direction=direction,
                unit=unit,
            )
            for direction in _crossover_directions(manifest)
        )
    return tuple(conditions)


def _validate_condition(
    manifest: AnalyticCrossoverManifest,
    condition: AnalyticCondition,
) -> None:
    fault = manifest.fault_sweep
    grid = _fault_grid(manifest)
    if (
        condition.fault_family != fault.kind
        or condition.fault_axis != fault.axis
        or condition.unit != fault.unit
    ):
        raise ValueError("condition fault coordinate does not match the manifest")
    if condition.index < 0 or condition.index >= len(grid):
        raise ValueError("condition index is outside the manifest grid")
    if condition.magnitude != grid[condition.index]:
        raise ValueError("condition magnitude does not match its manifest grid index")
    expected = ("identity",) if condition.index == 0 else _crossover_directions(manifest)
    if condition.direction not in expected:
        raise ValueError("condition direction is invalid for its manifest grid index")


def _base_sensor_population(
    manifest: AnalyticCrossoverManifest,
    modality: Literal["camera", "lidar"],
) -> _SensorPopulation:
    sensor = getattr(manifest.observations, modality)
    return _SensorPopulation(
        mean_xy_m=sensor.true_error_mean_xy_m,
        actual_variance_xy_m2=(
            sensor.actual_std_xy_m[0] ** 2,
            sensor.actual_std_xy_m[1] ** 2,
        ),
        reported_variance_xy_m2=(
            sensor.reported_std_xy_m[0] ** 2,
            sensor.reported_std_xy_m[1] ** 2,
        ),
    )


def _replace_target(
    *,
    target: Literal["camera", "lidar"],
    replacement: _SensorPopulation,
    camera: _SensorPopulation,
    lidar: _SensorPopulation,
) -> tuple[_SensorPopulation, _SensorPopulation]:
    if target == "camera":
        return replacement, lidar
    return camera, replacement


def _faulted_sensor_populations(
    manifest: AnalyticCrossoverManifest,
    condition: AnalyticCondition,
) -> tuple[_SensorPopulation, _SensorPopulation]:
    _validate_condition(manifest, condition)
    camera = _base_sensor_population(manifest, "camera")
    lidar = _base_sensor_population(manifest, "lidar")
    fault = manifest.fault_sweep
    target = camera if fault.target == "camera" else lidar

    if isinstance(fault, AdditivePositionBiasFault):
        mean = list(target.mean_xy_m)
        if condition.direction != "identity":
            axis_index = 0 if fault.axis == "x" else 1
            sign = -1.0 if condition.direction == "negative" else 1.0
            mean[axis_index] = sign * condition.magnitude
        replacement = _SensorPopulation(
            mean_xy_m=(mean[0], mean[1]),
            actual_variance_xy_m2=target.actual_variance_xy_m2,
            reported_variance_xy_m2=target.reported_variance_xy_m2,
        )
        return _replace_target(
            target=fault.target,
            replacement=replacement,
            camera=camera,
            lidar=lidar,
        )

    scale_squared = condition.magnitude**2
    actual = (
        target.actual_variance_xy_m2[0] * scale_squared,
        target.actual_variance_xy_m2[1] * scale_squared,
    )
    reported = target.reported_variance_xy_m2
    if isinstance(fault, CorrectlyReportedNoiseFault):
        reported = (
            reported[0] * scale_squared,
            reported[1] * scale_squared,
        )
    replacement = _SensorPopulation(
        mean_xy_m=target.mean_xy_m,
        actual_variance_xy_m2=actual,
        reported_variance_xy_m2=reported,
    )
    return _replace_target(
        target=fault.target,
        replacement=replacement,
        camera=camera,
        lidar=lidar,
    )


def _fixed_fusion_population(
    camera: _SensorPopulation,
    lidar: _SensorPopulation,
) -> _SensorPopulation:
    camera_weights = tuple(
        lidar.reported_variance_xy_m2[axis]
        / (camera.reported_variance_xy_m2[axis] + lidar.reported_variance_xy_m2[axis])
        for axis in range(2)
    )
    lidar_weights = (1.0 - camera_weights[0], 1.0 - camera_weights[1])
    mean = tuple(
        camera_weights[axis] * camera.mean_xy_m[axis] + lidar_weights[axis] * lidar.mean_xy_m[axis]
        for axis in range(2)
    )
    actual = tuple(
        camera_weights[axis] ** 2 * camera.actual_variance_xy_m2[axis]
        + lidar_weights[axis] ** 2 * lidar.actual_variance_xy_m2[axis]
        for axis in range(2)
    )
    reported = tuple(
        (camera.reported_variance_xy_m2[axis] * lidar.reported_variance_xy_m2[axis])
        / (camera.reported_variance_xy_m2[axis] + lidar.reported_variance_xy_m2[axis])
        for axis in range(2)
    )
    return _SensorPopulation(
        mean_xy_m=(mean[0], mean[1]),
        actual_variance_xy_m2=(actual[0], actual[1]),
        reported_variance_xy_m2=(reported[0], reported[1]),
    )


def _squared_error_moments(population: _SensorPopulation) -> tuple[float, float]:
    mse = sum(
        population.actual_variance_xy_m2[axis] + population.mean_xy_m[axis] ** 2
        for axis in range(2)
    )
    variance = sum(
        2.0 * population.actual_variance_xy_m2[axis] ** 2
        + (4.0 * population.mean_xy_m[axis] ** 2 * population.actual_variance_xy_m2[axis])
        for axis in range(2)
    )
    return mse, variance


def gaussian_population(
    manifest: AnalyticCrossoverManifest,
    condition: AnalyticCondition,
    method_id: GaussianMethodId,
) -> GaussianPopulation:
    """Return independent diagonal-Gaussian truth for one supported method."""

    if method_id not in ANALYTIC_GAUSSIAN_METHODS:
        raise ValueError(f"method {method_id!r} is not a Gaussian analytic method")
    camera, lidar = _faulted_sensor_populations(manifest, condition)
    if method_id == "camera-only":
        method_population = camera
    elif method_id == "lidar-only":
        method_population = lidar
    else:
        method_population = _fixed_fusion_population(camera, lidar)
    mse, variance = _squared_error_moments(method_population)
    return GaussianPopulation(
        method_id=method_id,
        mean_xy_m=method_population.mean_xy_m,
        actual_variance_xy_m2=method_population.actual_variance_xy_m2,
        reported_variance_xy_m2=method_population.reported_variance_xy_m2,
        mse_m2=mse,
        mse_variance_m4=variance,
    )


def population_points(
    manifest: AnalyticCrossoverManifest,
) -> tuple[AnalyticPopulationPoint, ...]:
    """Return condition-major, preregistered-method-major population truth."""

    return tuple(
        AnalyticPopulationPoint(
            condition=condition,
            population=gaussian_population(manifest, condition, method_id),
        )
        for condition in expand_conditions(manifest)
        for method_id in ANALYTIC_GAUSSIAN_METHODS
    )


def population_contrast(
    manifest: AnalyticCrossoverManifest,
    condition: AnalyticCondition,
) -> float:
    """Return population fixed-fusion MSE minus the healthy modality MSE."""

    healthy_method: GaussianMethodId = (
        "lidar-only" if manifest.fault_sweep.target == "camera" else "camera-only"
    )
    fused = gaussian_population(manifest, condition, "fixed-fusion")
    healthy = gaussian_population(manifest, condition, healthy_method)
    return fused.mse_m2 - healthy.mse_m2


def pava_non_decreasing(values: Sequence[float]) -> tuple[float, ...]:
    """Independent equal-weight nondecreasing pool-adjacent-violators fit."""

    source = tuple(float(value) for value in values)
    if not source:
        raise ValueError("PAVA requires a nonempty vector")
    if not all(math.isfinite(value) for value in source):
        raise ValueError("PAVA values must be finite")

    means: list[float] = []
    weights: list[int] = []
    for value in source:
        means.append(value)
        weights.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            right_mean = means.pop()
            left_mean = means.pop()
            right_weight = weights.pop()
            left_weight = weights.pop()
            pooled_weight = left_weight + right_weight
            means.append((left_mean * left_weight + right_mean * right_weight) / pooled_weight)
            weights.append(pooled_weight)

    fitted: list[float] = []
    for mean, weight in zip(means, weights, strict=True):
        fitted.extend(mean for _ in range(weight))
    return tuple(fitted)


def first_zero_grid_root(
    magnitudes: Sequence[float],
    fitted_values: Sequence[float],
    *,
    zero_tolerance: float,
) -> float | None:
    """Interpolate the first zero of a validated nondecreasing fitted curve."""

    grid = tuple(float(value) for value in magnitudes)
    fitted = tuple(float(value) for value in fitted_values)
    if len(grid) < 2 or len(grid) != len(fitted):
        raise ValueError("crossover grid and fitted values must be aligned vectors")
    if not all(math.isfinite(value) for value in (*grid, *fitted)):
        raise ValueError("crossover inputs must be finite")
    if any(right <= left for left, right in pairwise(grid)):
        raise ValueError("crossover magnitudes must be strictly increasing")
    if any(right < left for left, right in pairwise(fitted)):
        raise ValueError("crossover fitted values must be nondecreasing")
    if not math.isfinite(zero_tolerance) or zero_tolerance <= 0.0:
        raise ValueError("zero_tolerance must be finite and positive")

    adjusted = tuple(0.0 if abs(value) <= zero_tolerance else value for value in fitted)
    if adjusted[0] >= 0.0:
        return grid[0]
    for index in range(1, len(grid)):
        if adjusted[index] < 0.0:
            continue
        if adjusted[index] == 0.0:
            return grid[index]
        fraction = -adjusted[index - 1] / (adjusted[index] - adjusted[index - 1])
        return grid[index - 1] + fraction * (grid[index] - grid[index - 1])
    return None


def grid_crossover_root(
    magnitudes: Sequence[float],
    contrasts: Sequence[float],
    *,
    zero_tolerance: float,
) -> float | None:
    """Fit independent PAVA and return its first linearly interpolated zero."""

    return first_zero_grid_root(
        magnitudes,
        pava_non_decreasing(contrasts),
        zero_tolerance=zero_tolerance,
    )


def _identity_contrast(manifest: AnalyticCrossoverManifest) -> float:
    return population_contrast(manifest, expand_conditions(manifest)[0])


def _nominal_target_fusion_weights(manifest: AnalyticCrossoverManifest) -> XY:
    camera = _base_sensor_population(manifest, "camera")
    lidar = _base_sensor_population(manifest, "lidar")
    camera_weights = tuple(
        lidar.reported_variance_xy_m2[axis]
        / (camera.reported_variance_xy_m2[axis] + lidar.reported_variance_xy_m2[axis])
        for axis in range(2)
    )
    if manifest.fault_sweep.target == "camera":
        return camera_weights[0], camera_weights[1]
    return 1.0 - camera_weights[0], 1.0 - camera_weights[1]


def continuous_crossover_root(manifest: AnalyticCrossoverManifest) -> float | None:
    """Return the continuous first-crossing reference, or no finite root."""

    fault = manifest.fault_sweep
    identity_contrast = _identity_contrast(manifest)
    target_weights = _nominal_target_fusion_weights(manifest)
    if isinstance(fault, AdditivePositionBiasFault):
        axis = 0 if fault.axis == "x" else 1
        if identity_contrast >= 0.0:
            return 0.0
        return math.sqrt(-identity_contrast / target_weights[axis] ** 2)
    if isinstance(fault, CorrectlyReportedNoiseFault):
        return None

    target = _base_sensor_population(manifest, fault.target)
    scale_coefficient = sum(
        target_weights[axis] ** 2 * target.actual_variance_xy_m2[axis] for axis in range(2)
    )
    if identity_contrast >= 0.0:
        return 1.0
    return math.sqrt(1.0 - identity_contrast / scale_coefficient)


def population_crossover_references(
    manifest: AnalyticCrossoverManifest,
) -> tuple[CrossoverReference, ...]:
    """Return grid/PAVA and distinct continuous population references."""

    all_conditions = expand_conditions(manifest)
    identity = all_conditions[0]
    continuous_root = continuous_crossover_root(manifest)
    references: list[CrossoverReference] = []
    for direction in _crossover_directions(manifest):
        conditions = (
            identity,
            *(condition for condition in all_conditions[1:] if condition.direction == direction),
        )
        magnitudes = tuple(condition.magnitude for condition in conditions)
        contrasts = tuple(population_contrast(manifest, condition) for condition in conditions)
        grid_root = grid_crossover_root(
            magnitudes,
            contrasts,
            zero_tolerance=manifest.evaluation.crossover.zero_tolerance_m2,
        )
        references.append(
            CrossoverReference(
                direction=direction,
                severity_unit=identity.unit,
                tested_maximum=magnitudes[-1],
                grid_status="crossed" if grid_root is not None else "not-crossed",
                grid_point_estimate=grid_root,
                grid_censoring=("none" if grid_root is not None else "right-above-tested-maximum"),
                continuous_status=("finite" if continuous_root is not None else "no-finite-root"),
                continuous_point_estimate=continuous_root,
            )
        )
    return tuple(references)
