"""Aggregate-only M5 replay descriptors and the frozen M3 comparator."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from fusion_fault_bench.contracts.procedural_profile_v1 import MainProceduralProfile
from fusion_fault_bench.replay_plan import LoadedReplayPlan
from fusion_fault_bench.replay_source import (
    DescriptorQuantiles,
    SceneDescriptorPrimitives,
    descriptor_quantiles,
)
from fusion_fault_bench.scenarios.procedural import generate_procedural_sequences

type DescriptorPopulation = Literal["nuscenes-mini-replay", "m3-main-test-comparator"]
type DescriptorStatistic = Literal[
    "count",
    "fraction",
    "minimum",
    "median",
    "maximum",
    "q0",
    "q25",
    "q50",
    "q75",
    "q100",
    "not-modeled",
]
type DescriptorStatus = Literal["ok", "not-applicable"]
type DescriptorUnit = Literal[
    "count",
    "fraction",
    "frames",
    "s",
    "m",
    "rad",
    "m/s",
    "m/s^2",
    "unitless",
]

_ACROSS_STATISTICS: tuple[Literal["minimum", "median", "maximum"], ...] = (
    "minimum",
    "median",
    "maximum",
)
_QUANTILE_FIELDS: tuple[
    tuple[Literal["q0", "q25", "q50", "q75", "q100"], str],
    ...,
] = (
    ("q0", "minimum"),
    ("q25", "lower_quartile"),
    ("q50", "median"),
    ("q75", "upper_quartile"),
    ("q100", "maximum"),
)


@dataclass(frozen=True, slots=True)
class ReplayDescriptorAggregate:
    """One privacy-safe descriptor row before artifact provenance binding."""

    descriptor_id: str
    population: DescriptorPopulation
    population_count: Literal[10, 200]
    statistic: DescriptorStatistic
    category_label: str | None
    status: DescriptorStatus
    value: float | None
    unit: DescriptorUnit

    def __post_init__(self) -> None:
        if not self.descriptor_id:
            raise ValueError("descriptor_id must be nonempty")
        if self.population_count != (10 if self.population == "nuscenes-mini-replay" else 200):
            raise ValueError("descriptor population count is invalid")
        if self.category_label is not None and (
            not self.category_label or "/" in self.category_label or "\\" in self.category_label
        ):
            raise ValueError("descriptor category label is not public-safe")
        if self.status == "ok":
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("defined descriptor requires a finite value")
        elif self.value is not None or self.statistic != "not-modeled":
            raise ValueError("not-modeled descriptor must be null and not applicable")


def _across(values: tuple[float, ...]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape not in {(10,), (200,)} or not bool(np.all(np.isfinite(array))):
        raise ValueError("descriptor across-population values are invalid")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.quantile(array, 0.5, method="linear")),
        "maximum": float(np.max(array)),
    }


def _rows_for_scalar(
    *,
    descriptor_id: str,
    population: DescriptorPopulation,
    values: tuple[float, ...],
    unit: DescriptorUnit,
) -> tuple[ReplayDescriptorAggregate, ...]:
    population_count: Literal[10, 200] = 10 if population == "nuscenes-mini-replay" else 200
    summaries = _across(values)
    return tuple(
        ReplayDescriptorAggregate(
            descriptor_id=descriptor_id,
            population=population,
            population_count=population_count,
            statistic=statistic,
            category_label=None,
            status="ok",
            value=summaries[statistic],
            unit=unit,
        )
        for statistic in _ACROSS_STATISTICS
    )


def _rows_for_quantiles(
    *,
    descriptor_id: str,
    population: DescriptorPopulation,
    quantiles: tuple[DescriptorQuantiles, ...],
    unit: DescriptorUnit,
) -> tuple[ReplayDescriptorAggregate, ...]:
    output: list[ReplayDescriptorAggregate] = []
    for quantile_id, field_name in _QUANTILE_FIELDS:
        output.extend(
            _rows_for_scalar(
                descriptor_id=f"{descriptor_id}-{quantile_id}",
                population=population,
                values=tuple(float(getattr(row, field_name)) for row in quantiles),
                unit=unit,
            )
        )
    return tuple(output)


def _categorical_rows(
    *,
    descriptor_id: str,
    counts_by_scene: tuple[tuple[tuple[str, int], ...], ...],
    eligible_counts: tuple[int, ...],
) -> tuple[ReplayDescriptorAggregate, ...]:
    labels = tuple(
        sorted(
            {label for scene in counts_by_scene for label, _ in scene},
            key=lambda value: value.encode("utf-8"),
        )
    )
    total_eligible = sum(eligible_counts)
    if total_eligible <= 0:
        raise ValueError("categorical descriptors require nonempty eligible support")
    output: list[ReplayDescriptorAggregate] = []
    for label in labels:
        scene_counts = tuple(dict(scene).get(label, 0) for scene in counts_by_scene)
        scene_fractions = tuple(
            count / eligible for count, eligible in zip(scene_counts, eligible_counts, strict=True)
        )
        total_count = sum(scene_counts)
        output.extend(
            (
                ReplayDescriptorAggregate(
                    descriptor_id=descriptor_id,
                    population="nuscenes-mini-replay",
                    population_count=10,
                    statistic="count",
                    category_label=label,
                    status="ok",
                    value=float(total_count),
                    unit="count",
                ),
                ReplayDescriptorAggregate(
                    descriptor_id=descriptor_id,
                    population="nuscenes-mini-replay",
                    population_count=10,
                    statistic="fraction",
                    category_label=label,
                    status="ok",
                    value=total_count / total_eligible,
                    unit="fraction",
                ),
            )
        )
        summaries = _across(scene_fractions)
        output.extend(
            ReplayDescriptorAggregate(
                descriptor_id=f"{descriptor_id}-per-scene-fraction",
                population="nuscenes-mini-replay",
                population_count=10,
                statistic=statistic,
                category_label=label,
                status="ok",
                value=summaries[statistic],
                unit="fraction",
            )
            for statistic in _ACROSS_STATISTICS
        )
    return tuple(output)


def build_replay_descriptor_aggregates(
    primitives: tuple[SceneDescriptorPrimitives, ...],
    *,
    distinct_log_group_count: int,
) -> tuple[ReplayDescriptorAggregate, ...]:
    """Apply within-scene-first descriptor aggregation to exactly ten scenes."""

    if len(primitives) != 10:
        raise ValueError("M5 descriptors require exactly ten scene primitives")
    if type(distinct_log_group_count) is not int or not 1 <= distinct_log_group_count <= 10:
        raise ValueError("M5 descriptor log-group count is invalid")
    eligible = tuple(row.eligible_object_frame_count for row in primitives)
    if any(count <= 0 for count in eligible):
        raise ValueError("every replay scene must have nonempty base support")
    output: list[ReplayDescriptorAggregate] = []

    scalar_fields: tuple[tuple[str, str, DescriptorUnit], ...] = (
        ("sample-count", "sample_count", "count"),
        ("eligible-object-frame-count", "eligible_object_frame_count", "count"),
        ("unique-eligible-track-count", "unique_eligible_track_count", "count"),
        (
            "zero-order-hold-velocity-fraction",
            "zero_order_hold_velocity_fraction",
            "fraction",
        ),
    )
    for descriptor_id, field_name, unit in scalar_fields:
        output.extend(
            _rows_for_scalar(
                descriptor_id=descriptor_id,
                population="nuscenes-mini-replay",
                values=tuple(float(getattr(row, field_name)) for row in primitives),
                unit=unit,
            )
        )

    waterfall_fields = (
        ("support-all-annotations", "all_annotations"),
        ("support-roi-pass", "roi_pass"),
        ("support-camera-center-pass", "camera_center_pass"),
        ("support-lidar-points-positive", "lidar_points_positive"),
        ("support-final-eligible", "final_eligible"),
    )
    for descriptor_id, field_name in waterfall_fields:
        output.extend(
            _rows_for_scalar(
                descriptor_id=descriptor_id,
                population="nuscenes-mini-replay",
                values=tuple(
                    float(getattr(row.support_waterfall, field_name)) for row in primitives
                ),
                unit="count",
            )
        )

    vector_fields: tuple[tuple[str, str, DescriptorUnit], ...] = (
        ("reference-time-delta", "reference_time_delta_s", "s"),
        ("camera-minus-lidar-acquisition-offset", "camera_minus_lidar_offset_s", "s"),
        ("eligible-track-length", "eligible_track_length_frames", "frames"),
        ("ego-range", "ego_range_m", "m"),
        ("ego-bearing", "ego_bearing_rad", "rad"),
        ("box-width", "box_width_m", "m"),
        ("box-length", "box_length_m", "m"),
        ("box-height", "box_height_m", "m"),
        ("finite-difference-speed", "finite_difference_speed_mps", "m/s"),
        (
            "finite-difference-acceleration",
            "finite_difference_acceleration_mps2",
            "m/s^2",
        ),
        ("lidar-point-count", "num_lidar_points", "count"),
    )
    for descriptor_id, field_name, unit in vector_fields:
        output.extend(
            _rows_for_quantiles(
                descriptor_id=descriptor_id,
                population="nuscenes-mini-replay",
                quantiles=tuple(
                    descriptor_quantiles(getattr(row, field_name)) for row in primitives
                ),
                unit=unit,
            )
        )

    output.extend(
        _categorical_rows(
            descriptor_id="visibility-level",
            counts_by_scene=tuple(row.visibility_counts for row in primitives),
            eligible_counts=eligible,
        )
    )
    output.extend(
        _categorical_rows(
            descriptor_id="category-composition",
            counts_by_scene=tuple(row.category_counts for row in primitives),
            eligible_counts=eligible,
        )
    )
    output.append(
        ReplayDescriptorAggregate(
            descriptor_id="distinct-log-group-count",
            population="nuscenes-mini-replay",
            population_count=10,
            statistic="count",
            category_label=None,
            status="ok",
            value=float(distinct_log_group_count),
            unit="count",
        )
    )
    return tuple(output)


def _not_modeled(descriptor_id: str) -> ReplayDescriptorAggregate:
    return ReplayDescriptorAggregate(
        descriptor_id=descriptor_id,
        population="m3-main-test-comparator",
        population_count=200,
        statistic="not-modeled",
        category_label=None,
        status="not-applicable",
        value=None,
        unit="unitless",
    )


def build_m3_comparator_descriptor_aggregates(
    plan: LoadedReplayPlan,
) -> tuple[ReplayDescriptorAggregate, ...]:
    """Regenerate only the shared frozen M3 main-test descriptor fields."""

    profile = next(
        (
            value
            for value in plan.persistent_matrix.profiles
            if isinstance(value, MainProceduralProfile)
        ),
        None,
    )
    if profile is None or profile.profile_id != "constant-velocity-front-roi-v1":
        raise ValueError("M5 comparator requires the frozen M3 main profile")
    seed = plan.persistent_cases[0].source_manifest.rng.data_master_seed
    if seed != 1729:
        raise ValueError("M5 comparator requires the frozen M3 data seed")
    sequences = generate_procedural_sequences(
        profile,
        split="test",
        sequence_count=200,
        data_master_seed=seed,
    )
    output: list[ReplayDescriptorAggregate] = []
    output.extend(
        _rows_for_scalar(
            descriptor_id="sample-count",
            population="m3-main-test-comparator",
            values=tuple(float(sequence.frame_count) for sequence in sequences),
            unit="count",
        )
    )
    output.extend(
        _rows_for_quantiles(
            descriptor_id="reference-time-delta",
            population="m3-main-test-comparator",
            quantiles=tuple(
                descriptor_quantiles(tuple(np.diff(sequence.frame_times_s)))
                for sequence in sequences
            ),
            unit="s",
        )
    )
    output.extend(
        _rows_for_scalar(
            descriptor_id="eligible-object-frame-count",
            population="m3-main-test-comparator",
            values=tuple(float(sequence.eligible_object_frame_count) for sequence in sequences),
            unit="count",
        )
    )
    track_quantiles: list[DescriptorQuantiles] = []
    range_quantiles: list[DescriptorQuantiles] = []
    bearing_quantiles: list[DescriptorQuantiles] = []
    speed_quantiles: list[DescriptorQuantiles] = []
    for sequence in sequences:
        track_counts = Counter(int(value) for value in sequence.eligible_object_indices)
        track_quantiles.append(descriptor_quantiles(tuple(track_counts.values())))
        truth = sequence.eligible_truth_xy_m
        velocity = sequence.eligible_velocity_xy_mps
        range_quantiles.append(
            descriptor_quantiles(tuple(float(value) for value in np.linalg.norm(truth, axis=1)))
        )
        bearing_quantiles.append(
            descriptor_quantiles(
                tuple(float(value) for value in np.arctan2(truth[:, 1], truth[:, 0]))
            )
        )
        speed_quantiles.append(
            descriptor_quantiles(tuple(float(value) for value in np.linalg.norm(velocity, axis=1)))
        )
    shared_quantiles: tuple[
        tuple[str, tuple[DescriptorQuantiles, ...], DescriptorUnit],
        ...,
    ] = (
        ("eligible-track-length", tuple(track_quantiles), "frames"),
        ("ego-range", tuple(range_quantiles), "m"),
        ("ego-bearing", tuple(bearing_quantiles), "rad"),
        ("finite-difference-speed", tuple(speed_quantiles), "m/s"),
    )
    for descriptor_id, quantiles, unit in shared_quantiles:
        output.extend(
            _rows_for_quantiles(
                descriptor_id=descriptor_id,
                population="m3-main-test-comparator",
                quantiles=quantiles,
                unit=unit,
            )
        )
    output.extend(
        _not_modeled(descriptor_id)
        for descriptor_id in (
            "camera-minus-lidar-acquisition-offset",
            "box-width",
            "box-length",
            "box-height",
            "finite-difference-acceleration",
            "visibility-level",
            "lidar-point-count",
            "zero-order-hold-velocity-fraction",
            "category-composition",
            "distinct-log-group-count",
        )
    )
    return tuple(output)
