from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import fusion_fault_bench.replay_descriptors as replay_descriptors
from fusion_fault_bench.replay_descriptors import (
    ReplayDescriptorAggregate,
    build_m3_comparator_descriptor_aggregates,
    build_replay_descriptor_aggregates,
)
from fusion_fault_bench.replay_plan import load_replay_plan
from fusion_fault_bench.replay_source import (
    SceneDescriptorPrimitives,
    SupportWaterfall,
)

ROOT = Path(__file__).resolve().parents[1]


def _primitive(index: int) -> SceneDescriptorPrimitives:
    count = index + 1
    return SceneDescriptorPrimitives(
        sample_count=16 + index,
        reference_time_delta_s=(0.4, 0.5, 0.6),
        camera_minus_lidar_offset_s=(-0.01, 0.0, 0.01),
        support_waterfall=SupportWaterfall(20, 18, 16, 14, count),
        eligible_object_frame_count=count,
        unique_eligible_track_count=1,
        eligible_track_length_frames=(count,),
        ego_range_m=(10.0 + index,),
        ego_bearing_rad=(0.1,),
        box_width_m=(2.0,),
        box_length_m=(4.0,),
        box_height_m=(1.5,),
        finite_difference_speed_mps=(3.0,),
        finite_difference_acceleration_mps2=(0.5,),
        visibility_counts=(("v80-100", count),),
        num_lidar_points=(5,),
        zero_order_hold_velocity_fraction=index / 10,
        category_counts=(("vehicle.car", count),),
    )


def test_replay_descriptors_are_within_scene_first_and_aggregate_only() -> None:
    rows = build_replay_descriptor_aggregates(
        tuple(_primitive(index) for index in range(10)),
        distinct_log_group_count=2,
    )

    sample_median = next(
        row for row in rows if row.descriptor_id == "sample-count" and row.statistic == "median"
    )
    range_maximum = next(
        row for row in rows if row.descriptor_id == "ego-range-q100" and row.statistic == "maximum"
    )
    category = next(
        row
        for row in rows
        if row.descriptor_id == "category-composition" and row.statistic == "fraction"
    )
    assert sample_median.value == 20.5
    assert range_maximum.value == 19.0
    assert category.category_label == "vehicle.car"
    assert category.value == 1.0
    assert all("nuscenes:" not in repr(row) and "private-" not in repr(row) for row in rows)


def test_m3_comparator_is_exact_200_sequence_frozen_profile() -> None:
    rows = build_m3_comparator_descriptor_aggregates(load_replay_plan(source_root=ROOT))

    sample = next(
        row for row in rows if row.descriptor_id == "sample-count" and row.statistic == "median"
    )
    async_row = next(
        row for row in rows if row.descriptor_id == "camera-minus-lidar-acquisition-offset"
    )
    assert sample.population_count == 200
    assert sample.value == 48.0
    assert async_row.status == "not-applicable"
    assert async_row.statistic == "not-modeled"
    assert async_row.value is None


def test_descriptor_contract_and_population_shape_rejections() -> None:
    valid = ReplayDescriptorAggregate(
        descriptor_id="sample-count",
        population="nuscenes-mini-replay",
        population_count=10,
        statistic="minimum",
        category_label=None,
        status="ok",
        value=16.0,
        unit="count",
    )
    assert valid.value == 16.0
    for update in (
        {"descriptor_id": ""},
        {"population_count": 200},
        {"category_label": ""},
        {"category_label": "private/path"},
        {"category_label": r"private\path"},
        {"value": None},
        {"value": float("nan")},
        {"status": "not-applicable", "value": 1.0, "statistic": "not-modeled"},
        {"status": "not-applicable", "value": None, "statistic": "minimum"},
    ):
        with pytest.raises(ValueError):
            replace(valid, **update)

    primitives = tuple(_primitive(index) for index in range(10))
    for rows in (primitives[:-1], (*primitives, primitives[0])):
        with pytest.raises(ValueError, match="exactly ten"):
            build_replay_descriptor_aggregates(
                rows,
                distinct_log_group_count=2,
            )
    for count in (True, 0, 11):
        with pytest.raises(ValueError, match="log-group count"):
            build_replay_descriptor_aggregates(
                primitives,
                distinct_log_group_count=count,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="nonempty base support"):
        build_replay_descriptor_aggregates(
            (replace(primitives[0], eligible_object_frame_count=0), *primitives[1:]),
            distinct_log_group_count=2,
        )


def test_descriptor_internal_reducers_reject_empty_or_nonfinite_support() -> None:
    with pytest.raises(ValueError, match="across-population"):
        replay_descriptors._across((1.0,))
    with pytest.raises(ValueError, match="across-population"):
        replay_descriptors._across((float("nan"),) * 10)
    with pytest.raises(ValueError, match="nonempty eligible support"):
        replay_descriptors._categorical_rows(
            descriptor_id="category-composition",
            counts_by_scene=((("vehicle.car", 0),),) * 10,
            eligible_counts=(0,) * 10,
        )
