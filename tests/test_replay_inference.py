from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_HYPOTHESIS_COORDINATES,
)
from fusion_fault_bench.contracts.replay_v1 import M5_SCENE_NAMES
from fusion_fault_bench.replay_inference import (
    H5_B_SELECTORS,
    LeaveOutEstimate,
    PersistenceAssessment,
    ReplayHealthSequenceContrast,
    ReplayInterval,
    SceneSignCounts,
    classify_persistence,
    conditional_observed_mean_interval,
    equal_scene_contrast_interval,
    equal_scene_loss_interval,
    equal_scene_ratio_interval,
    equal_scene_value_interval,
    leave_one_log_group_out,
    leave_one_scene_out,
    observed_fraction_interval,
    pooled_availability_interval,
    pooled_conditional_loss_interval,
    replay_bootstrap_indices,
    replay_sequence_contrast_values,
    scene_sign_counts,
    supports_nonpositive_control,
)

_DIGEST = "a" * 64


def _identity_bootstrap(scene_count: int, replicates: int = 40) -> np.ndarray:
    return np.tile(np.arange(scene_count, dtype=np.int64), (replicates, 1))


def _contrast(
    sequence_index: int,
    *,
    fixed_sum: float = 10.0,
    policy_sum: float = 4.0,
    common_count: int = 2,
) -> ReplayHealthSequenceContrast:
    return ReplayHealthSequenceContrast(
        replay_experiment_identity_sha256=_DIGEST,
        sequence_id=f"nuscenes:{M5_SCENE_NAMES[sequence_index]}",
        condition_id="replay-lidar-output-y-bias",
        condition_selector="replay-lidar-output-y-bias:+3",
        policy="combined-health-gate",
        window="event",
        fixed_support_sha256=_DIGEST,
        policy_support_sha256=_DIGEST,
        fixed_policy_common_count=common_count,
        fixed_on_common_loss_sum_m2=fixed_sum,
        policy_on_fixed_common_loss_sum_m2=policy_sum,
        target_drop_applicable=True,
        policy_target_drop_common_count=common_count,
        policy_on_target_common_loss_sum_m2=policy_sum,
        target_drop_on_common_loss_sum_m2=2.0 if common_count else 0.0,
        target_drop_support_sha256=_DIGEST,
        frame_oracle_applicable=True,
        policy_frame_oracle_common_count=common_count,
        policy_on_oracle_common_loss_sum_m2=policy_sum,
        frame_oracle_on_common_loss_sum_m2=1.0 if common_count else 0.0,
        frame_oracle_support_sha256=_DIGEST,
    )


def test_replay_bootstrap_uses_exact_pcg64dxsm_defaults_and_shape() -> None:
    observed = replay_bootstrap_indices()
    expected = np.random.Generator(np.random.PCG64DXSM(1_618_033)).integers(
        0,
        10,
        size=(2_000, 10),
        dtype=np.int64,
    )
    assert observed.shape == (2_000, 10)
    assert np.array_equal(observed, expected)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: ReplayInterval(1.0, 0.0, 2.0, 40, 0), "positive"),
        (lambda: ReplayInterval(1.0, 0.0, 2.0, 41, 40), "within"),
        (lambda: ReplayInterval(1.0, None, 2.0, 40, 40), "both endpoints"),
        (lambda: ReplayInterval(1.0, 0.0, 2.0, 39, 40), "97.5"),
        (lambda: ReplayInterval(float("nan"), 0.0, 2.0, 40, 40), "finite"),
        (lambda: ReplayInterval(1.0, 2.0, 0.0, 40, 40), "lower endpoint"),
        (lambda: LeaveOutEstimate("", 1.0), "nonempty"),
        (lambda: LeaveOutEstimate("scene-ordinal:00", float("inf")), "finite"),
        (lambda: SceneSignCounts(-1, 0, 0), "nonnegative"),
    ],
)
def test_inference_value_objects_reject_inconsistent_intervals_and_counts(
    factory: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_persistence_value_objects_bind_group_rows_and_scene_count() -> None:
    signs = SceneSignCounts(1, 2, 3, 4)
    assert signs.total == 10
    row = LeaveOutEstimate("log-group:00", 1.0)
    with pytest.raises(ValueError, match="at least one log group"):
        PersistenceAssessment(
            label="non-persistent",
            expected_direction="positive",
            scene_signs=signs,
            leave_one_scene_out=(),
            distinct_log_group_count=0,
            leave_one_log_group_out=(),
        )
    with pytest.raises(ValueError, match="must match"):
        PersistenceAssessment(
            label="non-persistent",
            expected_direction="positive",
            scene_signs=signs,
            leave_one_scene_out=(),
            distinct_log_group_count=2,
            leave_one_log_group_out=(row,),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"policy": "unknown"}, "unknown replay health policy"),
        ({"window": "unknown"}, "unknown replay health window"),
        ({"target_drop_applicable": 1}, "must be a boolean"),
        ({"frame_oracle_applicable": 1}, "must be a boolean"),
        ({"replay_experiment_identity_sha256": "A" * 64}, "lowercase SHA-256"),
        ({"sequence_id": "unsafe/path"}, "safe replay identifier"),
        ({"sequence_id": "nuscenes:scene-9999"}, "frozen scene population"),
        ({"condition_id": "replay-not-preregistered"}, "frozen M5-B matrix"),
        (
            {"condition_selector": "replay-camera-output-y-bias:+3"},
            "bind the base condition_id",
        ),
        ({"fixed_policy_common_count": -1}, "nonnegative integer"),
        ({"fixed_on_common_loss_sum_m2": -1.0}, "finite and nonnegative"),
        (
            {"fixed_policy_common_count": 0},
            "zero common support requires zero loss sums",
        ),
        (
            {"policy_target_drop_common_count": None},
            "applicability must match every",
        ),
        (
            {
                "target_drop_applicable": False,
                "target_drop_support_sha256": None,
            },
            "inapplicable target-drop statistics",
        ),
        ({"target_drop_support_sha256": "bad"}, "lowercase SHA-256"),
        (
            {
                "policy_target_drop_common_count": 0,
                "policy_on_target_common_loss_sum_m2": 1.0,
            },
            "zero common support requires zero loss sums",
        ),
        (
            {"policy_frame_oracle_common_count": None},
            "applicability must match every",
        ),
        (
            {
                "frame_oracle_applicable": False,
                "frame_oracle_support_sha256": None,
            },
            "inapplicable frame-oracle statistics",
        ),
        ({"frame_oracle_support_sha256": "bad"}, "lowercase SHA-256"),
        (
            {
                "policy_frame_oracle_common_count": 0,
                "policy_on_oracle_common_loss_sum_m2": 1.0,
            },
            "zero common support requires zero loss sums",
        ),
        (
            {"policy_frame_oracle_common_count": 1},
            "identical nonempty three-way support",
        ),
    ],
)
def test_sequence_contrast_contract_rejects_semantic_and_support_rebinding(
    updates: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_contrast(0), **updates)


def test_sequence_contrast_reports_identical_three_way_support_applicability() -> None:
    row = _contrast(0)
    assert row.identical_support_recovery_applicable
    without_oracle = replace(
        row,
        frame_oracle_applicable=False,
        policy_frame_oracle_common_count=None,
        policy_on_oracle_common_loss_sum_m2=None,
        frame_oracle_on_common_loss_sum_m2=None,
        frame_oracle_support_sha256=None,
    )
    assert not without_oracle.identical_support_recovery_applicable


def test_equal_scene_loss_and_ratio_never_reduce_away_zero_support_scenes() -> None:
    indices = _identity_bootstrap(2)
    loss = equal_scene_loss_interval([2.0, 6.0], [2, 3], [2, 3], indices)
    assert loss.estimate == pytest.approx(1.5)
    assert loss.lower == pytest.approx(1.5)
    assert loss.upper == pytest.approx(1.5)

    assert equal_scene_loss_interval([0.0, 6.0], [0, 3], [0, 3], indices).estimate is None
    assert equal_scene_loss_interval([1.0, 6.0], [1, 2], [2, 3], indices).estimate is None
    assert equal_scene_ratio_interval([0, 1], [0, 2], indices).estimate is None


def test_equal_scene_value_interval_preserves_signed_scene_contrasts() -> None:
    indices = _identity_bootstrap(2)
    result = equal_scene_value_interval([-2.0, 1.0], indices)
    assert result.estimate == pytest.approx(-0.5)
    assert result.lower == pytest.approx(-0.5)
    assert result.upper == pytest.approx(-0.5)


def test_pooled_availability_retains_zero_count_scenes_and_reconstructs_counts() -> None:
    indices = _identity_bootstrap(2)
    result = pooled_availability_interval([0, 2], [0, 4], indices)
    assert result.estimate == pytest.approx(0.5)
    assert result.lower == pytest.approx(0.5)
    assert result.defined_replicates == 40
    assert result.defined_fraction == 1.0

    undefined = pooled_availability_interval([0, 0], [0, 0], indices)
    assert undefined.estimate is None
    assert undefined.defined_replicates == 0


def test_conditional_denominator_gate_is_strictly_greater_than_0975() -> None:
    boundary_indices = np.vstack(
        (
            np.ones((39, 2), dtype=np.int64),
            np.zeros((1, 2), dtype=np.int64),
        )
    )
    boundary = pooled_conditional_loss_interval(
        [0.0, 2.0],
        [0, 1],
        boundary_indices,
    )
    assert boundary.defined_replicates == 39
    assert boundary.defined_fraction == 0.975
    assert boundary.estimate is None

    all_defined = pooled_conditional_loss_interval(
        [0.0, 2.0],
        [0, 1],
        np.ones((40, 2), dtype=np.int64),
    )
    assert all_defined.estimate == pytest.approx(2.0)
    assert all_defined.defined_replicates == 40


def test_conditional_observed_mean_reconstructs_signed_values_without_imputation() -> None:
    result = conditional_observed_mean_interval(
        (-2.0, None, 4.0),
        _identity_bootstrap(3),
    )
    assert result.estimate == pytest.approx(1.0)
    assert result.lower == pytest.approx(1.0)

    boundary_indices = np.vstack(
        (
            np.ones((39, 3), dtype=np.int64) * 2,
            np.ones((1, 3), dtype=np.int64),
        )
    )
    censored = conditional_observed_mean_interval(
        (None, None, -2.0),
        boundary_indices,
    )
    assert censored.defined_replicates == 39
    assert censored.estimate is None

    support = observed_fraction_interval(
        (-2.0, None, 4.0),
        _identity_bootstrap(3),
    )
    assert support.estimate == pytest.approx(2.0 / 3.0)
    assert support.defined_replicates == 40


def test_policy_gain_is_reconstructed_only_from_paired_common_support_rows() -> None:
    rows = (
        _contrast(0),
        _contrast(1, fixed_sum=8.0, policy_sum=2.0),
    )
    values = replay_sequence_contrast_values(rows)
    assert values == (3.0, 3.0)
    result = equal_scene_contrast_interval(rows, _identity_bootstrap(2))
    assert result.estimate == pytest.approx(3.0)
    assert result.lower == pytest.approx(3.0)

    target_gap = replay_sequence_contrast_values(
        rows,
        contrast="policy-target-drop",
    )
    assert target_gap == (1.0, 0.0)
    oracle_gap = replay_sequence_contrast_values(
        rows,
        contrast="policy-frame-oracle",
    )
    assert oracle_gap == (1.5, 0.5)


def test_any_zero_common_support_makes_equal_scene_gain_undefined() -> None:
    rows = (
        _contrast(0),
        _contrast(1, fixed_sum=0.0, policy_sum=0.0, common_count=0),
    )
    result = equal_scene_contrast_interval(rows, _identity_bootstrap(2))
    assert result.estimate is None
    assert result.defined_replicates == 0
    assert replay_sequence_contrast_values(rows) == (3.0, None)


def test_common_support_contract_rejects_unpaired_and_mismatched_coordinates() -> None:
    with pytest.raises(ValueError, match="zero common support"):
        _contrast(0, fixed_sum=1.0, policy_sum=0.0, common_count=0)

    base = _contrast(0)
    with pytest.raises(ValueError, match="condition_selector"):
        replace(
            base,
            condition_selector="replay-camera-output-y-bias:+3",
        )
    with pytest.raises(ValueError, match="target-drop"):
        replace(
            base,
            target_drop_applicable=False,
        )

    other_coordinate = replace(
        _contrast(1),
        condition_selector="replay-lidar-output-y-bias:-3",
    )
    with pytest.raises(ValueError, match="condition_selector"):
        equal_scene_contrast_interval(
            (base, other_coordinate),
            _identity_bootstrap(2),
        )


def test_contrast_reconstruction_rejects_empty_duplicate_and_inapplicable_rows() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        replay_sequence_contrast_values(())

    row = _contrast(0)
    with pytest.raises(ValueError, match="duplicate scene"):
        replay_sequence_contrast_values((row, row))

    no_target = replace(
        row,
        target_drop_applicable=False,
        policy_target_drop_common_count=None,
        policy_on_target_common_loss_sum_m2=None,
        target_drop_on_common_loss_sum_m2=None,
        target_drop_support_sha256=None,
    )
    with pytest.raises(ValueError, match="target-drop contrast is inapplicable"):
        replay_sequence_contrast_values(
            (no_target,),
            contrast="policy-target-drop",
        )

    no_oracle = replace(
        row,
        frame_oracle_applicable=False,
        policy_frame_oracle_common_count=None,
        policy_on_oracle_common_loss_sum_m2=None,
        frame_oracle_on_common_loss_sum_m2=None,
        frame_oracle_support_sha256=None,
    )
    with pytest.raises(ValueError, match="frame-oracle contrast is inapplicable"):
        replay_sequence_contrast_values(
            (no_oracle,),
            contrast="policy-frame-oracle",
        )
    with pytest.raises(ValueError, match="unknown replay sequence contrast"):
        replay_sequence_contrast_values(
            (row,),
            contrast=cast(Any, "unknown"),
        )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: replay_bootstrap_indices(seed=-1), "seed"),
        (lambda: replay_bootstrap_indices(seed=True), "seed"),
        (lambda: replay_bootstrap_indices(replicates=0), "replicates"),
        (lambda: replay_bootstrap_indices(scene_count=0), "scene_count"),
        (
            lambda: equal_scene_value_interval([], np.ones((1, 0), dtype=np.int64)),
            "nonempty finite vector",
        ),
        (
            lambda: equal_scene_value_interval(
                [float("nan")],
                np.zeros((1, 1), dtype=np.int64),
            ),
            "nonempty finite vector",
        ),
        (
            lambda: equal_scene_value_interval([1.0, 2.0], np.zeros(2, dtype=np.int64)),
            "integer matrix",
        ),
        (
            lambda: equal_scene_value_interval(
                [1.0, 2.0],
                np.asarray([[0, 2]], dtype=np.int64),
            ),
            "outside the scene population",
        ),
        (
            lambda: equal_scene_loss_interval(
                [-1.0],
                [1],
                [1],
                np.zeros((1, 1), dtype=np.int64),
            ),
            "nonnegative",
        ),
        (
            lambda: equal_scene_loss_interval(
                [1.0],
                [1.0],
                [1],
                np.zeros((1, 1), dtype=np.int64),
            ),
            "integer vector",
        ),
        (
            lambda: equal_scene_loss_interval(
                [1.0],
                [2],
                [1],
                np.zeros((1, 1), dtype=np.int64),
            ),
            "cannot exceed",
        ),
        (
            lambda: equal_scene_loss_interval(
                [1.0],
                [0],
                [1],
                np.zeros((1, 1), dtype=np.int64),
            ),
            "zero valid support",
        ),
        (
            lambda: equal_scene_ratio_interval(
                [],
                [],
                np.ones((1, 0), dtype=np.int64),
            ),
            "nonempty integer vector",
        ),
        (
            lambda: equal_scene_ratio_interval(
                [2],
                [1],
                np.zeros((1, 1), dtype=np.int64),
            ),
            "cannot exceed",
        ),
        (
            lambda: pooled_availability_interval(
                [],
                [],
                np.ones((1, 0), dtype=np.int64),
            ),
            "nonempty integer vector",
        ),
        (
            lambda: pooled_availability_interval(
                [2],
                [1],
                np.zeros((1, 1), dtype=np.int64),
            ),
            "cannot exceed",
        ),
        (
            lambda: pooled_conditional_loss_interval(
                [-1.0],
                [1],
                np.zeros((1, 1), dtype=np.int64),
            ),
            "nonnegative",
        ),
        (
            lambda: pooled_conditional_loss_interval(
                [1.0],
                [0],
                np.zeros((1, 1), dtype=np.int64),
            ),
            "zero valid support",
        ),
        (
            lambda: conditional_observed_mean_interval(
                (),
                np.ones((1, 0), dtype=np.int64),
            ),
            "nonempty",
        ),
        (
            lambda: conditional_observed_mean_interval(
                (float("inf"),),
                np.zeros((1, 1), dtype=np.int64),
            ),
            "finite when observed",
        ),
        (
            lambda: observed_fraction_interval(
                (),
                np.ones((1, 0), dtype=np.int64),
            ),
            "nonempty",
        ),
        (
            lambda: observed_fraction_interval(
                (float("nan"),),
                np.zeros((1, 1), dtype=np.int64),
            ),
            "finite when defined",
        ),
    ],
)
def test_public_inference_functions_reject_invalid_shapes_support_and_values(
    call: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_leave_out_rows_are_opaque_complete_scene_and_complete_log_group_means() -> None:
    values = (1.0, 3.0, 5.0)
    loso = leave_one_scene_out(values)
    assert tuple(row.cluster_id for row in loso) == (
        "scene-ordinal:00",
        "scene-ordinal:01",
        "scene-ordinal:02",
    )
    assert tuple(row.estimate for row in loso) == (4.0, 3.0, 2.0)

    lolo = leave_one_log_group_out(
        values,
        ("z-private-token", "a-private-token", "z-private-token"),
    )
    assert tuple(row.cluster_id for row in lolo) == (
        "log-group:00",
        "log-group:01",
    )
    assert tuple(row.estimate for row in lolo) == (3.0, 3.0)
    assert "private" not in repr(lolo)


def test_leave_out_contracts_reject_misaligned_or_empty_log_groups() -> None:
    with pytest.raises(ValueError, match="align with scene values"):
        leave_one_log_group_out((1.0, 2.0), ("group-a",))
    with pytest.raises(ValueError, match="nonempty strings"):
        leave_one_log_group_out((1.0, 2.0), ("group-a", ""))

    undefined = leave_one_scene_out((1.0, None))
    assert tuple(row.estimate for row in undefined) == (None, 1.0)


def test_persistence_classification_uses_sign_interval_loso_and_lolo_gates() -> None:
    values = (1.0,) * 10
    groups = ("group-b",) * 5 + ("group-a",) * 5
    interval = ReplayInterval(1.0, 0.5, 1.5, 2_000, 2_000)
    robust = classify_persistence(interval, values, groups, "positive")
    assert robust.label == "robustly-persistent"
    assert robust.scene_signs.positive == 10
    assert robust.distinct_log_group_count == 2

    crossing = classify_persistence(
        ReplayInterval(1.0, -0.1, 1.5, 2_000, 2_000),
        values,
        groups,
        "positive",
    )
    assert crossing.label == "directionally-consistent"

    wrong = classify_persistence(
        ReplayInterval(-1.0, -1.5, -0.5, 2_000, 2_000),
        (-1.0,) * 10,
        groups,
        "positive",
    )
    assert wrong.label == "non-persistent"

    undefined = classify_persistence(
        ReplayInterval(None, None, None, 0, 2_000),
        (None, *values[1:]),
        groups,
        "positive",
    )
    assert undefined.label == "undefined"
    assert undefined.scene_signs.undefined == 1


def test_negative_persistence_and_classification_input_contracts() -> None:
    values = (-1.0,) * 10
    groups = ("group-a",) * 5 + ("group-b",) * 5
    robust = classify_persistence(
        ReplayInterval(-1.0, -1.5, -0.5, 2_000, 2_000),
        values,
        groups,
        "negative",
    )
    assert robust.label == "robustly-persistent"
    assert robust.scene_signs.negative == 10

    with pytest.raises(ValueError, match="exactly ten scenes"):
        classify_persistence(
            ReplayInterval(1.0, 0.5, 1.5, 40, 40),
            (1.0,),
            ("group-a",),
            "positive",
        )
    with pytest.raises(ValueError, match="align with scene values"):
        classify_persistence(
            ReplayInterval(1.0, 0.5, 1.5, 40, 40),
            (1.0,) * 10,
            ("group-a",) * 9,
            "positive",
        )
    with pytest.raises(ValueError, match="positive or negative"):
        classify_persistence(
            ReplayInterval(1.0, 0.5, 1.5, 40, 40),
            (1.0,) * 10,
            ("group-a",) * 10,
            cast(Any, "zero"),
        )


def test_scene_signs_and_nonpositive_control_use_exact_zero_boundaries() -> None:
    signs = scene_sign_counts((1.0, 0.0, -1.0, None))
    assert (signs.positive, signs.zero, signs.negative, signs.undefined) == (
        1,
        1,
        1,
        1,
    )
    assert supports_nonpositive_control(ReplayInterval(0.0, -1.0, 0.0, 2_000, 2_000))
    assert not supports_nonpositive_control(ReplayInterval(0.0, -1.0, 0.01, 2_000, 2_000))
    assert supports_nonpositive_control(ReplayInterval(None, None, None, 0, 2_000)) is None


def test_h5_b_selectors_match_the_frozen_exact_coordinates() -> None:
    inferred_coordinates = tuple(
        (
            row.hypothesis_id,
            row.selector,
            row.method,
            row.metric_name,
            row.window,
            row.unit,
            (
                "primary-directional"
                if row.assessment_rule == "persistence"
                else "nonpositive-control"
                if row.assessment_rule == "nonpositive-control"
                else "diagnostic"
            ),
            (
                row.expected_direction
                if row.expected_direction is not None
                else "nonpositive"
                if row.assessment_rule == "nonpositive-control"
                else "none"
            ),
        )
        for row in H5_B_SELECTORS
    )
    assert inferred_coordinates == M5_HEALTH_HYPOTHESIS_COORDINATES
