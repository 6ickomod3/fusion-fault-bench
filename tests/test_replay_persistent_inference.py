from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fusion_fault_bench.contracts.replay_v1 import M5_SCENE_NAMES
from fusion_fault_bench.replay_inference import replay_bootstrap_indices
from fusion_fault_bench.replay_persistent import (
    ReplayPersistentMethodResult,
    ReplayPersistentSceneEvaluation,
)
from fusion_fault_bench.replay_persistent_inference import (
    M5_A_DIRECTIONAL_EXPECTATIONS,
    ReplayPersistentCrossoverEstimate,
    ReplayPersistentPopulationMetric,
    aggregate_replay_persistent_case,
    evaluate_replay_persistent_crossovers,
)
from fusion_fault_bench.replay_plan import ReplayPersistentCase, load_replay_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = load_replay_plan(source_root=ROOT)
IDENTITY_BOOTSTRAP = np.tile(np.arange(10, dtype=np.int64), (40, 1))


def _case(experiment_id: str, selector: str) -> ReplayPersistentCase:
    return next(
        case
        for case in PLAN.persistent_cases
        if case.identity.experiment_id == experiment_id
        and case.fault_condition.selector == selector
    )


def _evaluations(
    case: ReplayPersistentCase,
    *,
    loss_by_method: dict[str, float],
    valid_by_method: dict[str, int] | None = None,
    eligible: int = 4,
    disagreement: float = 1.0,
) -> tuple[ReplayPersistentSceneEvaluation, ...]:
    valid = valid_by_method or {method: eligible for method in case.source_manifest.methods}
    base = ReplayPersistentSceneEvaluation(
        replay_experiment_identity_sha256=case.identity_sha256,
        sequence_id=f"nuscenes:{M5_SCENE_NAMES[0]}",
        condition_id=case.identity.experiment_id,
        condition_selector=case.fault_condition.selector,
        results=tuple(
            ReplayPersistentMethodResult(
                method=method,
                loss_sum_m2=loss_by_method[method],
                valid_object_frame_count=valid[method],
                eligible_object_frame_count=eligible,
            )
            for method in case.source_manifest.methods
        ),
        cross_modal_disagreement_sum_m2=disagreement,
        cross_modal_common_count=eligible,
    )
    return tuple(
        replace(base, sequence_id=f"nuscenes:{scene_name}") for scene_name in M5_SCENE_NAMES
    )


def test_persistent_equal_scene_losses_and_signed_delta_use_ten_scenes() -> None:
    case = _case("replay-lidar-y-bias", "replay-lidar-y-bias:+4")
    rows = aggregate_replay_persistent_case(
        case,
        _evaluations(
            case,
            loss_by_method={
                "camera-only": 8.0,
                "lidar-only": 4.0,
                "fixed-fusion": 2.0,
                "fault-target-drop-policy": 8.0,
                "performance-oracle": 2.0,
            },
        ),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )

    fixed = next(row for row in rows if row.method_id == "fixed-fusion")
    contrast = next(row for row in rows if row.metric_id == "fused-minus-healthy")
    assert fixed.interval.estimate == pytest.approx(0.5)
    assert contrast.interval.estimate == pytest.approx(-1.5)
    assert contrast.scene_values == (-1.5,) * 10


def test_persistent_dropout_reconstructs_pooled_counts_without_zero_loss() -> None:
    case = _case("replay-camera-dropout", "replay-camera-dropout:1")
    rows = aggregate_replay_persistent_case(
        case,
        _evaluations(
            case,
            loss_by_method={
                "camera-only": 0.0,
                "lidar-only": 4.0,
                "fixed-fusion": 0.0,
                "fault-target-drop-policy": 4.0,
            },
            valid_by_method={
                "camera-only": 0,
                "lidar-only": 4,
                "fixed-fusion": 0,
                "fault-target-drop-policy": 4,
            },
        ),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )

    fixed = tuple(row for row in rows if row.method_id == "fixed-fusion")
    assert next(row for row in fixed if row.metric_id == "coverage").interval.estimate == 0.0
    assert (
        next(
            row for row in fixed if row.metric_id == "conditional-matched-center-mse"
        ).interval.estimate
        is None
    )
    target = tuple(row for row in rows if row.method_id == "fault-target-drop-policy")
    assert next(row for row in target if row.metric_id == "coverage").interval.estimate == 1.0
    assert next(
        row for row in target if row.metric_id == "conditional-matched-center-mse"
    ).interval.estimate == pytest.approx(1.0)


def test_common_mode_reports_absolute_loss_and_disagreement_without_delta() -> None:
    case = _case("replay-common-mode-x", "replay-common-mode-x:+4")
    rows = aggregate_replay_persistent_case(
        case,
        _evaluations(
            case,
            loss_by_method={
                "camera-only": 16.0,
                "lidar-only": 16.0,
                "fixed-fusion": 16.0,
            },
            disagreement=0.0,
        ),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )

    assert not any(row.metric_id == "fused-minus-healthy" for row in rows)
    disagreement = next(row for row in rows if row.metric_id == "camera-lidar-disagreement-mse")
    assert disagreement.interval.estimate == 0.0


def test_persistent_crossover_matches_independent_linear_root() -> None:
    cases = tuple(
        case
        for case in PLAN.persistent_cases
        if case.identity.experiment_id == "replay-lidar-y-bias"
    )
    aggregates: list[ReplayPersistentPopulationMetric] = []
    for case in cases:
        scene_delta = case.source_condition.magnitude - 1.0
        rows = aggregate_replay_persistent_case(
            case,
            _evaluations(
                case,
                loss_by_method={
                    "camera-only": 4.0,
                    "lidar-only": 4.0,
                    "fixed-fusion": 4.0 * (scene_delta + 1.0),
                    "fault-target-drop-policy": 4.0,
                    "performance-oracle": 4.0,
                },
            ),
            bootstrap_indices=IDENTITY_BOOTSTRAP,
        )
        aggregates.extend(rows)

    crossovers = evaluate_replay_persistent_crossovers(
        cases,
        tuple(aggregates),
        bootstrap_indices=replay_bootstrap_indices(replicates=40),
    )
    assert len(crossovers) == 2
    assert all(row.status == "observed" for row in crossovers)
    assert all(row.point_estimate == pytest.approx(1.0) for row in crossovers)
    assert all(row.interval_lower == pytest.approx(1.0) for row in crossovers)
    assert all(row.interval_upper == pytest.approx(1.0) for row in crossovers)


def test_h5_a_directional_selector_set_is_exact_and_native_axis_specific() -> None:
    assert len(M5_A_DIRECTIONAL_EXPECTATIONS) == 16
    assert M5_A_DIRECTIONAL_EXPECTATIONS["replay-lidar-y-bias:+4"] == "positive"
    assert M5_A_DIRECTIONAL_EXPECTATIONS["replay-camera-noise-correctly-reported:4"] == "negative"


def test_persistent_population_contract_rejects_malformed_coordinates_and_support() -> None:
    case = _case("replay-camera-dropout", "replay-camera-dropout:1")
    rows = aggregate_replay_persistent_case(
        case,
        _evaluations(
            case,
            loss_by_method={
                "camera-only": 0.0,
                "lidar-only": 4.0,
                "fixed-fusion": 0.0,
                "fault-target-drop-policy": 4.0,
            },
            valid_by_method={
                "camera-only": 0,
                "lidar-only": 4,
                "fixed-fusion": 0,
                "fault-target-drop-policy": 4,
            },
        ),
        bootstrap_indices=IDENTITY_BOOTSTRAP,
    )
    conditional = next(
        row
        for row in rows
        if row.method_id == "fixed-fusion" and row.metric_id == "conditional-matched-center-mse"
    )
    assert conditional.scene_values == (None,) * 10
    for update in (
        {"replay_experiment_identity_sha256": "short"},
        {"condition_id": ""},
        {"condition_selector": "replay-clean:0"},
        {"method_id": ""},
        {"scene_numerators": conditional.scene_numerators[:-1]},
        {"scene_denominators": conditional.scene_denominators[:-1]},
        {"scene_numerators": (float("nan"), *conditional.scene_numerators[1:])},
        {"scene_denominators": (True, *conditional.scene_denominators[1:])},
        {"scene_denominators": (-1, *conditional.scene_denominators[1:])},
    ):
        with pytest.raises(ValueError):
            replace(conditional, **update)


def test_persistent_crossover_contract_and_input_grid_rejections() -> None:
    valid = ReplayPersistentCrossoverEstimate(
        replay_experiment_identity_sha256="a" * 64,
        condition_id="replay-lidar-y-bias",
        direction="positive",
        severity_unit="m",
        tested_maximum=4.0,
        status="observed",
        point_estimate=1.0,
        interval_lower=0.5,
        interval_upper=1.5,
        bootstrap_crossing_count=20,
        bootstrap_replicates=40,
    )
    assert valid.bootstrap_crossing_fraction == 0.5
    for update in (
        {"replay_experiment_identity_sha256": "short"},
        {"condition_id": ""},
        {"tested_maximum": float("nan")},
        {"tested_maximum": 0.0},
        {"bootstrap_crossing_count": True},
        {"bootstrap_replicates": True},
        {"bootstrap_crossing_count": -1},
        {"bootstrap_crossing_count": 41},
    ):
        with pytest.raises(ValueError):
            replace(valid, **update)

    geometry_cases = tuple(
        case
        for case in PLAN.persistent_cases
        if case.identity.experiment_id == "replay-lidar-y-bias"
    )
    aggregates: list[ReplayPersistentPopulationMetric] = []
    for case in geometry_cases:
        aggregates.extend(
            aggregate_replay_persistent_case(
                case,
                _evaluations(
                    case,
                    loss_by_method={
                        "camera-only": 4.0,
                        "lidar-only": 4.0,
                        "fixed-fusion": 4.0,
                        "fault-target-drop-policy": 4.0,
                        "performance-oracle": 4.0,
                    },
                ),
                bootstrap_indices=IDENTITY_BOOTSTRAP,
            )
        )
    with pytest.raises(ValueError, match="one geometry experiment"):
        evaluate_replay_persistent_crossovers((), tuple(aggregates))
    availability = _case("replay-camera-dropout", "replay-camera-dropout:1")
    with pytest.raises(ValueError, match="one geometry experiment"):
        evaluate_replay_persistent_crossovers((availability,), ())
    mixed = (
        geometry_cases[0],
        _case("replay-camera-calibration-x", "replay-camera-calibration-x:0"),
    )
    with pytest.raises(ValueError, match="one frozen experiment"):
        evaluate_replay_persistent_crossovers(mixed, tuple(aggregates))
    first_selector = geometry_cases[0].fault_condition.selector
    incomplete = tuple(
        row
        for row in aggregates
        if not (row.condition_selector == first_selector and row.metric_id == "fused-minus-healthy")
    )
    with pytest.raises(ValueError, match="complete severity curve"):
        evaluate_replay_persistent_crossovers(geometry_cases, incomplete)


def test_persistent_aggregation_rejects_scene_order_and_case_substitution() -> None:
    case = _case("replay-lidar-y-bias", "replay-lidar-y-bias:+4")
    evaluations = _evaluations(
        case,
        loss_by_method={
            "camera-only": 4.0,
            "lidar-only": 4.0,
            "fixed-fusion": 4.0,
            "fault-target-drop-policy": 4.0,
            "performance-oracle": 4.0,
        },
    )
    with pytest.raises(ValueError, match="frozen order"):
        aggregate_replay_persistent_case(
            case,
            tuple(reversed(evaluations)),
            bootstrap_indices=IDENTITY_BOOTSTRAP,
        )
    for update in (
        {"replay_experiment_identity_sha256": "0" * 64},
        {"condition_id": "replay-camera-calibration-x"},
        {"condition_selector": "replay-camera-calibration-x:0"},
    ):
        changed = (replace(evaluations[0], **update), *evaluations[1:])
        with pytest.raises(ValueError, match="frozen case"):
            aggregate_replay_persistent_case(
                case,
                changed,
                bootstrap_indices=IDENTITY_BOOTSTRAP,
            )
