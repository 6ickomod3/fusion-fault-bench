from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fusion_fault_bench.contracts.procedural_profile_v1 import load_procedural_profile
from fusion_fault_bench.experiments.health import generate_health_observations
from fusion_fault_bench.health import HealthCalibration, HealthThresholds
from fusion_fault_bench.health_aggregation import (
    aggregate_health_condition,
    recompute_row_derived_health_aggregates,
)
from fusion_fault_bench.health_evaluation import (
    evaluate_health_sequence,
    sequence_contrast_records,
    sequence_event_record,
    sequence_loss_records,
)
from fusion_fault_bench.health_fit import (
    compute_health_feature_trace,
    rescore_health_feature_trace,
)
from fusion_fault_bench.scenarios.health import (
    HealthFaultSpec,
    generate_health_base_sequences,
)


def _calibration() -> HealthCalibration:
    values = np.asarray([0.0, 1e12], dtype=np.float64)
    return HealthCalibration(
        camera_self_mean=values,
        camera_self_maximum=values,
        lidar_self_mean=values,
        lidar_self_maximum=values,
        camera_from_lidar_cross_mean=values,
        camera_from_lidar_cross_maximum=values,
        lidar_from_camera_cross_mean=values,
        lidar_from_camera_cross_maximum=values,
    )


def _population(fault: HealthFaultSpec):
    profile_path = (
        "examples/profiles/constant-velocity-fov-edge-v1.json"
        if fault.family == "common-mode-position-bias"
        else "examples/profiles/constant-velocity-front-roi-v1.json"
    )
    profile = load_procedural_profile(Path(profile_path))
    bases = generate_health_base_sequences(
        profile,
        split="test",
        sequence_count=2,
        data_master_seed=1729,
    )
    evaluations = []
    losses = []
    contrasts = []
    events = []
    for base in bases:
        observations = generate_health_observations(base, fault=fault)
        evidence = rescore_health_feature_trace(
            compute_health_feature_trace(observations.health_frame_inputs()),
            _calibration(),
        ).frames
        evaluation = evaluate_health_sequence(
            observations,
            condition_id="condition",
            fault=fault,
            evidence=evidence,
            thresholds=HealthThresholds(self_score=1.0, cross_score=1.0),
        )
        evaluations.append(evaluation)
        losses.extend(
            sequence_loss_records(
                evaluation,
                observations=observations,
                fault=fault,
            )
        )
        contrasts.extend(sequence_contrast_records(evaluation, fault=fault))
        events.extend(
            sequence_event_record(
                trace,
                observations=observations,
                condition_id="condition",
                fault=fault,
            )
            for trace in evaluation.policy_traces
        )
    return tuple(evaluations), tuple(losses), tuple(contrasts), tuple(events)


def _find(aggregates, metric, *, method, window):
    return next(
        record
        for record in aggregates
        if record.metric_name == metric and record.method == method and record.window == window
    )


def test_identity_aggregation_reports_losses_coverage_contrasts_and_events() -> None:
    fault = HealthFaultSpec(
        family="identity",
        target="none",
        axis="none",
        unit="identity",
        value=0.0,
    )
    evaluations, losses, contrasts, events = _population(fault)
    aggregates = aggregate_health_condition(
        condition_id="condition",
        fault=fault,
        sequence_ids=tuple(item.sequence_id for item in evaluations),
        sequence_losses=losses,
        sequence_contrasts=contrasts,
        sequence_events=events,
    )
    assert (
        _find(
            aggregates,
            "coverage",
            method="fixed-fusion",
            window="score",
        ).estimate
        == 1.0
    )
    assert (
        _find(
            aggregates,
            "undefined-output-rate",
            method="fixed-fusion",
            window="score",
        ).estimate
        == 0.0
    )
    assert (
        _find(
            aggregates,
            "detection-fraction",
            method="combined-health-gate",
            window="event",
        ).estimate
        == 0.0
    )
    assert (
        _find(
            aggregates,
            "state-healthy-occupancy",
            method="combined-health-gate",
            window="event",
        ).estimate
        == 1.0
    )
    assert _find(
        aggregates,
        "policy-gain-vs-fixed",
        method="combined-health-gate",
        window="score",
    ).estimate == pytest.approx(0.0)
    recomputed = recompute_row_derived_health_aggregates(
        condition_id="condition",
        fault=fault,
        sequence_ids=tuple(item.sequence_id for item in evaluations),
        sequence_losses=losses,
        sequence_contrasts=contrasts,
        sequence_events=events,
    )
    assert recomputed == aggregates


def test_dropout_uses_common_support_and_reports_realization_first() -> None:
    fault = HealthFaultSpec(
        family="dropout",
        target="camera",
        axis="availability",
        unit="probability",
        value=1.0,
    )
    evaluations, losses, contrasts, events = _population(fault)
    aggregates = aggregate_health_condition(
        condition_id="condition",
        fault=fault,
        sequence_ids=tuple(item.sequence_id for item in evaluations),
        sequence_losses=losses,
        sequence_contrasts=contrasts,
        sequence_events=events,
    )
    assert (
        _find(
            aggregates,
            "coverage",
            method="fixed-fusion",
            window="event",
        ).estimate
        == 0.0
    )
    assert (
        _find(
            aggregates,
            "coverage",
            method="combined-health-gate",
            window="event",
        ).estimate
        == 1.0
    )
    assert (
        _find(
            aggregates,
            "policy-gain-vs-fixed",
            method="combined-health-gate",
            window="event",
        ).status
        == "undefined"
    )
    assert (
        _find(
            aggregates,
            "realized-dropout-fraction",
            method=None,
            window="event",
        ).estimate
        == 1.0
    )
    assert (
        _find(
            aggregates,
            "first-missing-frame-minus-event-start",
            method=None,
            window="event",
        ).estimate
        == 0.0
    )
    assert (
        _find(
            aggregates,
            "detection-among-realized-dropout-fraction",
            method="direct-telemetry-gate",
            window="event",
        ).estimate
        == 1.0
    )
    assert (
        _find(
            aggregates,
            "detection-minus-first-missing",
            method="direct-telemetry-gate",
            window="event",
        ).estimate
        == 1.0
    )
    assert not any(
        record.metric_name == "frame-oracle-recoverable-loss-fraction"
        and record.method == "combined-health-gate"
        and record.window == "event"
        for record in aggregates
    )


def test_common_mode_reports_first_latch_labels_without_target_attribution() -> None:
    fault = HealthFaultSpec(
        family="common-mode-position-bias",
        target="both",
        axis="x",
        unit="m",
        value=1.0,
    )
    evaluations, losses, contrasts, events = _population(fault)
    aggregates = aggregate_health_condition(
        condition_id="condition",
        fault=fault,
        sequence_ids=tuple(item.sequence_id for item in evaluations),
        sequence_losses=losses,
        sequence_contrasts=contrasts,
        sequence_events=events,
    )
    metric_names = {record.metric_name for record in aggregates}
    assert {
        "first-latch-label-camera-fault-fraction",
        "first-latch-label-lidar-fault-fraction",
        "first-latch-label-ambiguous-fraction",
        "event-outcome-missed-fraction",
    }.issubset(metric_names)
    assert "attribution-fraction" not in metric_names
    assert "event-outcome-correct-fraction" not in metric_names


def test_aggregation_rejects_duplicate_or_incomplete_populations() -> None:
    fault = HealthFaultSpec(
        family="identity",
        target="none",
        axis="none",
        unit="identity",
        value=0.0,
    )
    evaluations, losses, contrasts, events = _population(fault)
    duplicate = (evaluations[0].sequence_id, evaluations[0].sequence_id)
    with pytest.raises(ValueError):
        aggregate_health_condition(
            condition_id="condition",
            fault=fault,
            sequence_ids=duplicate,
            sequence_losses=losses,
            sequence_contrasts=contrasts,
            sequence_events=events,
        )
    with pytest.raises(ValueError):
        aggregate_health_condition(
            condition_id="condition",
            fault=fault,
            sequence_ids=tuple(item.sequence_id for item in evaluations),
            sequence_losses=losses[:-1],
            sequence_contrasts=contrasts,
            sequence_events=events,
        )
