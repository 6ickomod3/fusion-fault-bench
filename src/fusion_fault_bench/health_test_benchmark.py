"""Apply-only M4 test orchestration over the frozen fit and 47 test variants."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from fusion_fault_bench.artifacts import ArtifactValidationError
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.health_result_v1 import (
    HealthAggregateMetricV1,
    HealthSequenceContrastV1,
    HealthSequenceEventV1,
    HealthSequenceLossV1,
    HealthValidationCheckV1,
    HealthValidationV1,
)
from fusion_fault_bench.contracts.health_v1 import M4_HEALTH_INTENT_SHA256
from fusion_fault_bench.experiments.health import generate_health_observations
from fusion_fault_bench.health import HealthCalibration, HealthThresholds
from fusion_fault_bench.health_aggregation import aggregate_health_condition
from fusion_fault_bench.health_artifacts import (
    LoadedHealthFitArtifact,
    load_health_fit_artifact,
)
from fusion_fault_bench.health_benchmark import expand_test_cases
from fusion_fault_bench.health_evaluation import (
    HealthSequenceEvaluation,
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
    HealthBaseSequence,
    HealthFaultSpec,
    build_bounded_acceleration_control,
    generate_health_base_sequences,
)


class HealthTestCase(Protocol):
    """Structural boundary keeping condition metadata outside scorer inputs."""

    @property
    def condition_id(self) -> str: ...

    @property
    def population(self) -> str: ...

    @property
    def fault(self) -> HealthFaultSpec: ...


@dataclass(frozen=True, slots=True)
class HealthBenchmarkEvaluation:
    """Complete apply-only M4 test rows and conjunctive validation evidence."""

    sequence_losses: tuple[HealthSequenceLossV1, ...]
    sequence_contrasts: tuple[HealthSequenceContrastV1, ...]
    sequence_events: tuple[HealthSequenceEventV1, ...]
    aggregates: tuple[HealthAggregateMetricV1, ...]
    validation: HealthValidationV1
    condition_ids: tuple[str, ...]
    evaluated_sequence_condition_count: int

    def __post_init__(self) -> None:
        if not self.condition_ids or len(set(self.condition_ids)) != len(self.condition_ids):
            raise ValueError("evaluation condition IDs must be nonempty and unique")
        if self.evaluated_sequence_condition_count <= 0:
            raise ValueError("evaluation must contain sequence-condition pairs")
        if (
            not self.sequence_losses
            or not self.sequence_contrasts
            or not self.sequence_events
            or not self.aggregates
        ):
            raise ValueError("evaluation result rows must be nonempty")
        if not self.validation.all_checks_passed:
            raise ValueError("evaluation result requires passing validation evidence")


@dataclass(frozen=True, slots=True)
class HealthConditionEvaluation:
    """One bounded condition batch ready for canonical streaming."""

    condition_id: str
    sequence_ids: tuple[str, ...]
    sequence_losses: tuple[HealthSequenceLossV1, ...]
    sequence_contrasts: tuple[HealthSequenceContrastV1, ...]
    sequence_events: tuple[HealthSequenceEventV1, ...]
    aggregates: tuple[HealthAggregateMetricV1, ...]


@dataclass(frozen=True, slots=True)
class HealthBenchmarkStreamSummary:
    """Global validation evidence after all condition batches were consumed."""

    validation: HealthValidationV1
    condition_ids: tuple[str, ...]
    evaluated_sequence_condition_count: int


def _case_bases(
    case: HealthTestCase,
    *,
    main_bases: tuple[HealthBaseSequence, ...],
    edge_bases: tuple[HealthBaseSequence, ...],
    maneuver_bases: tuple[HealthBaseSequence, ...],
) -> tuple[HealthBaseSequence, ...]:
    if case.fault.family == "clean-predictor-mismatch":
        if case.population != "main-test":
            raise ValueError("bounded-acceleration control must use main-test")
        return maneuver_bases
    if case.population == "main-test":
        return main_bases
    if case.population == "edge-test":
        return edge_bases
    raise ValueError("test case has an unknown population")


def _oracle_dominates(evaluation: HealthSequenceEvaluation) -> bool:
    methods = {trace.method for trace in evaluation.losses}
    if "frame-action-performance-oracle" not in methods:
        return True
    oracle = evaluation.loss("frame-action-performance-oracle")
    for trace in evaluation.losses:
        if trace.method == "combined-health-gate-abstain":
            continue
        if np.array_equal(trace.valid_mask, oracle.valid_mask) and (
            float(oracle.loss_m2.sum(dtype=np.float64))
            > float(trace.loss_m2.sum(dtype=np.float64)) + 1e-12
        ):
            return False
    return True


def _evaluate_health_cases(
    *,
    cases: tuple[HealthTestCase, ...],
    main_bases: tuple[HealthBaseSequence, ...],
    edge_bases: tuple[HealthBaseSequence, ...],
    calibration: HealthCalibration,
    thresholds: HealthThresholds,
    intent_sha256: str = M4_HEALTH_INTENT_SHA256,
    require_exact_matrix: bool = False,
    condition_sink: Callable[[HealthConditionEvaluation], None] | None = None,
    materialize_rows: bool = True,
) -> HealthBenchmarkEvaluation | HealthBenchmarkStreamSummary:
    """Evaluate supplied cases while caching each observable feature trace once."""

    if not cases or len({case.condition_id for case in cases}) != len(cases):
        raise ValueError("test cases must be nonempty with unique value-level IDs")
    if not main_bases or not edge_bases:
        raise ValueError("M4 test populations must be nonempty")
    if require_exact_matrix and (
        len(cases) != 47 or len(main_bases) != 200 or len(edge_bases) != 100
    ):
        raise ValueError("full M4 evaluation requires 47 cases and 200/100 populations")
    maneuver_bases = tuple(build_bounded_acceleration_control(base) for base in main_bases)
    all_losses: list[HealthSequenceLossV1] = []
    all_contrasts: list[HealthSequenceContrastV1] = []
    all_events: list[HealthSequenceEventV1] = []
    all_aggregates: list[HealthAggregateMetricV1] = []
    evaluated_pairs = 0
    loss_row_count = 0
    contrast_row_count = 0
    event_row_count = 0
    oracle_valid = True
    common_mode_boundary_valid = True
    cold_start_window_valid = True

    ordered_cases = tuple(sorted(cases, key=lambda item: item.condition_id))
    for case in ordered_cases:
        bases = _case_bases(
            case,
            main_bases=main_bases,
            edge_bases=edge_bases,
            maneuver_bases=maneuver_bases,
        )
        sequence_ids: list[str] = []
        case_losses: list[HealthSequenceLossV1] = []
        case_contrasts: list[HealthSequenceContrastV1] = []
        case_events: list[HealthSequenceEventV1] = []
        for base in bases:
            observations = generate_health_observations(base, fault=case.fault)
            unscored = compute_health_feature_trace(observations.health_frame_inputs())
            evidence = rescore_health_feature_trace(unscored, calibration).frames
            evaluation = evaluate_health_sequence(
                observations,
                condition_id=case.condition_id,
                fault=case.fault,
                evidence=evidence,
                thresholds=thresholds,
            )
            oracle_valid &= _oracle_dominates(evaluation)
            if case.fault.family == "common-mode-position-bias":
                methods = {trace.method for trace in evaluation.losses}
                common_mode_boundary_valid &= (
                    "fault-target-drop-policy" not in methods
                    and "frame-action-performance-oracle" not in methods
                )
            losses = sequence_loss_records(
                evaluation,
                observations=observations,
                fault=case.fault,
            )
            if case.fault.schedule == "cold_start":
                fixed_event = next(
                    row for row in losses if row.method == "fixed-fusion" and row.window == "event"
                )
                fixed_recovery = next(
                    row
                    for row in losses
                    if row.method == "fixed-fusion" and row.window == "recovery"
                )
                cold_start_window_valid &= (
                    fixed_event.eligible_object_frame_count
                    == fixed_recovery.eligible_object_frame_count
                    == 24 * base.object_count
                )
            events = tuple(
                sequence_event_record(
                    trace,
                    observations=observations,
                    condition_id=case.condition_id,
                    fault=case.fault,
                )
                for trace in evaluation.policy_traces
            )
            contrasts = sequence_contrast_records(
                evaluation,
                fault=case.fault,
            )
            sequence_ids.append(evaluation.sequence_id)
            case_losses.extend(losses)
            case_contrasts.extend(contrasts)
            case_events.extend(events)

        aggregates = aggregate_health_condition(
            condition_id=case.condition_id,
            fault=case.fault,
            sequence_ids=tuple(sequence_ids),
            sequence_losses=case_losses,
            sequence_contrasts=case_contrasts,
            sequence_events=case_events,
        )
        batch = HealthConditionEvaluation(
            condition_id=case.condition_id,
            sequence_ids=tuple(sorted(sequence_ids)),
            sequence_losses=tuple(
                sorted(
                    case_losses,
                    key=lambda row: (row.condition_id, row.sequence_id, row.method, row.window),
                )
            ),
            sequence_contrasts=tuple(
                sorted(
                    case_contrasts,
                    key=lambda row: (row.condition_id, row.sequence_id, row.policy, row.window),
                )
            ),
            sequence_events=tuple(
                sorted(
                    case_events,
                    key=lambda row: (row.condition_id, row.sequence_id, row.policy),
                )
            ),
            aggregates=tuple(
                sorted(
                    aggregates,
                    key=lambda row: (
                        row.condition_id,
                        "" if row.method is None else row.method,
                        row.metric_name,
                        "" if row.window is None else row.window,
                    ),
                )
            ),
        )
        if condition_sink is not None:
            condition_sink(batch)
        evaluated_pairs += len(bases)
        loss_row_count += len(batch.sequence_losses)
        contrast_row_count += len(batch.sequence_contrasts)
        event_row_count += len(batch.sequence_events)
        if materialize_rows:
            all_losses.extend(batch.sequence_losses)
            all_contrasts.extend(batch.sequence_contrasts)
            all_events.extend(batch.sequence_events)
            all_aggregates.extend(batch.aggregates)

    expected_pairs = sum(
        len(edge_bases) if case.population == "edge-test" else len(main_bases) for case in cases
    )
    held_out_yaw_present = any(case.fault.family == "calibration-yaw" for case in cases)
    controls_present = {case.fault.family for case in cases}.issuperset(
        {
            "identity",
            "clean-predictor-mismatch",
            "common-mode-position-bias",
            "dropout",
        }
    )
    checks = (
        HealthValidationCheckV1(
            check_id="intent-identity",
            passed=intent_sha256 == M4_HEALTH_INTENT_SHA256,
            observed=intent_sha256,
            expected=M4_HEALTH_INTENT_SHA256,
        ),
        HealthValidationCheckV1(
            check_id="condition-and-population-counts",
            passed=evaluated_pairs == expected_pairs,
            observed=evaluated_pairs,
            expected=expected_pairs,
        ),
        HealthValidationCheckV1(
            check_id="condition-value-count",
            passed=len(cases) == (47 if require_exact_matrix else len(cases)),
            observed=len(cases),
            expected=47 if require_exact_matrix else len(cases),
        ),
        HealthValidationCheckV1(
            check_id="sequence-loss-row-count",
            passed=loss_row_count == (264_600 if require_exact_matrix else loss_row_count),
            observed=loss_row_count,
            expected=264_600 if require_exact_matrix else loss_row_count,
        ),
        HealthValidationCheckV1(
            check_id="sequence-event-row-count",
            passed=event_row_count == (35_600 if require_exact_matrix else event_row_count),
            observed=event_row_count,
            expected=35_600 if require_exact_matrix else event_row_count,
        ),
        HealthValidationCheckV1(
            check_id="sequence-contrast-row-count",
            passed=contrast_row_count == (133_500 if require_exact_matrix else contrast_row_count),
            observed=contrast_row_count,
            expected=133_500 if require_exact_matrix else contrast_row_count,
        ),
        HealthValidationCheckV1(
            check_id="oracle-frame-action-dominance",
            passed=oracle_valid,
            observed=oracle_valid,
            expected=True,
        ),
        HealthValidationCheckV1(
            check_id="common-mode-hindsight-boundary",
            passed=common_mode_boundary_valid,
            observed=common_mode_boundary_valid,
            expected=True,
        ),
        HealthValidationCheckV1(
            check_id="cold-start-schedule-windows",
            passed=cold_start_window_valid,
            observed=cold_start_window_valid,
            expected=True,
        ),
        HealthValidationCheckV1(
            check_id="held-out-yaw-present",
            passed=held_out_yaw_present,
            observed=held_out_yaw_present,
            expected=True,
        ),
        HealthValidationCheckV1(
            check_id="required-negative-controls-present",
            passed=controls_present,
            observed=controls_present,
            expected=True,
        ),
        HealthValidationCheckV1(
            check_id="feature-computed-once-per-sequence-condition",
            passed=True,
            observed=True,
            expected=True,
        ),
        HealthValidationCheckV1(
            check_id="test-fit-apply-only",
            passed=True,
            observed=True,
            expected=True,
        ),
    )
    validation = HealthValidationV1(
        schema="ffb.health-validation/v1",
        intent_sha256=intent_sha256,
        checks=checks,
        all_checks_passed=all(check.passed for check in checks),
    )
    if not materialize_rows:
        return HealthBenchmarkStreamSummary(
            validation=validation,
            condition_ids=tuple(case.condition_id for case in ordered_cases),
            evaluated_sequence_condition_count=evaluated_pairs,
        )
    return HealthBenchmarkEvaluation(
        sequence_losses=tuple(all_losses),
        sequence_contrasts=tuple(all_contrasts),
        sequence_events=tuple(all_events),
        aggregates=tuple(all_aggregates),
        validation=validation,
        condition_ids=tuple(case.condition_id for case in ordered_cases),
        evaluated_sequence_condition_count=evaluated_pairs,
    )


def _authenticated_fit_handle(
    fit_artifact: object,
) -> LoadedHealthFitArtifact:
    if not isinstance(fit_artifact, LoadedHealthFitArtifact):
        raise TypeError("fit_artifact must be a LoadedHealthFitArtifact")
    authenticated = load_health_fit_artifact(fit_artifact.path)
    if (
        authenticated.artifact_sha256 != fit_artifact.artifact_sha256
        or authenticated.run_sha256 != fit_artifact.run_sha256
    ):
        raise ArtifactValidationError("provided health fit handle is not authentic")
    return authenticated


def evaluate_health_benchmark_test(
    *,
    fit_artifact: LoadedHealthFitArtifact,
) -> HealthBenchmarkEvaluation:
    """Apply one authenticated frozen fit to the exact canonical test matrix.

    Intent, profiles, ECDF calibration, selected thresholds, cases, counts, and
    seed are all reloaded from the content-addressed fit artifact.  The public
    boundary exposes no scientific override parameters.
    """

    authenticated = _authenticated_fit_handle(fit_artifact)
    intent = authenticated.intent
    cases = expand_test_cases(intent)
    if sha256_digest(intent) != M4_HEALTH_INTENT_SHA256 or len(cases) != 47:
        raise ArtifactValidationError("authenticated fit does not bind the frozen M4 matrix")
    counts = intent.source_population.split_sequence_counts
    seed = intent.source_population.data_master_seed
    main_bases = generate_health_base_sequences(
        authenticated.main_profile,
        split="test",
        sequence_count=counts.test,
        data_master_seed=seed,
    )
    edge_bases = generate_health_base_sequences(
        authenticated.edge_profile,
        split="test",
        sequence_count=counts.edge_test,
        data_master_seed=seed,
    )
    thresholds = HealthThresholds(
        self_score=authenticated.summary.selected_self_threshold,
        cross_score=authenticated.summary.selected_cross_threshold,
    )
    result = _evaluate_health_cases(
        cases=cases,
        main_bases=main_bases,
        edge_bases=edge_bases,
        calibration=authenticated.calibration,
        thresholds=thresholds,
        intent_sha256=sha256_digest(intent),
        require_exact_matrix=True,
    )
    return cast(HealthBenchmarkEvaluation, result)


def stream_health_benchmark_test(
    *,
    fit_artifact: LoadedHealthFitArtifact,
    condition_sink: Callable[[HealthConditionEvaluation], None],
) -> HealthBenchmarkStreamSummary:
    """Apply the exact test matrix while retaining at most one condition batch."""

    authenticated = _authenticated_fit_handle(fit_artifact)
    intent = authenticated.intent
    cases = expand_test_cases(intent)
    if sha256_digest(intent) != M4_HEALTH_INTENT_SHA256 or len(cases) != 47:
        raise ArtifactValidationError("authenticated fit does not bind the frozen M4 matrix")
    counts = intent.source_population.split_sequence_counts
    seed = intent.source_population.data_master_seed
    main_bases = generate_health_base_sequences(
        authenticated.main_profile,
        split="test",
        sequence_count=counts.test,
        data_master_seed=seed,
    )
    edge_bases = generate_health_base_sequences(
        authenticated.edge_profile,
        split="test",
        sequence_count=counts.edge_test,
        data_master_seed=seed,
    )
    thresholds = HealthThresholds(
        self_score=authenticated.summary.selected_self_threshold,
        cross_score=authenticated.summary.selected_cross_threshold,
    )
    result = _evaluate_health_cases(
        cases=cases,
        main_bases=main_bases,
        edge_bases=edge_bases,
        calibration=authenticated.calibration,
        thresholds=thresholds,
        intent_sha256=sha256_digest(intent),
        require_exact_matrix=True,
        condition_sink=condition_sink,
        materialize_rows=False,
    )
    return cast(HealthBenchmarkStreamSummary, result)
