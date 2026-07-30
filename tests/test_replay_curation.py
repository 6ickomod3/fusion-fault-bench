from __future__ import annotations

from copy import copy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest

import fusion_fault_bench.replay_curation as replay_curation
from fusion_fault_bench.artifacts import derive_run_id
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_FIT_RUN_SHA256,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_REPLAY_RELEASE_ID,
    M5_TRACKED_AGGREGATE_TERMS,
    REPLAY_CURATED_ARTIFACT_CONTRACT,
    ReplayDescriptorAggregateV1,
    ReplayExecutionResourceEvidenceV1,
    ReplayProfileSummaryV1,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_FIT_SHA256,
    M5_PERSISTENT_MATRIX_SHA256,
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_SHA256,
    M5_SCENE_NAMES,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.replay_curation import (
    ReplayLogGroupBinding,
    _curate_crossovers,
    _curate_replay_population_evidence,
    _group_health_rows,
    _health_record,
    _persistent_record,
    _require_release_interval,
    _validate_context,
    _validate_plan,
    aggregate_replay_health_evidence,
    aggregate_replay_persistent_evidence,
    assemble_replay_curated_write_request,
    curate_replay_evidence,
    expected_replay_health_coordinates,
    expected_replay_persistent_coordinates,
)
from fusion_fault_bench.replay_health_population import ReplayHealthPopulationMetric
from fusion_fault_bench.replay_inference import (
    H5_B_SELECTORS,
    ReplayHealthSequenceContrast,
    ReplayInterval,
)
from fusion_fault_bench.replay_persistent_inference import (
    M5_A_DIRECTIONAL_EXPECTATIONS,
    ReplayPersistentCrossoverEstimate,
    ReplayPersistentPopulationMetric,
)
from fusion_fault_bench.replay_plan import LoadedReplayPlan, load_replay_plan
from fusion_fault_bench.replay_resources import (
    M5_PUBLIC_REPLAY_COMMAND,
    replay_environment_sha256,
    replay_logical_command_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
_PLAN = load_replay_plan(source_root=ROOT)
_GROUPS = ("log-group:00",) * 5 + ("log-group:01",) * 5
_EXPECTED_CROSSOVERS = (
    ("replay-lidar-y-bias", "negative", "m", 4.0),
    ("replay-lidar-y-bias", "positive", "m", 4.0),
    ("replay-camera-noise-correctly-reported", "increase", "std-scale", 4.0),
    ("replay-camera-noise-underreported", "increase", "std-scale", 4.0),
    ("replay-camera-calibration-x", "negative", "m", 4.0),
    ("replay-camera-calibration-x", "positive", "m", 4.0),
    ("replay-camera-calibration-yaw", "negative", "rad", 0.08),
    ("replay-camera-calibration-yaw", "positive", "rad", 0.08),
    ("replay-camera-timestamp-offset", "negative", "s", 0.8),
    ("replay-camera-timestamp-offset", "positive", "s", 0.8),
)


def _run() -> RunRecordV1Alpha1:
    run_id = derive_run_id(
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        git_revision="1" * 40,
        lockfile_sha256="2" * 64,
        package_version="0.1.0",
        artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
    )
    started = datetime(2026, 1, 2, tzinfo=UTC)
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        package_version="0.1.0",
        git_revision="1" * 40,
        source_dirty=False,
        lockfile_sha256="2" * 64,
        command=(
            "ffb",
            "replay",
            "run",
            "--output-dir",
            "reports/generated/m5-primary",
        ),
        environment=RuntimeEnvironment(
            python_version="3.12.13",
            os_name="test-os",
            os_release="test-release",
            machine="test-machine",
            cpu_model="Test CPU",
            logical_cpu_count=4,
            memory_bytes=8 * 1024**3,
        ),
        started_at=started,
        ended_at=started + timedelta(seconds=10),
        status="succeeded",
        artifact_sha256="0" * 64,
    )


def _resource_evidence(
    run: RunRecordV1Alpha1,
) -> tuple[
    ReplayExecutionResourceEvidenceV1,
    ReplayExecutionResourceEvidenceV1,
]:
    def resource(
        run_label: str,
        *,
        elapsed_seconds: float,
        peak_rss_bytes: int,
        run_sha256: str,
    ) -> ReplayExecutionResourceEvidenceV1:
        return ReplayExecutionResourceEvidenceV1(
            schema="ffb.replay-execution-resource-evidence/v1",
            run_id=run.run_id,
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            run_label=run_label,  # type: ignore[arg-type]
            local_artifact_sha256="5" * 64,
            local_run_sha256=run_sha256,
            environment_sha256=replay_environment_sha256(run.environment),
            logical_command_sha256=replay_logical_command_sha256(tuple(run.command)),
            persisted_internal_elapsed_seconds=elapsed_seconds - 1.0,
            persisted_internal_peak_rss_bytes=peak_rss_bytes - 1,
            persisted_internal_measurement_scope=(
                "metadata-through-canonical-scientific-members-before-publication"
            ),
            tool_path="/usr/bin/time",
            tool_options=("-l",),
            parser_contract="ffb.darwin-time-l-strict/v1",
            raw_log_sha256=("8" if run_label == "primary" else "9") * 64,
            raw_log_byte_length=777,
            elapsed_seconds=elapsed_seconds,
            peak_rss_bytes=peak_rss_bytes,
            exit_status=0,
            scientific_replay_worker_count=1,
            cpu_process_scope=("one-scientific-replay-worker-no-benchmark-multiprocessing"),
            helper_process_policy=("sequential-provenance-and-resource-measurement-helpers-only"),
            accelerator_requested=False,
            wall_time_cap_seconds=1800.0,
            peak_rss_cap_bytes=1_073_741_824,
            wall_time_within_cap=True,
            peak_rss_within_cap=True,
            measurement_scope=(
                "operator-recorded-darwin-time-l-for-complete-replay-cli-lifetime;"
                "self-reported-not-independent-attestation"
            ),
        )

    return (
        resource(
            "primary",
            elapsed_seconds=9.0,
            peak_rss_bytes=200 * 1024**2,
            run_sha256="3" * 64,
        ),
        resource(
            "repeat",
            elapsed_seconds=10.0,
            peak_rss_bytes=256 * 1024**2,
            run_sha256="4" * 64,
        ),
    )


def _profile(run: RunRecordV1Alpha1) -> ReplayProfileSummaryV1:
    resources = _resource_evidence(run)
    return ReplayProfileSummaryV1(
        schema="ffb.replay-profile-summary/v1",
        run_id=run.run_id,
        replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
        replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
        release_id=M5_REPLAY_RELEASE_ID,
        replay_intent_byte_sha256=M5_REPLAY_INTENT_BYTE_SHA256,
        persistent_source_matrix_sha256=M5_PERSISTENT_MATRIX_SHA256,
        health_fit_artifact_sha256=M5_HEALTH_FIT_SHA256,
        health_fit_run_sha256=M5_HEALTH_FIT_RUN_SHA256,
        dataset_profile="official-nuscenes-v1.0-mini",
        adapter_profile="nuscenes-mini-matched-centers-v1",
        scene_count=10,
        persistent_experiment_count=8,
        health_experiment_count=14,
        replay_experiment_count=22,
        distinct_log_group_count=2,
        all_scenes_have_base_support=True,
        all_health_schedules_valid=True,
        raw_sensor_payload_reads=0,
        scientific_replay_worker_count=1,
        gpu_used=False,
        torch_imported=False,
        cuda_used=False,
        resource_evidence=resources,
        peak_rss_bytes=256 * 1024**2,
        elapsed_seconds=10.0,
        dataset_root_serialized=False,
        dataset_bytes_authenticated=False,
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        attribution_and_non_endorsement_required=True,
    )


def _descriptors(run: RunRecordV1Alpha1) -> tuple[ReplayDescriptorAggregateV1, ...]:
    return (
        ReplayDescriptorAggregateV1(
            schema="ffb.replay-descriptor-aggregate/v1",
            run_id=run.run_id,
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            descriptor_id="frame-count",
            population="nuscenes-mini-replay",
            population_count=10,
            statistic="median",
            status="ok",
            value=40.0,
            unit="frames",
            tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        ),
    )


@cache
def _persistent_metrics(
    plan: LoadedReplayPlan = _PLAN,
) -> tuple[ReplayPersistentPopulationMetric, ...]:
    cases = {case.fault_condition.selector: case for case in plan.persistent_cases}
    output: list[ReplayPersistentPopulationMetric] = []
    for selector, method_id, metric_id, aggregation in sorted(
        expected_replay_persistent_coordinates(plan)
    ):
        case = cases[selector]
        expected = M5_A_DIRECTIONAL_EXPECTATIONS.get(selector)
        value = -1.0 if expected == "negative" else 1.0
        if metric_id in {"coverage", "undefined-output-rate", "scene-equal-coverage"}:
            value = 0.5
        undefined = (
            selector == "replay-camera-dropout:1"
            and method_id == "fixed-fusion"
            and metric_id == "conditional-matched-center-mse"
        )
        if undefined:
            interval = ReplayInterval(
                estimate=None,
                lower=None,
                upper=None,
                defined_replicates=0,
                bootstrap_replicates=2_000,
            )
            numerators = (0.0,) * 10
            denominators = (0,) * 10
        else:
            interval = ReplayInterval(
                estimate=value,
                lower=value - 0.25,
                upper=value + 0.25,
                defined_replicates=2_000,
                bootstrap_replicates=2_000,
            )
            numerators = (value,) * 10
            denominators = (1,) * 10
        output.append(
            ReplayPersistentPopulationMetric(
                replay_experiment_identity_sha256=case.identity_sha256,
                condition_id=case.identity.experiment_id,
                condition_selector=selector,
                method_id=method_id,
                metric_id=metric_id,  # type: ignore[arg-type]
                aggregation=aggregation,  # type: ignore[arg-type]
                interval=interval,
                scene_numerators=numerators,
                scene_denominators=denominators,
            )
        )
    return tuple(output)


@cache
def _health_metrics(
    plan: LoadedReplayPlan = _PLAN,
) -> tuple[ReplayHealthPopulationMetric, ...]:
    selector_map = {row.selector: row for row in H5_B_SELECTORS}
    cases = {case.selector: case for case in plan.health_cases}
    required = expected_replay_health_coordinates(plan)
    support_incompatible_recovery = {
        coordinate
        for coordinate in required
        if coordinate[2] == "frame-oracle-recoverable-loss-fraction"
        and cases[coordinate[0]].family == "dropout"
        and coordinate[3] in {"score", "event"}
    }
    assert len(support_incompatible_recovery) == 60
    output: list[ReplayHealthPopulationMetric] = []
    for coordinate in sorted(required):
        selector, method_id, metric_id, window, unit, aggregation = coordinate
        case = cases[selector]
        selected = selector_map.get(selector)
        selected_coordinate = (
            selected is not None
            and method_id == selected.method
            and metric_id == selected.metric_name
            and window == selected.window
            and unit == selected.unit
            and aggregation == "equal-scene-mean"
        )
        value = 0.5 if unit == "fraction" else 1.0
        if selected_coordinate:
            assert selected is not None
            value = -1.0 if selected.expected_direction == "negative" else 1.0
            if selected.assessment_rule == "nonpositive-control":
                value = -1.0
        not_applicable = (
            case.family == "common-mode-position-bias"
            and metric_id
            in {
                "gap-vs-fault-target-drop",
                "gap-vs-frame-oracle",
                "frame-oracle-recoverable-loss-fraction",
            }
        ) or coordinate in support_incompatible_recovery
        interval = (
            ReplayInterval(
                estimate=None,
                lower=None,
                upper=None,
                defined_replicates=0,
                bootstrap_replicates=2_000,
            )
            if not_applicable
            else ReplayInterval(
                estimate=value,
                lower=value - 0.25,
                upper=value + 0.25,
                defined_replicates=2_000,
                bootstrap_replicates=2_000,
            )
        )
        output.append(
            ReplayHealthPopulationMetric(
                replay_experiment_identity_sha256=case.identity_sha256,
                condition_id=case.identity.experiment_id,
                condition_selector=case.selector,
                method_id=method_id,
                metric_id=metric_id,
                window=window,  # type: ignore[arg-type]
                unit=unit,  # type: ignore[arg-type]
                aggregation=aggregation,  # type: ignore[arg-type]
                status="not-applicable" if not_applicable else "ok",
                interval=interval,
                scene_numerators=((0.0,) * 10 if not_applicable else (value,) * 10),
                scene_denominators=((0.0,) * 10 if not_applicable else (1.0,) * 10),
                scene_defined=((False,) * 10 if not_applicable else (True,) * 10),
            )
        )
    return tuple(output)


@cache
def _health_contrasts(
    plan: LoadedReplayPlan = _PLAN,
) -> tuple[ReplayHealthSequenceContrast, ...]:
    support_sha256 = "c" * 64
    mismatched_sha256 = "d" * 64
    policies = (
        "self-nis-gate",
        "cross-nis-gate",
        "direct-telemetry-gate",
        "combined-health-gate",
        "combined-health-gate-abstain",
    )
    windows = ("score", "event", "recovery")
    output: list[ReplayHealthSequenceContrast] = []
    for case in plan.health_cases:
        applicable = case.family != "common-mode-position-bias"
        for sequence_id in (f"nuscenes:{name}" for name in M5_SCENE_NAMES):
            for policy in policies:
                for window in windows:
                    oracle_support_sha256 = (
                        mismatched_sha256
                        if case.family == "dropout" and window in {"score", "event"}
                        else support_sha256
                    )
                    output.append(
                        ReplayHealthSequenceContrast(
                            replay_experiment_identity_sha256=case.identity_sha256,
                            sequence_id=sequence_id,
                            condition_id=case.identity.experiment_id,
                            condition_selector=case.selector,
                            policy=policy,  # type: ignore[arg-type]
                            window=window,  # type: ignore[arg-type]
                            fixed_support_sha256=support_sha256,
                            policy_support_sha256=support_sha256,
                            fixed_policy_common_count=1,
                            fixed_on_common_loss_sum_m2=2.0,
                            policy_on_fixed_common_loss_sum_m2=1.0,
                            target_drop_applicable=applicable,
                            policy_target_drop_common_count=1 if applicable else None,
                            policy_on_target_common_loss_sum_m2=1.0 if applicable else None,
                            target_drop_on_common_loss_sum_m2=0.5 if applicable else None,
                            target_drop_support_sha256=(support_sha256 if applicable else None),
                            frame_oracle_applicable=applicable,
                            policy_frame_oracle_common_count=1 if applicable else None,
                            policy_on_oracle_common_loss_sum_m2=1.0 if applicable else None,
                            frame_oracle_on_common_loss_sum_m2=0.25 if applicable else None,
                            frame_oracle_support_sha256=(
                                oracle_support_sha256 if applicable else None
                            ),
                        )
                    )
    return tuple(output)


@cache
def _crossovers(
    plan: LoadedReplayPlan = _PLAN,
) -> tuple[ReplayPersistentCrossoverEstimate, ...]:
    identities = {
        case.identity.experiment_id: case.identity_sha256 for case in plan.persistent_cases
    }
    return tuple(
        ReplayPersistentCrossoverEstimate(
            replay_experiment_identity_sha256=identities[condition_id],
            condition_id=condition_id,
            direction=direction,  # type: ignore[arg-type]
            severity_unit=unit,
            tested_maximum=tested_maximum,
            status="not-observed",
            point_estimate=None,
            interval_lower=tested_maximum,
            interval_upper="positive-infinity",
            bootstrap_crossing_count=0,
            bootstrap_replicates=2_000,
        )
        for condition_id, direction, unit, tested_maximum in _EXPECTED_CROSSOVERS
    )


def _curate(
    *,
    persistent_metrics: tuple[ReplayPersistentPopulationMetric, ...] | None = None,
    health_metrics: tuple[ReplayHealthPopulationMetric, ...] | None = None,
    groups: tuple[str, ...] = _GROUPS,
):
    run = _run()
    return _curate_replay_population_evidence(
        plan=_PLAN,
        persistent_metrics=(
            _persistent_metrics() if persistent_metrics is None else persistent_metrics
        ),
        persistent_crossovers=_crossovers(),
        health_metrics=_health_metrics() if health_metrics is None else health_metrics,
        health_contrasts=_health_contrasts(),
        descriptor_aggregates=_descriptors(run),
        log_group_ordinals=groups,
        profile_summary=_profile(run),
        run=run,
    )


def _unsafe_plan_copy(**updates: object) -> LoadedReplayPlan:
    plan = copy(_PLAN)
    for field, value in updates.items():
        object.__setattr__(plan, field, value)
    return plan


def _log_group_bindings() -> tuple[ReplayLogGroupBinding, ...]:
    return tuple(
        ReplayLogGroupBinding(
            sequence_id=f"nuscenes:{scene_name}",
            log_group_ordinal=group,
        )
        for scene_name, group in zip(M5_SCENE_NAMES, _GROUPS, strict=True)
    )


def test_curation_builds_exact_claim_roles_sensitivity_and_stable_ids() -> None:
    first = _curate()
    second = _curate(
        persistent_metrics=tuple(reversed(_persistent_metrics())),
        health_metrics=tuple(reversed(_health_metrics())),
    )

    assert tuple(first.run.command) == M5_PUBLIC_REPLAY_COMMAND
    assert len(first.persistent_aggregates) == 464
    assert len(first.health_aggregates) == 14_988
    not_applicable = tuple(row for row in first.health_aggregates if row.status == "not-applicable")
    assert len(not_applicable) == 240
    assert sum(row.applicability_basis == "structural-unavailable" for row in not_applicable) == 180
    support_incompatible = tuple(
        row for row in not_applicable if row.applicability_basis == "support-incompatible"
    )
    assert len(support_incompatible) == 60
    assert all(row.recovery_support_compatible_scene_count == 0 for row in support_incompatible)
    assert all(row.inference_role == "descriptive" for row in not_applicable)
    assert all(
        row.positive_scene_count is None
        and row.zero_scene_count is None
        and row.negative_scene_count is None
        and row.undefined_scene_count is None
        for row in not_applicable
    )
    assert len(first.persistent_crossovers) == 10
    persistent_claims = tuple(
        row for row in first.persistent_aggregates if row.hypothesis_id is not None
    )
    health_claims = tuple(row for row in first.health_aggregates if row.hypothesis_id is not None)
    assert len(persistent_claims) == 33
    assert sum(row.hypothesis_id == "h5-a5" for row in persistent_claims) == 4
    assert sum(row.hypothesis_id == "h5-a6" for row in persistent_claims) == 13
    a5_conditional = next(
        row
        for row in persistent_claims
        if row.hypothesis_id == "h5-a5"
        and row.method_id == "fixed-fusion"
        and row.metric_id == "conditional-matched-center-mse"
    )
    assert a5_conditional.status == "undefined"
    assert all(
        row.inference_role == "diagnostic"
        for row in persistent_claims
        if row.hypothesis_id in {"h5-a5", "h5-a6"}
    )
    assert len(health_claims) == 11
    assert sum(row.inference_role == "primary-directional" for row in health_claims) == 8
    assert sum(row.inference_role == "nonpositive-control" for row in health_claims) == 2
    assert sum(row.inference_role == "diagnostic" for row in health_claims) == 1
    assert len(first.cluster_sensitivity) == (16 + 8 + 2) * (10 + 2)
    assert all(
        row.cluster_id.startswith(("scene-ordinal:", "log-group:"))
        for row in first.cluster_sensitivity
    )
    assert tuple(row.result_id for row in first.persistent_aggregates) == tuple(
        row.result_id for row in second.persistent_aggregates
    )
    assert tuple(row.result_id for row in first.health_aggregates) == tuple(
        row.result_id for row in second.health_aggregates
    )
    assert (
        len({row.result_id for row in (*first.persistent_aggregates, *first.health_aggregates)})
        == 15_452
    )
    source_by_id = {
        row.result_id: row for row in (*first.persistent_aggregates, *first.health_aggregates)
    }
    assert all(
        row.source_record_sha256 == sha256_digest(source_by_id[row.source_result_id])
        for row in first.cluster_sensitivity
    )


def test_curation_rejects_incomplete_matrices_and_hypothesis_reassignment() -> None:
    with pytest.raises(ValueError, match="464-row matrix"):
        _curate(persistent_metrics=_persistent_metrics()[:-1])

    health = list(_health_metrics())
    selected_index = next(
        index
        for index, row in enumerate(health)
        if row.condition_selector == H5_B_SELECTORS[0].selector
        and row.method_id == H5_B_SELECTORS[0].method
        and row.metric_id == H5_B_SELECTORS[0].metric_name
        and row.window == H5_B_SELECTORS[0].window
    )
    health[selected_index] = replace(health[selected_index], method_id="fixed-fusion")
    with pytest.raises(ValueError, match="14,988-row matrix"):
        _curate(health_metrics=tuple(health))


def test_curation_rejects_identity_rebinding_and_noncontiguous_group_ordinals() -> None:
    persistent = list(_persistent_metrics())
    persistent[0] = replace(
        persistent[0],
        replay_experiment_identity_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="replay identity"):
        _curate(persistent_metrics=tuple(persistent))

    with pytest.raises(ValueError, match="log-group ordinals"):
        _curate(groups=("log-group:00",) * 5 + ("log-group:02",) * 5)


def test_raw_health_curation_boundary_requires_all_three_exact_43_selector_grids() -> None:
    with pytest.raises(ValueError, match="exact 43-selector"):
        aggregate_replay_health_evidence(
            plan=_PLAN,
            health_results=(),
            health_contrasts=(),
            health_events=(),
        )


def test_public_curation_requires_keyed_scene_order_and_raw_persistent_evidence() -> None:
    run = _run()
    reversed_bindings = tuple(
        ReplayLogGroupBinding(
            sequence_id=f"nuscenes:{scene_name}",
            log_group_ordinal=group,
        )
        for scene_name, group in reversed(tuple(zip(M5_SCENE_NAMES, _GROUPS, strict=True)))
    )
    with pytest.raises(ValueError, match="frozen replay scene order"):
        curate_replay_evidence(
            plan=_PLAN,
            persistent_scene_evaluations=(),
            persistent_metrics=(),
            persistent_crossovers=(),
            health_results=(),
            health_contrasts=(),
            health_events=(),
            descriptor_aggregates=_descriptors(run),
            log_group_bindings=reversed_bindings,
            profile_summary=_profile(run),
            run=run,
        )
    with pytest.raises(ValueError, match="exact 71 selectors"):
        aggregate_replay_persistent_evidence(
            plan=_PLAN,
            scene_evaluations=(),
        )


def test_frozen_plan_authority_rejects_intent_and_selector_grid_tampering() -> None:
    invalid_intent = replace(_PLAN.intent, intent_sha256="0" * 64)
    with pytest.raises(ValueError, match="authenticated frozen M5 intent"):
        _validate_plan(replace(_PLAN, intent=invalid_intent))

    with pytest.raises(ValueError, match="incomplete M5-A selector grid"):
        _validate_plan(_unsafe_plan_copy(persistent_cases=_PLAN.persistent_cases[:-1]))
    with pytest.raises(ValueError, match="incomplete M5-B selector grid"):
        _validate_plan(_unsafe_plan_copy(health_cases=_PLAN.health_cases[:-1]))


def test_curation_context_rejects_run_profile_group_and_descriptor_rebinding() -> None:
    run = _run()
    profile = _profile(run)
    descriptors = _descriptors(run)
    invalid_runs = (
        run.model_copy(update={"manifest_sha256": "0" * 64}),
        run.model_copy(update={"source_dirty": True}),
        run.model_copy(update={"status": "failed"}),
        run.model_copy(update={"artifact_sha256": "1" * 64}),
        run.model_copy(update={"run_id": "f" * 64}),
    )
    for invalid in invalid_runs:
        with pytest.raises(ValueError, match="clean successful frozen-intent run"):
            _validate_context(
                profile_summary=profile,
                descriptor_aggregates=descriptors,
                log_group_ordinals=_GROUPS,
                run=invalid,
            )

    with pytest.raises(ValueError, match="profile summary does not bind"):
        _validate_context(
            profile_summary=profile.model_copy(update={"run_id": "f" * 64}),
            descriptor_aggregates=descriptors,
            log_group_ordinals=_GROUPS,
            run=run,
        )
    with pytest.raises(ValueError, match="one log-group ordinal"):
        _validate_context(
            profile_summary=profile,
            descriptor_aggregates=descriptors,
            log_group_ordinals=_GROUPS[:-1],
            run=run,
        )
    with pytest.raises(ValueError, match="contiguous or disagree"):
        _validate_context(
            profile_summary=profile.model_copy(update={"distinct_log_group_count": 3}),
            descriptor_aggregates=descriptors,
            log_group_ordinals=_GROUPS,
            run=run,
        )
    with pytest.raises(ValueError, match="requires source descriptor"):
        _validate_context(
            profile_summary=profile,
            descriptor_aggregates=(),
            log_group_ordinals=_GROUPS,
            run=run,
        )
    with pytest.raises(ValueError, match="descriptor aggregate does not bind"):
        _validate_context(
            profile_summary=profile,
            descriptor_aggregates=(descriptors[0].model_copy(update={"run_id": "f" * 64}),),
            log_group_ordinals=_GROUPS,
            run=run,
        )
    with pytest.raises(ValueError, match="duplicate coordinate"):
        _validate_context(
            profile_summary=profile,
            descriptor_aggregates=(descriptors[0], descriptors[0]),
            log_group_ordinals=_GROUPS,
            run=run,
        )

    later = descriptors[0].model_copy(update={"descriptor_id": "z-frame-count"})
    ordered, groups = _validate_context(
        profile_summary=profile,
        descriptor_aggregates=(later, descriptors[0]),
        log_group_ordinals=_GROUPS,
        run=run,
    )
    assert tuple(row.descriptor_id for row in ordered) == (
        "frame-count",
        "z-frame-count",
    )
    assert groups == _GROUPS


def test_health_grouping_rejects_unknown_and_incomplete_selector_sets() -> None:
    with pytest.raises(ValueError, match="unknown replay selector"):
        _group_health_rows(
            (SimpleNamespace(condition_selector="unknown-selector"),),
            expected_selectors={"known"},
            label="test rows",
        )
    with pytest.raises(ValueError, match="exact 43-selector"):
        _group_health_rows(
            (SimpleNamespace(condition_selector="first"),),
            expected_selectors={"first", "second"},
            label="test rows",
        )
    grouped = _group_health_rows(
        (
            SimpleNamespace(condition_selector="second"),
            SimpleNamespace(condition_selector="first"),
        ),
        expected_selectors={"first", "second"},
        label="test rows",
    )
    assert set(grouped) == {"first", "second"}


def test_health_aggregation_orchestrates_each_frozen_selector_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selectors = tuple(case.selector for case in _PLAN.health_cases)
    source_rows = tuple(SimpleNamespace(condition_selector=value) for value in selectors)
    calls: list[tuple[str, str]] = []
    validated: list[tuple[object, ...]] = []

    def aggregate(kind: str):
        def fake(
            case: SimpleNamespace,
            rows: tuple[object, ...],
        ) -> tuple[object, ...]:
            selector = case.selector
            assert len(rows) == 1
            calls.append((kind, selector))
            return (SimpleNamespace(kind=kind, condition_selector=selector),)

        return fake

    def validate(
        plan: LoadedReplayPlan,
        rows: tuple[object, ...],
        *,
        contrasts: tuple[object, ...],
    ) -> None:
        assert plan is _PLAN
        assert contrasts is source_rows
        validated.append(rows)

    monkeypatch.setattr(
        replay_curation,
        "aggregate_replay_health_results",
        aggregate("result"),
    )
    monkeypatch.setattr(
        replay_curation,
        "aggregate_replay_health_contrasts",
        aggregate("contrast"),
    )
    monkeypatch.setattr(
        replay_curation,
        "aggregate_replay_health_events",
        aggregate("event"),
    )
    monkeypatch.setattr(
        replay_curation,
        "validate_replay_health_population_grid",
        validate,
    )

    rows = aggregate_replay_health_evidence(
        plan=_PLAN,
        health_results=source_rows,  # type: ignore[arg-type]
        health_contrasts=source_rows,  # type: ignore[arg-type]
        health_events=source_rows,  # type: ignore[arg-type]
    )

    assert len(rows) == 3 * len(selectors)
    assert len(calls) == 3 * len(selectors)
    assert len(validated) == 1


def test_persistent_aggregation_rejects_unknown_selector_and_preserves_case_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="unknown selector"):
        aggregate_replay_persistent_evidence(
            plan=_PLAN,
            scene_evaluations=(SimpleNamespace(condition_selector="unknown-selector"),),  # type: ignore[arg-type]
        )

    source_rows = tuple(
        SimpleNamespace(condition_selector=case.fault_condition.selector)
        for case in _PLAN.persistent_cases
    )
    aggregated_selectors: list[str] = []
    crossover_experiments: list[str] = []

    def aggregate_case(
        case: SimpleNamespace,
        rows: tuple[object, ...],
    ) -> tuple[object, ...]:
        selector = case.fault_condition.selector
        assert len(rows) == 1
        aggregated_selectors.append(selector)
        return (
            SimpleNamespace(
                condition_id=case.identity.experiment_id,
                condition_selector=selector,
            ),
        )

    def evaluate(
        cases: tuple[SimpleNamespace, ...],
        metrics: tuple[SimpleNamespace, ...],
    ) -> tuple[object, ...]:
        experiment_id = cases[0].identity.experiment_id
        assert all(row.condition_id == experiment_id for row in metrics)
        crossover_experiments.append(experiment_id)
        return (SimpleNamespace(condition_id=experiment_id),)

    monkeypatch.setattr(
        replay_curation,
        "aggregate_replay_persistent_case",
        aggregate_case,
    )
    monkeypatch.setattr(
        replay_curation,
        "evaluate_replay_persistent_crossovers",
        evaluate,
    )

    metrics, crossovers = aggregate_replay_persistent_evidence(
        plan=_PLAN,
        scene_evaluations=source_rows,  # type: ignore[arg-type]
    )

    assert tuple(aggregated_selectors) == tuple(
        case.fault_condition.selector for case in _PLAN.persistent_cases
    )
    assert len(metrics) == len(_PLAN.persistent_cases)
    assert len(crossovers) == len(crossover_experiments)
    assert len(crossover_experiments) == len(set(crossover_experiments))


def test_record_curation_rejects_identity_bootstrap_and_recovery_support_tampering() -> None:
    persistent = _persistent_metrics()[0]
    persistent_case = next(
        case
        for case in _PLAN.persistent_cases
        if case.fault_condition.selector == persistent.condition_selector
    )
    with pytest.raises(ValueError, match="disagrees with its replay identity"):
        _persistent_record(
            run_id=_run().run_id,
            case_identity=persistent_case.identity,
            metric=replace(
                persistent,
                replay_experiment_identity_sha256="f" * 64,
            ),
            log_group_ordinals=_GROUPS,
        )
    with pytest.raises(ValueError, match="2,000-replicate bootstrap"):
        _persistent_record(
            run_id=_run().run_id,
            case_identity=persistent_case.identity,
            metric=replace(
                persistent,
                interval=replace(
                    persistent.interval,
                    defined_replicates=1_000,
                    bootstrap_replicates=1_000,
                ),
            ),
            log_group_ordinals=_GROUPS,
        )

    recovery = next(
        row
        for row in _health_metrics()
        if row.metric_id == "frame-oracle-recoverable-loss-fraction" and row.status == "ok"
    )
    health_case = next(
        case for case in _PLAN.health_cases if case.selector == recovery.condition_selector
    )
    with pytest.raises(ValueError, match="health population metric disagrees"):
        _health_record(
            run_id=_run().run_id,
            case_identity=health_case.identity,
            metric=replace(
                recovery,
                replay_experiment_identity_sha256="f" * 64,
            ),
            recovery_support_compatible_scene_count=10,
            log_group_ordinals=_GROUPS,
        )
    with pytest.raises(ValueError, match="lacks all-scene support evidence"):
        _health_record(
            run_id=_run().run_id,
            case_identity=health_case.identity,
            metric=recovery,
            recovery_support_compatible_scene_count=None,
            log_group_ordinals=_GROUPS,
        )
    with pytest.raises(ValueError, match="2,000-replicate bootstrap"):
        _require_release_interval(
            replace(
                recovery.interval,
                defined_replicates=1_000,
                bootstrap_replicates=1_000,
            )
        )


def test_crossover_curation_rejects_grid_binding_status_and_censoring_tampering() -> None:
    rows = _crossovers()
    run_id = _run().run_id

    with pytest.raises(ValueError, match="duplicate coordinate"):
        _curate_crossovers(
            plan=_PLAN,
            crossovers=(*rows, rows[0]),
            run_id=run_id,
        )
    with pytest.raises(ValueError, match="exact ten-coordinate"):
        _curate_crossovers(
            plan=_PLAN,
            crossovers=rows[:-1],
            run_id=run_id,
        )

    invalid_bindings = (
        replace(rows[0], replay_experiment_identity_sha256="f" * 64),
        replace(rows[0], bootstrap_replicates=1_000),
        replace(rows[0], tested_maximum=8.0, interval_lower=8.0),
    )
    for invalid in invalid_bindings:
        with pytest.raises(ValueError, match="does not bind frozen replay inference"):
            _curate_crossovers(
                plan=_PLAN,
                crossovers=(invalid, *rows[1:]),
                run_id=run_id,
            )

    with pytest.raises(ValueError, match="status contradicts"):
        _curate_crossovers(
            plan=_PLAN,
            crossovers=(replace(rows[0], status="observed"), *rows[1:]),
            run_id=run_id,
        )
    with pytest.raises(ValueError, match="finite interval"):
        _curate_crossovers(
            plan=_PLAN,
            crossovers=(
                replace(
                    rows[0],
                    status="observed",
                    point_estimate=1.0,
                    interval_lower=None,
                    interval_upper=None,
                    bootstrap_crossing_count=2_000,
                ),
                *rows[1:],
            ),
            run_id=run_id,
        )
    with pytest.raises(ValueError, match="exact right censoring"):
        _curate_crossovers(
            plan=_PLAN,
            crossovers=(replace(rows[0], interval_lower=0.0), *rows[1:]),
            run_id=run_id,
        )
    with pytest.raises(ValueError, match="cannot carry an interval"):
        _curate_crossovers(
            plan=_PLAN,
            crossovers=(
                replace(
                    rows[0],
                    status="undetermined",
                    point_estimate=1.0,
                    interval_lower=0.5,
                    interval_upper=None,
                    bootstrap_crossing_count=100,
                ),
                *rows[1:],
            ),
            run_id=run_id,
        )


def test_crossover_curation_emits_observed_and_undetermined_semantics() -> None:
    rows = _crossovers()
    observed = replace(
        rows[0],
        status="observed",
        point_estimate=1.0,
        interval_lower=0.5,
        interval_upper=1.5,
        bootstrap_crossing_count=2_000,
    )
    undetermined = replace(
        rows[1],
        status="undetermined",
        point_estimate=1.0,
        interval_lower=None,
        interval_upper=None,
        bootstrap_crossing_count=100,
    )
    curated = _curate_crossovers(
        plan=_PLAN,
        crossovers=(observed, undetermined, *rows[2:]),
        run_id=_run().run_id,
    )

    assert curated[0].status == "observed"
    assert curated[0].censoring == "none"
    assert curated[1].status == "undetermined"
    assert curated[1].censoring == "mixed-bootstrap"
    assert curated[2].status == "not-observed"
    assert curated[2].censoring == "right-above-tested-maximum"


def test_public_curation_compares_recomputed_population_evidence_before_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recomputed_metrics = _persistent_metrics()
    recomputed_crossovers = _crossovers()
    delegated: list[dict[str, object]] = []
    sentinel = object()

    def aggregate_persistent(**_: object):
        return recomputed_metrics, recomputed_crossovers

    def aggregate_health(**_: object):
        return ()

    def curate_population(**kwargs: object):
        delegated.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        replay_curation,
        "aggregate_replay_persistent_evidence",
        aggregate_persistent,
    )
    monkeypatch.setattr(
        replay_curation,
        "aggregate_replay_health_evidence",
        aggregate_health,
    )
    monkeypatch.setattr(
        replay_curation,
        "_curate_replay_population_evidence",
        curate_population,
    )

    run = _run()
    common = {
        "plan": _PLAN,
        "persistent_scene_evaluations": (),
        "persistent_crossovers": recomputed_crossovers,
        "health_results": (),
        "health_contrasts": (),
        "health_events": (),
        "descriptor_aggregates": _descriptors(run),
        "log_group_bindings": _log_group_bindings(),
        "profile_summary": _profile(run),
        "run": run,
    }
    with pytest.raises(ValueError, match="differs from deterministic recomputation"):
        curate_replay_evidence(
            **common,
            persistent_metrics=recomputed_metrics[:-1],
        )

    reordered = {
        **common,
        "persistent_crossovers": tuple(reversed(recomputed_crossovers)),
    }
    result = curate_replay_evidence(
        **reordered,
        persistent_metrics=tuple(reversed(recomputed_metrics)),
    )
    assert result is sentinel
    assert len(delegated) == 1
    assert delegated[0]["persistent_metrics"] == recomputed_metrics
    assert delegated[0]["persistent_crossovers"] == recomputed_crossovers
    assert delegated[0]["log_group_ordinals"] == _GROUPS


def test_write_request_assembly_only_attaches_supplied_release_evidence() -> None:
    run = _run()
    evidence = SimpleNamespace(
        profile_summary=_profile(run),
        descriptor_aggregates=_descriptors(run),
        persistent_aggregates=(),
        persistent_crossovers=(),
        health_aggregates=(),
        cluster_sensitivity=(),
        run=run,
    )
    validation = SimpleNamespace(name="validation")
    repeat = SimpleNamespace(name="repeat")
    figure = SimpleNamespace(name="figure")
    commitment = SimpleNamespace(name="commitment")

    request = assemble_replay_curated_write_request(
        evidence,  # type: ignore[arg-type]
        validation=validation,  # type: ignore[arg-type]
        repeat_verification=repeat,  # type: ignore[arg-type]
        figures=[figure],  # type: ignore[list-item]
        source_commitments=[commitment],  # type: ignore[list-item]
    )

    assert request.validation is validation
    assert request.repeat_verification is repeat
    assert request.figures == (figure,)
    assert request.source_commitments == (commitment,)
