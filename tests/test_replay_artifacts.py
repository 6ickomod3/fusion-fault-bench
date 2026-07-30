from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path

import pytest

import fusion_fault_bench.replay_artifacts as replay_artifacts
from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    derive_run_id,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_HEALTH_FIT_RUN_SHA256,
    M5_HEALTH_HYPOTHESIS_COORDINATES,
    M5_PERSISTENT_HYPOTHESIS_COORDINATES,
    M5_RELEASE_VALIDATION_CHECK_IDS,
    M5_REPLAY_IDENTITY_SET_SHA256,
    M5_REPLAY_RELEASE_ID,
    M5_SCIENTIFIC_SOURCE_ROLES,
    M5_TRACKED_AGGREGATE_TERMS,
    REPLAY_ARTIFACT_PATHS,
    REPLAY_CURATED_ARTIFACT_CONTRACT,
    REPLAY_DESCRIPTOR_AGGREGATES_FILE,
    REPLAY_INTENT_FILE,
    REPLAY_PERSISTENT_AGGREGATES_FILE,
    REPLAY_RELEASE_INDEX_FILE,
    REPLAY_RUN_FILE,
    REPLAY_SUCCESS_FILE,
    ReplayClusterSensitivityV1,
    ReplayDescriptorAggregateV1,
    ReplayExecutionResourceEvidenceV1,
    ReplayFigureRecordV1,
    ReplayHealthAggregateV1,
    ReplayPersistentAggregateV1,
    ReplayPersistentCrossoverV1,
    ReplayProfileSummaryV1,
    ReplayRepeatVerificationV1,
    ReplaySourceMemberCommitmentV1,
    ReplayValidationCheckV1,
    ReplayValidationV1,
    replay_resource_evidence_sha256,
)
from fusion_fault_bench.contracts.replay_v1 import (
    M5_HEALTH_FIT_SHA256,
    M5_HEALTH_PANEL_ID,
    M5_PERSISTENT_MATRIX_SHA256,
    M5_PERSISTENT_PANEL_ID,
    M5_REPLAY_INTENT_BYTE_SHA256,
    M5_REPLAY_INTENT_SHA256,
    expected_replay_identities,
    replay_experiment_identity_sha256,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.replay_artifacts import (
    ReplayCuratedArtifactWriteRequest,
    canonical_replay_ndjson_bytes,
    load_replay_curated_artifact,
    validate_replay_candidate_bytes,
    write_replay_curated_artifact,
)
from fusion_fault_bench.replay_curation import (
    expected_replay_health_coordinates,
    expected_replay_persistent_coordinates,
)
from fusion_fault_bench.replay_plan import load_replay_plan
from fusion_fault_bench.replay_resources import (
    replay_environment_sha256,
    replay_logical_command_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
_DIGEST = "a" * 64
_PLAN = load_replay_plan(source_root=ROOT)
_IDENTITY_INDEX = {
    (identity.panel_id, identity.experiment_id): index
    for index, identity in enumerate(expected_replay_identities())
}
_HEALTH_HYPOTHESIS_BY_COORDINATE = {
    (
        coordinate[1],
        coordinate[2],
        coordinate[3],
        coordinate[4],
        coordinate[5],
        "equal-scene-mean",
    ): coordinate
    for coordinate in M5_HEALTH_HYPOTHESIS_COORDINATES
}
_PERSISTENT_HYPOTHESIS_BY_COORDINATE = {
    (coordinate[1], coordinate[2], coordinate[3], coordinate[6]): coordinate
    for coordinate in M5_PERSISTENT_HYPOTHESIS_COORDINATES
}
_PERSISTENT_CASE_BY_SELECTOR = {
    case.fault_condition.selector: case for case in _PLAN.persistent_cases
}
_HEALTH_CASE_BY_SELECTOR = {case.selector: case for case in _PLAN.health_cases}
_COMMON_MODE_STRUCTURAL_NA_METRICS = {
    "gap-vs-fault-target-drop",
    "gap-vs-frame-oracle",
    "frame-oracle-recoverable-loss-fraction",
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _stable_id(prefix: str, coordinate: dict[str, object]) -> str:
    return f"{prefix}-{sha256_digest(coordinate)}"


def _run() -> RunRecordV1Alpha1:
    git_revision = "1" * 40
    lockfile_sha256 = "2" * 64
    package_version = "0.1.0"
    run_id = derive_run_id(
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        git_revision=git_revision,
        lockfile_sha256=lockfile_sha256,
        package_version=package_version,
        artifact_contract=REPLAY_CURATED_ARTIFACT_CONTRACT,
    )
    started_at = datetime(2026, 1, 2, tzinfo=UTC)
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=M5_REPLAY_INTENT_SHA256,
        package_version=package_version,
        git_revision=git_revision,
        source_dirty=False,
        lockfile_sha256=lockfile_sha256,
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
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=10),
        status="succeeded",
        artifact_sha256="0" * 64,
    )


def _global(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "replay_intent_sha256": M5_REPLAY_INTENT_SHA256,
        "replay_identity_set_sha256": M5_REPLAY_IDENTITY_SET_SHA256,
    }


def _identity(identity_index: int, run_id: str) -> dict[str, object]:
    identity = expected_replay_identities()[identity_index]
    return {
        **_global(run_id),
        "identity": identity,
        "replay_identity_sha256": replay_experiment_identity_sha256(identity),
    }


def _resource_evidence(
    run: RunRecordV1Alpha1,
) -> tuple[
    ReplayExecutionResourceEvidenceV1,
    ReplayExecutionResourceEvidenceV1,
]:
    def resource(
        run_label: str,
        *,
        run_sha256: str,
        elapsed_seconds: float,
        peak_rss_bytes: int,
    ) -> ReplayExecutionResourceEvidenceV1:
        return ReplayExecutionResourceEvidenceV1(
            schema="ffb.replay-execution-resource-evidence/v1",
            **_global(run.run_id),
            run_label=run_label,  # type: ignore[arg-type]
            local_artifact_sha256=_digest("local-artifact"),
            local_run_sha256=run_sha256,
            environment_sha256=replay_environment_sha256(run.environment),
            logical_command_sha256=replay_logical_command_sha256(tuple(run.command)),
            persisted_internal_elapsed_seconds=elapsed_seconds - 10.0,
            persisted_internal_peak_rss_bytes=peak_rss_bytes - 1024,
            persisted_internal_measurement_scope=(
                "metadata-through-canonical-scientific-members-before-publication"
            ),
            tool_path="/usr/bin/time",
            tool_options=("-l",),
            parser_contract="ffb.darwin-time-l-strict/v1",
            raw_log_sha256=_digest(f"{run_label}-resource-log"),
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
            run_sha256=_digest("primary-run"),
            elapsed_seconds=90.0,
            peak_rss_bytes=200 * 1024**2,
        ),
        resource(
            "repeat",
            run_sha256=_digest("repeat-run"),
            elapsed_seconds=100.0,
            peak_rss_bytes=256 * 1024**2,
        ),
    )


def _profile(run: RunRecordV1Alpha1) -> ReplayProfileSummaryV1:
    resources = _resource_evidence(run)
    return ReplayProfileSummaryV1(
        schema="ffb.replay-profile-summary/v1",
        **_global(run.run_id),
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
        distinct_log_group_count=4,
        all_scenes_have_base_support=True,
        all_health_schedules_valid=True,
        raw_sensor_payload_reads=0,
        scientific_replay_worker_count=1,
        gpu_used=False,
        torch_imported=False,
        cuda_used=False,
        resource_evidence=resources,
        peak_rss_bytes=256 * 1024**2,
        elapsed_seconds=100.0,
        dataset_root_serialized=False,
        dataset_bytes_authenticated=False,
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
        attribution_and_non_endorsement_required=True,
    )


def _descriptor(run_id: str) -> ReplayDescriptorAggregateV1:
    return ReplayDescriptorAggregateV1(
        schema="ffb.replay-descriptor-aggregate/v1",
        **_global(run_id),
        descriptor_id="frame-count",
        population="nuscenes-mini-replay",
        population_count=10,
        statistic="median",
        status="ok",
        value=40.0,
        unit="frames",
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
    )


def _persistent(
    index: int,
    run_id: str,
    *,
    identity_index: int,
    coordinate: tuple[str, str, str, str],
) -> ReplayPersistentAggregateV1:
    condition_selector, method_id, metric_id, aggregation = coordinate
    identity_sha256 = replay_experiment_identity_sha256(
        expected_replay_identities()[identity_index]
    )
    hypothesis_coordinate = _PERSISTENT_HYPOTHESIS_BY_COORDINATE.get(coordinate)
    if hypothesis_coordinate is None:
        hypothesis_id = None
        inference_role = "descriptive"
        expected_direction = "none"
        unit = (
            "fraction"
            if metric_id in {"coverage", "undefined-output-rate", "scene-equal-coverage"}
            else "m^2"
        )
    else:
        (
            hypothesis_id,
            _,
            method_id,
            metric_id,
            _,
            unit,
            aggregation,
            inference_role,
            expected_direction,
        ) = hypothesis_coordinate
    if expected_direction == "negative":
        estimate, interval_lower, interval_upper = -1.0, -1.5, -0.5
        positive_scene_count, zero_scene_count, negative_scene_count = 0, 0, 10
        persistence_label = "robustly-persistent"
    elif expected_direction == "positive":
        estimate, interval_lower, interval_upper = 1.0, 0.5, 1.5
        positive_scene_count, zero_scene_count, negative_scene_count = 10, 0, 0
        persistence_label = "robustly-persistent"
    else:
        estimate, interval_lower, interval_upper = (
            (0.5, 0.25, 0.75) if unit == "fraction" else (1.0, -0.5, 1.5)
        )
        positive_scene_count = None
        zero_scene_count = None
        negative_scene_count = None
        persistence_label = "not-applicable"
    return ReplayPersistentAggregateV1(
        schema="ffb.replay-persistent-aggregate/v1",
        **_identity(identity_index, run_id),
        result_id=_stable_id(
            "replay-result",
            {
                "schema": "ffb.replay-result-coordinate/v1",
                "panel_id": M5_PERSISTENT_PANEL_ID,
                "replay_experiment_identity_sha256": identity_sha256,
                "condition_selector": condition_selector,
                "method_id": method_id,
                "metric_id": metric_id,
                "window": "full",
            },
        ),
        condition_id=expected_replay_identities()[identity_index].experiment_id,
        condition_selector=condition_selector,
        hypothesis_id=hypothesis_id,  # type: ignore[arg-type]
        method_id=method_id,
        metric_id=metric_id,
        window="full",
        inference_role=inference_role,  # type: ignore[arg-type]
        unit=unit,  # type: ignore[arg-type]
        status="ok",
        estimate=estimate,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        bootstrap_replicates=2000,
        defined_bootstrap_replicates=2000,
        confidence_level=0.95,
        interval_method="paired-scene-percentile-pointwise",
        aggregation=aggregation,  # type: ignore[arg-type]
        scene_count=10,
        positive_scene_count=positive_scene_count,
        zero_scene_count=zero_scene_count,
        negative_scene_count=negative_scene_count,
        undefined_scene_count=0 if positive_scene_count is not None else None,
        expected_direction=expected_direction,  # type: ignore[arg-type]
        persistence_label=persistence_label,  # type: ignore[arg-type]
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
    )


def _health(
    index: int,
    run_id: str,
    *,
    identity_index: int,
    coordinate: tuple[str, str, str, str, str, str],
) -> ReplayHealthAggregateV1:
    condition_selector, method_id, metric_id, window, unit, aggregation = coordinate
    identity_sha256 = replay_experiment_identity_sha256(
        expected_replay_identities()[identity_index]
    )
    hypothesis = _HEALTH_HYPOTHESIS_BY_COORDINATE.get(coordinate)
    structural_not_applicable = (
        _HEALTH_CASE_BY_SELECTOR[condition_selector].family == "common-mode-position-bias"
        and metric_id in _COMMON_MODE_STRUCTURAL_NA_METRICS
    )
    if hypothesis is None:
        hypothesis_id = None
        inference_role = "descriptive"
        estimate, interval_lower, interval_upper = (
            (0.5, 0.25, 0.75) if unit == "fraction" else (1.0, 0.5, 1.5)
        )
        positive_scene_count = None
        zero_scene_count = None
        negative_scene_count = None
        undefined_scene_count = None
        expected_direction = "none"
        persistence_label = "not-applicable"
        nonpositive_control_supported = None
    else:
        (
            hypothesis_id,
            _,
            _,
            _,
            _,
            _,
            inference_role,
            expected_direction,
        ) = hypothesis
        if expected_direction == "positive":
            estimate, interval_lower, interval_upper = 0.5, 0.1, 1.0
            positive_scene_count, zero_scene_count, negative_scene_count = 8, 0, 2
            persistence_label = "robustly-persistent"
            nonpositive_control_supported = None
        elif expected_direction == "negative":
            estimate, interval_lower, interval_upper = -0.5, -1.0, -0.1
            positive_scene_count, zero_scene_count, negative_scene_count = 2, 0, 8
            persistence_label = "robustly-persistent"
            nonpositive_control_supported = None
        elif expected_direction == "nonpositive":
            estimate, interval_lower, interval_upper = -0.5, -1.0, 0.0
            positive_scene_count, zero_scene_count, negative_scene_count = 2, 0, 8
            persistence_label = "not-applicable"
            nonpositive_control_supported = True
        else:
            estimate, interval_lower, interval_upper = 0.5, -0.25, 1.0
            positive_scene_count = None
            zero_scene_count = None
            negative_scene_count = None
            persistence_label = "not-applicable"
            nonpositive_control_supported = None
        undefined_scene_count = 0 if positive_scene_count is not None else None
    if structural_not_applicable:
        status = "not-applicable"
        estimate = None
        interval_lower = None
        interval_upper = None
        defined_bootstrap_replicates = 0
        persistence_label = "undefined"
        applicability_basis = "structural-unavailable"
        recovery_support_compatible_scene_count = None
    else:
        status = "ok"
        defined_bootstrap_replicates = 2000
        applicability_basis = "applicable"
        recovery_support_compatible_scene_count = (
            10 if metric_id == "frame-oracle-recoverable-loss-fraction" else None
        )
    return ReplayHealthAggregateV1(
        schema="ffb.replay-health-aggregate/v1",
        **_identity(identity_index, run_id),
        result_id=_stable_id(
            "replay-result",
            {
                "schema": "ffb.replay-result-coordinate/v1",
                "panel_id": M5_HEALTH_PANEL_ID,
                "replay_experiment_identity_sha256": identity_sha256,
                "condition_selector": condition_selector,
                "method_id": method_id,
                "metric_id": metric_id,
                "window": window,
            },
        ),
        condition_id=expected_replay_identities()[identity_index].experiment_id,
        condition_selector=condition_selector,
        hypothesis_id=hypothesis_id,  # type: ignore[arg-type]
        method_id=method_id,
        metric_id=metric_id,
        window=window,  # type: ignore[arg-type]
        inference_role=inference_role,  # type: ignore[arg-type]
        unit=unit,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        estimate=estimate,
        interval_lower=interval_lower,
        interval_upper=interval_upper,
        bootstrap_replicates=2000,
        defined_bootstrap_replicates=defined_bootstrap_replicates,
        confidence_level=0.95,
        interval_method="paired-scene-percentile-pointwise",
        aggregation=aggregation,  # type: ignore[arg-type]
        scene_count=10,
        positive_scene_count=positive_scene_count,
        zero_scene_count=zero_scene_count,
        negative_scene_count=negative_scene_count,
        undefined_scene_count=undefined_scene_count,
        expected_direction=expected_direction,  # type: ignore[arg-type]
        persistence_label=persistence_label,  # type: ignore[arg-type]
        nonpositive_control_supported=nonpositive_control_supported,
        applicability_basis=applicability_basis,  # type: ignore[arg-type]
        recovery_support_compatible_scene_count=recovery_support_compatible_scene_count,
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
    )


def _crossover(
    index: int,
    run_id: str,
    *,
    identity_index: int,
    direction: str,
    severity_unit: str,
    tested_maximum: float,
) -> ReplayPersistentCrossoverV1:
    identity_sha256 = replay_experiment_identity_sha256(
        expected_replay_identities()[identity_index]
    )
    return ReplayPersistentCrossoverV1(
        schema="ffb.replay-persistent-crossover/v1",
        **_identity(identity_index, run_id),
        crossover_id=_stable_id(
            "replay-crossover",
            {
                "schema": "ffb.replay-crossover-coordinate/v1",
                "replay_experiment_identity_sha256": identity_sha256,
                "direction": direction,
                "severity_unit": severity_unit,
            },
        ),
        direction=direction,  # type: ignore[arg-type]
        severity_unit=severity_unit,  # type: ignore[arg-type]
        tested_maximum=tested_maximum,
        status="not-observed",
        point_curve_crossed=False,
        point_estimate=None,
        interval_lower=tested_maximum,
        interval_upper="positive-infinity",
        censoring="right-above-tested-maximum",
        bootstrap_replicates=2000,
        bootstrap_crossing_count=0,
        bootstrap_crossing_fraction=0.0,
        confidence_level=0.95,
        interval_method="right-censored-paired-scene-percentile",
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
    )


def _sensitivity(
    identity_index: int,
    run_id: str,
    source: ReplayPersistentAggregateV1 | ReplayHealthAggregateV1,
    *,
    source_ordinal: int,
    ordinal: int,
    cluster_kind: str,
    cluster_id: str,
) -> ReplayClusterSensitivityV1:
    return ReplayClusterSensitivityV1(
        schema="ffb.replay-cluster-sensitivity/v1",
        **_identity(identity_index, run_id),
        sensitivity_id=_stable_id(
            "replay-sensitivity",
            {
                "schema": "ffb.replay-sensitivity-coordinate/v1",
                "source_result_id": source.result_id,
                "cluster_kind": cluster_kind,
                "cluster_id": cluster_id,
            },
        ),
        source_result_id=source.result_id,
        source_record_sha256=sha256_digest(source),
        cluster_kind=cluster_kind,  # type: ignore[arg-type]
        cluster_id=cluster_id,
        status="ok",
        estimate=-0.25 if source.expected_direction == "negative" else 0.25,
        unit=source.unit,
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
    )


def _figure(
    identity_index: int,
    run_id: str,
    source: ReplayPersistentAggregateV1 | ReplayHealthAggregateV1,
) -> ReplayFigureRecordV1:
    return ReplayFigureRecordV1(
        schema="ffb.replay-figure-record/v1",
        **_identity(identity_index, run_id),
        figure_id=f"figure-{identity_index:02d}",
        figure_kind="panel-summary",
        source_result_id=source.result_id,
        source_record_sha256=sha256_digest(source),
        figure_spec_sha256=_digest(f"figure-spec-{identity_index}"),
        tracked_aggregate_terms=M5_TRACKED_AGGREGATE_TERMS,
    )


def _commitments(run_id: str) -> tuple[ReplaySourceMemberCommitmentV1, ...]:
    return tuple(
        ReplaySourceMemberCommitmentV1(
            schema="ffb.replay-source-member-commitment/v1",
            **_global(run_id),
            relative_role=role,
            primary_byte_length=100 + index,
            repeat_byte_length=100 + index,
            primary_record_count=10 + index,
            repeat_record_count=10 + index,
            primary_sha256=_digest(f"member-{index}"),
            repeat_sha256=_digest(f"member-{index}"),
            equal=True,
        )
        for index, role in enumerate(M5_SCIENTIFIC_SOURCE_ROLES)
    )


@cache
def _request() -> ReplayCuratedArtifactWriteRequest:
    run = _run()
    persistent = tuple(
        _persistent(
            index,
            run.run_id,
            identity_index=_IDENTITY_INDEX[
                (
                    _PERSISTENT_CASE_BY_SELECTOR[coordinate[0]].identity.panel_id,
                    _PERSISTENT_CASE_BY_SELECTOR[coordinate[0]].identity.experiment_id,
                )
            ],
            coordinate=coordinate,
        )
        for index, coordinate in enumerate(sorted(expected_replay_persistent_coordinates(_PLAN)))
    )
    health = tuple(
        _health(
            index,
            run.run_id,
            identity_index=_IDENTITY_INDEX[
                (
                    _HEALTH_CASE_BY_SELECTOR[coordinate[0]].identity.panel_id,
                    _HEALTH_CASE_BY_SELECTOR[coordinate[0]].identity.experiment_id,
                )
            ],
            coordinate=coordinate,
        )
        for index, coordinate in enumerate(sorted(expected_replay_health_coordinates(_PLAN)))
    )
    source_by_identity: dict[int, ReplayPersistentAggregateV1 | ReplayHealthAggregateV1] = {}
    for source in (*persistent, *health):
        identity_index = _IDENTITY_INDEX[(source.identity.panel_id, source.identity.experiment_id)]
        source_by_identity.setdefault(identity_index, source)
    identity_sources = tuple(source_by_identity[index] for index in range(22))
    sensitivity_by_result: dict[str, ReplayPersistentAggregateV1 | ReplayHealthAggregateV1] = {}
    for source in (*persistent, *health):
        if source.inference_role in {"primary-directional", "nonpositive-control"}:
            sensitivity_by_result[source.result_id] = source
    sensitivity_sources = tuple(sensitivity_by_result.values())
    commitments = _commitments(run.run_id)
    commitment_bytes = canonical_replay_ndjson_bytes(commitments)
    repeat = ReplayRepeatVerificationV1(
        schema="ffb.replay-repeat-verification/v1",
        **_global(run.run_id),
        primary_local_artifact_sha256=_digest("local-artifact"),
        repeat_local_artifact_sha256=_digest("local-artifact"),
        primary_run_sha256=_digest("primary-run"),
        repeat_run_sha256=_digest("repeat-run"),
        source_member_commitments_sha256=hashlib.sha256(commitment_bytes).hexdigest(),
        scientific_member_count=len(commitments),
        mismatch_count=0,
        scientific_members_all_equal=True,
        run_records_distinct=True,
        source_paths_and_inodes_independent=True,
        same_named_cpu_environment=True,
        evidence_scope=("distinct-path-inode-run-and-member-consistency-not-cryptographic-proof"),
        all_checks_passed=True,
    )
    profile = _profile(run)
    validation = ReplayValidationV1(
        schema="ffb.replay-validation/v1",
        **_global(run.run_id),
        checks=tuple(
            ReplayValidationCheckV1(
                check_id=check_id,
                passed=True,
                evidence_sha256=(
                    replay_resource_evidence_sha256(profile.resource_evidence)
                    if check_id == "cpu-and-memory-caps"
                    else _digest(check_id)
                ),
            )
            for check_id in M5_RELEASE_VALIDATION_CHECK_IDS
        ),
        scene_count=10,
        replay_experiment_count=22,
        raw_sensor_payload_reads=0,
        all_checks_passed=True,
    )
    return ReplayCuratedArtifactWriteRequest(
        profile_summary=profile,
        descriptor_aggregates=(_descriptor(run.run_id),),
        persistent_aggregates=persistent,
        persistent_crossovers=tuple(
            _crossover(
                index,
                run.run_id,
                identity_index=identity_index,
                direction=direction,
                severity_unit=unit,
                tested_maximum=tested_maximum,
            )
            for index, (identity_index, direction, unit, tested_maximum) in enumerate(
                (
                    (0, "negative", "m", 4.0),
                    (0, "positive", "m", 4.0),
                    (1, "increase", "std-scale", 4.0),
                    (2, "increase", "std-scale", 4.0),
                    (3, "negative", "m", 4.0),
                    (3, "positive", "m", 4.0),
                    (4, "negative", "rad", 0.08),
                    (4, "positive", "rad", 0.08),
                    (5, "negative", "s", 0.8),
                    (5, "positive", "s", 0.8),
                )
            )
        ),
        health_aggregates=health,
        cluster_sensitivity=tuple(
            _sensitivity(
                _IDENTITY_INDEX[(source.identity.panel_id, source.identity.experiment_id)],
                run.run_id,
                source,
                source_ordinal=source_ordinal,
                ordinal=ordinal,
                cluster_kind=cluster_kind,
                cluster_id=cluster_id,
            )
            for source_ordinal, source in enumerate(sensitivity_sources)
            for ordinal, (cluster_kind, cluster_id) in enumerate(
                (
                    *(("leave-one-scene-out", f"scene-ordinal:{index:02d}") for index in range(10)),
                    *(("leave-one-log-group-out", f"log-group:{index:02d}") for index in range(4)),
                )
            )
        ),
        validation=validation,
        repeat_verification=repeat,
        figures=tuple(
            _figure(index, run.run_id, source) for index, source in enumerate(identity_sources)
        ),
        source_commitments=commitments,
        run=run,
    )


def _minimal_candidate_files() -> dict[str, bytes]:
    files = {path: b"{}\n" for path in REPLAY_ARTIFACT_PATHS}
    files[REPLAY_INTENT_FILE] = _PLAN.intent.path.read_bytes()
    return files


def _write_minimal_artifact_tree(root: Path) -> None:
    root.mkdir()
    for path, value in _minimal_candidate_files().items():
        (root / path).write_bytes(value)


def _primary_with_sensitivity() -> tuple[
    ReplayPersistentAggregateV1 | ReplayHealthAggregateV1,
    tuple[ReplayClusterSensitivityV1, ...],
]:
    request = _request()
    source = next(
        row
        for row in (*request.persistent_aggregates, *request.health_aggregates)
        if row.inference_role == "primary-directional"
    )
    rows = tuple(
        row for row in request.cluster_sensitivity if row.source_result_id == source.result_id
    )
    assert len(rows) == 14
    return source, rows


def _validate_request_links(request: ReplayCuratedArtifactWriteRequest) -> None:
    (
        descriptors,
        persistent,
        crossovers,
        health,
        sensitivity,
        figures,
        commitments,
    ) = replay_artifacts._ordered_records(request)
    replay_artifacts._validate_record_links(
        run_id=request.run.run_id,
        runtime_environment_sha256=replay_environment_sha256(request.run.environment),
        logical_command_sha256=replay_logical_command_sha256(tuple(request.run.command)),
        profile_summary=request.profile_summary,
        descriptors=descriptors,
        persistent=persistent,
        crossovers=crossovers,
        health=health,
        sensitivity=sensitivity,
        validation=request.validation,
        repeat=request.repeat_verification,
        figures=figures,
        commitments=commitments,
    )


def test_round_trip_is_aggregate_only_and_fully_bound(tmp_path: Path) -> None:
    destination = tmp_path / "m5-release"
    loaded = write_replay_curated_artifact(
        _request(),
        destination,
        source_root=ROOT,
        git_metadata_dirs=(),
    )

    assert set(path.name for path in destination.iterdir()) == set(REPLAY_ARTIFACT_PATHS)
    assert len(loaded.persistent_aggregates) == 464
    assert len(loaded.health_aggregates) == 14_988
    assert len(loaded.cluster_sensitivity) == 26 * 14
    assert len(loaded.figures) == 22
    assert hashlib.sha256(loaded.intent_bytes).hexdigest() == M5_REPLAY_INTENT_BYTE_SHA256
    assert loaded.run.artifact_sha256 == loaded.artifact_sha256
    assert loaded.release_index.run_id == loaded.run.run_id
    assert (destination / REPLAY_SUCCESS_FILE).is_file()
    assert all(
        not path.name.endswith((".jpg", ".png", ".bin", ".pcd")) for path in destination.iterdir()
    )


def test_scientific_members_and_artifact_digest_are_destination_independent(
    tmp_path: Path,
) -> None:
    first = write_replay_curated_artifact(
        _request(),
        tmp_path / "first",
        source_root=ROOT,
        git_metadata_dirs=(),
    )
    second = write_replay_curated_artifact(
        _request(),
        tmp_path / "second",
        source_root=ROOT,
        git_metadata_dirs=(),
    )

    assert first.artifact_sha256 == second.artifact_sha256
    for path in REPLAY_ARTIFACT_PATHS:
        assert (first.path / path).read_bytes() == (second.path / path).read_bytes()


def test_publication_never_overwrites_existing_or_dangling_destinations(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_text("owned")
    with pytest.raises(FileExistsError):
        write_replay_curated_artifact(
            _request(),
            existing,
            source_root=ROOT,
            git_metadata_dirs=(),
        )
    assert marker.read_text() == "owned"

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        write_replay_curated_artifact(
            _request(),
            dangling,
            source_root=ROOT,
            git_metadata_dirs=(),
        )
    assert dangling.is_symlink()


def test_loader_rejects_mutation_extra_members_symlinks_and_hardlinks(
    tmp_path: Path,
) -> None:
    base = write_replay_curated_artifact(
        _request(),
        tmp_path / "base",
        source_root=ROOT,
        git_metadata_dirs=(),
    ).path

    aggregate_path = base / REPLAY_PERSISTENT_AGGREGATES_FILE
    original = aggregate_path.read_bytes()
    aggregate_path.write_bytes(original + b"\n")
    with pytest.raises(ArtifactValidationError):
        load_replay_curated_artifact(base)
    aggregate_path.write_bytes(original)

    extra = base / "private-output.ndjson"
    extra.write_text("{}\n")
    with pytest.raises(ArtifactValidationError):
        load_replay_curated_artifact(base)
    extra.unlink()

    run_path = base / REPLAY_RUN_FILE
    run_bytes = run_path.read_bytes()
    run_path.unlink()
    run_path.symlink_to(REPLAY_RELEASE_INDEX_FILE)
    with pytest.raises(ArtifactValidationError):
        load_replay_curated_artifact(base)
    run_path.unlink()
    run_path.write_bytes(run_bytes)

    success_path = base / REPLAY_SUCCESS_FILE
    hardlink = tmp_path / "success-hardlink"
    os.link(success_path, hardlink)
    with pytest.raises(ArtifactValidationError):
        load_replay_curated_artifact(base)


def test_writer_rejects_incomplete_identity_coverage_and_rebound_references(
    tmp_path: Path,
) -> None:
    request = _request()
    incomplete = replace(
        request,
        persistent_aggregates=request.persistent_aggregates[:-1],
    )
    with pytest.raises(ArtifactValidationError, match="aggregate coordinate matrix"):
        write_replay_curated_artifact(
            incomplete,
            tmp_path / "incomplete",
            source_root=ROOT,
            git_metadata_dirs=(),
        )

    health_incomplete = replace(
        request,
        health_aggregates=request.health_aggregates[:-1],
    )
    with pytest.raises(ArtifactValidationError, match="aggregate coordinate matrix"):
        write_replay_curated_artifact(
            health_incomplete,
            tmp_path / "health-incomplete",
            source_root=ROOT,
            git_metadata_dirs=(),
        )

    substituted_index = next(
        index
        for index, row in enumerate(request.health_aggregates)
        if row.hypothesis_id is None and row.status == "ok"
    )
    substituted = request.health_aggregates[substituted_index].model_copy(
        update={"method_id": "substituted-method"}
    )
    health_substituted = replace(
        request,
        health_aggregates=(
            *request.health_aggregates[:substituted_index],
            substituted,
            *request.health_aggregates[substituted_index + 1 :],
        ),
    )
    with pytest.raises(ArtifactValidationError, match="aggregate coordinate matrix"):
        write_replay_curated_artifact(
            health_substituted,
            tmp_path / "health-substituted",
            source_root=ROOT,
            git_metadata_dirs=(),
        )

    omitted = request.health_aggregates[-1]
    duplicate_source = next(
        row
        for row in request.health_aggregates[:-1]
        if row.replay_identity_sha256 == omitted.replay_identity_sha256
    )
    duplicated = duplicate_source.model_copy(
        update={"result_id": "health-result-duplicate-coordinate"}
    )
    health_duplicated = replace(
        request,
        health_aggregates=(*request.health_aggregates[:-1], duplicated),
    )
    with pytest.raises(ArtifactValidationError, match="aggregate coordinate matrix"):
        write_replay_curated_artifact(
            health_duplicated,
            tmp_path / "health-duplicated",
            source_root=ROOT,
            git_metadata_dirs=(),
        )

    wrong_maximum = request.persistent_crossovers[-1].model_copy(
        update={
            "tested_maximum": 8.0,
            "interval_lower": 8.0,
        }
    )
    crossover_substituted = replace(
        request,
        persistent_crossovers=(*request.persistent_crossovers[:-1], wrong_maximum),
    )
    with pytest.raises(ArtifactValidationError, match="crossover coordinate coverage"):
        write_replay_curated_artifact(
            crossover_substituted,
            tmp_path / "crossover-substituted",
            source_root=ROOT,
            git_metadata_dirs=(),
        )

    recovery_index = next(
        index
        for index, row in enumerate(request.health_aggregates)
        if row.metric_id == "frame-oracle-recoverable-loss-fraction" and row.status == "ok"
    )
    relabeled_recovery = request.health_aggregates[recovery_index].model_copy(
        update={
            "status": "not-applicable",
            "estimate": None,
            "interval_lower": None,
            "interval_upper": None,
            "defined_bootstrap_replicates": 0,
            "persistence_label": "undefined",
        }
    )
    status_substituted = replace(
        request,
        health_aggregates=(
            *request.health_aggregates[:recovery_index],
            relabeled_recovery,
            *request.health_aggregates[recovery_index + 1 :],
        ),
    )
    with pytest.raises(ArtifactValidationError, match="applicability status"):
        write_replay_curated_artifact(
            status_substituted,
            tmp_path / "status-substituted",
            source_root=ROOT,
            git_metadata_dirs=(),
        )

    not_applicable_source = next(
        row for row in request.health_aggregates if row.status == "not-applicable"
    )
    source_identity_index = _IDENTITY_INDEX[
        (
            not_applicable_source.identity.panel_id,
            not_applicable_source.identity.experiment_id,
        )
    ]
    forbidden_sensitivity = tuple(
        _sensitivity(
            source_identity_index,
            request.run.run_id,
            not_applicable_source,
            source_ordinal=999,
            ordinal=ordinal,
            cluster_kind=cluster_kind,
            cluster_id=cluster_id,
        )
        for ordinal, (cluster_kind, cluster_id) in enumerate(
            (
                *(("leave-one-scene-out", f"scene-ordinal:{index:02d}") for index in range(10)),
                *(("leave-one-log-group-out", f"log-group:{index:02d}") for index in range(4)),
            )
        )
    )
    with pytest.raises(ArtifactValidationError, match="descriptive aggregate carries"):
        write_replay_curated_artifact(
            replace(
                request,
                cluster_sensitivity=(*request.cluster_sensitivity, *forbidden_sensitivity),
            ),
            tmp_path / "not-applicable-sensitivity",
            source_root=ROOT,
            git_metadata_dirs=(),
        )

    first = request.cluster_sensitivity[0]
    rebound = first.model_copy(update={"source_record_sha256": _DIGEST})
    invalid_reference = replace(
        request,
        cluster_sensitivity=(rebound, *request.cluster_sensitivity[1:]),
    )
    with pytest.raises(ArtifactValidationError, match="source binding"):
        write_replay_curated_artifact(
            invalid_reference,
            tmp_path / "rebound",
            source_root=ROOT,
            git_metadata_dirs=(),
        )


def test_persistence_label_is_derived_from_complete_cluster_evidence(
    tmp_path: Path,
) -> None:
    request = _request()
    sensitivity_source_ids = {row.source_result_id for row in request.cluster_sensitivity}
    original_index, original = next(
        (index, row)
        for index, row in enumerate(request.persistent_aggregates)
        if row.inference_role == "primary-directional"
        and row.expected_direction == "positive"
        and row.result_id in sensitivity_source_ids
    )
    primary = original.model_copy(
        update={
            "inference_role": "primary-directional",
            "estimate": 1.0,
            "interval_lower": 0.5,
            "interval_upper": 1.5,
            "positive_scene_count": 8,
            "zero_scene_count": 0,
            "negative_scene_count": 2,
            "undefined_scene_count": 0,
            "expected_direction": "positive",
            "persistence_label": "robustly-persistent",
        }
    )
    source_sha256 = sha256_digest(primary)
    sensitivity = tuple(
        row.model_copy(
            update={
                "source_record_sha256": source_sha256,
                "estimate": 0.25,
            }
        )
        if row.source_result_id == primary.result_id
        else row
        for row in request.cluster_sensitivity
    )
    figures = tuple(
        row.model_copy(update={"source_record_sha256": source_sha256})
        if row.source_result_id == primary.result_id
        else row
        for row in request.figures
    )
    valid = replace(
        request,
        persistent_aggregates=(
            *request.persistent_aggregates[:original_index],
            primary,
            *request.persistent_aggregates[original_index + 1 :],
        ),
        cluster_sensitivity=sensitivity,
        figures=figures,
    )
    loaded = write_replay_curated_artifact(
        valid,
        tmp_path / "persistent",
        source_root=ROOT,
        git_metadata_dirs=(),
    )
    assert (
        next(
            row for row in loaded.persistent_aggregates if row.result_id == primary.result_id
        ).persistence_label
        == "robustly-persistent"
    )

    mislabeled = primary.model_copy(update={"persistence_label": "directionally-consistent"})
    mislabeled_sha256 = sha256_digest(mislabeled)
    invalid = replace(
        valid,
        persistent_aggregates=(
            *valid.persistent_aggregates[:original_index],
            mislabeled,
            *valid.persistent_aggregates[original_index + 1 :],
        ),
        cluster_sensitivity=tuple(
            row.model_copy(update={"source_record_sha256": mislabeled_sha256})
            if row.source_result_id == primary.result_id
            else row
            for row in valid.cluster_sensitivity
        ),
        figures=tuple(
            row.model_copy(update={"source_record_sha256": mislabeled_sha256})
            if row.source_result_id == primary.result_id
            else row
            for row in valid.figures
        ),
    )
    with pytest.raises(ArtifactValidationError, match="not derived"):
        write_replay_curated_artifact(
            invalid,
            tmp_path / "mislabeled",
            source_root=ROOT,
            git_metadata_dirs=(),
        )


def test_health_hypothesis_coordinate_map_is_mechanically_complete(
    tmp_path: Path,
) -> None:
    request = _request()
    target_index = next(
        index for index, row in enumerate(request.health_aggregates) if row.hypothesis_id == "h5-b1"
    )
    health = list(request.health_aggregates)
    health[target_index] = health[target_index].model_copy(update={"hypothesis_id": None})
    with pytest.raises(ArtifactValidationError, match="hypothesis coordinate map"):
        write_replay_curated_artifact(
            replace(request, health_aggregates=tuple(health)),
            tmp_path / "missing-hypothesis",
            source_root=ROOT,
            git_metadata_dirs=(),
        )


def test_candidate_privacy_scan_rejects_paths_scenes_secrets_and_raw_payloads() -> None:
    files = _minimal_candidate_files()
    cases = (
        b'{"local":"/Users/person/dataset"}\n',
        b'{"private":"scene-0061"}\n',
        b'{"credential":"api_key=abcdefghijk"}\n',
        b'{"payload":"sweep.bin"}\n',
    )
    for candidate in cases:
        mutated = dict(files)
        mutated[REPLAY_DESCRIPTOR_AGGREGATES_FILE] = candidate
        with pytest.raises(ArtifactValidationError):
            validate_replay_candidate_bytes(mutated)


def test_candidate_requires_exact_frozen_intent_and_allowlist() -> None:
    files = _minimal_candidate_files()

    mutated = dict(files)
    mutated[REPLAY_INTENT_FILE] += b" "
    with pytest.raises(ArtifactValidationError, match="byte-frozen intent"):
        validate_replay_candidate_bytes(mutated)

    missing = dict(files)
    del missing[REPLAY_SUCCESS_FILE]
    with pytest.raises(ArtifactValidationError, match="allowlist"):
        validate_replay_candidate_bytes(missing)


def test_index_and_envelope_members_are_canonical_json(tmp_path: Path) -> None:
    loaded = write_replay_curated_artifact(
        _request(),
        tmp_path / "valid",
        source_root=ROOT,
        git_metadata_dirs=(),
    )
    assert (loaded.path / REPLAY_RELEASE_INDEX_FILE).read_bytes() == canonical_json_bytes(
        loaded.release_index
    )
    assert (loaded.path / REPLAY_RUN_FILE).read_bytes() == canonical_json_bytes(loaded.run)


def test_canonical_ndjson_enforces_empty_count_record_and_member_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor(_run().run_id)
    encoded = canonical_replay_ndjson_bytes((descriptor,))
    assert encoded == canonical_json_bytes(descriptor)
    assert replay_artifacts.compute_replay_curated_artifact_digest(encoded) != (
        replay_artifacts.compute_replay_curated_artifact_digest(encoded + b"\n")
    )

    with pytest.raises(ArtifactValidationError, match="must not be empty"):
        canonical_replay_ndjson_bytes(())
    with monkeypatch.context() as patch:
        patch.setattr(replay_artifacts, "REPLAY_MAX_NDJSON_RECORDS", 0)
        with pytest.raises(ArtifactValidationError, match="record-count cap"):
            canonical_replay_ndjson_bytes((descriptor,))
    with monkeypatch.context() as patch:
        patch.setattr(replay_artifacts, "REPLAY_MAX_RECORD_BYTES", len(encoded) - 1)
        with pytest.raises(ArtifactValidationError, match="record 0"):
            canonical_replay_ndjson_bytes((descriptor,))
    with monkeypatch.context() as patch:
        patch.setattr(replay_artifacts, "REPLAY_MAX_MEMBER_BYTES", len(encoded) - 1)
        with pytest.raises(ArtifactValidationError, match="member cap"):
            canonical_replay_ndjson_bytes((descriptor,))


def test_run_and_frozen_intent_authorities_reject_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run()
    replay_artifacts._validate_run(run)
    invalid_runs = (
        (run.model_copy(update={"manifest_sha256": "f" * 64}), "frozen intent"),
        (run.model_copy(update={"run_id": "f" * 64}), "run_id"),
        (run.model_copy(update={"source_dirty": True}), "clean successful"),
        (run.model_copy(update={"status": "failed"}), "clean successful"),
        (run.model_copy(update={"artifact_sha256": "f" * 64}), "unfinalized"),
        (
            run.model_copy(update={"command": ("fusion-fault-bench", "replay", "curate")}),
            "exact M5 authority",
        ),
        (
            run.model_copy(
                update={
                    "command": (
                        "ffb",
                        "replay",
                        "run",
                        "--output-dir",
                        "/private/local-output",
                    )
                }
            ),
            "exact M5 authority",
        ),
    )
    for invalid, message in invalid_runs:
        with pytest.raises(ArtifactValidationError, match=message):
            replay_artifacts._validate_run(invalid)

    finalized = replay_artifacts._finalize_run(run, "a" * 64)
    replay_artifacts._validate_run(finalized, artifact_sha256="a" * 64)
    with pytest.raises(ArtifactValidationError, match="artifact identity"):
        replay_artifacts._validate_run(finalized, artifact_sha256="b" * 64)

    intent_bytes = _PLAN.intent.path.read_bytes()
    replay_artifacts._validate_frozen_intent_bytes(intent_bytes)
    with monkeypatch.context() as patch:
        patch.setattr(
            replay_artifacts,
            "REPLAY_MAX_MEMBER_BYTES",
            len(intent_bytes) - 1,
        )
        with pytest.raises(ArtifactValidationError, match="byte-frozen intent"):
            replay_artifacts._validate_frozen_intent_bytes(intent_bytes)

    malformed_values = (
        (b"not-json\n", "invalid JSON"),
        (b"[]\n", "canonical identity"),
        (b'{"schema":"wrong"}\n', "canonical identity"),
    )
    for value, message in malformed_values:
        with monkeypatch.context() as patch:
            patch.setattr(
                replay_artifacts,
                "M5_REPLAY_INTENT_BYTE_SHA256",
                hashlib.sha256(value).hexdigest(),
            )
            with pytest.raises(ArtifactValidationError, match=message):
                replay_artifacts._validate_frozen_intent_bytes(value)


def test_candidate_parser_rejects_malformed_json_caps_and_nested_private_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _minimal_candidate_files()
    validate_replay_candidate_bytes(files)

    malformed = (
        (b"", "must not be empty"),
        (b"{}", "LF-delimited"),
        (b"{}\r\n", "LF-delimited"),
        (b"not-json\n", "invalid JSON"),
        (b"[]\n", "JSON objects"),
    )
    for value, message in malformed:
        candidate = dict(files)
        candidate[REPLAY_DESCRIPTOR_AGGREGATES_FILE] = value
        with pytest.raises(ArtifactValidationError, match=message):
            validate_replay_candidate_bytes(candidate)

    nested_private = (
        b'{"public":[{"dataset-root":"/safe-looking"}]}\n',
        b'{"public":["interview/notes.md"]}\n',
    )
    for value in nested_private:
        candidate = dict(files)
        candidate[REPLAY_DESCRIPTOR_AGGREGATES_FILE] = value
        with pytest.raises(ArtifactValidationError, match=r"private|privacy"):
            validate_replay_candidate_bytes(candidate)

    with pytest.raises(ArtifactValidationError, match="non-string JSON key"):
        replay_artifacts._scan_public_value({1: "value"})

    with monkeypatch.context() as patch:
        patch.setattr(replay_artifacts, "REPLAY_MAX_MEMBER_BYTES", 1)
        with pytest.raises(ArtifactValidationError, match="member exceeds"):
            validate_replay_candidate_bytes(files)
    with monkeypatch.context() as patch:
        patch.setattr(
            replay_artifacts,
            "REPLAY_MAX_ARTIFACT_BYTES",
            sum(len(value) for value in files.values()) - 1,
        )
        with pytest.raises(ArtifactValidationError, match="50 MiB cap"):
            validate_replay_candidate_bytes(files)
    with monkeypatch.context() as patch:
        patch.setattr(replay_artifacts, "REPLAY_MAX_RECORD_BYTES", 2)
        with pytest.raises(ArtifactValidationError, match="record exceeds"):
            validate_replay_candidate_bytes(files)


def test_safe_tree_and_snapshot_reject_malformed_or_raced_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    _write_minimal_artifact_tree(root)
    snapshot = replay_artifacts._require_safe_tree(root)
    assert set(snapshot.entries) == set(REPLAY_ARTIFACT_PATHS)

    extra = root / "extra.json"
    extra.write_bytes(b"{}\n")
    allowlist_race = replay_artifacts._TreeSnapshot(
        root_stat=os.lstat(root),
        entries=snapshot.entries,
    )
    with pytest.raises(ArtifactValidationError, match="allowlist changed"):
        replay_artifacts._verify_tree_snapshot(root, allowlist_race)
    extra.unlink()

    snapshot = replay_artifacts._require_safe_tree(root)
    member = root / REPLAY_DESCRIPTOR_AGGREGATES_FILE
    original = member.read_bytes()
    member.write_bytes(original + b" ")
    with pytest.raises(ArtifactValidationError, match="member changed"):
        replay_artifacts._verify_tree_snapshot(root, snapshot)
    with pytest.raises(ArtifactValidationError, match="member changed"):
        replay_artifacts._open_regular_member(
            root,
            REPLAY_DESCRIPTOR_AGGREGATES_FILE,
            expected_stat=snapshot.entries[REPLAY_DESCRIPTOR_AGGREGATES_FILE],
        )
    current = os.lstat(member)
    with pytest.raises(ArtifactValidationError, match="exceeds its cap"):
        replay_artifacts._read_member(
            root,
            REPLAY_DESCRIPTOR_AGGREGATES_FILE,
            expected_stat=current,
            byte_cap=1,
        )

    file_root = tmp_path / "not-a-directory"
    file_root.write_bytes(b"{}\n")
    with pytest.raises(ArtifactValidationError, match="real directory"):
        replay_artifacts._require_safe_tree(file_root)

    symlink_target = tmp_path / "symlink-target"
    _write_minimal_artifact_tree(symlink_target)
    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(symlink_target, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="contains a symlink"):
        replay_artifacts._require_safe_tree(symlink_root)

    with monkeypatch.context() as patch:
        patch.setattr(replay_artifacts, "REPLAY_MAX_MEMBER_BYTES", 1)
        with pytest.raises(ArtifactValidationError, match="member exceeds"):
            replay_artifacts._require_safe_tree(symlink_target)
    with monkeypatch.context() as patch:
        patch.setattr(replay_artifacts, "REPLAY_MAX_ARTIFACT_BYTES", 1)
        with pytest.raises(ArtifactValidationError, match="50 MiB tree cap"):
            replay_artifacts._require_safe_tree(symlink_target)


def test_model_and_ndjson_loaders_require_strict_canonical_nonempty_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(_run())
    profile_bytes = canonical_json_bytes(profile)
    loaded = replay_artifacts._load_model(
        profile_bytes,
        label="profile",
        validate=ReplayProfileSummaryV1.model_validate_json,
    )
    assert loaded == profile

    profile_value = profile.model_dump(mode="json", by_alias=True)
    reversed_value = dict(reversed(tuple(profile_value.items())))
    noncanonical = (
        json.dumps(
            reversed_value,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    with pytest.raises(ArtifactValidationError, match="not canonical"):
        replay_artifacts._load_model(
            noncanonical,
            label="profile",
            validate=ReplayProfileSummaryV1.model_validate_json,
        )
    with pytest.raises(ArtifactValidationError, match="strict canonical"):
        replay_artifacts._load_model(
            b"{}\n",
            label="profile",
            validate=ReplayProfileSummaryV1.model_validate_json,
        )

    rows_path = tmp_path / "rows.ndjson"
    descriptor = _descriptor(_run().run_id)
    rows_path.write_bytes(canonical_json_bytes(descriptor))
    records = replay_artifacts._load_ndjson(
        tmp_path,
        rows_path.name,
        expected_stat=os.lstat(rows_path),
        validate=ReplayDescriptorAggregateV1.model_validate_json,
    )
    assert records == (descriptor,)

    rows_path.write_bytes(b"")
    with pytest.raises(ArtifactValidationError, match="must not be empty"):
        replay_artifacts._load_ndjson(
            tmp_path,
            rows_path.name,
            expected_stat=os.lstat(rows_path),
            validate=ReplayDescriptorAggregateV1.model_validate_json,
        )
    rows_path.write_bytes(canonical_json_bytes(descriptor))
    with monkeypatch.context() as patch:
        patch.setattr(replay_artifacts, "REPLAY_MAX_RECORD_BYTES", 2)
        with pytest.raises(ArtifactValidationError, match="record cap"):
            replay_artifacts._load_ndjson(
                tmp_path,
                rows_path.name,
                expected_stat=os.lstat(rows_path),
                validate=ReplayDescriptorAggregateV1.model_validate_json,
            )
    with monkeypatch.context() as patch:
        patch.setattr(replay_artifacts, "REPLAY_MAX_NDJSON_RECORDS", 0)
        with pytest.raises(ArtifactValidationError, match="record-count cap"):
            replay_artifacts._load_ndjson(
                tmp_path,
                rows_path.name,
                expected_stat=os.lstat(rows_path),
                validate=ReplayDescriptorAggregateV1.model_validate_json,
            )


def test_sensitivity_authority_rejects_unknown_incomplete_descriptive_and_unit_rebinding() -> None:
    request = _request()
    source, rows = _primary_with_sensitivity()
    replay_artifacts._validate_sensitivity_coverage(
        profile_summary=request.profile_summary,
        aggregates=(source,),
        sensitivity=rows,
    )

    with pytest.raises(ArtifactValidationError, match="unknown aggregate"):
        replay_artifacts._validate_sensitivity_coverage(
            profile_summary=request.profile_summary,
            aggregates=(source,),
            sensitivity=(rows[0].model_copy(update={"source_result_id": "unknown-result"}),),
        )
    with pytest.raises(ArtifactValidationError, match="coverage is incomplete"):
        replay_artifacts._validate_sensitivity_coverage(
            profile_summary=request.profile_summary,
            aggregates=(source,),
            sensitivity=rows[:-1],
        )
    with pytest.raises(ArtifactValidationError, match="lacks cluster sensitivity"):
        replay_artifacts._validate_sensitivity_coverage(
            profile_summary=request.profile_summary,
            aggregates=(source,),
            sensitivity=(),
        )

    descriptive = next(
        row
        for row in (*request.persistent_aggregates, *request.health_aggregates)
        if row.inference_role == "descriptive" and row.status == "ok"
    )
    rebound = tuple(
        row.model_copy(
            update={
                "source_result_id": descriptive.result_id,
                "unit": descriptive.unit,
            }
        )
        for row in rows
    )
    with pytest.raises(ArtifactValidationError, match="descriptive aggregate carries"):
        replay_artifacts._validate_sensitivity_coverage(
            profile_summary=request.profile_summary,
            aggregates=(descriptive,),
            sensitivity=rebound,
        )

    wrong_unit = "s" if source.unit != "s" else "m^2"
    with pytest.raises(ArtifactValidationError, match="unit disagrees"):
        replay_artifacts._validate_sensitivity_coverage(
            profile_summary=request.profile_summary,
            aggregates=(source,),
            sensitivity=(rows[0].model_copy(update={"unit": wrong_unit}), *rows[1:]),
        )


def test_directional_classification_is_derived_from_point_interval_scene_and_cluster_evidence() -> (
    None
):
    source, rows = _primary_with_sensitivity()
    assert (
        replay_artifacts._classify_directional_aggregate(
            source,
            rows,
            distinct_log_group_count=4,
        )
        == "robustly-persistent"
    )
    assert (
        replay_artifacts._classify_directional_aggregate(
            source.model_copy(update={"status": "undefined"}),
            rows,
            distinct_log_group_count=4,
        )
        == "undefined"
    )

    opposite = -1.0 if source.expected_direction == "positive" else 1.0
    assert (
        replay_artifacts._classify_directional_aggregate(
            source.model_copy(update={"estimate": opposite}),
            rows,
            distinct_log_group_count=4,
        )
        == "non-persistent"
    )
    assert (
        replay_artifacts._classify_directional_aggregate(
            source,
            rows,
            distinct_log_group_count=1,
        )
        == "directionally-consistent"
    )
    assert (
        replay_artifacts._classify_directional_aggregate(
            source,
            (rows[0].model_copy(update={"status": "undefined", "estimate": None}), *rows[1:]),
            distinct_log_group_count=4,
        )
        == "directionally-consistent"
    )


def test_health_applicability_and_coordinate_ids_are_revalidated_after_model_construction() -> None:
    request = _request()
    structural = next(row for row in request.health_aggregates if row.status == "not-applicable")
    recovery = next(
        row
        for row in request.health_aggregates
        if row.metric_id == "frame-oracle-recoverable-loss-fraction" and row.status == "ok"
    )
    ordinary = next(
        row
        for row in request.health_aggregates
        if row.metric_id != "frame-oracle-recoverable-loss-fraction" and row.status == "ok"
    )
    replay_artifacts._validate_health_not_applicable_semantics((structural, recovery, ordinary))

    invalid_rows = (
        structural.model_copy(update={"applicability_basis": "applicable"}),
        recovery.model_copy(update={"recovery_support_compatible_scene_count": 9}),
        ordinary.model_copy(update={"applicability_basis": "support-incompatible"}),
        structural.model_copy(update={"positive_scene_count": 0}),
    )
    for invalid in invalid_rows:
        with pytest.raises(ArtifactValidationError, match="applicability status"):
            replay_artifacts._validate_health_not_applicable_semantics((invalid,))

    source, sensitivity = _primary_with_sensitivity()
    with pytest.raises(ArtifactValidationError, match="panel result identifier"):
        replay_artifacts._validate_deterministic_ids(
            persistent=(
                source.model_copy(update={"result_id": "forged-result"}),  # type: ignore[arg-type]
            ),
            crossovers=(),
            health=(),
            sensitivity=(),
        )
    with pytest.raises(ArtifactValidationError, match="crossover identifier"):
        replay_artifacts._validate_deterministic_ids(
            persistent=(),
            crossovers=(
                request.persistent_crossovers[0].model_copy(
                    update={"crossover_id": "forged-crossover"}
                ),
            ),
            health=(),
            sensitivity=(),
        )
    with pytest.raises(ArtifactValidationError, match="sensitivity identifier"):
        replay_artifacts._validate_deterministic_ids(
            persistent=(),
            crossovers=(),
            health=(),
            sensitivity=(
                sensitivity[0].model_copy(update={"sensitivity_id": "forged-sensitivity"}),
            ),
        )


def test_record_link_authority_rejects_run_figure_repeat_and_validation_tampering() -> None:
    request = _request()
    _validate_request_links(request)

    with pytest.raises(ArtifactValidationError, match="run_id binding"):
        _validate_request_links(
            replace(
                request,
                profile_summary=request.profile_summary.model_copy(update={"run_id": "f" * 64}),
            )
        )
    with pytest.raises(ArtifactValidationError, match="figure-record identity coverage"):
        _validate_request_links(replace(request, figures=request.figures[:-1]))

    invalid_figure = request.figures[0].model_copy(update={"source_record_sha256": "f" * 64})
    with pytest.raises(ArtifactValidationError, match="invalid aggregate binding"):
        _validate_request_links(replace(request, figures=(invalid_figure, *request.figures[1:])))

    descriptor_figure = request.figures[0].model_copy(
        update={
            "source_result_id": request.descriptor_aggregates[0].descriptor_id,
            "source_record_sha256": sha256_digest(request.descriptor_aggregates[0]),
            "figure_kind": "panel-summary",
        }
    )
    with pytest.raises(ArtifactValidationError, match="descriptor-comparison"):
        _validate_request_links(replace(request, figures=(descriptor_figure, *request.figures[1:])))

    with pytest.raises(ArtifactValidationError, match="frozen canonical set"):
        _validate_request_links(replace(request, source_commitments=()))
    with pytest.raises(ArtifactValidationError, match="repeat verification disagrees"):
        _validate_request_links(
            replace(
                request,
                repeat_verification=request.repeat_verification.model_copy(
                    update={"source_member_commitments_sha256": "f" * 64}
                ),
            )
        )
    with pytest.raises(ArtifactValidationError, match="did not pass"):
        _validate_request_links(
            replace(
                request,
                validation=request.validation.model_copy(update={"all_checks_passed": False}),
            )
        )
    with pytest.raises(ArtifactValidationError, match="check order"):
        _validate_request_links(
            replace(
                request,
                validation=request.validation.model_copy(
                    update={"checks": tuple(reversed(request.validation.checks))}
                ),
            )
        )

    primary, repeat = request.profile_summary.resource_evidence
    for invalid_profile in (
        request.profile_summary.model_copy(
            update={
                "resource_evidence": (
                    primary.model_copy(update={"local_run_sha256": "f" * 64}),
                    repeat,
                )
            }
        ),
        request.profile_summary.model_copy(
            update={
                "resource_evidence": (
                    primary,
                    repeat.model_copy(update={"local_artifact_sha256": "f" * 64}),
                )
            }
        ),
        request.profile_summary.model_copy(
            update={
                "resource_evidence": (
                    primary.model_copy(update={"environment_sha256": "f" * 64}),
                    repeat,
                )
            }
        ),
        request.profile_summary.model_copy(
            update={
                "resource_evidence": (
                    primary.model_copy(update={"logical_command_sha256": "f" * 64}),
                    repeat,
                )
            }
        ),
        request.profile_summary.model_copy(update={"elapsed_seconds": 99.0}),
    ):
        with pytest.raises(ArtifactValidationError, match="resource evidence"):
            _validate_request_links(replace(request, profile_summary=invalid_profile))

    with pytest.raises(ArtifactValidationError, match="resource evidence"):
        _validate_request_links(
            replace(
                request,
                repeat_verification=request.repeat_verification.model_copy(
                    update={"primary_local_artifact_sha256": "f" * 64}
                ),
            )
        )

    resource_check_index = tuple(check.check_id for check in request.validation.checks).index(
        "cpu-and-memory-caps"
    )
    invalid_checks = list(request.validation.checks)
    invalid_checks[resource_check_index] = invalid_checks[resource_check_index].model_copy(
        update={"evidence_sha256": "f" * 64}
    )
    with pytest.raises(ArtifactValidationError, match="CPU-and-memory validation"):
        _validate_request_links(
            replace(
                request,
                validation=request.validation.model_copy(update={"checks": tuple(invalid_checks)}),
            )
        )


def test_commitment_roles_are_the_exact_frozen_order_and_bind_repeat_summary() -> None:
    request = _request()
    commitments = request.source_commitments
    assert tuple(row.relative_role for row in commitments) == M5_SCIENTIFIC_SOURCE_ROLES

    arbitrary_two = (
        commitments[0].model_copy(update={"relative_role": "aggregate-members"}),
        commitments[1].model_copy(update={"relative_role": "validation-members"}),
    )
    mutations = (
        arbitrary_two,
        commitments[:-1],
        (
            *commitments,
            commitments[-1].model_copy(update={"relative_role": "extra-scientific-role"}),
        ),
        tuple(reversed(commitments)),
    )
    for invalid in mutations:
        with pytest.raises(ArtifactValidationError, match="frozen canonical set"):
            _validate_request_links(replace(request, source_commitments=invalid))

    with pytest.raises(
        ArtifactValidationError,
        match="repeat verification disagrees",
    ):
        _validate_request_links(
            replace(
                request,
                repeat_verification=request.repeat_verification.model_copy(
                    update={"scientific_member_count": len(M5_SCIENTIFIC_SOURCE_ROLES) - 1}
                ),
            )
        )
    reordered_bytes = canonical_replay_ndjson_bytes(tuple(reversed(commitments)))
    with pytest.raises(ArtifactValidationError, match="frozen canonical set"):
        _validate_request_links(
            replace(
                request,
                source_commitments=tuple(reversed(commitments)),
                repeat_verification=request.repeat_verification.model_copy(
                    update={
                        "source_member_commitments_sha256": hashlib.sha256(
                            reordered_bytes
                        ).hexdigest()
                    }
                ),
            )
        )


def test_ordering_rejects_records_outside_frozen_identity_authority() -> None:
    request = _request()
    unknown_identity = request.persistent_aggregates[0].identity.model_copy(
        update={"experiment_id": "unknown-experiment"}
    )
    rebound = request.persistent_aggregates[0].model_copy(
        update={
            "identity": unknown_identity,
            "replay_identity_sha256": replay_experiment_identity_sha256(unknown_identity),
        }
    )
    with pytest.raises(ArtifactValidationError, match="cannot be ordered"):
        replay_artifacts._ordered_records(
            replace(
                request,
                persistent_aggregates=(rebound, *request.persistent_aggregates[1:]),
            )
        )
    with pytest.raises(ArtifactValidationError, match="missing its experiment identity"):
        replay_artifacts._identity_digest(object())


def test_no_overwrite_rechecks_destination_after_opening_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "raced-destination"
    prepared = replay_artifacts._PreparedReplayArtifact(
        files={},
        artifact_sha256="a" * 64,
        run_sha256="b" * 64,
    )
    original_open = replay_artifacts.open_or_create_real_directory

    def open_parent_and_create_destination(parent: Path) -> int:
        descriptor = original_open(parent)
        destination.mkdir()
        return descriptor

    monkeypatch.setattr(
        replay_artifacts,
        "_prepare_replay_artifact",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        replay_artifacts,
        "open_or_create_real_directory",
        open_parent_and_create_destination,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_replay_curated_artifact(
            _request(),
            destination,
            source_root=ROOT,
            git_metadata_dirs=(),
        )
    assert destination.is_dir()


def test_public_loader_normalizes_missing_artifact_errors(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError):
        load_replay_curated_artifact(tmp_path / "missing")
