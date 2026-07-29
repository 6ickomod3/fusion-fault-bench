from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

import fusion_fault_bench.health_artifacts as health_artifacts_module
import fusion_fault_bench.health_release as release_module
from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    compute_run_record_digest,
    derive_run_id,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import SuccessMarkerV1Alpha1
from fusion_fault_bench.contracts.health_artifact_v1 import (
    HEALTH_AGGREGATES_FILE,
    HEALTH_CANDIDATES_FILE,
    HEALTH_ECDF_FILE,
    HEALTH_EVAL_ARTIFACT_CONTRACT,
    HEALTH_EVAL_INDEXED_PATHS,
    HEALTH_EVAL_VALIDATION_FILE,
    HEALTH_FIT_ARTIFACT_CONTRACT,
    HEALTH_FIT_INDEXED_PATHS,
    HEALTH_FIT_REFERENCE_FILE,
    HEALTH_FIT_SUMMARY_FILE,
    HEALTH_FIT_VALIDATION_FILE,
    HEALTH_INTENT_FILE,
    HEALTH_MAIN_PROFILE_FILE,
    HEALTH_PAYLOAD_INDEX_FILE,
    HEALTH_RUN_FILE,
    HEALTH_SEQUENCE_CONTRASTS_FILE,
    HEALTH_SEQUENCE_EVENTS_FILE,
    HEALTH_SEQUENCE_LOSSES_FILE,
    HEALTH_SUCCESS_FILE,
    HealthFitReferenceV1,
    HealthPayloadFileEntryV1,
    HealthPayloadIndexV1,
)
from fusion_fault_bench.contracts.health_result_v1 import (
    CROSS_THRESHOLDS,
    SELF_THRESHOLDS,
    HealthAggregateMetricV1,
    HealthFitSummaryV1,
    HealthThresholdCandidateV1,
    HealthValidationCheckV1,
    HealthValidationV1,
)
from fusion_fault_bench.contracts.health_v1 import load_health_benchmark_intent
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.health_artifacts import (
    HEALTH_ECDF_CHANNEL_ORDER,
    HEALTH_FIT_ARTIFACT_PATHS,
    HealthEcdfArraysV1,
    HealthEcdfChannelV1,
    LoadedHealthFitArtifact,
    compute_health_artifact_digest,
)
from fusion_fault_bench.health_benchmark import (
    expand_test_cases,
    load_health_population_profiles,
)
from fusion_fault_bench.health_release import (
    HEALTH_RELEASE_INDEXED_PATHS,
    AggregateQuantitativeClaimV1,
    FitQuantitativeClaimV1,
    HealthQuantitativeClaimV1,
    HealthReleaseValidationError,
    HealthReleaseWriteRequest,
    HealthRunResources,
    ResourceQuantitativeClaimV1,
    build_health_resource_measurement,
    load_health_release,
    validate_health_quantitative_claims,
    validate_health_release_candidate_bytes,
    write_health_release,
)

ROOT = Path(__file__).parents[1]
_CPU_MODEL = "Test M4 Release CPU"
_GIT_REVISION = "a" * 40
_LOCKFILE_SHA256 = "b" * 64
_SELECTED_CANDIDATE_INDEX = 35


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _EvaluationHandle:
    path: Path
    fit_reference: HealthFitReferenceV1
    aggregates: tuple[HealthAggregateMetricV1, ...]
    validation: HealthValidationV1
    payload_index: HealthPayloadIndexV1
    run: RunRecordV1Alpha1
    success: SuccessMarkerV1Alpha1
    artifact_sha256: str
    run_sha256: str


@dataclass(frozen=True, slots=True)
class _SourceSet:
    official_fit: LoadedHealthFitArtifact
    repeat_fit: LoadedHealthFitArtifact
    primary_evaluation: _EvaluationHandle
    repeat_evaluation: _EvaluationHandle
    aggregates: tuple[HealthAggregateMetricV1, ...]


def _validation(
    intent_sha256: str,
    *,
    checks: tuple[tuple[str, Any, Any], ...],
) -> HealthValidationV1:
    return HealthValidationV1(
        schema="ffb.health-validation/v1",
        intent_sha256=intent_sha256,
        checks=tuple(
            HealthValidationCheckV1(
                check_id=check_id,
                passed=True,
                observed=observed,
                expected=expected,
            )
            for check_id, observed, expected in checks
        ),
        all_checks_passed=True,
    )


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        python_version="3.12.13",
        os_name="Darwin",
        os_release="25.0",
        machine="arm64",
        cpu_model=_CPU_MODEL,
        logical_cpu_count=4,
        memory_bytes=16 * 1024**3,
    )


def _run(
    *,
    artifact_contract: str,
    artifact_sha256: str,
    run_id: str,
    intent_sha256: str,
    started_at: datetime,
    output_name: str,
) -> RunRecordV1Alpha1:
    fit = artifact_contract == HEALTH_FIT_ARTIFACT_CONTRACT
    command = (
        ("ffb", "health", "fit", "--output-dir", output_name)
        if fit
        else (
            "ffb",
            "health",
            "evaluate",
            "reports/generated/official-fit",
            "--output-dir",
            output_name,
        )
    )
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=intent_sha256,
        package_version="0.1.0",
        git_revision=_GIT_REVISION,
        source_dirty=False,
        lockfile_sha256=_LOCKFILE_SHA256,
        command=command,
        environment=_environment(),
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
        status="succeeded",
        artifact_sha256=artifact_sha256,
    )


def _payload_entry(
    path: str,
    value: bytes,
    *,
    record_count: int | None,
) -> HealthPayloadFileEntryV1:
    return HealthPayloadFileEntryV1(
        path=cast(Any, path),
        byte_length=len(value),
        sha256=hashlib.sha256(value).hexdigest(),
        record_count=record_count,
    )


def _write_source_tree(
    root: Path,
    *,
    indexed_paths: tuple[str, ...],
    scientific_files: dict[str, bytes],
    record_counts: dict[str, int],
    artifact_contract: str,
    intent_sha256: str,
    main_profile_sha256: str,
    edge_profile_sha256: str,
    started_at: datetime,
    output_name: str,
) -> tuple[HealthPayloadIndexV1, RunRecordV1Alpha1, SuccessMarkerV1Alpha1, str, str]:
    root.mkdir()
    run_id = derive_run_id(
        manifest_sha256=intent_sha256,
        git_revision=_GIT_REVISION,
        lockfile_sha256=_LOCKFILE_SHA256,
        package_version="0.1.0",
        artifact_contract=artifact_contract,
    )
    index = HealthPayloadIndexV1(
        schema="ffb.health-payload-index/v1",
        artifact_contract=cast(Any, artifact_contract),
        run_id=run_id,
        intent_sha256=intent_sha256,
        main_profile_sha256=main_profile_sha256,
        edge_profile_sha256=edge_profile_sha256,
        files=tuple(
            _payload_entry(
                path,
                scientific_files[path],
                record_count=record_counts.get(path),
            )
            for path in indexed_paths
        ),
    )
    index_bytes = canonical_json_bytes(index)
    artifact_sha256 = compute_health_artifact_digest(
        index_bytes,
        artifact_contract=cast(Any, artifact_contract),
    )
    run = _run(
        artifact_contract=artifact_contract,
        artifact_sha256=artifact_sha256,
        run_id=run_id,
        intent_sha256=intent_sha256,
        started_at=started_at,
        output_name=output_name,
    )
    run_bytes = canonical_json_bytes(run)
    run_sha256 = compute_run_record_digest(run_bytes)
    success = SuccessMarkerV1Alpha1(
        schema="ffb.success/v1alpha1",
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )
    for path, value in scientific_files.items():
        (root / path).write_bytes(value)
    (root / HEALTH_PAYLOAD_INDEX_FILE).write_bytes(index_bytes)
    (root / HEALTH_RUN_FILE).write_bytes(run_bytes)
    (root / HEALTH_SUCCESS_FILE).write_bytes(canonical_json_bytes(success))
    return index, run, success, artifact_sha256, run_sha256


def _fit_validation(
    intent_sha256: str,
    *,
    main_sha256: str,
    edge_sha256: str,
) -> HealthValidationV1:
    return _validation(
        intent_sha256,
        checks=(
            ("intent-digest", intent_sha256, intent_sha256),
            ("main-profile-digest", main_sha256, main_sha256),
            ("edge-profile-digest", edge_sha256, edge_sha256),
            ("train-sequence-count", 200, 200),
            ("validation-sequence-count", 200, 200),
            ("train-validation-sequence-id-overlap", 0, 0),
            ("validation-value-case-count", 33, 33),
            ("test-value-case-count", 47, 47),
            ("value-case-id-uniqueness", 80, 80),
            ("selection-value-case-count", 20, 20),
            ("identity-outside-active-event", 4_000, 4_000),
            ("ecdf-channel-count", 8, 8),
            ("ecdf-values-per-channel", 9_200, 9_200),
            ("threshold-candidate-count", 36, 36),
            ("threshold-candidate-order", True, True),
            ("metadata-leakage-boundary", True, True),
            ("future-prefix-causality", True, True),
            ("current-preupdate-causality", True, True),
            ("independent-modality-histories", True, True),
            ("frame-oracle-comparison-count", 144_000, 144_000),
            ("frame-oracle-dominance", 0, 0),
            ("candidate-frame-evaluation-cap", 7_257_600, 50_000_000),
            ("bootstrap-cell-cap", 400_000, 100_000_000),
            ("scientific-feature-trace-count", 4_400, 4_400),
            ("selected-candidate-feasible", True, True),
        ),
    )


def _evaluation_validation(intent_sha256: str) -> HealthValidationV1:
    return _validation(
        intent_sha256,
        checks=(
            ("intent-identity", intent_sha256, intent_sha256),
            ("condition-and-population-counts", 8_900, 8_900),
            ("condition-value-count", 47, 47),
            ("sequence-loss-row-count", 264_600, 264_600),
            ("sequence-event-row-count", 35_600, 35_600),
            ("sequence-contrast-row-count", 133_500, 133_500),
            ("oracle-frame-action-dominance", True, True),
            ("common-mode-hindsight-boundary", True, True),
            ("cold-start-schedule-windows", True, True),
            ("held-out-yaw-present", True, True),
            ("required-negative-controls-present", True, True),
            ("feature-computed-once-per-sequence-condition", True, True),
            ("test-fit-apply-only", True, True),
        ),
    )


def _ecdf() -> HealthEcdfArraysV1:
    return HealthEcdfArraysV1(
        schema="ffb.health-ecdf-arrays/v1",
        channels=tuple(
            HealthEcdfChannelV1(
                channel=channel,
                values=tuple(float(index) for _ in range(9_200)),
            )
            for index, channel in enumerate(HEALTH_ECDF_CHANNEL_ORDER)
        ),
    )


def _candidates() -> tuple[HealthThresholdCandidateV1, ...]:
    return tuple(
        HealthThresholdCandidateV1(
            schema="ffb.health-threshold-candidate/v1",
            candidate_index=index,
            self_threshold=SELF_THRESHOLDS[index // 6],
            cross_threshold=CROSS_THRESHOLDS[index % 6],
            mean_clean_regression_m2=0.0,
            upper_95pct_clean_regression_m2=0.0,
            false_alert_episode_starts_per_sequence=0.0,
            clean_coverage=1.0,
            fixed_clean_coverage=1.0,
            feasible=True,
            validation_regret_m2=float(35 - index),
        )
        for index in range(36)
    )


def _aggregates(intent: Any) -> tuple[HealthAggregateMetricV1, ...]:
    rows: list[HealthAggregateMetricV1] = []
    for case in expand_test_cases(intent):
        methods = tuple(
            method
            for method in intent.methods
            if not (
                case.fault.family == "common-mode-position-bias"
                and method
                in {
                    "fault-target-drop-policy",
                    "frame-action-performance-oracle",
                }
            )
        )
        required, _ = health_artifacts_module._expected_aggregate_keys(  # pyright: ignore[reportPrivateUsage]
            condition_id=case.condition_id,
            fault_family=case.fault.family,
            fault_target=case.fault.target,
            methods=methods,
        )
        sequence_count = 100 if case.population == "edge-test" else 200
        rows.extend(
            HealthAggregateMetricV1(
                schema="ffb.health-aggregate/v1",
                condition_id=condition_id,
                method=cast(Any, method),
                metric_name=metric_name,
                window=cast(Any, window),
                unit="fraction",
                status="ok",
                estimate=0.1,
                interval_lower=0.1,
                interval_upper=0.1,
                sequence_count=sequence_count,
                bootstrap_replicates=2000,
                defined_bootstrap_replicates=2000,
            )
            for condition_id, method, metric_name, window in required
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.condition_id,
                "" if row.method is None else row.method,
                row.metric_name,
                "" if row.window is None else row.window,
            ),
        )
    )


def _make_sources(tmp_path: Path) -> _SourceSet:
    tmp_path.mkdir(parents=True, exist_ok=True)
    loaded_intent = load_health_benchmark_intent(source_root=ROOT)
    intent = loaded_intent.intent
    profiles = load_health_population_profiles(intent, source_root=ROOT)
    main_profile = profiles.main_profile
    edge_profile = profiles.edge_profile
    intent_sha256 = sha256_digest(intent)
    main_sha256 = sha256_digest(main_profile)
    edge_sha256 = sha256_digest(edge_profile)
    fit_validation = _fit_validation(
        intent_sha256,
        main_sha256=main_sha256,
        edge_sha256=edge_sha256,
    )
    evaluation_validation = _evaluation_validation(intent_sha256)
    fit_summary = HealthFitSummaryV1(
        schema="ffb.health-fit-summary/v1",
        intent_sha256=intent_sha256,
        main_profile_sha256=main_sha256,
        edge_profile_sha256=edge_sha256,
        train_sequence_count=200,
        validation_sequence_count=200,
        ecdf_channel_count=8,
        ecdf_values_per_channel=9200,
        candidate_count=36,
        selected_candidate_index=_SELECTED_CANDIDATE_INDEX,
        selected_self_threshold=SELF_THRESHOLDS[_SELECTED_CANDIDATE_INDEX // 6],
        selected_cross_threshold=CROSS_THRESHOLDS[_SELECTED_CANDIDATE_INDEX % 6],
        selection_status="selected",
    )
    candidates = b"".join(canonical_json_bytes(candidate) for candidate in _candidates())
    fit_files = {
        HEALTH_INTENT_FILE: canonical_json_bytes(intent),
        HEALTH_MAIN_PROFILE_FILE: canonical_json_bytes(main_profile),
        "edge-profile.json": canonical_json_bytes(edge_profile),
        HEALTH_ECDF_FILE: canonical_json_bytes(_ecdf()),
        HEALTH_CANDIDATES_FILE: candidates,
        HEALTH_FIT_SUMMARY_FILE: canonical_json_bytes(fit_summary),
        HEALTH_FIT_VALIDATION_FILE: canonical_json_bytes(fit_validation),
    }
    fit_record_counts = {HEALTH_CANDIDATES_FILE: 36}
    now = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)
    official_fit_path = tmp_path / "official-fit"
    repeat_fit_path = tmp_path / "repeat-fit"
    official_fit_parts = _write_source_tree(
        official_fit_path,
        indexed_paths=HEALTH_FIT_INDEXED_PATHS,
        scientific_files=fit_files,
        record_counts=fit_record_counts,
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
        intent_sha256=intent_sha256,
        main_profile_sha256=main_sha256,
        edge_profile_sha256=edge_sha256,
        started_at=now,
        output_name="reports/generated/official-fit",
    )
    repeat_fit_parts = _write_source_tree(
        repeat_fit_path,
        indexed_paths=HEALTH_FIT_INDEXED_PATHS,
        scientific_files=fit_files,
        record_counts=fit_record_counts,
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
        intent_sha256=intent_sha256,
        main_profile_sha256=main_sha256,
        edge_profile_sha256=edge_sha256,
        started_at=now + timedelta(minutes=1),
        output_name="reports/generated/repeat-fit",
    )
    official_fit = LoadedHealthFitArtifact(
        path=official_fit_path,
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        calibration=cast(Any, None),
        candidates=(),
        summary=fit_summary,
        validation=fit_validation,
        payload_index=official_fit_parts[0],
        run=official_fit_parts[1],
        success=official_fit_parts[2],
        artifact_sha256=official_fit_parts[3],
        run_sha256=official_fit_parts[4],
    )
    repeat_fit = LoadedHealthFitArtifact(
        path=repeat_fit_path,
        intent=intent,
        main_profile=main_profile,
        edge_profile=edge_profile,
        calibration=cast(Any, None),
        candidates=(),
        summary=fit_summary,
        validation=fit_validation,
        payload_index=repeat_fit_parts[0],
        run=repeat_fit_parts[1],
        success=repeat_fit_parts[2],
        artifact_sha256=repeat_fit_parts[3],
        run_sha256=repeat_fit_parts[4],
    )
    fit_reference = HealthFitReferenceV1(
        schema="ffb.health-fit-reference/v1",
        fit_artifact_sha256=official_fit.artifact_sha256,
        fit_run_sha256=official_fit.run_sha256,
        intent_sha256=intent_sha256,
        selected_candidate_index=fit_summary.selected_candidate_index,
        selected_self_threshold=fit_summary.selected_self_threshold,
        selected_cross_threshold=fit_summary.selected_cross_threshold,
    )
    aggregates = _aggregates(intent)
    aggregate_bytes = b"".join(canonical_json_bytes(record) for record in aggregates)
    evaluation_files = {
        HEALTH_INTENT_FILE: canonical_json_bytes(intent),
        HEALTH_MAIN_PROFILE_FILE: canonical_json_bytes(main_profile),
        "edge-profile.json": canonical_json_bytes(edge_profile),
        HEALTH_FIT_REFERENCE_FILE: canonical_json_bytes(fit_reference),
        HEALTH_SEQUENCE_LOSSES_FILE: b"{}\n",
        HEALTH_SEQUENCE_CONTRASTS_FILE: b"{}\n",
        HEALTH_SEQUENCE_EVENTS_FILE: b"{}\n",
        HEALTH_AGGREGATES_FILE: aggregate_bytes,
        HEALTH_EVAL_VALIDATION_FILE: canonical_json_bytes(evaluation_validation),
    }
    evaluation_counts = {
        HEALTH_SEQUENCE_LOSSES_FILE: 264_600,
        HEALTH_SEQUENCE_CONTRASTS_FILE: 133_500,
        HEALTH_SEQUENCE_EVENTS_FILE: 35_600,
        HEALTH_AGGREGATES_FILE: len(aggregates),
    }
    primary_evaluation_path = tmp_path / "primary-evaluation"
    repeat_evaluation_path = tmp_path / "repeat-evaluation"
    primary_evaluation_parts = _write_source_tree(
        primary_evaluation_path,
        indexed_paths=HEALTH_EVAL_INDEXED_PATHS,
        scientific_files=evaluation_files,
        record_counts=evaluation_counts,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
        intent_sha256=intent_sha256,
        main_profile_sha256=main_sha256,
        edge_profile_sha256=edge_sha256,
        started_at=now + timedelta(minutes=2),
        output_name="reports/generated/primary-evaluation",
    )
    repeat_evaluation_parts = _write_source_tree(
        repeat_evaluation_path,
        indexed_paths=HEALTH_EVAL_INDEXED_PATHS,
        scientific_files=evaluation_files,
        record_counts=evaluation_counts,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
        intent_sha256=intent_sha256,
        main_profile_sha256=main_sha256,
        edge_profile_sha256=edge_sha256,
        started_at=now + timedelta(minutes=3),
        output_name="reports/generated/repeat-evaluation",
    )

    def evaluation(
        path: Path,
        parts: tuple[
            HealthPayloadIndexV1,
            RunRecordV1Alpha1,
            SuccessMarkerV1Alpha1,
            str,
            str,
        ],
    ) -> _EvaluationHandle:
        return _EvaluationHandle(
            path=path,
            fit_reference=fit_reference,
            aggregates=aggregates,
            validation=evaluation_validation,
            payload_index=parts[0],
            run=parts[1],
            success=parts[2],
            artifact_sha256=parts[3],
            run_sha256=parts[4],
        )

    return _SourceSet(
        official_fit=official_fit,
        repeat_fit=repeat_fit,
        primary_evaluation=evaluation(
            primary_evaluation_path,
            primary_evaluation_parts,
        ),
        repeat_evaluation=evaluation(
            repeat_evaluation_path,
            repeat_evaluation_parts,
        ),
        aggregates=aggregates,
    )


def _patch_source_loaders(
    monkeypatch: pytest.MonkeyPatch,
    sources: _SourceSet,
) -> None:
    fits = {
        sources.official_fit.path: sources.official_fit,
        sources.repeat_fit.path: sources.repeat_fit,
    }
    evaluations = {
        sources.primary_evaluation.path: sources.primary_evaluation,
        sources.repeat_evaluation.path: sources.repeat_evaluation,
    }

    def load_fit(path: Path) -> LoadedHealthFitArtifact:
        return fits[path]

    def load_evaluation(
        path: Path,
        *,
        fit_artifact: LoadedHealthFitArtifact,
    ) -> _EvaluationHandle:
        assert fit_artifact.artifact_sha256 == sources.official_fit.artifact_sha256
        return evaluations[path]

    monkeypatch.setattr(release_module, "load_health_fit_artifact", load_fit)
    monkeypatch.setattr(
        release_module,
        "load_health_evaluation_artifact",
        load_evaluation,
    )


def _claims(sources: _SourceSet) -> tuple[HealthQuantitativeClaimV1, ...]:
    return (
        AggregateQuantitativeClaimV1(
            schema="ffb.health-quantitative-claim/v1",
            source_kind="aggregate",
            claim_id="aggregate-clean-fixed",
            presentation_id="figure-health-summary",
            aggregate=sources.aggregates[0],
        ),
        FitQuantitativeClaimV1(
            schema="ffb.health-quantitative-claim/v1",
            source_kind="fit-summary",
            claim_id="selected-candidate",
            presentation_id="table-fit-summary",
            field="selected-candidate-index",
            value=float(_SELECTED_CANDIDATE_INDEX),
            unit="count",
        ),
        ResourceQuantitativeClaimV1(
            schema="ffb.health-quantitative-claim/v1",
            source_kind="resource",
            claim_id="primary-evaluation-wall",
            presentation_id="table-resource-summary",
            run_label="primary-evaluation",
            metric="wall-time-seconds",
            value=4.0,
            unit="s",
            cpu_model=_CPU_MODEL,
            evidence_scope=("operator-recorded-time-l-sidecar-not-independent-attestation"),
        ),
    )


def _request(sources: _SourceSet) -> HealthReleaseWriteRequest:
    def resources(
        run_label: Any,
        artifact: Any,
        wall_time_seconds: float,
        peak_rss_bytes: int,
    ) -> HealthRunResources:
        time_l_log = (
            f"        {wall_time_seconds:.2f} real         0.50 user"
            f"         0.25 sys\n{peak_rss_bytes:20d}"
            "  maximum resident set size\n"
        ).encode("ascii")
        return HealthRunResources(
            time_l_log=time_l_log,
            measurement=build_health_resource_measurement(
                run_label,
                artifact,
                time_l_log,
            ),
        )

    return HealthReleaseWriteRequest(
        official_fit_path=sources.official_fit.path,
        repeat_fit_path=sources.repeat_fit.path,
        primary_evaluation_path=sources.primary_evaluation.path,
        repeat_evaluation_path=sources.repeat_evaluation.path,
        primary_fit_resources=resources(
            "primary-fit",
            sources.official_fit,
            2.0,
            100_000_000,
        ),
        repeat_fit_resources=resources(
            "repeat-fit",
            sources.repeat_fit,
            2.1,
            101_000_000,
        ),
        primary_evaluation_resources=resources(
            "primary-evaluation",
            sources.primary_evaluation,
            4.0,
            110_000_000,
        ),
        repeat_evaluation_resources=resources(
            "repeat-evaluation",
            sources.repeat_evaluation,
            4.1,
            111_000_000,
        ),
        quantitative_claims=_claims(sources),
    )


def _reseal_release(
    root: Path,
    *,
    record_count_overrides: dict[str, int] | None = None,
) -> None:
    overrides = {} if record_count_overrides is None else record_count_overrides
    index_path = root / release_module.HEALTH_RELEASE_INDEX_FILE
    original_index = release_module.HealthReleaseIndexV1.model_validate_json(
        index_path.read_bytes()
    )
    updated_entries = tuple(
        entry.model_copy(
            update={
                "byte_length": len(value := (root / entry.path).read_bytes()),
                "sha256": hashlib.sha256(value).hexdigest(),
                "record_count": overrides.get(entry.path, entry.record_count),
            }
        )
        for entry in original_index.files
    )
    updated_index = release_module.HealthReleaseIndexV1.model_validate(
        original_index.model_copy(update={"files": updated_entries}),
    )
    index_bytes = canonical_json_bytes(updated_index)
    index_path.write_bytes(index_bytes)
    success = release_module.HealthReleaseSuccessV1(
        schema="ffb.health-release-success/v1",
        release_artifact_sha256=release_module.compute_health_release_digest(index_bytes),
        release_summary_sha256=hashlib.sha256(
            (root / release_module.HEALTH_RELEASE_SUMMARY_FILE).read_bytes()
        ).hexdigest(),
        repeat_verification_sha256=hashlib.sha256(
            (root / release_module.HEALTH_RELEASE_REPEAT_FILE).read_bytes()
        ).hexdigest(),
    )
    (root / release_module.HEALTH_RELEASE_SUCCESS_FILE).write_bytes(canonical_json_bytes(success))


def test_release_round_trip_preserves_fit_aggregates_and_exact_omissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _make_sources(tmp_path)
    _patch_source_loaders(monkeypatch, sources)

    release = write_health_release(
        _request(sources),
        tmp_path / "curated-release",
        git_metadata_dirs=(),
    )

    assert release.summary.condition_count == 47
    assert release.summary.aggregate_record_count == len(sources.aggregates)
    assert release.summary.quantitative_claim_count == 3
    assert release.summary.omitted_sequence_member_count == 3
    assert release.summary.omitted_sequence_record_count == 433_700
    assert release.repeat.total_member_comparison_count == 16
    assert release.repeat.mismatch_count == 0
    assert release.repeat.all_checks_passed
    assert len(release.commitments) == 16
    assert tuple(
        item.path
        for item in release.commitments
        if item.retention_scope == "omitted-commitment-only-not-independently-recomputable"
    ) == (
        HEALTH_SEQUENCE_LOSSES_FILE,
        HEALTH_SEQUENCE_CONTRASTS_FILE,
        HEALTH_SEQUENCE_EVENTS_FILE,
    )
    fit_intent_commitment = next(
        item
        for item in release.commitments
        if item.artifact_kind == "fit" and item.path == HEALTH_INTENT_FILE
    )
    aggregate_commitment = next(
        item
        for item in release.commitments
        if item.artifact_kind == "evaluation" and item.path == HEALTH_AGGREGATES_FILE
    )
    assert fit_intent_commitment.primary_retained_release_path == "primary-fit-intent.json"
    assert fit_intent_commitment.repeat_retained_release_path == "repeat-fit-intent.json"
    assert aggregate_commitment.primary_retained_release_path == "aggregate-metrics.ndjson"
    assert aggregate_commitment.repeat_retained_release_path == "aggregate-metrics.ndjson"
    assert not any(
        path in {member.name for member in release.path.iterdir()}
        for path in (
            HEALTH_SEQUENCE_LOSSES_FILE,
            HEALTH_SEQUENCE_CONTRASTS_FILE,
            HEALTH_SEQUENCE_EVENTS_FILE,
        )
    )
    assert (release.path / "aggregate-metrics.ndjson").read_bytes() == (
        sources.primary_evaluation.path / HEALTH_AGGREGATES_FILE
    ).read_bytes()
    for source_path in HEALTH_FIT_ARTIFACT_PATHS:
        normalized = "success.json" if source_path == HEALTH_SUCCESS_FILE else source_path
        release_path = f"primary-fit-{normalized}"
        assert (release.path / release_path).read_bytes() == (
            sources.official_fit.path / source_path
        ).read_bytes()
    assert tuple(entry.path for entry in release.release_index.files) == (
        HEALTH_RELEASE_INDEXED_PATHS
    )
    assert load_health_release(release.path).release_artifact_sha256 == (
        release.release_artifact_sha256
    )
    with pytest.raises(FileExistsError):
        write_health_release(
            _request(sources),
            release.path,
            git_metadata_dirs=(),
        )


def test_claim_validation_rejects_changed_aggregate_fit_and_resource_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _make_sources(tmp_path)
    _patch_source_loaders(monkeypatch, sources)
    release = write_health_release(
        _request(sources),
        tmp_path / "claim-release",
        git_metadata_dirs=(),
    )
    repeat = release.repeat
    valid = _claims(sources)
    assert validate_health_quantitative_claims(
        valid,
        aggregates=sources.aggregates,
        fit_summary=sources.official_fit.summary,
        repeat=repeat,
    ) == tuple(sorted(valid, key=lambda claim: claim.claim_id))

    wrong_aggregate = cast(
        AggregateQuantitativeClaimV1,
        valid[0],
    ).model_copy(update={"aggregate": sources.aggregates[0].model_copy(update={"estimate": 99.0})})
    wrong_fit = cast(FitQuantitativeClaimV1, valid[1]).model_copy(update={"value": 1.0})
    wrong_resource = cast(ResourceQuantitativeClaimV1, valid[2]).model_copy(update={"value": 5.0})
    for wrong in (wrong_aggregate, wrong_fit, wrong_resource):
        claims = (wrong, *tuple(item for item in valid if item.claim_id != wrong.claim_id))
        with pytest.raises(HealthReleaseValidationError, match=r"claim disagrees|not an exact"):
            validate_health_quantitative_claims(
                claims,
                aggregates=sources.aggregates,
                fit_summary=sources.official_fit.summary,
                repeat=repeat,
            )


def test_aggregate_structure_and_raw_resource_provenance_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _make_sources(tmp_path)
    _patch_source_loaders(monkeypatch, sources)
    assert (
        health_artifacts_module.validate_health_aggregate_structure(
            sources.official_fit.intent,
            sources.official_fit.main_profile,
            sources.official_fit.edge_profile,
            sources.aggregates,
        )
        == sources.aggregates
    )
    wrong_denominator = (
        sources.aggregates[0].model_copy(
            update={"sequence_count": sources.aggregates[0].sequence_count - 1}
        ),
        *sources.aggregates[1:],
    )
    with pytest.raises(ArtifactValidationError, match="denominator"):
        health_artifacts_module.validate_health_aggregate_structure(
            sources.official_fit.intent,
            sources.official_fit.main_profile,
            sources.official_fit.edge_profile,
            wrong_denominator,
        )

    request = _request(sources)
    malformed = replace(
        request,
        primary_fit_resources=replace(
            request.primary_fit_resources,
            time_l_log=b"2.00 real 0.50 user 0.25 sys\n",
        ),
    )
    with pytest.raises(HealthReleaseValidationError, match="exactly one"):
        write_health_release(
            malformed,
            tmp_path / "malformed-resource-release",
            git_metadata_dirs=(),
        )
    unbound = replace(
        request,
        primary_fit_resources=replace(
            request.primary_fit_resources,
            measurement=request.primary_fit_resources.measurement.model_copy(
                update={"run_sha256": _digest("unbound-resource-sidecar")}
            ),
        ),
    )
    with pytest.raises(HealthReleaseValidationError, match="sidecar"):
        write_health_release(
            unbound,
            tmp_path / "unbound-resource-release",
            git_metadata_dirs=(),
        )


def test_repeat_gate_rejects_scientific_mismatch_and_hard_linked_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _make_sources(tmp_path)
    _patch_source_loaders(monkeypatch, sources)
    request = _request(sources)

    files = list(sources.repeat_evaluation.payload_index.files)
    aggregate_index = next(
        index for index, entry in enumerate(files) if entry.path == HEALTH_AGGREGATES_FILE
    )
    files[aggregate_index] = files[aggregate_index].model_copy(
        update={"sha256": _digest("repeat-aggregate-mismatch")}
    )
    mismatched_index = sources.repeat_evaluation.payload_index.model_copy(
        update={"files": tuple(files)}
    )
    mismatched_evaluation = replace(
        sources.repeat_evaluation,
        payload_index=mismatched_index,
        artifact_sha256=_digest("repeat-evaluation-artifact-mismatch"),
    )

    def load_mismatched_evaluation(
        path: Path,
        *,
        fit_artifact: LoadedHealthFitArtifact,
    ) -> _EvaluationHandle:
        del fit_artifact
        return (
            sources.primary_evaluation
            if path == sources.primary_evaluation.path
            else mismatched_evaluation
        )

    monkeypatch.setattr(
        release_module,
        "load_health_evaluation_artifact",
        load_mismatched_evaluation,
    )
    request = _request(
        replace(
            sources,
            repeat_evaluation=mismatched_evaluation,
        )
    )
    with pytest.raises(HealthReleaseValidationError, match="release gates did not all pass"):
        write_health_release(
            request,
            tmp_path / "mismatched-release",
            git_metadata_dirs=(),
        )

    sources = _make_sources(tmp_path / "hardlink-sources")
    _patch_source_loaders(monkeypatch, sources)
    repeat_member = sources.repeat_fit.path / HEALTH_INTENT_FILE
    repeat_member.unlink()
    os.link(sources.official_fit.path / HEALTH_INTENT_FILE, repeat_member)
    with pytest.raises(HealthReleaseValidationError, match="independent inodes"):
        write_health_release(
            _request(sources),
            tmp_path / "hardlinked-release",
            git_metadata_dirs=(),
        )


def test_release_loader_rejects_mutation_extra_file_and_private_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _make_sources(tmp_path)
    _patch_source_loaders(monkeypatch, sources)
    release = write_health_release(
        _request(sources),
        tmp_path / "release",
        git_metadata_dirs=(),
    )
    aggregate_path = release.path / "aggregate-metrics.ndjson"
    aggregate_path.write_bytes(aggregate_path.read_bytes() + b"{}\n")
    with pytest.raises(HealthReleaseValidationError, match="release member disagrees"):
        load_health_release(release.path)

    sources = _make_sources(tmp_path / "extra-sources")
    _patch_source_loaders(monkeypatch, sources)
    release = write_health_release(
        _request(sources),
        tmp_path / "extra-release",
        git_metadata_dirs=(),
    )
    (release.path / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(HealthReleaseValidationError, match="allowlist"):
        load_health_release(release.path)

    sources = _make_sources(tmp_path / "semantic-sources")
    _patch_source_loaders(monkeypatch, sources)
    semantic_base = write_health_release(
        _request(sources),
        tmp_path / "semantic-base",
        git_metadata_dirs=(),
    )
    semantic_root = tmp_path / "semantic-release"
    shutil.copytree(semantic_base.path, semantic_root, dirs_exist_ok=False)
    count_root = tmp_path / "count-release"
    shutil.copytree(semantic_base.path, count_root, dirs_exist_ok=False)
    resource_root = tmp_path / "resource-release"
    shutil.copytree(semantic_base.path, resource_root, dirs_exist_ok=False)
    commitment_path = semantic_root / release_module.HEALTH_RELEASE_COMMITMENTS_FILE
    commitments = tuple(
        release_module.HealthSourceMemberCommitmentV1.model_validate_json(line)
        for line in commitment_path.read_bytes().splitlines(keepends=True)
    )
    commitments = (
        commitments[0].model_copy(
            update={"primary_artifact_sha256": _digest("unbound-fit-artifact")}
        ),
        *commitments[1:],
    )
    commitment_bytes = b"".join(canonical_json_bytes(item) for item in commitments)
    commitment_path.write_bytes(commitment_bytes)
    repeat_path = semantic_root / release_module.HEALTH_RELEASE_REPEAT_FILE
    repeat = release_module.HealthRepeatVerificationV1.model_validate_json(
        repeat_path.read_bytes()
    ).model_copy(
        update={"source_member_commitments_sha256": hashlib.sha256(commitment_bytes).hexdigest()}
    )
    repeat_path.write_bytes(canonical_json_bytes(repeat))
    _reseal_release(semantic_root)
    with pytest.raises(HealthReleaseValidationError, match="member commitments"):
        load_health_release(semantic_root)

    _reseal_release(
        count_root,
        record_count_overrides={"aggregate-metrics.ndjson": 46},
    )
    with pytest.raises(HealthReleaseValidationError, match="record count"):
        load_health_release(count_root)

    resource_log = resource_root / "primary-fit-time-l.txt"
    resource_log.write_bytes(resource_log.read_bytes().replace(b"100000000", b"100000001"))
    _reseal_release(resource_root)
    with pytest.raises(HealthReleaseValidationError, match="resource evidence"):
        load_health_release(resource_root)

    with pytest.raises(HealthReleaseValidationError, match="privacy scan"):
        validate_health_release_candidate_bytes(
            {"candidate.json": b'{"dataset":"/Users/private/dataset"}\n'}
        )
