from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import fusion_fault_bench.health_artifacts as health_artifact_module
from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    compute_run_record_digest,
    derive_run_id,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import SuccessMarkerV1Alpha1
from fusion_fault_bench.contracts.health_artifact_v1 import (
    HEALTH_CANDIDATES_FILE,
    HEALTH_ECDF_FILE,
    HEALTH_EVAL_ARTIFACT_CONTRACT,
    HEALTH_FIT_ARTIFACT_CONTRACT,
    HEALTH_FIT_REFERENCE_FILE,
    HEALTH_PAYLOAD_INDEX_FILE,
    HEALTH_RUN_FILE,
    HEALTH_SEQUENCE_CONTRASTS_FILE,
    HEALTH_SEQUENCE_LOSSES_FILE,
    HEALTH_SUCCESS_FILE,
    HealthArtifactContract,
    HealthPayloadIndexV1,
)
from fusion_fault_bench.contracts.health_result_v1 import (
    CROSS_THRESHOLDS,
    SELF_THRESHOLDS,
    HealthAggregateMetricV1,
    HealthFitSummaryV1,
    HealthSequenceContrastV1,
    HealthSequenceEventV1,
    HealthSequenceLossV1,
    HealthThresholdCandidateV1,
    HealthValidationCheckV1,
    HealthValidationV1,
)
from fusion_fault_bench.contracts.health_v1 import (
    HEALTH_BENCHMARK_INTENT_ADAPTER,
    HealthBenchmarkIntentV1,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    EdgeProceduralProfile,
    MainProceduralProfile,
    load_procedural_profile,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.health import HealthCalibration
from fusion_fault_bench.health_artifacts import (
    HEALTH_ECDF_CHANNEL_ORDER,
    HEALTH_EVAL_ARTIFACT_PATHS,
    HEALTH_FIT_ARTIFACT_PATHS,
    HealthEvaluationArtifactTransaction,
    HealthEvaluationArtifactWriteRequest,
    HealthFitArtifactWriteRequest,
    LoadedHealthEvaluationArtifact,
    LoadedHealthFitArtifact,
    build_health_fit_reference,
    compute_health_artifact_digest,
    load_health_evaluation_artifact,
    load_health_fit_artifact,
    write_health_evaluation_artifact,
    write_health_fit_artifact,
)
from fusion_fault_bench.scenarios.health import HealthFaultSpec

REPOSITORY_ROOT = Path(__file__).parents[1]
INTENT_PATH = REPOSITORY_ROOT / "examples/health/m4-health-v1.json"
MAIN_PROFILE_PATH = REPOSITORY_ROOT / "examples/profiles/constant-velocity-front-roi-v1.json"
EDGE_PROFILE_PATH = REPOSITORY_ROOT / "examples/profiles/constant-velocity-fov-edge-v1.json"
GIT_REVISION = "a" * 40
LOCKFILE_SHA256 = "b" * 64
PLACEHOLDER_DIGEST = "0" * 64
_EXACT_EVALUATION_ROWS_VALIDATOR = vars(health_artifact_module)["_require_exact_evaluation_rows"]


def _intent() -> HealthBenchmarkIntentV1:
    return HEALTH_BENCHMARK_INTENT_ADAPTER.validate_json(INTENT_PATH.read_text(encoding="utf-8"))


def _profiles() -> tuple[MainProceduralProfile, EdgeProceduralProfile]:
    main = load_procedural_profile(MAIN_PROFILE_PATH)
    edge = load_procedural_profile(EDGE_PROFILE_PATH)
    assert isinstance(main, MainProceduralProfile)
    assert isinstance(edge, EdgeProceduralProfile)
    return main, edge


def _run(
    intent: HealthBenchmarkIntentV1,
    *,
    artifact_contract: HealthArtifactContract,
    started_at: datetime | None = None,
) -> RunRecordV1Alpha1:
    intent_sha256 = sha256_digest(intent)
    started = datetime(2026, 7, 29, 12, 0, tzinfo=UTC) if started_at is None else started_at
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=derive_run_id(
            manifest_sha256=intent_sha256,
            git_revision=GIT_REVISION,
            lockfile_sha256=LOCKFILE_SHA256,
            package_version="0.1.0",
            artifact_contract=artifact_contract,
        ),
        manifest_sha256=intent_sha256,
        package_version="0.1.0",
        git_revision=GIT_REVISION,
        source_dirty=False,
        lockfile_sha256=LOCKFILE_SHA256,
        command=("ffb", "health", "fit" if "fit" in artifact_contract else "evaluate"),
        environment=RuntimeEnvironment(
            python_version="3.12.13",
            os_name="Darwin",
            os_release="24.5.0",
            machine="arm64",
            cpu_model="Test CPU",
            logical_cpu_count=4,
            memory_bytes=8 * 1024**3,
        ),
        started_at=started,
        ended_at=started + timedelta(seconds=1),
        status="succeeded",
        artifact_sha256=PLACEHOLDER_DIGEST,
    )


def _calibration() -> HealthCalibration:
    arrays = [
        np.linspace(float(index), float(index + 1), 9200, dtype=np.float64) for index in range(8)
    ]
    return HealthCalibration(
        camera_self_mean=arrays[0],
        camera_self_maximum=arrays[1],
        lidar_self_mean=arrays[2],
        lidar_self_maximum=arrays[3],
        camera_from_lidar_cross_mean=arrays[4],
        camera_from_lidar_cross_maximum=arrays[5],
        lidar_from_camera_cross_mean=arrays[6],
        lidar_from_camera_cross_maximum=arrays[7],
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


def _fit_validation(intent: HealthBenchmarkIntentV1) -> HealthValidationV1:
    digest = sha256_digest(intent)
    values = (
        ("intent-digest", digest, digest),
        (
            "main-profile-digest",
            intent.source_population.profile_sha256,
            intent.source_population.profile_sha256,
        ),
        (
            "edge-profile-digest",
            intent.source_population.edge_profile_sha256,
            intent.source_population.edge_profile_sha256,
        ),
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
    )
    return HealthValidationV1(
        schema="ffb.health-validation/v1",
        intent_sha256=digest,
        checks=tuple(
            HealthValidationCheckV1(
                check_id=check_id,
                passed=True,
                observed=observed,
                expected=expected,
            )
            for check_id, observed, expected in values
        ),
        all_checks_passed=True,
    )


def _evaluation_validation(intent: HealthBenchmarkIntentV1) -> HealthValidationV1:
    digest = sha256_digest(intent)
    values = (
        ("intent-identity", digest, digest),
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
    )
    return HealthValidationV1(
        schema="ffb.health-validation/v1",
        intent_sha256=digest,
        checks=tuple(
            HealthValidationCheckV1(
                check_id=check_id,
                passed=True,
                observed=observed,
                expected=expected,
            )
            for check_id, observed, expected in values
        ),
        all_checks_passed=True,
    )


def _fit_request() -> HealthFitArtifactWriteRequest:
    intent = _intent()
    main, edge = _profiles()
    return HealthFitArtifactWriteRequest(
        intent=intent,
        main_profile=main,
        edge_profile=edge,
        calibration=_calibration(),
        candidates=tuple(reversed(_candidates())),
        summary=HealthFitSummaryV1(
            schema="ffb.health-fit-summary/v1",
            intent_sha256=sha256_digest(intent),
            main_profile_sha256=sha256_digest(main),
            edge_profile_sha256=sha256_digest(edge),
            train_sequence_count=200,
            validation_sequence_count=200,
            ecdf_channel_count=8,
            ecdf_values_per_channel=9200,
            candidate_count=36,
            selected_candidate_index=35,
            selected_self_threshold=1.0,
            selected_cross_threshold=1.0,
            selection_status="selected",
        ),
        validation=_fit_validation(intent),
        run=_run(intent, artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT),
    )


def _sequence_losses() -> tuple[HealthSequenceLossV1, ...]:
    return (
        HealthSequenceLossV1(
            schema="ffb.health-sequence-loss/v1",
            sequence_id="sequence:000",
            condition_id="condition-a",
            method="fixed-fusion",
            window="event",
            loss_sum_m2=2.0,
            valid_object_frame_count=10,
            eligible_object_frame_count=10,
        ),
        HealthSequenceLossV1(
            schema="ffb.health-sequence-loss/v1",
            sequence_id="sequence:000",
            condition_id="condition-a",
            method="camera-only",
            window="event",
            loss_sum_m2=3.0,
            valid_object_frame_count=10,
            eligible_object_frame_count=10,
        ),
    )


def _sequence_event(policy: str) -> HealthSequenceEventV1:
    return HealthSequenceEventV1.model_validate(
        {
            "schema": "ffb.health-sequence-event/v1",
            "sequence_id": "sequence:000",
            "condition_id": "condition-a",
            "policy": policy,
            "detected": True,
            "detection_latency_frames": 2,
            "first_latch_label": "camera-fault",
            "outcome": "correct",
            "correctly_attributed": True,
            "attribution_latency_frames": 2,
            "realized_dropout": False,
            "first_missing_frame_minus_event_start": None,
            "detection_minus_first_missing_frames": None,
            "latch_episode_count": 1,
            "false_alert_episode_count": 0,
            "early_clear": False,
            "final_active_state": "camera-fault",
            "active_frame_count": 24,
            "active_healthy_frames": 0,
            "active_camera_fault_frames": 24,
            "active_lidar_fault_frames": 0,
            "active_ambiguous_frames": 0,
            "active_camera_action_frames": 0,
            "active_lidar_action_frames": 24,
            "active_fixed_action_frames": 0,
            "active_undefined_action_frames": 0,
            "recovery_eligible": True,
            "recovered": True,
            "recovery_latency_frames": 3,
        }
    )


def _sequence_contrast(policy: str) -> HealthSequenceContrastV1:
    return HealthSequenceContrastV1.model_validate(
        {
            "schema": "ffb.health-sequence-contrast/v1",
            "sequence_id": "sequence:000",
            "condition_id": "condition-a",
            "policy": policy,
            "window": "event",
            "fixed_support_sha256": "c" * 64,
            "policy_support_sha256": "d" * 64,
            "fixed_policy_common_count": 10,
            "fixed_on_common_loss_sum_m2": 2.0,
            "policy_on_fixed_common_loss_sum_m2": 1.0,
            "target_drop_applicable": False,
            "policy_target_drop_common_count": None,
            "policy_on_target_common_loss_sum_m2": None,
            "target_drop_on_common_loss_sum_m2": None,
            "frame_oracle_applicable": False,
            "policy_frame_oracle_common_count": None,
            "policy_on_oracle_common_loss_sum_m2": None,
            "frame_oracle_on_common_loss_sum_m2": None,
            "frame_oracle_support_sha256": None,
        }
    )


def _sequence_contrasts() -> tuple[HealthSequenceContrastV1, ...]:
    return (
        _sequence_contrast("self-nis-gate"),
        _sequence_contrast("combined-health-gate"),
    )


def _sequence_events() -> tuple[HealthSequenceEventV1, ...]:
    return (
        _sequence_event("self-nis-gate"),
        _sequence_event("combined-health-gate"),
    )


def _aggregates() -> tuple[HealthAggregateMetricV1, ...]:
    return (
        HealthAggregateMetricV1(
            schema="ffb.health-aggregate/v1",
            condition_id="condition-a",
            method="fixed-fusion",
            metric_name="matched-center-mse",
            window="event",
            unit="m^2",
            status="ok",
            estimate=0.2,
            interval_lower=0.1,
            interval_upper=0.3,
            sequence_count=200,
            bootstrap_replicates=2000,
            defined_bootstrap_replicates=2000,
        ),
        HealthAggregateMetricV1(
            schema="ffb.health-aggregate/v1",
            condition_id="condition-a",
            method="camera-only",
            metric_name="matched-center-mse",
            window="event",
            unit="m^2",
            status="ok",
            estimate=0.3,
            interval_lower=0.2,
            interval_upper=0.4,
            sequence_count=200,
            bootstrap_replicates=2000,
            defined_bootstrap_replicates=2000,
        ),
    )


def _evaluation_request(
    fit: LoadedHealthFitArtifact,
) -> HealthEvaluationArtifactWriteRequest:
    intent = _intent()
    main, edge = _profiles()
    return HealthEvaluationArtifactWriteRequest(
        intent=intent,
        main_profile=main,
        edge_profile=edge,
        fit_reference=build_health_fit_reference(fit),
        sequence_losses=_sequence_losses(),
        sequence_contrasts=_sequence_contrasts(),
        sequence_events=_sequence_events(),
        aggregates=_aggregates(),
        validation=_evaluation_validation(intent),
        run=_run(intent, artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT),
    )


def _copy_artifact(source: Path, parent: Path, name: str) -> Path:
    destination = parent / name
    shutil.copytree(source, destination)
    return destination


def _rebind_artifact(path: Path, *, artifact_contract: HealthArtifactContract) -> None:
    index_path = path / HEALTH_PAYLOAD_INDEX_FILE
    payload_index = HealthPayloadIndexV1.model_validate_json(index_path.read_bytes())
    entries = []
    for entry in payload_index.files:
        member_bytes = (path / entry.path).read_bytes()
        entries.append(
            entry.model_copy(
                update={
                    "byte_length": len(member_bytes),
                    "sha256": hashlib.sha256(member_bytes).hexdigest(),
                }
            )
        )
    payload_index = payload_index.model_copy(update={"files": tuple(entries)})
    index_bytes = canonical_json_bytes(payload_index)
    index_path.write_bytes(index_bytes)
    artifact_sha256 = compute_health_artifact_digest(
        index_bytes,
        artifact_contract=artifact_contract,
    )
    run = RunRecordV1Alpha1.model_validate_json((path / HEALTH_RUN_FILE).read_bytes())
    run = run.model_copy(update={"artifact_sha256": artifact_sha256})
    run_bytes = canonical_json_bytes(run)
    (path / HEALTH_RUN_FILE).write_bytes(run_bytes)
    success = SuccessMarkerV1Alpha1(
        schema="ffb.success/v1alpha1",
        artifact_sha256=artifact_sha256,
        run_sha256=compute_run_record_digest(run_bytes),
    )
    (path / HEALTH_SUCCESS_FILE).write_bytes(canonical_json_bytes(success))


@pytest.fixture(scope="module")
def written_fit(tmp_path_factory: pytest.TempPathFactory) -> LoadedHealthFitArtifact:
    return write_health_fit_artifact(
        _fit_request(),
        tmp_path_factory.mktemp("health-fit") / "artifact",
        git_metadata_dirs=(),
    )


@pytest.fixture(scope="module", autouse=True)
def isolate_filesystem_envelope_from_full_matrix() -> Iterator[None]:
    """Keep low-level filesystem tests compact; exact rows have a separate gate."""

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        health_artifact_module,
        "_require_exact_evaluation_rows",
        lambda **_: None,
    )
    yield
    patcher.undo()


@pytest.fixture(scope="module")
def written_evaluation(
    tmp_path_factory: pytest.TempPathFactory,
    written_fit: LoadedHealthFitArtifact,
) -> LoadedHealthEvaluationArtifact:
    return write_health_evaluation_artifact(
        _evaluation_request(written_fit),
        tmp_path_factory.mktemp("health-evaluation") / "artifact",
        fit_artifact=written_fit,
        git_metadata_dirs=(),
    )


def test_health_artifact_digest_uses_distinct_exact_domains() -> None:
    index_bytes = b"{}\n"
    fit_expected = hashlib.sha256(
        b"fusion-fault-bench/health-fit-artifact/v1\x00"
        + len(index_bytes).to_bytes(8, "big")
        + index_bytes
    ).hexdigest()
    evaluation_expected = hashlib.sha256(
        b"fusion-fault-bench/health-eval-artifact/v1\x00"
        + len(index_bytes).to_bytes(8, "big")
        + index_bytes
    ).hexdigest()

    assert (
        compute_health_artifact_digest(
            index_bytes,
            artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
        )
        == fit_expected
    )
    assert (
        compute_health_artifact_digest(
            index_bytes,
            artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
        )
        == evaluation_expected
    )
    assert fit_expected != evaluation_expected


def test_scientific_validator_rejects_incomplete_evaluation_matrix() -> None:
    intent = _intent()
    main, edge = _profiles()
    with pytest.raises(ArtifactValidationError, match="frozen matrix"):
        _EXACT_EVALUATION_ROWS_VALIDATOR(
            intent=intent,
            main_profile=main,
            edge_profile=edge,
            losses=_sequence_losses(),
            contrasts=_sequence_contrasts(),
            events=_sequence_events(),
            aggregates=_aggregates(),
        )


def _focused_exact_case(*, identity: bool) -> SimpleNamespace:
    fault = (
        HealthFaultSpec(
            family="identity",
            target="none",
            axis="none",
            unit="identity",
            value=0.0,
        )
        if identity
        else HealthFaultSpec(
            family="calibration-translation",
            target="camera",
            axis="x",
            unit="m",
            value=1.0,
        )
    )
    return SimpleNamespace(
        condition_id="focused-condition",
        population="main-test",
        fault=fault,
    )


def _focused_dropout_case(*, probability: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        condition_id="focused-condition",
        population="main-test",
        fault=HealthFaultSpec(
            family="dropout",
            target="camera",
            axis="availability",
            unit="probability",
            value=probability,
        ),
    )


def _focused_exact_losses() -> tuple[HealthSequenceLossV1, ...]:
    intent = _intent()
    eligible_by_window = {"score": 276, "event": 144, "recovery": 72}
    return tuple(
        HealthSequenceLossV1(
            schema="ffb.health-sequence-loss/v1",
            sequence_id="sequence:000",
            condition_id="focused-condition",
            method=method,
            window=window,
            loss_sum_m2=1.0,
            valid_object_frame_count=eligible,
            eligible_object_frame_count=eligible,
        )
        for method in intent.methods
        for window, eligible in eligible_by_window.items()
    )


def _focused_full_dropout_losses() -> tuple[HealthSequenceLossV1, ...]:
    expected_target_support = {"score": 132, "event": 0, "recovery": 72}
    records: list[HealthSequenceLossV1] = []
    for row in _focused_exact_losses():
        if row.method not in {"camera-only", "fixed-fusion"}:
            records.append(row)
            continue
        valid_count = expected_target_support[row.window]
        records.append(
            HealthSequenceLossV1.model_validate(
                row.model_dump(by_alias=True)
                | {
                    "valid_object_frame_count": valid_count,
                    "loss_sum_m2": 0.0 if valid_count == 0 else 1.0,
                }
            )
        )
    return tuple(records)


def _focused_exact_contrasts() -> tuple[HealthSequenceContrastV1, ...]:
    eligible_by_window = {"score": 276, "event": 144, "recovery": 72}
    policies = (
        "self-nis-gate",
        "cross-nis-gate",
        "direct-telemetry-gate",
        "combined-health-gate",
        "combined-health-gate-abstain",
    )
    return tuple(
        HealthSequenceContrastV1(
            schema="ffb.health-sequence-contrast/v1",
            sequence_id="sequence:000",
            condition_id="focused-condition",
            policy=policy,
            window=window,
            fixed_support_sha256="c" * 64,
            policy_support_sha256="c" * 64,
            fixed_policy_common_count=eligible,
            fixed_on_common_loss_sum_m2=1.0,
            policy_on_fixed_common_loss_sum_m2=1.0,
            target_drop_applicable=True,
            policy_target_drop_common_count=eligible,
            policy_on_target_common_loss_sum_m2=1.0,
            target_drop_on_common_loss_sum_m2=1.0,
            frame_oracle_applicable=True,
            policy_frame_oracle_common_count=eligible,
            policy_on_oracle_common_loss_sum_m2=1.0,
            frame_oracle_on_common_loss_sum_m2=1.0,
            frame_oracle_support_sha256="c" * 64,
        )
        for policy in policies
        for window, eligible in eligible_by_window.items()
    )


def _focused_full_dropout_contrasts() -> tuple[HealthSequenceContrastV1, ...]:
    eligible_by_window = {"score": 276, "event": 144, "recovery": 72}
    fixed_by_window = {"score": 132, "event": 0, "recovery": 72}
    policies = (
        "self-nis-gate",
        "cross-nis-gate",
        "direct-telemetry-gate",
        "combined-health-gate",
        "combined-health-gate-abstain",
    )
    records: list[HealthSequenceContrastV1] = []
    for policy in policies:
        for window, eligible in eligible_by_window.items():
            fixed_count = fixed_by_window[window]
            fixed_loss = 0.0 if fixed_count == 0 else 1.0
            policy_on_fixed = 1.0 if fixed_count == eligible else (0.5 if fixed_count else 0.0)
            fixed_digest = "c" * 64 if fixed_count == eligible else "f" * 64
            records.append(
                HealthSequenceContrastV1(
                    schema="ffb.health-sequence-contrast/v1",
                    sequence_id="sequence:000",
                    condition_id="focused-condition",
                    policy=policy,
                    window=window,
                    fixed_support_sha256=fixed_digest,
                    policy_support_sha256="c" * 64,
                    fixed_policy_common_count=fixed_count,
                    fixed_on_common_loss_sum_m2=fixed_loss,
                    policy_on_fixed_common_loss_sum_m2=policy_on_fixed,
                    target_drop_applicable=True,
                    policy_target_drop_common_count=eligible,
                    policy_on_target_common_loss_sum_m2=1.0,
                    target_drop_on_common_loss_sum_m2=1.0,
                    frame_oracle_applicable=True,
                    policy_frame_oracle_common_count=eligible,
                    policy_on_oracle_common_loss_sum_m2=1.0,
                    frame_oracle_on_common_loss_sum_m2=1.0,
                    frame_oracle_support_sha256="c" * 64,
                )
            )
    return tuple(records)


def _focused_event(*, identity: bool) -> HealthSequenceEventV1:
    payload = {
        "schema": "ffb.health-sequence-event/v1",
        "sequence_id": "sequence:000",
        "condition_id": "focused-condition",
        "policy": "combined-health-gate",
        "detected": not identity,
        "detection_latency_frames": None if identity else 0,
        "first_latch_label": None if identity else "camera-fault",
        "outcome": "missed" if identity else "correct",
        "correctly_attributed": not identity,
        "attribution_latency_frames": None if identity else 0,
        "realized_dropout": None,
        "first_missing_frame_minus_event_start": None,
        "detection_minus_first_missing_frames": None,
        "latch_episode_count": 0 if identity else 1,
        "false_alert_episode_count": 0,
        "early_clear": False,
        "final_active_state": "healthy" if identity else "camera-fault",
        "active_frame_count": 24,
        "active_healthy_frames": 24 if identity else 0,
        "active_camera_fault_frames": 0 if identity else 24,
        "active_lidar_fault_frames": 0,
        "active_ambiguous_frames": 0,
        "active_camera_action_frames": 0,
        "active_lidar_action_frames": 0 if identity else 24,
        "active_fixed_action_frames": 24 if identity else 0,
        "active_undefined_action_frames": 0,
        "recovery_eligible": not identity,
        "recovered": False,
        "recovery_latency_frames": None,
    }
    return HealthSequenceEventV1.model_validate(payload)


def _focused_dropout_event(*, realized: bool) -> HealthSequenceEventV1:
    if realized:
        updates: dict[str, object] = {
            "realized_dropout": True,
            "first_missing_frame_minus_event_start": 0,
            "detection_minus_first_missing_frames": 0,
        }
        base = _focused_event(identity=False)
    else:
        updates = {"realized_dropout": False}
        base = _focused_event(identity=True)
    return HealthSequenceEventV1.model_validate(base.model_dump(by_alias=True) | updates)


def _focused_exact_validation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    case: SimpleNamespace,
    losses: tuple[HealthSequenceLossV1, ...],
    contrasts: tuple[HealthSequenceContrastV1, ...],
    events: tuple[HealthSequenceEventV1, ...],
) -> None:
    cases = tuple(
        SimpleNamespace(
            condition_id=("focused-condition" if index == 0 else f"focused-condition-{index:02d}"),
            population=case.population,
            fault=case.fault,
        )
        for index in range(47)
    )

    def for_all_conditions(records: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            record.model_copy(update={"condition_id": matrix_case.condition_id})
            for matrix_case in cases
            for record in records
        )

    monkeypatch.setattr(
        "fusion_fault_bench.health_benchmark.expand_test_cases",
        lambda _intent: cases,
    )
    monkeypatch.setattr(
        health_artifact_module,
        "_expected_case_sequence_ids",
        lambda **_kwargs: ("sequence:000",),
    )
    _EXACT_EVALUATION_ROWS_VALIDATOR(
        intent=_intent(),
        main_profile=_profiles()[0],
        edge_profile=_profiles()[1],
        losses=for_all_conditions(losses),
        contrasts=for_all_conditions(contrasts),
        events=for_all_conditions(events),
        aggregates=(),
    )


def test_exact_validator_rejects_denominator_and_availability_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _focused_exact_case(identity=True)
    losses = _focused_exact_losses()
    wrong_denominator = (
        losses[0].model_copy(update={"eligible_object_frame_count": 275}),
        *losses[1:],
    )
    with pytest.raises(ArtifactValidationError, match="nonfrozen denominator"):
        _focused_exact_validation(
            monkeypatch,
            case=case,
            losses=wrong_denominator,
            contrasts=(),
            events=(),
        )

    wrong_support = (
        losses[0].model_copy(update={"valid_object_frame_count": 275}),
        *losses[1:],
    )
    with pytest.raises(ArtifactValidationError, match="must cover exact eligibility"):
        _focused_exact_validation(
            monkeypatch,
            case=case,
            losses=wrong_support,
            contrasts=(),
            events=(),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"fixed_on_common_loss_sum_m2": 2.0}, "common loss exceeds"),
        ({"policy_support_sha256": "d" * 64}, "commitments or losses"),
    ),
)
def test_exact_validator_rejects_contrast_sum_and_support_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    mutation: dict[str, object],
    message: str,
) -> None:
    contrasts = _focused_exact_contrasts()
    mutated = (
        HealthSequenceContrastV1.model_validate(contrasts[0].model_dump(by_alias=True) | mutation),
        *contrasts[1:],
    )
    with pytest.raises(ArtifactValidationError, match=message):
        _focused_exact_validation(
            monkeypatch,
            case=_focused_exact_case(identity=True),
            losses=_focused_exact_losses(),
            contrasts=mutated,
            events=(),
        )


def test_exact_validator_rejects_known_support_nesting_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    losses = _focused_exact_losses()
    policy_support_rebound = tuple(
        HealthSequenceLossV1.model_validate(
            row.model_dump(by_alias=True) | {"valid_object_frame_count": 143}
        )
        if row.method == "self-nis-gate" and row.window == "event"
        else row
        for row in losses
    )
    with pytest.raises(ArtifactValidationError, match="nonabstaining policy support"):
        _focused_exact_validation(
            monkeypatch,
            case=_focused_exact_case(identity=False),
            losses=policy_support_rebound,
            contrasts=(),
            events=(),
        )

    contrasts = _focused_exact_contrasts()
    mutations = (
        (
            {
                "fixed_support_sha256": "d" * 64,
                "fixed_policy_common_count": 143,
                "fixed_on_common_loss_sum_m2": 0.9,
                "policy_on_fixed_common_loss_sum_m2": 0.9,
            },
            "fixed-policy",
        ),
        (
            {
                "policy_target_drop_common_count": 143,
                "policy_on_target_common_loss_sum_m2": 0.9,
                "target_drop_on_common_loss_sum_m2": 0.9,
            },
            "policy-target-drop",
        ),
        (
            {
                "frame_oracle_support_sha256": "d" * 64,
                "policy_frame_oracle_common_count": 143,
                "policy_on_oracle_common_loss_sum_m2": 0.9,
                "frame_oracle_on_common_loss_sum_m2": 0.9,
            },
            "policy-frame-oracle",
        ),
    )
    for mutation, message in mutations:
        target_index = next(
            index
            for index, row in enumerate(contrasts)
            if row.policy == "self-nis-gate" and row.window == "event"
        )
        mutated_row = HealthSequenceContrastV1.model_validate(
            contrasts[target_index].model_dump(by_alias=True) | mutation
        )
        mutated = (
            *contrasts[:target_index],
            mutated_row,
            *contrasts[target_index + 1 :],
        )
        with pytest.raises(ArtifactValidationError, match=message):
            _focused_exact_validation(
                monkeypatch,
                case=_focused_exact_case(identity=False),
                losses=losses,
                contrasts=mutated,
                events=(),
            )


@pytest.mark.parametrize(
    ("digest_mutation", "message"),
    (
        (
            {
                "fixed_support_sha256": "c" * 64,
                "frame_oracle_support_sha256": "e" * 64,
            },
            "equal fixed-policy support",
        ),
        (
            {
                "fixed_support_sha256": "f" * 64,
                "frame_oracle_support_sha256": "c" * 64,
            },
            "equal policy-oracle support",
        ),
    ),
)
def test_exact_validator_rejects_equal_digest_with_unequal_full_support(
    monkeypatch: pytest.MonkeyPatch,
    digest_mutation: dict[str, object],
    message: str,
) -> None:
    losses = tuple(
        row.model_copy(
            update={
                "valid_object_frame_count": 200,
                "loss_sum_m2": 0.8,
            }
        )
        if row.method == "combined-health-gate-abstain" and row.window == "score"
        else row
        for row in _focused_full_dropout_losses()
    )
    contrasts = _focused_full_dropout_contrasts()
    target_index = next(
        index
        for index, row in enumerate(contrasts)
        if row.policy == "combined-health-gate-abstain" and row.window == "score"
    )
    target = contrasts[target_index]
    mutated_target = HealthSequenceContrastV1.model_validate(
        target.model_dump(by_alias=True)
        | {
            "policy_target_drop_common_count": 200,
            "policy_on_target_common_loss_sum_m2": 0.8,
            "target_drop_on_common_loss_sum_m2": 0.75,
            "policy_frame_oracle_common_count": 200,
            "policy_on_oracle_common_loss_sum_m2": 0.8,
            "frame_oracle_on_common_loss_sum_m2": 0.75,
        }
        | digest_mutation
    )
    mutated = (
        *contrasts[:target_index],
        mutated_target,
        *contrasts[target_index + 1 :],
    )
    with pytest.raises(ArtifactValidationError, match=message):
        _focused_exact_validation(
            monkeypatch,
            case=_focused_dropout_case(probability=0.5),
            losses=losses,
            contrasts=mutated,
            events=(),
        )


def test_exact_validator_enforces_full_dropout_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    losses = _focused_full_dropout_losses()
    fixed_event_index = next(
        index
        for index, row in enumerate(losses)
        if row.method == "fixed-fusion" and row.window == "event"
    )
    target_event_index = next(
        index
        for index, row in enumerate(losses)
        if row.method == "camera-only" and row.window == "event"
    )
    rebound_losses = tuple(
        HealthSequenceLossV1.model_validate(
            row.model_dump(by_alias=True) | {"valid_object_frame_count": 1, "loss_sum_m2": 1.0}
        )
        if index in {fixed_event_index, target_event_index}
        else row
        for index, row in enumerate(losses)
    )
    with pytest.raises(ArtifactValidationError, match="full-dropout fixed and target support"):
        _focused_exact_validation(
            monkeypatch,
            case=_focused_dropout_case(),
            losses=rebound_losses,
            contrasts=(),
            events=(),
        )

    baseline = _focused_dropout_event(realized=True)
    mutations = (
        (
            baseline.model_dump(by_alias=True)
            | {
                "realized_dropout": False,
                "first_missing_frame_minus_event_start": None,
                "detection_minus_first_missing_frames": None,
                "detected": False,
                "detection_latency_frames": None,
                "first_latch_label": None,
                "outcome": "missed",
                "correctly_attributed": False,
                "attribution_latency_frames": None,
                "latch_episode_count": 0,
            }
        ),
        (
            baseline.model_dump(by_alias=True)
            | {
                "first_missing_frame_minus_event_start": 1,
                "detection_minus_first_missing_frames": -1,
            }
        ),
        (
            baseline.model_dump(by_alias=True)
            | {
                "active_lidar_action_frames": 0,
                "active_fixed_action_frames": 24,
            }
        ),
    )
    for payload in mutations:
        mutated_event = HealthSequenceEventV1.model_validate(payload)
        with pytest.raises(ArtifactValidationError, match="full-dropout event"):
            _focused_exact_validation(
                monkeypatch,
                case=_focused_dropout_case(),
                losses=losses,
                contrasts=_focused_full_dropout_contrasts(),
                events=(mutated_event,),
            )


def test_exact_validator_treats_unrealized_dropout_as_fully_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _focused_dropout_event(realized=False)
    mutated = HealthSequenceEventV1.model_validate(
        baseline.model_dump(by_alias=True)
        | {
            "active_lidar_action_frames": 24,
            "active_fixed_action_frames": 0,
        }
    )
    with pytest.raises(ArtifactValidationError, match="fully-available actions"):
        _focused_exact_validation(
            monkeypatch,
            case=_focused_dropout_case(probability=0.5),
            losses=_focused_exact_losses(),
            contrasts=_focused_exact_contrasts(),
            events=(mutated,),
        )


def test_exact_validator_rejects_event_detection_and_action_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _focused_exact_case(identity=False)
    event = _focused_event(identity=False)
    suppressed_detection = HealthSequenceEventV1.model_validate(
        event.model_dump(by_alias=True)
        | {
            "detected": False,
            "detection_latency_frames": None,
            "first_latch_label": None,
            "outcome": "missed",
            "correctly_attributed": False,
            "attribution_latency_frames": None,
        }
    )
    with pytest.raises(ArtifactValidationError, match="latch episodes"):
        _focused_exact_validation(
            monkeypatch,
            case=case,
            losses=_focused_exact_losses(),
            contrasts=_focused_exact_contrasts(),
            events=(suppressed_detection,),
        )

    wrong_actions = HealthSequenceEventV1.model_validate(
        event.model_dump(by_alias=True)
        | {
            "active_lidar_action_frames": 0,
            "active_fixed_action_frames": 24,
        }
    )
    with pytest.raises(ArtifactValidationError, match="state table"):
        _focused_exact_validation(
            monkeypatch,
            case=case,
            losses=_focused_exact_losses(),
            contrasts=_focused_exact_contrasts(),
            events=(wrong_actions,),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        (
            {
                "recovered": True,
                "recovery_latency_frames": 12,
            },
            "recovery latency",
        ),
        ({"latch_episode_count": 13}, "latch episodes"),
        ({"early_clear": True}, "early-clear"),
    ),
)
def test_exact_validator_rejects_event_window_and_transition_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    message: str,
) -> None:
    event = _focused_event(identity=False)
    mutated = HealthSequenceEventV1.model_validate(event.model_dump(by_alias=True) | updates)
    with pytest.raises(ArtifactValidationError, match=message):
        _focused_exact_validation(
            monkeypatch,
            case=_focused_exact_case(identity=False),
            losses=_focused_exact_losses(),
            contrasts=_focused_exact_contrasts(),
            events=(mutated,),
        )


def test_exact_validator_requires_early_clear_when_active_state_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _focused_event(identity=False)
    mutated = HealthSequenceEventV1.model_validate(
        event.model_dump(by_alias=True)
        | {
            "final_active_state": "healthy",
            "active_healthy_frames": 24,
            "active_camera_fault_frames": 0,
            "active_lidar_action_frames": 0,
            "active_fixed_action_frames": 24,
            "recovery_eligible": False,
        }
    )
    with pytest.raises(ArtifactValidationError, match="early-clear"):
        _focused_exact_validation(
            monkeypatch,
            case=_focused_exact_case(identity=False),
            losses=_focused_exact_losses(),
            contrasts=_focused_exact_contrasts(),
            events=(mutated,),
        )


def test_exact_validator_caps_standard_score_false_alerts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = _focused_event(identity=True)
    mutated = HealthSequenceEventV1.model_validate(
        event.model_dump(by_alias=True) | {"false_alert_episode_count": 24}
    )
    with pytest.raises(ArtifactValidationError, match="false alerts"):
        _focused_exact_validation(
            monkeypatch,
            case=_focused_exact_case(identity=True),
            losses=_focused_exact_losses(),
            contrasts=_focused_exact_contrasts(),
            events=(mutated,),
        )


def test_fit_writer_rejects_arbitrary_validation_conjunction(tmp_path: Path) -> None:
    request = _fit_request()
    fabricated = HealthValidationV1(
        schema="ffb.health-validation/v1",
        intent_sha256=sha256_digest(request.intent),
        checks=(
            HealthValidationCheckV1(
                check_id="invented-pass",
                passed=True,
                observed=True,
                expected=True,
            ),
        ),
        all_checks_passed=True,
    )
    with pytest.raises(ArtifactValidationError, match="frozen conjunction"):
        write_health_fit_artifact(
            replace(request, validation=fabricated),
            tmp_path / "fit",
            git_metadata_dirs=(),
        )


def test_fit_build_load_commits_exact_members_and_channel_order(
    written_fit: LoadedHealthFitArtifact,
) -> None:
    fit = written_fit
    assert {member.name for member in fit.path.iterdir()} == set(HEALTH_FIT_ARTIFACT_PATHS)
    assert tuple(candidate.candidate_index for candidate in fit.candidates) == tuple(range(36))
    assert (
        next(
            entry.record_count
            for entry in fit.payload_index.files
            if entry.path == HEALTH_CANDIDATES_FILE
        )
        == 36
    )
    ecdf = json.loads((fit.path / HEALTH_ECDF_FILE).read_bytes())
    assert tuple(channel["channel"] for channel in ecdf["channels"]) == (HEALTH_ECDF_CHANNEL_ORDER)
    assert all(len(channel["values"]) == 9200 for channel in ecdf["channels"])
    assert fit.run.artifact_sha256 == fit.artifact_sha256
    assert fit.success.artifact_sha256 == fit.artifact_sha256
    assert fit.success.run_sha256 == fit.run_sha256
    reloaded = load_health_fit_artifact(fit.path)
    assert reloaded.artifact_sha256 == fit.artifact_sha256
    assert reloaded.run_sha256 == fit.run_sha256
    assert np.array_equal(
        reloaded.calibration.camera_self_mean,
        fit.calibration.camera_self_mean,
    )


def test_fit_timestamp_only_rerun_preserves_scientific_identity(tmp_path: Path) -> None:
    first_request = _fit_request()
    first = write_health_fit_artifact(
        first_request,
        tmp_path / "first",
        git_metadata_dirs=(),
    )
    later_run = _run(
        first_request.intent,
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
        started_at=first_request.run.started_at + timedelta(minutes=1),
    )
    second = write_health_fit_artifact(
        replace(first_request, run=later_run),
        tmp_path / "second",
        git_metadata_dirs=(),
    )

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.run_sha256 != second.run_sha256
    for entry in first.payload_index.files:
        assert (first.path / entry.path).read_bytes() == (second.path / entry.path).read_bytes()
    assert (first.path / HEALTH_PAYLOAD_INDEX_FILE).read_bytes() == (
        second.path / HEALTH_PAYLOAD_INDEX_FILE
    ).read_bytes()


def test_evaluation_build_load_binds_exact_fit_and_records(
    written_fit: LoadedHealthFitArtifact,
    written_evaluation: LoadedHealthEvaluationArtifact,
) -> None:
    evaluation = written_evaluation
    assert evaluation.sequence_rows_materialized
    assert {member.name for member in evaluation.path.iterdir()} == set(HEALTH_EVAL_ARTIFACT_PATHS)
    assert evaluation.fit_reference == build_health_fit_reference(written_fit)
    assert evaluation.fit_reference.fit_artifact_sha256 == written_fit.artifact_sha256
    assert evaluation.fit_reference.fit_run_sha256 == written_fit.run_sha256
    assert tuple(record.method for record in evaluation.sequence_losses) == (
        "camera-only",
        "fixed-fusion",
    )
    assert tuple(record.policy for record in evaluation.sequence_events) == (
        "combined-health-gate",
        "self-nis-gate",
    )
    assert tuple(record.policy for record in evaluation.sequence_contrasts) == (
        "combined-health-gate",
        "self-nis-gate",
    )
    assert (
        next(
            entry.record_count
            for entry in evaluation.payload_index.files
            if entry.path == HEALTH_SEQUENCE_CONTRASTS_FILE
        )
        == 2
    )
    with pytest.raises(ArtifactValidationError, match="frozen matrix"):
        load_health_evaluation_artifact(
            evaluation.path,
            fit_artifact=written_fit,
        )


def test_condition_stream_is_byte_identical_and_returns_metadata_light_handle(
    written_fit: LoadedHealthFitArtifact,
    written_evaluation: LoadedHealthEvaluationArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _evaluation_request(written_fit)
    case = SimpleNamespace(
        condition_id="condition-a",
        population="main-test",
        fault=HealthFaultSpec(
            family="identity",
            target="none",
            axis="none",
            unit="identity",
            value=0.0,
        ),
    )
    monkeypatch.setattr(
        "fusion_fault_bench.health_benchmark.expand_test_cases",
        lambda _intent: (case,),
    )
    monkeypatch.setattr(health_artifact_module, "_EXPECTED_EVALUATION_CONDITION_COUNT", 1)
    monkeypatch.setattr(health_artifact_module, "_MAX_EVALUATION_LOSS_RECORDS", 2)
    monkeypatch.setattr(health_artifact_module, "_MAX_EVALUATION_CONTRAST_RECORDS", 2)
    monkeypatch.setattr(health_artifact_module, "_MAX_EVALUATION_EVENT_RECORDS", 2)
    destination = tmp_path / "streamed"
    batch = SimpleNamespace(
        condition_id="condition-a",
        sequence_losses=tuple(
            sorted(request.sequence_losses, key=health_artifact_module._loss_key)
        ),
        sequence_contrasts=tuple(
            sorted(request.sequence_contrasts, key=health_artifact_module._contrast_key)
        ),
        sequence_events=tuple(
            sorted(request.sequence_events, key=health_artifact_module._event_key)
        ),
        aggregates=tuple(sorted(request.aggregates, key=health_artifact_module._aggregate_key)),
    )
    with HealthEvaluationArtifactTransaction(
        destination,
        fit_artifact=written_fit,
        git_metadata_dirs=(),
    ) as transaction:
        transaction.append_condition(batch)
        streamed = transaction.finalize(
            validation=request.validation,
            run=request.run,
        )

    for name in (
        HEALTH_SEQUENCE_LOSSES_FILE,
        HEALTH_SEQUENCE_CONTRASTS_FILE,
        "sequence-events.ndjson",
        "aggregate-metrics.ndjson",
    ):
        assert (streamed.path / name).read_bytes() == (written_evaluation.path / name).read_bytes()
    assert streamed.sequence_losses == ()
    assert streamed.sequence_contrasts == ()
    assert streamed.sequence_events == ()
    assert not streamed.sequence_rows_materialized
    assert streamed.aggregates == tuple(
        sorted(request.aggregates, key=health_artifact_module._aggregate_key)
    )


def test_condition_stream_reauthenticates_fit_before_publication(
    written_fit: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _evaluation_request(written_fit)
    case = SimpleNamespace(
        condition_id="condition-a",
        population="main-test",
        fault=HealthFaultSpec(
            family="identity",
            target="none",
            axis="none",
            unit="identity",
            value=0.0,
        ),
    )
    monkeypatch.setattr(
        "fusion_fault_bench.health_benchmark.expand_test_cases",
        lambda _intent: (case,),
    )
    monkeypatch.setattr(health_artifact_module, "_EXPECTED_EVALUATION_CONDITION_COUNT", 1)
    monkeypatch.setattr(health_artifact_module, "_MAX_EVALUATION_LOSS_RECORDS", 2)
    monkeypatch.setattr(health_artifact_module, "_MAX_EVALUATION_CONTRAST_RECORDS", 2)
    monkeypatch.setattr(health_artifact_module, "_MAX_EVALUATION_EVENT_RECORDS", 2)
    destination = tmp_path / "fit-mutation-evaluation"
    batch = SimpleNamespace(
        condition_id="condition-a",
        sequence_losses=tuple(
            sorted(request.sequence_losses, key=health_artifact_module._loss_key)
        ),
        sequence_contrasts=tuple(
            sorted(request.sequence_contrasts, key=health_artifact_module._contrast_key)
        ),
        sequence_events=tuple(
            sorted(request.sequence_events, key=health_artifact_module._event_key)
        ),
        aggregates=tuple(sorted(request.aggregates, key=health_artifact_module._aggregate_key)),
    )
    transaction = HealthEvaluationArtifactTransaction(
        destination,
        fit_artifact=written_fit,
        git_metadata_dirs=(),
    )
    transaction.append_condition(batch)
    moved_fit = tmp_path / "moved-fit"
    written_fit.path.rename(moved_fit)
    try:
        with pytest.raises(ArtifactValidationError, match="fit source changed"):
            transaction.finalize(
                validation=request.validation,
                run=request.run,
            )
    finally:
        moved_fit.rename(written_fit.path)
    transaction.abort()
    assert not destination.exists()
    assert not transaction.staging_path.exists()


def test_fit_loader_rejects_rebound_candidate_order_and_truncation(
    written_fit: LoadedHealthFitArtifact,
    tmp_path: Path,
) -> None:
    wrong_order = _copy_artifact(written_fit.path, tmp_path, "wrong-order")
    candidate_path = wrong_order / HEALTH_CANDIDATES_FILE
    lines = candidate_path.read_bytes().splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    candidate_path.write_bytes(b"".join(lines))
    _rebind_artifact(
        wrong_order,
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
    )
    with pytest.raises(ArtifactValidationError, match="canonical"):
        load_health_fit_artifact(wrong_order)

    truncated = _copy_artifact(written_fit.path, tmp_path, "truncated")
    truncated_path = truncated / HEALTH_CANDIDATES_FILE
    truncated_path.write_bytes(truncated_path.read_bytes()[:-1])
    _rebind_artifact(
        truncated,
        artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT,
    )
    with pytest.raises(ArtifactValidationError, match="terminal LF"):
        load_health_fit_artifact(truncated)


def test_fit_loader_rejects_rebound_ecdf_channel_order(
    written_fit: LoadedHealthFitArtifact,
    tmp_path: Path,
) -> None:
    copied = _copy_artifact(written_fit.path, tmp_path, "ecdf-order")
    ecdf_path = copied / HEALTH_ECDF_FILE
    value = cast(dict[str, Any], json.loads(ecdf_path.read_bytes()))
    channels = cast(list[dict[str, Any]], value["channels"])
    channels[0], channels[1] = channels[1], channels[0]
    ecdf_path.write_bytes(canonical_json_bytes(value))
    _rebind_artifact(copied, artifact_contract=HEALTH_FIT_ARTIFACT_CONTRACT)

    with pytest.raises(ArtifactValidationError, match="fixed schema"):
        load_health_fit_artifact(copied)


def test_evaluation_loader_rejects_rebound_order_and_fit_tamper(
    written_fit: LoadedHealthFitArtifact,
    written_evaluation: LoadedHealthEvaluationArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health_artifact_module, "_MAX_EVALUATION_LOSS_RECORDS", 2)
    monkeypatch.setattr(health_artifact_module, "_MAX_EVALUATION_CONTRAST_RECORDS", 2)
    monkeypatch.setattr(health_artifact_module, "_MAX_EVALUATION_EVENT_RECORDS", 2)
    wrong_order = _copy_artifact(written_evaluation.path, tmp_path, "eval-order")
    loss_path = wrong_order / HEALTH_SEQUENCE_LOSSES_FILE
    lines = loss_path.read_bytes().splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    loss_path.write_bytes(b"".join(lines))
    _rebind_artifact(
        wrong_order,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
    )
    with pytest.raises(ArtifactValidationError, match="canonical"):
        load_health_evaluation_artifact(wrong_order, fit_artifact=written_fit)

    wrong_contrast_order = _copy_artifact(
        written_evaluation.path,
        tmp_path,
        "eval-contrast-order",
    )
    contrast_path = wrong_contrast_order / HEALTH_SEQUENCE_CONTRASTS_FILE
    contrast_lines = contrast_path.read_bytes().splitlines(keepends=True)
    contrast_lines[0], contrast_lines[1] = contrast_lines[1], contrast_lines[0]
    contrast_path.write_bytes(b"".join(contrast_lines))
    _rebind_artifact(
        wrong_contrast_order,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
    )
    with pytest.raises(ArtifactValidationError, match="canonical"):
        load_health_evaluation_artifact(
            wrong_contrast_order,
            fit_artifact=written_fit,
        )

    wrong_fit = _copy_artifact(written_evaluation.path, tmp_path, "wrong-fit")
    reference_path = wrong_fit / HEALTH_FIT_REFERENCE_FILE
    reference = json.loads(reference_path.read_bytes())
    reference["fit_artifact_sha256"] = "f" * 64
    reference_path.write_bytes(canonical_json_bytes(reference))
    _rebind_artifact(
        wrong_fit,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
    )
    with pytest.raises(ArtifactValidationError, match="fit reference"):
        load_health_evaluation_artifact(wrong_fit, fit_artifact=written_fit)


def test_writer_never_overwrites_existing_or_dangling_destination(
    written_fit: LoadedHealthFitArtifact,
    tmp_path: Path,
) -> None:
    request = _fit_request()
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_health_fit_artifact(request, existing, git_metadata_dirs=())
    assert sentinel.read_text(encoding="utf-8") == "keep"

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        write_health_fit_artifact(request, dangling, git_metadata_dirs=())
    assert dangling.is_symlink()

    evaluation_destination = tmp_path / "existing-evaluation"
    evaluation_destination.mkdir()
    evaluation_sentinel = evaluation_destination / "keep"
    evaluation_sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_health_evaluation_artifact(
            _evaluation_request(written_fit),
            evaluation_destination,
            fit_artifact=written_fit,
            git_metadata_dirs=(),
        )
    assert evaluation_sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("member_kind", ["symlink", "hardlink", "fifo"])
def test_loader_rejects_unsafe_member_types(
    written_fit: LoadedHealthFitArtifact,
    tmp_path: Path,
    member_kind: str,
) -> None:
    copied = _copy_artifact(written_fit.path, tmp_path, member_kind)
    run_path = copied / HEALTH_RUN_FILE
    run_bytes = run_path.read_bytes()
    run_path.unlink()
    if member_kind == "symlink":
        run_path.symlink_to(written_fit.path / HEALTH_RUN_FILE)
    elif member_kind == "hardlink":
        backing = tmp_path / "hardlink-backing"
        backing.write_bytes(run_bytes)
        os.link(backing, run_path)
    else:
        os.mkfifo(run_path)

    expected = (
        "symlink"
        if member_kind == "symlink"
        else ("hard-linked" if member_kind == "hardlink" else "regular files")
    )
    with pytest.raises(ArtifactValidationError, match=expected):
        load_health_fit_artifact(copied)


def test_loader_binds_reads_to_scanned_descriptors(
    written_fit: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = _copy_artifact(written_fit.path, tmp_path, "inode-swap")
    intent_bytes = (copied / "intent.json").read_bytes()
    replacement = tmp_path / "replacement-intent.json"
    original_read = vars(health_artifact_module)["_read_member"]
    swapped = False

    def swap_before_read(
        root: Path,
        name: str,
        *,
        expected_stat: os.stat_result,
        byte_cap: int,
    ) -> bytes:
        nonlocal swapped
        if name == "intent.json" and not swapped:
            replacement.write_bytes(intent_bytes)
            replacement.replace(root / name)
            swapped = True
        return original_read(
            root,
            name,
            expected_stat=expected_stat,
            byte_cap=byte_cap,
        )

    monkeypatch.setattr(health_artifact_module, "_read_member", swap_before_read)
    with pytest.raises(ArtifactValidationError, match="changed during validation"):
        load_health_fit_artifact(copied)


def test_evaluation_rejects_a_different_fit_handle(
    written_fit: LoadedHealthFitArtifact,
    written_evaluation: LoadedHealthEvaluationArtifact,
) -> None:
    fabricated = replace(written_fit, artifact_sha256="e" * 64)
    with pytest.raises(ArtifactValidationError, match="not authentic"):
        load_health_evaluation_artifact(
            written_evaluation.path,
            fit_artifact=fabricated,
        )
