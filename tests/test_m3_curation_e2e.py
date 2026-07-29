from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import fusion_fault_bench.procedural_release as release_module
from fusion_fault_bench.artifacts import (
    canonical_json_bytes,
    compute_run_record_digest,
    derive_run_id,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import SuccessMarkerV1Alpha1
from fusion_fault_bench.contracts.bundle_v1alpha1 import expected_conditions
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    ProceduralSource,
)
from fusion_fault_bench.contracts.matrix_v1 import (
    LoadedExperimentMatrix,
    load_experiment_matrix,
)
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    PROCEDURAL_AGGREGATE_METRICS_FILE,
    PROCEDURAL_ARTIFACT_CONTRACT,
    PROCEDURAL_CROSSOVERS_FILE,
    PROCEDURAL_INDEXED_PAYLOAD_PATHS,
    PROCEDURAL_MANIFEST_FILE,
    PROCEDURAL_PAYLOAD_INDEX_FILE,
    PROCEDURAL_PROFILE_FILE,
    PROCEDURAL_RUN_FILE,
    PROCEDURAL_SEQUENCE_METRICS_FILE,
    PROCEDURAL_SUCCESS_FILE,
    PROCEDURAL_VALIDATION_FILE,
    ProceduralPayloadFileEntryV1Alpha2,
    ProceduralPayloadIndexV1Alpha2,
)
from fusion_fault_bench.contracts.procedural_validation_v1 import ProceduralValidationV1
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    RunRecordV1Alpha1,
    SeverityCoordinate,
)
from fusion_fault_bench.procedural_artifacts import (
    LoadedProceduralArtifact,
    canonical_procedural_ndjson_bytes,
    compute_procedural_artifact_digest,
)
from fusion_fault_bench.procedural_release import (
    RepeatRunResources,
    build_m3_matrix_validation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = Path("examples/matrices/m3-procedural-v1.json")
PRIMARY_OUTPUT = "reports/generated/m3-e2e-primary"
REPEAT_OUTPUT = "reports/generated/m3-e2e-repeat"
PUBLIC_CI_RUN_ID = 123_456

sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))
curation = importlib.import_module("tools.m3_curation")
release_tool = importlib.import_module("tools.m3_release")
curation_fixtures = importlib.import_module("test_m3_curation")
release_fixtures = importlib.import_module("test_procedural_release")

M3CurationError = vars(curation)["M3CurationError"]
ProceduralReleaseDriverError = vars(release_tool)["ProceduralReleaseDriverError"]
_AGGREGATE_ROWS = vars(curation_fixtures)["_aggregate_rows"]
_CROSSOVER_ROWS = vars(curation_fixtures)["_crossover_rows"]
_FAKE_ARTIFACT_SET = vars(release_fixtures)["_fake_artifact_set"]
_READ_EVIDENCE_MEMBER = vars(release_tool)["_read_evidence_member"]
_READ_REGULAR = vars(curation)["_read_regular"]
_RELEASE_ALLOWLIST = vars(curation)["_release_allowlist"]
_SCAN_RELEASE = vars(curation)["_scan_release"]


@dataclass(frozen=True, slots=True)
class SyntheticRelease:
    matrix: LoadedExperimentMatrix
    first: tuple[LoadedProceduralArtifact, ...]
    second: tuple[LoadedProceduralArtifact, ...]
    matrix_validation: Any
    repeat_verification: Any
    public_ci_attestation_bytes: bytes
    results_review_attestation_bytes: bytes
    results_review_report_bytes: bytes
    identity_bytes: bytes
    release_root: Path


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _attestation_bytes(
    *,
    scientific_source_revision: str,
    artifact_set_sha256: str,
) -> tuple[bytes, bytes, bytes]:
    results_review_report = (
        f"# M3 adversarial results review — {curation.RELEASE_ID}\n"
        "\n"
        "Verdict: **PASS**\n"
        "\n"
        f"Artifact set: `{artifact_set_sha256}`\n"
    ).encode()
    public_ci = {
        "schema": curation.PUBLIC_CI_ATTESTATION_SCHEMA,
        "provider": "github-actions",
        "repository": curation.PUBLIC_CI_REPOSITORY,
        "workflow": "ci",
        "workflow_path": ".github/workflows/ci.yml",
        "run_id": PUBLIC_CI_RUN_ID,
        "url": f"{curation.PUBLIC_CI_URL_PREFIX}{PUBLIC_CI_RUN_ID}",
        "source_revision": scientific_source_revision,
        "conclusion": "success",
        "smoke_matrix_sha256": curation.M3_CI_SMOKE_MATRIX_SHA256,
        "release_evidence": False,
        "verification_scope": curation.PUBLIC_CI_VERIFICATION_SCOPE,
    }
    results_review = {
        "schema": curation.RESULTS_REVIEW_ATTESTATION_SCHEMA,
        "release_id": curation.RELEASE_ID,
        "status": "pass",
        "scope": list(curation.RESULTS_REVIEW_SCOPE),
        "reviewed_artifact_set_sha256": artifact_set_sha256,
        "reviewer": "independent-adversarial-agent",
        "reference": curation.RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH.as_posix(),
        "reference_sha256": _sha256(results_review_report),
        "reference_byte_length": len(results_review_report),
        "verification_scope": curation.RESULTS_REVIEW_VERIFICATION_SCOPE,
    }
    return (
        canonical_json_bytes(public_ci),
        canonical_json_bytes(results_review),
        results_review_report,
    )


def _synthetic_validation(
    manifest: Any,
    profile: Any,
    run: RunRecordV1Alpha1,
) -> ProceduralValidationV1:
    source = cast(ProceduralSource, manifest.source)
    conditions = expected_conditions(manifest)
    eligible_per_sequence = profile.source.frame_count * profile.source.object_count
    total_eligible = source.sequence_count * eligible_per_sequence
    availability = isinstance(manifest, AvailabilityControlManifest)
    common_mode = isinstance(manifest, CommonModeControlManifest)
    sequence_pairs = (
        len(manifest.methods) * len(manifest.evaluation.metrics)
        if availability
        else len(manifest.methods)
    )
    expected_loss_pairs = (
        ()
        if availability
        else (
            (
                ("camera-only", "matched-center-mse"),
                ("lidar-only", "matched-center-mse"),
                ("fixed-fusion", "matched-center-mse"),
            )
            if common_mode
            else (
                ("camera-only", "matched-center-mse"),
                ("lidar-only", "matched-center-mse"),
                ("fixed-fusion", "matched-center-mse"),
                ("fixed-fusion", "fused-minus-healthy"),
                ("fault-target-drop-policy", "matched-center-mse"),
            )
        )
    )
    expected_losses = tuple(
        {
            "check_id": (
                f"synthetic-{condition.severity_index}-{condition.direction}-{method}-{metric}"
            ),
            "fault_family": condition.fault_family,
            "fault_axis": condition.fault_axis,
            "severity": SeverityCoordinate(
                index=condition.severity_index,
                magnitude=condition.magnitude,
                direction=cast(Any, condition.direction),
                unit=cast(Any, condition.unit),
            ),
            "method_id": method,
            "metric_name": metric,
            "expected_value_m2": (
                condition.magnitude - 0.2
                if metric == "fused-minus-healthy"
                else 0.2 + condition.magnitude
            ),
            "empirical_value_m2": (
                condition.magnitude - 0.2
                if metric == "fused-minus-healthy"
                else 0.2 + condition.magnitude
            ),
            "analytic_standard_error_m2": 1.0,
            "absolute_standardized_error": 0.0,
            "standard_error_multiplier": 6.0,
            "passed": True,
        }
        for condition in conditions
        for method, metric in expected_loss_pairs
    )
    moment_keys = (
        ("camera-x-mean", "mean", "camera", "x", None, None),
        ("camera-y-mean", "mean", "camera", "y", None, None),
        ("lidar-x-mean", "mean", "lidar", "x", None, None),
        ("lidar-y-mean", "mean", "lidar", "y", None, None),
        ("camera-x-variance", "variance", "camera", "x", None, None),
        ("camera-y-variance", "variance", "camera", "y", None, None),
        ("lidar-x-variance", "variance", "lidar", "x", None, None),
        ("lidar-y-variance", "variance", "lidar", "y", None, None),
        (
            "camera-xy-covariance",
            "within-sensor-covariance",
            "camera",
            "x",
            "camera",
            "y",
        ),
        (
            "lidar-xy-covariance",
            "within-sensor-covariance",
            "lidar",
            "x",
            "lidar",
            "y",
        ),
        (
            "camera-x-lidar-x-covariance",
            "camera-lidar-cross-covariance",
            "camera",
            "x",
            "lidar",
            "x",
        ),
        (
            "camera-x-lidar-y-covariance",
            "camera-lidar-cross-covariance",
            "camera",
            "x",
            "lidar",
            "y",
        ),
        (
            "camera-y-lidar-x-covariance",
            "camera-lidar-cross-covariance",
            "camera",
            "y",
            "lidar",
            "x",
        ),
        (
            "camera-y-lidar-y-covariance",
            "camera-lidar-cross-covariance",
            "camera",
            "y",
            "lidar",
            "y",
        ),
    )
    moments = tuple(
        {
            "check_id": check_id,
            "statistic": statistic,
            "sensor_a": sensor_a,
            "coordinate_a": coordinate_a,
            "sensor_b": sensor_b,
            "coordinate_b": coordinate_b,
            "sample_count": total_eligible,
            "ddof": 1,
            "expectation": 0.0,
            "observed_value": 0.0,
            "six_standard_error_bound": 1.0,
            "absolute_discrepancy": 0.0,
            "unit": "m" if statistic == "mean" else "m^2",
            "passed": True,
        }
        for check_id, statistic, sensor_a, coordinate_a, sensor_b, coordinate_b in moment_keys
    )

    def oracle(check_id: str, unit: str = "m") -> dict[str, object]:
        return {
            "check_id": check_id,
            "unit": unit,
            "maximum_absolute_discrepancy": 0.0,
            "tolerance": 1e-12,
            "passed": True,
        }

    return ProceduralValidationV1.model_validate(
        {
            "schema": "ffb.procedural-validation/v1",
            "run_id": run.run_id,
            "manifest_sha256": run.manifest_sha256,
            "profile_id": profile.profile_id,
            "profile_sha256": sha256_digest(profile),
            "split": source.split,
            "sequence_count": source.sequence_count,
            "frame_count": profile.source.frame_count,
            "object_count": profile.source.object_count,
            "total_eligible_object_frame_count": total_eligible,
            "profile_checks": {
                "schema_valid": True,
                "profile_id_valid": True,
                "profile_digest_valid": True,
                "split_count_valid": True,
                "roi_valid": True,
                "isotropic_yaw_compatible": True,
                "split_family_support_valid": True,
                "canonical_ordering_valid": True,
                "all_checks_passed": True,
            },
            "eligibility": {
                "ordered_sequence_commitments_sha256": "c" * 64,
                "minimum_eligible_object_frame_count": eligible_per_sequence,
                "maximum_eligible_object_frame_count": eligible_per_sequence,
                "total_eligible_object_frame_count": total_eligible,
                "eligibility_invariant": True,
            },
            "oracle_checks": {
                "identity_center": oracle("identity-center"),
                "calibration_translation_center": oracle("calibration-translation-center"),
                "translation_bias_equivalence_center": oracle(
                    "translation-bias-equivalence-center"
                ),
                "translation_bias_equivalence_sequence_loss": oracle(
                    "translation-bias-equivalence-sequence-loss",
                    "m^2",
                ),
                "calibration_yaw_center": oracle("calibration-yaw-center"),
                "timestamp_alignment_center": oracle("timestamp-alignment-center"),
                "static_timestamp_center": oracle("static-timestamp-center"),
                "fault_cancellation_mutation_rejected": True,
                "all_checks_passed": True,
            },
            "moment_checks": moments,
            "expected_loss_checks": expected_losses,
            "dropout_validation": (
                {
                    "status": "applicable",
                    "uniform_vectors_sha256": "d" * 64,
                    "exact_mask_comparison_count": (source.sequence_count * len(conditions)),
                    "frame_sharing_passed": True,
                    "nesting_passed": True,
                    "endpoint_behavior_passed": True,
                    "maximum_mask_discrepancy": 0,
                    "all_checks_passed": True,
                }
                if availability
                else {"status": "not-applicable"}
            ),
            "identity_comparison": (
                {
                    "status": "not-applicable",
                    "reason": "edge-common-mode-has-no-comparable-profile-peer",
                }
                if common_mode
                else {
                    "status": "deferred-to-matrix",
                    "reason": "cross-manifest-identity-requires-complete-matrix",
                }
            ),
            "common_mode_validation": (
                {
                    "status": "applicable",
                    "maximum_disagreement_discrepancy_m": 0.0,
                    "tolerance_m": 1e-12,
                    "passed": True,
                }
                if common_mode
                else {"status": "not-applicable"}
            ),
            "deterministic_model_checks": {
                "reported_covariance_behavior": "passed",
                "expected_curve_response": ("not-applicable" if availability else "passed"),
                "complete_sequence_performance_oracle": (
                    "passed" if not availability and not common_mode else "not-applicable"
                ),
                "identity_row_reconstruction": ("not-applicable" if common_mode else "passed"),
                "all_checks_passed": True,
            },
            "resources": {
                "implied_sequence_row_count": (
                    source.sequence_count * len(conditions) * sequence_pairs
                ),
                "sequence_row_cap": 2_000_000,
                "implied_bootstrap_cell_count": (
                    source.sequence_count * manifest.evaluation.bootstrap.replicates
                ),
                "bootstrap_cell_cap": 20_000_000,
                "sequence_count": source.sequence_count,
                "sequence_count_cap": 10_000,
                "bootstrap_replicates": manifest.evaluation.bootstrap.replicates,
                "bootstrap_replicate_cap": 20_000,
                "sequence_rows_within_cap": True,
                "bootstrap_cells_within_cap": True,
                "sequence_count_within_cap": True,
                "bootstrap_replicates_within_cap": True,
                "all_checks_passed": True,
            },
            "all_checks_passed": True,
        }
    )


def _write_synthetic_artifact(
    *,
    template: LoadedProceduralArtifact,
    root: Path,
    logical_output: str,
    aggregate_offset: float = 0.0,
    run_time_offset_seconds: int = 0,
) -> LoadedProceduralArtifact:
    manifest = template.manifest
    profile = template.profile
    manifest_sha256 = sha256_digest(manifest)
    profile_sha256 = sha256_digest(profile)
    run_id = derive_run_id(
        manifest_sha256=manifest_sha256,
        git_revision=template.run.git_revision,
        lockfile_sha256=template.run.lockfile_sha256,
        package_version=template.run.package_version,
        artifact_contract=PROCEDURAL_ARTIFACT_CONTRACT,
    )
    provisional_run = RunRecordV1Alpha1.model_validate(
        {
            **template.run.model_dump(mode="python", by_alias=True),
            "run_id": run_id,
            "command": (
                "ffb",
                "procedural",
                "matrix",
                "run",
                MATRIX_PATH.as_posix(),
                "--output-dir",
                logical_output,
            ),
            "artifact_sha256": "0" * 64,
            "started_at": (template.run.started_at + timedelta(seconds=run_time_offset_seconds)),
            "ended_at": (
                template.run.ended_at + timedelta(seconds=run_time_offset_seconds)
                if template.run.ended_at is not None
                else None
            ),
        }
    )
    aggregate_rows = list(
        record.model_copy(update={"run_id": run_id})
        for record in cast(tuple[Any, ...], _AGGREGATE_ROWS(manifest))
    )
    if aggregate_offset:
        original = aggregate_rows[0]
        assert original.estimate is not None
        assert original.interval_lower is not None
        assert original.interval_upper is not None
        aggregate_rows[0] = AggregateMetricRecordV1Alpha1.model_validate(
            {
                **original.model_dump(mode="python", by_alias=True),
                "estimate": original.estimate + aggregate_offset,
                "interval_lower": original.interval_lower + aggregate_offset,
                "interval_upper": original.interval_upper + aggregate_offset,
            }
        )
    aggregates = tuple(aggregate_rows)
    crossovers = tuple(
        record.model_copy(update={"run_id": run_id})
        for record in cast(tuple[Any, ...], _CROSSOVER_ROWS(manifest))
    )
    validation = _synthetic_validation(manifest, profile, provisional_run)
    scientific_bytes = {
        PROCEDURAL_MANIFEST_FILE: canonical_json_bytes(manifest),
        PROCEDURAL_PROFILE_FILE: canonical_json_bytes(profile),
        PROCEDURAL_SEQUENCE_METRICS_FILE: (
            canonical_json_bytes(
                {
                    "synthetic_omitted_sequence_rows": manifest.experiment,
                }
            )
        ),
        PROCEDURAL_AGGREGATE_METRICS_FILE: canonical_procedural_ndjson_bytes(aggregates),
        PROCEDURAL_CROSSOVERS_FILE: canonical_procedural_ndjson_bytes(
            crossovers,
            allow_empty=True,
        ),
        PROCEDURAL_VALIDATION_FILE: canonical_json_bytes(validation),
    }
    payload_index = ProceduralPayloadIndexV1Alpha2(
        schema="ffb.payload-index/v1alpha2",
        artifact_contract=PROCEDURAL_ARTIFACT_CONTRACT,
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        profile_sha256=profile_sha256,
        files=tuple(
            ProceduralPayloadFileEntryV1Alpha2(
                path=cast(Any, name),
                byte_length=len(scientific_bytes[name]),
                sha256=_sha256(scientific_bytes[name]),
            )
            for name in PROCEDURAL_INDEXED_PAYLOAD_PATHS
        ),
    )
    payload_index_bytes = canonical_json_bytes(payload_index)
    artifact_sha256 = compute_procedural_artifact_digest(payload_index_bytes)
    run = RunRecordV1Alpha1.model_validate(
        {
            **provisional_run.model_dump(mode="python", by_alias=True),
            "artifact_sha256": artifact_sha256,
        }
    )
    run_bytes = canonical_json_bytes(run)
    run_sha256 = compute_run_record_digest(run_bytes)
    success = SuccessMarkerV1Alpha1(
        schema="ffb.success/v1alpha1",
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )
    artifact_root = root / manifest.experiment
    artifact_root.mkdir(parents=True)
    for name, value in {
        **scientific_bytes,
        PROCEDURAL_PAYLOAD_INDEX_FILE: payload_index_bytes,
        PROCEDURAL_RUN_FILE: run_bytes,
        PROCEDURAL_SUCCESS_FILE: canonical_json_bytes(success),
    }.items():
        (artifact_root / name).write_bytes(value)

    metrics = tuple(record.model_copy(update={"run_id": run_id}) for record in template.metrics)
    source = cast(ProceduralSource, manifest.source)
    assert validation.resources.implied_sequence_row_count > source.sequence_count
    return replace(
        template,
        path=artifact_root,
        metrics=metrics,
        aggregates=aggregates,
        crossovers=crossovers,
        validation=validation,
        payload_index=payload_index,
        run=run,
        artifact_sha256=artifact_sha256,
        run_sha256=run_sha256,
    )


@pytest.fixture(scope="module")
def synthetic_release(
    tmp_path_factory: pytest.TempPathFactory,
) -> SyntheticRelease:
    root = tmp_path_factory.mktemp("m3-curation-e2e")
    matrix = load_experiment_matrix(MATRIX_PATH, source_root=REPOSITORY_ROOT)
    templates = _FAKE_ARTIFACT_SET(matrix)
    first_root = root / "first"
    second_root = root / "second"
    first = tuple(
        _write_synthetic_artifact(
            template=template,
            root=first_root,
            logical_output=PRIMARY_OUTPUT,
        )
        for template in templates
    )
    second = tuple(
        _write_synthetic_artifact(
            template=template,
            root=second_root,
            logical_output=REPEAT_OUTPUT,
            run_time_offset_seconds=60,
        )
        for template in templates
    )
    matrix_validation = build_m3_matrix_validation(matrix, first)
    repeat_builder = vars(release_module)["_repeat_verification_from_artifacts"]
    repeat_verification = repeat_builder(
        matrix,
        first,
        second,
        first_resources=RepeatRunResources(
            wall_time_seconds=10.0,
            peak_memory_bytes=1_000,
        ),
        second_resources=RepeatRunResources(
            wall_time_seconds=11.0,
            peak_memory_bytes=1_100,
        ),
    )
    public_ci_bytes, results_review_bytes, results_review_report_bytes = _attestation_bytes(
        scientific_source_revision=first[0].run.git_revision,
        artifact_set_sha256=matrix_validation.artifact_set_sha256,
    )
    identity = curation.derive_official_identity(
        matrix,
        first,
        second,
        matrix_validation=matrix_validation,
        repeat_verification=repeat_verification,
        public_ci_attestation_bytes=public_ci_bytes,
        results_review_attestation_bytes=results_review_bytes,
        results_review_report_bytes=results_review_report_bytes,
        expected_first_output=PRIMARY_OUTPUT,
        expected_second_output=REPEAT_OUTPUT,
    )
    identity_bytes = canonical_json_bytes(identity)
    release_root = root / "release"
    curation.build_curated_release(
        matrix,
        first,
        second,
        matrix_validation=matrix_validation,
        repeat_verification=repeat_verification,
        official_identity_bytes=identity_bytes,
        results_review_report_bytes=results_review_report_bytes,
        output_dir=release_root,
        expected_first_output=PRIMARY_OUTPUT,
        expected_second_output=REPEAT_OUTPUT,
    )
    return SyntheticRelease(
        matrix=matrix,
        first=first,
        second=second,
        matrix_validation=matrix_validation,
        repeat_verification=repeat_verification,
        public_ci_attestation_bytes=public_ci_bytes,
        results_review_attestation_bytes=results_review_bytes,
        results_review_report_bytes=results_review_report_bytes,
        identity_bytes=identity_bytes,
        release_root=release_root,
    )


def _release_copy(
    synthetic_release: SyntheticRelease,
    tmp_path: Path,
) -> Path:
    destination = tmp_path / "release"
    shutil.copytree(synthetic_release.release_root, destination)
    return destination


def test_synthetic_release_builds_validates_and_omits_raw_sequence_rows(
    synthetic_release: SyntheticRelease,
) -> None:
    index = curation.validate_curated_release(
        synthetic_release.release_root,
        official_identity_bytes=synthetic_release.identity_bytes,
        results_review_report_bytes=synthetic_release.results_review_report_bytes,
    )

    assert index["selection_policy"]["omitted_member"] == (PROCEDURAL_SEQUENCE_METRICS_FILE)
    assert not any(
        path.name == PROCEDURAL_SEQUENCE_METRICS_FILE
        for path in synthetic_release.release_root.rglob("*")
    )
    assert _SCAN_RELEASE(synthetic_release.release_root) == _RELEASE_ALLOWLIST()
    assert all(
        primary.run_sha256 != repeated.run_sha256
        for primary, repeated in zip(
            synthetic_release.first,
            synthetic_release.second,
            strict=True,
        )
    )
    assert tuple(
        (row.method_id, row.metric_name) for row in synthetic_release.first[0].aggregates[:6]
    ) == (
        ("camera-only", "matched-center-mse"),
        ("lidar-only", "matched-center-mse"),
        ("fixed-fusion", "matched-center-mse"),
        ("fixed-fusion", "fused-minus-healthy"),
        ("fault-target-drop-policy", "matched-center-mse"),
        ("performance-oracle", "matched-center-mse"),
    )


def test_validator_rejects_tampered_aggregate(
    synthetic_release: SyntheticRelease,
    tmp_path: Path,
) -> None:
    release_root = _release_copy(synthetic_release, tmp_path)
    aggregate_path = (
        release_root / "records" / curation.EXPECTED_EXPERIMENTS[0] / "aggregate-metrics.ndjson"
    )
    lines = aggregate_path.read_bytes().splitlines(keepends=True)
    first = cast(dict[str, Any], json.loads(lines[0]))
    first["estimate"] = cast(float, first["estimate"]) + 0.01
    first["interval_lower"] = cast(float, first["interval_lower"]) + 0.01
    first["interval_upper"] = cast(float, first["interval_upper"]) + 0.01
    lines[0] = canonical_json_bytes(first)
    aggregate_path.write_bytes(b"".join(lines))

    with pytest.raises(M3CurationError):
        curation.validate_curated_release(
            release_root,
            official_identity_bytes=synthetic_release.identity_bytes,
            results_review_report_bytes=(synthetic_release.results_review_report_bytes),
        )


def test_public_validator_rejects_equal_primary_repeat_run_digests(
    synthetic_release: SyntheticRelease,
    tmp_path: Path,
) -> None:
    release_root = _release_copy(synthetic_release, tmp_path)
    evidence_path = release_root / curation.REPEAT_VERIFICATION_PATH
    evidence = cast(dict[str, Any], json.loads(evidence_path.read_bytes()))
    first_run = cast(dict[str, Any], evidence["first_run"])
    second_run = cast(dict[str, Any], evidence["second_run"])
    second_run["run_record_sha256s"] = list(cast(list[str], first_run["run_record_sha256s"]))
    evidence_path.write_bytes(canonical_json_bytes(evidence))

    with pytest.raises(M3CurationError):
        curation.validate_curated_release(
            release_root,
            official_identity_bytes=synthetic_release.identity_bytes,
            results_review_report_bytes=(synthetic_release.results_review_report_bytes),
        )


def test_coherent_rehash_requires_its_new_external_identity(
    synthetic_release: SyntheticRelease,
    tmp_path: Path,
) -> None:
    matrix = synthetic_release.matrix
    templates = _FAKE_ARTIFACT_SET(matrix)
    first_output = "reports/generated/m3-e2e-rehashed-primary"
    second_output = "reports/generated/m3-e2e-rehashed-repeat"
    first = tuple(
        _write_synthetic_artifact(
            template=template,
            root=tmp_path / "rehashed-first",
            logical_output=first_output,
            aggregate_offset=0.01 if index == 0 else 0.0,
        )
        for index, template in enumerate(templates)
    )
    second = tuple(
        _write_synthetic_artifact(
            template=template,
            root=tmp_path / "rehashed-second",
            logical_output=second_output,
            aggregate_offset=0.01 if index == 0 else 0.0,
            run_time_offset_seconds=60,
        )
        for index, template in enumerate(templates)
    )
    matrix_validation = build_m3_matrix_validation(matrix, first)
    repeat_builder = vars(release_module)["_repeat_verification_from_artifacts"]
    repeat_verification = repeat_builder(
        matrix,
        first,
        second,
        first_resources=RepeatRunResources(
            wall_time_seconds=12.0,
            peak_memory_bytes=1_200,
        ),
        second_resources=RepeatRunResources(
            wall_time_seconds=13.0,
            peak_memory_bytes=1_300,
        ),
    )
    public_ci_bytes, results_review_bytes, results_review_report_bytes = _attestation_bytes(
        scientific_source_revision=first[0].run.git_revision,
        artifact_set_sha256=matrix_validation.artifact_set_sha256,
    )
    identity_bytes = canonical_json_bytes(
        curation.derive_official_identity(
            matrix,
            first,
            second,
            matrix_validation=matrix_validation,
            repeat_verification=repeat_verification,
            public_ci_attestation_bytes=public_ci_bytes,
            results_review_attestation_bytes=results_review_bytes,
            results_review_report_bytes=results_review_report_bytes,
            expected_first_output=first_output,
            expected_second_output=second_output,
        )
    )
    release_root = tmp_path / "coherently-rehashed-release"
    curation.build_curated_release(
        matrix,
        first,
        second,
        matrix_validation=matrix_validation,
        repeat_verification=repeat_verification,
        official_identity_bytes=identity_bytes,
        results_review_report_bytes=results_review_report_bytes,
        output_dir=release_root,
        expected_first_output=first_output,
        expected_second_output=second_output,
    )

    assert (
        matrix_validation.artifact_set_sha256
        != synthetic_release.matrix_validation.artifact_set_sha256
    )
    assert identity_bytes != synthetic_release.identity_bytes
    curation.validate_curated_release(
        release_root,
        official_identity_bytes=identity_bytes,
        results_review_report_bytes=results_review_report_bytes,
    )
    with pytest.raises(M3CurationError):
        curation.validate_curated_release(
            release_root,
            official_identity_bytes=synthetic_release.identity_bytes,
            results_review_report_bytes=(synthetic_release.results_review_report_bytes),
        )


@pytest.mark.parametrize(
    "relative_path",
    (
        Path("README.md"),
        Path("figures/fusion-delta-curves.svg"),
        Path("evidence/results-review.md"),
    ),
)
def test_validator_rejects_tampered_derived_presentation(
    synthetic_release: SyntheticRelease,
    tmp_path: Path,
    relative_path: Path,
) -> None:
    release_root = _release_copy(synthetic_release, tmp_path)
    target = release_root / relative_path
    target.write_bytes(target.read_bytes() + b"tampered\n")

    with pytest.raises(M3CurationError):
        curation.validate_curated_release(
            release_root,
            official_identity_bytes=synthetic_release.identity_bytes,
            results_review_report_bytes=(synthetic_release.results_review_report_bytes),
        )


def test_bool_cannot_alias_integer_in_official_identity(
    synthetic_release: SyntheticRelease,
    tmp_path: Path,
) -> None:
    identity = cast(dict[str, Any], json.loads(synthetic_release.identity_bytes))
    ordered = cast(list[dict[str, Any]], identity["ordered_artifacts"])
    assert ordered[0]["execution_index"] == 0
    ordered[0]["execution_index"] = False
    mutated_bytes = canonical_json_bytes(identity)
    destination = tmp_path / "must-not-exist"

    with pytest.raises(M3CurationError, match="official identity disagrees"):
        curation.build_curated_release(
            synthetic_release.matrix,
            synthetic_release.first,
            synthetic_release.second,
            matrix_validation=synthetic_release.matrix_validation,
            repeat_verification=synthetic_release.repeat_verification,
            official_identity_bytes=mutated_bytes,
            results_review_report_bytes=(synthetic_release.results_review_report_bytes),
            output_dir=destination,
            expected_first_output=PRIMARY_OUTPUT,
            expected_second_output=REPEAT_OUTPUT,
        )
    assert not destination.exists()


def test_identity_candidate_rejects_output_nested_in_a_run_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "reports" / "generated" / "first"
    second_root = tmp_path / "reports" / "generated" / "second"
    evidence_root = tmp_path / "reports" / "generated" / "evidence"
    for root in (first_root, second_root, evidence_root):
        root.mkdir(parents=True)
    strict_inputs = SimpleNamespace(
        snapshot=SimpleNamespace(source_root=tmp_path),
        first_root=first_root,
        second_root=second_root,
        evidence_root=evidence_root,
    )

    def fake_strict_inputs(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return strict_inputs

    monkeypatch.setattr(
        release_tool,
        "_strict_release_inputs",
        fake_strict_inputs,
    )
    candidate = Path("reports/generated/first/identity.json")

    with pytest.raises(ProceduralReleaseDriverError, match="must be disjoint"):
        release_tool.derive_identity_candidate(
            MATRIX_PATH,
            first_output_dir=Path("reports/generated/first"),
            second_output_dir=Path("reports/generated/second"),
            evidence_dir=Path("reports/generated/evidence"),
            output_path=candidate,
            public_ci_attestation_path=curation.PUBLIC_CI_ATTESTATION_RELATIVE_PATH,
            results_review_attestation_path=(curation.RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH),
            results_review_report_path=(curation.RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH),
        )
    assert not (tmp_path / candidate).exists()


def test_curation_reader_rejects_lstat_to_open_symlink_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    victim = tmp_path / "victim"
    original = tmp_path / "original"
    secret = tmp_path / "secret"
    victim.write_bytes(b"public")
    secret.write_bytes(b"private")
    real_open = os.open

    def swap_then_open(path: Any, flags: int) -> int:
        if Path(path) == victim and victim.is_file():
            victim.rename(original)
            victim.symlink_to(secret)
        return real_open(path, flags)

    monkeypatch.setattr(curation.os, "open", swap_then_open)
    with pytest.raises(M3CurationError, match="release member"):
        _READ_REGULAR(victim)


def test_repeat_evidence_reader_rejects_lstat_to_open_symlink_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    victim = tmp_path / "evidence.json"
    original = tmp_path / "original.json"
    secret = tmp_path / "secret.json"
    victim.write_bytes(b'{"public":true}\n')
    secret.write_bytes(b'{"private":true}\n')
    real_open = os.open

    def swap_then_open(path: Any, flags: int) -> int:
        if Path(path) == victim and victim.is_file():
            victim.rename(original)
            victim.symlink_to(secret)
        return real_open(path, flags)

    monkeypatch.setattr(release_tool.os, "open", swap_then_open)
    with pytest.raises(ProceduralReleaseDriverError, match="evidence member"):
        _READ_EVIDENCE_MEMBER(tmp_path, victim.name)


def test_release_scan_rejects_hardlinked_allowlisted_member(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "release"
    for relative in _RELEASE_ALLOWLIST():
        path = release_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    os.link(release_root / curation.README_PATH, tmp_path / "outside-hardlink")

    with pytest.raises(M3CurationError, match="hard-linked"):
        _SCAN_RELEASE(release_root)
