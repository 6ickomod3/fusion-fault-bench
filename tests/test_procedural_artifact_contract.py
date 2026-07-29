from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

import fusion_fault_bench.procedural_artifacts as procedural_artifact_module
from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    compute_run_record_digest,
    derive_run_id,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import SuccessMarkerV1Alpha1
from fusion_fault_bench.contracts.bundle_v1alpha1 import (
    ConditionKey,
    expected_conditions,
    expected_sequence_ids,
)
from fusion_fault_bench.contracts.io import load_manifest, validate_manifest_mapping
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    FaultAxis,
    FaultFamily,
    GeometryCrossoverManifest,
    ProceduralSource,
    SeverityDirection,
    SeverityUnit,
)
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    PROCEDURAL_ARTIFACT_CONTRACT,
    PROCEDURAL_ARTIFACT_PATHS,
    PROCEDURAL_CROSSOVERS_FILE,
    PROCEDURAL_INDEXED_PAYLOAD_PATHS,
    PROCEDURAL_MANIFEST_FILE,
    PROCEDURAL_MAX_ARTIFACT_BYTES,
    PROCEDURAL_MAX_BOOTSTRAP_CELLS,
    PROCEDURAL_MAX_BOOTSTRAP_REPLICATES,
    PROCEDURAL_MAX_MEMBER_BYTES,
    PROCEDURAL_MAX_RECORD_BYTES,
    PROCEDURAL_MAX_SEQUENCE_COUNT,
    PROCEDURAL_MAX_SEQUENCE_ROWS,
    PROCEDURAL_PAYLOAD_INDEX_FILE,
    PROCEDURAL_RUN_FILE,
    PROCEDURAL_SEQUENCE_METRICS_FILE,
    PROCEDURAL_SUCCESS_FILE,
    ProceduralPayloadFileEntryV1Alpha2,
    ProceduralPayloadIndexV1Alpha2,
    procedural_payload_index_json_schema,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    ProceduralProfileV1,
    load_procedural_profile,
)
from fusion_fault_bench.contracts.procedural_validation_v1 import (
    ProceduralValidationV1,
    procedural_validation_json_schema,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    LocalizationMetricRecord,
    MetricRecordV1Alpha1,
    RateMetricRecord,
    RunRecordV1Alpha1,
    RuntimeEnvironment,
    SeverityCoordinate,
)
from fusion_fault_bench.experiments.procedural import (
    generate_procedural_sequence_metrics,
)
from fusion_fault_bench.procedural_artifacts import (
    LoadedProceduralArtifact,
    ProceduralArtifactWriteRequest,
    canonical_procedural_ndjson_bytes,
    compute_procedural_artifact_digest,
    load_procedural_artifact,
    validate_procedural_bundle,
    write_procedural_artifact,
)
from fusion_fault_bench.procedural_evaluation import evaluate_procedural_records
from fusion_fault_bench.procedural_validation import build_procedural_validation
from fusion_fault_bench.scenarios.procedural import generate_procedural_sequences

REPOSITORY_ROOT = Path(__file__).parents[1]
SMOKE_MANIFEST_PATH = REPOSITORY_ROOT / "examples/manifests/procedural-ci-smoke-v1alpha1.json"
SMOKE_PROFILE_PATH = REPOSITORY_ROOT / "examples/profiles/constant-velocity-ci-smoke-v1.json"
COMMON_MODE_MANIFEST_PATH = (
    REPOSITORY_ROOT / "examples/manifests/procedural-common-mode-x-edge-v1alpha1.json"
)
EDGE_PROFILE_PATH = REPOSITORY_ROOT / "examples/profiles/constant-velocity-fov-edge-v1.json"
PLACEHOLDER_DIGEST = "0" * 64
GIT_REVISION = "a" * 40
LOCKFILE_SHA256 = "b" * 64


type TestProceduralManifest = (
    GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest
)


def _smoke_manifest() -> GeometryCrossoverManifest:
    manifest = load_manifest(SMOKE_MANIFEST_PATH)
    assert isinstance(manifest, GeometryCrossoverManifest)
    return manifest


def _common_mode_manifest() -> CommonModeControlManifest:
    manifest = load_manifest(COMMON_MODE_MANIFEST_PATH)
    assert isinstance(manifest, CommonModeControlManifest)
    return manifest


def _availability_manifest() -> AvailabilityControlManifest:
    data = _smoke_manifest().model_dump(mode="json", by_alias=True)
    data["kind"] = "availability-control"
    data["experiment"] = "procedural-artifact-availability-test"
    data["fault_sweep"] = {
        "kind": "dropout",
        "target": "camera",
        "axis": "availability",
        "unit": "probability",
        "injection_site": "availability",
        "process": "shared-target-modality-frame-bernoulli",
        "probability_values": [0.0, 0.5, 1.0],
    }
    data["methods"] = [
        "camera-only",
        "lidar-only",
        "fixed-fusion",
        "fault-target-drop-policy",
    ]
    data["evaluation"] = {
        "mode": "availability-control",
        "metrics": [
            "coverage",
            "conditional-matched-center-mse",
            "undefined-output-rate",
        ],
        "missing_output_policy": "undefined-no-localization-penalty",
        "rate_aggregation": "count-ratio-with-sequence-bootstrap",
        "conditional_loss_aggregation": ("valid-object-frame-ratio-with-sequence-bootstrap"),
        "undefined_bootstrap_replicate_action": ("exclude-and-require-two-sided-support"),
        "unimodal_missing_input_action": "undefined",
        "fixed_fusion_missing_input_action": "undefined",
        "target_drop_identity_action": "fixed-fusion",
        "target_drop_nonidentity_action": "use-nontarget-modality",
        "bootstrap": {
            "method": "percentile",
            "unit": "sequence",
            "resampling": "paired-indices-across-severities-and-methods",
            "interval_scope": "pointwise",
            "replicates": 200,
            "confidence_level": 0.95,
        },
        "crossover": "not-applicable",
    }
    manifest = validate_manifest_mapping(data)
    assert isinstance(manifest, AvailabilityControlManifest)
    return manifest


def _profile_for(manifest: TestProceduralManifest) -> ProceduralProfileV1:
    path = (
        EDGE_PROFILE_PATH if isinstance(manifest, CommonModeControlManifest) else SMOKE_PROFILE_PATH
    )
    return load_procedural_profile(path)


def _run(
    manifest: TestProceduralManifest,
    *,
    started_at: datetime | None = None,
) -> RunRecordV1Alpha1:
    manifest_sha256 = sha256_digest(manifest)
    run_id = derive_run_id(
        manifest_sha256=manifest_sha256,
        git_revision=GIT_REVISION,
        lockfile_sha256=LOCKFILE_SHA256,
        package_version="0.1.0",
        artifact_contract=PROCEDURAL_ARTIFACT_CONTRACT,
    )
    started = datetime(2026, 7, 29, 12, 0, tzinfo=UTC) if started_at is None else started_at
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        package_version="0.1.0",
        git_revision=GIT_REVISION,
        source_dirty=False,
        lockfile_sha256=LOCKFILE_SHA256,
        command=(
            "ffb",
            "run",
            "examples/manifests/procedural-test-v1alpha1.json",
        ),
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


def _severity(condition: ConditionKey) -> SeverityCoordinate:
    return SeverityCoordinate(
        index=condition.severity_index,
        magnitude=condition.magnitude,
        direction=cast(SeverityDirection, condition.direction),
        unit=cast(SeverityUnit, condition.unit),
    )


def _localization_metrics(
    manifest: GeometryCrossoverManifest | CommonModeControlManifest,
    run: RunRecordV1Alpha1,
    profile: ProceduralProfileV1,
) -> tuple[MetricRecordV1Alpha1, ...]:
    eligible = profile.source.frame_count * profile.source.object_count
    records: list[MetricRecordV1Alpha1] = []
    for sequence_id in expected_sequence_ids(manifest):
        for condition in expected_conditions(manifest):
            for method in manifest.methods:
                if isinstance(manifest, CommonModeControlManifest):
                    value = 0.2 + condition.magnitude**2
                elif condition.severity_index == 0:
                    value = {
                        "camera-only": 2.0,
                        "lidar-only": 0.18,
                        "fixed-fusion": 0.165,
                        "fault-target-drop-policy": 0.165,
                        "performance-oracle": 0.165,
                    }[method]
                else:
                    value = {
                        "camera-only": 2.25,
                        "lidar-only": 0.18,
                        "fixed-fusion": 0.3,
                        "fault-target-drop-policy": 0.18,
                        "performance-oracle": 0.18,
                    }[method]
                records.append(
                    LocalizationMetricRecord(
                        schema="ffb.sequence-metric/v1alpha1",
                        record_level="sequence",
                        run_id=run.run_id,
                        manifest_sha256=run.manifest_sha256,
                        sequence_id=sequence_id,
                        fault_family=cast(FaultFamily, condition.fault_family),
                        fault_axis=cast(FaultAxis, condition.fault_axis),
                        severity=_severity(condition),
                        method_id=method,
                        eligible_object_frame_count=eligible,
                        valid_object_frame_count=eligible,
                        metric_name="matched-center-mse",
                        status="ok",
                        value=value,
                        unit="m^2",
                    )
                )
    return tuple(records)


def _availability_metrics(
    manifest: AvailabilityControlManifest,
    run: RunRecordV1Alpha1,
    profile: ProceduralProfileV1,
) -> tuple[MetricRecordV1Alpha1, ...]:
    eligible = profile.source.frame_count * profile.source.object_count
    records: list[MetricRecordV1Alpha1] = []
    for sequence_id in expected_sequence_ids(manifest):
        for condition in expected_conditions(manifest):
            for method in manifest.methods:
                target_or_fused = method in {"camera-only", "fixed-fusion"}
                valid = (
                    eligible
                    if condition.severity_index == 0 or not target_or_fused
                    else eligible // 2
                    if condition.magnitude == 0.5
                    else 0
                )
                loss = (
                    0.5
                    if target_or_fused
                    or (method == "fault-target-drop-policy" and condition.severity_index == 0)
                    else 0.2
                )
                common = {
                    "schema": "ffb.sequence-metric/v1alpha1",
                    "record_level": "sequence",
                    "run_id": run.run_id,
                    "manifest_sha256": run.manifest_sha256,
                    "sequence_id": sequence_id,
                    "fault_family": condition.fault_family,
                    "fault_axis": condition.fault_axis,
                    "severity": _severity(condition),
                    "method_id": method,
                    "eligible_object_frame_count": eligible,
                    "valid_object_frame_count": valid,
                }
                records.extend(
                    (
                        RateMetricRecord.model_validate(
                            {
                                **common,
                                "metric_name": "coverage",
                                "status": "ok",
                                "value": valid / eligible,
                                "unit": "fraction",
                            }
                        ),
                        LocalizationMetricRecord.model_validate(
                            {
                                **common,
                                "metric_name": "conditional-matched-center-mse",
                                "status": "ok" if valid else "undefined",
                                "value": loss if valid else None,
                                "unit": "m^2",
                            }
                        ),
                        RateMetricRecord.model_validate(
                            {
                                **common,
                                "metric_name": "undefined-output-rate",
                                "status": "ok",
                                "value": (eligible - valid) / eligible,
                                "unit": "fraction",
                            }
                        ),
                    )
                )
    return tuple(records)


def _oracle(check_id: str, *, unit: str = "m") -> dict[str, object]:
    return {
        "check_id": check_id,
        "unit": unit,
        "maximum_absolute_discrepancy": 0.0,
        "tolerance": 1e-12,
        "passed": True,
    }


def _validation(
    manifest: TestProceduralManifest,
    profile: ProceduralProfileV1,
    run: RunRecordV1Alpha1,
) -> ProceduralValidationV1:
    source = manifest.source
    assert isinstance(source, ProceduralSource)
    sequence_count = source.sequence_count
    eligible_per_sequence = profile.source.frame_count * profile.source.object_count
    total_eligible = sequence_count * eligible_per_sequence
    condition_count = len(expected_conditions(manifest))
    metric_pair_count = (
        len(manifest.methods) * len(manifest.evaluation.metrics)
        if isinstance(manifest, AvailabilityControlManifest)
        else len(manifest.methods)
    )
    implied_rows = sequence_count * condition_count * metric_pair_count
    common_mode = isinstance(manifest, CommonModeControlManifest)
    availability = isinstance(manifest, AvailabilityControlManifest)
    identity = expected_conditions(manifest)[0]
    value: dict[str, object] = {
        "schema": "ffb.procedural-validation/v1",
        "run_id": run.run_id,
        "manifest_sha256": run.manifest_sha256,
        "profile_id": profile.profile_id,
        "profile_sha256": sha256_digest(profile),
        "split": source.split,
        "sequence_count": sequence_count,
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
            "identity_center": _oracle("identity-center"),
            "calibration_translation_center": _oracle("translation-center"),
            "translation_bias_equivalence_center": _oracle("translation-bias-center"),
            "translation_bias_equivalence_sequence_loss": _oracle(
                "translation-bias-loss",
                unit="m^2",
            ),
            "calibration_yaw_center": _oracle("yaw-center"),
            "timestamp_alignment_center": _oracle("timestamp-center"),
            "static_timestamp_center": _oracle("static-timestamp-center"),
            "fault_cancellation_mutation_rejected": True,
            "all_checks_passed": True,
        },
        "moment_checks": (
            {
                "check_id": "camera-x-mean",
                "statistic": "mean",
                "sensor_a": "camera",
                "coordinate_a": "x",
                "sensor_b": None,
                "coordinate_b": None,
                "sample_count": total_eligible,
                "ddof": 1,
                "expectation": 0.0,
                "observed_value": 0.0,
                "six_standard_error_bound": 1.0,
                "absolute_discrepancy": 0.0,
                "unit": "m",
                "passed": True,
            },
        ),
        "expected_loss_checks": (
            ()
            if availability
            else (
                {
                    "check_id": "identity-reference-loss",
                    "fault_family": identity.fault_family,
                    "fault_axis": identity.fault_axis,
                    "severity": _severity(identity),
                    "method_id": manifest.methods[0],
                    "metric_name": "matched-center-mse",
                    "expected_value_m2": 0.2,
                    "empirical_value_m2": 0.2,
                    "analytic_standard_error_m2": 0.1,
                    "absolute_standardized_error": 0.0,
                    "standard_error_multiplier": 6.0,
                    "passed": True,
                },
            )
        ),
        "dropout_validation": (
            {
                "status": "applicable",
                "uniform_vectors_sha256": "d" * 64,
                "exact_mask_comparison_count": sequence_count * condition_count,
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
                "status": "applicable",
                "scope": "same-profile-split-observations-seeds-and-comparable-methods",
                "comparison_count": sequence_count,
                "maximum_absolute_value_discrepancy_m2": 0.0,
                "tolerance_m2": 1e-12,
                "passed": True,
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
        "resources": {
            "implied_sequence_row_count": implied_rows,
            "sequence_row_cap": PROCEDURAL_MAX_SEQUENCE_ROWS,
            "implied_bootstrap_cell_count": (
                sequence_count * manifest.evaluation.bootstrap.replicates
            ),
            "bootstrap_cell_cap": PROCEDURAL_MAX_BOOTSTRAP_CELLS,
            "sequence_count": sequence_count,
            "sequence_count_cap": PROCEDURAL_MAX_SEQUENCE_COUNT,
            "bootstrap_replicates": manifest.evaluation.bootstrap.replicates,
            "bootstrap_replicate_cap": PROCEDURAL_MAX_BOOTSTRAP_REPLICATES,
            "sequence_rows_within_cap": True,
            "bootstrap_cells_within_cap": True,
            "sequence_count_within_cap": True,
            "bootstrap_replicates_within_cap": True,
            "all_checks_passed": True,
        },
        "all_checks_passed": True,
    }
    return ProceduralValidationV1.model_validate(value)


def _request(
    manifest: TestProceduralManifest | None = None,
) -> ProceduralArtifactWriteRequest:
    selected_manifest = _smoke_manifest() if manifest is None else manifest
    profile = _profile_for(selected_manifest)
    run = _run(selected_manifest)
    source = selected_manifest.source
    assert isinstance(source, ProceduralSource)
    sequences = generate_procedural_sequences(
        profile,
        split=source.split,
        sequence_count=source.sequence_count,
        data_master_seed=selected_manifest.rng.data_master_seed,
    )
    metrics = generate_procedural_sequence_metrics(
        selected_manifest,
        profile=profile,
        run_id=run.run_id,
    )
    evaluated = evaluate_procedural_records(
        selected_manifest,
        run_id=run.run_id,
        metrics=metrics,
    )
    validation = build_procedural_validation(
        selected_manifest,
        profile=profile,
        run_id=run.run_id,
        sequences=sequences,
        metrics=evaluated.metrics,
    )
    return ProceduralArtifactWriteRequest(
        manifest=selected_manifest,
        profile=profile,
        metrics=evaluated.metrics,
        aggregates=evaluated.aggregates,
        crossovers=evaluated.crossovers,
        validation=validation,
        run=run,
    )


def _copy_artifact(source: Path, parent: Path, name: str) -> Path:
    destination = parent / name
    shutil.copytree(source, destination)
    return destination


@pytest.fixture
def written_procedural_artifact(tmp_path: Path) -> LoadedProceduralArtifact:
    return write_procedural_artifact(
        _request(),
        tmp_path / "artifact",
        git_metadata_dirs=(),
    )


def _entries() -> tuple[ProceduralPayloadFileEntryV1Alpha2, ...]:
    return tuple(
        ProceduralPayloadFileEntryV1Alpha2(
            path=path,
            byte_length=0 if path == PROCEDURAL_CROSSOVERS_FILE else 1,
            sha256="a" * 64,
        )
        for path in PROCEDURAL_INDEXED_PAYLOAD_PATHS
    )


def test_payload_index_requires_exact_six_member_order() -> None:
    value = {
        "schema": "ffb.payload-index/v1alpha2",
        "artifact_contract": "ffb.procedural-payload/v1",
        "run_id": "run:test",
        "manifest_sha256": "b" * 64,
        "profile_sha256": "c" * 64,
        "files": tuple(reversed(_entries())),
    }

    with pytest.raises(ValidationError, match="fixed six-member order"):
        ProceduralPayloadIndexV1Alpha2.model_validate(value)


def test_only_crossover_entry_may_be_zero_bytes() -> None:
    value = {
        "path": "manifest.json",
        "byte_length": 0,
        "sha256": "a" * 64,
    }
    with pytest.raises(ValidationError, match="only crossovers"):
        ProceduralPayloadFileEntryV1Alpha2.model_validate(value)

    crossover = ProceduralPayloadFileEntryV1Alpha2(
        path="crossovers.ndjson",
        byte_length=0,
        sha256=hashlib.sha256(b"").hexdigest(),
    )
    assert crossover.byte_length == 0


def test_procedural_artifact_digest_uses_exact_domain_and_length_framing() -> None:
    index = ProceduralPayloadIndexV1Alpha2(
        schema="ffb.payload-index/v1alpha2",
        artifact_contract="ffb.procedural-payload/v1",
        run_id="run:test",
        manifest_sha256="b" * 64,
        profile_sha256="c" * 64,
        files=_entries(),
    )
    index_bytes = canonical_json_bytes(index)
    expected = hashlib.sha256(
        b"fusion-fault-bench/procedural-artifact/v1\x00"
        + len(index_bytes).to_bytes(8, "big")
        + index_bytes
    ).hexdigest()

    assert compute_procedural_artifact_digest(index_bytes) == expected


def test_procedural_ndjson_has_explicit_empty_member_semantics() -> None:
    assert canonical_procedural_ndjson_bytes((), allow_empty=True) == b""
    with pytest.raises(ArtifactValidationError, match="at least one"):
        canonical_procedural_ndjson_bytes(())


def test_public_schemas_and_caps_are_exactly_frozen() -> None:
    index_schema = procedural_payload_index_json_schema()
    validation_schema = procedural_validation_json_schema()
    index_properties = cast(dict[str, Any], index_schema["properties"])
    validation_properties = cast(dict[str, Any], validation_schema["properties"])

    assert index_properties["schema"]["const"] == "ffb.payload-index/v1alpha2"
    assert index_properties["artifact_contract"]["const"] == "ffb.procedural-payload/v1"
    assert index_properties["files"]["minItems"] == 6
    assert index_properties["files"]["maxItems"] == 6
    assert validation_properties["schema"]["const"] == "ffb.procedural-validation/v1"
    assert (
        PROCEDURAL_MAX_SEQUENCE_COUNT,
        PROCEDURAL_MAX_BOOTSTRAP_REPLICATES,
        PROCEDURAL_MAX_BOOTSTRAP_CELLS,
        PROCEDURAL_MAX_SEQUENCE_ROWS,
        PROCEDURAL_MAX_MEMBER_BYTES,
        PROCEDURAL_MAX_ARTIFACT_BYTES,
        PROCEDURAL_MAX_RECORD_BYTES,
    ) == (
        10_000,
        20_000,
        20_000_000,
        2_000_000,
        512 * 1024 * 1024,
        1024 * 1024 * 1024,
        1024 * 1024,
    )


def test_writer_round_trips_exact_nine_file_identity_graph(
    written_procedural_artifact: LoadedProceduralArtifact,
) -> None:
    loaded = written_procedural_artifact
    assert {path.name for path in loaded.path.iterdir()} == set(PROCEDURAL_ARTIFACT_PATHS)
    assert all(path.is_file() and not path.is_symlink() for path in loaded.path.iterdir())
    assert tuple(entry.path for entry in loaded.payload_index.files) == (
        PROCEDURAL_INDEXED_PAYLOAD_PATHS
    )

    for entry in loaded.payload_index.files:
        value = (loaded.path / entry.path).read_bytes()
        assert entry.byte_length == len(value)
        assert entry.sha256 == hashlib.sha256(value).hexdigest()

    index_bytes = (loaded.path / PROCEDURAL_PAYLOAD_INDEX_FILE).read_bytes()
    run_bytes = (loaded.path / PROCEDURAL_RUN_FILE).read_bytes()
    marker = SuccessMarkerV1Alpha1.model_validate_json(
        (loaded.path / PROCEDURAL_SUCCESS_FILE).read_bytes()
    )
    assert loaded.artifact_sha256 == compute_procedural_artifact_digest(index_bytes)
    assert loaded.run_sha256 == compute_run_record_digest(run_bytes)
    assert loaded.run.artifact_sha256 == loaded.artifact_sha256
    assert marker.artifact_sha256 == loaded.artifact_sha256
    assert marker.run_sha256 == loaded.run_sha256
    assert load_procedural_artifact(loaded.path) == loaded


def test_timestamp_only_rerun_preserves_indexed_payload_identity(
    tmp_path: Path,
) -> None:
    first_request = _request()
    first = write_procedural_artifact(
        first_request,
        tmp_path / "first",
        git_metadata_dirs=(),
    )
    later_run = _run(
        first_request.manifest,
        started_at=first_request.run.started_at + timedelta(minutes=1),
    )
    second_request = replace(first_request, run=later_run)
    second = write_procedural_artifact(
        second_request,
        tmp_path / "second",
        git_metadata_dirs=(),
    )

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.run_sha256 != second.run_sha256
    for name in (*PROCEDURAL_INDEXED_PAYLOAD_PATHS, PROCEDURAL_PAYLOAD_INDEX_FILE):
        assert (first.path / name).read_bytes() == (second.path / name).read_bytes()
    assert (first.path / PROCEDURAL_RUN_FILE).read_bytes() != (
        second.path / PROCEDURAL_RUN_FILE
    ).read_bytes()
    assert (first.path / PROCEDURAL_SUCCESS_FILE).read_bytes() != (
        second.path / PROCEDURAL_SUCCESS_FILE
    ).read_bytes()


@pytest.mark.parametrize("manifest_factory", [_availability_manifest, _common_mode_manifest])
def test_controls_use_an_exact_zero_byte_crossover_member(
    tmp_path: Path,
    manifest_factory: Any,
) -> None:
    manifest = manifest_factory()
    loaded = write_procedural_artifact(
        _request(manifest),
        tmp_path / manifest.experiment,
        git_metadata_dirs=(),
    )
    crossover_path = loaded.path / PROCEDURAL_CROSSOVERS_FILE
    crossover_entry = next(
        entry for entry in loaded.payload_index.files if entry.path == PROCEDURAL_CROSSOVERS_FILE
    )

    assert crossover_path.read_bytes() == b""
    assert loaded.crossovers == ()
    assert crossover_entry.byte_length == 0
    assert crossover_entry.sha256 == hashlib.sha256(b"").hexdigest()


def test_loader_recomputes_scientific_bundle_after_metric_tamper(
    written_procedural_artifact: LoadedProceduralArtifact,
    tmp_path: Path,
) -> None:
    copied = _copy_artifact(
        written_procedural_artifact.path,
        tmp_path,
        "semantic-tamper",
    )
    metric_path = copied / PROCEDURAL_SEQUENCE_METRICS_FILE
    lines = metric_path.read_bytes().splitlines(keepends=True)
    first = json.loads(lines[0])
    first["value"] = 42.0
    lines[0] = canonical_json_bytes(first)
    metric_path.write_bytes(b"".join(lines))

    with pytest.raises(ArtifactValidationError, match="invalid M3 procedural artifact"):
        load_procedural_artifact(copied)


def test_bundle_rejects_contract_valid_but_fabricated_validation_commitment() -> None:
    request = _request()
    eligibility = request.validation.eligibility.model_copy(
        update={"ordered_sequence_commitments_sha256": "e" * 64}
    )
    fabricated = request.validation.model_copy(update={"eligibility": eligibility})

    with pytest.raises(
        ArtifactValidationError,
        match="disagrees with independent recomputation",
    ):
        validate_procedural_bundle(
            request.manifest,
            request.profile,
            request.metrics,
            request.aggregates,
            request.crossovers,
            fabricated,
            request.run,
        )


def test_internal_run_and_profile_links_reject_each_identity_mismatch() -> None:
    request = _request()
    validate_run = vars(procedural_artifact_module)["_validate_run"]
    validate_profile = vars(procedural_artifact_module)["_validate_profile_link"]

    bad_manifest_digest = request.run.model_copy(update={"manifest_sha256": "f" * 64})
    with pytest.raises(ArtifactValidationError, match="manifest identity"):
        validate_run(request.manifest, bad_manifest_digest)

    bad_run_id = request.run.model_copy(update={"run_id": "run:wrong"})
    with pytest.raises(ArtifactValidationError, match="run_id"):
        validate_run(request.manifest, bad_run_id)

    dirty = request.run.model_copy(update={"source_dirty": True})
    with pytest.raises(ArtifactValidationError, match="clean successful"):
        validate_run(request.manifest, dirty)

    with pytest.raises(ArtifactValidationError, match="artifact identity"):
        validate_run(
            request.manifest,
            request.run,
            artifact_sha256="f" * 64,
        )

    altered_profile = request.profile.model_copy(
        update={"source": request.profile.source.model_copy(update={"frame_period_s": 0.2})}
    )
    with pytest.raises(ArtifactValidationError, match="not preregistered"):
        validate_profile(request.manifest, altered_profile)

    source = cast(ProceduralSource, request.manifest.source)
    wrong_id_manifest = request.manifest.model_copy(
        update={"source": source.model_copy(update={"profile_id": "constant-velocity-fov-edge-v1"})}
    )
    with pytest.raises(ArtifactValidationError, match="profile ID"):
        validate_profile(wrong_id_manifest, request.profile)

    wrong_digest_manifest = request.manifest.model_copy(
        update={"source": source.model_copy(update={"profile_sha256": "f" * 64})}
    )
    with pytest.raises(ArtifactValidationError, match="profile digest disagrees"):
        validate_profile(wrong_digest_manifest, request.profile)

    wrong_count_manifest = request.manifest.model_copy(
        update={"source": source.model_copy(update={"sequence_count": 5})}
    )
    with pytest.raises(ArtifactValidationError, match="split count"):
        validate_profile(wrong_count_manifest, request.profile)

    wrong_roi_manifest = request.manifest.model_copy(
        update={"roi": request.manifest.roi.model_copy(update={"x_min_m": 6.0})}
    )
    with pytest.raises(ArtifactValidationError, match="ROI"):
        validate_profile(wrong_roi_manifest, request.profile)


def test_internal_validation_links_reject_unbound_evidence() -> None:
    request = _request()
    validate_links = vars(procedural_artifact_module)["_validate_validation_links"]

    mutations = (
        (
            request.validation.model_copy(update={"run_id": "run:wrong"}),
            "validation run_id",
        ),
        (
            request.validation.model_copy(update={"manifest_sha256": "f" * 64}),
            "manifest identity",
        ),
        (
            request.validation.model_copy(update={"profile_sha256": "f" * 64}),
            "profile identity",
        ),
        (
            request.validation.model_copy(update={"sequence_count": 5}),
            "population shape",
        ),
        (
            request.validation.model_copy(
                update={
                    "resources": request.validation.resources.model_copy(
                        update={"implied_sequence_row_count": 1}
                    )
                }
            ),
            "row count",
        ),
        (
            request.validation.model_copy(
                update={
                    "resources": request.validation.resources.model_copy(
                        update={"implied_bootstrap_cell_count": 1}
                    )
                }
            ),
            "bootstrap cell count",
        ),
        (
            request.validation.model_copy(
                update={
                    "resources": request.validation.resources.model_copy(
                        update={"bootstrap_replicates": 201}
                    )
                }
            ),
            "replicate count",
        ),
        (
            request.validation.model_copy(update={"expected_loss_checks": ()}),
            "expected-loss rows",
        ),
        (
            request.validation.model_copy(update={"all_checks_passed": False}),
            "did not pass",
        ),
    )
    for validation, message in mutations:
        with pytest.raises(ArtifactValidationError, match=message):
            validate_links(
                request.manifest,
                request.profile,
                validation,
                request.run,
            )


def test_execution_caps_are_checked_before_large_allocations() -> None:
    request = _request()
    validate_caps = vars(procedural_artifact_module)["_validate_execution_caps"]
    source = cast(ProceduralSource, request.manifest.source)

    too_many_sequences = request.manifest.model_copy(
        update={
            "source": source.model_copy(
                update={"sequence_count": PROCEDURAL_MAX_SEQUENCE_COUNT + 1}
            )
        }
    )
    with pytest.raises(ArtifactValidationError, match="sequence_count"):
        validate_caps(too_many_sequences)

    bootstrap = request.manifest.evaluation.bootstrap
    too_many_replicates = request.manifest.model_copy(
        update={
            "evaluation": request.manifest.evaluation.model_copy(
                update={
                    "bootstrap": bootstrap.model_copy(
                        update={"replicates": PROCEDURAL_MAX_BOOTSTRAP_REPLICATES + 1}
                    )
                }
            )
        }
    )
    with pytest.raises(ArtifactValidationError, match="bootstrap replicates"):
        validate_caps(too_many_replicates)

    too_many_cells = request.manifest.model_copy(
        update={
            "source": source.model_copy(update={"sequence_count": 10_000}),
            "evaluation": request.manifest.evaluation.model_copy(
                update={"bootstrap": bootstrap.model_copy(update={"replicates": 2_001})}
            ),
        }
    )
    with pytest.raises(ArtifactValidationError, match="bootstrap matrix"):
        validate_caps(too_many_cells)

    with pytest.raises(ArtifactValidationError, match="sequence rows"):
        validate_caps(
            request.manifest,
            sequence_rows=PROCEDURAL_MAX_SEQUENCE_ROWS + 1,
        )


def test_loader_rejects_noncanonical_order_and_file_allowlist_changes(
    written_procedural_artifact: LoadedProceduralArtifact,
    tmp_path: Path,
) -> None:
    wrong_order = _copy_artifact(
        written_procedural_artifact.path,
        tmp_path,
        "wrong-order",
    )
    metric_path = wrong_order / PROCEDURAL_SEQUENCE_METRICS_FILE
    lines = metric_path.read_bytes().splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    metric_path.write_bytes(b"".join(lines))
    with pytest.raises(ArtifactValidationError, match="canonical order"):
        load_procedural_artifact(wrong_order)

    extra = _copy_artifact(written_procedural_artifact.path, tmp_path, "extra")
    (extra / "extra.json").write_bytes(b"{}\n")
    with pytest.raises(ArtifactValidationError, match="allowlist mismatch"):
        load_procedural_artifact(extra)

    missing = _copy_artifact(written_procedural_artifact.path, tmp_path, "missing")
    (missing / PROCEDURAL_SUCCESS_FILE).unlink()
    with pytest.raises(ArtifactValidationError, match="allowlist mismatch"):
        load_procedural_artifact(missing)


def test_loader_rejects_member_root_and_parent_symlinks(
    written_procedural_artifact: LoadedProceduralArtifact,
    tmp_path: Path,
) -> None:
    linked_member = _copy_artifact(
        written_procedural_artifact.path,
        tmp_path,
        "linked-member",
    )
    run_path = linked_member / PROCEDURAL_RUN_FILE
    run_path.unlink()
    run_path.symlink_to(written_procedural_artifact.path / PROCEDURAL_RUN_FILE)
    with pytest.raises(ArtifactValidationError, match="symlink"):
        load_procedural_artifact(linked_member)

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(written_procedural_artifact.path, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="symlink"):
        load_procedural_artifact(linked_root)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    _copy_artifact(written_procedural_artifact.path, real_parent, "artifact")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="symlink"):
        load_procedural_artifact(linked_parent / "artifact")


def test_loader_binds_reads_to_the_scanned_inode(
    written_procedural_artifact: LoadedProceduralArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = _copy_artifact(
        written_procedural_artifact.path,
        tmp_path,
        "inode-swap",
    )
    manifest_bytes = (copied / PROCEDURAL_MANIFEST_FILE).read_bytes()
    replacement = tmp_path / "replacement-manifest.json"
    original_read = vars(procedural_artifact_module)["_read_member"]
    swapped = False

    def swap_before_read(
        root: Path,
        name: str,
        *,
        expected_stat: Any,
        byte_cap: int,
    ) -> bytes:
        nonlocal swapped
        if name == PROCEDURAL_MANIFEST_FILE and not swapped:
            replacement.write_bytes(manifest_bytes)
            replacement.replace(root / name)
            swapped = True
        return original_read(
            root,
            name,
            expected_stat=expected_stat,
            byte_cap=byte_cap,
        )

    monkeypatch.setattr(
        procedural_artifact_module,
        "_read_member",
        swap_before_read,
    )
    with pytest.raises(
        ArtifactValidationError,
        match="member changed during validation",
    ):
        load_procedural_artifact(copied)


def test_index_hashing_is_bounded_if_a_member_grows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = tmp_path / "member"
    member.write_bytes(b"x")
    calls: list[int] = []

    def endless_read(_descriptor: int, byte_count: int) -> bytes:
        calls.append(byte_count)
        return b"x"

    monkeypatch.setattr(procedural_artifact_module.os, "read", endless_read)
    sha256_member = vars(procedural_artifact_module)["_sha256_member"]
    with pytest.raises(ArtifactValidationError, match="changed during hashing"):
        sha256_member(
            tmp_path,
            member.name,
            expected_stat=member.stat(),
        )
    assert calls == [1, 1]


def test_writer_never_replaces_existing_or_dangling_destination(
    tmp_path: Path,
) -> None:
    request = _request()
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_procedural_artifact(request, existing, git_metadata_dirs=())
    assert sentinel.read_text(encoding="utf-8") == "keep"

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        write_procedural_artifact(request, dangling, git_metadata_dirs=())
    assert dangling.is_symlink()


def test_writer_rejects_symlink_parent_and_git_metadata(tmp_path: Path) -> None:
    request = _request()
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="not a real directory"):
        write_procedural_artifact(
            request,
            linked_parent / "artifact",
            git_metadata_dirs=(),
        )

    git_metadata = tmp_path / "git-metadata"
    git_metadata.mkdir()
    with pytest.raises(ArtifactValidationError, match="inside Git metadata"):
        write_procedural_artifact(
            request,
            git_metadata / "artifact",
            git_metadata_dirs=(git_metadata,),
        )
