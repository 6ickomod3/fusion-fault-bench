from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import fusion_fault_bench.procedural_release as release_module
from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.bundle_v1alpha1 import expected_conditions
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AvailabilityControlManifest,
    CommonModeControlManifest,
    GeometryCrossoverManifest,
    ProceduralSource,
)
from fusion_fault_bench.contracts.matrix_v1 import (
    LoadedExperimentMatrix,
    load_experiment_matrix,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    SeverityCoordinate,
)
from fusion_fault_bench.procedural_release import (
    RepeatRunResources,
    build_m3_matrix_validation,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = Path("examples/matrices/m3-procedural-v1.json")
PUBLIC_CI_RUN_ID = 123456
RESULTS_REVIEW_REFERENCE = "docs/reviews/m3-results-review.md"
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))
curation = importlib.import_module("tools.m3_curation")
release_tool = importlib.import_module("tools.m3_release")
release_fixtures = importlib.import_module("test_procedural_release")
validation_fixtures = importlib.import_module("test_procedural_validation_runtime")
CuratedExperiment = vars(curation)["CuratedExperiment"]
CuratedRelease = vars(curation)["CuratedRelease"]
M3CurationError = vars(curation)["M3CurationError"]
_FAKE_ARTIFACT_SET = vars(release_fixtures)["_fake_artifact_set"]
_SECOND_RUN_ARTIFACTS = vars(release_fixtures)["_second_run_artifacts"]
_EXPECTED_COUNTS = vars(curation)["_expected_counts"]
_VALIDATE_CURATED_ROWS = vars(curation)["_validate_curated_rows"]
_RELEASE_ALLOWLIST = vars(curation)["_release_allowlist"]
_SCAN_RELEASE = vars(curation)["_scan_release"]
_VALIDATE_INDEX_PATHS = vars(curation)["_validate_index_paths"]
_BUILD_PARSER = vars(release_tool)["_build_parser"]
_STRICT_RELEASE_INPUTS = vars(release_tool)["_strict_release_inputs"]
_BUILD_VALIDATION = vars(validation_fixtures)["_build"]
_SMOKE_VALIDATION_MANIFEST = vars(validation_fixtures)["_smoke_manifest"]


@pytest.fixture(scope="module")
def release_matrix() -> LoadedExperimentMatrix:
    return load_experiment_matrix(MATRIX_PATH, source_root=REPOSITORY_ROOT)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _with_command(
    artifacts: tuple[Any, ...],
    *,
    output_dir: str,
) -> tuple[Any, ...]:
    command = (
        "ffb",
        "procedural",
        "matrix",
        "run",
        MATRIX_PATH.as_posix(),
        "--output-dir",
        output_dir,
    )
    return tuple(
        replace(
            artifact,
            run=artifact.run.model_copy(update={"command": command}),
        )
        for artifact in artifacts
    )


def _identity_fixture(
    matrix: LoadedExperimentMatrix,
) -> tuple[
    tuple[Any, ...],
    tuple[Any, ...],
    Any,
    Any,
]:
    primary = _with_command(
        _FAKE_ARTIFACT_SET(matrix),
        output_dir="reports/generated/m3-primary",
    )
    repeated = _with_command(
        _SECOND_RUN_ARTIFACTS(primary),
        output_dir="reports/generated/m3-repeat",
    )
    matrix_evidence = build_m3_matrix_validation(matrix, primary)
    build_repeat = vars(release_module)["_repeat_verification_from_artifacts"]
    repeat_evidence = build_repeat(
        matrix,
        primary,
        repeated,
        first_resources=RepeatRunResources(
            wall_time_seconds=10.0,
            peak_memory_bytes=1_000,
        ),
        second_resources=RepeatRunResources(
            wall_time_seconds=11.0,
            peak_memory_bytes=1_100,
        ),
    )
    return primary, repeated, matrix_evidence, repeat_evidence


def _attestation_kwargs(
    primary: tuple[Any, ...],
    matrix_evidence: Any,
) -> dict[str, bytes]:
    results_review_report = (
        f"# M3 adversarial results review — {curation.RELEASE_ID}\n"
        "\n"
        "Verdict: **PASS**\n"
        "\n"
        f"Artifact set: `{matrix_evidence.artifact_set_sha256}`\n"
    ).encode()
    public_ci = {
        "schema": curation.PUBLIC_CI_ATTESTATION_SCHEMA,
        "provider": "github-actions",
        "repository": curation.PUBLIC_CI_REPOSITORY,
        "workflow": "ci",
        "workflow_path": ".github/workflows/ci.yml",
        "run_id": PUBLIC_CI_RUN_ID,
        "url": f"{curation.PUBLIC_CI_URL_PREFIX}{PUBLIC_CI_RUN_ID}",
        "source_revision": primary[0].run.git_revision,
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
        "reviewed_artifact_set_sha256": matrix_evidence.artifact_set_sha256,
        "reviewer": "independent-adversarial-agent",
        "reference": RESULTS_REVIEW_REFERENCE,
        "reference_sha256": hashlib.sha256(results_review_report).hexdigest(),
        "reference_byte_length": len(results_review_report),
        "verification_scope": curation.RESULTS_REVIEW_VERIFICATION_SCOPE,
    }
    return {
        "public_ci_attestation_bytes": canonical_json_bytes(public_ci),
        "results_review_attestation_bytes": canonical_json_bytes(results_review),
        "results_review_report_bytes": results_review_report,
    }


def test_official_identity_is_deterministic_and_pins_all_run_records(
    release_matrix: LoadedExperimentMatrix,
) -> None:
    primary, repeated, matrix_evidence, repeat_evidence = _identity_fixture(release_matrix)

    first = curation.derive_official_identity(
        release_matrix,
        primary,
        repeated,
        matrix_validation=matrix_evidence,
        repeat_verification=repeat_evidence,
        **_attestation_kwargs(primary, matrix_evidence),
        expected_first_output="reports/generated/m3-primary",
        expected_second_output="reports/generated/m3-repeat",
    )
    second = curation.derive_official_identity(
        release_matrix,
        primary,
        repeated,
        matrix_validation=matrix_evidence,
        repeat_verification=repeat_evidence,
        **_attestation_kwargs(primary, matrix_evidence),
        expected_first_output="reports/generated/m3-primary",
        expected_second_output="reports/generated/m3-repeat",
    )

    assert first == second
    assert first["matrix_sha256"] == release_matrix.matrix_sha256
    assert first["artifact_set_sha256"] == matrix_evidence.artifact_set_sha256
    entries = cast(list[dict[str, Any]], first["ordered_artifacts"])
    assert len(entries) == 8
    assert [entry["primary_run_sha256"] for entry in entries] == list(
        repeat_evidence.first_run.run_record_sha256s
    )
    assert [entry["repeat_run_sha256"] for entry in entries] == list(
        repeat_evidence.second_run.run_record_sha256s
    )


def test_official_identity_rejects_command_and_evidence_tampering(
    release_matrix: LoadedExperimentMatrix,
) -> None:
    primary, repeated, matrix_evidence, repeat_evidence = _identity_fixture(release_matrix)
    wrong_command = list(primary)
    wrong_command[0] = replace(
        wrong_command[0],
        run=wrong_command[0].run.model_copy(
            update={
                "command": (
                    "ffb",
                    "procedural",
                    "matrix",
                    "run",
                    MATRIX_PATH.as_posix(),
                    "--output-dir",
                    "reports/generated/other-root",
                )
            }
        ),
    )
    with pytest.raises(M3CurationError, match="command changed"):
        curation.derive_official_identity(
            release_matrix,
            tuple(wrong_command),
            repeated,
            matrix_validation=matrix_evidence,
            repeat_verification=repeat_evidence,
            **_attestation_kwargs(primary, matrix_evidence),
        )

    tampered_repeat = repeat_evidence.model_copy(
        update={
            "first_run": repeat_evidence.first_run.model_copy(
                update={"run_record_sha256s": (_digest("tampered"),) * 8}
            )
        }
    )
    with pytest.raises(M3CurationError, match="run-record digests"):
        curation.derive_official_identity(
            release_matrix,
            primary,
            repeated,
            matrix_validation=matrix_evidence,
            repeat_verification=tampered_repeat,
            **_attestation_kwargs(primary, matrix_evidence),
        )


def test_build_rejects_official_identity_mismatch_before_writing(
    release_matrix: LoadedExperimentMatrix,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary, repeated, matrix_evidence, repeat_evidence = _identity_fixture(release_matrix)
    attestations = _attestation_kwargs(primary, matrix_evidence)
    expected = curation.derive_official_identity(
        release_matrix,
        primary,
        repeated,
        matrix_validation=matrix_evidence,
        repeat_verification=repeat_evidence,
        **attestations,
    )
    wrong = {**expected, "artifact_set_sha256": _digest("wrong-artifact-set")}

    def skip_source_gate(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        curation,
        "_require_input_release_matrix",
        skip_source_gate,
    )

    with pytest.raises(M3CurationError, match="official identity disagrees"):
        curation.build_curated_release(
            release_matrix,
            primary,
            repeated,
            matrix_validation=matrix_evidence,
            repeat_verification=repeat_evidence,
            official_identity_bytes=canonical_json_bytes(wrong),
            results_review_report_bytes=attestations["results_review_report_bytes"],
            output_dir=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_curation_orchestrator_invokes_strict_both_root_eligibility(
    release_matrix: LoadedExperimentMatrix,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    primary, repeated, matrix_evidence, repeat_evidence = _identity_fixture(release_matrix)
    evidence = SimpleNamespace(
        matrix_validation=matrix_evidence,
        repeat_verification=repeat_evidence,
    )
    snapshot = SimpleNamespace(source_root=tmp_path)
    first_relative = Path("reports/generated/first")
    second_relative = Path("reports/generated/second")
    evidence_relative = Path("reports/generated/evidence")
    first_root = tmp_path / first_relative
    second_root = tmp_path / second_relative
    captured: dict[str, Any] = {}

    def fake_validate_repeat(*_args: Any, **_kwargs: Any) -> Any:
        return evidence

    def fake_snapshot(_path: Path) -> Any:
        return snapshot

    def fake_matrix(*_args: Any, **_kwargs: Any) -> LoadedExperimentMatrix:
        return release_matrix

    def fake_artifacts(_matrix: Any, root: Path) -> tuple[Any, ...]:
        return primary if root == first_root else repeated

    def skip_source_check(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(
        release_tool,
        "validate_procedural_repeat",
        fake_validate_repeat,
    )
    monkeypatch.setattr(release_tool, "_initial_snapshot", fake_snapshot)
    monkeypatch.setattr(
        release_tool,
        "_load_matrix_for_snapshot",
        fake_matrix,
    )
    monkeypatch.setattr(
        release_tool,
        "load_m3_artifact_set",
        fake_artifacts,
    )

    def capture_gate(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        release_tool,
        "validate_m3_release_eligibility",
        capture_gate,
    )
    monkeypatch.setattr(
        release_tool,
        "_verify_unchanged_source",
        skip_source_check,
    )

    loaded = _STRICT_RELEASE_INPUTS(
        MATRIX_PATH,
        first_output_dir=first_relative,
        second_output_dir=second_relative,
        evidence_dir=evidence_relative,
    )

    assert loaded.first_artifacts is primary
    assert loaded.second_artifacts is repeated
    assert captured["first_run_root"] == first_root
    assert captured["second_run_root"] == second_root
    assert captured["matrix"] is release_matrix


def _aggregate_pairs(
    manifest: GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest,
) -> tuple[tuple[str, str], ...]:
    if isinstance(manifest, AvailabilityControlManifest):
        return tuple(
            (method, metric)
            for method in manifest.methods
            for metric in manifest.evaluation.metrics
        )
    if isinstance(manifest, GeometryCrossoverManifest):
        return tuple(
            pair
            for method in manifest.methods
            for pair in (
                (
                    (method, "matched-center-mse"),
                    (method, "fused-minus-healthy"),
                )
                if method == "fixed-fusion"
                else ((method, "matched-center-mse"),)
            )
        )
    return tuple((method, "matched-center-mse") for method in manifest.methods)


def _aggregate_rows(
    manifest: GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest,
    *,
    only_fusion_delta: bool = False,
) -> tuple[AggregateMetricRecordV1Alpha1, ...]:
    run_id = f"run:{_digest(manifest.experiment)}"
    manifest_sha256 = sha256_digest(manifest)
    records: list[AggregateMetricRecordV1Alpha1] = []
    for condition in expected_conditions(manifest):
        severity = SeverityCoordinate(
            index=condition.severity_index,
            magnitude=condition.magnitude,
            direction=cast(Any, condition.direction),
            unit=cast(Any, condition.unit),
        )
        for method, metric in _aggregate_pairs(manifest):
            if only_fusion_delta and metric != "fused-minus-healthy":
                continue
            is_rate = metric in {"coverage", "undefined-output-rate"}
            undefined = (
                metric == "conditional-matched-center-mse"
                and method == "camera-only"
                and condition.magnitude == 1.0
            )
            if is_rate:
                estimate = 0.6
                lower = 0.55
                upper = 0.65
                unit = "fraction"
            elif metric == "fused-minus-healthy":
                estimate = condition.magnitude - 0.2
                lower = estimate - 0.05
                upper = estimate + 0.05
                unit = "m^2"
            else:
                estimate = 0.2 + condition.magnitude
                lower = estimate - 0.05
                upper = estimate + 0.05
                unit = "m^2"
            records.append(
                AggregateMetricRecordV1Alpha1(
                    schema="ffb.aggregate-metric/v1alpha1",
                    record_level="aggregate",
                    run_id=run_id,
                    manifest_sha256=manifest_sha256,
                    fault_family=cast(Any, condition.fault_family),
                    fault_axis=cast(Any, condition.fault_axis),
                    severity=severity,
                    method_id=cast(Any, method),
                    metric_name=cast(Any, metric),
                    status="undefined" if undefined else "ok",
                    estimate=None if undefined else estimate,
                    interval_lower=None if undefined else lower,
                    interval_upper=None if undefined else upper,
                    unit=cast(Any, unit),
                    sequence_count=cast(ProceduralSource, manifest.source).sequence_count,
                    contributing_sequence_count=(
                        0 if undefined else cast(ProceduralSource, manifest.source).sequence_count
                    ),
                    bootstrap_replicates=manifest.evaluation.bootstrap.replicates,
                    defined_bootstrap_replicates=(
                        0 if undefined else manifest.evaluation.bootstrap.replicates
                    ),
                    confidence_level=0.95,
                    interval_method="paired-sequence-percentile-pointwise",
                    aggregation=cast(
                        Any,
                        (
                            "count-ratio-with-sequence-bootstrap"
                            if metric in {"coverage", "undefined-output-rate"}
                            else (
                                "valid-object-frame-ratio-with-sequence-bootstrap"
                                if metric == "conditional-matched-center-mse"
                                else "object-frame-mean-then-sequence-mean"
                            )
                        ),
                    ),
                )
            )
    return tuple(records)


def _crossover_rows(
    manifest: GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest,
) -> tuple[CrossoverRecordV1Alpha1, ...]:
    if not isinstance(manifest, GeometryCrossoverManifest):
        return ()
    directions = tuple(
        dict.fromkeys(
            condition.direction
            for condition in expected_conditions(manifest)
            if condition.direction != "identity"
        )
    )
    maximum = max(condition.magnitude for condition in expected_conditions(manifest))
    return tuple(
        CrossoverRecordV1Alpha1(
            schema="ffb.crossover/v1alpha1",
            run_id=f"run:{_digest(manifest.experiment)}",
            manifest_sha256=sha256_digest(manifest),
            fault_family=manifest.fault_sweep.kind,
            fault_axis=manifest.fault_sweep.axis,
            direction=cast(Any, direction),
            severity_unit=manifest.fault_sweep.unit,
            status="not-observed",
            point_curve_crossed=False,
            point_estimate=None,
            interval_lower=maximum,
            interval_upper="positive-infinity",
            tested_maximum=maximum,
            censoring="right-above-tested-maximum",
            bootstrap_crossing_fraction=0.0,
            sequence_count=cast(ProceduralSource, manifest.source).sequence_count,
            bootstrap_replicates=manifest.evaluation.bootstrap.replicates,
            confidence_level=0.95,
            interval_method="right-censored-percentile",
        )
        for direction in directions
    )


def test_frozen_matrix_implies_literal_counts_and_exact_row_order(
    release_matrix: LoadedExperimentMatrix,
) -> None:
    first_manifest = cast(GeometryCrossoverManifest, release_matrix.manifests[0])
    assert _aggregate_pairs(first_manifest) == (
        ("camera-only", "matched-center-mse"),
        ("lidar-only", "matched-center-mse"),
        ("fixed-fusion", "matched-center-mse"),
        ("fixed-fusion", "fused-minus-healthy"),
        ("fault-target-drop-policy", "matched-center-mse"),
        ("performance-oracle", "matched-center-mse"),
    )
    totals = [0, 0, 0]
    for manifest in release_matrix.manifests:
        typed = cast(
            GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest,
            manifest,
        )
        counts = _EXPECTED_COUNTS(typed)
        assert counts == curation.EXPECTED_COUNTS[typed.experiment]
        aggregates = _aggregate_rows(typed)
        crossovers = _crossover_rows(typed)
        _VALIDATE_CURATED_ROWS(
            typed,
            run_id=f"run:{_digest(typed.experiment)}",
            manifest_sha256=sha256_digest(typed),
            aggregates=aggregates,
            crossovers=crossovers,
        )
        if len(aggregates) > 1:
            changed = (aggregates[1], aggregates[0], *aggregates[2:])
            with pytest.raises(M3CurationError, match="out of order"):
                _VALIDATE_CURATED_ROWS(
                    typed,
                    run_id=f"run:{_digest(typed.experiment)}",
                    manifest_sha256=sha256_digest(typed),
                    aggregates=changed,
                    crossovers=crossovers,
                )
            with pytest.raises(M3CurationError, match="incomplete"):
                _VALIDATE_CURATED_ROWS(
                    typed,
                    run_id=f"run:{_digest(typed.experiment)}",
                    manifest_sha256=sha256_digest(typed),
                    aggregates=aggregates[:-1],
                    crossovers=crossovers,
                )
        for index, value in enumerate(counts):
            totals[index] += value
    assert tuple(totals) == (71_700, 429, 10)


def _write_allowlisted_empty_tree(root: Path) -> None:
    for relative in _RELEASE_ALLOWLIST():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_release_tree_rejects_raw_rows_unlisted_files_and_symlinks(
    tmp_path: Path,
) -> None:
    release_root = tmp_path / "release"
    _write_allowlisted_empty_tree(release_root)
    assert _SCAN_RELEASE(release_root) == _RELEASE_ALLOWLIST()

    raw = release_root / "records" / curation.EXPECTED_EXPERIMENTS[0] / "sequence-metrics.ndjson"
    raw.write_bytes(b"{}\n")
    with pytest.raises(M3CurationError, match="raw sequence metrics"):
        _SCAN_RELEASE(release_root)
    raw.unlink()

    extra = release_root / "unexpected.txt"
    extra.write_bytes(b"x")
    with pytest.raises(M3CurationError, match="allowlist"):
        _SCAN_RELEASE(release_root)
    extra.unlink()

    readme = release_root / curation.README_PATH
    readme.unlink()
    readme.symlink_to(release_root / curation.VERIFICATION_PATH)
    with pytest.raises(M3CurationError, match="symlink"):
        _SCAN_RELEASE(release_root)

    linked_root = tmp_path / "linked-release"
    linked_root.symlink_to(release_root, target_is_directory=True)
    with pytest.raises(M3CurationError, match="symlink components"):
        _SCAN_RELEASE(linked_root)


@pytest.mark.parametrize("value", ("../escape", "records\\escape", "/absolute", "a/./b"))
def test_release_index_paths_reject_path_tricks(value: str) -> None:
    with pytest.raises(M3CurationError):
        _VALIDATE_INDEX_PATHS(
            {"files": [{"path": value}]},
            scanned={curation.RELEASE_INDEX_PATH, Path("README.md")},
        )


def test_release_index_rejects_duplicate_paths() -> None:
    with pytest.raises(M3CurationError, match="duplicate"):
        _VALIDATE_INDEX_PATHS(
            {"files": [{"path": "README.md"}, {"path": "README.md"}]},
            scanned={curation.RELEASE_INDEX_PATH, Path("README.md")},
        )


def _figure_release(matrix: LoadedExperimentMatrix) -> Any:
    profiles = {profile.profile_id: profile for profile in matrix.profiles}
    experiments: list[Any] = []
    for index, raw_manifest in enumerate(matrix.manifests):
        manifest = cast(
            GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest,
            raw_manifest,
        )
        source = cast(ProceduralSource, manifest.source)
        aggregates = (
            _aggregate_rows(manifest, only_fusion_delta=True)
            if index < 6
            else _aggregate_rows(manifest)
        )
        common = (
            SimpleNamespace(
                status="applicable",
                maximum_disagreement_discrepancy_m=0.0,
                tolerance_m=1e-12,
                passed=True,
            )
            if index == 7
            else SimpleNamespace(status="not-applicable")
        )
        validation = SimpleNamespace(
            all_checks_passed=True,
            common_mode_validation=common,
        )
        experiments.append(
            CuratedExperiment(
                execution_index=index,
                experiment=manifest.experiment,
                manifest=manifest,
                profile=profiles[source.profile_id],
                aggregates=aggregates,
                crossovers=_crossover_rows(manifest),
                validation=cast(Any, validation),
                payload_index=cast(Any, None),
                primary_run=cast(Any, None),
                primary_success=cast(Any, None),
                repeat_run=cast(Any, None),
                repeat_success=cast(Any, None),
                artifact_sha256=_digest(f"artifact:{index}"),
                primary_run_sha256=_digest(f"primary:{index}"),
                repeat_run_sha256=_digest(f"repeat:{index}"),
                source_sequence_byte_length=1,
                source_sequence_sha256=_digest(f"sequence:{index}"),
            )
        )
    repeat = SimpleNamespace(
        first_run=SimpleNamespace(
            cpu_model="Test CPU",
            wall_time_seconds=10.0,
            peak_memory_bytes=1_000,
        ),
        second_run=SimpleNamespace(
            cpu_model="Test CPU",
            wall_time_seconds=11.0,
            peak_memory_bytes=1_100,
        ),
        comparison_count=48,
        mismatch_count=0,
        resource_measurement_scope=(
            "self-reported-by-tracked-wait4-driver-not-independently-recomputable"
        ),
        execution_evidence_scope=(
            "distinct-path-inode-and-run-record-consistency-not-cryptographic-proof"
        ),
    )
    return CuratedRelease(
        matrix=matrix.matrix,
        matrix_sha256=matrix.matrix_sha256,
        profiles=matrix.profiles,
        experiments=tuple(experiments),
        matrix_validation=cast(Any, None),
        repeat_verification=cast(Any, repeat),
    )


def test_figures_and_documents_are_deterministic_and_result_complete(
    release_matrix: LoadedExperimentMatrix,
) -> None:
    release = _figure_release(release_matrix)
    identity = {
        "scientific_source_revision": "a" * 40,
        "lockfile_sha256": "b" * 64,
        "artifact_set_sha256": "c" * 64,
        "matrix_validation_sha256": "d" * 64,
        "repeat_verification_sha256": "e" * 64,
        "public_ci": {
            "attestation": {
                "run_id": PUBLIC_CI_RUN_ID,
                "url": f"{curation.PUBLIC_CI_URL_PREFIX}{PUBLIC_CI_RUN_ID}",
                "source_revision": "a" * 40,
                "conclusion": "success",
            }
        },
        "results_review": {
            "attestation": {
                "status": "pass",
                "reference": RESULTS_REVIEW_REFERENCE,
                "reviewed_artifact_set_sha256": "c" * 64,
            }
        },
    }
    renderers = (
        curation.render_fusion_delta_figure,
        curation.render_dropout_figure,
        curation.render_common_mode_figure,
    )
    for renderer in renderers:
        first = renderer(release)
        assert first == renderer(release)
        assert first.startswith(b'<?xml version="1.0" encoding="UTF-8"?>\n<svg ')
        assert b"<image" not in first
        assert b"<script" not in first

    assert b"All 54 unique" in curation.render_fusion_delta_figure(release)
    assert b"All 72 aggregate rows" in curation.render_dropout_figure(release)
    common_figure = curation.render_common_mode_figure(release)
    assert b"All 33 camera-only" in common_figure
    assert b"MSE rises" not in common_figure
    for renderer in (
        curation.render_readme,
        curation.render_verification,
        curation.render_claim_evidence,
    ):
        first = renderer(release, identity)
        assert first == renderer(release, identity)
        assert first.endswith(b"\n")
    readme = curation.render_readme(release, identity)
    assert b"rising absolute loss" not in readme
    assert b"](figures/fusion-delta-curves.svg)" in readme
    assert b"](figures/dropout-controls.svg)" in readme
    assert b"](figures/common-mode-control.svg)" in readme
    assert b"figure presentation order" in curation.render_claim_evidence(
        release,
        identity,
    )

    shortened = list(release.experiments)
    shortened[0] = replace(
        shortened[0],
        aggregates=shortened[0].aggregates[:-1],
    )
    with pytest.raises(M3CurationError, match="completeness"):
        curation.render_fusion_delta_figure(replace(release, experiments=tuple(shortened)))


def test_cli_parser_exposes_identity_and_release_stages() -> None:
    parser = _BUILD_PARSER()
    derived = parser.parse_args(
        [
            "derive-identity",
            MATRIX_PATH.as_posix(),
            "--first-output-dir",
            "reports/generated/first",
            "--second-output-dir",
            "reports/generated/second",
            "--evidence-dir",
            "reports/generated/evidence",
            "--output-path",
            "reports/generated/identity.json",
        ]
    )
    assert derived.command == "derive-identity"
    assert derived.public_ci_attestation == curation.PUBLIC_CI_ATTESTATION_RELATIVE_PATH
    assert derived.results_review_attestation == curation.RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH
    assert derived.results_review_report == curation.RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH
    built = parser.parse_args(
        [
            "build-release",
            MATRIX_PATH.as_posix(),
            "--first-output-dir",
            "reports/generated/first",
            "--second-output-dir",
            "reports/generated/second",
            "--evidence-dir",
            "reports/generated/evidence",
        ]
    )
    assert built.official_identity == curation.OFFICIAL_IDENTITY_RELATIVE_PATH
    assert built.output_dir == curation.RELEASE_RELATIVE_PATH
    validated = parser.parse_args(["validate-release", curation.RELEASE_RELATIVE_PATH.as_posix()])
    assert validated.command == "validate-release"


def test_validation_summary_never_pools_m_and_m2_oracle_discrepancies() -> None:
    validation = _BUILD_VALIDATION(_SMOKE_VALIDATION_MANIFEST())
    summary = curation._validation_summary(validation)
    assert "maximum_oracle_discrepancy" not in summary
    assert set(summary["maximum_oracle_discrepancy_by_unit"]) == {"m", "m^2"}


def test_exact_prose_float_and_cpu_model_are_presentation_safe() -> None:
    assert curation._exact_float_text(0.1) == "0.1"
    assert curation._specific_cpu_model("Apple M4 Pro")
    assert curation._specific_cpu_model("AMD EPYC 7763 64-Core Processor")
    assert not curation._specific_cpu_model("CPU `injected`")
    assert not curation._specific_cpu_model("CPU\ninjected")
