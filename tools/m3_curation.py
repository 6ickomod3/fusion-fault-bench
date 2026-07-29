"""Deterministic aggregate-only curation for the M3 procedural release tool."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from fusion_fault_bench.artifacts import (
    canonical_json_bytes,
    compute_run_record_digest,
    derive_run_id,
    publish_directory_no_replace,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import SuccessMarkerV1Alpha1
from fusion_fault_bench.contracts.bundle_v1alpha1 import expected_conditions
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    EXPERIMENT_MANIFEST_ADAPTER,
    AvailabilityControlManifest,
    CommonModeControlManifest,
    GeometryCrossoverManifest,
    ProceduralSource,
)
from fusion_fault_bench.contracts.matrix_v1 import (
    EXPERIMENT_MATRIX_ADAPTER,
    M3_CI_SMOKE_MATRIX_SHA256,
    M3_PROCEDURAL_MATRIX_SHA256,
    ExperimentMatrixV1,
    LoadedExperimentMatrix,
)
from fusion_fault_bench.contracts.procedural_artifact_v1 import (
    PROCEDURAL_AGGREGATE_METRICS_FILE,
    PROCEDURAL_ARTIFACT_CONTRACT,
    PROCEDURAL_CROSSOVERS_FILE,
    PROCEDURAL_INDEXED_PAYLOAD_PATHS,
    PROCEDURAL_MANIFEST_FILE,
    PROCEDURAL_MAX_RECORD_BYTES,
    PROCEDURAL_PAYLOAD_INDEX_FILE,
    PROCEDURAL_PROFILE_FILE,
    PROCEDURAL_RUN_FILE,
    PROCEDURAL_SEQUENCE_METRICS_FILE,
    PROCEDURAL_SUCCESS_FILE,
    PROCEDURAL_VALIDATION_FILE,
    ProceduralPayloadIndexV1Alpha2,
)
from fusion_fault_bench.contracts.procedural_profile_v1 import (
    PROCEDURAL_PROFILE_ADAPTER,
    ProceduralProfileV1,
)
from fusion_fault_bench.contracts.procedural_release_v1 import (
    M3MatrixValidationV1,
    RepeatVerificationV1,
)
from fusion_fault_bench.contracts.procedural_validation_v1 import (
    ProceduralValidationV1,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    AggregateMetricRecordV1Alpha1,
    CrossoverRecordV1Alpha1,
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.inference import pava_non_decreasing
from fusion_fault_bench.procedural_artifacts import (
    LoadedProceduralArtifact,
    compute_procedural_artifact_digest,
)

type ProceduralManifest = (
    GeometryCrossoverManifest | AvailabilityControlManifest | CommonModeControlManifest
)

RELEASE_ID = "m3-procedural-v0.1.0"
RELEASE_SCHEMA = "ffb.m3-release/v1"
OFFICIAL_IDENTITY_SCHEMA = "ffb.m3-official-identity/v1"
SUMMARY_SCHEMA = "ffb.m3-release-summary/v1"
PUBLIC_CI_REPOSITORY = "6ickomod3/fusion-fault-bench"
PUBLIC_CI_URL_PREFIX = f"https://github.com/{PUBLIC_CI_REPOSITORY}/actions/runs/"
PUBLIC_CI_ATTESTATION_SCHEMA = "ffb.public-ci-attestation/v1"
RESULTS_REVIEW_ATTESTATION_SCHEMA = "ffb.results-review-attestation/v1"
PUBLIC_CI_ATTESTATION_RELATIVE_PATH = Path("docs/reviews/m3-public-ci-attestation.json")
RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH = Path("docs/reviews/m3-results-review-attestation.json")
RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH = Path("docs/reviews/m3-results-review.md")
RESULTS_REVIEW_INCLUDED_PATH = Path("evidence/results-review.md")
PUBLIC_CI_VERIFICATION_SCOPE = (
    "tracked-operator-attestation; offline-validator-does-not-query-github"
)
RESULTS_REVIEW_VERIFICATION_SCOPE = (
    "tracked-review-attestation; reviewer-identity-is-not-cryptographically-authenticated"
)
RESULTS_REVIEW_SCOPE = (
    "semantics",
    "geometry",
    "leakage",
    "statistics",
    "selection",
    "claims",
    "privacy",
)
RELEASE_RELATIVE_PATH = Path(f"reports/releases/{RELEASE_ID}")
OFFICIAL_IDENTITY_RELATIVE_PATH = Path(f"examples/release-identities/{RELEASE_ID}.json")

MATRIX_PATH = Path("intent/matrix.json")
MATRIX_VALIDATION_PATH = Path("evidence/matrix-validation.json")
REPEAT_VERIFICATION_PATH = Path("evidence/repeat-verification.json")
OFFICIAL_IDENTITY_PATH = Path("evidence/official-identity.json")
RESULTS_REVIEW_REPORT_PATH = RESULTS_REVIEW_INCLUDED_PATH
SUMMARY_PATH = Path("release-summary.json")
README_PATH = Path("README.md")
VERIFICATION_PATH = Path("verification.md")
CLAIM_EVIDENCE_PATH = Path("claim-evidence.md")
FUSION_DELTA_FIGURE_PATH = Path("figures/fusion-delta-curves.svg")
DROPOUT_FIGURE_PATH = Path("figures/dropout-controls.svg")
COMMON_MODE_FIGURE_PATH = Path("figures/common-mode-control.svg")
RELEASE_INDEX_PATH = Path("release-index.json")

DOCUMENT_PATHS = (README_PATH, VERIFICATION_PATH, CLAIM_EVIDENCE_PATH)
FIGURE_PATHS = (
    FUSION_DELTA_FIGURE_PATH,
    DROPOUT_FIGURE_PATH,
    COMMON_MODE_FIGURE_PATH,
)
EVIDENCE_PATHS = (
    MATRIX_VALIDATION_PATH,
    REPEAT_VERIFICATION_PATH,
    OFFICIAL_IDENTITY_PATH,
    RESULTS_REVIEW_REPORT_PATH,
)

RECORD_MEMBER_DESTINATIONS = (
    (PROCEDURAL_AGGREGATE_METRICS_FILE, "aggregate-metrics.ndjson"),
    (PROCEDURAL_CROSSOVERS_FILE, "crossovers.ndjson"),
    (PROCEDURAL_VALIDATION_FILE, "procedural-validation.json"),
    (PROCEDURAL_PAYLOAD_INDEX_FILE, "source-payload-index.json"),
)
PRIMARY_RUN_DESTINATION = "primary-run.json"
PRIMARY_SUCCESS_DESTINATION = "primary-success.json"
REPEAT_RUN_DESTINATION = "repeat-run.json"
REPEAT_SUCCESS_DESTINATION = "repeat-success.json"

EXPECTED_EXPERIMENTS = (
    "procedural-lidar-y-bias",
    "procedural-camera-noise-correctly-reported",
    "procedural-camera-noise-underreported",
    "procedural-camera-calibration-x",
    "procedural-camera-calibration-yaw",
    "procedural-camera-timestamp-offset",
    "procedural-camera-dropout",
    "procedural-common-mode-x-fov-edge",
)
EXPECTED_PROFILE_IDS = (
    "constant-velocity-front-roi-v1",
    "constant-velocity-fov-edge-v1",
    "constant-velocity-ci-smoke-v1",
)
EXPECTED_COUNTS: dict[str, tuple[int, int, int]] = {
    "procedural-lidar-y-bias": (11_000, 66, 2),
    "procedural-camera-noise-correctly-reported": (5_000, 30, 1),
    "procedural-camera-noise-underreported": (5_000, 30, 1),
    "procedural-camera-calibration-x": (11_000, 66, 2),
    "procedural-camera-calibration-yaw": (11_000, 66, 2),
    "procedural-camera-timestamp-offset": (11_000, 66, 2),
    "procedural-camera-dropout": (14_400, 72, 0),
    "procedural-common-mode-x-fov-edge": (3_300, 33, 0),
}
EXPECTED_TOTAL_SEQUENCE_ROWS = 71_700
EXPECTED_TOTAL_AGGREGATE_ROWS = 429
EXPECTED_TOTAL_CROSSOVER_ROWS = 10
EXPECTED_FUSION_DELTA_ROWS = 54
EXPECTED_FUSION_DELTA_BRANCH_POINTS = 58
EXPECTED_FUSION_DELTA_PAVA_CURVES = 10

CURATION_COMMAND = (
    "uv",
    "run",
    "python",
    "tools/m3_release.py",
    "build-release",
    "examples/matrices/m3-procedural-v1.json",
    "--first-output-dir",
    "<primary-root>",
    "--second-output-dir",
    "<repeat-root>",
    "--evidence-dir",
    "<repeat-evidence>",
    "--official-identity",
    OFFICIAL_IDENTITY_RELATIVE_PATH.as_posix(),
    "--output-dir",
    RELEASE_RELATIVE_PATH.as_posix(),
)
VALIDATION_COMMAND = (
    "uv",
    "run",
    "python",
    "tools/m3_release.py",
    "validate-release",
    RELEASE_RELATIVE_PATH.as_posix(),
    "--official-identity",
    OFFICIAL_IDENTITY_RELATIVE_PATH.as_posix(),
)

_RECORD_MEMBER_NAMES = (
    *(destination for _, destination in RECORD_MEMBER_DESTINATIONS),
    PRIMARY_RUN_DESTINATION,
    PRIMARY_SUCCESS_DESTINATION,
    REPEAT_RUN_DESTINATION,
    REPEAT_SUCCESS_DESTINATION,
)
_MAX_MEMBER_BYTES = 64 * 1024 * 1024
_MAX_TREE_BYTES = 256 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_GENERIC_CPU_MODELS = frozenset(
    {
        "aarch64",
        "amd64",
        "arm",
        "arm64",
        "i386",
        "i686",
        "unknown",
        "unknown-cpu",
        "x86",
        "x86_64",
    }
)
_SAFE_CPU_MODEL = re.compile(r"[A-Za-z0-9 .,_()+@:/-]{1,200}\Z")


class M3CurationError(ValueError):
    """M3 release inputs or curated evidence failed closed."""


@dataclass(frozen=True, slots=True)
class CuratedExperiment:
    """Strictly parsed aggregate-only evidence for one matrix entry."""

    execution_index: int
    experiment: str
    manifest: ProceduralManifest
    profile: ProceduralProfileV1
    aggregates: tuple[AggregateMetricRecordV1Alpha1, ...]
    crossovers: tuple[CrossoverRecordV1Alpha1, ...]
    validation: ProceduralValidationV1
    payload_index: ProceduralPayloadIndexV1Alpha2
    primary_run: RunRecordV1Alpha1
    primary_success: SuccessMarkerV1Alpha1
    repeat_run: RunRecordV1Alpha1
    repeat_success: SuccessMarkerV1Alpha1
    artifact_sha256: str
    primary_run_sha256: str
    repeat_run_sha256: str
    source_sequence_byte_length: int
    source_sequence_sha256: str


@dataclass(frozen=True, slots=True)
class CuratedRelease:
    """Every parsed component needed to regenerate public release bytes."""

    matrix: ExperimentMatrixV1
    matrix_sha256: str
    profiles: tuple[ProceduralProfileV1, ...]
    experiments: tuple[CuratedExperiment, ...]
    matrix_validation: M3MatrixValidationV1
    repeat_verification: RepeatVerificationV1


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative_path(raw: object, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise M3CurationError(f"{label} must be a nonempty normalized POSIX path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or pure.as_posix() != raw:
        raise M3CurationError(f"{label} must be repository-relative")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise M3CurationError(f"{label} contains an unsafe path segment")
    return Path(*pure.parts)


def _reject_symlink_components(path: Path, *, require_exists: bool) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if require_exists:
                raise M3CurationError("release path contains a missing component") from None
            return
        except OSError as error:
            raise M3CurationError("release path components cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise M3CurationError("release paths must not contain symlink components")


def _read_regular(path: Path, *, cap: int = _MAX_MEMBER_BYTES) -> bytes:
    try:
        before = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise M3CurationError("release member is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        fingerprint = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        before_fingerprint = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or fingerprint != before_fingerprint
        ):
            raise M3CurationError("release member must be one stable real regular file")
        if opened.st_size > cap:
            raise M3CurationError("release member exceeds its byte cap")
        chunks: list[bytes] = []
        remaining = cap + 1
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        after = os.fstat(descriptor)
        after_fingerprint = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if len(value) > cap or len(value) != opened.st_size or after_fingerprint != fingerprint:
            raise M3CurationError("release member changed while reading")
        return value
    except OSError as error:
        raise M3CurationError("release member could not be read safely") from error
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, value: bytes) -> None:
    if len(value) > _MAX_MEMBER_BYTES:
        raise M3CurationError("curated member exceeds its byte cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _load_canonical_model[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
) -> tuple[ModelT, bytes]:
    value = _read_regular(path)
    try:
        parsed = model.model_validate_json(value)
    except (ValidationError, ValueError) as error:
        raise M3CurationError("release JSON member violates its strict schema") from error
    if canonical_json_bytes(parsed) != value:
        raise M3CurationError("release JSON member is not canonical")
    return parsed, value


def _load_canonical_mapping_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise M3CurationError(f"{label} is invalid JSON") from error
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        raise M3CurationError(f"{label} must be a JSON object")
    typed = cast(dict[str, Any], parsed)
    if canonical_json_bytes(typed) != value:
        raise M3CurationError(f"{label} is not canonical JSON")
    return typed


def _load_canonical_mapping(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    value = _read_regular(path)
    return _load_canonical_mapping_bytes(value, label=label), value


def _load_manifest(path: Path) -> tuple[ProceduralManifest, bytes]:
    value = _read_regular(path)
    try:
        parsed = EXPERIMENT_MANIFEST_ADAPTER.validate_json(value)
    except (ValidationError, ValueError) as error:
        raise M3CurationError("curated manifest violates its strict schema") from error
    if not isinstance(
        parsed,
        (GeometryCrossoverManifest, AvailabilityControlManifest, CommonModeControlManifest),
    ) or not isinstance(parsed.source, ProceduralSource):
        raise M3CurationError("curated manifest is not an M3 procedural manifest")
    if canonical_json_bytes(parsed) != value:
        raise M3CurationError("curated manifest is not canonical JSON")
    return parsed, value


def _load_profile(path: Path) -> tuple[ProceduralProfileV1, bytes]:
    value = _read_regular(path)
    try:
        parsed = cast(
            ProceduralProfileV1,
            PROCEDURAL_PROFILE_ADAPTER.validate_json(value),
        )
    except (ValidationError, ValueError) as error:
        raise M3CurationError("curated profile violates its strict schema") from error
    if canonical_json_bytes(parsed) != value:
        raise M3CurationError("curated profile is not canonical JSON")
    return parsed, value


def _load_matrix(path: Path) -> tuple[ExperimentMatrixV1, bytes]:
    value = _read_regular(path)
    try:
        parsed = EXPERIMENT_MATRIX_ADAPTER.validate_json(value)
    except (ValidationError, ValueError) as error:
        raise M3CurationError("curated matrix violates its strict schema") from error
    if canonical_json_bytes(parsed) != value:
        raise M3CurationError("curated matrix is not canonical JSON")
    digest = sha256_digest(parsed)
    if parsed.matrix_id != "m3-procedural-v1" or digest != M3_PROCEDURAL_MATRIX_SHA256:
        raise M3CurationError("curation requires the exact frozen M3 release matrix")
    return parsed, value


def _load_canonical_ndjson[ModelT: BaseModel](
    path: Path,
    model: type[ModelT],
    *,
    allow_empty: bool,
) -> tuple[tuple[ModelT, ...], bytes]:
    value = _read_regular(path)
    if not value:
        if allow_empty:
            return (), value
        raise M3CurationError("curated NDJSON must not be empty")
    rows: list[ModelT] = []
    for line in value.splitlines(keepends=True):
        if not line.endswith(b"\n") or len(line) > PROCEDURAL_MAX_RECORD_BYTES:
            raise M3CurationError("curated NDJSON framing or line cap is invalid")
        try:
            parsed = model.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise M3CurationError("curated NDJSON row violates its strict schema") from error
        if canonical_json_bytes(parsed) != line:
            raise M3CurationError("curated NDJSON row is not canonical")
        rows.append(parsed)
    return tuple(rows), value


def _expected_counts(manifest: ProceduralManifest) -> tuple[int, int, int]:
    source = manifest.source
    if not isinstance(source, ProceduralSource):
        raise M3CurationError("M3 curation counts require a procedural source")
    condition_count = len(expected_conditions(manifest))
    if isinstance(manifest, AvailabilityControlManifest):
        sequence_pairs = len(manifest.methods) * len(manifest.evaluation.metrics)
        aggregate_pairs = sequence_pairs
        crossover_count = 0
    elif isinstance(manifest, CommonModeControlManifest):
        sequence_pairs = len(manifest.methods)
        aggregate_pairs = sequence_pairs
        crossover_count = 0
    else:
        sequence_pairs = len(manifest.methods)
        aggregate_pairs = sequence_pairs + 1
        directions = {
            condition.direction
            for condition in expected_conditions(manifest)
            if condition.direction != "identity"
        }
        crossover_count = len(directions)
    return (
        source.sequence_count * condition_count * sequence_pairs,
        condition_count * aggregate_pairs,
        crossover_count,
    )


def _condition_key(record: Any) -> tuple[str, str, int, float, str, str]:
    return (
        record.fault_family,
        record.fault_axis,
        record.severity.index,
        record.severity.magnitude,
        record.severity.direction,
        record.severity.unit,
    )


def _expected_aggregate_pairs(
    manifest: ProceduralManifest,
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


def _validate_curated_rows(
    manifest: ProceduralManifest,
    *,
    run_id: str,
    manifest_sha256: str,
    aggregates: Sequence[AggregateMetricRecordV1Alpha1],
    crossovers: Sequence[CrossoverRecordV1Alpha1],
) -> None:
    source = manifest.source
    if not isinstance(source, ProceduralSource):
        raise M3CurationError("M3 curated rows require a procedural source")
    conditions = expected_conditions(manifest)
    condition_keys = tuple(
        (
            condition.fault_family,
            condition.fault_axis,
            condition.severity_index,
            condition.magnitude,
            condition.direction,
            condition.unit,
        )
        for condition in conditions
    )
    expected_aggregate_keys = tuple(
        (*condition, method, metric)
        for condition in condition_keys
        for method, metric in _expected_aggregate_pairs(manifest)
    )
    actual_aggregate_keys = tuple(
        (*_condition_key(record), record.method_id, record.metric_name) for record in aggregates
    )
    if actual_aggregate_keys != expected_aggregate_keys:
        raise M3CurationError("curated aggregate rows are incomplete or out of order")
    expected_directions = tuple(
        dict.fromkeys(
            condition.direction for condition in conditions if condition.direction != "identity"
        )
    )
    if isinstance(
        manifest,
        (AvailabilityControlManifest, CommonModeControlManifest),
    ):
        expected_directions = ()
    if tuple(record.direction for record in crossovers) != expected_directions:
        raise M3CurationError("curated crossover rows are incomplete or out of order")
    for record in (*aggregates, *crossovers):
        if record.run_id != run_id or record.manifest_sha256 != manifest_sha256:
            raise M3CurationError("curated result provenance is inconsistent")
        if (
            record.sequence_count != source.sequence_count
            or record.bootstrap_replicates != manifest.evaluation.bootstrap.replicates
        ):
            raise M3CurationError("curated result population or inference count changed")


def _specific_cpu_model(value: str) -> bool:
    normalized = value.strip()
    return (
        normalized == value
        and normalized.casefold() not in _GENERIC_CPU_MODELS
        and _SAFE_CPU_MODEL.fullmatch(normalized) is not None
    )


def _validated_public_ci_run_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise M3CurationError("public CI run ID must be a positive JSON integer")
    return value


def _public_ci_attestation(
    value: bytes,
    *,
    scientific_source_revision: str,
) -> dict[str, Any]:
    parsed = _load_canonical_mapping_bytes(value, label="public CI attestation")
    run_id = _validated_public_ci_run_id(parsed.get("run_id"))
    expected = {
        "schema": PUBLIC_CI_ATTESTATION_SCHEMA,
        "provider": "github-actions",
        "repository": PUBLIC_CI_REPOSITORY,
        "workflow": "ci",
        "workflow_path": ".github/workflows/ci.yml",
        "run_id": run_id,
        "url": f"{PUBLIC_CI_URL_PREFIX}{run_id}",
        "source_revision": scientific_source_revision,
        "conclusion": "success",
        "smoke_matrix_sha256": M3_CI_SMOKE_MATRIX_SHA256,
        "release_evidence": False,
        "verification_scope": PUBLIC_CI_VERIFICATION_SCOPE,
    }
    if value != canonical_json_bytes(expected):
        raise M3CurationError("public CI attestation is incomplete or contradictory")
    return expected


def _results_review_attestation(
    value: bytes,
    *,
    artifact_set_sha256: str,
    results_review_report_bytes: bytes,
) -> dict[str, Any]:
    _load_canonical_mapping_bytes(value, label="results-review attestation")
    try:
        report_text = results_review_report_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise M3CurationError("results-review report must be UTF-8 Markdown") from error
    if (
        not report_text.startswith(f"# M3 adversarial results review — {RELEASE_ID}\n")
        or "Verdict: **PASS**" not in report_text
        or f"Artifact set: `{artifact_set_sha256}`" not in report_text
        or not report_text.endswith("\n")
    ):
        raise M3CurationError(
            "results-review report does not record the passing reviewed artifact set"
        )
    expected = {
        "schema": RESULTS_REVIEW_ATTESTATION_SCHEMA,
        "release_id": RELEASE_ID,
        "status": "pass",
        "scope": list(RESULTS_REVIEW_SCOPE),
        "reviewed_artifact_set_sha256": artifact_set_sha256,
        "reviewer": "independent-adversarial-agent",
        "reference": RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH.as_posix(),
        "reference_sha256": _sha256(results_review_report_bytes),
        "reference_byte_length": len(results_review_report_bytes),
        "verification_scope": RESULTS_REVIEW_VERIFICATION_SCOPE,
    }
    if value != canonical_json_bytes(expected):
        raise M3CurationError("results-review attestation is incomplete or contradictory")
    return expected


def _attestation_wrapper(
    *,
    source_path: Path,
    value: bytes,
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "git_bound_source_path": source_path.as_posix(),
        "attestation_sha256": _sha256(value),
        "attestation": dict(attestation),
    }


def _identity_attestations(
    identity: Mapping[str, Any],
    *,
    scientific_source_revision: str,
    artifact_set_sha256: str,
    results_review_report_bytes: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_public_ci = identity.get("public_ci")
    raw_results_review = identity.get("results_review")
    if not isinstance(raw_public_ci, dict) or not isinstance(raw_results_review, dict):
        raise M3CurationError("official identity requires CI and results-review attestations")
    public_ci_wrapper = cast(dict[str, Any], raw_public_ci)
    results_review_wrapper = cast(dict[str, Any], raw_results_review)

    def canonical_attestation(
        wrapper: dict[str, Any],
        *,
        expected_path: Path,
        label: str,
    ) -> bytes:
        if wrapper.get("git_bound_source_path") != expected_path.as_posix():
            raise M3CurationError(f"{label} source path changed")
        raw_attestation = wrapper.get("attestation")
        if not isinstance(raw_attestation, dict):
            raise M3CurationError(f"{label} payload must be an object")
        value = canonical_json_bytes(cast(dict[str, Any], raw_attestation))
        if wrapper.get("attestation_sha256") != _sha256(value):
            raise M3CurationError(f"{label} payload digest changed")
        return value

    public_ci_bytes = canonical_attestation(
        public_ci_wrapper,
        expected_path=PUBLIC_CI_ATTESTATION_RELATIVE_PATH,
        label="public CI attestation",
    )
    review_bytes = canonical_attestation(
        results_review_wrapper,
        expected_path=RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH,
        label="results-review attestation",
    )
    public_ci = _public_ci_attestation(
        public_ci_bytes,
        scientific_source_revision=scientific_source_revision,
    )
    review = _results_review_attestation(
        review_bytes,
        artifact_set_sha256=artifact_set_sha256,
        results_review_report_bytes=results_review_report_bytes,
    )
    return (
        _attestation_wrapper(
            source_path=PUBLIC_CI_ATTESTATION_RELATIVE_PATH,
            value=public_ci_bytes,
            attestation=public_ci,
        ),
        _attestation_wrapper(
            source_path=RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH,
            value=review_bytes,
            attestation=review,
        ),
    )


def _identity_attestation_payloads(
    identity: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    public_ci_wrapper = identity.get("public_ci")
    results_review_wrapper = identity.get("results_review")
    if not isinstance(public_ci_wrapper, dict) or not isinstance(
        results_review_wrapper,
        dict,
    ):
        raise M3CurationError("rendering requires CI and results-review attestations")
    public_ci = public_ci_wrapper.get("attestation")
    results_review = results_review_wrapper.get("attestation")
    if not isinstance(public_ci, dict) or not isinstance(results_review, dict):
        raise M3CurationError("rendering requires attestation object payloads")
    return (
        cast(dict[str, Any], public_ci),
        cast(dict[str, Any], results_review),
    )


def _validated_run_command(run: RunRecordV1Alpha1) -> str:
    command = run.command
    if len(command) != 7 or command[:6] != (
        "ffb",
        "procedural",
        "matrix",
        "run",
        "examples/matrices/m3-procedural-v1.json",
        "--output-dir",
    ):
        raise M3CurationError("M3 release run command is not the frozen logical command")
    output = _safe_relative_path(command[6], label="run output")
    generated_root = Path("reports/generated")
    try:
        output.relative_to(generated_root)
    except ValueError:
        raise M3CurationError("M3 release run output must remain under reports/generated") from None
    if output == generated_root:
        raise M3CurationError("M3 release run output requires a matrix-specific directory")
    return output.as_posix()


def _validate_run_sets(
    first: Sequence[LoadedProceduralArtifact],
    second: Sequence[LoadedProceduralArtifact],
    *,
    repeat: RepeatVerificationV1,
    expected_first_output: str | None,
    expected_second_output: str | None,
) -> tuple[str, str, str, RuntimeEnvironment, str, str]:
    if len(first) != 8 or len(second) != 8:
        raise M3CurationError("M3 release requires exactly sixteen artifact run records")
    first_commands = {_validated_run_command(artifact.run) for artifact in first}
    second_commands = {_validated_run_command(artifact.run) for artifact in second}
    if len(first_commands) != 1 or len(second_commands) != 1:
        raise M3CurationError("run command changed within one matrix execution")
    first_output = next(iter(first_commands))
    second_output = next(iter(second_commands))
    if expected_first_output is not None and first_output != expected_first_output:
        raise M3CurationError("primary run command does not name its actual output root")
    if expected_second_output is not None and second_output != expected_second_output:
        raise M3CurationError("repeat run command does not name its actual output root")
    first_path = Path(first_output)
    second_path = Path(second_output)
    if (
        first_path == second_path
        or first_path in second_path.parents
        or second_path in first_path.parents
    ):
        raise M3CurationError("primary and repeat command outputs must be distinct and non-nested")

    runs = tuple(artifact.run for artifact in (*first, *second))
    revisions = {run.git_revision for run in runs}
    locks = {run.lockfile_sha256 for run in runs}
    packages = {run.package_version for run in runs}
    environments = {canonical_json_bytes(run.environment) for run in runs}
    if len(revisions) != 1 or len(locks) != 1 or len(packages) != 1 or len(environments) != 1:
        raise M3CurationError(
            "all sixteen M3 runs must share revision, lock, package, and environment"
        )
    if any(run.source_dirty or run.status != "succeeded" for run in runs):
        raise M3CurationError("all sixteen M3 runs must be clean and successful")
    environment = runs[0].environment
    if not _specific_cpu_model(environment.cpu_model):
        raise M3CurationError("M3 release requires a specific named CPU model")
    if (
        repeat.first_run.cpu_model != environment.cpu_model
        or repeat.second_run.cpu_model != environment.cpu_model
    ):
        raise M3CurationError("repeat resource evidence names a different CPU")
    if tuple(artifact.run_sha256 for artifact in first) != repeat.first_run.run_record_sha256s:
        raise M3CurationError("primary run-record digests disagree with repeat evidence")
    if tuple(artifact.run_sha256 for artifact in second) != repeat.second_run.run_record_sha256s:
        raise M3CurationError("repeat run-record digests disagree with repeat evidence")
    if any(
        primary.run_sha256 == repeated.run_sha256
        for primary, repeated in zip(first, second, strict=True)
    ):
        raise M3CurationError(
            "primary and repeat volatile run-record digests must differ elementwise"
        )
    return (
        next(iter(revisions)),
        next(iter(locks)),
        next(iter(packages)),
        environment,
        first_output,
        second_output,
    )


def derive_official_identity(
    matrix: LoadedExperimentMatrix,
    first: Sequence[LoadedProceduralArtifact],
    second: Sequence[LoadedProceduralArtifact],
    *,
    matrix_validation: M3MatrixValidationV1,
    repeat_verification: RepeatVerificationV1,
    public_ci_attestation_bytes: bytes,
    results_review_attestation_bytes: bytes,
    results_review_report_bytes: bytes,
    expected_first_output: str | None = None,
    expected_second_output: str | None = None,
) -> dict[str, Any]:
    """Derive the post-run identity that must be reviewed and frozen in Git."""

    if (
        matrix.matrix.matrix_id != "m3-procedural-v1"
        or matrix.matrix_sha256 != M3_PROCEDURAL_MATRIX_SHA256
        or len(first) != 8
        or len(second) != 8
    ):
        raise M3CurationError("official identity requires the exact full M3 release matrix")
    if (
        not matrix_validation.all_checks_passed
        or not repeat_verification.all_checks_passed
        or matrix_validation.matrix_sha256 != matrix.matrix_sha256
        or repeat_verification.matrix_sha256 != matrix.matrix_sha256
        or matrix_validation.artifact_set_sha256
        != repeat_verification.first_run.artifact_set_sha256
    ):
        raise M3CurationError("official identity requires passing cross-linked M3 evidence")
    (
        revision,
        lockfile,
        package,
        environment,
        first_output,
        second_output,
    ) = _validate_run_sets(
        first,
        second,
        repeat=repeat_verification,
        expected_first_output=expected_first_output,
        expected_second_output=expected_second_output,
    )
    entries: list[dict[str, Any]] = []
    for index, (manifest, primary, repeated, evidence) in enumerate(
        zip(
            matrix.manifests,
            first,
            second,
            matrix_validation.ordered_artifacts,
            strict=True,
        )
    ):
        if (
            manifest.experiment != EXPECTED_EXPERIMENTS[index]
            or primary.manifest != manifest
            or repeated.manifest != manifest
            or primary.artifact_sha256 != repeated.artifact_sha256
            or primary.artifact_sha256 != evidence.artifact_sha256
            or primary.run.run_id != repeated.run.run_id
        ):
            raise M3CurationError("official identity artifact order or identity graph changed")
        entries.append(
            {
                "execution_index": index,
                "experiment": manifest.experiment,
                "manifest_sha256": evidence.manifest_sha256,
                "profile_id": evidence.profile_id,
                "profile_sha256": evidence.profile_sha256,
                "artifact_sha256": evidence.artifact_sha256,
                "run_id": primary.run.run_id,
                "primary_run_sha256": primary.run_sha256,
                "repeat_run_sha256": repeated.run_sha256,
            }
        )
    matrix_bytes = canonical_json_bytes(matrix_validation)
    repeat_bytes = canonical_json_bytes(repeat_verification)
    public_ci_attestation = _public_ci_attestation(
        public_ci_attestation_bytes,
        scientific_source_revision=revision,
    )
    results_review_attestation = _results_review_attestation(
        results_review_attestation_bytes,
        artifact_set_sha256=matrix_validation.artifact_set_sha256,
        results_review_report_bytes=results_review_report_bytes,
    )
    return {
        "schema": OFFICIAL_IDENTITY_SCHEMA,
        "release_id": RELEASE_ID,
        "git_bound_source_path": OFFICIAL_IDENTITY_RELATIVE_PATH.as_posix(),
        "matrix_id": matrix.matrix.matrix_id,
        "matrix_sha256": matrix.matrix_sha256,
        "scientific_source_revision": revision,
        "lockfile_sha256": lockfile,
        "package_version": package,
        "environment": environment.model_dump(mode="json", by_alias=True),
        "primary_output_dir": first_output,
        "repeat_output_dir": second_output,
        "artifact_set_sha256": matrix_validation.artifact_set_sha256,
        "ordered_artifacts": entries,
        "matrix_validation_sha256": _sha256(matrix_bytes),
        "repeat_verification_sha256": _sha256(repeat_bytes),
        "first_run_resources": {
            "cpu_model": repeat_verification.first_run.cpu_model,
            "wall_time_seconds": repeat_verification.first_run.wall_time_seconds,
            "peak_memory_bytes": repeat_verification.first_run.peak_memory_bytes,
        },
        "second_run_resources": {
            "cpu_model": repeat_verification.second_run.cpu_model,
            "wall_time_seconds": repeat_verification.second_run.wall_time_seconds,
            "peak_memory_bytes": repeat_verification.second_run.peak_memory_bytes,
        },
        "resource_measurement_scope": repeat_verification.resource_measurement_scope,
        "execution_evidence_scope": repeat_verification.execution_evidence_scope,
        "public_ci": _attestation_wrapper(
            source_path=PUBLIC_CI_ATTESTATION_RELATIVE_PATH,
            value=public_ci_attestation_bytes,
            attestation=public_ci_attestation,
        ),
        "results_review": _attestation_wrapper(
            source_path=RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH,
            value=results_review_attestation_bytes,
            attestation=results_review_attestation,
        ),
    }


def _manifest_path(experiment: str) -> Path:
    return Path("intent/manifests") / f"{experiment}.json"


def _profile_path(profile_id: str) -> Path:
    return Path("intent/profiles") / f"{profile_id}.json"


def _record_path(experiment: str, name: str) -> Path:
    return Path("records") / experiment / name


def _release_allowlist() -> set[Path]:
    return {
        RELEASE_INDEX_PATH,
        MATRIX_PATH,
        SUMMARY_PATH,
        *DOCUMENT_PATHS,
        *FIGURE_PATHS,
        *EVIDENCE_PATHS,
        *(_manifest_path(experiment) for experiment in EXPECTED_EXPERIMENTS),
        *(_profile_path(profile_id) for profile_id in EXPECTED_PROFILE_IDS),
        *(
            _record_path(experiment, member)
            for experiment in EXPECTED_EXPERIMENTS
            for member in _RECORD_MEMBER_NAMES
        ),
    }


def _scan_release(root: Path) -> set[Path]:
    absolute = Path(os.path.abspath(os.fspath(root)))
    _reject_symlink_components(absolute, require_exists=True)
    try:
        root_stat = os.lstat(absolute)
    except OSError as error:
        raise M3CurationError("M3 release directory is unavailable") from error
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise M3CurationError("M3 release root must be a real directory")
    files: set[Path] = set()
    total_bytes = 0
    try:
        for path in absolute.rglob("*"):
            metadata = os.lstat(path)
            relative = path.relative_to(absolute)
            if stat.S_ISLNK(metadata.st_mode):
                raise M3CurationError("M3 release contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise M3CurationError("M3 release contains a non-file member")
            if metadata.st_nlink != 1:
                raise M3CurationError("M3 release contains a hard-linked member")
            if metadata.st_size > _MAX_MEMBER_BYTES:
                raise M3CurationError("M3 release member exceeds its byte cap")
            total_bytes += metadata.st_size
            files.add(relative)
    except M3CurationError:
        raise
    except OSError as error:
        raise M3CurationError("M3 release tree cannot be inspected") from error
    if total_bytes > _MAX_TREE_BYTES:
        raise M3CurationError("M3 release exceeds its tree byte cap")
    if any(path.name == PROCEDURAL_SEQUENCE_METRICS_FILE for path in files):
        raise M3CurationError("raw sequence metrics are forbidden in the public M3 release")
    if files != _release_allowlist():
        raise M3CurationError("M3 release file allowlist mismatch")
    return files


def _validate_payload_member(
    payload_index: ProceduralPayloadIndexV1Alpha2,
    *,
    source_name: str,
    value: bytes,
) -> None:
    entry = next(
        (item for item in payload_index.files if item.path == source_name),
        None,
    )
    if entry is None or entry.byte_length != len(value) or entry.sha256 != _sha256(value):
        raise M3CurationError("curated member disagrees with the source payload index")


def _validate_curated_run(
    run: RunRecordV1Alpha1,
    success: SuccessMarkerV1Alpha1,
    *,
    manifest_sha256: str,
    artifact_sha256: str,
    run_bytes: bytes,
) -> str:
    expected_run_id = derive_run_id(
        manifest_sha256=manifest_sha256,
        git_revision=run.git_revision,
        lockfile_sha256=run.lockfile_sha256,
        package_version=run.package_version,
        artifact_contract=PROCEDURAL_ARTIFACT_CONTRACT,
    )
    run_sha256 = compute_run_record_digest(run_bytes)
    if (
        run.manifest_sha256 != manifest_sha256
        or run.run_id != expected_run_id
        or run.artifact_sha256 != artifact_sha256
        or run.source_dirty
        or run.status != "succeeded"
        or success.artifact_sha256 != artifact_sha256
        or success.run_sha256 != run_sha256
    ):
        raise M3CurationError("curated run and completion identity graph is invalid")
    _validated_run_command(run)
    return run_sha256


def _validate_curated_provenance(
    experiments: Sequence[CuratedExperiment],
    *,
    repeat: RepeatVerificationV1,
) -> tuple[str, str, str, RuntimeEnvironment, str, str]:
    runs = tuple(
        run for experiment in experiments for run in (experiment.primary_run, experiment.repeat_run)
    )
    revisions = {run.git_revision for run in runs}
    locks = {run.lockfile_sha256 for run in runs}
    packages = {run.package_version for run in runs}
    environments = {canonical_json_bytes(run.environment) for run in runs}
    primary_commands = {_validated_run_command(item.primary_run) for item in experiments}
    repeat_commands = {_validated_run_command(item.repeat_run) for item in experiments}
    if (
        len(runs) != 16
        or len(revisions) != 1
        or len(locks) != 1
        or len(packages) != 1
        or len(environments) != 1
        or len(primary_commands) != 1
        or len(repeat_commands) != 1
    ):
        raise M3CurationError("curated sixteen-run provenance is not uniform")
    if any(run.source_dirty or run.status != "succeeded" for run in runs):
        raise M3CurationError("curated runs must all be clean and successful")
    environment = runs[0].environment
    if not _specific_cpu_model(environment.cpu_model):
        raise M3CurationError("curated release requires a specific CPU model")
    primary_output = next(iter(primary_commands))
    repeat_output = next(iter(repeat_commands))
    primary_path = Path(primary_output)
    repeat_path = Path(repeat_output)
    if (
        primary_path == repeat_path
        or primary_path in repeat_path.parents
        or repeat_path in primary_path.parents
    ):
        raise M3CurationError("curated primary and repeat outputs are not disjoint")
    if tuple(item.primary_run_sha256 for item in experiments) != (
        repeat.first_run.run_record_sha256s
    ) or tuple(item.repeat_run_sha256 for item in experiments) != (
        repeat.second_run.run_record_sha256s
    ):
        raise M3CurationError("curated run-record hashes disagree with repeat evidence")
    if any(item.primary_run_sha256 == item.repeat_run_sha256 for item in experiments):
        raise M3CurationError(
            "curated primary and repeat run-record digests must differ elementwise"
        )
    if (
        repeat.first_run.cpu_model != environment.cpu_model
        or repeat.second_run.cpu_model != environment.cpu_model
    ):
        raise M3CurationError("curated resource evidence names a different CPU")
    return (
        next(iter(revisions)),
        next(iter(locks)),
        next(iter(packages)),
        environment,
        primary_output,
        repeat_output,
    )


def _identity_from_curated(
    release: CuratedRelease,
    *,
    public_ci: Mapping[str, Any],
    results_review: Mapping[str, Any],
    results_review_report_bytes: bytes,
) -> dict[str, Any]:
    (
        revision,
        lockfile,
        package,
        environment,
        primary_output,
        repeat_output,
    ) = _validate_curated_provenance(
        release.experiments,
        repeat=release.repeat_verification,
    )
    entries = [
        {
            "execution_index": item.execution_index,
            "experiment": item.experiment,
            "manifest_sha256": sha256_digest(item.manifest),
            "profile_id": item.profile.profile_id,
            "profile_sha256": sha256_digest(item.profile),
            "artifact_sha256": item.artifact_sha256,
            "run_id": item.primary_run.run_id,
            "primary_run_sha256": item.primary_run_sha256,
            "repeat_run_sha256": item.repeat_run_sha256,
        }
        for item in release.experiments
    ]
    matrix_bytes = canonical_json_bytes(release.matrix_validation)
    repeat_bytes = canonical_json_bytes(release.repeat_verification)
    validated_public_ci, validated_results_review = _identity_attestations(
        {
            "public_ci": dict(public_ci),
            "results_review": dict(results_review),
        },
        scientific_source_revision=revision,
        artifact_set_sha256=release.matrix_validation.artifact_set_sha256,
        results_review_report_bytes=results_review_report_bytes,
    )
    return {
        "schema": OFFICIAL_IDENTITY_SCHEMA,
        "release_id": RELEASE_ID,
        "git_bound_source_path": OFFICIAL_IDENTITY_RELATIVE_PATH.as_posix(),
        "matrix_id": release.matrix.matrix_id,
        "matrix_sha256": release.matrix_sha256,
        "scientific_source_revision": revision,
        "lockfile_sha256": lockfile,
        "package_version": package,
        "environment": environment.model_dump(mode="json", by_alias=True),
        "primary_output_dir": primary_output,
        "repeat_output_dir": repeat_output,
        "artifact_set_sha256": release.matrix_validation.artifact_set_sha256,
        "ordered_artifacts": entries,
        "matrix_validation_sha256": _sha256(matrix_bytes),
        "repeat_verification_sha256": _sha256(repeat_bytes),
        "first_run_resources": {
            "cpu_model": release.repeat_verification.first_run.cpu_model,
            "wall_time_seconds": release.repeat_verification.first_run.wall_time_seconds,
            "peak_memory_bytes": release.repeat_verification.first_run.peak_memory_bytes,
        },
        "second_run_resources": {
            "cpu_model": release.repeat_verification.second_run.cpu_model,
            "wall_time_seconds": release.repeat_verification.second_run.wall_time_seconds,
            "peak_memory_bytes": release.repeat_verification.second_run.peak_memory_bytes,
        },
        "resource_measurement_scope": (release.repeat_verification.resource_measurement_scope),
        "execution_evidence_scope": release.repeat_verification.execution_evidence_scope,
        "public_ci": validated_public_ci,
        "results_review": validated_results_review,
    }


def _load_curated_release(root: Path) -> CuratedRelease:
    matrix, _ = _load_matrix(root / MATRIX_PATH)
    matrix_sha256 = sha256_digest(matrix)
    profiles: list[ProceduralProfileV1] = []
    profile_bytes_by_id: dict[str, bytes] = {}
    profile_by_id: dict[str, ProceduralProfileV1] = {}
    if len(matrix.profiles) != len(EXPECTED_PROFILE_IDS):
        raise M3CurationError("curated matrix profile count changed")
    for expected_id, entry in zip(EXPECTED_PROFILE_IDS, matrix.profiles, strict=True):
        profile, profile_bytes = _load_profile(root / _profile_path(expected_id))
        if profile.profile_id != expected_id or sha256_digest(profile) != entry.profile_sha256:
            raise M3CurationError("curated profile disagrees with the frozen matrix")
        profiles.append(profile)
        profile_by_id[profile.profile_id] = profile
        profile_bytes_by_id[profile.profile_id] = profile_bytes

    matrix_validation, _ = _load_canonical_model(
        root / MATRIX_VALIDATION_PATH,
        M3MatrixValidationV1,
    )
    repeat_verification, _ = _load_canonical_model(
        root / REPEAT_VERIFICATION_PATH,
        RepeatVerificationV1,
    )
    if (
        matrix_validation.matrix_id != "m3-procedural-v1"
        or repeat_verification.matrix_id != "m3-procedural-v1"
        or matrix_validation.matrix_sha256 != matrix_sha256
        or repeat_verification.matrix_sha256 != matrix_sha256
        or matrix_validation.artifact_set_sha256
        != repeat_verification.first_run.artifact_set_sha256
        or not matrix_validation.all_checks_passed
        or not repeat_verification.all_checks_passed
        or repeat_verification.comparison_count != 48
        or repeat_verification.mismatch_count != 0
    ):
        raise M3CurationError("curated M3 matrix/repeat evidence is not release eligible")

    pairs = {
        (pair.execution_index, pair.path): pair for pair in repeat_verification.indexed_member_pairs
    }
    if len(pairs) != 48:
        raise M3CurationError("curated repeat member evidence is incomplete")

    experiments: list[CuratedExperiment] = []
    total_sequence_rows = 0
    total_aggregate_rows = 0
    total_crossover_rows = 0
    for index, (expected_experiment, matrix_entry, evidence_entry) in enumerate(
        zip(
            EXPECTED_EXPERIMENTS,
            matrix.execution_order,
            matrix_validation.ordered_artifacts,
            strict=True,
        )
    ):
        manifest, manifest_bytes = _load_manifest(root / _manifest_path(expected_experiment))
        if (
            manifest.experiment != expected_experiment
            or sha256_digest(manifest) != matrix_entry.manifest_sha256
            or evidence_entry.execution_index != index
            or evidence_entry.experiment != expected_experiment
            or evidence_entry.manifest_sha256 != matrix_entry.manifest_sha256
        ):
            raise M3CurationError("curated manifest order or digest changed")
        source = manifest.source
        if not isinstance(source, ProceduralSource):
            raise M3CurationError("curated manifest source is not procedural")
        profile = profile_by_id.get(source.profile_id)
        profile_bytes = profile_bytes_by_id.get(source.profile_id)
        if (
            profile is None
            or profile_bytes is None
            or sha256_digest(profile) != source.profile_sha256
            or evidence_entry.profile_id != profile.profile_id
            or evidence_entry.profile_sha256 != source.profile_sha256
        ):
            raise M3CurationError("curated manifest/profile identity changed")

        record_root = root / "records" / expected_experiment
        aggregates, aggregate_bytes = _load_canonical_ndjson(
            record_root / "aggregate-metrics.ndjson",
            AggregateMetricRecordV1Alpha1,
            allow_empty=False,
        )
        crossovers, crossover_bytes = _load_canonical_ndjson(
            record_root / "crossovers.ndjson",
            CrossoverRecordV1Alpha1,
            allow_empty=isinstance(
                manifest,
                (AvailabilityControlManifest, CommonModeControlManifest),
            ),
        )
        validation, validation_bytes = _load_canonical_model(
            record_root / "procedural-validation.json",
            ProceduralValidationV1,
        )
        payload_index, payload_index_bytes = _load_canonical_model(
            record_root / "source-payload-index.json",
            ProceduralPayloadIndexV1Alpha2,
        )
        primary_run, primary_run_bytes = _load_canonical_model(
            record_root / PRIMARY_RUN_DESTINATION,
            RunRecordV1Alpha1,
        )
        primary_success, _ = _load_canonical_model(
            record_root / PRIMARY_SUCCESS_DESTINATION,
            SuccessMarkerV1Alpha1,
        )
        repeat_run, repeat_run_bytes = _load_canonical_model(
            record_root / REPEAT_RUN_DESTINATION,
            RunRecordV1Alpha1,
        )
        repeat_success, _ = _load_canonical_model(
            record_root / REPEAT_SUCCESS_DESTINATION,
            SuccessMarkerV1Alpha1,
        )

        if (
            payload_index.run_id != primary_run.run_id
            or payload_index.manifest_sha256 != matrix_entry.manifest_sha256
            or payload_index.profile_sha256 != source.profile_sha256
            or primary_run.run_id != repeat_run.run_id
        ):
            raise M3CurationError("curated payload/run identity graph changed")
        source_values = {
            PROCEDURAL_MANIFEST_FILE: manifest_bytes,
            PROCEDURAL_PROFILE_FILE: profile_bytes,
            PROCEDURAL_AGGREGATE_METRICS_FILE: aggregate_bytes,
            PROCEDURAL_CROSSOVERS_FILE: crossover_bytes,
            PROCEDURAL_VALIDATION_FILE: validation_bytes,
        }
        for source_name, value in source_values.items():
            _validate_payload_member(
                payload_index,
                source_name=source_name,
                value=value,
            )
        sequence_entry = next(
            item for item in payload_index.files if item.path == PROCEDURAL_SEQUENCE_METRICS_FILE
        )
        artifact_sha256 = compute_procedural_artifact_digest(payload_index_bytes)
        primary_run_sha256 = _validate_curated_run(
            primary_run,
            primary_success,
            manifest_sha256=matrix_entry.manifest_sha256,
            artifact_sha256=artifact_sha256,
            run_bytes=primary_run_bytes,
        )
        repeat_run_sha256 = _validate_curated_run(
            repeat_run,
            repeat_success,
            manifest_sha256=matrix_entry.manifest_sha256,
            artifact_sha256=artifact_sha256,
            run_bytes=repeat_run_bytes,
        )
        if artifact_sha256 != evidence_entry.artifact_sha256:
            raise M3CurationError("curated artifact digest disagrees with matrix evidence")
        for source_name in PROCEDURAL_INDEXED_PAYLOAD_PATHS:
            pair = pairs.get((index, source_name))
            source_entry = next(item for item in payload_index.files if item.path == source_name)
            if (
                pair is None
                or pair.experiment != expected_experiment
                or pair.manifest_sha256 != matrix_entry.manifest_sha256
                or pair.first_sha256 != source_entry.sha256
                or pair.second_sha256 != source_entry.sha256
                or not pair.equal
            ):
                raise M3CurationError("curated member disagrees with repeat evidence")

        expected_count = _expected_counts(manifest)
        if (
            expected_count != EXPECTED_COUNTS[expected_experiment]
            or len(aggregates) != expected_count[1]
            or len(crossovers) != expected_count[2]
            or validation.resources.implied_sequence_row_count != expected_count[0]
            or validation.run_id != primary_run.run_id
            or validation.manifest_sha256 != matrix_entry.manifest_sha256
            or validation.profile_id != profile.profile_id
            or validation.profile_sha256 != source.profile_sha256
            or not validation.all_checks_passed
        ):
            raise M3CurationError("curated row counts or validation links changed")
        _validate_curated_rows(
            manifest,
            run_id=primary_run.run_id,
            manifest_sha256=matrix_entry.manifest_sha256,
            aggregates=aggregates,
            crossovers=crossovers,
        )
        total_sequence_rows += expected_count[0]
        total_aggregate_rows += len(aggregates)
        total_crossover_rows += len(crossovers)
        experiments.append(
            CuratedExperiment(
                execution_index=index,
                experiment=expected_experiment,
                manifest=manifest,
                profile=profile,
                aggregates=aggregates,
                crossovers=crossovers,
                validation=validation,
                payload_index=payload_index,
                primary_run=primary_run,
                primary_success=primary_success,
                repeat_run=repeat_run,
                repeat_success=repeat_success,
                artifact_sha256=artifact_sha256,
                primary_run_sha256=primary_run_sha256,
                repeat_run_sha256=repeat_run_sha256,
                source_sequence_byte_length=sequence_entry.byte_length,
                source_sequence_sha256=sequence_entry.sha256,
            )
        )

    if (
        total_sequence_rows != EXPECTED_TOTAL_SEQUENCE_ROWS
        or total_aggregate_rows != EXPECTED_TOTAL_AGGREGATE_ROWS
        or total_crossover_rows != EXPECTED_TOTAL_CROSSOVER_ROWS
    ):
        raise M3CurationError("curated release literal completeness totals changed")
    release = CuratedRelease(
        matrix=matrix,
        matrix_sha256=matrix_sha256,
        profiles=tuple(profiles),
        experiments=tuple(experiments),
        matrix_validation=matrix_validation,
        repeat_verification=repeat_verification,
    )
    _validate_curated_provenance(
        release.experiments,
        repeat=release.repeat_verification,
    )
    return release


def _validation_summary(validation: ProceduralValidationV1) -> dict[str, Any]:
    expected_loss_max = max(
        (check.absolute_standardized_error for check in validation.expected_loss_checks),
        default=None,
    )
    oracle_checks = (
        validation.oracle_checks.identity_center,
        validation.oracle_checks.calibration_translation_center,
        validation.oracle_checks.translation_bias_equivalence_center,
        validation.oracle_checks.translation_bias_equivalence_sequence_loss,
        validation.oracle_checks.calibration_yaw_center,
        validation.oracle_checks.timestamp_alignment_center,
        validation.oracle_checks.static_timestamp_center,
    )
    oracle_maximum_by_unit: dict[str, float] = {}
    for check in oracle_checks:
        oracle_maximum_by_unit[check.unit] = max(
            oracle_maximum_by_unit.get(check.unit, 0.0),
            check.maximum_absolute_discrepancy,
        )
    return {
        "all_checks_passed": validation.all_checks_passed,
        "profile_checks_passed": validation.profile_checks.all_checks_passed,
        "eligibility_invariant": validation.eligibility.eligibility_invariant,
        "oracle_checks_passed": validation.oracle_checks.all_checks_passed,
        "oracle_check_count": len(oracle_checks),
        "maximum_oracle_discrepancy_by_unit": {
            unit: oracle_maximum_by_unit[unit] for unit in sorted(oracle_maximum_by_unit)
        },
        "moment_check_count": len(validation.moment_checks),
        "moment_checks_passed": all(check.passed for check in validation.moment_checks),
        "expected_loss_check_count": len(validation.expected_loss_checks),
        "expected_loss_checks_passed": all(
            check.passed for check in validation.expected_loss_checks
        ),
        "maximum_expected_loss_standardized_error": expected_loss_max,
        "dropout_validation": validation.dropout_validation.model_dump(
            mode="json",
            by_alias=True,
        ),
        "identity_comparison": validation.identity_comparison.model_dump(
            mode="json",
            by_alias=True,
        ),
        "common_mode_validation": validation.common_mode_validation.model_dump(
            mode="json",
            by_alias=True,
        ),
        "deterministic_model_checks": (
            validation.deterministic_model_checks.model_dump(
                mode="json",
                by_alias=True,
            )
        ),
        "resources": validation.resources.model_dump(mode="json", by_alias=True),
    }


def build_release_summary(release: CuratedRelease) -> dict[str, Any]:
    """Build the fixed all-row, aggregate-only machine-readable M3 summary."""

    experiments: list[dict[str, Any]] = []
    for item in release.experiments:
        manifest_sha256 = sha256_digest(item.manifest)
        aggregate_entry = next(
            entry
            for entry in item.payload_index.files
            if entry.path == PROCEDURAL_AGGREGATE_METRICS_FILE
        )
        crossover_entry = next(
            entry for entry in item.payload_index.files if entry.path == PROCEDURAL_CROSSOVERS_FILE
        )
        experiments.append(
            {
                "execution_index": item.execution_index,
                "experiment": item.experiment,
                "manifest_sha256": manifest_sha256,
                "profile_id": item.profile.profile_id,
                "profile_sha256": sha256_digest(item.profile),
                "fault_family": item.manifest.fault_sweep.kind,
                "fault_axis": item.manifest.fault_sweep.axis,
                "severity_unit": item.manifest.fault_sweep.unit,
                "source_sequence_metrics": {
                    "included": False,
                    "source_name": PROCEDURAL_SEQUENCE_METRICS_FILE,
                    "record_count": EXPECTED_COUNTS[item.experiment][0],
                    "byte_length": item.source_sequence_byte_length,
                    "sha256": item.source_sequence_sha256,
                    "omission_reason": (
                        "raw-generated-sequence-rows-retained-locally-and-"
                        "reproducible-from-frozen-intent"
                    ),
                },
                "aggregate_metrics": {
                    "included": True,
                    "source_name": PROCEDURAL_AGGREGATE_METRICS_FILE,
                    "record_count": len(item.aggregates),
                    "byte_length": aggregate_entry.byte_length,
                    "sha256": aggregate_entry.sha256,
                    "selection": "all-source-rows-in-source-order",
                },
                "crossovers": {
                    "included": True,
                    "source_name": PROCEDURAL_CROSSOVERS_FILE,
                    "record_count": len(item.crossovers),
                    "byte_length": crossover_entry.byte_length,
                    "sha256": crossover_entry.sha256,
                    "selection": "all-source-rows-in-source-order",
                    "records": [
                        record.model_dump(mode="json", by_alias=True) for record in item.crossovers
                    ],
                },
                "validation": _validation_summary(item.validation),
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "release_id": RELEASE_ID,
        "matrix_id": release.matrix.matrix_id,
        "matrix_sha256": release.matrix_sha256,
        "result_selection": release.matrix.result_selection,
        "selection_policy": {
            "matrix_entries": "all-eight-in-frozen-execution-order",
            "aggregate_rows": "all-429-source-rows-without-result-predicate",
            "crossover_rows": "all-10-source-rows-without-result-predicate",
            "omitted_scientific_member": PROCEDURAL_SEQUENCE_METRICS_FILE,
            "favorable_result_selection": False,
            "physical_severity_pooling": False,
        },
        "totals": {
            "experiment_count": len(release.experiments),
            "omitted_sequence_metric_record_count": EXPECTED_TOTAL_SEQUENCE_ROWS,
            "curated_aggregate_metric_record_count": EXPECTED_TOTAL_AGGREGATE_ROWS,
            "curated_crossover_record_count": EXPECTED_TOTAL_CROSSOVER_ROWS,
        },
        "experiments": experiments,
        "matrix_validation": {
            "artifact_set_sha256": release.matrix_validation.artifact_set_sha256,
            "identity_comparison": release.matrix_validation.identity_comparison.model_dump(
                mode="json",
                by_alias=True,
            ),
            "all_checks_passed": release.matrix_validation.all_checks_passed,
        },
        "repeat_verification": {
            "comparison_count": release.repeat_verification.comparison_count,
            "mismatch_count": release.repeat_verification.mismatch_count,
            "first_run": release.repeat_verification.first_run.model_dump(
                mode="json",
                by_alias=True,
            ),
            "second_run": release.repeat_verification.second_run.model_dump(
                mode="json",
                by_alias=True,
            ),
            "resource_measurement_scope": (release.repeat_verification.resource_measurement_scope),
            "execution_evidence_scope": (release.repeat_verification.execution_evidence_scope),
            "all_checks_passed": release.repeat_verification.all_checks_passed,
        },
    }


def _float_text(value: float | None) -> str:
    if value is None:
        return "undefined"
    return format(value, ".6g")


def _exact_float_text(value: float | None) -> str:
    if value is None:
        return "undefined"
    return repr(value)


def _svg_header(*, width: int, height: int, title: str, description: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        ),
        f"<title>{html.escape(title)}</title>",
        f"<desc>{html.escape(description)}</desc>",
        "<style>",
        (
            "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "fill:#182230}.muted{fill:#5b677a}.axis{stroke:#8391a5;stroke-width:1}"
            ".zero{stroke:#9d3346;stroke-width:1;stroke-dasharray:5 4}"
            ".interval{stroke-width:1.3}.fit{fill:none;stroke-width:2.4}"
            ".raw{stroke:#fff;stroke-width:1}.panel{fill:#fbfcfe;stroke:#dce3ec}"
        ),
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="48" y="42" font-size="25" font-weight="700">{html.escape(title)}</text>',
    ]


def _linear_scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if not math.isfinite(value) or not math.isfinite(low) or not math.isfinite(high):
        raise M3CurationError("figure inputs must be finite")
    if high <= low:
        return (start + end) / 2.0
    return start + (value - low) / (high - low) * (end - start)


def render_fusion_delta_figure(release: CuratedRelease) -> bytes:
    """Render every signed fusion-delta row and its preregistered PAVA branch."""

    panels: list[
        tuple[
            CuratedExperiment,
            list[
                tuple[
                    str,
                    tuple[AggregateMetricRecordV1Alpha1, ...],
                    tuple[float, ...],
                ]
            ],
        ]
    ] = []
    raw_row_count = 0
    branch_point_count = 0
    curve_count = 0
    for item in release.experiments[:6]:
        rows = tuple(
            record for record in item.aggregates if record.metric_name == "fused-minus-healthy"
        )
        raw_row_count += len(rows)
        identity = tuple(record for record in rows if record.severity.direction == "identity")
        if len(identity) != 1:
            raise M3CurationError("fusion-delta panel requires one identity row")
        directions = tuple(
            dict.fromkeys(
                record.severity.direction
                for record in rows
                if record.severity.direction != "identity"
            )
        )
        series: list[
            tuple[
                str,
                tuple[AggregateMetricRecordV1Alpha1, ...],
                tuple[float, ...],
            ]
        ] = []
        for direction in directions:
            branch = (
                *identity,
                *(record for record in rows if record.severity.direction == direction),
            )
            if any(
                record.status != "ok"
                or record.estimate is None
                or record.interval_lower is None
                or record.interval_upper is None
                for record in branch
            ):
                raise M3CurationError("fusion-delta rows must be fully defined")
            fitted_array = pava_non_decreasing([cast(float, record.estimate) for record in branch])
            fitted = tuple(float(value) for value in fitted_array)
            series.append((direction, branch, fitted))
            branch_point_count += len(branch)
            curve_count += 1
        panels.append((item, series))
    if (
        raw_row_count != EXPECTED_FUSION_DELTA_ROWS
        or branch_point_count != EXPECTED_FUSION_DELTA_BRANCH_POINTS
        or curve_count != EXPECTED_FUSION_DELTA_PAVA_CURVES
    ):
        raise M3CurationError("fusion-delta figure completeness counts changed")

    width = 1400
    height = 1080
    lines = _svg_header(
        width=width,
        height=height,
        title="M3 signed fusion delta under six single-sensor proxy faults",
        description=(
            "All 54 unique fused-minus-healthy aggregate rows, 58 direction-branch "
            "points including shared identities, pointwise 95 percent intervals, and "
            "10 equal-severity PAVA curves. Each panel retains its native physical unit."
        ),
    )
    lines.extend(
        [
            '<text class="muted" x="48" y="70" font-size="14">'
            "Signed estimate: fixed-fusion matched-center MSE minus designated healthy "
            "modality MSE. Positive is harmful within this benchmark.</text>",
            '<text class="muted" x="48" y="91" font-size="13">'
            "Intervals are pointwise paired-sequence bootstrap intervals; they are not "
            "simultaneous family-wise intervals.</text>",
        ]
    )
    colors = {"negative": "#3366a8", "positive": "#d06b2d", "increase": "#21856f"}
    panel_width = 642
    panel_height = 285
    for panel_index, (item, series) in enumerate(panels):
        column = panel_index % 2
        row = panel_index // 2
        x0 = 48 + column * 682
        y0 = 116 + row * 310
        plot_left = x0 + 72
        plot_right = x0 + panel_width - 26
        plot_top = y0 + 55
        plot_bottom = y0 + panel_height - 52
        all_records = tuple(record for _, branch, _ in series for record in branch)
        all_fitted = tuple(value for _, _, fitted in series for value in fitted)
        x_values = [record.severity.magnitude for record in all_records]
        y_values = (
            [
                value
                for record in all_records
                for value in (
                    cast(float, record.interval_lower),
                    cast(float, record.estimate),
                    cast(float, record.interval_upper),
                )
            ]
            + list(all_fitted)
            + [0.0]
        )
        x_low, x_high = min(x_values), max(x_values)
        y_low, y_high = min(y_values), max(y_values)
        pad = max((y_high - y_low) * 0.08, 1e-9)
        y_low -= pad
        y_high += pad
        title = item.experiment.removeprefix("procedural-").replace("-", " ")
        unit = item.manifest.fault_sweep.unit
        lines.extend(
            [
                (
                    f'<rect class="panel" x="{x0}" y="{y0}" width="{panel_width}" '
                    f'height="{panel_height}" rx="8"/>'
                ),
                (
                    f'<text x="{x0 + 18}" y="{y0 + 27}" font-size="16" '
                    f'font-weight="650">{html.escape(title)}</text>'
                ),
                (
                    f'<text class="muted" x="{x0 + 18}" y="{y0 + 46}" '
                    f'font-size="12">severity ({html.escape(unit)}), separate '
                    "direction branches</text>"
                ),
                (
                    f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" '
                    f'x2="{plot_right}" y2="{plot_bottom}"/>'
                ),
                (
                    f'<line class="axis" x1="{plot_left}" y1="{plot_top}" '
                    f'x2="{plot_left}" y2="{plot_bottom}"/>'
                ),
            ]
        )
        zero_y = _linear_scale(0.0, y_low, y_high, plot_bottom, plot_top)
        lines.append(
            f'<line class="zero" x1="{plot_left}" y1="{zero_y:.3f}" '
            f'x2="{plot_right}" y2="{zero_y:.3f}"/>'
        )
        for series_index, (direction, branch, fitted) in enumerate(series):
            color = colors[direction]
            fitted_points: list[str] = []
            for record, fitted_value in zip(branch, fitted, strict=True):
                x = _linear_scale(
                    record.severity.magnitude,
                    x_low,
                    x_high,
                    plot_left,
                    plot_right,
                )
                estimate = cast(float, record.estimate)
                lower = cast(float, record.interval_lower)
                upper = cast(float, record.interval_upper)
                y = _linear_scale(estimate, y_low, y_high, plot_bottom, plot_top)
                y_lower = _linear_scale(lower, y_low, y_high, plot_bottom, plot_top)
                y_upper = _linear_scale(upper, y_low, y_high, plot_bottom, plot_top)
                fitted_y = _linear_scale(
                    fitted_value,
                    y_low,
                    y_high,
                    plot_bottom,
                    plot_top,
                )
                fitted_points.append(f"{x:.3f},{fitted_y:.3f}")
                lines.extend(
                    [
                        (
                            f'<line class="interval" stroke="{color}" x1="{x:.3f}" '
                            f'y1="{y_lower:.3f}" x2="{x:.3f}" y2="{y_upper:.3f}"/>'
                        ),
                        (f'<circle class="raw" fill="{color}" cx="{x:.3f}" cy="{y:.3f}" r="4"/>'),
                    ]
                )
            lines.append(
                f'<polyline class="fit" stroke="{color}" points="{" ".join(fitted_points)}"/>'
            )
            legend_x = x0 + panel_width - 155
            legend_y = y0 + 24 + series_index * 18
            lines.extend(
                [
                    (
                        f'<line stroke="{color}" stroke-width="2.4" x1="{legend_x}" '
                        f'y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}"/>'
                    ),
                    (
                        f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-size="11">'
                        f"{html.escape(direction)}</text>"
                    ),
                ]
            )
        lines.extend(
            [
                (
                    f'<text class="muted" x="{plot_left}" y="{plot_bottom + 20}" '
                    f'font-size="11">{_float_text(x_low)}</text>'
                ),
                (
                    f'<text class="muted" x="{plot_right}" y="{plot_bottom + 20}" '
                    f'font-size="11" text-anchor="end">{_float_text(x_high)} '
                    f"{html.escape(unit)}</text>"
                ),
                (
                    f'<text class="muted" x="{plot_left - 8}" y="{plot_top + 4}" '
                    f'font-size="10" text-anchor="end">{_float_text(y_high)}</text>'
                ),
                (
                    f'<text class="muted" x="{plot_left - 8}" y="{plot_bottom}" '
                    f'font-size="10" text-anchor="end">{_float_text(y_low)} m²</text>'
                ),
            ]
        )
    lines.extend(
        [
            '<text class="muted" x="48" y="1050" font-size="12">'
            "Raw points and intervals are never replaced by the fitted PAVA curves. "
            "Meters, radians, seconds, and std-scales are not pooled.</text>",
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_dropout_figure(release: CuratedRelease) -> bytes:
    """Render all dropout coverage, undefined-rate, and conditional-loss rows."""

    item = release.experiments[6]
    if not isinstance(item.manifest, AvailabilityControlManifest) or len(item.aggregates) != 72:
        raise M3CurationError("dropout figure requires the exact 72-row control")
    metrics = (
        ("coverage", "Coverage", "fraction"),
        ("undefined-output-rate", "Undefined-output rate", "fraction"),
        (
            "conditional-matched-center-mse",
            "Conditional matched-center MSE",
            "m²",
        ),
    )
    colors = {
        "camera-only": "#3c6ca8",
        "lidar-only": "#7a5fa6",
        "fixed-fusion": "#d06b2d",
        "fault-target-drop-policy": "#21856f",
    }
    width = 1400
    height = 1030
    lines = _svg_header(
        width=width,
        height=height,
        title="M3 camera-dropout availability control",
        description=(
            "All 72 aggregate rows: every probability and method for coverage, "
            "undefined-output rate, then conditional matched-center loss. Undefined "
            "conditional losses are marked and never imputed as zero."
        ),
    )
    lines.extend(
        [
            '<text class="muted" x="48" y="70" font-size="14">'
            "Availability metrics are shown before conditional localization loss. "
            "Pointwise intervals use paired complete-sequence bootstrap resampling.</text>",
        ]
    )
    plot_left = 180
    plot_right = 1320
    panel_height = 245
    for metric_index, (metric_name, label, unit) in enumerate(metrics):
        panel_y = 102 + metric_index * 285
        plot_top = panel_y + 48
        plot_bottom = panel_y + panel_height - 38
        rows = tuple(record for record in item.aggregates if record.metric_name == metric_name)
        if len(rows) != 24:
            raise M3CurationError("dropout metric panel is missing a method/probability row")
        defined_values = [
            value
            for record in rows
            if record.status == "ok"
            for value in (
                cast(float, record.interval_lower),
                cast(float, record.estimate),
                cast(float, record.interval_upper),
            )
        ]
        if metric_name in {"coverage", "undefined-output-rate"}:
            y_low, y_high = 0.0, 1.0
        else:
            y_low = min([0.0, *defined_values])
            y_high = max(defined_values)
            y_high += max((y_high - y_low) * 0.08, 1e-9)
        lines.extend(
            [
                (
                    f'<rect class="panel" x="48" y="{panel_y}" width="1304" '
                    f'height="{panel_height}" rx="8"/>'
                ),
                (
                    f'<text x="68" y="{panel_y + 29}" font-size="17" '
                    f'font-weight="650">{html.escape(label)} ({html.escape(unit)})</text>'
                ),
                (
                    f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" '
                    f'x2="{plot_right}" y2="{plot_bottom}"/>'
                ),
                (
                    f'<line class="axis" x1="{plot_left}" y1="{plot_top}" '
                    f'x2="{plot_left}" y2="{plot_bottom}"/>'
                ),
            ]
        )
        for method_index, method in enumerate(item.manifest.methods):
            method_rows = tuple(record for record in rows if record.method_id == method)
            if len(method_rows) != 6:
                raise M3CurationError("dropout method series is incomplete")
            color = colors[method]
            points: list[str] = []
            for record in method_rows:
                x = _linear_scale(
                    record.severity.magnitude,
                    0.0,
                    1.0,
                    plot_left,
                    plot_right,
                )
                if record.status == "undefined":
                    y = plot_top + 11 + method_index * 8
                    lines.extend(
                        [
                            (
                                f'<line stroke="{color}" stroke-width="2" '
                                f'x1="{x - 4:.3f}" y1="{y - 4:.3f}" '
                                f'x2="{x + 4:.3f}" y2="{y + 4:.3f}"/>'
                            ),
                            (
                                f'<line stroke="{color}" stroke-width="2" '
                                f'x1="{x - 4:.3f}" y1="{y + 4:.3f}" '
                                f'x2="{x + 4:.3f}" y2="{y - 4:.3f}"/>'
                            ),
                        ]
                    )
                    continue
                estimate = cast(float, record.estimate)
                lower = cast(float, record.interval_lower)
                upper = cast(float, record.interval_upper)
                y = _linear_scale(estimate, y_low, y_high, plot_bottom, plot_top)
                y_lower = _linear_scale(lower, y_low, y_high, plot_bottom, plot_top)
                y_upper = _linear_scale(upper, y_low, y_high, plot_bottom, plot_top)
                points.append(f"{x:.3f},{y:.3f}")
                lines.extend(
                    [
                        (
                            f'<line class="interval" stroke="{color}" x1="{x:.3f}" '
                            f'y1="{y_lower:.3f}" x2="{x:.3f}" y2="{y_upper:.3f}"/>'
                        ),
                        (f'<circle class="raw" fill="{color}" cx="{x:.3f}" cy="{y:.3f}" r="4"/>'),
                    ]
                )
            if points:
                lines.append(
                    f'<polyline class="fit" stroke="{color}" points="{" ".join(points)}"/>'
                )
            legend_x = 520 + method_index * 205
            lines.extend(
                [
                    (
                        f'<line stroke="{color}" stroke-width="2.4" x1="{legend_x}" '
                        f'y1="{panel_y + 24}" x2="{legend_x + 20}" y2="{panel_y + 24}"/>'
                    ),
                    (
                        f'<text x="{legend_x + 26}" y="{panel_y + 28}" font-size="11">'
                        f"{html.escape(method)}</text>"
                    ),
                ]
            )
        lines.extend(
            [
                (
                    f'<text class="muted" x="{plot_left}" y="{plot_bottom + 20}" '
                    'font-size="11">0</text>'
                ),
                (
                    f'<text class="muted" x="{plot_right}" y="{plot_bottom + 20}" '
                    'font-size="11" text-anchor="end">1 dropout probability</text>'
                ),
                (
                    f'<text class="muted" x="{plot_left - 12}" y="{plot_top + 4}" '
                    f'font-size="10" text-anchor="end">{_float_text(y_high)}</text>'
                ),
                (
                    f'<text class="muted" x="{plot_left - 12}" y="{plot_bottom}" '
                    f'font-size="10" text-anchor="end">{_float_text(y_low)}</text>'
                ),
            ]
        )
    lines.extend(
        [
            '<text class="muted" x="48" y="978" font-size="12">'
            "An &#215; marks an undefined conditional-loss aggregate. Missing output "
            "receives no zero localization penalty; coverage and undefined rate retain "
            "the denominator.</text>",
            '<text class="muted" x="48" y="999" font-size="12">'
            "Dropout is an availability control and has no crossover estimand.</text>",
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_common_mode_figure(release: CuratedRelease) -> bytes:
    """Render all common-mode absolute losses and its disagreement blind spot."""

    item = release.experiments[7]
    common_validation = item.validation.common_mode_validation
    if (
        not isinstance(item.manifest, CommonModeControlManifest)
        or len(item.aggregates) != 33
        or common_validation.status != "applicable"
    ):
        raise M3CurationError("common-mode figure requires the exact 33-row control")
    colors = {
        "camera-only": "#3c6ca8",
        "lidar-only": "#7a5fa6",
        "fixed-fusion": "#d06b2d",
    }
    width = 1400
    height = 650
    plot_left = 150
    plot_right = 1330
    plot_top = 145
    plot_bottom = 520
    values = [
        value
        for record in item.aggregates
        for value in (
            cast(float, record.interval_lower),
            cast(float, record.estimate),
            cast(float, record.interval_upper),
        )
    ]
    y_low = min([0.0, *values])
    y_high = max(values)
    y_high += max((y_high - y_low) * 0.08, 1e-9)
    max_magnitude = max(record.severity.magnitude for record in item.aggregates)
    lines = _svg_header(
        width=width,
        height=height,
        title="M3 common-mode position-bias control",
        description=(
            "All 33 camera-only, lidar-only, and fixed-fusion absolute-loss aggregate "
            "rows with pointwise intervals on signed native-meter severity. Cross-modal "
            "disagreement invariance is reported as a validated blind spot."
        ),
    )
    lines.extend(
        [
            '<text class="muted" x="48" y="70" font-size="14">'
            "Absolute matched-center MSE under shared output bias is shown alongside "
            "validated camera-LiDAR disagreement invariance.</text>",
            (
                '<text class="muted" x="48" y="92" font-size="13">'
                "Validated maximum disagreement discrepancy: "
                f"{_float_text(common_validation.maximum_disagreement_discrepancy_m)} m; "
                f"tolerance {_float_text(common_validation.tolerance_m)} m; "
                f"passed={str(common_validation.passed).lower()}.</text>"
            ),
            ('<rect class="panel" x="48" y="116" width="1304" height="448" rx="8"/>'),
            (
                f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" '
                f'x2="{plot_right}" y2="{plot_bottom}"/>'
            ),
            (
                f'<line class="axis" x1="{plot_left}" y1="{plot_top}" '
                f'x2="{plot_left}" y2="{plot_bottom}"/>'
            ),
        ]
    )
    zero_x = _linear_scale(0.0, -max_magnitude, max_magnitude, plot_left, plot_right)
    lines.append(
        f'<line class="zero" x1="{zero_x:.3f}" y1="{plot_top}" '
        f'x2="{zero_x:.3f}" y2="{plot_bottom}"/>'
    )
    for method_index, method in enumerate(item.manifest.methods):
        rows = tuple(record for record in item.aggregates if record.method_id == method)
        if len(rows) != 11:
            raise M3CurationError("common-mode method series is incomplete")

        def signed_magnitude(record: AggregateMetricRecordV1Alpha1) -> float:
            if record.severity.direction == "negative":
                return -record.severity.magnitude
            return record.severity.magnitude

        ordered = tuple(sorted(rows, key=signed_magnitude))
        points: list[str] = []
        color = colors[method]
        for record in ordered:
            x_value = signed_magnitude(record)
            x = _linear_scale(
                x_value,
                -max_magnitude,
                max_magnitude,
                plot_left,
                plot_right,
            )
            estimate = cast(float, record.estimate)
            lower = cast(float, record.interval_lower)
            upper = cast(float, record.interval_upper)
            y = _linear_scale(estimate, y_low, y_high, plot_bottom, plot_top)
            y_lower = _linear_scale(lower, y_low, y_high, plot_bottom, plot_top)
            y_upper = _linear_scale(upper, y_low, y_high, plot_bottom, plot_top)
            points.append(f"{x:.3f},{y:.3f}")
            lines.extend(
                [
                    (
                        f'<line class="interval" stroke="{color}" x1="{x:.3f}" '
                        f'y1="{y_lower:.3f}" x2="{x:.3f}" y2="{y_upper:.3f}"/>'
                    ),
                    (f'<circle class="raw" fill="{color}" cx="{x:.3f}" cy="{y:.3f}" r="4"/>'),
                ]
            )
        lines.append(f'<polyline class="fit" stroke="{color}" points="{" ".join(points)}"/>')
        legend_x = 770 + method_index * 185
        lines.extend(
            [
                (
                    f'<line stroke="{color}" stroke-width="2.4" x1="{legend_x}" '
                    'y1="91" '
                    f'x2="{legend_x + 20}" y2="91"/>'
                ),
                (f'<text x="{legend_x + 26}" y="95" font-size="11">{html.escape(method)}</text>'),
            ]
        )
    lines.extend(
        [
            (
                f'<text class="muted" x="{plot_left}" y="{plot_bottom + 22}" '
                f'font-size="11">-{_float_text(max_magnitude)} m</text>'
            ),
            (
                f'<text class="muted" x="{zero_x}" y="{plot_bottom + 22}" '
                'font-size="11" text-anchor="middle">0 m</text>'
            ),
            (
                f'<text class="muted" x="{plot_right}" y="{plot_bottom + 22}" '
                f'font-size="11" text-anchor="end">+{_float_text(max_magnitude)} m</text>'
            ),
            (
                f'<text class="muted" x="{plot_left - 12}" y="{plot_top + 4}" '
                f'font-size="10" text-anchor="end">{_float_text(y_high)} m²</text>'
            ),
            (
                f'<text class="muted" x="{plot_left - 12}" y="{plot_bottom}" '
                f'font-size="10" text-anchor="end">{_float_text(y_low)} m²</text>'
            ),
            '<text class="muted" x="48" y="606" font-size="12">'
            "This is an explicit common-mode blind-spot control, not a healthy-reference "
            "crossover. No target-drop or performance oracle is defined.</text>",
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _crossover_interval(record: CrossoverRecordV1Alpha1) -> str:
    if record.status == "observed":
        return (
            f"[{_exact_float_text(record.interval_lower)}, "
            f"{_exact_float_text(cast(float, record.interval_upper))}]"
        )
    if record.status == "not-observed":
        return f"[{_exact_float_text(record.tested_maximum)}, +∞)"
    return "not two-sided"


def render_readme(release: CuratedRelease, identity: Mapping[str, Any]) -> bytes:
    """Render the public M3 release README from curated evidence only."""

    first = release.repeat_verification.first_run
    second = release.repeat_verification.second_run
    public_ci, results_review = _identity_attestation_payloads(identity)
    lines = [
        f"# Fusion Fault Bench — {RELEASE_ID}",
        "",
        "This is the aggregate-only evidence release for the frozen CPU M3",
        "procedural estimator-output benchmark. It publishes every matrix entry,",
        "all 429 aggregate rows, all 10 crossover rows, complete validation",
        "summaries, both run records per experiment, and three deterministic",
        "figures. No result, method, severity, direction, or outcome was selected",
        "after inspection.",
        "",
        "## What is included",
        "",
        "| # | Experiment | Fault axis | Aggregate rows | Crossovers | Validation |",
        "|---:|---|---|---:|---:|---|",
    ]
    for item in release.experiments:
        lines.append(
            "| "
            f"{item.execution_index} | `{item.experiment}` | "
            f"`{item.manifest.fault_sweep.axis}` "
            f"({item.manifest.fault_sweep.unit}) | "
            f"{len(item.aggregates)} | {len(item.crossovers)} | "
            f"{'PASS' if item.validation.all_checks_passed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "The only omitted scientific member is `sequence-metrics.ndjson`:",
            "71,700 generated sequence rows remain local. Each experiment retains",
            "the omitted member's exact source byte length, SHA-256, and an",
            "independently manifest-derived record count in `release-summary.json`.",
            "The public validator authenticates that commitment but cannot",
            "recompute aggregates or inspect omitted rows without regenerating the",
            "source artifacts.",
            "",
            "## Crossover outcomes",
            "",
            "These are signed, per-physical-axis stress-test results. An observed",
            "crossover is not a physical sensor tolerance.",
            "",
            "| Experiment | Direction | Status | Point estimate | "
            "95% interval / censoring | Unit |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for item in release.experiments:
        for record in item.crossovers:
            lines.append(
                "| "
                f"`{item.experiment}` | {record.direction} | {record.status} | "
                f"{_exact_float_text(record.point_estimate)} | "
                f"{_crossover_interval(record)} | {record.severity_unit} |"
            )
    lines.extend(
        [
            "",
            "Observed, not-observed, undetermined, negative, and contrary outcomes",
            "are retained. The fused-delta figure shows every one of the 54 unique",
            "signed aggregate rows, their pointwise intervals, and all 10",
            "direction-specific PAVA fits in native units.",
            "",
            "![M3 signed fusion delta curves](figures/fusion-delta-curves.svg)",
            "",
            "## Availability and common-mode controls",
            "",
            "The dropout figure contains all 72 method-by-probability aggregate",
            "rows. It displays coverage and undefined-output rate before",
            "conditional matched-center loss. Undefined conditional loss is marked",
            "and never imputed as zero. Dropout has no crossover estimand.",
            "",
            "![M3 dropout controls](figures/dropout-controls.svg)",
            "",
            "The common-mode figure contains all 33 camera-only, LiDAR-only, and",
            "fixed-fusion absolute-loss rows. It shows those result-derived curves",
            "alongside the independently validated camera-LiDAR",
            "disagreement-invariance blind spot. Common mode has no",
            "healthy-reference crossover.",
            "",
            "![M3 common-mode control](figures/common-mode-control.svg)",
            "",
            "## Provenance and repeat evidence",
            "",
            f"- Scientific source revision: `{identity['scientific_source_revision']}`",
            f"- Lockfile SHA-256: `{identity['lockfile_sha256']}`",
            f"- Artifact-set SHA-256: `{identity['artifact_set_sha256']}`",
            f"- CPU: `{first.cpu_model}`",
            (
                "- Public CI: "
                f"[GitHub Actions run {public_ci['run_id']}]({public_ci['url']}) "
                f"on `{public_ci['source_revision']}`; smoke only, not release evidence"
            ),
            (
                "- Independent adversarial results review: "
                f"`{results_review['status']}`; "
                "[included report](evidence/results-review.md), tracked source "
                f"`{results_review['reference']}`; artifact set "
                f"`{results_review['reviewed_artifact_set_sha256']}`; included "
                "byte-for-byte"
            ),
            (
                "- Primary complete-matrix measurement: "
                f"{_exact_float_text(first.wall_time_seconds)} s wall, "
                f"{first.peak_memory_bytes} bytes peak RSS"
            ),
            (
                "- Repeat complete-matrix measurement: "
                f"{_exact_float_text(second.wall_time_seconds)} s wall, "
                f"{second.peak_memory_bytes} bytes peak RSS"
            ),
            f"- Indexed scientific comparisons: {release.repeat_verification.comparison_count}",
            f"- Indexed scientific mismatches: {release.repeat_verification.mismatch_count}",
            "",
            "Both measurements are reported; neither was selected as preferred.",
            "Wall time and peak memory are self-reported by the tracked `wait4`",
            "driver and cannot be independently recomputed from this package.",
            "Distinct paths, inodes, logical commands, volatile run records, and",
            "completion markers are consistency evidence, not cryptographic proof",
            "that two executions occurred. The Git-bound official identity and source",
            "revision are the local provenance boundary. The CI and review entries",
            "are tracked attestations linked to public or human-readable references;",
            "the offline validator does not query GitHub or authenticate the reviewer.",
            "",
            "## Claim boundary",
            "",
            "M3 measures matched-center estimator-output behavior under declared",
            "procedural proxy faults. It does not evaluate a detector, raw sensor",
            "noise, association, planning, safety, production readiness,",
            "fleet-scale behavior, or real-world sensor tolerance. M3 uses no",
            "nuScenes data. The CI smoke matrix is explicitly not release evidence.",
            "",
            "See `claim-evidence.md` for exact selectors and `verification.md` for",
            "the strict validation command.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_verification(release: CuratedRelease, identity: Mapping[str, Any]) -> bytes:
    """Render deterministic reproduction and verification guidance."""

    public_ci, results_review = _identity_attestation_payloads(identity)
    lines = [
        f"# Verification — {RELEASE_ID}",
        "",
        "Run from the repository root in the locked `.venv`:",
        "",
        "```bash",
        " ".join(VALIDATION_COMMAND),
        "```",
        "",
        "The validator fails closed unless all of the following hold:",
        "",
        "- the exact `m3-procedural-v1` matrix and canonical digest are present;",
        "- the external Git-bound official identity exactly matches the included copy;",
        "- the exact Git-bound results-review report matches its included copy and",
        "  content-addressed review attestation;",
        "- all eight matrix entries occur once in frozen order;",
        "- exactly 429 aggregate and 10 crossover rows are present in source order;",
        "- the omitted 71,700 sequence rows retain exact source hashes, byte lengths,",
        "  and independently derived counts;",
        "- every curated member matches its source payload index and repeat pair;",
        "- payload indexes, artifact digests, matrix evidence, both run records,",
        "  run IDs, and both completion markers form one identity graph;",
        "- all sixteen runs share the pinned source, lock, package, full environment,",
        "  named CPU, clean-success state, and exact logical command;",
        "- generated summaries, documents, and all three SVG figures reproduce byte",
        "  for byte; and",
        "- the canonical release index exactly hashes and allowlists every member.",
        "",
        "Frozen official identity:",
        "",
        f"- source revision: `{identity['scientific_source_revision']}`",
        f"- artifact set: `{identity['artifact_set_sha256']}`",
        (f"- matrix evidence SHA-256: `{identity['matrix_validation_sha256']}`"),
        (f"- repeat evidence SHA-256: `{identity['repeat_verification_sha256']}`"),
        "",
        "Completeness facts:",
        "",
        f"- experiments: {len(release.experiments)}",
        f"- omitted sequence rows: {EXPECTED_TOTAL_SEQUENCE_ROWS}",
        f"- curated aggregate rows: {EXPECTED_TOTAL_AGGREGATE_ROWS}",
        f"- curated crossover rows: {EXPECTED_TOTAL_CROSSOVER_ROWS}",
        f"- repeat member comparisons: {release.repeat_verification.comparison_count}",
        f"- repeat member mismatches: {release.repeat_verification.mismatch_count}",
        "",
        "The release intentionally does not contain `sequence-metrics.ndjson`.",
        "Therefore standalone public validation cannot recompute aggregate values",
        "from sequence rows. Regenerate both source roots from the frozen matrix to",
        "repeat that computation.",
        "",
        "For a full fresh-clone regeneration, check out the scientific source",
        "revision above, create the locked environment, and run:",
        "",
        "```bash",
        "uv run python tools/m3_release.py execute examples/matrices/m3-procedural-v1.json \\",
        "  --first-output-dir reports/generated/m3-reproduction-first \\",
        "  --second-output-dir reports/generated/m3-reproduction-second \\",
        "  --evidence-dir reports/generated/m3-reproduction-evidence",
        "uv run python tools/m3_release.py validate examples/matrices/m3-procedural-v1.json \\",
        "  --first-output-dir reports/generated/m3-reproduction-first \\",
        "  --second-output-dir reports/generated/m3-reproduction-second \\",
        "  --evidence-dir reports/generated/m3-reproduction-evidence",
        "```",
        "",
        "Compare the regenerated artifact-set digest and indexed scientific-member",
        "hashes with `evidence/official-identity.json`. Volatile run-record hashes,",
        "wall time, and RSS are expected to differ. Byte-identical scientific",
        "members were demonstrated for the two named runs on the named locked CPU",
        "environment; cross-architecture byte identity is not claimed.",
        "",
        "Resource and execution authenticity boundary:",
        "",
        f"- `{release.repeat_verification.resource_measurement_scope}`",
        f"- `{release.repeat_verification.execution_evidence_scope}`",
        "",
        "Those literal scopes mean elapsed time and RSS are preserved observations,",
        "not independently reproducible facts, and the consistency controls do not",
        "cryptographically prove two executions. Git history binds the tracked",
        "driver and official identity. CI smoke is not M3 release evidence.",
        "",
        "Machine-bound review and CI attestations:",
        "",
        (
            f"- public CI run `{public_ci['run_id']}` concluded "
            f"`{public_ci['conclusion']}` for `{public_ci['source_revision']}`;"
        ),
        (
            f"- results review `{results_review['status']}` covers artifact set "
            f"`{results_review['reviewed_artifact_set_sha256']}` at "
            f"`{results_review['reference']}`."
        ),
        "",
        "These are content-addressed, Git-bound declarations. The offline validator",
        "checks their exact contents and links, but does not query GitHub or",
        "cryptographically authenticate the human or agent reviewer.",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_claim_evidence(
    release: CuratedRelease,
    identity: Mapping[str, Any],
) -> bytes:
    """Render the exact public claim-to-evidence ledger."""

    lines = [
        f"# Claim-evidence ledger - {RELEASE_ID}",
        "",
        "Every quantitative claim below is conditional on the frozen procedural",
        "population and proxy-fault contract. Shared provenance for every row:",
        "",
        f"- scientific revision `{identity['scientific_source_revision']}`;",
        f"- named hardware `{release.repeat_verification.first_run.cpu_model}`;",
        f"- artifact set `{identity['artifact_set_sha256']}`; and",
        "- pointwise paired complete-sequence bootstrap inference.",
        "",
        "| Claim family | Exact manifest / record selector | Figure | Validation evidence |",
        "|---|---|---|---|",
    ]
    for panel, item in enumerate(release.experiments[:6], start=1):
        lines.append(
            "| Signed fusion benefit/harm and crossover for "
            f"`{item.experiment}` | manifest `{sha256_digest(item.manifest)}`; "
            f"`records/{item.experiment}/aggregate-metrics.ndjson` where "
            "`metric_name=fused-minus-healthy`, all severities and directions; "
            f"all rows in `records/{item.experiment}/crossovers.ndjson` | "
            f"`figures/fusion-delta-curves.svg`, panel {panel} | "
            f"`records/{item.experiment}/procedural-validation.json`, all gates |"
        )
    dropout = release.experiments[6]
    lines.append(
        "| Dropout availability, undefined rate, and conditional localization loss | "
        f"manifest `{sha256_digest(dropout.manifest)}`; "
        f"`records/{dropout.experiment}/aggregate-metrics.ndjson`, all methods, "
        "probabilities, and all source metrics; figure presentation order is coverage, "
        "undefined-output-rate, conditional-matched-center-mse | "
        "`figures/dropout-controls.svg`, panels 1-3 | "
        f"`records/{dropout.experiment}/procedural-validation.json`, dropout and "
        "all other gates |"
    )
    common = release.experiments[7]
    lines.append(
        "| Common-mode absolute loss and disagreement blind spot | "
        f"manifest `{sha256_digest(common.manifest)}`; "
        f"`records/{common.experiment}/aggregate-metrics.ndjson`, all methods, "
        "severities, and directions | `figures/common-mode-control.svg` | "
        f"`records/{common.experiment}/procedural-validation.json`, "
        "`common_mode_validation` and all other gates |"
    )
    lines.extend(
        [
            "| Deterministic scientific repeat | every indexed member pair in "
            "`evidence/repeat-verification.json`; no run selection | no derived "
            "result figure | `evidence/matrix-validation.json` and "
            "`evidence/repeat-verification.json` |",
            "",
            "All aggregate and crossover selectors mean every matching source row",
            "in frozen order. No statement may omit an observed, not-observed,",
            "undetermined, negative, or contrary outcome. Physical severity units",
            "are never pooled.",
            "",
            "The strongest supported public wording is: “A deterministic CPU",
            "estimator-output benchmark measured matched-center fusion behavior",
            "under declared procedural geometry and proxy metadata faults.”",
            "",
            "Unsupported claims include real sensor-noise transfer, detector",
            "robustness, physical fault tolerance, planning or safety benefit,",
            "production readiness, and fleet generalization.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _file_role(path: Path) -> str:
    if path in DOCUMENT_PATHS:
        return "generated-document"
    if path in FIGURE_PATHS:
        return "generated-vector-figure"
    if path == SUMMARY_PATH:
        return "generated-machine-readable-summary"
    if path == MATRIX_PATH or path.parts[0] == "intent":
        return "frozen-experimental-intent"
    if path == OFFICIAL_IDENTITY_PATH:
        return "git-bound-official-identity-copy"
    if path == RESULTS_REVIEW_REPORT_PATH:
        return "git-bound-independent-adversarial-results-review"
    if path.parts[0] == "evidence":
        return "matrix-repeat-evidence"
    if path.parts[0] == "records":
        if path.name in {
            "aggregate-metrics.ndjson",
            "crossovers.ndjson",
            "procedural-validation.json",
            "source-payload-index.json",
        }:
            return "curated-primary-scientific-evidence"
        return "primary-repeat-runtime-provenance"
    raise M3CurationError("release file has no declared role")


def _file_entry(root: Path, relative: Path) -> dict[str, Any]:
    value = _read_regular(root / relative)
    return {
        "path": relative.as_posix(),
        "role": _file_role(relative),
        "byte_length": len(value),
        "sha256": _sha256(value),
    }


def _expected_release_index(
    root: Path,
    release: CuratedRelease,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    file_paths = sorted(_release_allowlist() - {RELEASE_INDEX_PATH})
    return {
        "schema": RELEASE_SCHEMA,
        "release_id": RELEASE_ID,
        "scope": "cpu-only-procedural-estimator-output-fault-benchmark",
        "matrix_id": release.matrix.matrix_id,
        "matrix_sha256": release.matrix_sha256,
        "scientific_source_revision": identity["scientific_source_revision"],
        "lockfile_sha256": identity["lockfile_sha256"],
        "package_version": identity["package_version"],
        "artifact_set_sha256": identity["artifact_set_sha256"],
        "official_identity": {
            "git_bound_source_path": OFFICIAL_IDENTITY_RELATIVE_PATH.as_posix(),
            "included_path": OFFICIAL_IDENTITY_PATH.as_posix(),
            "sha256": _sha256(canonical_json_bytes(identity)),
        },
        "public_ci": identity["public_ci"],
        "results_review": identity["results_review"],
        "selection_policy": {
            "matrix_entries": "all-eight-in-frozen-execution-order",
            "aggregate_rows": "all-429-source-rows-in-source-order",
            "crossover_rows": "all-10-source-rows-in-source-order",
            "omitted_member": PROCEDURAL_SEQUENCE_METRICS_FILE,
            "omitted_sequence_record_count": EXPECTED_TOTAL_SEQUENCE_ROWS,
            "favorable_result_selection": False,
        },
        "files": [_file_entry(root, path) for path in file_paths],
        "curation_command": list(CURATION_COMMAND),
        "validation_command": list(VALIDATION_COMMAND),
        "limitations": {
            "omitted_rows": (
                "public-validator-authenticates-commitments-but-cannot-recompute-"
                "aggregates-without-regeneration"
            ),
            "resource_measurement_scope": (release.repeat_verification.resource_measurement_scope),
            "execution_evidence_scope": (release.repeat_verification.execution_evidence_scope),
            "ci_smoke": "not-release-evidence",
            "attestation_scope": (
                "tracked-declarations; offline-validator-does-not-authenticate-"
                "GitHub-or-reviewer-identity"
            ),
        },
    }


def write_official_identity_candidate(
    destination: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one canonical no-overwrite identity candidate for review."""

    value = canonical_json_bytes(identity)
    parsed = _load_canonical_mapping_bytes(value, label="official identity candidate")
    output = Path(os.path.abspath(os.fspath(destination)))
    _reject_symlink_components(output.parent, require_exists=True)
    if os.path.lexists(output):
        raise FileExistsError("official identity candidate already exists")
    _write_exclusive(output, value)
    if _read_regular(output) != value:
        raise M3CurationError("official identity candidate changed after publication")
    return parsed


def _require_input_release_matrix(
    matrix: LoadedExperimentMatrix,
    first: Sequence[LoadedProceduralArtifact],
    second: Sequence[LoadedProceduralArtifact],
    matrix_validation: M3MatrixValidationV1,
    repeat_verification: RepeatVerificationV1,
) -> None:
    if (
        matrix.matrix.matrix_id != "m3-procedural-v1"
        or matrix.matrix_sha256 != M3_PROCEDURAL_MATRIX_SHA256
        or tuple(manifest.experiment for manifest in matrix.manifests) != EXPECTED_EXPERIMENTS
        or len(first) != 8
        or len(second) != 8
        or matrix_validation.matrix_id != "m3-procedural-v1"
        or repeat_verification.matrix_id != "m3-procedural-v1"
        or not matrix_validation.all_checks_passed
        or not repeat_verification.all_checks_passed
    ):
        raise M3CurationError("curation rejects non-release or failing M3 evidence")
    total_sequence = 0
    total_aggregate = 0
    total_crossover = 0
    for index, (manifest, primary, repeated) in enumerate(
        zip(matrix.manifests, first, second, strict=True)
    ):
        if not isinstance(
            manifest,
            (GeometryCrossoverManifest, AvailabilityControlManifest, CommonModeControlManifest),
        ):
            raise M3CurationError("curation matrix contains a non-procedural manifest")
        counts = _expected_counts(manifest)
        if (
            counts != EXPECTED_COUNTS[manifest.experiment]
            or len(primary.aggregates) != counts[1]
            or len(primary.crossovers) != counts[2]
            or primary.validation.resources.implied_sequence_row_count != counts[0]
            or primary.manifest != manifest
            or repeated.manifest != manifest
            or primary.artifact_sha256 != repeated.artifact_sha256
            or matrix_validation.ordered_artifacts[index].artifact_sha256 != primary.artifact_sha256
        ):
            raise M3CurationError("source artifacts violate literal curation completeness")
        for source_name in PROCEDURAL_INDEXED_PAYLOAD_PATHS:
            if _read_regular(primary.path / source_name) != _read_regular(
                repeated.path / source_name
            ):
                raise M3CurationError("source scientific bytes differ across repeat roots")
        total_sequence += counts[0]
        total_aggregate += counts[1]
        total_crossover += counts[2]
    if (
        total_sequence != EXPECTED_TOTAL_SEQUENCE_ROWS
        or total_aggregate != EXPECTED_TOTAL_AGGREGATE_ROWS
        or total_crossover != EXPECTED_TOTAL_CROSSOVER_ROWS
    ):
        raise M3CurationError("source matrix completeness totals changed")


def build_curated_release(
    matrix: LoadedExperimentMatrix,
    first: Sequence[LoadedProceduralArtifact],
    second: Sequence[LoadedProceduralArtifact],
    *,
    matrix_validation: M3MatrixValidationV1,
    repeat_verification: RepeatVerificationV1,
    official_identity_bytes: bytes,
    results_review_report_bytes: bytes,
    output_dir: Path,
    expected_first_output: str | None = None,
    expected_second_output: str | None = None,
) -> dict[str, Any]:
    """Build the atomic aggregate-only M3 release from strict complete roots."""

    _require_input_release_matrix(
        matrix,
        first,
        second,
        matrix_validation,
        repeat_verification,
    )
    official_identity = _load_canonical_mapping_bytes(
        official_identity_bytes,
        label="Git-bound official identity",
    )
    public_ci, results_review = _identity_attestations(
        official_identity,
        scientific_source_revision=first[0].run.git_revision,
        artifact_set_sha256=matrix_validation.artifact_set_sha256,
        results_review_report_bytes=results_review_report_bytes,
    )
    derived_identity = derive_official_identity(
        matrix,
        first,
        second,
        matrix_validation=matrix_validation,
        repeat_verification=repeat_verification,
        public_ci_attestation_bytes=canonical_json_bytes(public_ci["attestation"]),
        results_review_attestation_bytes=canonical_json_bytes(results_review["attestation"]),
        results_review_report_bytes=results_review_report_bytes,
        expected_first_output=expected_first_output,
        expected_second_output=expected_second_output,
    )
    if official_identity_bytes != canonical_json_bytes(derived_identity):
        raise M3CurationError("Git-bound official identity disagrees with strict source roots")

    output = Path(os.path.abspath(os.fspath(output_dir)))
    _reject_symlink_components(output, require_exists=False)
    if os.path.lexists(output):
        raise FileExistsError("M3 release destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output.parent, require_exists=True)
    staging = Path(tempfile.mkdtemp(prefix=".ffb-m3-release-", dir=output.parent))
    published = False
    try:
        _write_exclusive(staging / MATRIX_PATH, canonical_json_bytes(matrix.matrix))
        for manifest in matrix.manifests:
            _write_exclusive(
                staging / _manifest_path(manifest.experiment),
                canonical_json_bytes(manifest),
            )
        for profile in matrix.profiles:
            _write_exclusive(
                staging / _profile_path(profile.profile_id),
                canonical_json_bytes(profile),
            )
        for primary, repeated in zip(first, second, strict=True):
            experiment = primary.manifest.experiment
            record_root = staging / "records" / experiment
            for source_name, destination_name in RECORD_MEMBER_DESTINATIONS:
                _write_exclusive(
                    record_root / destination_name,
                    _read_regular(primary.path / source_name),
                )
            _write_exclusive(
                record_root / PRIMARY_RUN_DESTINATION,
                _read_regular(primary.path / PROCEDURAL_RUN_FILE),
            )
            _write_exclusive(
                record_root / PRIMARY_SUCCESS_DESTINATION,
                _read_regular(primary.path / PROCEDURAL_SUCCESS_FILE),
            )
            _write_exclusive(
                record_root / REPEAT_RUN_DESTINATION,
                _read_regular(repeated.path / PROCEDURAL_RUN_FILE),
            )
            _write_exclusive(
                record_root / REPEAT_SUCCESS_DESTINATION,
                _read_regular(repeated.path / PROCEDURAL_SUCCESS_FILE),
            )
        _write_exclusive(
            staging / MATRIX_VALIDATION_PATH,
            canonical_json_bytes(matrix_validation),
        )
        _write_exclusive(
            staging / REPEAT_VERIFICATION_PATH,
            canonical_json_bytes(repeat_verification),
        )
        _write_exclusive(
            staging / OFFICIAL_IDENTITY_PATH,
            official_identity_bytes,
        )
        _write_exclusive(
            staging / RESULTS_REVIEW_REPORT_PATH,
            results_review_report_bytes,
        )

        release = _load_curated_release(staging)
        if (
            canonical_json_bytes(
                _identity_from_curated(
                    release,
                    public_ci=public_ci,
                    results_review=results_review,
                    results_review_report_bytes=results_review_report_bytes,
                )
            )
            != official_identity_bytes
        ):
            raise M3CurationError("curated base files disagree with the official identity")
        _write_exclusive(
            staging / SUMMARY_PATH,
            canonical_json_bytes(build_release_summary(release)),
        )
        _write_exclusive(
            staging / FUSION_DELTA_FIGURE_PATH,
            render_fusion_delta_figure(release),
        )
        _write_exclusive(
            staging / DROPOUT_FIGURE_PATH,
            render_dropout_figure(release),
        )
        _write_exclusive(
            staging / COMMON_MODE_FIGURE_PATH,
            render_common_mode_figure(release),
        )
        _write_exclusive(
            staging / README_PATH,
            render_readme(release, official_identity),
        )
        _write_exclusive(
            staging / VERIFICATION_PATH,
            render_verification(release, official_identity),
        )
        _write_exclusive(
            staging / CLAIM_EVIDENCE_PATH,
            render_claim_evidence(release, official_identity),
        )
        index = _expected_release_index(staging, release, official_identity)
        _write_exclusive(staging / RELEASE_INDEX_PATH, canonical_json_bytes(index))
        validate_curated_release(
            staging,
            official_identity_bytes=official_identity_bytes,
            results_review_report_bytes=results_review_report_bytes,
        )
        publish_directory_no_replace(staging, output)
        published = True
        validate_curated_release(
            output,
            official_identity_bytes=official_identity_bytes,
            results_review_report_bytes=results_review_report_bytes,
        )
        return index
    except BaseException:
        if not published and staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_index_paths(
    index: Mapping[str, Any],
    *,
    scanned: set[Path],
) -> None:
    raw_files = index.get("files")
    if not isinstance(raw_files, list):
        raise M3CurationError("M3 release index files must be an array")
    paths: list[Path] = []
    for raw_entry in raw_files:
        if not isinstance(raw_entry, dict) or any(not isinstance(key, str) for key in raw_entry):
            raise M3CurationError("M3 release index file entry must be an object")
        typed_entry = cast(dict[str, object], raw_entry)
        paths.append(
            _safe_relative_path(
                typed_entry.get("path"),
                label="release index file path",
            )
        )
    if len(paths) != len(set(paths)):
        raise M3CurationError("M3 release index contains duplicate file paths")
    if set(paths) != scanned - {RELEASE_INDEX_PATH}:
        raise M3CurationError("M3 release index file paths do not exactly allowlist the tree")


def _validate_svg(value: bytes) -> None:
    if (
        not value.startswith(b'<?xml version="1.0" encoding="UTF-8"?>\n<svg ')
        or not value.endswith(b"</svg>\n")
        or b"<image" in value
        or b"data:" in value
        or b"<script" in value
    ):
        raise M3CurationError("M3 figure is not the expected vector-only SVG")


def validate_curated_release(
    root: Path,
    *,
    official_identity_bytes: bytes,
    results_review_report_bytes: bytes,
) -> dict[str, Any]:
    """Validate an exact M3 release against the external Git-bound identity."""

    release_root = Path(os.path.abspath(os.fspath(root)))
    scanned = _scan_release(release_root)
    official_identity = _load_canonical_mapping_bytes(
        official_identity_bytes,
        label="Git-bound official identity",
    )
    if _read_regular(release_root / OFFICIAL_IDENTITY_PATH) != official_identity_bytes:
        raise M3CurationError("included official identity differs from its Git-bound source")
    if _read_regular(release_root / RESULTS_REVIEW_REPORT_PATH) != results_review_report_bytes:
        raise M3CurationError("included results-review report differs from its Git-bound source")

    release = _load_curated_release(release_root)
    public_ci, results_review = _identity_attestations(
        official_identity,
        scientific_source_revision=release.experiments[0].primary_run.git_revision,
        artifact_set_sha256=release.matrix_validation.artifact_set_sha256,
        results_review_report_bytes=results_review_report_bytes,
    )
    if (
        canonical_json_bytes(
            _identity_from_curated(
                release,
                public_ci=public_ci,
                results_review=results_review,
                results_review_report_bytes=results_review_report_bytes,
            )
        )
        != official_identity_bytes
    ):
        raise M3CurationError("curated release disagrees with the Git-bound official identity")

    expected_summary = canonical_json_bytes(build_release_summary(release))
    if _read_regular(release_root / SUMMARY_PATH) != expected_summary:
        raise M3CurationError("M3 release summary does not reproduce from curated evidence")
    expected_figures = {
        FUSION_DELTA_FIGURE_PATH: render_fusion_delta_figure(release),
        DROPOUT_FIGURE_PATH: render_dropout_figure(release),
        COMMON_MODE_FIGURE_PATH: render_common_mode_figure(release),
    }
    for path, expected in expected_figures.items():
        observed = _read_regular(release_root / path)
        _validate_svg(observed)
        if observed != expected:
            raise M3CurationError("M3 release figure does not reproduce from curated evidence")
    expected_documents = {
        README_PATH: render_readme(release, official_identity),
        VERIFICATION_PATH: render_verification(release, official_identity),
        CLAIM_EVIDENCE_PATH: render_claim_evidence(release, official_identity),
    }
    for path, expected in expected_documents.items():
        if _read_regular(release_root / path) != expected:
            raise M3CurationError("M3 release document does not reproduce from evidence")

    observed_index, observed_index_bytes = _load_canonical_mapping(
        release_root / RELEASE_INDEX_PATH,
        label="M3 release index",
    )
    _validate_index_paths(observed_index, scanned=scanned)
    expected_index = _expected_release_index(
        release_root,
        release,
        official_identity,
    )
    if observed_index_bytes != canonical_json_bytes(expected_index):
        raise M3CurationError("M3 release index is incomplete or contradictory")
    return observed_index
