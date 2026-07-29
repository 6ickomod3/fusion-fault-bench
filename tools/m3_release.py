"""Execute and validate measured two-run Fusion Fault Bench M3 evidence."""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    publish_directory_no_replace,
)
from fusion_fault_bench.contracts.matrix_v1 import (
    LoadedExperimentMatrix,
    load_experiment_matrix,
)
from fusion_fault_bench.contracts.procedural_release_v1 import (
    M3MatrixValidationV1,
    RepeatVerificationV1,
)
from fusion_fault_bench.procedural_artifacts import LoadedProceduralArtifact

if __package__:
    from tools.m3_curation import (
        OFFICIAL_IDENTITY_RELATIVE_PATH,
        PUBLIC_CI_ATTESTATION_RELATIVE_PATH,
        RELEASE_RELATIVE_PATH,
        RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH,
        RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH,
        build_curated_release,
        derive_official_identity,
        validate_curated_release,
        write_official_identity_candidate,
    )
else:
    from m3_curation import (  # pyright: ignore[reportMissingImports]
        OFFICIAL_IDENTITY_RELATIVE_PATH,
        PUBLIC_CI_ATTESTATION_RELATIVE_PATH,
        RELEASE_RELATIVE_PATH,
        RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH,
        RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH,
        build_curated_release,
        derive_official_identity,
        validate_curated_release,
        write_official_identity_candidate,
    )
from fusion_fault_bench.procedural_release import (
    RepeatRunResources,
    build_m3_repeat_evidence,
    load_m3_artifact_set,
    validate_m3_release_eligibility,
)
from fusion_fault_bench.provenance import (
    CleanSourceSnapshot,
    discover_clean_source,
    verify_locked_execution,
)

M3_MATRIX_VALIDATION_FILE = "matrix-validation.json"
M3_REPEAT_VERIFICATION_FILE = "repeat-verification.json"
M3_REPEAT_EVIDENCE_PATHS = (
    M3_MATRIX_VALIDATION_FILE,
    M3_REPEAT_VERIFICATION_FILE,
)
_GENERATED_ROOT = Path("reports/generated")
_EVIDENCE_MEMBER_CAP_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


class ProceduralReleaseDriverError(ValueError):
    """A measured M3 repeat run or evidence envelope failed closed."""


@dataclass(frozen=True, slots=True)
class LoadedM3RepeatEvidence:
    """Canonical matrix and repeat evidence loaded from one exact directory."""

    path: Path
    matrix_validation: M3MatrixValidationV1
    repeat_verification: RepeatVerificationV1


@dataclass(frozen=True, slots=True)
class StrictM3ReleaseInputs:
    """Strict full-matrix roots and persisted evidence ready for curation."""

    snapshot: CleanSourceSnapshot
    matrix: LoadedExperimentMatrix
    first_root: Path
    second_root: Path
    evidence_root: Path
    first_artifacts: tuple[LoadedProceduralArtifact, ...]
    second_artifacts: tuple[LoadedProceduralArtifact, ...]
    evidence: LoadedM3RepeatEvidence


def _reject_symlink_components(path: Path, *, require_exists: bool) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if require_exists:
                raise ProceduralReleaseDriverError(
                    "M3 repeat path contains a missing component"
                ) from None
            return
        except OSError as error:
            raise ProceduralReleaseDriverError(
                "M3 repeat path components cannot be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ProceduralReleaseDriverError(
                "M3 repeat paths must not contain symlink components"
            )


def _initial_snapshot(matrix_path: Path) -> CleanSourceSnapshot:
    try:
        snapshot = discover_clean_source(matrix_path)
        verify_locked_execution(snapshot)
        return snapshot
    except (OSError, ValueError) as error:
        raise ProceduralReleaseDriverError(
            "M3 repeat execution requires a clean locked source checkout"
        ) from error


def _verify_unchanged_source(
    matrix_path: Path,
    *,
    initial: CleanSourceSnapshot,
) -> None:
    try:
        final = discover_clean_source(matrix_path)
        verify_locked_execution(final)
    except (OSError, ValueError) as error:
        raise ProceduralReleaseDriverError(
            "M3 source validation failed after a measured child run"
        ) from error
    if final != initial:
        raise ProceduralReleaseDriverError("M3 source provenance changed during repeat execution")


def _generated_destination(
    value: Path,
    *,
    source_root: Path,
    label: str,
) -> Path:
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise ProceduralReleaseDriverError(f"{label} must be a normalized repository-relative path")
    try:
        value.relative_to(_GENERATED_ROOT)
    except ValueError:
        raise ProceduralReleaseDriverError(f"{label} must remain under reports/generated") from None
    if value == _GENERATED_ROOT:
        raise ProceduralReleaseDriverError(f"{label} requires a matrix-specific directory")
    destination = source_root / value
    _reject_symlink_components(destination, require_exists=False)
    return destination


def _require_disjoint_destinations(
    destinations: tuple[Path, Path, Path],
    *,
    require_absent: bool,
) -> None:
    if len(set(destinations)) != len(destinations):
        raise ProceduralReleaseDriverError("repeat run and evidence destinations must be distinct")
    if any(
        left in right.parents or right in left.parents
        for index, left in enumerate(destinations)
        for right in destinations[index + 1 :]
    ):
        raise ProceduralReleaseDriverError(
            "repeat run and evidence destinations must not be nested"
        )
    if require_absent and any(os.path.lexists(path) for path in destinations):
        raise FileExistsError("M3 repeat execution never overwrites a destination")


def _peak_memory_bytes(usage: resource.struct_rusage) -> int:
    raw = float(usage.ru_maxrss)
    if not raw > 0.0:
        raise ProceduralReleaseDriverError(
            "child resource usage did not report positive peak memory"
        )
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(raw * multiplier)


def _run_measured_matrix_child(
    *,
    source_root: Path,
    matrix_relative_path: str,
    output_relative_path: str,
) -> RepeatRunResources:
    command = (
        sys.executable,
        "-m",
        "fusion_fault_bench",
        "procedural",
        "matrix",
        "run",
        matrix_relative_path,
        "--output-dir",
        output_relative_path,
    )
    started = time.perf_counter()
    try:
        process = subprocess.Popen(command, cwd=source_root)
        while True:
            try:
                waited_pid, status, usage = os.wait4(process.pid, 0)
                break
            except InterruptedError:
                continue
    except OSError as error:
        raise ProceduralReleaseDriverError(
            "could not execute the measured M3 matrix child"
        ) from error
    if waited_pid != process.pid:
        raise ProceduralReleaseDriverError(
            "measured M3 child returned an unexpected process identity"
        )
    return_code = os.waitstatus_to_exitcode(status)
    process.returncode = return_code
    wall_time_seconds = time.perf_counter() - started
    if return_code != 0:
        raise ProceduralReleaseDriverError("measured M3 matrix child did not complete successfully")
    if not wall_time_seconds > 0.0:
        raise ProceduralReleaseDriverError("measured M3 wall time must be positive")
    return RepeatRunResources(
        wall_time_seconds=wall_time_seconds,
        peak_memory_bytes=_peak_memory_bytes(usage),
    )


def _read_evidence_member(root: Path, name: str) -> bytes:
    path = root / name
    try:
        before = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ProceduralReleaseDriverError("M3 repeat evidence member is unavailable") from error
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
            raise ProceduralReleaseDriverError(
                "M3 repeat evidence members must be one stable real regular file"
            )
        if opened.st_size > _EVIDENCE_MEMBER_CAP_BYTES:
            raise ProceduralReleaseDriverError("M3 repeat evidence member exceeds its byte cap")
        chunks: list[bytes] = []
        remaining = _EVIDENCE_MEMBER_CAP_BYTES + 1
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
        if (
            len(value) > _EVIDENCE_MEMBER_CAP_BYTES
            or len(value) != opened.st_size
            or after_fingerprint != fingerprint
        ):
            raise ProceduralReleaseDriverError("M3 repeat evidence member changed while reading")
        return value
    except OSError as error:
        raise ProceduralReleaseDriverError(
            "M3 repeat evidence member could not be read safely"
        ) from error
    finally:
        os.close(descriptor)


def load_m3_repeat_evidence(path: Path) -> LoadedM3RepeatEvidence:
    """Load one exact canonical two-file M3 repeat evidence envelope."""

    root = Path(os.path.abspath(os.fspath(path)))
    _reject_symlink_components(root, require_exists=True)
    try:
        metadata = os.lstat(root)
        entries = tuple(root.iterdir())
    except OSError as error:
        raise ProceduralReleaseDriverError(
            "M3 repeat evidence directory cannot be inspected"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProceduralReleaseDriverError("M3 repeat evidence path must be a real directory")
    if {entry.name for entry in entries} != set(M3_REPEAT_EVIDENCE_PATHS):
        raise ProceduralReleaseDriverError("M3 repeat evidence file allowlist is invalid")
    try:
        matrix_bytes = _read_evidence_member(root, M3_MATRIX_VALIDATION_FILE)
        repeat_bytes = _read_evidence_member(root, M3_REPEAT_VERIFICATION_FILE)
        matrix = M3MatrixValidationV1.model_validate_json(matrix_bytes)
        repeat = RepeatVerificationV1.model_validate_json(repeat_bytes)
    except (ValidationError, ValueError) as error:
        raise ProceduralReleaseDriverError(
            "M3 repeat evidence violates its strict schema"
        ) from error
    if canonical_json_bytes(matrix) != matrix_bytes or canonical_json_bytes(repeat) != repeat_bytes:
        raise ProceduralReleaseDriverError("M3 repeat evidence must use canonical JSON bytes")
    if (
        matrix.matrix_id != repeat.matrix_id
        or matrix.matrix_sha256 != repeat.matrix_sha256
        or matrix.artifact_set_sha256 != repeat.first_run.artifact_set_sha256
    ):
        raise ProceduralReleaseDriverError("M3 repeat evidence files are not cross-linked")
    return LoadedM3RepeatEvidence(
        path=root,
        matrix_validation=matrix,
        repeat_verification=repeat,
    )


def _write_m3_repeat_evidence(
    destination: Path,
    *,
    matrix_validation: M3MatrixValidationV1,
    repeat_verification: RepeatVerificationV1,
) -> LoadedM3RepeatEvidence:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ffb-m3-repeat-", dir=parent))
    published = False
    try:
        payloads = {
            M3_MATRIX_VALIDATION_FILE: canonical_json_bytes(matrix_validation),
            M3_REPEAT_VERIFICATION_FILE: canonical_json_bytes(repeat_verification),
        }
        for name in M3_REPEAT_EVIDENCE_PATHS:
            with (staging / name).open("xb") as stream:
                stream.write(payloads[name])
                stream.flush()
                os.fsync(stream.fileno())
        load_m3_repeat_evidence(staging)
        publish_directory_no_replace(staging, destination)
        published = True
        return load_m3_repeat_evidence(destination)
    except (ArtifactValidationError, OSError, ValueError) as error:
        if isinstance(error, (FileExistsError, ProceduralReleaseDriverError)):
            raise
        raise ProceduralReleaseDriverError(
            "M3 repeat evidence could not be published atomically"
        ) from error
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _load_matrix_for_snapshot(
    matrix_path: Path,
    *,
    snapshot: CleanSourceSnapshot,
) -> LoadedExperimentMatrix:
    matrix = load_experiment_matrix(
        matrix_path,
        source_root=snapshot.source_root,
    )
    expected_relative = matrix.path.relative_to(snapshot.source_root).as_posix()
    if snapshot.manifest_relative_path != expected_relative:
        raise ProceduralReleaseDriverError(
            "clean-source snapshot does not identify the loaded M3 matrix"
        )
    return matrix


def execute_procedural_repeat(
    matrix_path: Path,
    *,
    first_output_dir: Path,
    second_output_dir: Path,
    evidence_dir: Path,
) -> LoadedM3RepeatEvidence:
    """Run a frozen matrix twice in children, measure it, and publish evidence."""

    snapshot = _initial_snapshot(matrix_path)
    matrix = _load_matrix_for_snapshot(matrix_path, snapshot=snapshot)
    first_root = _generated_destination(
        first_output_dir,
        source_root=snapshot.source_root,
        label="first output",
    )
    second_root = _generated_destination(
        second_output_dir,
        source_root=snapshot.source_root,
        label="second output",
    )
    evidence_root = _generated_destination(
        evidence_dir,
        source_root=snapshot.source_root,
        label="evidence output",
    )
    _require_disjoint_destinations(
        (first_root, second_root, evidence_root),
        require_absent=True,
    )
    matrix_relative = matrix.path.relative_to(snapshot.source_root).as_posix()
    first_resources = _run_measured_matrix_child(
        source_root=snapshot.source_root,
        matrix_relative_path=matrix_relative,
        output_relative_path=first_output_dir.as_posix(),
    )
    _verify_unchanged_source(matrix_path, initial=snapshot)
    second_resources = _run_measured_matrix_child(
        source_root=snapshot.source_root,
        matrix_relative_path=matrix_relative,
        output_relative_path=second_output_dir.as_posix(),
    )
    _verify_unchanged_source(matrix_path, initial=snapshot)
    matrix_validation, repeat_verification = build_m3_repeat_evidence(
        matrix,
        first_root,
        second_root,
        first_resources=first_resources,
        second_resources=second_resources,
    )
    written = _write_m3_repeat_evidence(
        evidence_root,
        matrix_validation=matrix_validation,
        repeat_verification=repeat_verification,
    )
    _verify_unchanged_source(matrix_path, initial=snapshot)
    return written


def validate_procedural_repeat(
    matrix_path: Path,
    *,
    first_output_dir: Path,
    second_output_dir: Path,
    evidence_dir: Path,
) -> LoadedM3RepeatEvidence:
    """Strictly rebuild a persisted repeat envelope from both artifact roots."""

    snapshot = _initial_snapshot(matrix_path)
    matrix = _load_matrix_for_snapshot(matrix_path, snapshot=snapshot)
    first_root = _generated_destination(
        first_output_dir,
        source_root=snapshot.source_root,
        label="first output",
    )
    second_root = _generated_destination(
        second_output_dir,
        source_root=snapshot.source_root,
        label="second output",
    )
    evidence_root = _generated_destination(
        evidence_dir,
        source_root=snapshot.source_root,
        label="evidence output",
    )
    _require_disjoint_destinations(
        (first_root, second_root, evidence_root),
        require_absent=False,
    )
    persisted = load_m3_repeat_evidence(evidence_root)
    first_resources = RepeatRunResources(
        wall_time_seconds=persisted.repeat_verification.first_run.wall_time_seconds,
        peak_memory_bytes=persisted.repeat_verification.first_run.peak_memory_bytes,
    )
    second_resources = RepeatRunResources(
        wall_time_seconds=persisted.repeat_verification.second_run.wall_time_seconds,
        peak_memory_bytes=persisted.repeat_verification.second_run.peak_memory_bytes,
    )
    rebuilt_matrix, rebuilt_repeat = build_m3_repeat_evidence(
        matrix,
        first_root,
        second_root,
        first_resources=first_resources,
        second_resources=second_resources,
    )
    if (
        persisted.matrix_validation != rebuilt_matrix
        or persisted.repeat_verification != rebuilt_repeat
    ):
        raise ProceduralReleaseDriverError(
            "persisted M3 repeat evidence disagrees with strict artifact roots"
        )
    _verify_unchanged_source(matrix_path, initial=snapshot)
    return persisted


def _strict_release_inputs(
    matrix_path: Path,
    *,
    first_output_dir: Path,
    second_output_dir: Path,
    evidence_dir: Path,
) -> StrictM3ReleaseInputs:
    persisted = validate_procedural_repeat(
        matrix_path,
        first_output_dir=first_output_dir,
        second_output_dir=second_output_dir,
        evidence_dir=evidence_dir,
    )
    snapshot = _initial_snapshot(matrix_path)
    matrix = _load_matrix_for_snapshot(matrix_path, snapshot=snapshot)
    first_root = _generated_destination(
        first_output_dir,
        source_root=snapshot.source_root,
        label="first output",
    )
    second_root = _generated_destination(
        second_output_dir,
        source_root=snapshot.source_root,
        label="second output",
    )
    evidence_root = _generated_destination(
        evidence_dir,
        source_root=snapshot.source_root,
        label="evidence output",
    )
    _require_disjoint_destinations(
        (first_root, second_root, evidence_root),
        require_absent=False,
    )
    first_artifacts = load_m3_artifact_set(matrix, first_root)
    second_artifacts = load_m3_artifact_set(matrix, second_root)
    first_resources = RepeatRunResources(
        wall_time_seconds=persisted.repeat_verification.first_run.wall_time_seconds,
        peak_memory_bytes=persisted.repeat_verification.first_run.peak_memory_bytes,
    )
    second_resources = RepeatRunResources(
        wall_time_seconds=persisted.repeat_verification.second_run.wall_time_seconds,
        peak_memory_bytes=persisted.repeat_verification.second_run.peak_memory_bytes,
    )
    validate_m3_release_eligibility(
        persisted.matrix_validation,
        persisted.repeat_verification,
        matrix=matrix,
        first_run_root=first_root,
        second_run_root=second_root,
        first_resources=first_resources,
        second_resources=second_resources,
    )
    _verify_unchanged_source(matrix_path, initial=snapshot)
    return StrictM3ReleaseInputs(
        snapshot=snapshot,
        matrix=matrix,
        first_root=first_root,
        second_root=second_root,
        evidence_root=evidence_root,
        first_artifacts=first_artifacts,
        second_artifacts=second_artifacts,
        evidence=persisted,
    )


def _generated_identity_destination(
    value: Path,
    *,
    source_root: Path,
) -> Path:
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise ProceduralReleaseDriverError(
            "identity candidate must be a normalized repository-relative path"
        )
    try:
        value.relative_to(_GENERATED_ROOT)
    except ValueError:
        raise ProceduralReleaseDriverError(
            "identity candidate must remain under reports/generated"
        ) from None
    if value == _GENERATED_ROOT or value.suffix != ".json":
        raise ProceduralReleaseDriverError(
            "identity candidate must name a JSON file below reports/generated"
        )
    destination = source_root / value
    _reject_symlink_components(destination, require_exists=False)
    return destination


def derive_identity_candidate(
    matrix_path: Path,
    *,
    first_output_dir: Path,
    second_output_dir: Path,
    evidence_dir: Path,
    output_path: Path,
    public_ci_attestation_path: Path,
    results_review_attestation_path: Path,
    results_review_report_path: Path,
) -> dict[str, object]:
    """Derive a generated post-run identity candidate after every strict gate."""

    inputs = _strict_release_inputs(
        matrix_path,
        first_output_dir=first_output_dir,
        second_output_dir=second_output_dir,
        evidence_dir=evidence_dir,
    )
    destination = _generated_identity_destination(
        output_path,
        source_root=inputs.snapshot.source_root,
    )
    if any(
        destination == root or destination in root.parents or root in destination.parents
        for root in (
            inputs.first_root,
            inputs.second_root,
            inputs.evidence_root,
        )
    ):
        raise ProceduralReleaseDriverError(
            "identity candidate must be disjoint from both run roots and repeat evidence"
        )
    public_ci_attestation_bytes = _git_bound_auxiliary(
        public_ci_attestation_path,
        expected_source=inputs.snapshot,
        expected_relative=PUBLIC_CI_ATTESTATION_RELATIVE_PATH,
        label="public CI attestation",
    )
    results_review_attestation_bytes = _git_bound_auxiliary(
        results_review_attestation_path,
        expected_source=inputs.snapshot,
        expected_relative=RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH,
        label="results-review attestation",
    )
    results_review_report_bytes = _git_bound_auxiliary(
        results_review_report_path,
        expected_source=inputs.snapshot,
        expected_relative=RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH,
        label="results-review report",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(destination.parent, require_exists=True)
    identity = derive_official_identity(
        inputs.matrix,
        inputs.first_artifacts,
        inputs.second_artifacts,
        matrix_validation=inputs.evidence.matrix_validation,
        repeat_verification=inputs.evidence.repeat_verification,
        public_ci_attestation_bytes=public_ci_attestation_bytes,
        results_review_attestation_bytes=results_review_attestation_bytes,
        results_review_report_bytes=results_review_report_bytes,
        expected_first_output=first_output_dir.as_posix(),
        expected_second_output=second_output_dir.as_posix(),
    )
    written = write_official_identity_candidate(destination, identity)
    _verify_unchanged_source(matrix_path, initial=inputs.snapshot)
    return cast(dict[str, object], written)


def _git_bound_auxiliary(
    path: Path,
    *,
    expected_source: CleanSourceSnapshot,
    expected_relative: Path,
    label: str,
) -> bytes:
    auxiliary_snapshot = _initial_snapshot(path)
    if (
        auxiliary_snapshot.source_root != expected_source.source_root
        or auxiliary_snapshot.git_revision != expected_source.git_revision
        or auxiliary_snapshot.lockfile_sha256 != expected_source.lockfile_sha256
        or auxiliary_snapshot.package_version != expected_source.package_version
        or auxiliary_snapshot.manifest_relative_path != expected_relative.as_posix()
    ):
        raise ProceduralReleaseDriverError(f"{label} must be its exact tracked Git-bound M3 source")
    absolute = auxiliary_snapshot.source_root / auxiliary_snapshot.manifest_relative_path
    return _read_evidence_member(absolute.parent, absolute.name)


def _git_bound_identity(
    path: Path,
    *,
    expected_source: CleanSourceSnapshot,
) -> bytes:
    return _git_bound_auxiliary(
        path,
        expected_source=expected_source,
        expected_relative=OFFICIAL_IDENTITY_RELATIVE_PATH,
        label="official identity",
    )


def build_procedural_curated_release(
    matrix_path: Path,
    *,
    first_output_dir: Path,
    second_output_dir: Path,
    evidence_dir: Path,
    official_identity_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Build the official aggregate-only M3 package from strict full roots."""

    inputs = _strict_release_inputs(
        matrix_path,
        first_output_dir=first_output_dir,
        second_output_dir=second_output_dir,
        evidence_dir=evidence_dir,
    )
    if output_dir != RELEASE_RELATIVE_PATH:
        raise ProceduralReleaseDriverError(
            f"official M3 output must be {RELEASE_RELATIVE_PATH.as_posix()}"
        )
    output = inputs.snapshot.source_root / output_dir
    _reject_symlink_components(output, require_exists=False)
    if any(
        output == root or output in root.parents or root in output.parents
        for root in (inputs.first_root, inputs.second_root, inputs.evidence_root)
    ):
        raise ProceduralReleaseDriverError(
            "release output must be disjoint from both run roots and repeat evidence"
        )
    identity_bytes = _git_bound_identity(
        official_identity_path,
        expected_source=inputs.snapshot,
    )
    results_review_report_bytes = _git_bound_auxiliary(
        RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH,
        expected_source=inputs.snapshot,
        expected_relative=RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH,
        label="results-review report",
    )
    index = build_curated_release(
        inputs.matrix,
        inputs.first_artifacts,
        inputs.second_artifacts,
        matrix_validation=inputs.evidence.matrix_validation,
        repeat_verification=inputs.evidence.repeat_verification,
        official_identity_bytes=identity_bytes,
        results_review_report_bytes=results_review_report_bytes,
        output_dir=output,
        expected_first_output=first_output_dir.as_posix(),
        expected_second_output=second_output_dir.as_posix(),
    )
    return cast(dict[str, object], index)


def validate_procedural_curated_release(
    release_dir: Path,
    *,
    official_identity_path: Path,
) -> dict[str, object]:
    """Validate official M3 evidence against a clean tracked identity."""

    snapshot = _initial_snapshot(official_identity_path)
    if snapshot.manifest_relative_path != OFFICIAL_IDENTITY_RELATIVE_PATH.as_posix():
        raise ProceduralReleaseDriverError(
            "release validation requires the official tracked M3 identity path"
        )
    if release_dir.is_absolute() or any(part in {"", ".", ".."} for part in release_dir.parts):
        raise ProceduralReleaseDriverError(
            "release validation path must be normalized and repository-relative"
        )
    release_root = snapshot.source_root / release_dir
    identity_root = snapshot.source_root / OFFICIAL_IDENTITY_RELATIVE_PATH
    identity_bytes = _read_evidence_member(identity_root.parent, identity_root.name)
    results_review_report_bytes = _git_bound_auxiliary(
        RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH,
        expected_source=snapshot,
        expected_relative=RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH,
        label="results-review report",
    )
    index = validate_curated_release(
        release_root,
        official_identity_bytes=identity_bytes,
        results_review_report_bytes=results_review_report_bytes,
    )
    _verify_unchanged_source(official_identity_path, initial=snapshot)
    return cast(dict[str, object], index)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Execute, validate, identity-freeze, and curate measured M3 evidence.")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command_name in ("execute", "validate"):
        command = commands.add_parser(command_name)
        command.add_argument("matrix", type=Path)
        command.add_argument("--first-output-dir", type=Path, required=True)
        command.add_argument("--second-output-dir", type=Path, required=True)
        command.add_argument("--evidence-dir", type=Path, required=True)
    derive = commands.add_parser(
        "derive-identity",
        help="derive a generated official-identity candidate after strict validation",
    )
    derive.add_argument("matrix", type=Path)
    derive.add_argument("--first-output-dir", type=Path, required=True)
    derive.add_argument("--second-output-dir", type=Path, required=True)
    derive.add_argument("--evidence-dir", type=Path, required=True)
    derive.add_argument("--output-path", type=Path, required=True)
    derive.add_argument(
        "--public-ci-attestation",
        type=Path,
        default=PUBLIC_CI_ATTESTATION_RELATIVE_PATH,
    )
    derive.add_argument(
        "--results-review-attestation",
        type=Path,
        default=RESULTS_REVIEW_ATTESTATION_RELATIVE_PATH,
    )
    derive.add_argument(
        "--results-review-report",
        type=Path,
        default=RESULTS_REVIEW_MARKDOWN_RELATIVE_PATH,
    )

    build_release = commands.add_parser(
        "build-release",
        help="build the official no-overwrite aggregate-only M3 release",
    )
    build_release.add_argument("matrix", type=Path)
    build_release.add_argument("--first-output-dir", type=Path, required=True)
    build_release.add_argument("--second-output-dir", type=Path, required=True)
    build_release.add_argument("--evidence-dir", type=Path, required=True)
    build_release.add_argument(
        "--official-identity",
        type=Path,
        default=OFFICIAL_IDENTITY_RELATIVE_PATH,
    )
    build_release.add_argument(
        "--output-dir",
        type=Path,
        default=RELEASE_RELATIVE_PATH,
    )

    validate_release = commands.add_parser(
        "validate-release",
        help="validate curated M3 evidence against its tracked official identity",
    )
    validate_release.add_argument("release_dir", type=Path)
    validate_release.add_argument(
        "--official-identity",
        type=Path,
        default=OFFICIAL_IDENTITY_RELATIVE_PATH,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the repository-local M3 release tool."""

    args = _build_parser().parse_args(argv)
    try:
        if args.command in {"execute", "validate"}:
            action = (
                execute_procedural_repeat
                if args.command == "execute"
                else validate_procedural_repeat
            )
            evidence = action(
                args.matrix,
                first_output_dir=args.first_output_dir,
                second_output_dir=args.second_output_dir,
                evidence_dir=args.evidence_dir,
            )
            repeat = evidence.repeat_verification
            print(
                f"valid {repeat.schema_id} "
                f"matrix_sha256={repeat.matrix_sha256} "
                f"comparison_count={repeat.comparison_count} "
                f"evidence_dir={args.evidence_dir.as_posix()}"
            )
        elif args.command == "derive-identity":
            identity = derive_identity_candidate(
                args.matrix,
                first_output_dir=args.first_output_dir,
                second_output_dir=args.second_output_dir,
                evidence_dir=args.evidence_dir,
                output_path=args.output_path,
                public_ci_attestation_path=args.public_ci_attestation,
                results_review_attestation_path=args.results_review_attestation,
                results_review_report_path=args.results_review_report,
            )
            print(
                f"derived {identity['schema']} "
                f"artifact_set_sha256={identity['artifact_set_sha256']} "
                f"output_path={args.output_path.as_posix()}"
            )
        elif args.command == "build-release":
            index = build_procedural_curated_release(
                args.matrix,
                first_output_dir=args.first_output_dir,
                second_output_dir=args.second_output_dir,
                evidence_dir=args.evidence_dir,
                official_identity_path=args.official_identity,
                output_dir=args.output_dir,
            )
            print(f"built {index['release_id']} artifact_set_sha256={index['artifact_set_sha256']}")
        else:
            index = validate_procedural_curated_release(
                args.release_dir,
                official_identity_path=args.official_identity,
            )
            print(f"valid {index['release_id']} artifact_set_sha256={index['artifact_set_sha256']}")
    except (OSError, ValueError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
