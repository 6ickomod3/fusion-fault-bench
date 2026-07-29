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
from fusion_fault_bench.procedural_release import (
    RepeatRunResources,
    build_m3_repeat_evidence,
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


class ProceduralReleaseDriverError(ValueError):
    """A measured M3 repeat run or evidence envelope failed closed."""


@dataclass(frozen=True, slots=True)
class LoadedM3RepeatEvidence:
    """Canonical matrix and repeat evidence loaded from one exact directory."""

    path: Path
    matrix_validation: M3MatrixValidationV1
    repeat_verification: RepeatVerificationV1


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
        metadata = os.lstat(path)
    except OSError as error:
        raise ProceduralReleaseDriverError("M3 repeat evidence member is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProceduralReleaseDriverError("M3 repeat evidence members must be real regular files")
    if metadata.st_size > _EVIDENCE_MEMBER_CAP_BYTES:
        raise ProceduralReleaseDriverError("M3 repeat evidence member exceeds its byte cap")
    value = path.read_bytes()
    if len(value) != metadata.st_size:
        raise ProceduralReleaseDriverError("M3 repeat evidence member changed while reading")
    return value


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute or validate a measured M3 two-run evidence envelope."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command_name in ("execute", "validate"):
        command = commands.add_parser(command_name)
        command.add_argument("matrix", type=Path)
        command.add_argument("--first-output-dir", type=Path, required=True)
        command.add_argument("--second-output-dir", type=Path, required=True)
        command.add_argument("--evidence-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the repository-local M3 release tool."""

    args = _build_parser().parse_args(argv)
    action = execute_procedural_repeat if args.command == "execute" else validate_procedural_repeat
    try:
        evidence = action(
            args.matrix,
            first_output_dir=args.first_output_dir,
            second_output_dir=args.second_output_dir,
            evidence_dir=args.evidence_dir,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    repeat = evidence.repeat_verification
    print(
        f"valid {repeat.schema_id} "
        f"matrix_sha256={repeat.matrix_sha256} "
        f"comparison_count={repeat.comparison_count} "
        f"evidence_dir={args.evidence_dir.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
