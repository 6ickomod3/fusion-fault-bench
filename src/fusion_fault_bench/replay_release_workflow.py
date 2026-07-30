"""Source-authenticated orchestration for the frozen M5 release workflow."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_release_v1 import M5_RELEASE_DESTINATION_PATH
from fusion_fault_bench.contracts.replay_v1 import M5_REPLAY_INTENT_PATH
from fusion_fault_bench.provenance import (
    CleanSourceSnapshot,
    RuntimeEnvironment,
    collect_runtime_environment,
    discover_clean_source,
    verify_locked_execution,
)
from fusion_fault_bench.replay_release_authority import (
    ImplementationSnapshot,
    build_implementation_snapshot,
)
from fusion_fault_bench.replay_release_validation import (
    load_implementation_review_attestation,
)

_IMPLEMENTATION_REVIEW_PATH = Path("docs/reviews/m5-release-implementation-review.md")
_IMPLEMENTATION_ATTESTATION_PATH = Path(
    "docs/reviews/m5-release-implementation-review-attestation.json"
)
_THREAD_ENVIRONMENT_KEYS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_RUN_DIRECTORY_PATTERN = re.compile(r"^m5-replay-(primary|repeat)-([0-9a-f]{40})-r([1-9][0-9]*)$")


class ReplayReleaseWorkflowError(ValueError):
    """An M5 workflow authority or deterministic reconstruction failed closed."""


@dataclass(frozen=True, slots=True)
class ReplayExecutionAuthority:
    """Outcome-blind authority that must remain identical around one replay."""

    source_root: Path
    run_label: str
    output_argument: str
    time_l_argument: str
    dataset_root_identity: tuple[int, int]
    uv_cache_root_identity: tuple[int, int]
    scientific_git_revision: str
    lockfile_sha256: str
    package_version: str
    implementation_snapshot_sha256: str
    implementation_attestation_sha256: str
    upstream_ref: str
    environment: RuntimeEnvironment
    ffb_executable: str
    time_executable: str


def _git_text(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", os.fspath(source_root), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ReplayReleaseWorkflowError("M5 Git execution authority is unavailable")
    return result.stdout.strip()


def _git_bytes(source_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", os.fspath(source_root), *arguments),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ReplayReleaseWorkflowError("M5 Git byte authority is unavailable")
    return result.stdout


def _tracked_head_blob(source_root: Path, path: Path) -> bytes:
    raw_path = path.as_posix()
    encoded_path = raw_path.encode("utf-8")
    index = _git_bytes(source_root, "ls-files", "--stage", "-z", "--", raw_path)
    tree = _git_bytes(source_root, "ls-tree", "-z", "HEAD", "--", raw_path)
    object_id = rb"([0-9a-f]{40}(?:[0-9a-f]{24})?)"
    index_match = re.fullmatch(
        rb"100644 " + object_id + rb" 0\t" + re.escape(encoded_path) + rb"\x00",
        index,
    )
    tree_match = re.fullmatch(
        rb"100644 blob " + object_id + rb"\t" + re.escape(encoded_path) + rb"\x00",
        tree,
    )
    if index_match is None or tree_match is None or index_match.group(1) != tree_match.group(1):
        raise ReplayReleaseWorkflowError(
            "M5 implementation review evidence is not an exact tracked HEAD blob"
        )
    return _git_bytes(source_root, "cat-file", "blob", tree_match.group(1).decode("ascii"))


def _clean_authority(source_root: Path) -> tuple[CleanSourceSnapshot, ImplementationSnapshot]:
    absolute = Path(os.path.abspath(os.fspath(source_root)))
    try:
        snapshot = discover_clean_source(absolute / M5_REPLAY_INTENT_PATH)
        verify_locked_execution(snapshot)
        implementation = build_implementation_snapshot(absolute)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ReplayReleaseWorkflowError(
            "M5 operation requires the exact clean locked implementation revision"
        ) from error
    if (
        snapshot.source_root != absolute
        or snapshot.git_revision != implementation.scientific_git_revision
    ):
        raise ReplayReleaseWorkflowError("M5 clean source and implementation snapshot disagree")
    return snapshot, implementation


def _read_review_authority(
    source_root: Path,
    implementation: ImplementationSnapshot,
) -> tuple[bytes, bytes]:
    report_path = source_root / _IMPLEMENTATION_REVIEW_PATH
    attestation_path = source_root / _IMPLEMENTATION_ATTESTATION_PATH
    try:
        if report_path.is_symlink() or attestation_path.is_symlink():
            raise OSError
        report = report_path.read_bytes()
        attestation = attestation_path.read_bytes()
        if report != _tracked_head_blob(
            source_root, _IMPLEMENTATION_REVIEW_PATH
        ) or attestation != _tracked_head_blob(source_root, _IMPLEMENTATION_ATTESTATION_PATH):
            raise OSError
        parsed = load_implementation_review_attestation(
            attestation,
            review_report=report,
            snapshot=implementation,
            require_release_permitting=True,
        )
    except (OSError, TypeError, ValueError) as error:
        raise ReplayReleaseWorkflowError(
            "M5 whole-revision implementation review is absent, stale, or blocking"
        ) from error
    if canonical_json_bytes(parsed) != attestation:
        raise ReplayReleaseWorkflowError("M5 implementation review attestation is noncanonical")
    return report, attestation


def _require_upstream_sync(source_root: Path, revision: str) -> str:
    upstream = _git_text(
        source_root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
    )
    if not upstream or _git_text(source_root, "rev-parse", "@{upstream}") != revision:
        raise ReplayReleaseWorkflowError(
            "M5 authoritative replay requires HEAD to equal its upstream tracking revision"
        )
    return upstream


def _normalized_run_paths(
    *,
    run_label: str,
    revision: str,
    output_dir: Path,
    time_l_output: Path,
) -> tuple[str, str]:
    if run_label not in {"primary", "repeat"}:
        raise ReplayReleaseWorkflowError("M5 replay label is not primary or repeat")
    for path, label in ((output_dir, "output"), (time_l_output, "timing log")):
        if path.is_absolute() or path.as_posix() != os.fspath(path):
            raise ReplayReleaseWorkflowError(f"M5 replay {label} path is not normalized")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ReplayReleaseWorkflowError(f"M5 replay {label} path is not normalized")
        if tuple(path.parts[:2]) != ("reports", "generated"):
            raise ReplayReleaseWorkflowError(f"M5 replay {label} is outside reports/generated")
    match = _RUN_DIRECTORY_PATTERN.fullmatch(output_dir.name)
    if match is None or match.group(1) != run_label or match.group(2) != revision:
        raise ReplayReleaseWorkflowError(
            "M5 replay output does not bind its label and scientific revision"
        )
    expected_log = output_dir.with_name(f"{output_dir.name}.time-l.txt")
    if time_l_output != expected_log:
        raise ReplayReleaseWorkflowError("M5 replay timing log does not match its output attempt")
    return output_dir.as_posix(), time_l_output.as_posix()


def _authenticated_executable(path: Path, *, label: str) -> str:
    try:
        absolute = path.resolve(strict=True)
        metadata = os.lstat(path)
    except OSError as error:
        raise ReplayReleaseWorkflowError(f"M5 {label} executable is unavailable") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not os.access(absolute, os.X_OK)
    ):
        raise ReplayReleaseWorkflowError(f"M5 {label} executable is not a safe regular file")
    return os.fspath(absolute)


def _authenticated_input_directory(
    raw: str | None,
    *,
    label: str,
    require_private: bool,
) -> tuple[Path, tuple[int, int]]:
    if not raw:
        raise ReplayReleaseWorkflowError(f"M5 replay {label} is not an absolute input")
    path = Path(raw)
    if (
        not path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReplayReleaseWorkflowError(f"M5 replay {label} is not a normalized absolute input")
    try:
        metadata = os.lstat(path)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReplayReleaseWorkflowError(f"M5 replay {label} is unavailable") from error
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or (require_private and stat.S_IMODE(metadata.st_mode) & 0o077 != 0)
    ):
        raise ReplayReleaseWorkflowError(f"M5 replay {label} is not a safe real directory")
    return resolved, (metadata.st_dev, metadata.st_ino)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _execution_authority(
    *,
    source_root: Path,
    run_label: str,
    output_dir: Path,
    time_l_output: Path,
) -> ReplayExecutionAuthority:
    clean, implementation = _clean_authority(source_root)
    _report, attestation = _read_review_authority(clean.source_root, implementation)
    output_argument, time_argument = _normalized_run_paths(
        run_label=run_label,
        revision=clean.git_revision,
        output_dir=output_dir,
        time_l_output=time_l_output,
    )
    if any(os.environ.get(name) != "1" for name in _THREAD_ENVIRONMENT_KEYS):
        raise ReplayReleaseWorkflowError("M5 replay thread environment is not frozen to one")
    dataset_root, dataset_identity = _authenticated_input_directory(
        os.environ.get("NUSCENES_ROOT"),
        label="dataset root",
        require_private=False,
    )
    cache_root, cache_identity = _authenticated_input_directory(
        os.environ.get("UV_CACHE_DIR"),
        label="UV cache root",
        require_private=True,
    )
    if (
        _paths_overlap(dataset_root, clean.source_root)
        or _paths_overlap(cache_root, clean.source_root)
        or _paths_overlap(cache_root, dataset_root)
    ):
        raise ReplayReleaseWorkflowError(
            "M5 replay dataset and cache inputs must be isolated from source and each other"
        )
    environment = collect_runtime_environment()
    if environment.os_name != "Darwin":
        raise ReplayReleaseWorkflowError("M5 authoritative replay requires the named Darwin CPU")
    expected_ffb = clean.source_root / ".venv/bin/ffb"
    discovered_ffb = shutil.which("ffb")
    if discovered_ffb is None:
        raise ReplayReleaseWorkflowError("M5 locked ffb executable is unavailable")
    ffb_executable = _authenticated_executable(Path(discovered_ffb), label="ffb")
    if Path(ffb_executable) != expected_ffb.resolve(strict=True):
        raise ReplayReleaseWorkflowError("M5 ffb executable is outside the locked environment")
    time_executable = _authenticated_executable(Path("/usr/bin/time"), label="Darwin time")
    return ReplayExecutionAuthority(
        source_root=clean.source_root,
        run_label=run_label,
        output_argument=output_argument,
        time_l_argument=time_argument,
        dataset_root_identity=dataset_identity,
        uv_cache_root_identity=cache_identity,
        scientific_git_revision=clean.git_revision,
        lockfile_sha256=clean.lockfile_sha256,
        package_version=clean.package_version,
        implementation_snapshot_sha256=implementation.sha256,
        implementation_attestation_sha256=hashlib.sha256(attestation).hexdigest(),
        upstream_ref=_require_upstream_sync(clean.source_root, clean.git_revision),
        environment=environment,
        ffb_executable=ffb_executable,
        time_executable=time_executable,
    )


def authenticate_replay_execution(
    *,
    source_root: Path,
    run_label: str,
    output_dir: Path,
    time_l_output: Path,
) -> ReplayExecutionAuthority:
    """Authenticate every outcome-blind authority before launching one replay."""

    return _execution_authority(
        source_root=source_root,
        run_label=run_label,
        output_dir=output_dir,
        time_l_output=time_l_output,
    )


def verify_replay_execution_unchanged(
    *,
    token: ReplayExecutionAuthority,
    source_root: Path,
    run_label: str,
    output_dir: Path,
    time_l_output: Path,
) -> None:
    """Require the exact same source/review/runtime authority after a replay."""

    observed = _execution_authority(
        source_root=source_root,
        run_label=run_label,
        output_dir=output_dir,
        time_l_output=time_l_output,
    )
    if observed != token:
        raise ReplayReleaseWorkflowError("M5 replay execution authority changed during the run")


def verify_software(*, source_root: Path, output: Path) -> object:
    from fusion_fault_bench.replay_release_software import (
        verify_software as run_software_verification,
    )

    clean, implementation = _clean_authority(source_root)
    _read_review_authority(clean.source_root, implementation)
    verification = run_software_verification(
        source_root=clean.source_root,
        output=output,
        clean_snapshot=clean,
        implementation_snapshot=implementation,
    )
    observed_clean, observed_implementation = _clean_authority(source_root)
    if observed_clean != clean or observed_implementation != implementation:
        raise ReplayReleaseWorkflowError("M5 source authority changed during software verification")
    _read_review_authority(observed_clean.source_root, observed_implementation)
    return verification


def prepare_review_candidate(
    *,
    primary_artifact: Path,
    repeat_artifact: Path,
    primary_time_l: Path,
    repeat_time_l: Path,
    software_verification: Path,
    output_dir: Path,
    source_root: Path,
) -> object:
    from fusion_fault_bench.replay_release_candidate import (
        load_validated_review_candidate as load_candidate,
    )
    from fusion_fault_bench.replay_release_candidate import (
        prepare_review_candidate as prepare_candidate,
    )

    clean, implementation = _clean_authority(source_root)
    report, attestation = _read_review_authority(clean.source_root, implementation)
    expected_output = Path(f"reports/generated/m5-review-candidate-{clean.git_revision}")
    if output_dir != expected_output:
        raise ReplayReleaseWorkflowError(
            "M5 review candidate destination does not bind the scientific revision"
        )
    prepare_candidate(
        primary_artifact=primary_artifact,
        repeat_artifact=repeat_artifact,
        primary_time_l=primary_time_l,
        repeat_time_l=repeat_time_l,
        software_verification=software_verification,
        output_dir=output_dir,
        source_root=clean.source_root,
        clean_snapshot=clean,
        implementation_snapshot=implementation,
        implementation_report=report,
        implementation_attestation=attestation,
    )
    observed_clean, observed_implementation = _clean_authority(source_root)
    if observed_clean != clean or observed_implementation != implementation:
        raise ReplayReleaseWorkflowError("M5 source authority changed during candidate preparation")
    return load_candidate(
        path=output_dir,
        source_root=clean.source_root,
        clean_snapshot=clean,
        implementation_snapshot=implementation,
        implementation_report=report,
        implementation_attestation=attestation,
    )


def load_validated_review_candidate(*, path: Path, source_root: Path) -> object:
    from fusion_fault_bench.replay_release_candidate import (
        load_validated_review_candidate as load_candidate,
    )

    clean, implementation = _clean_authority(source_root)
    report, attestation = _read_review_authority(clean.source_root, implementation)
    candidate = load_candidate(
        path=path,
        source_root=clean.source_root,
        clean_snapshot=clean,
        implementation_snapshot=implementation,
        implementation_report=report,
        implementation_attestation=attestation,
    )
    observed_clean, observed_implementation = _clean_authority(source_root)
    if observed_clean != clean or observed_implementation != implementation:
        raise ReplayReleaseWorkflowError("M5 source authority changed during candidate validation")
    return candidate


def validate_review_candidate(*, path: Path, source_root: Path) -> str:
    candidate = load_validated_review_candidate(path=path, source_root=source_root)
    digest = getattr(candidate, "candidate_sha256", None)
    if not isinstance(digest, str):
        raise ReplayReleaseWorkflowError("M5 validated candidate has no semantic digest")
    return digest


def build_reviewed_release(
    *,
    candidate: Path,
    results_review: Path,
    results_review_attestation: Path,
    primary_artifact: Path,
    repeat_artifact: Path,
    primary_time_l: Path,
    repeat_time_l: Path,
    software_verification: Path,
    output_dir: Path,
    source_root: Path,
) -> object:
    from fusion_fault_bench.replay_release_workflow_build import (
        orchestrate_reviewed_release,
    )

    try:
        return orchestrate_reviewed_release(
            candidate=candidate,
            results_review=results_review,
            results_review_attestation=results_review_attestation,
            primary_artifact=primary_artifact,
            repeat_artifact=repeat_artifact,
            primary_time_l=primary_time_l,
            repeat_time_l=repeat_time_l,
            software_verification=software_verification,
            output_dir=output_dir,
            source_root=source_root,
            clean_authority=_clean_authority,
            review_authority=_read_review_authority,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ReplayReleaseWorkflowError(
            "M5 reviewed release construction failed closed"
        ) from error


def _validated_digest(value: object, *, subject: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReplayReleaseWorkflowError(f"M5 validated {subject} has no semantic digest")
    return value


def validate_release_package(*, path: Path) -> str:
    from fusion_fault_bench.replay_release_package import (
        validate_release_package as validate_package,
    )

    validated = validate_package(path)
    return _validated_digest(
        validated.release_package_sha256,
        subject="release package",
    )


def validate_publication(*, release: Path, source_root: Path) -> str:
    from fusion_fault_bench.replay_release_package import (
        validate_publication as validate_repository_publication,
    )

    clean, implementation = _clean_authority(source_root)
    expected_release = clean.source_root / M5_RELEASE_DESTINATION_PATH
    observed_release = Path(
        os.path.abspath(
            os.fspath(release if release.is_absolute() else clean.source_root / release)
        )
    )
    if observed_release != expected_release:
        raise ReplayReleaseWorkflowError(
            "M5 publication validation requires the frozen tracked package path"
        )
    digest = validate_repository_publication(observed_release, clean.source_root)
    observed_clean, observed_implementation = _clean_authority(source_root)
    if observed_clean != clean or observed_implementation != implementation:
        raise ReplayReleaseWorkflowError(
            "M5 source authority changed during publication validation"
        )
    return _validated_digest(digest, subject="publication")


__all__ = [
    "ReplayExecutionAuthority",
    "ReplayReleaseWorkflowError",
    "authenticate_replay_execution",
    "build_reviewed_release",
    "load_validated_review_candidate",
    "prepare_review_candidate",
    "validate_publication",
    "validate_release_package",
    "validate_review_candidate",
    "verify_replay_execution_unchanged",
    "verify_software",
]
