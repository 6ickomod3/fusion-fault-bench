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
from fusion_fault_bench.replay_release import LoadedReplayReviewCandidate
from fusion_fault_bench.replay_release_authority import (
    ImplementationSnapshot,
    build_implementation_snapshot,
)
from fusion_fault_bench.replay_release_validation import (
    load_implementation_review_attestation,
    load_software_verification,
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
_RUN_DIRECTORY_PATTERN = re.compile(r"^m5-replay-(primary|repeat)-([0-9a-f]{40})-r1$")
_SOFTWARE_VERIFICATION_BYTE_CAP = 16 * 1024 * 1024
_EXECUTABLE_BYTE_CAP = 128 * 1024 * 1024
_TIMING_LOG_BYTE_CAP = 1024 * 1024
_SUCCESS_RECEIPT_BYTE_CAP = 64 * 1024
_LIVE_UPSTREAM_TIMEOUT_SECONDS = 20.0
_STRICT_SSH_COMMAND = "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes"


class ReplayReleaseWorkflowError(ValueError):
    """An M5 workflow authority or deterministic reconstruction failed closed."""


@dataclass(frozen=True, slots=True)
class ReplayExecutableFingerprint:
    """Stable metadata and content authority for one launched executable."""

    device: int
    inode: int
    mode: int
    link_count: int
    owner_uid: int
    owner_gid: int
    byte_length: int
    modified_time_ns: int
    changed_time_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ReplayExecutionSuccessReceipt:
    """Exclusive local receipt authorizing the next frozen lifecycle step."""

    path: Path
    value: bytes


@dataclass(frozen=True, slots=True)
class ReplayExecutionAuthority:
    """Outcome-blind authority that must remain identical around one replay."""

    source_root: Path
    run_label: str
    output_argument: str
    time_l_argument: str
    success_argument: str
    dataset_root_identity: tuple[int, int]
    uv_cache_root_identity: tuple[int, int]
    scientific_git_revision: str
    lockfile_sha256: str
    package_version: str
    implementation_snapshot_sha256: str
    implementation_attestation_sha256: str
    software_verification_argument: str
    software_verification_sha256: str
    upstream_ref: str
    environment: RuntimeEnvironment
    ffb_executable: str
    ffb_executable_fingerprint: ReplayExecutableFingerprint
    time_executable: str
    time_executable_fingerprint: ReplayExecutableFingerprint


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


def _live_remote_bytes(
    source_root: Path,
    *,
    remote: str,
    merge_ref: str,
    uses_ssh: bool,
) -> bytes:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    if uses_ssh:
        environment["GIT_SSH_COMMAND"] = _STRICT_SSH_COMMAND
    try:
        result = subprocess.run(
            (
                "git",
                "-C",
                os.fspath(source_root),
                "ls-remote",
                "--exit-code",
                remote,
                merge_ref,
            ),
            check=False,
            capture_output=True,
            env=environment,
            timeout=_LIVE_UPSTREAM_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise ReplayReleaseWorkflowError(
            "M5 authoritative replay could not authenticate the upstream revision"
        ) from None
    if result.returncode != 0:
        raise ReplayReleaseWorkflowError(
            "M5 authoritative replay could not authenticate the upstream revision"
        )
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
    if Path(os.path.abspath(os.curdir)) != absolute:
        raise ReplayReleaseWorkflowError(
            "M5 operation working directory must equal the authenticated source root"
        )
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
    try:
        branch = _git_text(source_root, "symbolic-ref", "--quiet", "--short", "HEAD")
        remote = _git_text(source_root, "config", "--get", f"branch.{branch}.remote")
        merge_ref = _git_text(source_root, "config", "--get", f"branch.{branch}.merge")
        upstream = _git_text(
            source_root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
        tracking_ref = _git_text(
            source_root,
            "rev-parse",
            "--symbolic-full-name",
            "@{upstream}",
        )
        tracking_revision = _git_text(source_root, "rev-parse", "@{upstream}")
        remote_url = _git_text(source_root, "remote", "get-url", remote)
    except ReplayReleaseWorkflowError as error:
        raise ReplayReleaseWorkflowError(
            "M5 authoritative replay requires an authenticated remote-tracking upstream"
        ) from error
    if (
        not branch
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", remote) is None
        or remote == "."
        or re.fullmatch(r"refs/heads/[A-Za-z0-9._/-]+", merge_ref) is None
        or ".." in merge_ref
        or "//" in merge_ref
        or "@{" in merge_ref
        or merge_ref.endswith(("/", ".", ".lock"))
    ):
        raise ReplayReleaseWorkflowError(
            "M5 authoritative replay requires an authenticated remote-tracking upstream"
        )
    branch_tail = merge_ref.removeprefix("refs/heads/")
    expected_tracking_ref = f"refs/remotes/{remote}/{branch_tail}"
    if (
        upstream != f"{remote}/{branch_tail}"
        or tracking_ref != expected_tracking_ref
        or tracking_revision != revision
    ):
        raise ReplayReleaseWorkflowError(
            "M5 authoritative replay requires HEAD to equal its upstream tracking revision"
        )
    https_authority = (
        remote_url.removeprefix("https://").split("/", 1)[0]
        if remote_url.startswith("https://")
        else ""
    )
    uses_ssh = (
        remote_url.startswith("ssh://")
        or re.fullmatch(r"[^/@:\s]+@[^/:\s]+:.+", remote_url) is not None
    )
    authenticated_transport = uses_ssh or (bool(https_authority) and "@" not in https_authority)
    if not authenticated_transport:
        raise ReplayReleaseWorkflowError(
            "M5 authoritative replay requires an authenticated SSH or HTTPS upstream"
        )
    remote_result = _live_remote_bytes(
        source_root,
        remote=remote,
        merge_ref=merge_ref,
        uses_ssh=uses_ssh,
    )
    expected_remote = f"{revision}\t{merge_ref}\n".encode("ascii")
    if remote_result != expected_remote:
        raise ReplayReleaseWorkflowError(
            "M5 authoritative replay requires the live upstream revision to equal HEAD"
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
    expected_output = Path(f"reports/generated/m5-replay-{run_label}-{revision}-r1")
    if (
        match is None
        or match.group(1) != run_label
        or match.group(2) != revision
        or output_dir != expected_output
    ):
        raise ReplayReleaseWorkflowError(
            "M5 replay output is not the frozen r1 attempt for its label and revision"
        )
    expected_log = output_dir.with_name(f"{output_dir.name}.time-l.txt")
    if time_l_output != expected_log:
        raise ReplayReleaseWorkflowError("M5 replay timing log does not match its output attempt")
    return output_dir.as_posix(), time_l_output.as_posix()


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_existing_directory(path: Path) -> int:
    if not path.is_absolute():
        raise OSError("M5 authority directory is not absolute")
    descriptor = os.open(path.anchor, _directory_open_flags())
    try:
        for component in path.parts[1:]:
            child = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _stable_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_stable_regular_file(
    path: Path,
    *,
    byte_cap: int,
    label: str,
    require_private: bool,
) -> tuple[bytes, os.stat_result]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_descriptor: int | None = None
    descriptor: int | None = None
    try:
        parent_descriptor = _open_existing_directory(absolute.parent)
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > byte_cap
            or (require_private and stat.S_IMODE(before.st_mode) != 0o600)
        ):
            raise OSError(f"M5 {label} is not a bounded private regular file")
        chunks: list[bytes] = []
        remaining = byte_cap + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        value = b"".join(chunks)
        after = os.fstat(descriptor)
        reopened = os.stat(absolute.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            len(value) != before.st_size
            or len(value) > byte_cap
            or _stable_identity(before) != _stable_identity(after)
            or _stable_identity(before) != _stable_identity(reopened)
        ):
            raise OSError(f"M5 {label} changed while it was read")
        return value, before
    except OSError as error:
        raise ReplayReleaseWorkflowError(f"M5 {label} is unavailable or unsafe") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def _authenticated_executable(
    path: Path,
    *,
    label: str,
) -> tuple[str, ReplayExecutableFingerprint]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        value, metadata = _read_stable_regular_file(
            absolute,
            byte_cap=_EXECUTABLE_BYTE_CAP,
            label=f"{label} executable",
            require_private=False,
        )
        resolved = absolute.resolve(strict=True)
    except (OSError, ReplayReleaseWorkflowError) as error:
        raise ReplayReleaseWorkflowError(f"M5 {label} executable is unavailable") from error
    if resolved != absolute or metadata.st_mode & 0o111 == 0 or not os.access(absolute, os.X_OK):
        raise ReplayReleaseWorkflowError(f"M5 {label} executable is not a safe regular file")
    fingerprint = ReplayExecutableFingerprint(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=metadata.st_mode,
        link_count=metadata.st_nlink,
        owner_uid=metadata.st_uid,
        owner_gid=metadata.st_gid,
        byte_length=metadata.st_size,
        modified_time_ns=metadata.st_mtime_ns,
        changed_time_ns=metadata.st_ctime_ns,
        sha256=hashlib.sha256(value).hexdigest(),
    )
    return os.fspath(absolute), fingerprint


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
    descriptor: int | None = None
    try:
        descriptor = _open_existing_directory(path)
        metadata = os.fstat(descriptor)
        reopened = os.stat(path, follow_symlinks=False)
        resolved = path.resolve(strict=True)
        if (
            resolved != path
            or not stat.S_ISDIR(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (reopened.st_dev, reopened.st_ino)
            or (require_private and stat.S_IMODE(metadata.st_mode) & 0o077 != 0)
        ):
            raise OSError
    except OSError as error:
        raise ReplayReleaseWorkflowError(
            f"M5 replay {label} is not a safe real directory"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return path, (metadata.st_dev, metadata.st_ino)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _software_verification_authority(
    clean: CleanSourceSnapshot,
    implementation: ImplementationSnapshot,
) -> tuple[str, str]:
    relative = Path(f"reports/generated/m5-software-verification-{clean.git_revision}.json")
    try:
        value, _metadata = _read_stable_regular_file(
            clean.source_root / relative,
            byte_cap=_SOFTWARE_VERIFICATION_BYTE_CAP,
            label="software verification",
            require_private=True,
        )
        verification = load_software_verification(
            value,
            snapshot=implementation,
            lockfile_sha256=clean.lockfile_sha256,
            package_version=clean.package_version,
        )
    except (OSError, TypeError, ValueError) as error:
        raise ReplayReleaseWorkflowError(
            "M5 fixed revision-specific software verification is absent or stale"
        ) from error
    if canonical_json_bytes(verification) != value:
        raise ReplayReleaseWorkflowError("M5 software verification is noncanonical")
    return relative.as_posix(), hashlib.sha256(value).hexdigest()


def _attempt_arguments(revision: str, run_label: str) -> tuple[str, str, str]:
    output = f"reports/generated/m5-replay-{run_label}-{revision}-r1"
    return output, f"{output}.time-l.txt", f"{output}.success.json"


def _attempt_entries(source_root: Path, revision: str) -> dict[str, os.stat_result]:
    generated = source_root / "reports/generated"
    descriptor: int | None = None
    try:
        descriptor = _open_existing_directory(generated)
        names = os.listdir(descriptor)
        prefixes = (
            f"m5-replay-primary-{revision}-r",
            f"m5-replay-repeat-{revision}-r",
        )
        selected: dict[str, os.stat_result] = {}
        for name in names:
            if name.startswith(prefixes):
                selected[name] = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        return selected
    except OSError as error:
        raise ReplayReleaseWorkflowError(
            "M5 replay lifecycle directory is unavailable or unsafe"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fingerprint_record(value: ReplayExecutableFingerprint) -> dict[str, int | str]:
    return {
        "device": value.device,
        "inode": value.inode,
        "mode": value.mode,
        "link_count": value.link_count,
        "owner_uid": value.owner_uid,
        "owner_gid": value.owner_gid,
        "byte_length": value.byte_length,
        "modified_time_ns": value.modified_time_ns,
        "changed_time_ns": value.changed_time_ns,
        "sha256": value.sha256,
    }


def _environment_record(environment: RuntimeEnvironment) -> object:
    model_dump = getattr(environment, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", by_alias=True)
    return {
        name: getattr(environment, name)
        for name in (
            "python_version",
            "os_name",
            "os_release",
            "machine",
            "cpu_model",
            "logical_cpu_count",
            "memory_bytes",
        )
        if hasattr(environment, name)
    }


def _success_receipt_bytes(
    token: ReplayExecutionAuthority,
    *,
    run_label: str,
) -> bytes:
    output, timing, _success = _attempt_arguments(token.scientific_git_revision, run_label)
    return canonical_json_bytes(
        {
            "schema": "ffb.m5-replay-execution-success/v1",
            "run_label": run_label,
            "scientific_git_revision": token.scientific_git_revision,
            "output_argument": output,
            "time_l_argument": timing,
            "dataset_root_identity": token.dataset_root_identity,
            "uv_cache_root_identity": token.uv_cache_root_identity,
            "lockfile_sha256": token.lockfile_sha256,
            "package_version": token.package_version,
            "implementation_snapshot_sha256": token.implementation_snapshot_sha256,
            "implementation_attestation_sha256": token.implementation_attestation_sha256,
            "software_verification_argument": token.software_verification_argument,
            "software_verification_sha256": token.software_verification_sha256,
            "upstream_ref": token.upstream_ref,
            "environment": _environment_record(token.environment),
            "ffb_executable": token.ffb_executable,
            "ffb_executable_fingerprint": _fingerprint_record(token.ffb_executable_fingerprint),
            "time_executable": token.time_executable,
            "time_executable_fingerprint": _fingerprint_record(token.time_executable_fingerprint),
        }
    )


def _require_output_entry(
    entries: dict[str, os.stat_result],
    *,
    name: str,
) -> None:
    metadata = entries[name]
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ReplayReleaseWorkflowError("M5 replay lifecycle output is not a real directory")


def _require_private_entry(
    source_root: Path,
    entries: dict[str, os.stat_result],
    *,
    name: str,
    byte_cap: int,
    label: str,
) -> bytes:
    value, metadata = _read_stable_regular_file(
        source_root / "reports/generated" / name,
        byte_cap=byte_cap,
        label=label,
        require_private=True,
    )
    if _stable_identity(metadata) != _stable_identity(entries[name]):
        raise ReplayReleaseWorkflowError(f"M5 {label} changed during lifecycle validation")
    return value


def _require_attempt_lifecycle(
    token: ReplayExecutionAuthority,
    *,
    phase: str,
) -> None:
    entries = _attempt_entries(token.source_root, token.scientific_git_revision)
    current = _attempt_arguments(token.scientific_git_revision, token.run_label)
    prior = (
        _attempt_arguments(token.scientific_git_revision, "primary")
        if token.run_label == "repeat"
        else None
    )
    required: set[str] = set(prior or ())
    allowed = set(required)
    if phase == "preflight":
        pass
    elif phase == "postflight":
        required.add(current[1])
        allowed.update((current[0], current[1]))
    elif phase == "successful":
        required.update(current[:2])
        allowed.update(current[:2])
    elif phase == "completed":
        required.update(current)
        allowed.update(current)
    else:
        raise ReplayReleaseWorkflowError("M5 replay lifecycle phase is invalid")
    observed = set(entries)
    if not required.issubset(observed) or not observed.issubset(allowed):
        raise ReplayReleaseWorkflowError(
            "M5 replay lifecycle has a prior, sibling, partial, or out-of-order attempt"
        )
    labels = ("primary",) if prior is not None else ()
    if phase == "completed":
        labels = (*labels, token.run_label)
    for label in labels:
        output, timing, receipt = _attempt_arguments(token.scientific_git_revision, label)
        _require_output_entry(entries, name=Path(output).name)
        _require_private_entry(
            token.source_root,
            entries,
            name=Path(timing).name,
            byte_cap=_TIMING_LOG_BYTE_CAP,
            label=f"{label} timing log",
        )
        receipt_bytes = _require_private_entry(
            token.source_root,
            entries,
            name=Path(receipt).name,
            byte_cap=_SUCCESS_RECEIPT_BYTE_CAP,
            label=f"{label} success receipt",
        )
        if receipt_bytes != _success_receipt_bytes(token, run_label=label):
            raise ReplayReleaseWorkflowError(
                "M5 replay success receipt does not match the current execution authority"
            )
    if phase in {"successful", "completed"}:
        _require_output_entry(entries, name=Path(current[0]).name)
        _require_private_entry(
            token.source_root,
            entries,
            name=Path(current[1]).name,
            byte_cap=_TIMING_LOG_BYTE_CAP,
            label=f"{token.run_label} timing log",
        )


def _execution_authority(
    *,
    source_root: Path,
    run_label: str,
    output_dir: Path,
    time_l_output: Path,
    lifecycle_phase: str = "preflight",
) -> ReplayExecutionAuthority:
    clean, implementation = _clean_authority(source_root)
    _report, attestation = _read_review_authority(clean.source_root, implementation)
    software_argument, software_sha256 = _software_verification_authority(clean, implementation)
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
    ffb_executable, ffb_fingerprint = _authenticated_executable(Path(discovered_ffb), label="ffb")
    if Path(ffb_executable) != expected_ffb.resolve(strict=True):
        raise ReplayReleaseWorkflowError("M5 ffb executable is outside the locked environment")
    time_executable, time_fingerprint = _authenticated_executable(
        Path("/usr/bin/time"), label="Darwin time"
    )
    token = ReplayExecutionAuthority(
        source_root=clean.source_root,
        run_label=run_label,
        output_argument=output_argument,
        time_l_argument=time_argument,
        success_argument=f"{output_argument}.success.json",
        dataset_root_identity=dataset_identity,
        uv_cache_root_identity=cache_identity,
        scientific_git_revision=clean.git_revision,
        lockfile_sha256=clean.lockfile_sha256,
        package_version=clean.package_version,
        implementation_snapshot_sha256=implementation.sha256,
        implementation_attestation_sha256=hashlib.sha256(attestation).hexdigest(),
        software_verification_argument=software_argument,
        software_verification_sha256=software_sha256,
        upstream_ref=_require_upstream_sync(clean.source_root, clean.git_revision),
        environment=environment,
        ffb_executable=ffb_executable,
        ffb_executable_fingerprint=ffb_fingerprint,
        time_executable=time_executable,
        time_executable_fingerprint=time_fingerprint,
    )
    _require_attempt_lifecycle(token, phase=lifecycle_phase)
    return token


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
        lifecycle_phase="postflight",
    )
    if observed != token:
        raise ReplayReleaseWorkflowError("M5 replay execution authority changed during the run")


def build_replay_execution_success_receipt(
    *,
    token: ReplayExecutionAuthority,
) -> ReplayExecutionSuccessReceipt:
    """Build the exclusive receipt only after a complete successful r1 attempt."""

    _require_attempt_lifecycle(token, phase="successful")
    return ReplayExecutionSuccessReceipt(
        path=Path(token.success_argument),
        value=_success_receipt_bytes(token, run_label=token.run_label),
    )


def verify_replay_execution_success_receipt(
    *,
    token: ReplayExecutionAuthority,
) -> None:
    """Reload the just-published receipt and close the lifecycle transition."""

    _require_attempt_lifecycle(token, phase="completed")


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


def _require_frozen_candidate_inputs(
    *,
    revision: str,
    primary_artifact: Path,
    repeat_artifact: Path,
    primary_time_l: Path,
    repeat_time_l: Path,
    software_verification: Path,
) -> None:
    primary, primary_time, _primary_receipt = _attempt_arguments(revision, "primary")
    repeat, repeat_time, _repeat_receipt = _attempt_arguments(revision, "repeat")
    observed = (
        primary_artifact,
        repeat_artifact,
        primary_time_l,
        repeat_time_l,
        software_verification,
    )
    expected = (
        Path(primary),
        Path(repeat),
        Path(primary_time),
        Path(repeat_time),
        Path(f"reports/generated/m5-software-verification-{revision}.json"),
    )
    if observed != expected:
        raise ReplayReleaseWorkflowError(
            "M5 review candidate inputs are not the frozen primary/repeat r1 evidence"
        )


def _authenticate_completed_replays(
    *,
    source_root: Path,
    revision: str,
) -> ReplayExecutionAuthority:
    repeat, repeat_time, _receipt = _attempt_arguments(revision, "repeat")
    token = _execution_authority(
        source_root=source_root,
        run_label="repeat",
        output_dir=Path(repeat),
        time_l_output=Path(repeat_time),
        lifecycle_phase="completed",
    )
    if token.scientific_git_revision != revision:
        raise ReplayReleaseWorkflowError(
            "M5 completed replay lifecycle changed scientific revision"
        )
    return token


def prepare_review_candidate(
    *,
    primary_artifact: Path,
    repeat_artifact: Path,
    primary_time_l: Path,
    repeat_time_l: Path,
    software_verification: Path,
    output_dir: Path,
    source_root: Path,
) -> LoadedReplayReviewCandidate:
    from fusion_fault_bench.replay_release_candidate import (
        load_validated_review_candidate as load_candidate,
    )
    from fusion_fault_bench.replay_release_candidate import (
        prepare_review_candidate as prepare_candidate,
    )

    clean, implementation = _clean_authority(source_root)
    report, attestation = _read_review_authority(clean.source_root, implementation)
    _authenticate_completed_replays(
        source_root=clean.source_root,
        revision=clean.git_revision,
    )
    _require_frozen_candidate_inputs(
        revision=clean.git_revision,
        primary_artifact=primary_artifact,
        repeat_artifact=repeat_artifact,
        primary_time_l=primary_time_l,
        repeat_time_l=repeat_time_l,
        software_verification=software_verification,
    )
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


def load_validated_review_candidate(
    *, path: Path, source_root: Path
) -> LoadedReplayReviewCandidate:
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
        clean, _implementation = _clean_authority(source_root)
        _authenticate_completed_replays(
            source_root=clean.source_root,
            revision=clean.git_revision,
        )
        _require_frozen_candidate_inputs(
            revision=clean.git_revision,
            primary_artifact=primary_artifact,
            repeat_artifact=repeat_artifact,
            primary_time_l=primary_time_l,
            repeat_time_l=repeat_time_l,
            software_verification=software_verification,
        )
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
    from fusion_fault_bench.replay_publication_authority import (
        authenticate_pending_publication,
        require_scientific_revision_ancestor,
        validate_current_implementation_review,
        verify_pending_publication_unchanged,
    )
    from fusion_fault_bench.replay_release_package import (
        validate_publication as validate_repository_publication,
    )
    from fusion_fault_bench.replay_release_package import (
        validate_release_package as validate_package,
    )

    root = Path(os.path.abspath(os.fspath(source_root)))
    expected_release = root / M5_RELEASE_DESTINATION_PATH
    observed_release = Path(
        os.path.abspath(os.fspath(release if release.is_absolute() else root / release))
    )
    if observed_release != expected_release:
        raise ReplayReleaseWorkflowError(
            "M5 publication validation requires the frozen tracked package path"
        )

    try:
        validated_before = validate_package(observed_release)
        scientific_revision = validated_before.artifact.run.git_revision
        clean: CleanSourceSnapshot | None = None
        pending = None
        try:
            clean, implementation = _clean_authority(root)
        except ReplayReleaseWorkflowError:
            pending = authenticate_pending_publication(
                root,
                scientific_git_revision=scientific_revision,
            )
            implementation = build_implementation_snapshot(root)
            if implementation.scientific_git_revision != scientific_revision:
                raise ReplayReleaseWorkflowError(
                    "M5 pending publication changed scientific revision"
                ) from None
            report, attestation = _read_review_authority(root, implementation)
        else:
            require_scientific_revision_ancestor(
                clean.source_root,
                scientific_revision,
            )
            report, attestation = _read_review_authority(
                clean.source_root,
                implementation,
            )

        validate_current_implementation_review(
            validated_before,
            snapshot=implementation,
            tracked_report=report,
            tracked_attestation=attestation,
        )
        digest = _validated_digest(
            validate_repository_publication(observed_release, root),
            subject="publication",
        )
        validated_after = validate_package(observed_release)
        final_digest = _validated_digest(
            validated_after.release_package_sha256,
            subject="release package",
        )
        if (
            digest
            != _validated_digest(
                validated_before.release_package_sha256,
                subject="release package",
            )
            or final_digest != digest
        ):
            raise ReplayReleaseWorkflowError("M5 publication package changed during validation")
        validate_current_implementation_review(
            validated_after,
            snapshot=implementation,
            tracked_report=report,
            tracked_attestation=attestation,
        )

        if pending is not None:
            verify_pending_publication_unchanged(pending)
        else:
            if clean is None:
                raise ReplayReleaseWorkflowError("M5 clean publication authority is unavailable")
            observed_clean, observed_implementation = _clean_authority(root)
            if observed_clean != clean or observed_implementation != implementation:
                raise ReplayReleaseWorkflowError(
                    "M5 source authority changed during publication validation"
                )
            require_scientific_revision_ancestor(
                observed_clean.source_root,
                scientific_revision,
            )
            observed_report, observed_attestation = _read_review_authority(
                observed_clean.source_root,
                observed_implementation,
            )
            if observed_report != report or observed_attestation != attestation:
                raise ReplayReleaseWorkflowError(
                    "M5 implementation review authority changed during publication validation"
                )
        return digest
    except ReplayReleaseWorkflowError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ReplayReleaseWorkflowError("M5 publication validation failed closed") from error


__all__ = [
    "ReplayExecutableFingerprint",
    "ReplayExecutionAuthority",
    "ReplayExecutionSuccessReceipt",
    "ReplayReleaseWorkflowError",
    "authenticate_replay_execution",
    "build_replay_execution_success_receipt",
    "build_reviewed_release",
    "load_validated_review_candidate",
    "prepare_review_candidate",
    "validate_publication",
    "validate_release_package",
    "validate_review_candidate",
    "verify_replay_execution_success_receipt",
    "verify_replay_execution_unchanged",
    "verify_software",
]
