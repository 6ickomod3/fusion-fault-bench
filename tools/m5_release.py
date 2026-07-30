"""Drive the frozen M5 review and release workflow.

The command layer deliberately contains no scientific selection logic.  It
either delegates to the source-authenticating workflow facade or performs one
of the two mechanical review canonicalizations and the exact replay wrapper
frozen in ``docs/m5-release-pipeline-plan.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import selectors
import signal
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from types import ModuleType

from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.replay_release import (
    attest_results_review,
    load_release_package,
    sync_reviewed_evidence,
)
from fusion_fault_bench.replay_release_authority import build_implementation_snapshot
from fusion_fault_bench.replay_release_validation import (
    build_implementation_review_attestation,
    load_implementation_review_decision,
    load_results_review_decision,
)

_GENERATED_ROOT_PARTS = ("reports", "generated")
_TIME_EXECUTABLE = "/usr/bin/time"
_READ_CHUNK_BYTES = 1024 * 1024
_REVIEW_REPORT_BYTE_CAP = 1024 * 1024
_DECISION_BYTE_CAP = 1024 * 1024
_CHILD_OUTPUT_BYTE_CAP = 64 * 1024
_CHILD_OUTPUT_READ_BYTES = 8192
_DIGEST_ATTRIBUTES = (
    "candidate_sha256",
    "release_package_sha256",
    "sha256",
)


class M5ReleaseDriverError(ValueError):
    """The repository-local M5 release command failed closed."""


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _open_existing_directory(path: Path) -> int:
    """Open an absolute directory without following a symlink component."""

    if not path.is_absolute():
        raise M5ReleaseDriverError("M5 path parent must be absolute")
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


def _entry_exists_at(directory_descriptor: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _read_bounded_regular_file(path: Path, *, byte_cap: int, label: str) -> bytes:
    absolute = _absolute_lexical(path)
    parent_descriptor = _open_existing_directory(absolute.parent)
    try:
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > byte_cap
            ):
                raise M5ReleaseDriverError(f"{label} is not a bounded private regular file")
            chunks: list[bytes] = []
            remaining = byte_cap + 1
            while remaining:
                chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            value = b"".join(chunks)
            after = os.fstat(descriptor)
            reopened = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if (
                len(value) > byte_cap
                or any(getattr(before, field) != getattr(after, field) for field in identity_fields)
                or any(
                    getattr(before, field) != getattr(reopened, field) for field in identity_fields
                )
            ):
                raise M5ReleaseDriverError(f"{label} changed while it was read")
            return value
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


def _write_exclusive_file(path: Path, value: bytes) -> None:
    """Write one canonical public record without following or replacing paths."""

    absolute = _absolute_lexical(path)
    parent_descriptor = _open_existing_directory(absolute.parent)
    descriptor: int | None = None
    try:
        if _entry_exists_at(parent_descriptor, absolute.name):
            raise FileExistsError("M5 release output already exists")
        descriptor = os.open(
            absolute.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        remaining = memoryview(value)
        while remaining:
            written = os.write(descriptor, remaining)
            if written == 0:
                raise OSError("short write while publishing M5 release record")
            remaining = remaining[written:]
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        reopened = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or (observed.st_dev, observed.st_ino, observed.st_size)
            != (reopened.st_dev, reopened.st_ino, reopened.st_size)
            or observed.st_size != len(value)
        ):
            raise M5ReleaseDriverError("M5 release output changed during publication")
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _workflow_api() -> ModuleType:
    """Load the aggregate-only workflow facade only for commands that need it."""

    try:
        return importlib.import_module("fusion_fault_bench.replay_release_workflow")
    except ImportError as error:
        raise M5ReleaseDriverError("M5 release workflow facade is unavailable") from error


def _call_workflow(name: str, /, **arguments: object) -> object:
    function = getattr(_workflow_api(), name, None)
    if not callable(function):
        raise M5ReleaseDriverError("M5 release workflow operation is unavailable")
    return function(**arguments)


def _result_digest(result: object) -> str | None:
    if isinstance(result, str):
        candidate = result
    else:
        candidate = ""
        for attribute in _DIGEST_ATTRIBUTES:
            value = getattr(result, attribute, None)
            if isinstance(value, str):
                candidate = value
                break
    if len(candidate) == 64 and all(character in "0123456789abcdef" for character in candidate):
        return candidate
    return None


def _print_success(command: str, result: object | None = None) -> None:
    digest = _result_digest(result) if result is not None else None
    suffix = f" sha256={digest}" if digest is not None else ""
    print(f"ok: {command}{suffix}")


def attest_implementation_review_command(args: argparse.Namespace) -> object:
    report = _read_bounded_regular_file(
        args.review_report,
        byte_cap=_REVIEW_REPORT_BYTE_CAP,
        label="implementation review report",
    )
    decision_bytes = _read_bounded_regular_file(
        args.decision,
        byte_cap=_DECISION_BYTE_CAP,
        label="implementation review decision",
    )
    snapshot = build_implementation_snapshot(args.source_root)
    decision = load_implementation_review_decision(decision_bytes)
    attestation = build_implementation_review_attestation(
        decision,
        review_report=report,
        snapshot=snapshot,
    )
    value = canonical_json_bytes(attestation)
    _write_exclusive_file(args.output, value)
    return hashlib.sha256(value).hexdigest()


def verify_software_command(args: argparse.Namespace) -> object:
    return _call_workflow(
        "verify_software",
        source_root=args.source_root,
        output=args.output,
    )


def _require_generated_path(path: Path, *, label: str) -> tuple[str, Path]:
    raw = os.fspath(path)
    if path.is_absolute() or path.as_posix() != raw:
        raise M5ReleaseDriverError(f"{label} must be a normalized repository-relative path")
    if (
        len(path.parts) <= len(_GENERATED_ROOT_PARTS)
        or tuple(path.parts[:2]) != _GENERATED_ROOT_PARTS
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise M5ReleaseDriverError(f"{label} must be below reports/generated")
    return raw, _absolute_lexical(path)


def _kill_child_group(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def _run_bounded_child(command: tuple[str, ...], *, pass_fds: tuple[int, ...]) -> int:
    """Run one child while draining both output streams into fixed-size buffers."""

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=pass_fds,
        start_new_session=True,
    )
    if process.stdout is None or process.stderr is None:
        _kill_child_group(process)
        process.wait()
        raise M5ReleaseDriverError("M5 replay child output pipes are unavailable")

    output = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = False
    selector = selectors.DefaultSelector()
    streams = {
        process.stdout.fileno(): ("stdout", process.stdout),
        process.stderr.fileno(): ("stderr", process.stderr),
    }
    try:
        for descriptor in streams:
            selector.register(descriptor, selectors.EVENT_READ)
        while selector.get_map():
            events = selector.select(timeout=1.0)
            if not events and process.poll() is not None:
                continue
            for key, _mask in events:
                descriptor = int(key.fd)
                chunk = os.read(descriptor, _CHILD_OUTPUT_READ_BYTES)
                if not chunk:
                    selector.unregister(descriptor)
                    streams[descriptor][1].close()
                    continue
                name = streams[descriptor][0]
                remaining = _CHILD_OUTPUT_BYTE_CAP - len(output[name])
                if len(chunk) > remaining:
                    output[name].extend(chunk[: max(remaining, 0)])
                    overflow = True
                    if process.poll() is None:
                        _kill_child_group(process)
                elif not overflow:
                    output[name].extend(chunk)
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            _kill_child_group(process)
        process.wait()
        raise
    finally:
        selector.close()
        for _name, stream in streams.values():
            if not stream.closed:
                stream.close()
    if overflow:
        raise M5ReleaseDriverError("M5 replay child output exceeded its byte cap")
    return returncode


def run_replay_command(args: argparse.Namespace) -> int:
    """Run exactly one timed replay attempt; never delete, overwrite, or retry."""

    output_argument, output_absolute = _require_generated_path(
        args.output_dir,
        label="replay output",
    )
    _log_argument, log_absolute = _require_generated_path(
        args.time_l_output,
        label="replay timing log",
    )
    if output_absolute == log_absolute or output_absolute in log_absolute.parents:
        raise M5ReleaseDriverError("replay output and timing log paths overlap")

    execution_token = _call_workflow(
        "authenticate_replay_execution",
        source_root=args.source_root,
        run_label=args.run_label,
        output_dir=args.output_dir,
        time_l_output=args.time_l_output,
    )
    time_executable = getattr(execution_token, "time_executable", None)
    ffb_executable = getattr(execution_token, "ffb_executable", None)
    if (
        time_executable != _TIME_EXECUTABLE
        or not isinstance(ffb_executable, str)
        or not Path(ffb_executable).is_absolute()
    ):
        raise M5ReleaseDriverError("M5 replay execution authority is invalid")

    output_parent_descriptor = _open_existing_directory(output_absolute.parent)
    try:
        if _entry_exists_at(output_parent_descriptor, output_absolute.name):
            raise FileExistsError("M5 replay output destination already exists")
    finally:
        os.close(output_parent_descriptor)

    log_parent_descriptor = _open_existing_directory(log_absolute.parent)
    log_descriptor: int | None = None
    try:
        if _entry_exists_at(log_parent_descriptor, log_absolute.name):
            raise FileExistsError("M5 replay timing log already exists")
        log_descriptor = os.open(
            log_absolute.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=log_parent_descriptor,
        )
        os.fchmod(log_descriptor, 0o600)
        reserved = os.fstat(log_descriptor)
        if (
            not stat.S_ISREG(reserved.st_mode)
            or reserved.st_nlink != 1
            or stat.S_IMODE(reserved.st_mode) != 0o600
        ):
            raise M5ReleaseDriverError("M5 replay timing log reservation is invalid")

        command = (
            time_executable,
            "-l",
            "-o",
            f"/dev/fd/{log_descriptor}",
            ffb_executable,
            "replay",
            "run",
            "--output-dir",
            output_argument,
        )
        child_attempted = False
        try:
            child_attempted = True
            returncode = _run_bounded_child(command, pass_fds=(log_descriptor,))
            os.fsync(log_descriptor)
            final = os.fstat(log_descriptor)
            reopened = os.stat(
                log_absolute.name,
                dir_fd=log_parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(final.st_mode)
                or final.st_nlink != 1
                or stat.S_IMODE(final.st_mode) != 0o600
                or final.st_size == 0
                or (final.st_dev, final.st_ino, final.st_size)
                != (reopened.st_dev, reopened.st_ino, reopened.st_size)
            ):
                raise M5ReleaseDriverError("M5 replay timing log is incomplete")
            os.fsync(log_parent_descriptor)
        finally:
            if child_attempted:
                _call_workflow(
                    "verify_replay_execution_unchanged",
                    token=execution_token,
                    source_root=args.source_root,
                    run_label=args.run_label,
                    output_dir=args.output_dir,
                    time_l_output=args.time_l_output,
                )
        return returncode
    finally:
        if log_descriptor is not None:
            os.close(log_descriptor)
        os.close(log_parent_descriptor)


def prepare_review_command(args: argparse.Namespace) -> object:
    return _call_workflow(
        "prepare_review_candidate",
        primary_artifact=args.primary_artifact,
        repeat_artifact=args.repeat_artifact,
        primary_time_l=args.primary_time_l,
        repeat_time_l=args.repeat_time_l,
        software_verification=args.software_verification,
        output_dir=args.output_dir,
        source_root=args.source_root,
    )


def validate_review_candidate_command(args: argparse.Namespace) -> object:
    return _call_workflow(
        "validate_review_candidate",
        path=args.path,
        source_root=args.source_root,
    )


def attest_results_review_command(args: argparse.Namespace) -> object:
    candidate = _call_workflow(
        "load_validated_review_candidate",
        path=args.candidate,
        source_root=args.source_root,
    )
    report = _read_bounded_regular_file(
        args.review_report,
        byte_cap=_REVIEW_REPORT_BYTE_CAP,
        label="results review report",
    )
    decision_bytes = _read_bounded_regular_file(
        args.decision,
        byte_cap=_DECISION_BYTE_CAP,
        label="results review decision",
    )
    decision = load_results_review_decision(decision_bytes)
    attestation = attest_results_review(
        candidate,
        review_report=report,
        decision=decision,
    )
    value = canonical_json_bytes(attestation)
    _write_exclusive_file(args.output, value)
    return hashlib.sha256(value).hexdigest()


def build_release_command(args: argparse.Namespace) -> object:
    return _call_workflow(
        "build_reviewed_release",
        candidate=args.candidate,
        results_review=args.results_review,
        results_review_attestation=args.results_review_attestation,
        primary_artifact=args.primary_artifact,
        repeat_artifact=args.repeat_artifact,
        primary_time_l=args.primary_time_l,
        repeat_time_l=args.repeat_time_l,
        software_verification=args.software_verification,
        output_dir=args.output_dir,
        source_root=args.source_root,
    )


def sync_reviewed_evidence_command(args: argparse.Namespace) -> None:
    package = load_release_package(args.release)
    sync_reviewed_evidence(
        package,
        report_output=args.review_report_output,
        attestation_output=args.review_attestation_output,
        source_root=args.source_root,
    )


def validate_release_command(args: argparse.Namespace) -> object:
    return _call_workflow("validate_release_package", path=args.path)


def validate_publication_command(args: argparse.Namespace) -> object:
    return _call_workflow(
        "validate_publication",
        release=args.release,
        source_root=args.source_root,
    )


def _add_source_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-root", type=Path, default=Path("."))


def _add_replay_authorities(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--primary-artifact", type=Path, required=True)
    parser.add_argument("--repeat-artifact", type=Path, required=True)
    parser.add_argument("--primary-time-l", type=Path, required=True)
    parser.add_argument("--repeat-time-l", type=Path, required=True)
    parser.add_argument("--software-verification", type=Path, required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    implementation = commands.add_parser("attest-implementation-review")
    implementation.add_argument("--review-report", type=Path, required=True)
    implementation.add_argument("--decision", type=Path, required=True)
    implementation.add_argument("--output", type=Path, required=True)
    _add_source_root(implementation)

    software = commands.add_parser("verify-software")
    software.add_argument("--output", type=Path, required=True)
    _add_source_root(software)

    run = commands.add_parser("run-replay")
    run.add_argument("--run-label", choices=("primary", "repeat"), required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--time-l-output", type=Path, required=True)

    _add_source_root(run)
    prepare = commands.add_parser("prepare-review")
    _add_replay_authorities(prepare)
    prepare.add_argument("--output-dir", type=Path, required=True)
    _add_source_root(prepare)

    candidate = commands.add_parser("validate-review-candidate")
    candidate.add_argument("path", type=Path)
    _add_source_root(candidate)

    results = commands.add_parser("attest-results-review")
    results.add_argument("--candidate", type=Path, required=True)
    results.add_argument("--review-report", type=Path, required=True)
    results.add_argument("--decision", type=Path, required=True)
    results.add_argument("--output", type=Path, required=True)
    _add_source_root(results)

    build = commands.add_parser("build-release")
    build.add_argument("--candidate", type=Path, required=True)
    build.add_argument("--results-review", type=Path, required=True)
    build.add_argument("--results-review-attestation", type=Path, required=True)
    _add_replay_authorities(build)
    build.add_argument("--output-dir", type=Path, required=True)
    _add_source_root(build)

    sync = commands.add_parser("sync-reviewed-evidence")
    sync.add_argument("--release", type=Path, required=True)
    sync.add_argument("--review-report-output", type=Path, required=True)
    sync.add_argument("--review-attestation-output", type=Path, required=True)
    _add_source_root(sync)

    release = commands.add_parser("validate-release")
    release.add_argument("path", type=Path)

    publication = commands.add_parser("validate-publication")
    publication.add_argument("--release", type=Path, required=True)
    publication.add_argument("--source-root", type=Path, required=True)
    return parser


_COMMANDS: dict[str, Callable[[argparse.Namespace], object]] = {
    "attest-implementation-review": attest_implementation_review_command,
    "verify-software": verify_software_command,
    "prepare-review": prepare_review_command,
    "validate-review-candidate": validate_review_candidate_command,
    "attest-results-review": attest_results_review_command,
    "build-release": build_release_command,
    "sync-reviewed-evidence": sync_reviewed_evidence_command,
    "validate-release": validate_release_command,
    "validate-publication": validate_publication_command,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one repository-local M5 release operation."""

    args = _build_parser().parse_args(argv)
    if args.command == "run-replay":
        try:
            exit_status = run_replay_command(args)
        except Exception:
            print("error: run-replay failed closed", file=sys.stderr)
            return 2
        if exit_status == 0:
            _print_success(args.command)
        return exit_status
    try:
        result = _COMMANDS[args.command](args)
        _print_success(args.command, result)
    except Exception:
        print(f"error: {args.command} failed closed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
