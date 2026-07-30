"""Operate the preregistered Fusion Fault Bench M5 release pipeline.

This repository-local driver deliberately contains only command orchestration.
Scientific curation, review binding, publication, and validation live in
``fusion_fault_bench.replay_release`` so the installed package and this tool
share one authority.
"""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Never

from fusion_fault_bench import replay_release as release_api

_GENERATED_ROOT = Path("reports/generated")
_TIME_TOOL = "/usr/bin/time"
_TIME_LOG_BYTES_MAX = 65_536
_PRIVATE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


class M5ReleaseToolError(ValueError):
    """A sanitized M5 release-tool failure."""


class _ReplayProcessExit(M5ReleaseToolError):
    """A timed replay returned a nonzero process status."""

    def __init__(self, returncode: int) -> None:
        super().__init__("timed replay process failed")
        if 0 < returncode <= 255:
            self.returncode = returncode
        elif -127 <= returncode < 0:
            self.returncode = 128 - returncode
        else:
            self.returncode = 2


class _SanitizedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise M5ReleaseToolError("invalid M5 release command arguments")


def _normalized_repository_path(value: Path, *, label: str) -> Path:
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise M5ReleaseToolError(f"{label} must be normalized and repository-relative")
    return value


def _generated_destination(value: Path, *, source_root: Path, label: str) -> Path:
    relative = _normalized_repository_path(value, label=label)
    try:
        generated_relative = relative.relative_to(_GENERATED_ROOT)
    except ValueError:
        raise M5ReleaseToolError(f"{label} must remain under reports/generated") from None
    if not generated_relative.parts:
        raise M5ReleaseToolError(f"{label} must name a generated member")
    return source_root / relative


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _authenticate_directory(path: Path, descriptor: int, *, label: str) -> None:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise M5ReleaseToolError(f"{label} parent cannot be reauthenticated") from error
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_ISLNK(named.st_mode)
        or not stat.S_ISDIR(named.st_mode)
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise M5ReleaseToolError(f"{label} parent must remain one real directory")


def _open_real_parent(path: Path, *, source_root: Path, label: str) -> int:
    relative = path.relative_to(source_root)
    try:
        current = os.open(source_root, _directory_flags())
    except OSError as error:
        raise M5ReleaseToolError(f"{label} parent must be a real directory") from error
    current_path = source_root
    try:
        _authenticate_directory(current_path, current, label=label)
        for part in relative.parent.parts:
            child = os.open(part, _directory_flags(), dir_fd=current)
            os.close(current)
            current = child
            current_path /= part
            _authenticate_directory(current_path, current, label=label)
        return current
    except OSError as error:
        os.close(current)
        raise M5ReleaseToolError(f"{label} parent must be a real directory") from error
    except BaseException:
        os.close(current)
        raise


def _require_absent_at(parent: int, name: str, *, label: str) -> None:
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as error:
        raise M5ReleaseToolError(f"{label} destination cannot be inspected safely") from error
    raise M5ReleaseToolError(f"{label} destination must not already exist")


def _reserve_time_log(parent: int, name: str) -> int:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, _PRIVATE_FILE_MODE, dir_fd=parent)
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        os.fsync(parent)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise M5ReleaseToolError("timing log cannot be reserved safely") from error
    return descriptor


def _authenticate_reserved_log(parent: int, name: str, descriptor: int) -> None:
    try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as error:
        raise M5ReleaseToolError("timing log cannot be reauthenticated") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) != _PRIVATE_FILE_MODE
        or not 0 < opened.st_size <= _TIME_LOG_BYTES_MAX
        or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise M5ReleaseToolError("timing log is not one complete private regular file")


def _authenticate_run_output(parent: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as error:
        raise M5ReleaseToolError("replay command did not publish its declared output") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise M5ReleaseToolError("replay command did not publish a real output directory")


def run_replay(
    *,
    run_label: str,
    output_dir: Path,
    time_l_output: Path,
    source_root: Path | None = None,
) -> None:
    """Run one separately timed replay without accepting or echoing dataset paths."""

    if run_label not in {"primary", "repeat"}:
        raise M5ReleaseToolError("run label must be primary or repeat")
    root = Path.cwd() if source_root is None else source_root
    try:
        root = root.resolve(strict=True)
    except OSError as error:
        raise M5ReleaseToolError("repository root cannot be resolved") from error
    output_path = _generated_destination(
        output_dir,
        source_root=root,
        label="replay output",
    )
    log_path = _generated_destination(
        time_l_output,
        source_root=root,
        label="timing log",
    )
    if output_path == log_path:
        raise M5ReleaseToolError("replay output and timing log must be distinct")

    output_parent = _open_real_parent(
        output_path,
        source_root=root,
        label="replay output",
    )
    try:
        log_parent = _open_real_parent(
            log_path,
            source_root=root,
            label="timing log",
        )
        try:
            _require_absent_at(output_parent, output_path.name, label="replay output")
            _require_absent_at(log_parent, log_path.name, label="timing log")
            descriptor = _reserve_time_log(log_parent, log_path.name)
            try:
                _authenticate_directory(
                    output_path.parent,
                    output_parent,
                    label="replay output",
                )
                _authenticate_directory(
                    log_path.parent,
                    log_parent,
                    label="timing log",
                )
                command = (
                    _TIME_TOOL,
                    "-l",
                    "-o",
                    f"/dev/fd/{descriptor}",
                    "ffb",
                    "replay",
                    "run",
                    "--output-dir",
                    output_dir.as_posix(),
                )
                try:
                    completed = subprocess.run(
                        command,
                        cwd=root,
                        check=False,
                        pass_fds=(descriptor,),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except OSError as error:
                    raise M5ReleaseToolError(
                        "timed replay process could not be launched"
                    ) from error
                try:
                    os.fsync(descriptor)
                except OSError as error:
                    raise M5ReleaseToolError("timing log could not be made durable") from error
                _authenticate_reserved_log(log_parent, log_path.name, descriptor)
                _authenticate_directory(
                    log_path.parent,
                    log_parent,
                    label="timing log",
                )
                os.fsync(log_parent)
                if completed.returncode != 0:
                    raise _ReplayProcessExit(completed.returncode)
                _authenticate_directory(
                    output_path.parent,
                    output_parent,
                    label="replay output",
                )
                _authenticate_run_output(output_parent, output_path.name)
                os.fsync(output_parent)
            finally:
                os.close(descriptor)
        finally:
            os.close(log_parent)
    finally:
        os.close(output_parent)


def _local_inputs(args: argparse.Namespace) -> release_api.M5ReleaseLocalInputs:
    return release_api.M5ReleaseLocalInputs(
        primary_artifact=args.primary_artifact,
        repeat_artifact=args.repeat_artifact,
        primary_time_l=args.primary_time_l,
        repeat_time_l=args.repeat_time_l,
        software_verification=args.software_verification,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = _SanitizedArgumentParser(
        description="Operate the preregistered Fusion Fault Bench M5 release pipeline."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    implementation_review = commands.add_parser("attest-implementation-review")
    implementation_review.add_argument("--review-report", type=Path, required=True)
    implementation_review.add_argument("--decision", type=Path, required=True)
    implementation_review.add_argument("--output", type=Path, required=True)

    software = commands.add_parser("verify-software")
    software.add_argument("--output", type=Path, required=True)

    replay = commands.add_parser("run-replay")
    replay.add_argument("--run-label", choices=("primary", "repeat"), required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--time-l-output", type=Path, required=True)

    prepare = commands.add_parser("prepare-review")
    prepare.add_argument("--primary-artifact", type=Path, required=True)
    prepare.add_argument("--repeat-artifact", type=Path, required=True)
    prepare.add_argument("--primary-time-l", type=Path, required=True)
    prepare.add_argument("--repeat-time-l", type=Path, required=True)
    prepare.add_argument("--software-verification", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    validate_candidate = commands.add_parser("validate-review-candidate")
    validate_candidate.add_argument("path", type=Path)

    results_review = commands.add_parser("attest-results-review")
    results_review.add_argument("--candidate", type=Path, required=True)
    results_review.add_argument("--review-report", type=Path, required=True)
    results_review.add_argument("--decision", type=Path, required=True)
    results_review.add_argument("--output", type=Path, required=True)

    build = commands.add_parser("build-release")
    build.add_argument("--candidate", type=Path, required=True)
    build.add_argument("--results-review", type=Path, required=True)
    build.add_argument("--results-review-attestation", type=Path, required=True)
    build.add_argument("--primary-artifact", type=Path, required=True)
    build.add_argument("--repeat-artifact", type=Path, required=True)
    build.add_argument("--primary-time-l", type=Path, required=True)
    build.add_argument("--repeat-time-l", type=Path, required=True)
    build.add_argument("--software-verification", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)

    sync = commands.add_parser("sync-reviewed-evidence")
    sync.add_argument("--release", type=Path, required=True)
    sync.add_argument("--review-report-output", type=Path, required=True)
    sync.add_argument("--review-attestation-output", type=Path, required=True)

    validate = commands.add_parser("validate-release")
    validate.add_argument("path", type=Path)

    publication = commands.add_parser("validate-publication")
    publication.add_argument("--release", type=Path, required=True)
    publication.add_argument("--source-root", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> None:
    source_root = Path.cwd()
    if args.command == "attest-implementation-review":
        release_api.attest_implementation_review(
            source_root=source_root,
            review_report=args.review_report,
            decision=args.decision,
            output=args.output,
        )
    elif args.command == "verify-software":
        release_api.run_software_verification_checks(
            source_root=source_root,
            output=args.output,
        )
    elif args.command == "run-replay":
        run_replay(
            run_label=args.run_label,
            output_dir=args.output_dir,
            time_l_output=args.time_l_output,
            source_root=source_root,
        )
    elif args.command == "prepare-review":
        release_api.prepare_review_candidate(
            _local_inputs(args),
            output_dir=args.output_dir,
            source_root=source_root,
        )
    elif args.command == "validate-review-candidate":
        release_api.validate_review_candidate(args.path)
    elif args.command == "attest-results-review":
        release_api.attest_results_review(
            candidate=args.candidate,
            review_report=args.review_report,
            decision=args.decision,
            output=args.output,
        )
    elif args.command == "build-release":
        release_api.build_release(
            candidate=args.candidate,
            results_review=args.results_review,
            results_review_attestation=args.results_review_attestation,
            inputs=_local_inputs(args),
            output_dir=args.output_dir,
            source_root=source_root,
        )
    elif args.command == "sync-reviewed-evidence":
        release_api.sync_reviewed_evidence(
            release=args.release,
            review_report_output=args.review_report_output,
            review_attestation_output=args.review_attestation_output,
        )
    elif args.command == "validate-release":
        release_api.validate_release(args.path)
    else:
        release_api.validate_publication(
            release=args.release,
            source_root=args.source_root,
        )


def main(argv: list[str] | None = None) -> int:
    """Run the M5 release driver with path-free success and failure output."""

    try:
        args = _build_parser().parse_args(argv)
        _run(args)
    except _ReplayProcessExit as error:
        print("error: M5 release command failed closed", file=sys.stderr)
        return error.returncode
    except Exception:
        print("error: M5 release command failed closed", file=sys.stderr)
        return 2
    print("M5 release command completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
