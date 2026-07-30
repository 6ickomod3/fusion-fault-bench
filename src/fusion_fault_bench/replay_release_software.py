"""Deterministic, source-bound software verification for the frozen M5 release."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

from fusion_fault_bench import __version__
from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_SOFTWARE_VERIFICATION_CATEGORIES,
    ReplaySoftwareVerificationCheckV1,
    ReplaySoftwareVerificationV1,
)
from fusion_fault_bench.contracts.replay_v1 import M5_REPLAY_INTENT_PATH
from fusion_fault_bench.provenance import (
    CleanSourceSnapshot,
    discover_clean_source,
    verify_locked_execution,
)
from fusion_fault_bench.replay_release_authority import (
    ImplementationSnapshot,
    build_implementation_snapshot,
)
from fusion_fault_bench.replay_release_validation import (
    M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK,
    M5_SOFTWARE_VERIFICATION_CHECK_IDS,
    build_software_verification,
)

type Command = tuple[str, ...]

M5_RUFF_FORMAT_COMMAND: Final[Command] = (
    "uv",
    "run",
    "--frozen",
    "--no-sync",
    "ruff",
    "format",
    "--check",
    ".",
)
M5_RUFF_LINT_COMMAND: Final[Command] = (
    "uv",
    "run",
    "--frozen",
    "--no-sync",
    "ruff",
    "check",
    ".",
)
M5_PYRIGHT_COMMAND: Final[Command] = (
    "uv",
    "run",
    "--frozen",
    "--no-sync",
    "pyright",
)
M5_PYTEST_RELEASE_AUTHORITY_COMMAND: Final[Command] = (
    "uv",
    "run",
    "--frozen",
    "--no-sync",
    "pytest",
)
M5_DISTRIBUTION_BUILD_COMMAND: Final[Command] = (
    "uv",
    "build",
    "--no-sources",
)
M5_BUILT_WHEEL_SMOKE_COMMAND: Final[Command] = (
    "uv",
    "run",
    "--frozen",
    "--no-sync",
    "python",
    "-m",
    "fusion_fault_bench.replay_release_software",
    "built-wheel-smoke",
)
M5_PRIVACY_AUDIT_TEST_NODES: Final[tuple[str, ...]] = (
    "tests/test_replay_runner.py::test_metadata_guard_is_scoped_and_blocks_payload_open",
    "tests/test_replay_runner.py::test_raw_payload_hardlink_alias_attempt_is_counted_and_blocked",
    "tests/test_m5_release_privacy.py",
    "tests/test_m5_methodology_placeholders.py",
)
M5_PRIVACY_AUDIT_COMMAND: Final[Command] = (
    "uv",
    "run",
    "--frozen",
    "--no-sync",
    "pytest",
    "--no-cov",
    "-q",
    *M5_PRIVACY_AUDIT_TEST_NODES,
)

_COMMAND_PAIRS: Final[tuple[tuple[str, Command], ...]] = (
    ("ruff-format", M5_RUFF_FORMAT_COMMAND),
    ("ruff-lint", M5_RUFF_LINT_COMMAND),
    ("pyright", M5_PYRIGHT_COMMAND),
    ("pytest-release-authority", M5_PYTEST_RELEASE_AUTHORITY_COMMAND),
    ("distribution-build", M5_DISTRIBUTION_BUILD_COMMAND),
    ("built-wheel-smoke", M5_BUILT_WHEEL_SMOKE_COMMAND),
    ("privacy-audit", M5_PRIVACY_AUDIT_COMMAND),
)
M5_SOFTWARE_COMMAND_BY_CHECK: Mapping[str, Command] = MappingProxyType(dict(_COMMAND_PAIRS))

_NORMALIZATION = "stable-command-output-with-runtime-paths-and-durations-removed"
_COMMAND_OUTPUT_BYTE_CAP = 32 * 1024 * 1024
_WHEEL_BYTE_CAP = 50 * 1024 * 1024
_GENERATED_RELATIVE = Path("reports/generated")
_PYTHON_VERSION_PATTERN = re.compile(r"3\.12\.[0-9]+")
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_FILE_URI_PATTERN = re.compile(r"file:(?://)?/[^\s\"'<>]+")
_WINDOWS_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>]+")
_POSIX_PATH_PATTERN = re.compile(r"(?<![:/A-Za-z0-9])/(?:[^\s\"'<>:/]+/)*[^\s\"'<>:]*")
_DURATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9.])(?:\d+(?:\.\d+)?\s*(?:ns|us|µs|ms|s|sec|secs|second|seconds|"
    r"min|mins|minute|minutes|h|hr|hrs|hour|hours)|\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)"
    r"(?![A-Za-z0-9.])",
    re.IGNORECASE,
)
_ISO_TIME_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)


class ReplaySoftwareVerificationError(ValueError):
    """The M5 software authority failed, drifted, or could not be published safely."""


def _software_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "NUSCENES_ROOT",
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "UV_OFFLINE": "1",
            "PIP_NO_INDEX": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
            "NO_COLOR": "1",
            "PYTHONHASHSEED": "0",
            "LC_ALL": "C",
            "LANG": "C",
            "TERM": "dumb",
        }
    )
    return environment


def normalize_command_output(
    stdout: bytes,
    stderr: bytes,
    *,
    runtime_paths: Sequence[Path] = (),
) -> bytes:
    """Remove machine-local paths, terminal escapes, and volatile durations."""

    if len(stdout) > _COMMAND_OUTPUT_BYTE_CAP or len(stderr) > _COMMAND_OUTPUT_BYTE_CAP:
        raise ReplaySoftwareVerificationError("M5 software check output exceeds its byte cap")
    try:
        streams = (stdout.decode("utf-8"), stderr.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ReplaySoftwareVerificationError("M5 software check output is not UTF-8") from error

    replacements = sorted(
        {os.fspath(path) for path in runtime_paths if os.fspath(path) not in {"", "."}},
        key=len,
        reverse=True,
    )
    normalized: list[str] = []
    for stream in streams:
        value = stream.replace("\r\n", "\n").replace("\r", "\n")
        value = _ANSI_ESCAPE_PATTERN.sub("", value)
        for path in replacements:
            value = value.replace(path, "<RUNTIME_PATH>")
        value = _FILE_URI_PATTERN.sub("<RUNTIME_PATH>", value)
        value = _WINDOWS_PATH_PATTERN.sub("<RUNTIME_PATH>", value)
        value = _POSIX_PATH_PATTERN.sub("<RUNTIME_PATH>", value)
        value = _ISO_TIME_PATTERN.sub("<TIMESTAMP>", value)
        value = _DURATION_PATTERN.sub("<DURATION>", value)
        normalized.append("\n".join(line.rstrip() for line in value.split("\n")).strip())

    return ("stdout:\n" + normalized[0] + "\nstderr:\n" + normalized[1] + "\n").encode("utf-8")


def _run_command(
    command: Command,
    *,
    source_root: Path,
    environment: Mapping[str, str],
    runtime_paths: Sequence[Path] = (),
) -> bytes:
    try:
        result = subprocess.run(
            command,
            cwd=source_root,
            env=dict(environment),
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ReplaySoftwareVerificationError("M5 software check could not be started") from error
    if result.returncode != 0:
        raise ReplaySoftwareVerificationError("M5 software check returned a failure status")
    return normalize_command_output(
        result.stdout,
        result.stderr,
        runtime_paths=(source_root, *runtime_paths),
    )


def _regular_file_bytes(path: Path, *, byte_cap: int, label: str) -> bytes:
    try:
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > byte_cap
        ):
            raise OSError
        value = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise ReplaySoftwareVerificationError(
            f"M5 {label} is not a bounded regular file"
        ) from error
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if len(value) != before.st_size or any(
        getattr(before, field) != getattr(after, field) for field in identity
    ):
        raise ReplaySoftwareVerificationError(f"M5 {label} changed while it was read")
    return value


def _built_wheel(source_root: Path, package_version: str) -> tuple[Path, bytes]:
    expected_name = f"fusion_fault_bench-{package_version}-py3-none-any.whl"
    dist = source_root / "dist"
    try:
        names = tuple(sorted(path.name for path in dist.iterdir() if path.suffix == ".whl"))
    except OSError as error:
        raise ReplaySoftwareVerificationError("M5 distribution build produced no wheel") from error
    if names != (expected_name,):
        raise ReplaySoftwareVerificationError(
            "M5 distribution build did not produce one exact wheel"
        )
    wheel = dist / expected_name
    return wheel, _regular_file_bytes(wheel, byte_cap=_WHEEL_BYTE_CAP, label="built wheel")


def _generated_root(source_root: Path) -> Path:
    reports = source_root / "reports"
    generated = source_root / _GENERATED_RELATIVE
    try:
        reports_state = reports.lstat()
        if not stat.S_ISDIR(reports_state.st_mode) or reports.is_symlink():
            raise OSError
        generated.mkdir(mode=0o700, exist_ok=True)
        generated_state = generated.lstat()
        if not stat.S_ISDIR(generated_state.st_mode) or generated.is_symlink():
            raise OSError
    except OSError as error:
        raise ReplaySoftwareVerificationError(
            "M5 generated-output parent is not a safe directory"
        ) from error
    return generated


def _remove_owned_temp(path: Path, *, device: int, inode: int) -> None:
    try:
        observed = path.lstat()
        if (
            not stat.S_ISDIR(observed.st_mode)
            or path.is_symlink()
            or (observed.st_dev, observed.st_ino) != (device, inode)
        ):
            raise OSError
        shutil.rmtree(path)
    except OSError as error:
        raise ReplaySoftwareVerificationError(
            "M5 wheel-smoke temporary environment could not be cleaned safely"
        ) from error


def _run_built_wheel_smoke(
    *,
    source_root: Path,
    package_version: str,
    environment: Mapping[str, str],
) -> bytes:
    wheel, wheel_bytes = _built_wheel(source_root, package_version)
    python_pin_bytes = _regular_file_bytes(
        source_root / ".python-version",
        byte_cap=64,
        label="Python version pin",
    )
    python_pin = python_pin_bytes.decode("ascii").strip()
    if _PYTHON_VERSION_PATTERN.fullmatch(python_pin) is None:
        raise ReplaySoftwareVerificationError("M5 Python version pin is invalid")
    generated = _generated_root(source_root)
    temp = Path(tempfile.mkdtemp(prefix=".m5-wheel-smoke-", dir=generated))
    owned = temp.lstat()
    try:
        os.chmod(temp, 0o700)
        temp_relative = temp.relative_to(source_root)
        venv_relative = temp_relative / "venv"
        python_relative = venv_relative / "bin/python"
        ffb_relative = venv_relative / "bin/ffb"
        wheel_relative = wheel.relative_to(source_root)
        commands: tuple[Command, ...] = (
            ("uv", "venv", "--python", python_pin, venv_relative.as_posix()),
            (
                "uv",
                "pip",
                "install",
                "--python",
                python_relative.as_posix(),
                "--offline",
                wheel_relative.as_posix(),
            ),
            (
                python_relative.as_posix(),
                "-c",
                (
                    "import os; import fusion_fault_bench as package; "
                    "from fusion_fault_bench import replay_release_workflow as workflow; "
                    "assert 'NUSCENES_ROOT' not in os.environ; "
                    "assert os.environ.get('UV_OFFLINE') == '1'; "
                    f"assert package.__version__ == {package_version!r}; "
                    "assert callable(workflow.validate_release_package); "
                    "print(package.__version__)"
                ),
            ),
            (ffb_relative.as_posix(), "--version"),
            (ffb_relative.as_posix(), "replay", "--help"),
            (ffb_relative.as_posix(), "schema", "show", "replay-validation"),
            (
                ffb_relative.as_posix(),
                "schema",
                "show",
                "replay-execution-resource-evidence",
            ),
            (ffb_relative.as_posix(), "replay", "release", "--help"),
        )
        output = bytearray(
            f"wheel_sha256={hashlib.sha256(wheel_bytes).hexdigest()}\n".encode("ascii")
        )
        for index, command in enumerate(commands):
            normalized = _run_command(
                command,
                source_root=source_root,
                environment=environment,
                runtime_paths=(temp,),
            )
            output.extend(f"step={index}\n".encode("ascii"))
            output.extend(normalized)
        reloaded_wheel = _regular_file_bytes(
            wheel,
            byte_cap=_WHEEL_BYTE_CAP,
            label="built wheel",
        )
        if hashlib.sha256(reloaded_wheel).digest() != hashlib.sha256(wheel_bytes).digest():
            raise ReplaySoftwareVerificationError("M5 built wheel changed during smoke testing")
        return bytes(output)
    finally:
        _remove_owned_temp(temp, device=owned.st_dev, inode=owned.st_ino)


def _require_source_authority(
    *,
    source_root: Path,
    clean_snapshot: CleanSourceSnapshot,
    implementation_snapshot: ImplementationSnapshot,
) -> None:
    try:
        observed_clean = discover_clean_source(source_root / M5_REPLAY_INTENT_PATH)
        verify_locked_execution(observed_clean)
        observed_implementation = build_implementation_snapshot(source_root)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ReplaySoftwareVerificationError(
            "M5 software verification requires the clean locked source authority"
        ) from error
    if (
        observed_clean != clean_snapshot
        or observed_implementation != implementation_snapshot
        or clean_snapshot.source_root != source_root
        or clean_snapshot.git_revision != implementation_snapshot.scientific_git_revision
    ):
        raise ReplaySoftwareVerificationError("M5 software verification source authority changed")


def _expected_output(
    source_root: Path,
    output: Path,
    *,
    revision: str,
) -> tuple[Path, str]:
    relative = Path(f"reports/generated/m5-software-verification-{revision}.json")
    expected = source_root / relative
    if output.is_absolute():
        if output != expected:
            raise ReplaySoftwareVerificationError(
                "M5 software verification output does not bind the scientific revision"
            )
    elif output.as_posix() != relative.as_posix():
        raise ReplaySoftwareVerificationError(
            "M5 software verification output does not bind the scientific revision"
        )
    return expected, expected.name


def _require_absent(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ReplaySoftwareVerificationError(
            "M5 software verification output state is unavailable"
        ) from error
    raise ReplaySoftwareVerificationError("M5 software verification output already exists")


def _publish_exclusive(path: Path, value: bytes) -> None:
    parent_descriptor = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor: int | None = None
    created_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        os.fchmod(descriptor, 0o600)
        initial = os.fstat(descriptor)
        created_identity = (initial.st_dev, initial.st_ino)
        remaining = memoryview(value)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short M5 software verification write")
            remaining = remaining[written:]
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        reopened = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or stat.S_IMODE(final.st_mode) != 0o600
            or final.st_size != len(value)
            or (final.st_dev, final.st_ino, final.st_size)
            != (reopened.st_dev, reopened.st_ino, reopened.st_size)
        ):
            raise OSError("invalid M5 software verification output")
        os.fsync(parent_descriptor)
        created_identity = None
    except FileExistsError as error:
        raise ReplaySoftwareVerificationError(
            "M5 software verification output already exists"
        ) from error
    except OSError as error:
        if created_identity is not None:
            try:
                observed = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
                if (observed.st_dev, observed.st_ino) == created_identity:
                    os.unlink(path.name, dir_fd=parent_descriptor)
                    os.fsync(parent_descriptor)
            except OSError:
                pass
        raise ReplaySoftwareVerificationError(
            "M5 software verification output could not be published durably"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def software_verification_sha256(verification: ReplaySoftwareVerificationV1) -> str:
    """Return the digest of the exact canonical attestation bytes."""

    return hashlib.sha256(canonical_json_bytes(verification)).hexdigest()


def verify_software(
    *,
    source_root: Path,
    output: Path,
    clean_snapshot: CleanSourceSnapshot,
    implementation_snapshot: ImplementationSnapshot,
) -> ReplaySoftwareVerificationV1:
    """Run all seven checks and exclusively publish their source-bound attestation."""

    absolute_root = Path(os.path.abspath(os.fspath(source_root)))
    expected_output, _output_name = _expected_output(
        absolute_root,
        output,
        revision=clean_snapshot.git_revision,
    )
    _require_source_authority(
        source_root=absolute_root,
        clean_snapshot=clean_snapshot,
        implementation_snapshot=implementation_snapshot,
    )
    _generated_root(absolute_root)
    _require_absent(expected_output)

    environment = _software_environment()
    checks: list[ReplaySoftwareVerificationCheckV1] = []
    for check_id, category in zip(
        M5_SOFTWARE_VERIFICATION_CHECK_IDS,
        M5_SOFTWARE_VERIFICATION_CATEGORIES,
        strict=True,
    ):
        command = M5_SOFTWARE_COMMAND_BY_CHECK[check_id]
        normalized = _run_command(
            command,
            source_root=absolute_root,
            environment=environment,
        )
        checks.append(
            ReplaySoftwareVerificationCheckV1(
                check_id=check_id,
                category=category,
                command=command,
                required_test_ids=M5_SOFTWARE_REQUIRED_TEST_IDS_BY_CHECK[check_id],
                exit_status=0,
                output_sha256=hashlib.sha256(normalized).hexdigest(),
                output_normalization=_NORMALIZATION,
            )
        )

    _require_source_authority(
        source_root=absolute_root,
        clean_snapshot=clean_snapshot,
        implementation_snapshot=implementation_snapshot,
    )
    verification = build_software_verification(
        checks,
        snapshot=implementation_snapshot,
        lockfile_sha256=clean_snapshot.lockfile_sha256,
        package_version=clean_snapshot.package_version,
    )
    _publish_exclusive(expected_output, canonical_json_bytes(verification))
    return verification


def _built_wheel_smoke_main() -> int:
    source_root = Path.cwd().resolve(strict=True)
    try:
        output = _run_built_wheel_smoke(
            source_root=source_root,
            package_version=__version__,
            environment=_software_environment(),
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        print("error: M5 built-wheel smoke failed closed", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(output)
    return 0


def _main(arguments: Sequence[str]) -> int:
    if tuple(arguments) != ("built-wheel-smoke",):
        print("error: invalid M5 software-verification command", file=sys.stderr)
        return 2
    return _built_wheel_smoke_main()


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "M5_BUILT_WHEEL_SMOKE_COMMAND",
    "M5_DISTRIBUTION_BUILD_COMMAND",
    "M5_PRIVACY_AUDIT_COMMAND",
    "M5_PRIVACY_AUDIT_TEST_NODES",
    "M5_PYRIGHT_COMMAND",
    "M5_PYTEST_RELEASE_AUTHORITY_COMMAND",
    "M5_RUFF_FORMAT_COMMAND",
    "M5_RUFF_LINT_COMMAND",
    "M5_SOFTWARE_COMMAND_BY_CHECK",
    "ReplaySoftwareVerificationError",
    "normalize_command_output",
    "software_verification_sha256",
    "verify_software",
]
