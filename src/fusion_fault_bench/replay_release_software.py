"""Deterministic, source-bound software verification for the frozen M5 release."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.metadata
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    "--no-cache",
    ".",
)
M5_RUFF_LINT_COMMAND: Final[Command] = (
    "uv",
    "run",
    "--frozen",
    "--no-sync",
    "ruff",
    "check",
    "--no-cache",
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
    "-p",
    "no:cacheprovider",
    "-p",
    "pytest_cov",
    "-o",
    "addopts=",
    "-q",
    "--strict-config",
    "--strict-markers",
    "--cov=fusion_fault_bench",
    "--cov-branch",
    "--cov-report=term-missing",
    "--cov-fail-under=90",
    "tests",
)
M5_DISTRIBUTION_BUILD_COMMAND: Final[Command] = (
    "uv",
    "run",
    "--frozen",
    "--no-sync",
    "python",
    "-m",
    "fusion_fault_bench.replay_release_software",
    "build-wheel",
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
    "-p",
    "no:cacheprovider",
    "-p",
    "pytest_cov",
    "-o",
    "addopts=",
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
_EXECUTABLE_BYTE_CAP = 128 * 1024 * 1024
_WHEEL_BYTE_CAP = 50 * 1024 * 1024
_GENERATED_RELATIVE = Path("reports/generated")
_EXPECTED_UV_VERSION = "0.11.8"
_SAFE_PATH = "/usr/bin:/bin"
_LOCKED_TOOL_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "coverage",
    "iniconfig",
    "nodeenv",
    "packaging",
    "pluggy",
    "pygments",
    "pyright",
    "pytest",
    "pytest-cov",
    "ruff",
)
_PYTHON_VERSION_PATTERN = re.compile(r"3\.12\.[0-9]+")
_PYTEST_PASSED_PATTERN = re.compile(rb"(?<![A-Za-z0-9])([1-9][0-9]*) passed\b")
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


@dataclass(frozen=True)
class _ExecutableFingerprint:
    path: Path
    device: int
    inode: int
    byte_length: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class _SoftwareToolAuthority:
    python: _ExecutableFingerprint
    ruff: _ExecutableFingerprint
    uv: _ExecutableFingerprint
    site_packages: Path
    installed_tools_sha256: str


def _software_environment() -> dict[str, str]:
    # Start from an allowlist. This excludes Python/pytest/coverage startup hooks,
    # dynamic-loader injection, tool config overrides, and hostile PATH entries.
    environment = {
        "HOME": os.devnull,
        "LANG": "C",
        "LC_ALL": "C",
        "NO_COLOR": "1",
        "NO_PROXY": "*",
        "PATH": _SAFE_PATH,
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PIP_NO_INPUT": "1",
        "PYTHONHASHSEED": "0",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "TERM": "dumb",
        "TMPDIR": "/tmp",
        "UV_NO_CONFIG": "1",
        "UV_NO_PROGRESS": "1",
        "UV_OFFLINE": "1",
        "no_proxy": "*",
    }
    raw_cache = os.environ.get("UV_CACHE_DIR")
    if raw_cache is not None:
        cache = Path(raw_cache)
        try:
            resolved_cache = cache.resolve(strict=True)
            state = resolved_cache.lstat()
        except OSError as error:
            raise ReplaySoftwareVerificationError(
                "M5 software verification UV cache is unavailable"
            ) from error
        if (
            not cache.is_absolute()
            or cache.is_symlink()
            or not stat.S_ISDIR(state.st_mode)
            or stat.S_IMODE(state.st_mode) & 0o077
        ):
            raise ReplaySoftwareVerificationError(
                "M5 software verification UV cache is not a private absolute directory"
            )
        environment["UV_CACHE_DIR"] = os.fspath(resolved_cache)
    return environment


def _stable_regular_bytes(path: Path, *, byte_cap: int, label: str) -> bytes:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > byte_cap:
            raise OSError
        value = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise ReplaySoftwareVerificationError(f"M5 {label} is unavailable") from error
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if len(value) != before.st_size or any(
        getattr(before, field) != getattr(after, field) for field in identity
    ):
        raise ReplaySoftwareVerificationError(f"M5 {label} changed while it was read")
    return value


def _fingerprint_executable(path: Path, *, label: str) -> _ExecutableFingerprint:
    try:
        resolved = path.resolve(strict=True)
        state = resolved.lstat()
    except OSError as error:
        raise ReplaySoftwareVerificationError(f"M5 {label} executable is unavailable") from error
    if (
        not stat.S_ISREG(state.st_mode)
        or state.st_nlink != 1
        or state.st_size <= 0
        or state.st_size > _EXECUTABLE_BYTE_CAP
        or not state.st_mode & stat.S_IXUSR
        or state.st_mode & 0o022
        or state.st_uid != os.geteuid()
    ):
        raise ReplaySoftwareVerificationError(f"M5 {label} executable is not trusted")
    value = _stable_regular_bytes(resolved, byte_cap=_EXECUTABLE_BYTE_CAP, label=label)
    after = resolved.lstat()
    return _ExecutableFingerprint(
        path=resolved,
        device=after.st_dev,
        inode=after.st_ino,
        byte_length=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=hashlib.sha256(value).hexdigest(),
    )


def _locked_package_versions(lockfile: Path) -> dict[str, str]:
    try:
        document = tomllib.loads(lockfile.read_text(encoding="utf-8"))
        packages = document["package"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise ReplaySoftwareVerificationError("M5 locked tool authority is unavailable") from error
    if not isinstance(packages, list):
        raise ReplaySoftwareVerificationError("M5 locked tool authority is malformed")
    versions: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ReplaySoftwareVerificationError("M5 locked tool authority is malformed")
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str) or name in versions:
            raise ReplaySoftwareVerificationError("M5 locked tool authority is ambiguous")
        versions[name] = version
    return versions


def _require_locked_tool_install(source_root: Path) -> tuple[Path, str]:
    versions = _locked_package_versions(source_root / "uv.lock")
    environment_root = (source_root / ".venv").resolve(strict=True)
    rows = bytearray()
    site_packages: Path | None = None
    for name in _LOCKED_TOOL_DISTRIBUTIONS:
        locked_version = versions.get(name)
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ReplaySoftwareVerificationError(
                "M5 locked software-verification tool is not installed"
            ) from error
        if locked_version is None or distribution.version != locked_version:
            raise ReplaySoftwareVerificationError(
                "M5 installed software-verification tool differs from uv.lock"
            )
        try:
            distribution_root = Path(str(distribution.locate_file(""))).resolve(strict=True)
            distribution_root.relative_to(environment_root)
        except (OSError, ValueError) as error:
            raise ReplaySoftwareVerificationError(
                "M5 software-verification tool is outside the locked environment"
            ) from error
        if name == "pytest":
            site_packages = distribution_root
        files = distribution.files
        if files is None:
            raise ReplaySoftwareVerificationError(
                "M5 software-verification tool has no installed-file authority"
            )
        rows.extend(f"distribution={name}=={locked_version}\n".encode("ascii"))
        for relative in sorted(files, key=os.fspath):
            expected_hash = relative.hash
            if expected_hash is None:
                if not os.fspath(relative).endswith(".dist-info/RECORD"):
                    raise ReplaySoftwareVerificationError(
                        "M5 software-verification tool has an unhashed installed file"
                    )
                continue
            if expected_hash.mode != "sha256":
                raise ReplaySoftwareVerificationError(
                    "M5 software-verification tool uses an unsupported installed-file hash"
                )
            try:
                installed = Path(str(distribution.locate_file(relative))).resolve(strict=True)
                installed.relative_to(environment_root)
            except (OSError, ValueError) as error:
                raise ReplaySoftwareVerificationError(
                    "M5 software-verification tool file escaped the locked environment"
                ) from error
            value = _stable_regular_bytes(
                installed,
                byte_cap=_EXECUTABLE_BYTE_CAP,
                label="installed tool file",
            )
            actual_hash = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=")
            if actual_hash.decode("ascii") != expected_hash.value:
                raise ReplaySoftwareVerificationError(
                    "M5 software-verification tool file differs from its installation record"
                )
            rows.extend(os.fspath(relative).encode("utf-8"))
            rows.extend(b"\0")
            rows.extend(hashlib.sha256(value).digest())
    if site_packages is None:
        raise ReplaySoftwareVerificationError("M5 pytest installation authority is unavailable")
    return site_packages, hashlib.sha256(rows).hexdigest()


def _trusted_uv_candidate_paths() -> tuple[Path, ...]:
    try:
        home = Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except (KeyError, OSError) as error:
        raise ReplaySoftwareVerificationError("M5 trusted uv location is unavailable") from error
    return (
        home / ".local/bin/uv",
        home / ".cargo/bin/uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    )


def _trusted_uv_authority(environment: Mapping[str, str]) -> _ExecutableFingerprint:
    matches: list[_ExecutableFingerprint] = []
    seen: set[Path] = set()
    for candidate in _trusted_uv_candidate_paths():
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if candidate.is_symlink() or resolved in seen:
            continue
        seen.add(resolved)
        try:
            fingerprint = _fingerprint_executable(resolved, label="uv")
            result = subprocess.run(
                (os.fspath(fingerprint.path), "--version"),
                cwd="/",
                env=dict(environment),
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired, ReplaySoftwareVerificationError):
            continue
        try:
            version_fields = result.stdout.decode("ascii").strip().split()
        except UnicodeDecodeError:
            version_fields = []
        if (
            result.returncode == 0
            and result.stderr == b""
            and version_fields[:2] == ["uv", _EXPECTED_UV_VERSION]
        ):
            matches.append(fingerprint)
    if len(matches) != 1:
        raise ReplaySoftwareVerificationError(
            "M5 requires one exact fixed-location uv executable at the frozen version"
        )
    return matches[0]


def _build_tool_authority(
    source_root: Path,
    environment: Mapping[str, str],
) -> _SoftwareToolAuthority:
    try:
        environment_root = (source_root / ".venv").resolve(strict=True)
        if Path(sys.prefix).resolve(strict=True) != environment_root:
            raise OSError
    except OSError as error:
        raise ReplaySoftwareVerificationError(
            "M5 software verification is outside the locked environment"
        ) from error
    site_packages, installed_tools_sha256 = _require_locked_tool_install(source_root)
    return _SoftwareToolAuthority(
        python=_fingerprint_executable(Path(sys.executable), label="Python"),
        ruff=_fingerprint_executable(source_root / ".venv/bin/ruff", label="Ruff"),
        uv=_trusted_uv_authority(environment),
        site_packages=site_packages,
        installed_tools_sha256=installed_tools_sha256,
    )


def _require_tool_authority_unchanged(authority: _SoftwareToolAuthority) -> None:
    for expected, label in (
        (authority.python, "Python"),
        (authority.ruff, "Ruff"),
        (authority.uv, "uv"),
    ):
        if _fingerprint_executable(expected.path, label=label) != expected:
            raise ReplaySoftwareVerificationError("M5 software tool authority changed")


_ISOLATED_MODULE_RUNNER = (
    "import runpy,sys;site,source,module,*args=sys.argv[1:];"
    "sys.path[:0]=[site,source];sys.argv=[module,*args];"
    "runpy.run_module(module,run_name='__main__')"
)


def _isolated_module_command(
    authority: _SoftwareToolAuthority,
    source_root: Path,
    module: str,
    arguments: Sequence[str],
) -> Command:
    return (
        os.fspath(authority.python.path),
        "-I",
        "-S",
        "-B",
        "-c",
        _ISOLATED_MODULE_RUNNER,
        os.fspath(authority.site_packages),
        os.fspath(source_root / "src"),
        module,
        *arguments,
    )


def _runtime_command(
    logical: Command,
    *,
    source_root: Path,
    authority: _SoftwareToolAuthority,
) -> Command:
    run_prefix = ("uv", "run", "--frozen", "--no-sync")
    if logical[:4] == run_prefix and len(logical) >= 5:
        tool = logical[4]
        arguments = logical[5:]
        if tool == "ruff":
            return (os.fspath(authority.ruff.path), *arguments)
        if tool in {"pyright", "pytest"}:
            return _isolated_module_command(authority, source_root, tool, arguments)
        if tool == "python" and len(arguments) == 3:
            module_flag, module, action = arguments
            if (
                module_flag == "-m"
                and module == "fusion_fault_bench.replay_release_software"
                and action in {"build-wheel", "built-wheel-smoke"}
            ):
                return _isolated_module_command(
                    authority,
                    source_root,
                    module,
                    (action,),
                )
    raise ReplaySoftwareVerificationError("M5 software check command is not allowlisted")


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
    require_test_execution: bool = False,
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
    if (
        require_test_execution
        and _PYTEST_PASSED_PATTERN.search(result.stdout + b"\n" + result.stderr) is None
    ):
        raise ReplaySoftwareVerificationError(
            "M5 pytest check did not report any executed passing tests"
        )
    return normalize_command_output(
        result.stdout,
        result.stderr,
        runtime_paths=(source_root, *runtime_paths),
    )


def _run_software_command(
    logical: Command,
    *,
    source_root: Path,
    environment: Mapping[str, str],
    authority: _SoftwareToolAuthority,
    require_test_execution: bool = False,
) -> bytes:
    _require_tool_authority_unchanged(authority)
    runtime = _runtime_command(logical, source_root=source_root, authority=authority)
    cache = environment.get("UV_CACHE_DIR")
    runtime_paths = (
        authority.python.path,
        authority.ruff.path,
        authority.uv.path,
        authority.site_packages,
        *((Path(cache),) if cache is not None else ()),
    )
    output = _run_command(
        runtime,
        source_root=source_root,
        environment=environment,
        runtime_paths=runtime_paths,
        require_test_execution=require_test_execution,
    )
    _require_tool_authority_unchanged(authority)
    return output


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


def _wheel_record_hash(value: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode("ascii")


def _wheel_member(name: str, value: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ReplaySoftwareVerificationError("M5 wheel member path is unsafe")
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info, value


def _build_distribution_wheel(source_root: Path, package_version: str) -> tuple[Path, bytes]:
    package_root = source_root / "src/fusion_fault_bench"
    dist_info = f"fusion_fault_bench-{package_version}.dist-info"
    members: dict[str, bytes] = {}
    try:
        source_paths = tuple(sorted(package_root.rglob("*.py")))
    except OSError as error:
        raise ReplaySoftwareVerificationError("M5 wheel source tree is unavailable") from error
    if not source_paths or package_root / "__init__.py" not in source_paths:
        raise ReplaySoftwareVerificationError("M5 wheel source tree is incomplete")
    for source in source_paths:
        try:
            relative = source.relative_to(source_root / "src").as_posix()
        except ValueError as error:
            raise ReplaySoftwareVerificationError("M5 wheel source escaped the package") from error
        members[relative] = _stable_regular_bytes(
            source,
            byte_cap=4 * 1024 * 1024,
            label="wheel source",
        )
    readme = _stable_regular_bytes(
        source_root / "README.md", byte_cap=4 * 1024 * 1024, label="wheel README"
    )
    license_bytes = _stable_regular_bytes(
        source_root / "LICENSE", byte_cap=1024 * 1024, label="wheel license"
    )
    metadata = (
        "Metadata-Version: 2.3\n"
        "Name: fusion-fault-bench\n"
        f"Version: {package_version}\n"
        "Summary: Deterministic evaluation of camera-LiDAR estimator fusion "
        "under controlled faults\n"
        "Author: Fusion Fault Bench contributors\n"
        "License-File: LICENSE\n"
        "Requires-Dist: numpy>=2.0,<3\n"
        "Requires-Dist: pydantic>=2.13,<3\n"
        "Requires-Python: >=3.12,<3.13\n"
        "Description-Content-Type: text/markdown\n\n"
    ).encode() + readme
    members.update(
        {
            f"{dist_info}/WHEEL": (
                "Wheel-Version: 1.0\n"
                "Generator: fusion-fault-bench-m5-hermetic-builder\n"
                "Root-Is-Purelib: true\n"
                "Tag: py3-none-any\n"
            ).encode("ascii"),
            f"{dist_info}/entry_points.txt": (
                "[console_scripts]\nffb = fusion_fault_bench.cli:main\n"
            ).encode("ascii"),
            f"{dist_info}/licenses/LICENSE": license_bytes,
            f"{dist_info}/METADATA": metadata,
        }
    )
    record_path = f"{dist_info}/RECORD"
    record = bytearray()
    for name, value in sorted(members.items()):
        record.extend(f"{name},sha256={_wheel_record_hash(value)},{len(value)}\n".encode())
    record.extend(f"{record_path},,\n".encode("ascii"))
    members[record_path] = bytes(record)

    dist = source_root / "dist"
    try:
        dist.mkdir(mode=0o700, exist_ok=True)
        dist_state = dist.lstat()
        if not stat.S_ISDIR(dist_state.st_mode) or dist.is_symlink():
            raise OSError
        expected_name = f"fusion_fault_bench-{package_version}-py3-none-any.whl"
        foreign_wheels = tuple(
            path.name
            for path in dist.iterdir()
            if path.suffix == ".whl" and path.name != expected_name
        )
        if foreign_wheels:
            raise OSError
        descriptor, raw_temporary = tempfile.mkstemp(prefix=".m5-wheel-", suffix=".tmp", dir=dist)
        os.close(descriptor)
        temporary = Path(raw_temporary)
        try:
            with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in sorted(members.items()):
                    info, payload = _wheel_member(name, value)
                    archive.writestr(
                        info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
                    )
            os.chmod(temporary, 0o600)
            wheel = dist / expected_name
            os.replace(temporary, wheel)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
    except (OSError, zipfile.BadZipFile) as error:
        raise ReplaySoftwareVerificationError("M5 deterministic wheel build failed") from error
    wheel_bytes = _regular_file_bytes(wheel, byte_cap=_WHEEL_BYTE_CAP, label="built wheel")
    return wheel, wheel_bytes


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
    uv_authority = _trusted_uv_authority(environment)
    python_authority = _fingerprint_executable(Path(sys.executable), label="Python")
    locked_site_packages, _installed_tools_sha256 = _require_locked_tool_install(source_root)
    generated = _generated_root(source_root)
    temp = Path(tempfile.mkdtemp(prefix=".m5-wheel-smoke-", dir=generated))
    owned = temp.lstat()
    try:
        os.chmod(temp, 0o700)
        smoke_cache = temp / "uv-cache"
        smoke_cache.mkdir(mode=0o700)
        smoke_environment = dict(environment)
        smoke_environment["UV_CACHE_DIR"] = os.fspath(smoke_cache)
        venv = temp / "venv"
        python = venv / "bin/python"
        ffb = venv / "bin/ffb"
        wheel_site_packages = venv / "lib" / "python3.12" / "site-packages"
        import_runner = (
            "import os,sys;from pathlib import Path;wheel,locked=sys.argv[1:3];"
            "sys.path[:0]=[wheel,locked];import fusion_fault_bench as package;"
            "from fusion_fault_bench import replay_release_workflow as workflow;"
            "assert 'NUSCENES_ROOT' not in os.environ;"
            "assert os.environ.get('UV_OFFLINE') == '1';"
            "assert Path(package.__file__).resolve().is_relative_to(Path(wheel).resolve());"
            f"assert package.__version__ == {package_version!r};"
            "assert callable(workflow.validate_release_package);print(package.__version__)"
        )
        cli_runner = (
            "import sys;wheel,locked,*args=sys.argv[1:];"
            "sys.path[:0]=[wheel,locked];from fusion_fault_bench.cli import main;"
            "raise SystemExit(main(args))"
        )
        isolated_prefix = (
            os.fspath(python),
            "-I",
            "-S",
            "-B",
            "-c",
        )
        isolated_paths = (
            os.fspath(wheel_site_packages),
            os.fspath(locked_site_packages),
        )
        commands: tuple[Command, ...] = (
            (
                os.fspath(uv_authority.path),
                "venv",
                "--python",
                os.fspath(python_authority.path),
                os.fspath(venv),
            ),
            (
                os.fspath(uv_authority.path),
                "pip",
                "install",
                "--python",
                os.fspath(python),
                "--offline",
                "--no-deps",
                os.fspath(wheel),
            ),
            (*isolated_prefix, import_runner, *isolated_paths),
            (*isolated_prefix, cli_runner, *isolated_paths, "--version"),
            (*isolated_prefix, cli_runner, *isolated_paths, "replay", "--help"),
            (
                *isolated_prefix,
                cli_runner,
                *isolated_paths,
                "schema",
                "show",
                "replay-validation",
            ),
            (
                *isolated_prefix,
                cli_runner,
                *isolated_paths,
                "schema",
                "show",
                "replay-execution-resource-evidence",
            ),
            (
                *isolated_prefix,
                cli_runner,
                *isolated_paths,
                "replay",
                "release",
                "--help",
            ),
        )
        output = bytearray(
            f"wheel_sha256={hashlib.sha256(wheel_bytes).hexdigest()}\n".encode("ascii")
        )
        for index, command in enumerate(commands):
            normalized = _run_command(
                command,
                source_root=source_root,
                environment=smoke_environment,
                runtime_paths=(temp, locked_site_packages),
            )
            if index == 1:
                _regular_file_bytes(ffb, byte_cap=16 * 1024, label="installed ffb entrypoint")
            output.extend(f"step={index}\n".encode("ascii"))
            output.extend(normalized)
        if _fingerprint_executable(uv_authority.path, label="uv") != uv_authority:
            raise ReplaySoftwareVerificationError("M5 uv authority changed during wheel smoke")
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
    tool_authority = _build_tool_authority(absolute_root, environment)
    checks: list[ReplaySoftwareVerificationCheckV1] = []
    for check_id, category in zip(
        M5_SOFTWARE_VERIFICATION_CHECK_IDS,
        M5_SOFTWARE_VERIFICATION_CATEGORIES,
        strict=True,
    ):
        command = M5_SOFTWARE_COMMAND_BY_CHECK[check_id]
        normalized = _run_software_command(
            command,
            source_root=absolute_root,
            environment=environment,
            authority=tool_authority,
            require_test_execution=check_id in {"pytest-release-authority", "privacy-audit"},
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
    if _build_tool_authority(absolute_root, environment) != tool_authority:
        raise ReplaySoftwareVerificationError("M5 locked software tool authority changed")
    verification = build_software_verification(
        checks,
        snapshot=implementation_snapshot,
        lockfile_sha256=clean_snapshot.lockfile_sha256,
        package_version=clean_snapshot.package_version,
    )
    _publish_exclusive(expected_output, canonical_json_bytes(verification))
    return verification


def _build_wheel_main() -> int:
    source_root = Path.cwd().resolve(strict=True)
    try:
        _wheel, wheel_bytes = _build_distribution_wheel(source_root, __version__)
    except (OSError, RuntimeError, TypeError, ValueError):
        print("error: M5 deterministic wheel build failed closed", file=sys.stderr)
        return 2
    print(f"wheel_sha256={hashlib.sha256(wheel_bytes).hexdigest()}")
    return 0


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
    if tuple(arguments) == ("build-wheel",):
        return _build_wheel_main()
    if tuple(arguments) == ("built-wheel-smoke",):
        return _built_wheel_smoke_main()
    print("error: invalid M5 software-verification command", file=sys.stderr)
    return 2


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
