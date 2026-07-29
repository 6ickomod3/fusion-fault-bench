"""Clean-source and runtime provenance for release-grade experiment runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import stat
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fusion_fault_bench import __version__
from fusion_fault_bench.contracts.result_v1alpha1 import RuntimeEnvironment


class ProvenanceError(ValueError):
    """Release provenance cannot be discovered without ambiguity."""


@dataclass(frozen=True)
class CleanSourceSnapshot:
    """Stable source facts required before experiment evaluation begins."""

    source_root: Path
    git_revision: str
    git_dir: Path
    git_common_dir: Path
    lockfile_sha256: str
    package_version: str
    manifest_relative_path: str


_PROJECT_DISTRIBUTION = "fusion-fault-bench"
_SAFE_PATH_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _git_output(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(source), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise ProvenanceError(f"Git provenance unavailable: {detail}")
    return result.stdout.strip()


def _git_bytes(source: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ("git", "-C", str(source), *arguments),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = os.fsdecode(result.stderr).strip() or "unknown Git error"
        raise ProvenanceError(f"Git provenance unavailable: {detail}")
    return result.stdout


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _head_tree_entries(
    source_root: Path,
    revision: str,
) -> dict[bytes, tuple[str, str]]:
    entries: dict[bytes, tuple[str, str]] = {}
    raw = _git_bytes(source_root, "ls-tree", "-r", "-z", "--full-tree", revision)
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, path = record.split(b"\t", maxsplit=1)
            mode, object_type, object_id = metadata.split(b" ", maxsplit=2)
        except ValueError as error:
            raise ProvenanceError("HEAD contains an unparseable tree entry") from error
        if object_type != b"blob" or mode not in {b"100644", b"100755"}:
            raise ProvenanceError("release checkout contains an unsupported Git tree entry")
        if path in entries:
            raise ProvenanceError("HEAD contains duplicate tree paths")
        entries[path] = (mode.decode("ascii"), object_id.decode("ascii"))
    return entries


def _index_entries(source_root: Path) -> dict[bytes, tuple[str, str]]:
    entries: dict[bytes, tuple[str, str]] = {}
    raw = _git_bytes(source_root, "ls-files", "--stage", "-z")
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, path = record.split(b"\t", maxsplit=1)
            mode, object_id, stage = metadata.split(b" ", maxsplit=2)
        except ValueError as error:
            raise ProvenanceError("Git index contains an unparseable entry") from error
        if stage != b"0" or path in entries:
            raise ProvenanceError("Git index contains an unmerged or duplicate entry")
        entries[path] = (mode.decode("ascii"), object_id.decode("ascii"))
    return entries


def _tracked_worktree_bytes(path: Path, *, mode: str) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ProvenanceError(f"tracked path is unavailable: {path}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ProvenanceError(f"tracked path is not a regular file: {path}")
    expected_executable = mode == "100755"
    if bool(metadata.st_mode & stat.S_IXUSR) != expected_executable:
        raise ProvenanceError(f"tracked path executable mode differs from HEAD: {path}")
    return path.read_bytes()


def _verify_exact_head_checkout(source_root: Path, *, revision: str) -> None:
    flagged = [
        record
        for record in _git_bytes(source_root, "ls-files", "-v", "-z").split(b"\0")
        if record and not record.startswith(b"H ")
    ]
    if flagged:
        raise ProvenanceError(
            "Git index uses assume-unchanged, skip-worktree, or another non-normal flag"
        )
    head_entries = _head_tree_entries(source_root, revision)
    if _index_entries(source_root) != head_entries:
        raise ProvenanceError("Git index does not exactly match HEAD")
    untracked = _git_bytes(
        source_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if untracked:
        raise ProvenanceError("release runs require no untracked non-ignored files")
    for raw_path, (mode, expected_object_id) in head_entries.items():
        path = source_root / os.fsdecode(raw_path)
        actual_object_id = _git_blob_sha1(_tracked_worktree_bytes(path, mode=mode))
        if actual_object_id != expected_object_id:
            raise ProvenanceError(f"tracked path content differs from HEAD: {path}")


def _resolve_git_path(source_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = source_root / path
    return path.resolve(strict=True)


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise ProvenanceError(f"{label} path must not contain symlinks")


def discover_clean_source(manifest_path: Path) -> CleanSourceSnapshot:
    """Resolve a tracked manifest and reject any dirty or unavailable source."""

    _reject_symlink_components(manifest_path, label="manifest")
    manifest_absolute = manifest_path.resolve(strict=True)
    if not manifest_absolute.is_file():
        raise ProvenanceError("manifest must be a regular file")

    source_root = Path(
        _git_output(manifest_absolute.parent, "rev-parse", "--show-toplevel")
    ).resolve(strict=True)
    try:
        manifest_relative = manifest_absolute.relative_to(source_root)
    except ValueError as error:
        raise ProvenanceError("manifest must be inside the source checkout") from error
    manifest_relative_posix = manifest_relative.as_posix()
    if any(
        _SAFE_PATH_SEGMENT.fullmatch(part) is None for part in manifest_relative_posix.split("/")
    ):
        raise ProvenanceError(
            "manifest path must use conservative POSIX repository-relative segments"
        )
    _git_output(
        source_root,
        "ls-files",
        "--error-unmatch",
        "--",
        manifest_relative_posix,
    )

    revision = _git_output(source_root, "rev-parse", "HEAD")
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ProvenanceError("Git revision is not a lowercase 40-character object ID")
    _verify_exact_head_checkout(source_root, revision=revision)
    if _git_output(source_root, "rev-parse", "HEAD") != revision:
        raise ProvenanceError("Git HEAD changed during clean-source discovery")

    lockfile = source_root / "uv.lock"
    if not lockfile.is_file() or lockfile.is_symlink():
        raise ProvenanceError("source checkout must contain a regular uv.lock")
    lock_digest = hashlib.sha256(lockfile.read_bytes()).hexdigest()

    git_dir = _resolve_git_path(
        source_root,
        _git_output(source_root, "rev-parse", "--absolute-git-dir"),
    )
    git_common_dir = _resolve_git_path(
        source_root,
        _git_output(source_root, "rev-parse", "--git-common-dir"),
    )
    return CleanSourceSnapshot(
        source_root=source_root,
        git_revision=revision,
        git_dir=git_dir,
        git_common_dir=git_common_dir,
        lockfile_sha256=lock_digest,
        package_version=__version__,
        manifest_relative_path=manifest_relative_posix,
    )


def _locked_runtime_versions(lockfile: Path) -> dict[str, str]:
    document = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    raw_packages = document.get("package")
    if not isinstance(raw_packages, list):
        raise ProvenanceError("uv.lock does not contain a package list")
    packages: dict[str, dict[str, Any]] = {}
    for raw_package in raw_packages:
        if not isinstance(raw_package, dict):
            raise ProvenanceError("uv.lock contains a malformed package entry")
        name = raw_package.get("name")
        version = raw_package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ProvenanceError("uv.lock package entries require names and versions")
        if name in packages:
            raise ProvenanceError(f"uv.lock has an ambiguous runtime package: {name}")
        packages[name] = raw_package

    pending = [_PROJECT_DISTRIBUTION]
    resolved: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if name in resolved:
            continue
        package = packages.get(name)
        if package is None:
            raise ProvenanceError(f"uv.lock is missing runtime package {name}")
        version = package["version"]
        assert isinstance(version, str)
        resolved[name] = version
        dependencies = package.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ProvenanceError(f"uv.lock has malformed dependencies for {name}")
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                raise ProvenanceError(f"uv.lock has a malformed dependency for {name}")
            dependency_name = dependency.get("name")
            if not isinstance(dependency_name, str):
                raise ProvenanceError(f"uv.lock has a malformed dependency for {name}")
            pending.append(dependency_name)
    return resolved


def verify_locked_execution(snapshot: CleanSourceSnapshot) -> None:
    """Prove this process is executing checkout A in its locked project environment."""

    if Path.cwd().resolve(strict=True) != snapshot.source_root:
        raise ProvenanceError("release runs must execute from the clean source root")

    expected_package = snapshot.source_root / "src" / "fusion_fault_bench"
    _reject_symlink_components(expected_package, label="package source")
    if Path(__file__).resolve(strict=True).parent != expected_package.resolve(strict=True):
        raise ProvenanceError("executing package does not come from the clean source checkout")

    environment_path = snapshot.source_root / ".venv"
    _reject_symlink_components(environment_path, label="project environment")
    expected_environment = environment_path.resolve(strict=True)
    if Path(sys.prefix).resolve(strict=True) != expected_environment:
        raise ProvenanceError("release runs require the checkout's locked .venv")

    lockfile = snapshot.source_root / "uv.lock"
    actual_lock_digest = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    if actual_lock_digest != snapshot.lockfile_sha256:
        raise ProvenanceError("uv.lock changed after clean-source discovery")

    python_pin = snapshot.source_root / ".python-version"
    if not python_pin.is_file() or python_pin.is_symlink():
        raise ProvenanceError("source checkout must contain a regular .python-version")
    if platform.python_version() != python_pin.read_text(encoding="utf-8").strip():
        raise ProvenanceError("executing Python does not match .python-version")

    locked_versions = _locked_runtime_versions(lockfile)
    locked_project_version = locked_versions[_PROJECT_DISTRIBUTION]
    if snapshot.package_version != __version__ or __version__ != locked_project_version:
        raise ProvenanceError("snapshot, package, and locked project versions do not agree")
    for distribution, locked_version in locked_versions.items():
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise ProvenanceError(
                f"locked runtime distribution is not installed: {distribution}"
            ) from error
        if installed_version != locked_version:
            raise ProvenanceError(
                f"installed {distribution} {installed_version} "
                f"does not match locked version {locked_version}"
            )


def logical_reproduction_command(
    *,
    snapshot: CleanSourceSnapshot,
    experiment: str,
    manifest_sha256: str,
) -> tuple[str, ...]:
    """Return the path-independent command interpreted from the source root."""

    output = f"reports/generated/{experiment}-{manifest_sha256[:12]}"
    return (
        "ffb",
        "run",
        snapshot.manifest_relative_path,
        "--output-dir",
        output,
    )


def _sysctl_value(name: str) -> str | None:
    result = subprocess.run(
        ("sysctl", "-n", name),
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        value = _sysctl_value("machdep.cpu.brand_string")
        if value is not None:
            return value
    if platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    value = line.split(":", maxsplit=1)[1].strip()
                    if value:
                        return value
    return platform.processor() or platform.machine() or "unknown-cpu"


def _memory_bytes() -> int:
    if platform.system() == "Darwin":
        value = _sysctl_value("hw.memsize")
        if value is not None:
            return int(value)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        page_size = 0
        page_count = 0
    total = int(page_size) * int(page_count)
    if total <= 0:
        raise ProvenanceError("physical memory size is unavailable")
    return total


def collect_runtime_environment() -> RuntimeEnvironment:
    """Collect the public, hostname-free CPU environment record."""

    return RuntimeEnvironment(
        python_version=platform.python_version(),
        os_name=platform.system(),
        os_release=platform.release(),
        machine=platform.machine() or "unknown-machine",
        cpu_model=_cpu_model(),
        logical_cpu_count=os.cpu_count() or 1,
        memory_bytes=_memory_bytes(),
    )
