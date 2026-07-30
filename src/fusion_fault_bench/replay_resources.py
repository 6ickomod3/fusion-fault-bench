"""Strict external resource-log import for the CPU-only M5 replay.

Raw ``/usr/bin/time -l`` logs remain local.  The public record contains only
their digest, length, parsed resource values, and content-addressed bindings to
the already validated local artifact and run record.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import ValidationError

from fusion_fault_bench.artifacts import canonical_json_bytes, compute_run_record_digest
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.replay_artifact_v1 import (
    M5_REPLAY_IDENTITY_SET_SHA256,
    ReplayExecutionResourceEvidenceV1,
)
from fusion_fault_bench.contracts.replay_v1 import M5_REPLAY_INTENT_SHA256
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)

_MAX_LOG_BYTES = 65_536
_MAX_SIGNED_64 = (1 << 63) - 1
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_OUTPUT_PART_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TIME_LINE_PATTERN = re.compile(
    rb"^ *(?P<real>(?:0|[1-9][0-9]*)\.[0-9]{2}) real"
    rb" +(?P<user>(?:0|[1-9][0-9]*)\.[0-9]{2}) user"
    rb" +(?P<sys>(?:0|[1-9][0-9]*)\.[0-9]{2}) sys$"
)
_RESOURCE_LINE_PATTERN = re.compile(rb"^ *(?P<value>(?:0|[1-9][0-9]*))  (?P<label>[a-z ]+)$")
_DARWIN_RESOURCE_LABELS = (
    "maximum resident set size",
    "average shared memory size",
    "average unshared data size",
    "average unshared stack size",
    "page reclaims",
    "page faults",
    "swaps",
    "block input operations",
    "block output operations",
    "messages sent",
    "messages received",
    "signals received",
    "voluntary context switches",
    "involuntary context switches",
    "instructions retired",
    "cycles elapsed",
    "peak memory footprint",
)
type ReplayRunLabel = Literal["primary", "repeat"]
type ReplayInternalMeasurementScope = Literal[
    "metadata-through-canonical-scientific-members-before-publication"
]
_INTERNAL_SCOPE: ReplayInternalMeasurementScope = (
    "metadata-through-canonical-scientific-members-before-publication"
)
M5_PUBLIC_REPLAY_COMMAND = (
    "ffb",
    "replay",
    "run",
    "--output-dir",
    "<local-output>",
)


def _normalized_replay_command(command: tuple[str, ...]) -> tuple[str, ...]:
    if command == M5_PUBLIC_REPLAY_COMMAND:
        return M5_PUBLIC_REPLAY_COMMAND
    if (
        type(command) is not tuple
        or len(command) != 5
        or command[:4] != ("ffb", "replay", "run", "--output-dir")
        or any(type(item) is not str for item in command)
    ):
        raise ValueError("M5 replay command is not the exact run command")
    output = command[4]
    path = PurePosixPath(output)
    if (
        output != path.as_posix()
        or path.is_absolute()
        or len(path.parts) < 3
        or path.parts[:2] != ("reports", "generated")
        or any(
            part in {".", ".."} or _SAFE_OUTPUT_PART_PATTERN.fullmatch(part) is None
            for part in path.parts[2:]
        )
    ):
        raise ValueError("M5 replay output argument is not a safe canonical local path")
    return M5_PUBLIC_REPLAY_COMMAND


class ReplayResourceEvidenceError(ValueError):
    """A local resource log or its exact run binding failed closed."""


@dataclass(frozen=True, slots=True)
class ParsedDarwinTimeL:
    """The two release-gating values from one canonical Darwin time log."""

    elapsed_seconds: float
    peak_rss_bytes: int


@dataclass(frozen=True, slots=True, repr=False)
class ReplayResourceRunBinding:
    """Non-private facts from one strictly loaded local replay artifact."""

    run: RunRecordV1Alpha1
    local_artifact_sha256: str
    local_run_sha256: str
    persisted_internal_elapsed_seconds: float
    persisted_internal_peak_rss_bytes: int
    persisted_internal_measurement_scope: ReplayInternalMeasurementScope

    def __post_init__(self) -> None:
        _normalized_replay_command(tuple(self.run.command))
        elapsed = self.persisted_internal_elapsed_seconds
        peak_rss = self.persisted_internal_peak_rss_bytes
        if (
            self.run.status != "succeeded"
            or self.run.ended_at is None
            or self.run.source_dirty
            or self.run.manifest_sha256 != M5_REPLAY_INTENT_SHA256
            or self.run.artifact_sha256 != "0" * 64
            or self.run.environment.os_name != "Darwin"
            or _DIGEST_PATTERN.fullmatch(self.local_artifact_sha256) is None
            or _DIGEST_PATTERN.fullmatch(self.local_run_sha256) is None
            or self.local_run_sha256 != compute_run_record_digest(canonical_json_bytes(self.run))
            or type(elapsed) is not float
            or not math.isfinite(elapsed)
            or elapsed <= 0.0
            or type(peak_rss) is not int
            or peak_rss <= 0
            or self.persisted_internal_measurement_scope != _INTERNAL_SCOPE
        ):
            raise ValueError("replay resource run binding is invalid")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(frozen=True, slots=True, repr=False)
class _AuthenticatedResourceLog:
    value: bytes
    device: int
    inode: int

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def replay_environment_sha256(environment: RuntimeEnvironment) -> str:
    """Digest one exact named runtime environment with a fixed domain label."""

    return sha256_digest(
        {
            "schema": "ffb.replay-runtime-environment-binding/v1",
            "environment": environment.model_dump(mode="json", by_alias=True),
        }
    )


def replay_logical_command_sha256(command: tuple[str, ...]) -> str:
    """Digest one exact logical command after normalizing its volatile output."""

    normalized = _normalized_replay_command(command)
    return sha256_digest(
        {
            "schema": "ffb.replay-logical-command-binding/v1",
            "logical_command": list(normalized),
        }
    )


def parse_darwin_time_l(value: bytes) -> ParsedDarwinTimeL:
    """Parse only the complete canonical 18-line Darwin ``time -l`` format."""

    if (
        type(value) is not bytes
        or not 0 < len(value) <= _MAX_LOG_BYTES
        or not value.endswith(b"\n")
        or b"\r" in value
        or b"\t" in value
    ):
        raise ReplayResourceEvidenceError(
            "Darwin time -l log must be bounded canonical LF-terminated bytes"
        )
    try:
        value.decode("ascii")
    except UnicodeDecodeError as error:
        raise ReplayResourceEvidenceError("Darwin time -l log must be ASCII") from error
    lines = value[:-1].split(b"\n")
    if len(lines) != 1 + len(_DARWIN_RESOURCE_LABELS):
        raise ReplayResourceEvidenceError(
            "Darwin time -l log does not have the exact canonical line count"
        )
    time_match = _TIME_LINE_PATTERN.fullmatch(lines[0])
    if time_match is None:
        raise ReplayResourceEvidenceError("Darwin time -l timing line is noncanonical")
    time_text = tuple(time_match.group(name).decode("ascii") for name in ("real", "user", "sys"))
    if any(len(item) > 12 for item in time_text):
        raise ReplayResourceEvidenceError("Darwin time -l timing value exceeds its field")
    expected_time_line = (
        f"{time_text[0]:>12} real {time_text[1]:>12} user {time_text[2]:>12} sys"
    ).encode("ascii")
    if lines[0] != expected_time_line:
        raise ReplayResourceEvidenceError("Darwin time -l timing spacing is noncanonical")

    resource_values: dict[str, int] = {}
    for line, expected_label in zip(lines[1:], _DARWIN_RESOURCE_LABELS, strict=True):
        match = _RESOURCE_LINE_PATTERN.fullmatch(line)
        if match is None or match.group("label").decode("ascii") != expected_label:
            raise ReplayResourceEvidenceError(
                "Darwin time -l resource labels are not exact and ordered"
            )
        numeric_text = match.group("value").decode("ascii")
        if len(numeric_text) > 20:
            raise ReplayResourceEvidenceError("Darwin time -l resource value exceeds its field")
        numeric_value = int(numeric_text)
        if numeric_value > _MAX_SIGNED_64:
            raise ReplayResourceEvidenceError("Darwin time -l resource value is out of range")
        if line != f"{numeric_text:>20}  {expected_label}".encode("ascii"):
            raise ReplayResourceEvidenceError("Darwin time -l resource spacing is noncanonical")
        resource_values[expected_label] = numeric_value

    elapsed_seconds = float(time_text[0])
    peak_rss_bytes = resource_values["maximum resident set size"]
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0 or peak_rss_bytes <= 0:
        raise ReplayResourceEvidenceError("Darwin time -l gating values must be positive")
    return ParsedDarwinTimeL(
        elapsed_seconds=elapsed_seconds,
        peak_rss_bytes=peak_rss_bytes,
    )


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _component_snapshot(path: Path) -> tuple[tuple[Path, os.stat_result], ...]:
    current = Path(path.anchor)
    output: list[tuple[Path, os.stat_result]] = []
    try:
        root_metadata = os.lstat(current)
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
            raise ReplayResourceEvidenceError("resource log path root is unsafe")
        output.append((current, root_metadata))
        for index, part in enumerate(path.parts[1:], start=1):
            current /= part
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode):
                raise ReplayResourceEvidenceError("resource log path contains a symlink")
            if index < len(path.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                raise ReplayResourceEvidenceError("resource log parent is not a directory")
            output.append((current, metadata))
    except ReplayResourceEvidenceError:
        raise
    except OSError as error:
        raise ReplayResourceEvidenceError("resource log path cannot be authenticated") from error
    return tuple(output)


def _read_authenticated_log(path: Path) -> _AuthenticatedResourceLog:
    absolute = Path(os.path.abspath(os.fspath(path)))
    before_components = _component_snapshot(absolute)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ReplayResourceEvidenceError("resource log cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= _MAX_LOG_BYTES
        ):
            raise ReplayResourceEvidenceError(
                "resource log must be one bounded private regular file"
            )
        output = bytearray()
        while len(output) <= _MAX_LOG_BYTES:
            chunk = os.read(
                descriptor,
                min(_MAX_LOG_BYTES + 1 - len(output), _MAX_LOG_BYTES),
            )
            if not chunk:
                break
            output.extend(chunk)
        after = os.fstat(descriptor)
        after_components = _component_snapshot(absolute)
        if _stat_fingerprint(before) != _stat_fingerprint(after):
            raise ReplayResourceEvidenceError("resource log changed while it was read")
        if _stat_fingerprint(before) != _stat_fingerprint(
            before_components[-1][1]
        ) or _stat_fingerprint(after) != _stat_fingerprint(after_components[-1][1]):
            raise ReplayResourceEvidenceError("resource log path changed while it was read")
        if len(before_components) != len(after_components) or any(
            first_path != second_path
            or _stat_fingerprint(first_metadata) != _stat_fingerprint(second_metadata)
            for (first_path, first_metadata), (second_path, second_metadata) in zip(
                before_components,
                after_components,
                strict=True,
            )
        ):
            raise ReplayResourceEvidenceError("resource log path changed while it was read")
        if len(output) != before.st_size:
            raise ReplayResourceEvidenceError("resource log size changed while it was read")
        parsed = bytes(output)
        parse_darwin_time_l(parsed)
        return _AuthenticatedResourceLog(
            value=parsed,
            device=before.st_dev,
            inode=before.st_ino,
        )
    except ReplayResourceEvidenceError:
        raise
    except OSError as error:
        raise ReplayResourceEvidenceError("resource log cannot be read safely") from error
    finally:
        os.close(descriptor)


def _build_resource_record(
    *,
    run_label: ReplayRunLabel,
    binding: ReplayResourceRunBinding,
    log: _AuthenticatedResourceLog,
) -> ReplayExecutionResourceEvidenceV1:
    measurement = parse_darwin_time_l(log.value)
    try:
        return ReplayExecutionResourceEvidenceV1(
            schema="ffb.replay-execution-resource-evidence/v1",
            run_id=binding.run.run_id,
            replay_intent_sha256=M5_REPLAY_INTENT_SHA256,
            replay_identity_set_sha256=M5_REPLAY_IDENTITY_SET_SHA256,
            run_label=run_label,
            local_artifact_sha256=binding.local_artifact_sha256,
            local_run_sha256=binding.local_run_sha256,
            environment_sha256=replay_environment_sha256(binding.run.environment),
            logical_command_sha256=replay_logical_command_sha256(tuple(binding.run.command)),
            persisted_internal_elapsed_seconds=(binding.persisted_internal_elapsed_seconds),
            persisted_internal_peak_rss_bytes=(binding.persisted_internal_peak_rss_bytes),
            persisted_internal_measurement_scope=(binding.persisted_internal_measurement_scope),
            tool_path="/usr/bin/time",
            tool_options=("-l",),
            parser_contract="ffb.darwin-time-l-strict/v1",
            raw_log_sha256=hashlib.sha256(log.value).hexdigest(),
            raw_log_byte_length=len(log.value),
            elapsed_seconds=measurement.elapsed_seconds,
            peak_rss_bytes=measurement.peak_rss_bytes,
            exit_status=0,
            scientific_replay_worker_count=1,
            cpu_process_scope=("one-scientific-replay-worker-no-benchmark-multiprocessing"),
            helper_process_policy=("sequential-provenance-and-resource-measurement-helpers-only"),
            accelerator_requested=False,
            wall_time_cap_seconds=1800.0,
            peak_rss_cap_bytes=1_073_741_824,
            wall_time_within_cap=True,
            peak_rss_within_cap=True,
            measurement_scope=(
                "operator-recorded-darwin-time-l-for-complete-replay-cli-lifetime;"
                "self-reported-not-independent-attestation"
            ),
        )
    except (TypeError, ValidationError, ValueError) as error:
        raise ReplayResourceEvidenceError(
            "external replay resource evidence violates its release contract"
        ) from error


def import_replay_execution_resource_evidence(
    *,
    primary_log_path: Path,
    repeat_log_path: Path,
    primary: ReplayResourceRunBinding,
    repeat: ReplayResourceRunBinding,
) -> tuple[ReplayExecutionResourceEvidenceV1, ReplayExecutionResourceEvidenceV1]:
    """Safely import two distinct raw logs in exact primary/repeat order."""

    if primary.run.run_id != repeat.run.run_id or primary.run.environment != repeat.run.environment:
        raise ReplayResourceEvidenceError(
            "primary and repeat resource bindings disagree on run identity or environment"
        )
    primary_log = _read_authenticated_log(primary_log_path)
    repeat_log = _read_authenticated_log(repeat_log_path)
    if (primary_log.device, primary_log.inode) == (
        repeat_log.device,
        repeat_log.inode,
    ):
        raise ReplayResourceEvidenceError(
            "primary and repeat resource logs must be independent files"
        )
    return (
        _build_resource_record(
            run_label="primary",
            binding=primary,
            log=primary_log,
        ),
        _build_resource_record(
            run_label="repeat",
            binding=repeat,
            log=repeat_log,
        ),
    )
