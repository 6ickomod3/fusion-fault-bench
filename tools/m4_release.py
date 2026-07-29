"""Build and validate the curated Fusion Fault Bench M4 health release."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from fusion_fault_bench.contracts.health_result_v1 import (
    HealthAggregateMetricV1,
    HealthFitSummaryV1,
)
from fusion_fault_bench.contracts.health_v1 import M4_HEALTH_INTENT_PATH
from fusion_fault_bench.health_artifacts import (
    LoadedHealthEvaluationArtifact,
    LoadedHealthFitArtifact,
    load_health_evaluation_artifact,
    load_health_fit_artifact,
)
from fusion_fault_bench.health_release import (
    AggregateQuantitativeClaimV1,
    FitQuantitativeClaimV1,
    HealthQuantitativeClaimV1,
    HealthReleaseWriteRequest,
    HealthRunResourceEvidenceV1,
    HealthRunResources,
    ResourceQuantitativeClaimV1,
    build_health_resource_measurement,
    load_health_release,
    write_health_release,
)
from fusion_fault_bench.provenance import (
    CleanSourceSnapshot,
    discover_clean_source,
    verify_locked_execution,
)

RELEASE_RELATIVE_PATH = Path("reports/releases/m4-health-v0.1.0")
_GENERATED_ROOT = Path("reports/generated")
_TIME_L_LOG_BYTES_MAX = 65_536

type AggregateKey = tuple[str, str, str, str]
type RunArtifact = LoadedHealthFitArtifact | LoadedHealthEvaluationArtifact

_OUTCOME_CLAIM_KEYS: tuple[AggregateKey, ...] = tuple(
    (
        condition_id,
        "combined-health-gate",
        "policy-gain-vs-fixed",
        window,
    )
    for condition_id, window in (
        ("test-camera-output-y-bias.value-03", "event"),
        ("test-lidar-output-y-bias.value-03", "event"),
        ("test-camera-noise-underreported.value-01", "event"),
        ("test-lidar-noise-underreported.value-01", "event"),
        ("test-camera-noise-correctly-reported.value-01", "event"),
        ("test-lidar-noise-correctly-reported.value-01", "event"),
        ("test-camera-timestamp-offset.value-03", "event"),
        ("test-lidar-timestamp-offset.value-03", "event"),
        ("test-camera-dropout.value-02", "event"),
        ("test-lidar-dropout.value-02", "event"),
        ("test-camera-calibration-x.value-03", "event"),
        ("test-camera-calibration-yaw.value-03", "event"),
        ("test-main-clean.value-00", "score"),
        ("test-edge-clean.value-00", "score"),
        ("test-clean-bounded-acceleration.value-00", "score"),
        ("test-common-mode-x-edge.value-03", "event"),
        ("test-cold-start-camera-calibration-x.value-00", "event"),
        ("test-cold-start-lidar-y-bias.value-00", "event"),
    )
)


class HealthReleaseDriverError(ValueError):
    """The repository-local M4 release workflow failed closed."""


@dataclass(frozen=True, slots=True)
class _ResourceLog:
    value: bytes
    device: int
    inode: int


def _clean_snapshot() -> CleanSourceSnapshot:
    try:
        snapshot = discover_clean_source(M4_HEALTH_INTENT_PATH)
        verify_locked_execution(snapshot)
    except (OSError, ValueError) as error:
        raise HealthReleaseDriverError(
            "M4 release construction requires a clean locked source checkout"
        ) from error
    if snapshot.manifest_relative_path != M4_HEALTH_INTENT_PATH.as_posix():
        raise HealthReleaseDriverError("M4 release source does not bind the frozen intent")
    return snapshot


def _reject_symlink_components(
    path: Path,
    *,
    source_root: Path,
    label: str,
) -> None:
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        raise HealthReleaseDriverError(f"{label} must remain inside the repository") from None
    current = source_root
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise HealthReleaseDriverError(f"{label} path cannot be inspected") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise HealthReleaseDriverError(f"{label} must not use symlink components")


def _repository_path(
    value: Path,
    *,
    source_root: Path,
    label: str,
    generated_only: bool,
) -> Path:
    if value.is_absolute() or any(part in {"", ".", ".."} for part in value.parts):
        raise HealthReleaseDriverError(f"{label} must be normalized and repository-relative")
    if generated_only:
        try:
            value.relative_to(_GENERATED_ROOT)
        except ValueError:
            raise HealthReleaseDriverError(f"{label} must remain under reports/generated") from None
        if value == _GENERATED_ROOT:
            raise HealthReleaseDriverError(f"{label} must name a specific generated member")
    resolved = source_root / value
    if generated_only:
        _reject_symlink_components(
            resolved,
            source_root=source_root,
            label=label,
        )
    return resolved


def _read_time_l_log(path: Path) -> _ResourceLog:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise HealthReleaseDriverError("resource log cannot be opened safely") from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= _TIME_L_LOG_BYTES_MAX
        ):
            raise HealthReleaseDriverError(
                "resource log must be one bounded independent regular file"
            )
        output = bytearray()
        while len(output) <= _TIME_L_LOG_BYTES_MAX:
            chunk = os.read(descriptor, min(65_536, _TIME_L_LOG_BYTES_MAX + 1 - len(output)))
            if not chunk:
                break
            output.extend(chunk)
        final = os.fstat(descriptor)
        try:
            path_metadata = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise HealthReleaseDriverError("resource log path cannot be reauthenticated") from error
        fingerprint = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        if fingerprint != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ):
            raise HealthReleaseDriverError("resource log changed while it was read")
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise HealthReleaseDriverError("resource log path changed while it was read")
        if len(output) != metadata.st_size:
            raise HealthReleaseDriverError("resource log size changed while it was read")
        return _ResourceLog(
            value=bytes(output),
            device=metadata.st_dev,
            inode=metadata.st_ino,
        )
    except OSError as error:
        raise HealthReleaseDriverError("resource log cannot be read safely") from error
    finally:
        os.close(descriptor)


def _read_independent_resource_logs(
    paths: Mapping[str, Path],
) -> Mapping[str, _ResourceLog]:
    if len(paths) != 4 or len(set(paths.values())) != 4:
        raise HealthReleaseDriverError("four resource logs must use distinct paths")
    logs = {label: _read_time_l_log(path) for label, path in paths.items()}
    identities = {(log.device, log.inode) for log in logs.values()}
    if len(identities) != 4:
        raise HealthReleaseDriverError("four resource logs must use distinct files")
    return logs


def _require_current_common_provenance(
    snapshot: CleanSourceSnapshot,
    artifacts: Sequence[RunArtifact],
) -> None:
    expected = (
        snapshot.git_revision,
        snapshot.lockfile_sha256,
        snapshot.package_version,
        False,
    )
    if len(artifacts) != 4 or any(
        (
            artifact.run.git_revision,
            artifact.run.lockfile_sha256,
            artifact.run.package_version,
            artifact.run.source_dirty,
        )
        != expected
        for artifact in artifacts
    ):
        raise HealthReleaseDriverError(
            "all four M4 artifacts must come from the current clean source snapshot"
        )


def _aggregate_key(record: HealthAggregateMetricV1) -> AggregateKey:
    return (
        record.condition_id,
        "" if record.method is None else record.method,
        record.metric_name,
        "" if record.window is None else record.window,
    )


def derive_quantitative_claims(
    *,
    fit_summary: HealthFitSummaryV1,
    aggregates: Sequence[HealthAggregateMetricV1],
    resources: Sequence[HealthRunResourceEvidenceV1],
) -> tuple[HealthQuantitativeClaimV1, ...]:
    """Project a frozen, result-independent set of public numeric evidence."""

    aggregate_index: Mapping[AggregateKey, HealthAggregateMetricV1] = {
        _aggregate_key(record): record for record in aggregates
    }
    if len(aggregate_index) != len(aggregates):
        raise HealthReleaseDriverError("M4 aggregate matrix contains duplicate keys")
    missing = tuple(key for key in _OUTCOME_CLAIM_KEYS if key not in aggregate_index)
    if missing:
        raise HealthReleaseDriverError(
            f"M4 aggregate matrix is missing {len(missing)} frozen public claim rows"
        )

    claims: list[HealthQuantitativeClaimV1] = [
        AggregateQuantitativeClaimV1(
            schema="ffb.health-quantitative-claim/v1",
            source_kind="aggregate",
            claim_id=f"outcome-{condition_id}-{window}",
            presentation_id="m4-primary-outcome-table",
            aggregate=aggregate_index[(condition_id, method, metric, window)],
        )
        for condition_id, method, metric, window in _OUTCOME_CLAIM_KEYS
    ]
    for field, value, unit in (
        (
            "selected-candidate-index",
            float(fit_summary.selected_candidate_index),
            "count",
        ),
        (
            "selected-self-threshold",
            float(fit_summary.selected_self_threshold),
            "fraction",
        ),
        (
            "selected-cross-threshold",
            float(fit_summary.selected_cross_threshold),
            "fraction",
        ),
    ):
        claims.append(
            FitQuantitativeClaimV1(
                schema="ffb.health-quantitative-claim/v1",
                source_kind="fit-summary",
                claim_id=f"fit-{field}",
                presentation_id="m4-fit-summary-table",
                field=cast(object, field),
                value=value,
                unit=cast(object, unit),
            )
        )
    for resource in resources:
        for metric, value, unit in (
            ("wall-time-seconds", resource.wall_time_seconds, "s"),
            ("peak-rss-bytes", float(resource.peak_rss_bytes), "bytes"),
        ):
            claims.append(
                ResourceQuantitativeClaimV1(
                    schema="ffb.health-quantitative-claim/v1",
                    source_kind="resource",
                    claim_id=f"resource-{resource.run_label}-{metric}",
                    presentation_id="m4-resource-summary-table",
                    run_label=resource.run_label,
                    metric=cast(object, metric),
                    value=value,
                    unit=cast(object, unit),
                    cpu_model=resource.cpu_model,
                    evidence_scope=("operator-recorded-time-l-sidecar-not-independent-attestation"),
                )
            )
    return tuple(claims)


def _resource(
    run_label: str,
    artifact: RunArtifact,
    log: _ResourceLog,
) -> HealthRunResources:
    measurement = build_health_resource_measurement(
        cast(object, run_label),
        artifact,
        log.value,
    )
    return HealthRunResources(time_l_log=log.value, measurement=measurement)


def build_release(
    *,
    official_fit: Path,
    repeat_fit: Path,
    primary_evaluation: Path,
    repeat_evaluation: Path,
    primary_fit_time_l: Path,
    repeat_fit_time_l: Path,
    primary_evaluation_time_l: Path,
    repeat_evaluation_time_l: Path,
    output_dir: Path = RELEASE_RELATIVE_PATH,
) -> str:
    """Authenticate four runs and publish one no-overwrite curated release."""

    snapshot = _clean_snapshot()
    generated = {
        label: _repository_path(
            path,
            source_root=snapshot.source_root,
            label=label,
            generated_only=True,
        )
        for label, path in (
            ("official fit", official_fit),
            ("repeat fit", repeat_fit),
            ("primary evaluation", primary_evaluation),
            ("repeat evaluation", repeat_evaluation),
            ("primary fit resource log", primary_fit_time_l),
            ("repeat fit resource log", repeat_fit_time_l),
            ("primary evaluation resource log", primary_evaluation_time_l),
            ("repeat evaluation resource log", repeat_evaluation_time_l),
        )
    }
    destination = _repository_path(
        output_dir,
        source_root=snapshot.source_root,
        label="release output",
        generated_only=False,
    )
    if output_dir != RELEASE_RELATIVE_PATH:
        raise HealthReleaseDriverError(
            f"M4 release output must be {RELEASE_RELATIVE_PATH.as_posix()}"
        )

    official = load_health_fit_artifact(generated["official fit"])
    repeated_fit = load_health_fit_artifact(generated["repeat fit"])
    primary = load_health_evaluation_artifact(
        generated["primary evaluation"],
        fit_artifact=official,
    )
    repeated_evaluation = load_health_evaluation_artifact(
        generated["repeat evaluation"],
        fit_artifact=official,
    )
    _require_current_common_provenance(
        snapshot,
        (official, repeated_fit, primary, repeated_evaluation),
    )
    resource_logs = _read_independent_resource_logs(
        {
            label: generated[label]
            for label in (
                "primary fit resource log",
                "repeat fit resource log",
                "primary evaluation resource log",
                "repeat evaluation resource log",
            )
        }
    )
    resource_inputs = (
        _resource(
            "primary-fit",
            official,
            resource_logs["primary fit resource log"],
        ),
        _resource(
            "repeat-fit",
            repeated_fit,
            resource_logs["repeat fit resource log"],
        ),
        _resource(
            "primary-evaluation",
            primary,
            resource_logs["primary evaluation resource log"],
        ),
        _resource(
            "repeat-evaluation",
            repeated_evaluation,
            resource_logs["repeat evaluation resource log"],
        ),
    )
    claims = derive_quantitative_claims(
        fit_summary=official.summary,
        aggregates=primary.aggregates,
        resources=tuple(resource.measurement for resource in resource_inputs),
    )
    request = HealthReleaseWriteRequest(
        official_fit_path=generated["official fit"],
        repeat_fit_path=generated["repeat fit"],
        primary_evaluation_path=generated["primary evaluation"],
        repeat_evaluation_path=generated["repeat evaluation"],
        primary_fit_resources=resource_inputs[0],
        repeat_fit_resources=resource_inputs[1],
        primary_evaluation_resources=resource_inputs[2],
        repeat_evaluation_resources=resource_inputs[3],
        quantitative_claims=claims,
    )
    release = write_health_release(
        request,
        destination,
        source_root=snapshot.source_root,
        git_metadata_dirs=(snapshot.git_dir, snapshot.git_common_dir),
    )
    return release.release_artifact_sha256


def validate_release(path: Path) -> str:
    """Strictly validate one curated M4 release tree."""

    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise HealthReleaseDriverError(
            "release validation path must be normalized and repository-relative"
        )
    release = load_health_release(Path.cwd() / path)
    return release.release_artifact_sha256


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate the curated M4 observable-health release."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-release")
    for flag in (
        "official-fit",
        "repeat-fit",
        "primary-evaluation",
        "repeat-evaluation",
        "primary-fit-time-l",
        "repeat-fit-time-l",
        "primary-evaluation-time-l",
        "repeat-evaluation-time-l",
    ):
        build.add_argument(f"--{flag}", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, default=RELEASE_RELATIVE_PATH)
    validate = commands.add_parser("validate-release")
    validate.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the repository-local M4 release tool."""

    args = _build_parser().parse_args(argv)
    try:
        if args.command == "build-release":
            digest = build_release(
                official_fit=args.official_fit,
                repeat_fit=args.repeat_fit,
                primary_evaluation=args.primary_evaluation,
                repeat_evaluation=args.repeat_evaluation,
                primary_fit_time_l=args.primary_fit_time_l,
                repeat_fit_time_l=args.repeat_fit_time_l,
                primary_evaluation_time_l=args.primary_evaluation_time_l,
                repeat_evaluation_time_l=args.repeat_evaluation_time_l,
                output_dir=args.output_dir,
            )
            print(f"built m4-health-v0.1.0 release_artifact_sha256={digest}")
        else:
            digest = validate_release(args.path)
            print(f"valid m4-health-v0.1.0 release_artifact_sha256={digest}")
    except (OSError, ValueError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
