"""Clean-source orchestration for the local-only M2 geometry validation."""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from fusion_fault_bench.adapters.nuscenes import (
    NuScenesAdapterError,
    load_nuscenes_mini,
)
from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    assert_directory_descriptor_matches_path,
    derive_run_id,
    open_or_create_real_directory,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.geometry_validation_v1 import (
    EXPECTED_KEYFRAME_BLOB_CHECK_COUNT,
    FROZEN_GEOMETRY_MANIFEST_SHA256,
    GEOMETRY_ARTIFACT_CONTRACT,
    GEOMETRY_LOGICAL_COMMAND,
    DatasetValidationV1,
    GeometryValidationManifestV1,
    GeometryValidationV1,
)
from fusion_fault_bench.contracts.result_v1alpha1 import RunRecordV1Alpha1
from fusion_fault_bench.geometry_artifacts import (
    GeometryArtifactWriteRequest,
    LoadedGeometryArtifact,
    write_geometry_validation_artifact,
)
from fusion_fault_bench.geometry_validation import (
    GeometryValidationComputationError,
    build_covariance_validation,
    build_synthetic_geometry_validation,
)
from fusion_fault_bench.nuscenes_geometry import (
    build_production_projection_diagnostic,
    projection_crosscheck_passes,
)
from fusion_fault_bench.provenance import (
    CleanSourceSnapshot,
    ProvenanceError,
    collect_runtime_environment,
    discover_clean_source,
    verify_locked_execution,
)
from fusion_fault_bench.reference.nuscenes_projection import (
    ScalarProjectionReferenceError,
    build_scalar_projection_diagnostic,
    render_scalar_diagnostic_svg,
)

_DATASET_ROOT_ENV = "NUSCENES_ROOT"
_MANIFEST_RELATIVE_PATH = Path("examples/validation/m2-geometry-v1.json")
_OUTPUT_RELATIVE_PATH = Path("reports/generated/m2-geometry")
_DIAGNOSTIC_RELATIVE_PATH = Path("reports/generated/m2-geometry-diagnostic.svg")
_DIAGNOSTIC_BYTE_CAP = 5 * 1024 * 1024


class GeometryRunnerError(ValueError):
    """M2 execution failed without exposing private local details."""


def _write_exclusive_diagnostic_at(
    directory_fd: int,
    name: str,
    value: bytes,
) -> None:
    if len(value) > _DIAGNOSTIC_BYTE_CAP:
        raise GeometryRunnerError("M2 diagnostic publication failed")
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        created = True
        remaining = memoryview(value)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short diagnostic write")
            remaining = remaining[written:]
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = os.read(descriptor, len(value) + 1)
        if observed != value:
            raise OSError("diagnostic verification failed")
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if created:
            with suppress(OSError):
                os.unlink(name, dir_fd=directory_fd)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current /= component
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode):
                raise GeometryRunnerError("M2 dataset root validation failed")
    except GeometryRunnerError:
        raise
    except OSError:
        raise GeometryRunnerError("M2 dataset root validation failed") from None


def _resolve_dataset_root(
    *,
    environment_name: str,
    source_root: Path,
) -> Path:
    if environment_name != _DATASET_ROOT_ENV:
        raise GeometryRunnerError("M2 requires the fixed dataset environment name")
    raw_value = os.environ.get(environment_name)
    if raw_value is None or not raw_value:
        raise GeometryRunnerError("M2 dataset environment is unavailable")
    candidate = Path(raw_value)
    if not candidate.is_absolute():
        raise GeometryRunnerError("M2 dataset root must be absolute")
    _reject_symlink_components(candidate)
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_dir():
            raise GeometryRunnerError("M2 dataset root validation failed")
    except GeometryRunnerError:
        raise
    except (OSError, RuntimeError):
        raise GeometryRunnerError("M2 dataset root validation failed") from None
    try:
        resolved.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise GeometryRunnerError("M2 dataset root must remain outside the source checkout")
    try:
        source_root.relative_to(resolved)
    except ValueError:
        return resolved
    raise GeometryRunnerError("M2 dataset root must remain outside the source checkout")


def _load_frozen_manifest(
    manifest_path: Path,
    *,
    snapshot: CleanSourceSnapshot,
) -> GeometryValidationManifestV1:
    if (
        snapshot.manifest_relative_path != _MANIFEST_RELATIVE_PATH.as_posix()
        or manifest_path != _MANIFEST_RELATIVE_PATH
    ):
        raise GeometryRunnerError("M2 requires the frozen repository-relative manifest path")
    try:
        manifest = GeometryValidationManifestV1.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValidationError, ValueError):
        raise GeometryRunnerError("M2 manifest validation failed") from None
    if (
        sha256_digest(manifest) != FROZEN_GEOMETRY_MANIFEST_SHA256
        or tuple(manifest.artifact.logical_command) != GEOMETRY_LOGICAL_COMMAND
    ):
        raise GeometryRunnerError("M2 manifest is not the frozen preregistered intent")
    return manifest


def _validate_output_destination(output_dir: Path) -> None:
    if output_dir != _OUTPUT_RELATIVE_PATH:
        raise GeometryRunnerError("M2 requires the frozen repository-relative output path")


def _write_local_diagnostic(
    *,
    source_root: Path,
    svg: str,
) -> None:
    if str(source_root) in svg:
        raise GeometryRunnerError("M2 diagnostic privacy validation failed")
    destination = source_root / _DIAGNOSTIC_RELATIVE_PATH
    parent = destination.parent
    descriptor: int | None = None
    created = False
    try:
        descriptor = open_or_create_real_directory(parent)
        assert_directory_descriptor_matches_path(
            descriptor,
            parent,
            label="M2 diagnostic parent",
        )
        _write_exclusive_diagnostic_at(
            descriptor,
            destination.name,
            svg.encode("utf-8"),
        )
        created = True
        os.fsync(descriptor)
        assert_directory_descriptor_matches_path(
            descriptor,
            parent,
            label="M2 diagnostic parent",
        )
    except (ArtifactValidationError, OSError):
        if created and descriptor is not None:
            try:
                os.unlink(destination.name, dir_fd=descriptor)
                os.fsync(descriptor)
            except OSError:
                pass
        raise GeometryRunnerError("M2 diagnostic publication failed") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _initial_snapshot(manifest_path: Path) -> CleanSourceSnapshot:
    try:
        snapshot = discover_clean_source(manifest_path)
        verify_locked_execution(snapshot)
        return snapshot
    except (OSError, ProvenanceError, ValueError):
        raise GeometryRunnerError("M2 clean-source validation failed") from None


def _final_snapshot(
    manifest_path: Path,
    initial: CleanSourceSnapshot,
) -> None:
    try:
        final = discover_clean_source(manifest_path)
        verify_locked_execution(final)
    except (OSError, ProvenanceError, ValueError):
        raise GeometryRunnerError("M2 final clean-source validation failed") from None
    if final != initial:
        raise GeometryRunnerError("M2 source provenance changed during validation")


def run_geometry_validation(
    manifest_path: Path,
    *,
    dataset_root_env: str,
    output_dir: Path,
) -> LoadedGeometryArtifact:
    """Run every frozen M2 gate and atomically publish sanitized evidence."""

    _validate_output_destination(output_dir)
    snapshot = _initial_snapshot(manifest_path)
    manifest = _load_frozen_manifest(manifest_path, snapshot=snapshot)
    dataset_root = _resolve_dataset_root(
        environment_name=dataset_root_env,
        source_root=snapshot.source_root,
    )
    manifest_digest = sha256_digest(manifest)
    run_id = derive_run_id(
        manifest_sha256=manifest_digest,
        git_revision=snapshot.git_revision,
        lockfile_sha256=snapshot.lockfile_sha256,
        package_version=snapshot.package_version,
        artifact_contract=GEOMETRY_ARTIFACT_CONTRACT,
    )
    started_at = datetime.now(UTC)
    try:
        metadata = load_nuscenes_mini(dataset_root)
        scalar_diagnostic = build_scalar_projection_diagnostic(dataset_root)
        production_diagnostic = build_production_projection_diagnostic(metadata)
        local_crosscheck_passed = projection_crosscheck_passes(
            production_diagnostic,
            scalar_diagnostic,
            pixel_tolerance_px=(
                manifest.property_validation.local_scalar_projection_max_abs_tolerance_px
            ),
            depth_tolerance_m=(manifest.property_validation.local_scalar_depth_max_abs_tolerance_m),
            minimum_annotation_count=(
                manifest.dataset.diagnostic_nonvacuity.minimum_annotation_count
            ),
            minimum_finite_positive_depth_center_count=(
                manifest.dataset.diagnostic_nonvacuity.minimum_finite_positive_depth_center_count
            ),
        )
    except (
        KeyError,
        NuScenesAdapterError,
        OSError,
        ScalarProjectionReferenceError,
        TypeError,
        ValueError,
    ):
        raise GeometryRunnerError("M2 local dataset geometry validation failed") from None

    adapter_validation = metadata.validation
    if (
        not adapter_validation.headline_profile_passed_attested
        or not adapter_validation.structural_integrity_passed_attested
        or adapter_validation.keyframe_blob_check_count != EXPECTED_KEYFRAME_BLOB_CHECK_COUNT
        or not adapter_validation.keyframe_blob_validation_passed_attested
        or not local_crosscheck_passed
    ):
        raise GeometryRunnerError("M2 local dataset acceptance gate failed")

    svg = render_scalar_diagnostic_svg(scalar_diagnostic)
    _write_local_diagnostic(source_root=snapshot.source_root, svg=svg)
    try:
        synthetic = build_synthetic_geometry_validation(
            manifest,
            source_root=snapshot.source_root,
        )
        covariance = build_covariance_validation(
            manifest,
            source_root=snapshot.source_root,
        )
        dataset = DatasetValidationV1(
            profile=manifest.dataset.profile,
            expected_headline_counts=manifest.dataset.expected_headline_counts,
            headline_profile_passed_attested=(adapter_validation.headline_profile_passed_attested),
            structural_integrity_passed_attested=(
                adapter_validation.structural_integrity_passed_attested
            ),
            keyframe_blob_check_count=(adapter_validation.keyframe_blob_check_count),
            keyframe_blob_validation_passed_attested=(
                adapter_validation.keyframe_blob_validation_passed_attested
            ),
            local_projection_crosscheck_passed_attested=local_crosscheck_passed,
            diagnostic_svg_generated_attested=True,
            dataset_authentication=manifest.dataset.dataset_authentication,
            all_checks_passed=True,
        )
        validation = GeometryValidationV1(
            schema="ffb.geometry-validation/v1",
            run_id=run_id,
            manifest_sha256=manifest_digest,
            dataset_terms=manifest.public_dataset_terms,
            dataset_validation=dataset,
            synthetic_geometry_validation=synthetic,
            covariance_validation=covariance,
            all_checks_passed=(
                dataset.all_checks_passed
                and synthetic.all_checks_passed
                and covariance.all_checks_passed
            ),
        )
    except (GeometryValidationComputationError, ValidationError, ValueError):
        raise GeometryRunnerError("M2 synthetic acceptance gate failed") from None
    if not validation.all_checks_passed:
        raise GeometryRunnerError("M2 acceptance gate failed")

    try:
        environment = collect_runtime_environment()
    except (OSError, ProvenanceError, ValueError):
        raise GeometryRunnerError("M2 runtime environment validation failed") from None
    _final_snapshot(manifest_path, snapshot)
    ended_at = datetime.now(UTC)
    run = RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=manifest_digest,
        package_version=snapshot.package_version,
        git_revision=snapshot.git_revision,
        source_dirty=False,
        lockfile_sha256=snapshot.lockfile_sha256,
        command=manifest.artifact.logical_command,
        environment=environment,
        started_at=started_at,
        ended_at=ended_at,
        status="succeeded",
        artifact_sha256="0" * 64,
    )
    try:
        return write_geometry_validation_artifact(
            GeometryArtifactWriteRequest(
                manifest=manifest,
                validation=validation,
                run=run,
            ),
            snapshot.source_root / output_dir,
            source_root=snapshot.source_root,
            git_metadata_dirs=(snapshot.git_dir, snapshot.git_common_dir),
        )
    except (ArtifactValidationError, FileExistsError, OSError, ValueError):
        raise GeometryRunnerError("M2 artifact publication failed") from None
