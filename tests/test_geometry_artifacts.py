from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import fusion_fault_bench.geometry_artifacts as geometry_artifact_module
from fusion_fault_bench.artifacts import (
    ArtifactValidationError,
    canonical_json_bytes,
    compute_artifact_digest,
    compute_run_record_digest,
    derive_run_id,
)
from fusion_fault_bench.canonical import sha256_digest
from fusion_fault_bench.contracts.artifact_v1alpha1 import SuccessMarkerV1Alpha1
from fusion_fault_bench.contracts.geometry_validation_v1 import (
    FROZEN_GEOMETRY_MANIFEST_SHA256,
    GEOMETRY_ARTIFACT_CONTRACT,
    GEOMETRY_ARTIFACT_PATHS,
    GEOMETRY_INDEXED_PAYLOAD_PATHS,
    GEOMETRY_MANIFEST_FILE,
    GEOMETRY_MEMBER_BYTE_CAP,
    GEOMETRY_PAYLOAD_INDEX_FILE,
    GEOMETRY_RUN_FILE,
    GEOMETRY_SUCCESS_FILE,
    GEOMETRY_VALIDATION_FILE,
    CovarianceEntryV1,
    CovarianceValidationV1,
    DatasetValidationV1,
    GeometryPayloadIndexV1,
    GeometryValidationManifestV1,
    GeometryValidationV1,
    PayloadFileEntryV1,
    SyntheticGeometryValidationV1,
)
from fusion_fault_bench.contracts.result_v1alpha1 import (
    RunRecordV1Alpha1,
    RuntimeEnvironment,
)
from fusion_fault_bench.geometry_artifacts import (
    GeometryArtifactWriteRequest,
    load_geometry_validation_artifact,
    validate_geometry_validation_bundle,
    write_geometry_validation_artifact,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "examples/validation/m2-geometry-v1.json"
PLACEHOLDER_DIGEST = "0" * 64


def _manifest() -> GeometryValidationManifestV1:
    return GeometryValidationManifestV1.model_validate_json(MANIFEST_PATH.read_bytes())


def _run(
    manifest: GeometryValidationManifestV1,
    *,
    started_at: datetime | None = None,
) -> RunRecordV1Alpha1:
    manifest_sha256 = sha256_digest(manifest)
    git_revision = "a" * 40
    lockfile_sha256 = "b" * 64
    package_version = "0.1.0"
    run_id = derive_run_id(
        manifest_sha256=manifest_sha256,
        git_revision=git_revision,
        lockfile_sha256=lockfile_sha256,
        package_version=package_version,
        artifact_contract=GEOMETRY_ARTIFACT_CONTRACT,
    )
    started = datetime(2026, 7, 29, 12, 0, tzinfo=UTC) if started_at is None else started_at
    return RunRecordV1Alpha1(
        schema="ffb.run/v1alpha1",
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        package_version=package_version,
        git_revision=git_revision,
        source_dirty=False,
        lockfile_sha256=lockfile_sha256,
        command=manifest.artifact.logical_command,
        environment=RuntimeEnvironment(
            python_version="3.12.13",
            os_name="Darwin",
            os_release="24.5.0",
            machine="arm64",
            cpu_model="Test CPU",
            logical_cpu_count=4,
            memory_bytes=8 * 1024**3,
        ),
        started_at=started,
        ended_at=started + timedelta(seconds=1),
        status="succeeded",
        artifact_sha256=PLACEHOLDER_DIGEST,
    )


def _validation(
    manifest: GeometryValidationManifestV1,
    run: RunRecordV1Alpha1,
) -> GeometryValidationV1:
    entries = tuple(
        CovarianceEntryV1(
            entry=entry,
            absolute_error_m2=0.0,
            allowed_error_m2=1.0,
            gate_ratio=0.0,
            passed=True,
        )
        for entry in ("xx", "xy", "yy")
    )
    return GeometryValidationV1(
        schema="ffb.geometry-validation/v1",
        run_id=run.run_id,
        manifest_sha256=run.manifest_sha256,
        dataset_terms=manifest.public_dataset_terms,
        dataset_validation=DatasetValidationV1(
            profile=manifest.dataset.profile,
            expected_headline_counts=manifest.dataset.expected_headline_counts,
            headline_profile_passed_attested=True,
            structural_integrity_passed_attested=True,
            keyframe_blob_check_count=808,
            keyframe_blob_validation_passed_attested=True,
            local_projection_crosscheck_passed_attested=True,
            diagnostic_svg_generated_attested=True,
            dataset_authentication=manifest.dataset.dataset_authentication,
            all_checks_passed=True,
        ),
        synthetic_geometry_validation=SyntheticGeometryValidationV1(
            fixture_id=manifest.synthetic_fixture.fixture_id,
            fixture_file_sha256=manifest.synthetic_fixture.file_sha256,
            rotation_max_abs_error=0.0,
            translation_max_abs_error_m=0.0,
            point_round_trip_max_abs_error_m=0.0,
            quaternion_sign_max_abs_error=0.0,
            projection_max_abs_error_px=0.0,
            depth_max_abs_error_m=0.0,
            box_corner_max_abs_error_m=0.0,
            all_checks_passed=True,
        ),
        covariance_validation=CovarianceValidationV1(
            finite_difference_max_abs_error=0.0,
            monte_carlo_sample_count=200_000,
            covariance_entries=entries,
            covariance_entry_max_abs_error_m2=0.0,
            covariance_entry_max_allowed_error_m2=1.0,
            covariance_entry_max_gate_ratio=0.0,
            actual_sampling_gate_passed=True,
            reported_role_separation_passed_attested=True,
            all_checks_passed=True,
        ),
        all_checks_passed=True,
    )


def _request() -> GeometryArtifactWriteRequest:
    manifest = _manifest()
    run = _run(manifest)
    return GeometryArtifactWriteRequest(
        manifest=manifest,
        validation=_validation(manifest, run),
        run=run,
    )


def _copy_artifact(source: Path, tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(source, destination)
    return destination


def _rechain_false_result(root: Path) -> None:
    validation_value = json.loads((root / GEOMETRY_VALIDATION_FILE).read_bytes())
    validation_value["dataset_validation"]["headline_profile_passed_attested"] = False
    validation_value["dataset_validation"]["all_checks_passed"] = False
    validation_value["all_checks_passed"] = False
    validation = GeometryValidationV1.model_validate_json(json.dumps(validation_value))
    validation_bytes = canonical_json_bytes(validation)
    (root / GEOMETRY_VALIDATION_FILE).write_bytes(validation_bytes)

    index = GeometryPayloadIndexV1.model_validate_json(
        (root / GEOMETRY_PAYLOAD_INDEX_FILE).read_bytes()
    )
    files = []
    for entry in index.files:
        value = (root / entry.path).read_bytes()
        files.append(
            PayloadFileEntryV1(
                path=entry.path,
                byte_length=len(value),
                sha256=hashlib.sha256(value).hexdigest(),
            )
        )
    index_value = index.model_dump(mode="python", by_alias=True)
    index_value["files"] = tuple(files)
    index = GeometryPayloadIndexV1.model_validate(index_value)
    index_bytes = canonical_json_bytes(index)
    (root / GEOMETRY_PAYLOAD_INDEX_FILE).write_bytes(index_bytes)

    artifact_sha256 = compute_artifact_digest(index_bytes)
    run = RunRecordV1Alpha1.model_validate_json((root / GEOMETRY_RUN_FILE).read_bytes())
    run_value = run.model_dump(mode="python", by_alias=True)
    run_value["artifact_sha256"] = artifact_sha256
    run = RunRecordV1Alpha1.model_validate(run_value)
    run_bytes = canonical_json_bytes(run)
    (root / GEOMETRY_RUN_FILE).write_bytes(run_bytes)

    success = SuccessMarkerV1Alpha1(
        schema="ffb.success/v1alpha1",
        artifact_sha256=artifact_sha256,
        run_sha256=compute_run_record_digest(run_bytes),
    )
    (root / GEOMETRY_SUCCESS_FILE).write_bytes(canonical_json_bytes(success))


@pytest.fixture
def written_geometry_artifact(tmp_path: Path):
    return write_geometry_validation_artifact(
        _request(),
        tmp_path / "artifact",
        git_metadata_dirs=(),
    )


def test_frozen_manifest_digest_and_run_id_domains() -> None:
    manifest = _manifest()
    assert sha256_digest(manifest) == FROZEN_GEOMETRY_MANIFEST_SHA256

    fields = {
        "manifest_sha256": "a" * 64,
        "git_revision": "b" * 40,
        "lockfile_sha256": "c" * 64,
        "package_version": "0.1.0",
    }

    def expected(contract: bytes) -> str:
        values = (
            fields["manifest_sha256"].encode(),
            fields["git_revision"].encode(),
            fields["lockfile_sha256"].encode(),
            fields["package_version"].encode(),
            contract,
        )
        preimage = b"fusion-fault-bench/run-id/v1\x00" + b"".join(
            len(value).to_bytes(4, "big") + value for value in values
        )
        return f"run:{hashlib.sha256(preimage).hexdigest()}"

    assert derive_run_id(**fields) == expected(b"ffb.scientific-payload/v1")
    geometry_run_id = derive_run_id(
        **fields,
        artifact_contract=GEOMETRY_ARTIFACT_CONTRACT,
    )
    assert geometry_run_id == expected(GEOMETRY_ARTIFACT_CONTRACT.encode())
    assert geometry_run_id != derive_run_id(**fields)


def test_manifest_layout_allowlists_and_command_are_code_owned() -> None:
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["public_summary_allowlist"]["dataset_validation_fields"].reverse()
    with pytest.raises(ValidationError):
        GeometryValidationManifestV1.model_validate_json(json.dumps(raw))

    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["artifact"]["ordered_files"].append("dataset.json")
    with pytest.raises(ValidationError):
        GeometryValidationManifestV1.model_validate_json(json.dumps(raw))

    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["artifact"]["logical_command"].extend(("--dataset-root", "../../nuscenes"))
    with pytest.raises(ValidationError):
        GeometryValidationManifestV1.model_validate_json(json.dumps(raw))


def test_result_contract_rejects_leakage_and_derived_contradictions() -> None:
    request = _request()
    raw = request.validation.model_dump(mode="python", by_alias=True)
    raw["dataset_root"] = "/private/datasets/nuscenes"
    with pytest.raises(ValidationError):
        GeometryValidationV1.model_validate(raw)

    raw = request.validation.model_dump(mode="python", by_alias=True)
    raw["dataset_validation"]["headline_profile_passed_attested"] = False
    with pytest.raises(ValidationError, match="conjunction"):
        GeometryValidationV1.model_validate(raw)

    raw = request.validation.model_dump(mode="python", by_alias=True)
    raw["covariance_validation"]["covariance_entries"][0]["gate_ratio"] = 0.5
    with pytest.raises(ValidationError, match="gate_ratio"):
        GeometryValidationV1.model_validate(raw)

    raw = request.validation.model_dump(mode="python", by_alias=True)
    raw["covariance_validation"]["covariance_entries"][0]["entry"] = "yy"
    with pytest.raises(ValidationError, match="exact xx, xy, yy order"):
        GeometryValidationV1.model_validate(raw)

    raw = request.validation.model_dump(mode="python", by_alias=True)
    raw["synthetic_geometry_validation"]["rotation_max_abs_error"] = 1.1e-12
    with pytest.raises(ValidationError, match="frozen error tolerances"):
        GeometryValidationV1.model_validate(raw)


def test_bundle_rejects_wrong_logical_command_and_m1_run_identity() -> None:
    request = _request()
    run_value = request.run.model_dump(mode="python", by_alias=True)
    run_value["command"] = (
        "ffb",
        "geometry",
        "validate",
        "examples/validation/m2-geometry-v1.json",
        "--dataset-root",
        "datasets/nuscenes",
        "--output-dir",
        "reports/generated/m2-geometry",
    )
    wrong_command = RunRecordV1Alpha1.model_validate(run_value)
    with pytest.raises(ArtifactValidationError, match="frozen logical command"):
        validate_geometry_validation_bundle(
            request.manifest,
            request.validation,
            wrong_command,
        )

    run_value = request.run.model_dump(mode="python", by_alias=True)
    run_value["run_id"] = derive_run_id(
        manifest_sha256=request.run.manifest_sha256,
        git_revision=request.run.git_revision,
        lockfile_sha256=request.run.lockfile_sha256,
        package_version=request.run.package_version,
    )
    wrong_identity = RunRecordV1Alpha1.model_validate(run_value)
    with pytest.raises(ArtifactValidationError, match="run_id"):
        validate_geometry_validation_bundle(
            request.manifest,
            request.validation,
            wrong_identity,
        )

    run_value = request.run.model_dump(mode="python", by_alias=True)
    run_value["environment"]["cpu_model"] = "/private/datasets/nuscenes"
    path_cpu = RunRecordV1Alpha1.model_validate(run_value)
    with pytest.raises(ArtifactValidationError, match="sanitized hardware name"):
        validate_geometry_validation_bundle(
            request.manifest,
            request.validation,
            path_cpu,
        )


def test_writer_round_trips_exact_five_file_identity_graph(
    written_geometry_artifact,
) -> None:
    loaded = written_geometry_artifact
    assert {path.name for path in loaded.path.iterdir()} == set(GEOMETRY_ARTIFACT_PATHS)
    assert all(path.is_file() and not path.is_symlink() for path in loaded.path.iterdir())
    assert sum(path.stat().st_size for path in loaded.path.iterdir()) < 5 * 1024 * 1024
    for path in loaded.path.iterdir():
        data = path.read_bytes()
        assert len(data) <= GEOMETRY_MEMBER_BYTE_CAP
        assert data.endswith(b"\n")
        assert b"\n" not in data[:-1]

    assert tuple(entry.path for entry in loaded.payload_index.files) == (
        GEOMETRY_INDEXED_PAYLOAD_PATHS
    )
    for entry in loaded.payload_index.files:
        value = (loaded.path / entry.path).read_bytes()
        assert entry.byte_length == len(value)
        assert entry.sha256 == hashlib.sha256(value).hexdigest()

    index_bytes = (loaded.path / GEOMETRY_PAYLOAD_INDEX_FILE).read_bytes()
    run_bytes = (loaded.path / GEOMETRY_RUN_FILE).read_bytes()
    marker = SuccessMarkerV1Alpha1.model_validate_json(
        (loaded.path / GEOMETRY_SUCCESS_FILE).read_bytes()
    )
    assert loaded.artifact_sha256 == compute_artifact_digest(index_bytes)
    assert loaded.run_sha256 == compute_run_record_digest(run_bytes)
    assert loaded.run.artifact_sha256 == loaded.artifact_sha256
    assert marker.artifact_sha256 == loaded.artifact_sha256
    assert marker.run_sha256 == loaded.run_sha256
    assert load_geometry_validation_artifact(loaded.path) == loaded


def test_success_is_written_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    original = geometry_artifact_module._write_exclusive_at

    def record_write(directory_fd: int, name: str, value: bytes) -> None:
        observed.append(name)
        original(directory_fd, name, value)

    monkeypatch.setattr(geometry_artifact_module, "_write_exclusive_at", record_write)
    write_geometry_validation_artifact(
        _request(),
        tmp_path / "artifact",
        git_metadata_dirs=(),
    )
    assert tuple(observed) == GEOMETRY_ARTIFACT_PATHS
    assert observed[-1] == GEOMETRY_SUCCESS_FILE


def test_timestamp_only_rerun_preserves_stable_payload_identity(tmp_path: Path) -> None:
    first_request = _request()
    first = write_geometry_validation_artifact(
        first_request,
        tmp_path / "first",
        git_metadata_dirs=(),
    )
    later_run = _run(
        first_request.manifest,
        started_at=first_request.run.started_at + timedelta(minutes=1),
    )
    second = write_geometry_validation_artifact(
        GeometryArtifactWriteRequest(
            manifest=first_request.manifest,
            validation=_validation(first_request.manifest, later_run),
            run=later_run,
        ),
        tmp_path / "second",
        git_metadata_dirs=(),
    )

    assert first.artifact_sha256 == second.artifact_sha256
    assert first.run_sha256 != second.run_sha256
    for name in (*GEOMETRY_INDEXED_PAYLOAD_PATHS, GEOMETRY_PAYLOAD_INDEX_FILE):
        assert (first.path / name).read_bytes() == (second.path / name).read_bytes()
    assert (first.path / GEOMETRY_RUN_FILE).read_bytes() != (
        second.path / GEOMETRY_RUN_FILE
    ).read_bytes()
    assert (first.path / GEOMETRY_SUCCESS_FILE).read_bytes() != (
        second.path / GEOMETRY_SUCCESS_FILE
    ).read_bytes()


def test_loader_rejects_semantic_failure_after_full_identity_rechain(
    written_geometry_artifact,
    tmp_path: Path,
) -> None:
    copied = _copy_artifact(written_geometry_artifact.path, tmp_path, "false-result")
    _rechain_false_result(copied)
    with pytest.raises(ArtifactValidationError, match="contradicts"):
        load_geometry_validation_artifact(copied)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        (
            "extra-file",
            lambda root: (root / "extra.json").write_bytes(b"{}\n"),
        ),
        (
            "noncanonical",
            lambda root: (root / GEOMETRY_RUN_FILE).write_bytes(
                (root / GEOMETRY_RUN_FILE).read_bytes().replace(b'{"', b'{ "', 1)
            ),
        ),
        (
            "oversized",
            lambda root: (root / GEOMETRY_VALIDATION_FILE).write_bytes(
                b"x" * (GEOMETRY_MEMBER_BYTE_CAP + 1)
            ),
        ),
    ],
)
def test_loader_rejects_tree_and_canonical_mutations(
    written_geometry_artifact,
    tmp_path: Path,
    name: str,
    mutate: Any,
) -> None:
    copied = _copy_artifact(written_geometry_artifact.path, tmp_path, name)
    mutate(copied)
    with pytest.raises(ArtifactValidationError):
        load_geometry_validation_artifact(copied)


def test_loader_rejects_member_and_root_symlinks(
    written_geometry_artifact,
    tmp_path: Path,
) -> None:
    linked_member = _copy_artifact(written_geometry_artifact.path, tmp_path, "linked-member")
    run_path = linked_member / GEOMETRY_RUN_FILE
    run_path.unlink()
    run_path.symlink_to(written_geometry_artifact.path / GEOMETRY_RUN_FILE)
    with pytest.raises(ArtifactValidationError, match="symlink"):
        load_geometry_validation_artifact(linked_member)

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(written_geometry_artifact.path, target_is_directory=True)
    with pytest.raises(ArtifactValidationError, match="symlink"):
        load_geometry_validation_artifact(linked_root)


def test_geometry_loader_binds_reads_to_the_scanned_inode(
    written_geometry_artifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = _copy_artifact(
        written_geometry_artifact.path,
        tmp_path,
        "inode-swap",
    )
    manifest_bytes = (copied / GEOMETRY_MANIFEST_FILE).read_bytes()
    replacement = tmp_path / "replacement-manifest.json"
    original_read = geometry_artifact_module._read_member
    swapped = False

    def swap_before_read(
        root: Path,
        name: str,
        *,
        expected_stat,
    ) -> bytes:
        nonlocal swapped
        if name == GEOMETRY_MANIFEST_FILE and not swapped:
            replacement.write_bytes(manifest_bytes)
            replacement.replace(root / name)
            swapped = True
        return original_read(
            root,
            name,
            expected_stat=expected_stat,
        )

    monkeypatch.setattr(
        geometry_artifact_module,
        "_read_member",
        swap_before_read,
    )
    with pytest.raises(
        ArtifactValidationError,
        match="member changed during validation",
    ):
        load_geometry_validation_artifact(copied)


def test_writer_never_replaces_existing_or_dangling_destination(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_geometry_validation_artifact(
            _request(),
            existing,
            git_metadata_dirs=(),
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        write_geometry_validation_artifact(
            _request(),
            dangling,
            git_metadata_dirs=(),
        )
    assert dangling.is_symlink()


def test_failed_write_cleans_geometry_member_and_preserves_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    keep = parent / "keep"
    keep.mkdir()
    original = geometry_artifact_module._write_exclusive_at

    def fail_after_validation(directory_fd: int, name: str, value: bytes) -> None:
        original(directory_fd, name, value)
        if name == GEOMETRY_VALIDATION_FILE:
            raise OSError("injected")

    monkeypatch.setattr(
        geometry_artifact_module,
        "_write_exclusive_at",
        fail_after_validation,
    )
    with pytest.raises(ArtifactValidationError, match="publication failed"):
        write_geometry_validation_artifact(
            _request(),
            parent / "artifact",
            git_metadata_dirs=(),
        )

    assert list(parent.iterdir()) == [keep]
