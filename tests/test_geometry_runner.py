from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import fusion_fault_bench.geometry_runner as geometry_runner
from fusion_fault_bench.adapters.nuscenes import NuScenesMiniValidation
from fusion_fault_bench.contracts.result_v1alpha1 import RuntimeEnvironment
from fusion_fault_bench.provenance import CleanSourceSnapshot

MANIFEST_PATH = Path("examples/validation/m2-geometry-v1.json")
OUTPUT_PATH = Path("reports/generated/m2-geometry")


def _snapshot(
    *,
    source_root: Path | None = None,
    revision: str = "a" * 40,
    manifest_relative_path: str = MANIFEST_PATH.as_posix(),
) -> CleanSourceSnapshot:
    root = Path.cwd().resolve() if source_root is None else source_root.resolve()
    git_dir = (root / ".git").resolve()
    return CleanSourceSnapshot(
        source_root=root,
        git_revision=revision,
        git_dir=git_dir,
        git_common_dir=git_dir,
        lockfile_sha256=hashlib.sha256((Path.cwd() / "uv.lock").read_bytes()).hexdigest(),
        package_version="0.1.0",
        manifest_relative_path=manifest_relative_path,
    )


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        python_version="3.12.13",
        os_name="TestOS",
        os_release="1.0",
        machine="test-machine",
        cpu_model="test CPU",
        logical_cpu_count=4,
        memory_bytes=8_000_000_000,
    )


def _adapter_validation(
    *,
    headline: bool = True,
    keyframe_count: int = 808,
) -> NuScenesMiniValidation:
    return NuScenesMiniValidation(
        headline_profile_passed_attested=headline,
        structural_integrity_passed_attested=True,
        keyframe_blob_check_count=keyframe_count,
        keyframe_blob_validation_passed_attested=True,
    )


def _patch_success_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    snapshot: CleanSourceSnapshot,
    validation: NuScenesMiniValidation | None = None,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        geometry_runner,
        "discover_clean_source",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(
        geometry_runner,
        "verify_locked_execution",
        lambda _snapshot: None,
    )
    metadata = SimpleNamespace(
        validation=_adapter_validation() if validation is None else validation
    )
    monkeypatch.setattr(geometry_runner, "load_nuscenes_mini", lambda _root: metadata)
    monkeypatch.setattr(
        geometry_runner,
        "build_scalar_projection_diagnostic",
        lambda _root: object(),
    )
    monkeypatch.setattr(
        geometry_runner,
        "build_production_projection_diagnostic",
        lambda _metadata: object(),
    )
    monkeypatch.setattr(
        geometry_runner,
        "projection_crosscheck_passes",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        geometry_runner,
        "render_scalar_diagnostic_svg",
        lambda _diagnostic: "<svg/>\n",
    )
    monkeypatch.setattr(
        geometry_runner,
        "_write_local_diagnostic",
        lambda **kwargs: captured.update(diagnostic=kwargs),
    )
    monkeypatch.setattr(
        geometry_runner,
        "collect_runtime_environment",
        _environment,
    )
    sentinel = object()

    def capture_write(request, destination, **kwargs):
        captured["request"] = request
        captured["destination"] = destination
        captured["write_kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(
        geometry_runner,
        "write_geometry_validation_artifact",
        capture_write,
    )
    captured["sentinel"] = sentinel
    return captured


def test_runner_builds_passing_sanitized_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot()
    captured = _patch_success_dependencies(monkeypatch, snapshot=snapshot)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    monkeypatch.setenv("NUSCENES_ROOT", str(dataset_root.resolve()))

    observed = geometry_runner.run_geometry_validation(
        MANIFEST_PATH,
        dataset_root_env="NUSCENES_ROOT",
        output_dir=OUTPUT_PATH,
    )

    assert observed is captured["sentinel"]
    request = captured["request"]
    assert request.validation.all_checks_passed
    assert request.validation.dataset_validation.keyframe_blob_check_count == 808
    assert request.validation.synthetic_geometry_validation.all_checks_passed
    assert request.validation.covariance_validation.all_checks_passed
    assert request.run.command == tuple(request.manifest.artifact.logical_command)
    assert request.run.source_dirty is False
    assert captured["destination"] == snapshot.source_root / OUTPUT_PATH
    assert captured["diagnostic"]["svg"] == "<svg/>\n"


def test_local_diagnostic_is_private_exclusive_and_token_free(tmp_path: Path) -> None:
    geometry_runner._write_local_diagnostic(
        source_root=tmp_path,
        svg="<svg><circle/></svg>\n",
    )
    destination = tmp_path / "reports/generated/m2-geometry-diagnostic.svg"

    assert destination.read_text(encoding="utf-8") == "<svg><circle/></svg>\n"
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="diagnostic publication failed",
    ):
        geometry_runner._write_local_diagnostic(
            source_root=tmp_path,
            svg="<svg/>\n",
        )
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="diagnostic privacy validation failed",
    ):
        geometry_runner._write_local_diagnostic(
            source_root=tmp_path / "other",
            svg=f"<svg>{tmp_path / 'other'}</svg>\n",
        )


def test_local_diagnostic_parent_swap_cannot_redirect_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "reports/generated"
    moved_parent = tmp_path / "reports/generated-pinned"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_write = geometry_runner._write_exclusive_diagnostic_at

    def swap_then_write(
        directory_fd: int,
        name: str,
        value: bytes,
    ) -> None:
        parent.rename(moved_parent)
        parent.symlink_to(outside, target_is_directory=True)
        real_write(directory_fd, name, value)

    monkeypatch.setattr(
        geometry_runner,
        "_write_exclusive_diagnostic_at",
        swap_then_write,
    )

    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="diagnostic publication failed",
    ):
        geometry_runner._write_local_diagnostic(
            source_root=tmp_path,
            svg="<svg/>\n",
        )

    assert not (outside / "m2-geometry-diagnostic.svg").exists()
    assert not (moved_parent / "m2-geometry-diagnostic.svg").exists()


def test_dataset_root_policy_rejects_name_relative_inside_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="fixed dataset environment name",
    ):
        geometry_runner._resolve_dataset_root(
            environment_name="OTHER_ROOT",
            source_root=source_root,
        )

    monkeypatch.delenv("NUSCENES_ROOT", raising=False)
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="environment is unavailable",
    ):
        geometry_runner._resolve_dataset_root(
            environment_name="NUSCENES_ROOT",
            source_root=source_root,
        )

    monkeypatch.setenv("NUSCENES_ROOT", "relative-dataset")
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="must be absolute",
    ):
        geometry_runner._resolve_dataset_root(
            environment_name="NUSCENES_ROOT",
            source_root=source_root,
        )

    inside = source_root / "dataset"
    inside.mkdir()
    monkeypatch.setenv("NUSCENES_ROOT", str(inside))
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="outside the source checkout",
    ):
        geometry_runner._resolve_dataset_root(
            environment_name="NUSCENES_ROOT",
            source_root=source_root,
        )

    monkeypatch.setenv("NUSCENES_ROOT", str(tmp_path))
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="outside the source checkout",
    ):
        geometry_runner._resolve_dataset_root(
            environment_name="NUSCENES_ROOT",
            source_root=source_root,
        )

    external = tmp_path / "external"
    external.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(external, target_is_directory=True)
    monkeypatch.setenv("NUSCENES_ROOT", str(linked))
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="root validation failed",
    ):
        geometry_runner._resolve_dataset_root(
            environment_name="NUSCENES_ROOT",
            source_root=source_root,
        )


def test_runner_rejects_nonfrozen_paths_and_failed_dataset_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="frozen repository-relative output path",
    ):
        geometry_runner.run_geometry_validation(
            MANIFEST_PATH,
            dataset_root_env="NUSCENES_ROOT",
            output_dir=tmp_path,
        )

    snapshot = _snapshot()
    _patch_success_dependencies(
        monkeypatch,
        snapshot=snapshot,
        validation=_adapter_validation(headline=False),
    )
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    monkeypatch.setenv("NUSCENES_ROOT", str(dataset_root.resolve()))
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="local dataset acceptance gate failed",
    ):
        geometry_runner.run_geometry_validation(
            MANIFEST_PATH,
            dataset_root_env="NUSCENES_ROOT",
            output_dir=OUTPUT_PATH,
        )


def test_runner_sanitizes_local_geometry_and_publication_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot()
    _patch_success_dependencies(monkeypatch, snapshot=snapshot)
    dataset_root = tmp_path / "dataset-secret"
    dataset_root.mkdir()
    monkeypatch.setenv("NUSCENES_ROOT", str(dataset_root.resolve()))
    monkeypatch.setattr(
        geometry_runner,
        "load_nuscenes_mini",
        lambda _root: (_ for _ in ()).throw(OSError(str(dataset_root))),
    )
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="local dataset geometry validation failed",
    ) as captured:
        geometry_runner.run_geometry_validation(
            MANIFEST_PATH,
            dataset_root_env="NUSCENES_ROOT",
            output_dir=OUTPUT_PATH,
        )
    assert str(dataset_root) not in str(captured.value)

    captured_dependencies = _patch_success_dependencies(
        monkeypatch,
        snapshot=snapshot,
    )
    monkeypatch.setattr(
        geometry_runner,
        "write_geometry_validation_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(str(dataset_root))),
    )
    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="artifact publication failed",
    ) as publication:
        geometry_runner.run_geometry_validation(
            MANIFEST_PATH,
            dataset_root_env="NUSCENES_ROOT",
            output_dir=OUTPUT_PATH,
        )
    assert str(dataset_root) not in str(publication.value)
    assert captured_dependencies["diagnostic"]["svg"] == "<svg/>\n"


def test_runner_rejects_source_change_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _snapshot()
    changed = _snapshot(revision="b" * 40)
    snapshots = iter((initial, changed))
    monkeypatch.setattr(
        geometry_runner,
        "discover_clean_source",
        lambda _path: next(snapshots),
    )
    monkeypatch.setattr(
        geometry_runner,
        "verify_locked_execution",
        lambda _snapshot: None,
    )
    _patch_success_dependencies(monkeypatch, snapshot=initial)
    snapshots = iter((initial, changed))
    monkeypatch.setattr(
        geometry_runner,
        "discover_clean_source",
        lambda _path: next(snapshots),
    )
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    monkeypatch.setenv("NUSCENES_ROOT", str(dataset_root.resolve()))

    with pytest.raises(
        geometry_runner.GeometryRunnerError,
        match="source provenance changed",
    ):
        geometry_runner.run_geometry_validation(
            MANIFEST_PATH,
            dataset_root_env="NUSCENES_ROOT",
            output_dir=OUTPUT_PATH,
        )
