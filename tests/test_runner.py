from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fusion_fault_bench.artifacts import INDEXED_PAYLOAD_PATHS, load_artifact
from fusion_fault_bench.cli import main
from fusion_fault_bench.contracts.io import load_manifest
from fusion_fault_bench.contracts.manifest_v1alpha1 import (
    AdditivePositionBiasFault,
    AnalyticCrossoverManifest,
)
from fusion_fault_bench.contracts.result_v1alpha1 import RuntimeEnvironment
from fusion_fault_bench.provenance import CleanSourceSnapshot
from fusion_fault_bench.runner import (
    preflight_analytic_manifest,
    run_analytic_experiment,
)
from fusion_fault_bench.validation import build_analytic_validation


def _manifest(name: str = "analytic-bias-v1alpha1.json") -> AnalyticCrossoverManifest:
    manifest = load_manifest(Path("examples/manifests") / name)
    assert isinstance(manifest, AnalyticCrossoverManifest)
    return manifest


def _snapshot(manifest_path: Path) -> CleanSourceSnapshot:
    source_root = Path.cwd().resolve()
    git_dir = (source_root / ".git").resolve()
    return CleanSourceSnapshot(
        source_root=source_root,
        git_revision="a" * 40,
        git_dir=git_dir,
        git_common_dir=git_dir,
        lockfile_sha256=hashlib.sha256((source_root / "uv.lock").read_bytes()).hexdigest(),
        package_version="0.1.0",
        manifest_relative_path=manifest_path.as_posix(),
    )


def _environment() -> RuntimeEnvironment:
    return RuntimeEnvironment(
        python_version="3.12.0",
        os_name="TestOS",
        os_release="1.0",
        machine="test-machine",
        cpu_model="test CPU",
        logical_cpu_count=4,
        memory_bytes=8_000_000_000,
    )


def test_runner_writes_a_strict_artifact_and_bundle_cli_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = Path("examples/manifests/analytic-noise-correct-v1alpha1.json")
    snapshot = _snapshot(manifest_path)
    monkeypatch.setattr(
        "fusion_fault_bench.runner.discover_clean_source",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(
        "fusion_fault_bench.runner.collect_runtime_environment",
        _environment,
    )
    destination = tmp_path / "artifact"

    artifact = run_analytic_experiment(manifest_path, output_dir=destination)

    assert artifact.path == destination.resolve()
    assert artifact.run.git_revision == "a" * 40
    assert artifact.run.command == (
        "ffb",
        "run",
        "examples/manifests/analytic-noise-correct-v1alpha1.json",
        "--output-dir",
        "reports/generated/analytic-camera-noise-correctly-reported-3ea7ffc2949c",
    )
    assert artifact.analytic_validation.all_monte_carlo_checks_passed
    assert load_artifact(destination).artifact_sha256 == artifact.artifact_sha256
    assert main(["bundle", "validate", str(destination)]) == 0
    assert capsys.readouterr().out.startswith("valid ffb.scientific-payload/v1 ")
    monkeypatch.setattr(
        "fusion_fault_bench.cli.run_analytic_experiment",
        lambda _path, *, output_dir: artifact,
    )
    assert (
        main(
            [
                "run",
                str(manifest_path),
                "--output-dir",
                str(tmp_path / "ignored-by-stub"),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.startswith(f"wrote {destination.resolve()} ")


def test_two_runs_have_identical_indexed_scientific_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = Path("examples/manifests/analytic-noise-underreported-v1alpha1.json")
    snapshot = _snapshot(manifest_path)
    monkeypatch.setattr(
        "fusion_fault_bench.runner.discover_clean_source",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(
        "fusion_fault_bench.runner.collect_runtime_environment",
        _environment,
    )

    first = run_analytic_experiment(manifest_path, output_dir=tmp_path / "first")
    second = run_analytic_experiment(manifest_path, output_dir=tmp_path / "second")

    assert first.artifact_sha256 == second.artifact_sha256
    for member in (*INDEXED_PAYLOAD_PATHS, "payload-index.json"):
        assert (first.path / member).read_bytes() == (second.path / member).read_bytes()


def test_runner_refuses_a_failed_analytic_acceptance_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = Path("examples/manifests/analytic-noise-correct-v1alpha1.json")
    snapshot = _snapshot(manifest_path)
    monkeypatch.setattr(
        "fusion_fault_bench.runner.discover_clean_source",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(
        "fusion_fault_bench.runner.collect_runtime_environment",
        _environment,
    )

    def failed_validation(*args: object, **kwargs: object) -> object:
        validation = build_analytic_validation(*args, **kwargs)  # type: ignore[arg-type]
        return validation.model_copy(update={"all_monte_carlo_checks_passed": False})

    monkeypatch.setattr(
        "fusion_fault_bench.runner.build_analytic_validation",
        failed_validation,
    )
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="6-SE acceptance gate"):
        run_analytic_experiment(manifest_path, output_dir=destination)
    assert not destination.exists()


def test_runner_refuses_source_provenance_change_during_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = Path("examples/manifests/analytic-noise-correct-v1alpha1.json")
    initial = _snapshot(manifest_path)
    changed = CleanSourceSnapshot(
        source_root=initial.source_root,
        git_revision="c" * 40,
        git_dir=initial.git_dir,
        git_common_dir=initial.git_common_dir,
        lockfile_sha256=initial.lockfile_sha256,
        package_version=initial.package_version,
        manifest_relative_path=initial.manifest_relative_path,
    )
    snapshots = iter((initial, changed))
    monkeypatch.setattr(
        "fusion_fault_bench.runner.discover_clean_source",
        lambda _path: next(snapshots),
    )
    monkeypatch.setattr(
        "fusion_fault_bench.runner.collect_runtime_environment",
        _environment,
    )

    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="changed during"):
        run_analytic_experiment(manifest_path, output_dir=destination)
    assert not destination.exists()


def test_preflight_rejects_each_execution_cap() -> None:
    manifest = _manifest()

    excessive_sequences = manifest.model_copy(
        update={"source": manifest.source.model_copy(update={"sequence_count": 10_001})}
    )
    with pytest.raises(ValueError, match="sequence_count"):
        preflight_analytic_manifest(excessive_sequences)

    excessive_bootstrap = manifest.model_copy(
        update={
            "evaluation": manifest.evaluation.model_copy(
                update={
                    "bootstrap": manifest.evaluation.bootstrap.model_copy(
                        update={"replicates": 20_040}
                    )
                }
            )
        }
    )
    with pytest.raises(ValueError, match="bootstrap replicates"):
        preflight_analytic_manifest(excessive_bootstrap)

    excessive_matrix = manifest.model_copy(
        update={
            "source": manifest.source.model_copy(update={"sequence_count": 2_000}),
            "evaluation": manifest.evaluation.model_copy(
                update={
                    "bootstrap": manifest.evaluation.bootstrap.model_copy(
                        update={"replicates": 20_000}
                    )
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="bootstrap matrix"):
        preflight_analytic_manifest(excessive_matrix)

    fault = manifest.fault_sweep
    assert isinstance(fault, AdditivePositionBiasFault)
    excessive_rows = manifest.model_copy(
        update={
            "source": manifest.source.model_copy(update={"sequence_count": 10_000}),
            "fault_sweep": fault.model_copy(
                update={"magnitude_values_m": tuple(float(value) for value in range(22))}
            ),
        }
    )
    with pytest.raises(ValueError, match="sequence records"):
        preflight_analytic_manifest(excessive_rows)
