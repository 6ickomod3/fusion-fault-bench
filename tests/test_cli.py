from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

from fusion_fault_bench.cli import main


def test_manifest_validate_json_output(example_path, capsys) -> None:
    result = main(["manifest", "validate", str(example_path), "--json"])

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "manifest_sha256": "a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e",
        "schema": "ffb.manifest/v1alpha1",
        "valid": True,
    }


def test_manifest_validate_human_output(example_path, capsys) -> None:
    result = main(["manifest", "validate", str(example_path)])

    assert result == 0
    assert capsys.readouterr().out.startswith("valid ffb.manifest/v1alpha1 ")


def test_manifest_digest_output(example_path, capsys) -> None:
    result = main(["manifest", "digest", str(example_path)])

    assert result == 0
    assert capsys.readouterr().out.strip() == (
        "a603d090f77ad97f20f92b4ec685fe19624d0974dc7d1be4328e2cd3c963bd3e"
    )


def test_schema_show_outputs_json(capsys) -> None:
    result = main(["schema", "show", "metric"])

    assert result == 0
    schema = json.loads(capsys.readouterr().out)
    assert "oneOf" in schema
    assert {"LocalizationMetricRecord", "RateMetricRecord"}.issubset(schema["$defs"])


def test_invalid_manifest_returns_exit_code_two(tmp_path, capsys) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")

    assert main(["manifest", "validate", str(path)]) == 2
    assert capsys.readouterr().err.startswith("error:")


def test_geometry_cli_uses_path_free_success_output(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    artifact = SimpleNamespace(
        artifact_sha256="a" * 64,
        run_sha256="b" * 64,
        payload_index=SimpleNamespace(artifact_contract="ffb.geometry-validation-payload/v1"),
    )
    monkeypatch.setattr(
        "fusion_fault_bench.cli.run_geometry_validation",
        lambda *_args, **_kwargs: artifact,
    )
    result = main(
        [
            "geometry",
            "validate",
            "examples/validation/m2-geometry-v1.json",
            "--dataset-root-env",
            "NUSCENES_ROOT",
            "--output-dir",
            "reports/generated/m2-geometry",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert output.startswith(
        f"wrote reports/generated/m2-geometry artifact_sha256={'a' * 64} run_sha256={'b' * 64}"
    )
    assert str(tmp_path) not in output

    monkeypatch.setattr(
        "fusion_fault_bench.cli.load_geometry_validation_artifact",
        lambda _path: artifact,
    )
    assert (
        main(
            [
                "geometry",
                "bundle",
                "validate",
                str(tmp_path / "private-artifact-path"),
            ]
        )
        == 0
    )
    bundle_output = capsys.readouterr().out
    assert bundle_output.startswith("valid ffb.geometry-validation-payload/v1 ")
    assert str(tmp_path) not in bundle_output


def test_geometry_schemas_are_exposed(capsys) -> None:
    for name in (
        "geometry-manifest",
        "geometry-payload-index",
        "geometry-validation",
    ):
        assert main(["schema", "show", name]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema["type"] == "object"


def test_procedural_cli_runs_matrix_and_validates_bundle(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    artifact = SimpleNamespace(
        path=tmp_path / "procedural-ci-smoke",
        artifact_sha256="c" * 64,
        run_sha256="d" * 64,
        payload_index=SimpleNamespace(artifact_contract="ffb.procedural-payload/v1"),
    )
    monkeypatch.setattr(
        "fusion_fault_bench.cli.run_procedural_matrix",
        lambda *_args, **_kwargs: (artifact,),
    )

    result = main(
        [
            "procedural",
            "matrix",
            "run",
            "examples/matrices/m3-ci-smoke-v1.json",
            "--output-dir",
            "reports/generated/m3-ci-smoke",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert f"artifact_sha256={'c' * 64}" in output
    assert "completed procedural matrix artifact_count=1" in output
    assert str(tmp_path) not in output

    monkeypatch.setattr(
        "fusion_fault_bench.cli.load_procedural_artifact",
        lambda _path: artifact,
    )
    assert (
        main(
            [
                "procedural",
                "bundle",
                "validate",
                str(tmp_path / "private-artifact-path"),
            ]
        )
        == 0
    )
    bundle_output = capsys.readouterr().out
    assert bundle_output.startswith("valid ffb.procedural-payload/v1 ")
    assert str(tmp_path) not in bundle_output


def test_procedural_schemas_are_exposed(capsys) -> None:
    for name in (
        "matrix",
        "m3-matrix-validation",
        "procedural-payload-index",
        "procedural-profile",
        "procedural-validation",
        "repeat-verification",
    ):
        assert main(["schema", "show", name]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema


def test_health_cli_fits_and_evaluates_with_path_free_success_output(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    fit = SimpleNamespace(
        artifact_sha256="e" * 64,
        run_sha256="f" * 64,
        payload_index=SimpleNamespace(artifact_contract="ffb.health-fit-payload/v1"),
    )
    evaluation = SimpleNamespace(
        artifact_sha256="1" * 64,
        run_sha256="2" * 64,
        payload_index=SimpleNamespace(artifact_contract="ffb.health-eval-payload/v1"),
    )
    monkeypatch.setattr(
        "fusion_fault_bench.cli.fit_health_benchmark_artifact",
        lambda **_kwargs: fit,
    )
    assert (
        main(
            [
                "health",
                "fit",
                "--output-dir",
                "reports/generated/m4-health-fit",
            ]
        )
        == 0
    )
    fit_output = capsys.readouterr().out
    assert fit_output.startswith(
        f"wrote reports/generated/m4-health-fit artifact_sha256={'e' * 64}"
    )

    monkeypatch.setattr(
        "fusion_fault_bench.cli.evaluate_health_benchmark_artifact",
        lambda *_args, **_kwargs: evaluation,
    )
    assert (
        main(
            [
                "health",
                "evaluate",
                "reports/generated/m4-health-fit",
                "--output-dir",
                "reports/generated/m4-health-evaluation",
            ]
        )
        == 0
    )
    evaluation_output = capsys.readouterr().out
    assert evaluation_output.startswith(
        f"wrote reports/generated/m4-health-evaluation artifact_sha256={'1' * 64}"
    )
    assert str(tmp_path) not in fit_output + evaluation_output


def test_health_cli_strictly_validates_fit_bound_bundles_without_echoing_paths(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    fit = SimpleNamespace(
        artifact_sha256="3" * 64,
        run_sha256="4" * 64,
        payload_index=SimpleNamespace(artifact_contract="ffb.health-fit-payload/v1"),
    )
    evaluation = SimpleNamespace(
        artifact_sha256="5" * 64,
        run_sha256="6" * 64,
        payload_index=SimpleNamespace(artifact_contract="ffb.health-eval-payload/v1"),
    )
    loaded: list[object] = []

    def load_fit(path):
        loaded.append(path)
        return fit

    def load_evaluation(path, *, fit_artifact):
        loaded.extend((path, fit_artifact))
        return evaluation

    monkeypatch.setattr("fusion_fault_bench.cli.load_health_fit_artifact", load_fit)
    monkeypatch.setattr(
        "fusion_fault_bench.cli.load_health_evaluation_artifact",
        load_evaluation,
    )
    private_fit = tmp_path / "private-fit"
    private_evaluation = tmp_path / "private-evaluation"

    assert (
        main(
            [
                "health",
                "bundle",
                "fit",
                "validate",
                str(private_fit),
            ]
        )
        == 0
    )
    fit_output = capsys.readouterr().out
    assert fit_output.startswith("valid ffb.health-fit-payload/v1 ")
    assert str(tmp_path) not in fit_output

    assert (
        main(
            [
                "health",
                "bundle",
                "evaluation",
                "validate",
                str(private_evaluation),
                "--fit-artifact",
                str(private_fit),
            ]
        )
        == 0
    )
    evaluation_output = capsys.readouterr().out
    assert evaluation_output.startswith("valid ffb.health-eval-payload/v1 ")
    assert str(tmp_path) not in evaluation_output
    assert loaded == [private_fit, private_fit, private_evaluation, fit]


def test_health_schemas_are_exposed(capsys) -> None:
    for name in (
        "health-aggregate",
        "health-fit-reference",
        "health-fit-summary",
        "health-intent",
        "health-payload-index",
        "health-sequence-contrast",
        "health-sequence-event",
        "health-sequence-loss",
        "health-threshold-candidate",
        "health-validation",
    ):
        assert main(["schema", "show", name]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema


def test_replay_cli_runs_repeats_and_validates_without_echoing_paths(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    primary_artifact = SimpleNamespace(
        artifact_sha256="7" * 64,
        run_sha256="8" * 64,
    )
    repeat_artifact = SimpleNamespace(
        artifact_sha256="9" * 64,
        run_sha256="a" * 64,
    )
    primary = SimpleNamespace(artifact=primary_artifact)
    repeat = SimpleNamespace(artifact=repeat_artifact)
    repeated = SimpleNamespace(
        primary=primary,
        repeat=repeat,
        repeat_verification={"schema": "ffb.test-replay-repeat/v1"},
    )
    loaded_repeat = SimpleNamespace(
        primary=primary_artifact,
        repeat=repeat_artifact,
        repeat_verification={"schema": "ffb.test-loaded-replay-repeat/v1"},
    )
    curated_input = SimpleNamespace(
        profile_summary={
            "schema": "ffb.test-replay-profile/v1",
            "resource_evidence": "externally-timed-primary-repeat",
        },
    )
    curated = SimpleNamespace(
        release_index=SimpleNamespace(
            artifact_contract="ffb.replay-curated-payload/v1",
        ),
        artifact_sha256="b" * 64,
        run_sha256="c" * 64,
    )
    calls: list[object] = []

    def run_local(path):
        calls.append(path)
        return primary

    def run_repeat(*, primary_output_dir, repeat_output_dir):
        calls.extend((primary_output_dir, repeat_output_dir))
        return repeated

    def verify_repeat(*, primary_path, repeat_path):
        calls.append(("verify-repeat", primary_path, repeat_path))
        return loaded_repeat

    def curate_repeat(evidence, *, primary_log_path, repeat_log_path):
        calls.append(("curate-repeat", evidence, primary_log_path, repeat_log_path))
        return curated_input

    def load_local(path):
        calls.append(path)
        return primary_artifact

    def load_curated(path):
        calls.append(path)
        return curated

    monkeypatch.setattr("fusion_fault_bench.cli.run_replay_local", run_local)
    monkeypatch.setattr("fusion_fault_bench.cli.run_replay_repeat", run_repeat)
    monkeypatch.setattr("fusion_fault_bench.cli.verify_replay_repeat_artifacts", verify_repeat)
    monkeypatch.setattr("fusion_fault_bench.cli.curate_replay_verified_repeat", curate_repeat)
    monkeypatch.setattr("fusion_fault_bench.cli.load_replay_local_artifact", load_local)
    monkeypatch.setattr("fusion_fault_bench.cli.load_replay_curated_artifact", load_curated)

    private_run = tmp_path / "private-replay-run"
    assert (
        main(
            [
                "replay",
                "run",
                "--output-dir",
                str(private_run),
            ]
        )
        == 0
    )
    run_output = capsys.readouterr().out
    assert run_output.startswith(f"wrote M5 replay local artifact artifact_sha256={'7' * 64}")
    assert str(tmp_path) not in run_output

    private_primary = tmp_path / "private-primary"
    private_repeat = tmp_path / "private-repeat"
    assert (
        main(
            [
                "replay",
                "repeat",
                "--primary-output-dir",
                str(private_primary),
                "--repeat-output-dir",
                str(private_repeat),
            ]
        )
        == 0
    )
    repeat_output = capsys.readouterr().out
    assert repeat_output.startswith(
        f"verified M5 replay repeat primary_artifact_sha256={'7' * 64} "
        f"repeat_artifact_sha256={'9' * 64}"
    )
    assert str(tmp_path) not in repeat_output

    primary_time_log = tmp_path / "private-primary-time-l.txt"
    repeat_time_log = tmp_path / "private-repeat-time-l.txt"
    assert (
        main(
            [
                "replay",
                "verify-repeat",
                "--primary-artifact",
                str(private_primary),
                "--repeat-artifact",
                str(private_repeat),
                "--primary-time-log",
                str(primary_time_log),
                "--repeat-time-log",
                str(repeat_time_log),
            ]
        )
        == 0
    )
    verified_output = capsys.readouterr().out
    assert verified_output.startswith(
        f"verified separately timed M5 replay artifacts "
        f"primary_artifact_sha256={'7' * 64} "
        f"repeat_artifact_sha256={'9' * 64}"
    )
    assert str(tmp_path) not in verified_output

    private_local = tmp_path / "private-local-artifact"
    assert main(["replay", "local", "validate", str(private_local)]) == 0
    local_output = capsys.readouterr().out
    assert local_output.startswith("valid ffb.replay-local-source/v1 ")
    assert str(tmp_path) not in local_output

    private_bundle = tmp_path / "private-curated-artifact"
    assert main(["replay", "bundle", "validate", str(private_bundle)]) == 0
    bundle_output = capsys.readouterr().out
    assert bundle_output.startswith("valid ffb.replay-curated-payload/v1 ")
    assert str(tmp_path) not in bundle_output
    assert calls == [
        private_run,
        private_primary,
        private_repeat,
        ("verify-repeat", private_primary, private_repeat),
        ("curate-repeat", loaded_repeat, primary_time_log, repeat_time_log),
        private_local,
        private_bundle,
    ]


def test_replay_schemas_are_exposed(capsys) -> None:
    for name in (
        "replay-cluster-sensitivity",
        "replay-descriptor-aggregate",
        "replay-execution-resource-evidence",
        "replay-experiment-identity",
        "replay-figure-record",
        "replay-health-aggregate",
        "replay-persistent-aggregate",
        "replay-persistent-crossover",
        "replay-profile-summary",
        "replay-release-index",
        "replay-repeat-verification",
        "replay-source-commitment",
        "replay-validation",
    ):
        assert main(["schema", "show", name]) == 0
        schema = json.loads(capsys.readouterr().out)
        assert schema["type"] == "object"


def test_geometry_cli_never_echoes_rejected_dataset_arguments() -> None:
    secret_path = "/Users/private-owner/datasets/nuScenes"
    base = [
        sys.executable,
        "-m",
        "fusion_fault_bench",
        "geometry",
        "validate",
        "examples/validation/m2-geometry-v1.json",
    ]
    cases = (
        [
            *base,
            "--dataset-root-env",
            secret_path,
            "--output-dir",
            "reports/generated/m2-geometry",
        ],
        [
            *base,
            "--dataset-root",
            secret_path,
            "--output-dir",
            "reports/generated/m2-geometry",
        ],
    )

    for command in cases:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        combined = completed.stdout + completed.stderr
        assert completed.returncode == 2
        assert secret_path not in combined
        assert combined == "error: invalid command arguments\n" or combined.startswith("error: M2 ")


def test_replay_cli_never_accepts_or_echoes_dataset_arguments() -> None:
    secret_path = "/Users/private-owner/datasets/nuScenes"
    base = [
        sys.executable,
        "-m",
        "fusion_fault_bench",
        "replay",
        "run",
    ]
    cases = (
        [
            *base,
            "--dataset-root",
            secret_path,
            "--output-dir",
            "reports/generated/m5-replay",
        ],
        [
            *base,
            secret_path,
            "--output-dir",
            "reports/generated/m5-replay",
        ],
    )

    for command in cases:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        combined = completed.stdout + completed.stderr
        assert completed.returncode == 2
        assert secret_path not in combined
        assert combined == "error: invalid command arguments\n"
