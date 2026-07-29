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
