# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false, reportUnnecessaryCast=false
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel
from test_m5_review_candidate import _candidate_content

import fusion_fault_bench.replay_release_candidate as candidate_module
from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_release_v1 import (
    M5_FIGURE_PATHS,
    M5_REVIEW_CANDIDATE_INDEXED_PATHS,
)
from fusion_fault_bench.replay_release_candidate import ReplayCandidateWorkflowError


class _SmallModel(BaseModel):
    value: int


def test_candidate_canonical_model_and_ndjson_helpers_fail_closed() -> None:
    one = canonical_json_bytes(_SmallModel(value=1))
    two = canonical_json_bytes(_SmallModel(value=2))
    assert candidate_module._model(one, _SmallModel, label="small").value == 1
    assert [
        row.value for row in candidate_module._ndjson(one + two, _SmallModel, label="rows")
    ] == [
        1,
        2,
    ]

    with pytest.raises(ReplayCandidateWorkflowError, match="not canonical"):
        candidate_module._model(b'{"value": 1}\n', _SmallModel, label="small")
    with pytest.raises(ReplayCandidateWorkflowError, match="violates"):
        candidate_module._model(b'{"value":"bad"}\n', _SmallModel, label="small")
    with pytest.raises(ReplayCandidateWorkflowError, match="canonical NDJSON"):
        candidate_module._ndjson(b'{"value":1}', _SmallModel, label="rows")


def test_candidate_file_assembly_is_exact_and_rejects_omission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = dict(_candidate_content().files)
    machine = {
        path.removeprefix("machine/"): expected[path]
        for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[:10]
    }
    plans = {path: expected[path] for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[10:13]}
    presentation = {path: expected[path] for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[28:31]}
    specs = {path: expected[path] for path in M5_FIGURE_PATHS[::2]}
    svgs = {path: expected[path] for path in M5_FIGURE_PATHS[1::2]}
    validation = object()
    summary = object()
    registry = object()
    serialized = {
        id(validation): expected["evidence/validation-inputs.json"],
        id(summary): expected["presentation/release-summary.json"],
        id(registry): expected["presentation/public-claim-projections.json"],
    }
    monkeypatch.setattr(
        candidate_module,
        "canonical_json_bytes",
        lambda value: serialized[id(value)],
    )

    arguments: dict[str, Any] = {
        "machine_files": machine,
        "plan_files": plans,
        "implementation_report": expected["evidence/implementation-review.md"],
        "implementation_attestation": expected["evidence/implementation-review-attestation.json"],
        "validation_inputs": validation,
        "software_bytes": expected["evidence/software-verification.json"],
        "privacy_bytes": expected["evidence/privacy-license-attestation.json"],
        "registry": registry,
        "summary": summary,
        "presentation": presentation,
        "specs": specs,
        "svgs": svgs,
    }
    assert candidate_module._candidate_files(**arguments) == expected

    arguments["machine_files"] = dict(tuple(machine.items())[1:])
    with pytest.raises(ReplayCandidateWorkflowError, match="omitted"):
        candidate_module._candidate_files(**arguments)


def test_prepare_candidate_orchestrates_all_review_authorities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    software_path = tmp_path / "software.json"
    software_path.write_bytes(b"software\n")
    row = SimpleNamespace(run_id="run-id")
    repeat = SimpleNamespace(
        primary=SimpleNamespace(artifact_sha256="1" * 64, run_sha256="2" * 64),
        repeat=SimpleNamespace(artifact_sha256="3" * 64, run_sha256="4" * 64),
        repeat_verification=row,
        source_commitments=(row,),
    )
    curated = SimpleNamespace(
        profile_summary=row,
        descriptor_aggregates=(row,),
        persistent_aggregates=(row,),
        persistent_crossovers=(row,),
        health_aggregates=(row,),
        cluster_sensitivity=(row,),
        run=row,
    )
    software = object()
    registry = object()
    summary = object()
    figures = SimpleNamespace(bindings=(row,), specs=(object(),), svgs={"figure.svg": b"svg\n"})
    machine_files = {
        path.removeprefix("machine/"): b"member\n"
        for path in M5_REVIEW_CANDIDATE_INDEXED_PATHS[:10]
    }
    machine = SimpleNamespace(files=machine_files)
    privacy = object()
    validation = object()
    final_files = dict(_candidate_content().files)
    published = object()
    observed: dict[str, Any] = {}

    monkeypatch.setattr(candidate_module, "verify_replay_repeat_artifacts", lambda **_kw: repeat)
    monkeypatch.setattr(
        candidate_module, "curate_replay_verified_repeat", lambda *_a, **_k: curated
    )
    monkeypatch.setattr(candidate_module, "load_software_verification", lambda *_a, **_k: software)
    monkeypatch.setattr(candidate_module, "_claim_evidence", lambda **_kw: object())
    monkeypatch.setattr(candidate_module, "build_public_claim_projections", lambda _e: registry)
    monkeypatch.setattr(candidate_module, "build_release_summary", lambda *_a: summary)
    monkeypatch.setattr(candidate_module, "build_presentation_files", lambda *_a: {})
    monkeypatch.setattr(candidate_module, "build_figure_bundle", lambda *_a: figures)
    monkeypatch.setattr(
        candidate_module, "prepare_replay_review_machine_members", lambda *_a, **_k: machine
    )
    monkeypatch.setattr(
        candidate_module, "build_privacy_license_attestation", lambda **_kw: privacy
    )
    monkeypatch.setattr(candidate_module, "canonical_json_bytes", lambda _value: b"canonical\n")
    monkeypatch.setattr(candidate_module, "_plan_files", lambda _root: {})
    monkeypatch.setattr(candidate_module, "_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        candidate_module,
        "derive_pre_review_evidence_from_parts",
        lambda **_kw: {"check": {"part": b"value"}},
    )
    monkeypatch.setattr(
        candidate_module,
        "derive_pre_review_validation_inputs",
        lambda **_kw: validation,
    )
    monkeypatch.setattr(candidate_module, "_candidate_files", lambda **_kw: final_files)
    monkeypatch.setattr(candidate_module, "canonical_figure_spec_files", lambda _specs: {})

    def publish(content: object, output: Path) -> object:
        observed["content"] = content
        observed["output"] = output
        return published

    monkeypatch.setattr(candidate_module, "publish_review_candidate", publish)
    clean = SimpleNamespace(
        source_root=tmp_path,
        git_revision="a" * 40,
        lockfile_sha256="b" * 64,
        package_version="0.1.0",
    )
    result = candidate_module.prepare_review_candidate(
        primary_artifact=Path("primary"),
        repeat_artifact=Path("repeat"),
        primary_time_l=Path("primary-time"),
        repeat_time_l=Path("repeat-time"),
        software_verification=software_path,
        output_dir=Path("candidate"),
        source_root=tmp_path,
        clean_snapshot=clean,
        implementation_snapshot=cast(Any, object()),
        implementation_report=b"report\n",
        implementation_attestation=b"attestation\n",
    )

    assert result is published
    content = cast(Any, observed["content"])
    assert content.files == final_files
    assert content.run_id == "run-id"
    assert content.primary_local_artifact_sha256 == "1" * 64
    assert observed["output"] == Path("candidate")


def test_prepare_candidate_sanitizes_low_level_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        candidate_module,
        "verify_replay_repeat_artifacts",
        lambda **_kw: (_ for _ in ()).throw(OSError("private detail")),
    )
    with pytest.raises(ReplayCandidateWorkflowError, match="preparation failed"):
        candidate_module.prepare_review_candidate(
            primary_artifact=Path("primary"),
            repeat_artifact=Path("repeat"),
            primary_time_l=Path("primary-time"),
            repeat_time_l=Path("repeat-time"),
            software_verification=Path("software"),
            output_dir=Path("candidate"),
            source_root=tmp_path,
            clean_snapshot=SimpleNamespace(),
            implementation_snapshot=cast(Any, object()),
            implementation_report=b"report\n",
            implementation_attestation=b"attestation\n",
        )


def test_results_review_bindings_cover_every_reviewed_surface() -> None:
    content = _candidate_content()
    candidate = SimpleNamespace(
        files=content.files,
        candidate_sha256="a" * 64,
        candidate_index_sha256="b" * 64,
    )
    bindings = candidate_module.candidate_results_review_bindings(cast(Any, candidate))
    assert bindings.candidate_sha256 == "a" * 64
    assert bindings.candidate_index_sha256 == "b" * 64
    assert len(bindings.scientific_member_set_sha256) == 64
    assert len(bindings.figure_spec_set_sha256) == 64
    assert len(bindings.rendered_figure_set_sha256) == 64
    assert len(bindings.presentation_template_set_sha256) == 64
