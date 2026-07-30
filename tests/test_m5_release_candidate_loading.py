# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownArgumentType=false
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_m5_review_candidate import _candidate_content
from test_replay_artifacts import ROOT, _request

import fusion_fault_bench.replay_release_candidate as candidate_module
from fusion_fault_bench.artifacts import canonical_json_bytes
from fusion_fault_bench.contracts.replay_release_v1 import M5_FIGURE_PATHS
from fusion_fault_bench.replay_artifacts import (
    ReplayReviewMachineMembersRequest,
    prepare_replay_review_machine_members,
)


def test_candidate_machine_parser_reads_all_ten_real_review_members() -> None:
    request = _request()
    machine = prepare_replay_review_machine_members(
        ReplayReviewMachineMembersRequest(
            profile_summary=request.profile_summary,
            descriptor_aggregates=request.descriptor_aggregates,
            persistent_aggregates=request.persistent_aggregates,
            persistent_crossovers=request.persistent_crossovers,
            health_aggregates=request.health_aggregates,
            cluster_sensitivity=request.cluster_sensitivity,
            repeat_verification=request.repeat_verification,
            figures=request.figures,
            source_commitments=request.source_commitments,
            run=request.run,
        ),
        source_root=ROOT,
    )
    candidate = SimpleNamespace(
        files={f"machine/{path}": value for path, value in machine.files.items()}
    )

    parsed = candidate_module._parse_candidate_machine(cast(Any, candidate))

    def record_set(rows: object) -> set[bytes]:
        return {canonical_json_bytes(row) for row in cast(Any, rows)}

    assert parsed[0] == request.profile_summary
    assert record_set(parsed[1]) == record_set(request.descriptor_aggregates)
    assert record_set(parsed[2]) == record_set(request.persistent_aggregates)
    assert record_set(parsed[3]) == record_set(request.persistent_crossovers)
    assert record_set(parsed[4]) == record_set(request.health_aggregates)
    assert record_set(parsed[5]) == record_set(request.cluster_sensitivity)
    assert parsed[6] == request.repeat_verification
    assert record_set(parsed[7]) == record_set(request.figures)
    assert record_set(parsed[8]) == record_set(request.source_commitments)


def test_semantic_candidate_loader_regenerates_every_reviewed_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    content = _candidate_content()
    files = dict(content.files)
    report = b"implementation report\n"
    attestation = b"implementation attestation\n"
    files["evidence/implementation-review.md"] = report
    files["evidence/implementation-review-attestation.json"] = attestation
    files["presentation/release-summary.json"] = b"summary\n"
    index = SimpleNamespace(
        scientific_git_revision="a" * 40,
        lockfile_sha256="b" * 64,
        package_version="0.1.0",
        run_id="run-id",
    )
    candidate = SimpleNamespace(index=index, files=files)
    clean = SimpleNamespace(
        source_root=tmp_path,
        git_revision=index.scientific_git_revision,
        lockfile_sha256=index.lockfile_sha256,
        package_version=index.package_version,
    )
    row = SimpleNamespace(run_id="run-id")
    profile = row
    descriptors = (row,)
    persistent = (row,)
    crossovers = (row,)
    health = (row,)
    sensitivity = (row,)
    repeat = row
    bindings = (row,)
    commitments = (row,)
    software = object()
    evidence = object()
    registry = object()
    summary = object()
    privacy = object()
    implementation = object()
    specs = iter(SimpleNamespace(figure_file=path) for path in M5_FIGURE_PATHS[1::2])
    observed: dict[str, Any] = {}

    monkeypatch.setattr(candidate_module, "load_review_candidate", lambda _path: candidate)
    monkeypatch.setattr(candidate_module, "_plan_files", lambda _root: {})
    monkeypatch.setattr(
        candidate_module,
        "_parse_candidate_machine",
        lambda _candidate: (
            profile,
            descriptors,
            persistent,
            crossovers,
            health,
            sensitivity,
            repeat,
            bindings,
            commitments,
        ),
    )
    monkeypatch.setattr(candidate_module, "load_software_verification", lambda *_a, **_k: software)
    monkeypatch.setattr(candidate_module, "_claim_evidence", lambda **_kw: evidence)

    def parse_model(_value: bytes, model_type: type[object], **_kwargs: object) -> object:
        name = model_type.__name__
        if name == "ReplayPublicClaimProjectionsV1":
            return registry
        if name == "ReplayFigureSpecV1":
            return next(specs)
        return implementation

    monkeypatch.setattr(candidate_module, "_model", parse_model)
    monkeypatch.setattr(candidate_module, "validate_public_claim_projections", lambda *_a: None)
    monkeypatch.setattr(candidate_module, "build_release_summary", lambda *_a: summary)
    monkeypatch.setattr(candidate_module, "canonical_json_bytes", lambda _value: b"summary\n")
    monkeypatch.setattr(candidate_module, "build_presentation_files", lambda *_a: {})
    monkeypatch.setattr(candidate_module, "validate_figure_bundle", lambda *_a: None)
    monkeypatch.setattr(
        candidate_module, "load_privacy_license_attestation", lambda *_a, **_k: privacy
    )
    monkeypatch.setattr(
        candidate_module,
        "derive_pre_review_evidence_from_parts",
        lambda **_kw: {"check": {"part": b"value"}},
    )

    def load_inputs(value: bytes, **arguments: object) -> None:
        observed["inputs"] = value
        observed.update(arguments)

    monkeypatch.setattr(candidate_module, "load_pre_review_validation_inputs", load_inputs)
    monkeypatch.setattr(candidate_module, "_results_bindings", lambda _candidate: object())

    result = candidate_module.load_validated_review_candidate(
        path=Path("candidate"),
        source_root=tmp_path,
        clean_snapshot=clean,
        implementation_snapshot=cast(Any, object()),
        implementation_report=report,
        implementation_attestation=attestation,
    )

    assert result is candidate
    assert observed["inputs"] == files["evidence/validation-inputs.json"]
    assert observed["run_id"] == "run-id"
    assert observed["evidence_by_check"] == {"check": {"part": b"value"}}
