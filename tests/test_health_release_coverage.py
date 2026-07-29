from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest
import test_health_release as existing
from pydantic import BaseModel, ValidationError

import fusion_fault_bench.health_release as release_module
from fusion_fault_bench.artifacts import ArtifactValidationError, canonical_json_bytes
from fusion_fault_bench.contracts.health_artifact_v1 import (
    HEALTH_AGGREGATES_FILE,
    HEALTH_CANDIDATES_FILE,
    HEALTH_FIT_SUMMARY_FILE,
    HEALTH_FIT_VALIDATION_FILE,
    HEALTH_INTENT_FILE,
    HEALTH_SEQUENCE_LOSSES_FILE,
)
from fusion_fault_bench.health_release import (
    HealthReleaseValidationError,
    ResourceQuantitativeClaimV1,
)


@dataclass(frozen=True, slots=True)
class _PublishedRelease:
    sources: Any
    request: Any
    release: Any


class _BlobRecord(BaseModel):
    payload: str


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _revalidate(model: BaseModel, **updates: Any) -> BaseModel:
    candidate = model.model_copy(update=updates)
    return type(model).model_validate_json(canonical_json_bytes(candidate))


def _release_files(root: Path) -> dict[str, bytes]:
    return {
        name: (root / name).read_bytes() for name in release_module.HEALTH_RELEASE_ARTIFACT_PATHS
    }


def _copy_release(published: _PublishedRelease, tmp_path: Path, name: str) -> Path:
    destination = tmp_path / name
    shutil.copytree(published.release.path, destination)
    return destination


def _write_success(root: Path, **updates: Any) -> None:
    path = root / release_module.HEALTH_RELEASE_SUCCESS_FILE
    success = release_module.HealthReleaseSuccessV1.model_validate_json(path.read_bytes())
    path.write_bytes(canonical_json_bytes(success.model_copy(update=updates)))


def _source_components(
    published: _PublishedRelease,
) -> tuple[
    dict[str, bytes],
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
    Any,
]:
    files = _release_files(published.release.path)
    repeat = published.release.repeat
    primary_fit_index, primary_fit_run = release_module._source_graph(
        files,
        prefix_mapping=release_module._PRIMARY_FIT_RELEASE_PATHS,
        expected_artifact_sha256=repeat.official_fit_artifact_sha256,
        expected_run_sha256=repeat.official_fit_run_sha256,
    )
    repeat_fit_index, repeat_fit_run = release_module._source_graph(
        files,
        prefix_mapping=release_module._REPEAT_FIT_RELEASE_PATHS,
        expected_artifact_sha256=repeat.repeat_fit_artifact_sha256,
        expected_run_sha256=repeat.repeat_fit_run_sha256,
    )
    primary_evaluation_index, primary_evaluation_run = release_module._source_graph(
        files,
        prefix_mapping=release_module._PRIMARY_EVALUATION_ENVELOPE_PATHS,
        expected_artifact_sha256=repeat.primary_evaluation_artifact_sha256,
        expected_run_sha256=repeat.primary_evaluation_run_sha256,
    )
    repeat_evaluation_index, repeat_evaluation_run = release_module._source_graph(
        files,
        prefix_mapping=release_module._REPEAT_EVALUATION_ENVELOPE_PATHS,
        expected_artifact_sha256=repeat.repeat_evaluation_artifact_sha256,
        expected_run_sha256=repeat.repeat_evaluation_run_sha256,
    )
    return (
        files,
        primary_fit_index,
        repeat_fit_index,
        primary_evaluation_index,
        repeat_evaluation_index,
        primary_fit_run,
        repeat_fit_run,
        primary_evaluation_run,
        repeat_evaluation_run,
    )


@pytest.fixture(scope="module")
def published_release(tmp_path_factory: pytest.TempPathFactory) -> _PublishedRelease:
    root = tmp_path_factory.mktemp("health-release-coverage")
    sources = existing._make_sources(root / "sources")
    request = existing._request(sources)
    with pytest.MonkeyPatch.context() as monkeypatch:
        existing._patch_source_loaders(monkeypatch, sources)
        release = release_module.write_health_release(
            request,
            root / "release",
            git_metadata_dirs=(),
        )
    return _PublishedRelease(sources=sources, request=request, release=release)


def test_strict_release_models_reject_cross_field_contradictions(
    published_release: _PublishedRelease,
) -> None:
    release = published_release.release
    fit_intent = next(
        item
        for item in release.commitments
        if item.artifact_kind == "fit" and item.path == HEALTH_INTENT_FILE
    )
    fit_candidates = next(
        item
        for item in release.commitments
        if item.artifact_kind == "fit" and item.path == HEALTH_CANDIDATES_FILE
    )
    evaluation_aggregate = next(
        item
        for item in release.commitments
        if item.artifact_kind == "evaluation" and item.path == HEALTH_AGGREGATES_FILE
    )
    omitted = next(
        item
        for item in release.commitments
        if item.artifact_kind == "evaluation" and item.path == HEALTH_SEQUENCE_LOSSES_FILE
    )

    commitment_cases = (
        (fit_intent, {"path": "not-indexed.json"}, "not indexed"),
        (fit_intent, {"primary_record_count": 1}, "primary record count"),
        (fit_candidates, {"repeat_record_count": None}, "repeat record count"),
        (fit_intent, {"equal": False}, "equality contradicts"),
        (
            omitted,
            {
                "primary_record_count": cast(int, omitted.primary_record_count) + 1,
                "repeat_record_count": cast(int, omitted.repeat_record_count) + 1,
            },
            "invalid retention record",
        ),
        (fit_intent, {"primary_retained_release_path": None}, "requires exact release paths"),
        (
            evaluation_aggregate,
            {"primary_retained_release_path": "wrong-aggregate.ndjson"},
            "mapping is not canonical",
        ),
    )
    for model, updates, message in commitment_cases:
        with pytest.raises(ValidationError, match=message):
            _revalidate(model, **updates)

    resource = release.repeat.resources[0]
    resource_cases = (
        ({"wall_time_cap_seconds": 1799.0}, "frozen M4 value"),
        ({"phase": "evaluation"}, "phase disagrees"),
        ({"artifact_contract": "ffb.health-eval-payload/v1"}, "contract disagrees"),
        ({"raw_log_path": "wrong-time-l.txt"}, "not canonical"),
        (
            {"maximum_resident_set_size_raw": resource.peak_rss_bytes + 1},
            "interpreted directly as bytes",
        ),
        ({"wall_time_within_cap": False}, "wall-time resource gate"),
        ({"peak_rss_within_cap": False}, "peak-RSS resource gate"),
    )
    for updates, message in resource_cases:
        with pytest.raises(ValidationError, match=message):
            _revalidate(resource, **updates)

    repeat = release.repeat
    repeat_cases = (
        ({"resources": tuple(reversed(repeat.resources))}, "canonical run order"),
        ({"mismatch_count": 1}, "scientific equality contradicts"),
        ({"volatile_run_records_distinct": False}, "exact conjunction"),
    )
    for updates, message in repeat_cases:
        with pytest.raises(ValidationError, match=message):
            _revalidate(repeat, **updates)

    fit_claim = next(claim for claim in release.claims if claim.source_kind == "fit-summary")
    with pytest.raises(ValidationError, match="fit claim unit"):
        _revalidate(fit_claim, unit="fraction")

    resource_claim = next(claim for claim in release.claims if claim.source_kind == "resource")
    with pytest.raises(ValidationError, match="resource claim unit"):
        _revalidate(resource_claim, unit="bytes")

    with pytest.raises(ValidationError, match="status counts do not partition"):
        _revalidate(
            release.summary,
            aggregate_ok_count=release.summary.aggregate_ok_count + 1,
        )

    json_entry = next(
        entry for entry in release.release_index.files if entry.path.endswith(".json")
    )
    with pytest.raises(ValidationError, match="record count must be present exactly"):
        _revalidate(json_entry, record_count=1)

    files = release.release_index.files
    with pytest.raises(ValidationError, match="exact member order"):
        _revalidate(release.release_index, files=(files[1], files[0], *files[2:]))


def test_raw_time_log_command_and_sidecar_parsing_fail_closed(
    published_release: _PublishedRelease,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value, message in (
        (b"not newline terminated", "bounded newline-terminated"),
        (b"\xff\n", "must be ASCII"),
        (
            b"0.00 real 0.50 user 0.25 sys\n1 maximum resident set size\n",
            "must be positive",
        ),
    ):
        with pytest.raises(HealthReleaseValidationError, match=message):
            release_module._parse_darwin_time_l(value)

    source_run = published_release.sources.official_fit.run
    with pytest.raises(HealthReleaseValidationError, match="lacks timestamps"):
        release_module._run_wall_time(source_run.model_copy(update={"ended_at": None}))

    short_log = b"0.50 real 0.25 user 0.10 sys\n1000000 maximum resident set size\n"
    with pytest.raises(HealthReleaseValidationError, match="does not cover"):
        release_module.build_health_resource_measurement(
            "primary-fit",
            published_release.sources.official_fit,
            short_log,
        )

    with pytest.raises(HealthReleaseValidationError, match="missing --output-dir"):
        release_module._normalized_command(
            source_run.model_copy(update={"command": ("ffb", "health", "fit")})
        )
    with pytest.raises(HealthReleaseValidationError, match="noncanonical output"):
        release_module._normalized_command(
            source_run.model_copy(
                update={
                    "command": (
                        "ffb",
                        "health",
                        "fit",
                        "--output-dir",
                        "fit",
                        "--unexpected",
                    )
                }
            )
        )

    with pytest.raises(HealthReleaseValidationError, match="payload index is incomplete"):
        release_module._payload_entry(
            published_release.sources.official_fit,
            "missing-member.json",
        )

    mismatched_reference = published_release.sources.primary_evaluation.fit_reference.model_copy(
        update={"fit_artifact_sha256": _digest("wrong-fit-reference")}
    )
    mismatched_sources = replace(
        published_release.sources,
        primary_evaluation=replace(
            published_release.sources.primary_evaluation,
            fit_reference=mismatched_reference,
        ),
    )
    existing._patch_source_loaders(monkeypatch, mismatched_sources)
    with pytest.raises(HealthReleaseValidationError, match="designated official fit"):
        release_module._authenticate_sources(published_release.request)


def test_claim_collection_and_serialization_caps_are_strict(
    published_release: _PublishedRelease,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = published_release.sources
    repeat = published_release.release.repeat
    fit_summary = sources.official_fit.summary
    claims = existing._claims(sources)

    with pytest.raises(HealthReleaseValidationError, match="at least one"):
        release_module.validate_health_quantitative_claims(
            (),
            aggregates=sources.aggregates,
            fit_summary=fit_summary,
            repeat=repeat,
        )
    with pytest.raises(HealthReleaseValidationError, match="IDs must be unique"):
        release_module.validate_health_quantitative_claims(
            (claims[0], claims[0]),
            aggregates=sources.aggregates,
            fit_summary=fit_summary,
            repeat=repeat,
        )
    with pytest.raises(HealthReleaseValidationError, match="duplicate keys"):
        release_module.validate_health_quantitative_claims(
            claims,
            aggregates=(*sources.aggregates, sources.aggregates[0]),
            fit_summary=fit_summary,
            repeat=repeat,
        )

    measured = repeat.resources[0]
    peak_claim = ResourceQuantitativeClaimV1(
        schema="ffb.health-quantitative-claim/v1",
        source_kind="resource",
        claim_id="primary-fit-peak-rss",
        presentation_id="resource-table",
        run_label="primary-fit",
        metric="peak-rss-bytes",
        value=float(measured.peak_rss_bytes),
        unit="bytes",
        cpu_model=measured.cpu_model,
        evidence_scope="operator-recorded-time-l-sidecar-not-independent-attestation",
    )
    assert release_module.validate_health_quantitative_claims(
        (peak_claim,),
        aggregates=sources.aggregates,
        fit_summary=fit_summary,
        repeat=repeat,
    ) == (peak_claim,)

    with pytest.raises(HealthReleaseValidationError, match="must be nonempty"):
        release_module._canonical_records(())

    blob = _BlobRecord(payload="x" * 32)
    monkeypatch.setattr(release_module, "_MAX_RECORD_BYTES", 16)
    with pytest.raises(HealthReleaseValidationError, match="record exceeds"):
        release_module._canonical_records((blob,))

    monkeypatch.setattr(release_module, "_MAX_RECORD_BYTES", 1_000)
    monkeypatch.setattr(release_module, "_CURATED_RELEASE_BYTES_MAX", 64)
    with pytest.raises(HealthReleaseValidationError, match="NDJSON exceeds"):
        release_module._canonical_records((blob, blob))

    with pytest.raises(HealthReleaseValidationError, match="allowlist"):
        release_module.build_health_release_index(
            {},
            summary=published_release.release.summary,
            record_counts={},
        )


def test_source_member_path_privacy_and_race_checks(
    published_release: _PublishedRelease,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "directory-member"
    directory.mkdir()
    with pytest.raises(HealthReleaseValidationError, match="bounded regular file"):
        release_module._read_source_member(tmp_path, directory.name)
    with pytest.raises(HealthReleaseValidationError, match="cannot be read"):
        release_module._read_source_member(tmp_path, "missing.json")

    member = tmp_path / "member.json"
    member.write_bytes(b"{}\n")
    other = tmp_path / "other.json"
    other.write_bytes(b"{}\n")
    os.utime(
        other,
        ns=(other.stat().st_atime_ns, other.stat().st_mtime_ns + 1_000_000_000),
    )
    original_fstat = os.fstat
    with monkeypatch.context() as context:
        context.setattr(release_module.os, "fstat", lambda descriptor: other.stat())
        with pytest.raises(HealthReleaseValidationError, match="changed before"):
            release_module._read_source_member(tmp_path, member.name)

    calls = 0

    def changing_fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        return original_fstat(descriptor) if calls == 1 else other.stat()

    with monkeypatch.context() as context:
        context.setattr(release_module.os, "fstat", changing_fstat)
        with pytest.raises(HealthReleaseValidationError, match="changed during"):
            release_module._read_source_member(tmp_path, member.name)

    fit = published_release.sources.official_fit
    entries = list(fit.payload_index.files)
    intent_index = next(
        index for index, entry in enumerate(entries) if entry.path == HEALTH_INTENT_FILE
    )
    entries[intent_index] = entries[intent_index].model_copy(
        update={"sha256": _digest("wrong-source-member")}
    )
    mismatched_fit = replace(
        fit,
        payload_index=fit.payload_index.model_copy(update={"files": tuple(entries)}),
    )
    with pytest.raises(HealthReleaseValidationError, match="payload commitments"):
        release_module._copy_source_members(
            {},
            mismatched_fit,
            {HEALTH_INTENT_FILE: "copied-intent.json"},
        )

    authenticated = release_module._AuthenticatedSources(
        official_fit=published_release.sources.official_fit,
        repeat_fit=published_release.sources.repeat_fit,
        primary_evaluation=cast(Any, published_release.sources.primary_evaluation),
        repeat_evaluation=cast(Any, published_release.sources.repeat_evaluation),
    )
    with pytest.raises(HealthReleaseValidationError, match="privacy scan"):
        release_module._privacy_scan(
            {"member.json": os.fsencode(published_release.sources.official_fit.path)},
            authenticated,
        )

    for private_bytes in (
        b'{"token":"ghp_' + b"abcdefghijklmnopqrstuvwxyz" + b'"}\n',
        b'{"notes":"interview/private.md"}\n',
        b'{"path":"C:\\\\private\\\\dataset"}\n',
    ):
        with pytest.raises(HealthReleaseValidationError, match="privacy scan"):
            release_module.validate_health_release_candidate_bytes({"member.json": private_bytes})


def test_release_tree_and_member_reads_detect_filesystem_substitution(
    published_release: _PublishedRelease,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(HealthReleaseValidationError, match="contains a symlink"):
        release_module._reject_symlink_components(link / "release")

    with pytest.raises(HealthReleaseValidationError, match="cannot be inspected"):
        release_module._reject_symlink_components(tmp_path / "missing" / "release")

    root_file = tmp_path / "not-a-directory"
    root_file.write_text("file", encoding="utf-8")
    with pytest.raises(HealthReleaseValidationError, match="real directory"):
        release_module._require_safe_release_tree(root_file)

    invalid_tree = tmp_path / "invalid-tree"
    invalid_tree.mkdir()
    (invalid_tree / "nested").mkdir()
    with pytest.raises(HealthReleaseValidationError, match="regular files"):
        release_module._require_safe_release_tree(invalid_tree)

    inspected_tree = tmp_path / "inspected-tree"
    inspected_tree.mkdir()

    def fail_scandir(path: Path) -> Any:
        del path
        raise OSError("simulated directory race")

    with monkeypatch.context() as context:
        context.setattr(release_module.os, "scandir", fail_scandir)
        with pytest.raises(HealthReleaseValidationError, match="tree cannot be inspected"):
            release_module._require_safe_release_tree(inspected_tree)

    with monkeypatch.context() as context:
        context.setattr(release_module, "_CURATED_RELEASE_BYTES_MAX", 1)
        with pytest.raises(HealthReleaseValidationError, match="tree exceeds"):
            release_module._require_safe_release_tree(published_release.release.path)

    member = tmp_path / "race-member"
    member.write_bytes(b"abc")
    other = tmp_path / "different-member"
    other.write_bytes(b"xyz")
    with pytest.raises(HealthReleaseValidationError, match="changed during validation"):
        release_module._read_release_member(
            tmp_path,
            member.name,
            expected_stat=other.stat(),
        )

    with monkeypatch.context() as context:
        context.setattr(release_module.os, "read", lambda descriptor, size: b"")
        with pytest.raises(HealthReleaseValidationError, match="changed during reading"):
            release_module._read_release_member(
                tmp_path,
                member.name,
                expected_stat=member.stat(),
            )

    original_read = os.read

    def growing_read(descriptor: int, size: int) -> bytes:
        return b"x" if size == 1 else original_read(descriptor, size)

    with monkeypatch.context() as context:
        context.setattr(release_module.os, "read", growing_read)
        with pytest.raises(HealthReleaseValidationError, match="grew during reading"):
            release_module._read_release_member(
                tmp_path,
                member.name,
                expected_stat=member.stat(),
            )


def test_canonical_loaders_and_profile_discriminator_reject_ambiguous_bytes(
    published_release: _PublishedRelease,
) -> None:
    with pytest.raises(HealthReleaseValidationError, match="strict contract"):
        release_module._load_canonical_model(
            b"{}\n",
            validate=release_module.HealthReleaseSuccessV1.model_validate_json,
            label="success",
        )

    success = published_release.release.success
    noncanonical = b" " + canonical_json_bytes(success)
    with pytest.raises(HealthReleaseValidationError, match="not canonical JSON"):
        release_module._load_canonical_model(
            noncanonical,
            validate=release_module.HealthReleaseSuccessV1.model_validate_json,
            label="success",
        )

    with pytest.raises(HealthReleaseValidationError, match="not canonical NDJSON"):
        release_module._load_canonical_ndjson(
            b"",
            validate=release_module.HealthSourceMemberCommitmentV1.model_validate_json,
            label="commitments",
        )
    with pytest.raises(HealthReleaseValidationError, match="not canonical NDJSON"):
        release_module._load_canonical_ndjson(
            canonical_json_bytes(published_release.release.commitments[0]).rstrip(b"\n"),
            validate=release_module.HealthSourceMemberCommitmentV1.model_validate_json,
            label="commitments",
        )

    smoke_profile = (
        existing.ROOT / "examples/profiles/constant-velocity-ci-smoke-v1.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="not main or edge"):
        release_module._parse_retained_profile(smoke_profile)


def test_retained_envelopes_recompute_all_cross_file_bindings(
    published_release: _PublishedRelease,
) -> None:
    (
        files,
        primary_fit_index,
        repeat_fit_index,
        primary_evaluation_index,
        repeat_evaluation_index,
        primary_fit_run,
        repeat_fit_run,
        primary_evaluation_run,
        repeat_evaluation_run,
    ) = _source_components(published_release)
    repeat = published_release.release.repeat
    commitments = published_release.release.commitments

    with pytest.raises(HealthReleaseValidationError, match="identity graph"):
        release_module._source_graph(
            files,
            prefix_mapping=release_module._PRIMARY_FIT_RELEASE_PATHS,
            expected_artifact_sha256=_digest("wrong-source-graph"),
            expected_run_sha256=repeat.official_fit_run_sha256,
        )

    bad_fit_files = dict(files)
    bad_fit_files[release_module._PRIMARY_FIT_RELEASE_PATHS[HEALTH_INTENT_FILE]] = b"changed\n"
    with pytest.raises(HealthReleaseValidationError, match="fit source bytes"):
        release_module._validate_fit_copies(
            bad_fit_files,
            mapping=release_module._PRIMARY_FIT_RELEASE_PATHS,
            index=primary_fit_index,
        )

    unequal_commitments = (
        commitments[0].model_copy(update={"equal": False}),
        *commitments[1:],
    )
    with pytest.raises(HealthReleaseValidationError, match="mismatches disagree"):
        release_module._validate_commitment_envelope(
            unequal_commitments,
            repeat=repeat,
            primary_fit_index=primary_fit_index,
            repeat_fit_index=repeat_fit_index,
            primary_evaluation_index=primary_evaluation_index,
            repeat_evaluation_index=repeat_evaluation_index,
        )

    with pytest.raises(HealthReleaseValidationError, match="commitments are incomplete"):
        release_module._validate_evaluation_commitments(
            files,
            commitments=commitments[:-1],
            primary_index=primary_evaluation_index,
            repeat_index=repeat_evaluation_index,
        )

    repeat_entries = list(repeat_evaluation_index.files)
    repeat_entries[0] = repeat_entries[0].model_copy(
        update={"sha256": _digest("wrong-evaluation-envelope")}
    )
    bad_repeat_evaluation_index = repeat_evaluation_index.model_copy(
        update={"files": tuple(repeat_entries)}
    )
    with pytest.raises(HealthReleaseValidationError, match="envelope disagrees"):
        release_module._validate_evaluation_commitments(
            files,
            commitments=commitments,
            primary_index=primary_evaluation_index,
            repeat_index=bad_repeat_evaluation_index,
        )

    bad_evaluation_files = dict(files)
    bad_evaluation_files[release_module._EVALUATION_SCIENCE_RELEASE_PATHS[HEALTH_INTENT_FILE]] = (
        b"changed\n"
    )
    with pytest.raises(HealthReleaseValidationError, match="source bytes are invalid"):
        release_module._validate_evaluation_commitments(
            bad_evaluation_files,
            commitments=commitments,
            primary_index=primary_evaluation_index,
            repeat_index=repeat_evaluation_index,
        )

    candidate_entries = list(primary_fit_index.files)
    candidate_index = next(
        index
        for index, entry in enumerate(candidate_entries)
        if entry.path == HEALTH_CANDIDATES_FILE
    )
    candidate_entries[candidate_index] = candidate_entries[candidate_index].model_copy(
        update={"record_count": 35}
    )
    bad_candidate_index = primary_fit_index.model_copy(update={"files": tuple(candidate_entries)})
    with pytest.raises(HealthReleaseValidationError, match="frozen grid"):
        release_module._validate_release_record_counts(
            published_release.release.release_index,
            primary_fit_index=bad_candidate_index,
            repeat_fit_index=repeat_fit_index,
            aggregates=published_release.release.aggregates,
            commitments=commitments,
            claims=published_release.release.claims,
        )

    with pytest.raises(HealthReleaseValidationError, match="source envelopes"):
        release_module._validate_retained_source_evidence(
            files,
            repeat=repeat.model_copy(update={"fit_payload_index_equal": False}),
            primary_fit_index=primary_fit_index,
            repeat_fit_index=repeat_fit_index,
            primary_evaluation_index=primary_evaluation_index,
            repeat_evaluation_index=repeat_evaluation_index,
            primary_fit_run=primary_fit_run,
            repeat_fit_run=repeat_fit_run,
            primary_evaluation_run=primary_evaluation_run,
            repeat_evaluation_run=repeat_evaluation_run,
        )

    invalid_validation_files = dict(files)
    validation_path = release_module._PRIMARY_FIT_RELEASE_PATHS[HEALTH_FIT_VALIDATION_FILE]
    validation = release_module.HealthValidationV1.model_validate_json(
        invalid_validation_files[validation_path]
    )
    invalid_validation_files[validation_path] = canonical_json_bytes(
        validation.model_copy(update={"intent_sha256": _digest("wrong-validation-intent")})
    )
    with pytest.raises(HealthReleaseValidationError, match="did not all pass"):
        release_module._validate_retained_validations(
            invalid_validation_files,
            intent_sha256=repeat.intent_sha256,
        )

    fit_summary = release_module.HealthFitSummaryV1.model_validate_json(
        files[release_module._PRIMARY_FIT_RELEASE_PATHS[HEALTH_FIT_SUMMARY_FILE]]
    )
    with pytest.raises(HealthReleaseValidationError, match="summary does not recompute"):
        release_module._validate_loaded_summary(
            published_release.release.summary.model_copy(update={"package_version": "9.9.9"}),
            aggregates=published_release.release.aggregates,
            claims=published_release.release.claims,
            commitments=commitments,
            repeat=repeat,
            fit_summary=fit_summary,
            official_fit_run=primary_fit_run,
        )


def test_loader_rejects_success_summary_and_index_identity_mismatches(
    published_release: _PublishedRelease,
    tmp_path: Path,
) -> None:
    invalid_success = _copy_release(published_release, tmp_path, "invalid-success")
    _write_success(invalid_success, release_artifact_sha256=_digest("wrong-release"))
    with pytest.raises(HealthReleaseValidationError, match="success digest"):
        release_module.load_health_release(invalid_success)

    invalid_binding = _copy_release(published_release, tmp_path, "invalid-binding")
    _write_success(invalid_binding, release_summary_sha256=_digest("wrong-summary"))
    with pytest.raises(HealthReleaseValidationError, match="binding is invalid"):
        release_module.load_health_release(invalid_binding)

    invalid_summary = _copy_release(published_release, tmp_path, "invalid-summary")
    summary_path = invalid_summary / release_module.HEALTH_RELEASE_SUMMARY_FILE
    summary = release_module.HealthReleaseSummaryV1.model_validate_json(summary_path.read_bytes())
    summary_path.write_bytes(
        canonical_json_bytes(summary.model_copy(update={"package_version": "9.9.9"}))
    )
    existing._reseal_release(invalid_summary)
    with pytest.raises(HealthReleaseValidationError, match="summary does not recompute"):
        release_module.load_health_release(invalid_summary)

    invalid_index = _copy_release(published_release, tmp_path, "invalid-index")
    index_path = invalid_index / release_module.HEALTH_RELEASE_INDEX_FILE
    index = release_module.HealthReleaseIndexV1.model_validate_json(index_path.read_bytes())
    index_bytes = canonical_json_bytes(
        index.model_copy(update={"intent_sha256": _digest("wrong-index-intent")})
    )
    index_path.write_bytes(index_bytes)
    _write_success(
        invalid_index,
        release_artifact_sha256=release_module.compute_health_release_digest(index_bytes),
    )
    with pytest.raises(HealthReleaseValidationError, match="identity disagrees"):
        release_module.load_health_release(invalid_index)


def test_loader_normalizes_unexpected_internal_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_load(path: Path) -> Any:
        del path
        raise KeyError("simulated missing retained member")

    monkeypatch.setattr(release_module, "_load_health_release", fail_load)
    with pytest.raises(HealthReleaseValidationError, match="invalid M4 health release"):
        release_module.load_health_release(tmp_path / "release")


def test_atomic_publication_rejects_races_and_cleans_staging(
    published_release: _PublishedRelease,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = release_module._PreparedHealthRelease(
        files=_release_files(published_release.release.path)
    )

    existing_target = tmp_path / "existing"
    existing_target.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        release_module._publish_health_release(
            prepared,
            existing_target,
            source_root=None,
            git_metadata_dirs=(),
        )

    first_race_parent = tmp_path / "first-race"
    first_race_parent.mkdir()
    with monkeypatch.context() as context:
        context.setattr(release_module, "entry_exists_at", lambda descriptor, name: True)
        with pytest.raises(FileExistsError, match="already exists"):
            release_module._publish_health_release(
                prepared,
                first_race_parent / "release",
                source_root=None,
                git_metadata_dirs=(),
            )

    verification_parent = tmp_path / "verification-race"
    verification_parent.mkdir()
    with monkeypatch.context() as context:
        context.setattr(
            release_module,
            "read_file_at",
            lambda descriptor, name, *, byte_cap: b"substituted",
        )
        with pytest.raises(HealthReleaseValidationError, match="staging verification"):
            release_module._publish_health_release(
                prepared,
                verification_parent / "release",
                source_root=None,
                git_metadata_dirs=(),
            )
    assert not tuple(verification_parent.iterdir())

    second_race_parent = tmp_path / "second-race"
    second_race_parent.mkdir()
    entry_checks = iter((False, True))
    with monkeypatch.context() as context:
        context.setattr(
            release_module,
            "entry_exists_at",
            lambda descriptor, name: next(entry_checks),
        )
        with pytest.raises(FileExistsError, match="already exists"):
            release_module._publish_health_release(
                prepared,
                second_race_parent / "release",
                source_root=None,
                git_metadata_dirs=(),
            )
    assert not tuple(second_race_parent.iterdir())


def test_write_rejects_overlapping_destination_and_wraps_artifact_errors(
    published_release: _PublishedRelease,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(HealthReleaseValidationError, match="disjoint"):
        release_module.write_health_release(
            published_release.request,
            published_release.sources.official_fit.path / "nested-release",
            git_metadata_dirs=(),
        )

    def fail_prepare(request: Any) -> Any:
        del request
        raise ArtifactValidationError("simulated source authentication failure")

    monkeypatch.setattr(release_module, "_prepare_health_release", fail_prepare)
    with pytest.raises(HealthReleaseValidationError, match="publication failed"):
        release_module.write_health_release(
            published_release.request,
            tmp_path / "wrapped-error",
            git_metadata_dirs=(),
        )
