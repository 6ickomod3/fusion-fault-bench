from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_health_artifacts import (
    _copy_artifact,
    _evaluation_request,
    _fit_request,
    _rebind_artifact,
)

import fusion_fault_bench.health_artifacts as health_artifacts
from fusion_fault_bench.artifacts import ArtifactValidationError, canonical_json_bytes
from fusion_fault_bench.contracts.health_artifact_v1 import (
    HEALTH_EVAL_ARTIFACT_CONTRACT,
    HEALTH_PAYLOAD_INDEX_FILE,
    HEALTH_SEQUENCE_LOSSES_FILE,
    HealthPayloadIndexV1,
)
from fusion_fault_bench.health_artifacts import (
    HealthEvaluationArtifactTransaction,
    LoadedHealthFitArtifact,
    load_health_evaluation_artifact,
    load_health_fit_artifact,
    write_health_evaluation_artifact,
    write_health_fit_artifact,
)
from fusion_fault_bench.scenarios.health import HealthFaultSpec


@pytest.fixture(scope="module")
def fit_artifact(tmp_path_factory: pytest.TempPathFactory) -> LoadedHealthFitArtifact:
    return write_health_fit_artifact(
        _fit_request(),
        tmp_path_factory.mktemp("health-artifact-security-fit") / "artifact",
        git_metadata_dirs=(),
    )


def _compact_case() -> SimpleNamespace:
    return SimpleNamespace(
        condition_id="condition-a",
        population="main-test",
        fault=HealthFaultSpec(
            family="identity",
            target="none",
            axis="none",
            unit="identity",
            value=0.0,
        ),
    )


def _configure_compact_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fusion_fault_bench.health_benchmark.expand_test_cases",
        lambda _intent: (_compact_case(),),
    )
    monkeypatch.setattr(health_artifacts, "_EXPECTED_EVALUATION_CONDITION_COUNT", 1)
    monkeypatch.setattr(health_artifacts, "_MAX_EVALUATION_LOSS_RECORDS", 2)
    monkeypatch.setattr(health_artifacts, "_MAX_EVALUATION_CONTRAST_RECORDS", 2)
    monkeypatch.setattr(health_artifacts, "_MAX_EVALUATION_EVENT_RECORDS", 2)
    monkeypatch.setattr(health_artifacts, "_require_exact_evaluation_rows", lambda **_: None)


def _compact_batch(fit_artifact: LoadedHealthFitArtifact) -> SimpleNamespace:
    request = _evaluation_request(fit_artifact)
    return SimpleNamespace(
        condition_id="condition-a",
        sequence_losses=tuple(sorted(request.sequence_losses, key=health_artifacts._loss_key)),
        sequence_contrasts=tuple(
            sorted(request.sequence_contrasts, key=health_artifacts._contrast_key)
        ),
        sequence_events=tuple(sorted(request.sequence_events, key=health_artifacts._event_key)),
        aggregates=tuple(sorted(request.aggregates, key=health_artifacts._aggregate_key)),
    )


def test_fit_publisher_removes_staging_after_readback_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health_artifacts, "read_file_at", lambda *_args, **_kwargs: b"tampered")
    destination = tmp_path / "fit"

    with pytest.raises(ArtifactValidationError, match="staging verification"):
        write_health_fit_artifact(
            _fit_request(),
            destination,
            git_metadata_dirs=(),
        )

    assert not destination.exists()
    assert tuple(tmp_path.iterdir()) == ()


def test_fit_publisher_preserves_competing_destination_at_commit_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "fit"
    original_entry_exists = health_artifacts.entry_exists_at
    calls = 0

    def inject_competing_destination(parent_fd: int, name: str) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.mkdir()
            (destination / "sentinel").write_text("competitor", encoding="utf-8")
            return True
        return original_entry_exists(parent_fd, name)

    monkeypatch.setattr(
        health_artifacts,
        "entry_exists_at",
        inject_competing_destination,
    )

    with pytest.raises(FileExistsError, match="destination already exists"):
        write_health_fit_artifact(
            _fit_request(),
            destination,
            git_metadata_dirs=(),
        )

    assert (destination / "sentinel").read_text(encoding="utf-8") == "competitor"
    assert tuple(path.name for path in tmp_path.iterdir()) == ("fit",)


def test_transaction_initialization_failure_removes_partial_staging(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = health_artifacts.write_exclusive_file_at
    calls = 0

    def fail_after_first_member(
        directory_fd: int,
        name: str,
        value: bytes,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated staging failure")
        original_write(directory_fd, name, value)

    monkeypatch.setattr(
        health_artifacts,
        "write_exclusive_file_at",
        fail_after_first_member,
    )

    with pytest.raises(OSError, match="simulated staging failure"):
        HealthEvaluationArtifactTransaction(
            tmp_path / "evaluation",
            fit_artifact=fit_artifact,
            git_metadata_dirs=(),
        )

    assert tuple(tmp_path.iterdir()) == ()


def test_transaction_initialization_preserves_racing_destination(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "evaluation"

    def inject_competing_destination(_parent_fd: int, _name: str) -> bool:
        destination.mkdir()
        (destination / "sentinel").write_text("competitor", encoding="utf-8")
        return True

    monkeypatch.setattr(
        health_artifacts,
        "entry_exists_at",
        inject_competing_destination,
    )

    with pytest.raises(FileExistsError, match="destination already exists"):
        HealthEvaluationArtifactTransaction(
            destination,
            fit_artifact=fit_artifact,
            git_metadata_dirs=(),
        )

    assert (destination / "sentinel").read_text(encoding="utf-8") == "competitor"


def test_transaction_context_aborts_unpublished_staging(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evaluation"

    with (
        pytest.raises(RuntimeError, match="caller failed"),
        HealthEvaluationArtifactTransaction(
            destination,
            fit_artifact=fit_artifact,
            git_metadata_dirs=(),
        ) as transaction,
    ):
        assert transaction.staging_path.exists()
        raise RuntimeError("caller failed")

    assert transaction._closed
    assert not transaction.staging_path.exists()
    assert not destination.exists()


def test_transaction_rejects_out_of_order_and_closed_appends(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_compact_evaluation(monkeypatch)
    transaction = HealthEvaluationArtifactTransaction(
        tmp_path / "evaluation",
        fit_artifact=fit_artifact,
        git_metadata_dirs=(),
    )
    batch = _compact_batch(fit_artifact)
    wrong_batch = SimpleNamespace(**(vars(batch) | {"condition_id": "wrong-condition"}))

    with pytest.raises(ArtifactValidationError, match="canonical order"):
        transaction.append_condition(wrong_batch)

    transaction.abort()
    with pytest.raises(ArtifactValidationError, match="not appendable"):
        transaction.append_condition(batch)
    transaction.abort()


def test_transaction_rejects_incomplete_and_wrong_global_counts(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_compact_evaluation(monkeypatch)
    request = _evaluation_request(fit_artifact)

    with (
        HealthEvaluationArtifactTransaction(
            tmp_path / "incomplete",
            fit_artifact=fit_artifact,
            git_metadata_dirs=(),
        ) as incomplete,
        pytest.raises(ArtifactValidationError, match="condition stream is incomplete"),
    ):
        incomplete.finalize(validation=request.validation, run=request.run)

    monkeypatch.setattr(health_artifacts, "_MAX_EVALUATION_LOSS_RECORDS", 3)
    with HealthEvaluationArtifactTransaction(
        tmp_path / "wrong-count",
        fit_artifact=fit_artifact,
        git_metadata_dirs=(),
    ) as wrong_count:
        wrong_count.append_condition(_compact_batch(fit_artifact))
        with pytest.raises(ArtifactValidationError, match="wrong global counts"):
            wrong_count.finalize(validation=request.validation, run=request.run)


def test_transaction_rejects_staged_identity_drift_and_cleans_up(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_compact_evaluation(monkeypatch)
    request = _evaluation_request(fit_artifact)
    destination = tmp_path / "evaluation"
    original_load = health_artifacts._load_health_evaluation_artifact

    def load_with_drift(
        path: Path,
        *,
        fit_artifact: LoadedHealthFitArtifact,
    ) -> Any:
        loaded = original_load(path, fit_artifact=fit_artifact)
        return replace(loaded, artifact_sha256="e" * 64)

    with HealthEvaluationArtifactTransaction(
        destination,
        fit_artifact=fit_artifact,
        git_metadata_dirs=(),
    ) as transaction:
        transaction.append_condition(_compact_batch(fit_artifact))
        monkeypatch.setattr(
            health_artifacts,
            "_load_health_evaluation_artifact",
            load_with_drift,
        )
        with pytest.raises(ArtifactValidationError, match=r"staged.*identity changed"):
            transaction.finalize(validation=request.validation, run=request.run)

    assert not destination.exists()
    assert not transaction.staging_path.exists()


def test_transaction_rolls_back_after_post_rename_path_failure(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_compact_evaluation(monkeypatch)
    request = _evaluation_request(fit_artifact)
    destination = tmp_path / "evaluation"
    original_assert = health_artifacts.assert_directory_descriptor_matches_path
    failed_once = False

    def fail_first_published_assertion(
        directory_fd: int,
        path: Path,
        *,
        label: str,
    ) -> None:
        nonlocal failed_once
        if label == "published artifact" and not failed_once:
            failed_once = True
            raise ArtifactValidationError("simulated published-path substitution")
        original_assert(directory_fd, path, label=label)

    with HealthEvaluationArtifactTransaction(
        destination,
        fit_artifact=fit_artifact,
        git_metadata_dirs=(),
    ) as transaction:
        transaction.append_condition(_compact_batch(fit_artifact))
        monkeypatch.setattr(
            health_artifacts,
            "assert_directory_descriptor_matches_path",
            fail_first_published_assertion,
        )
        with pytest.raises(ArtifactValidationError, match="published-path substitution"):
            transaction.finalize(validation=request.validation, run=request.run)

    assert failed_once
    assert not destination.exists()
    assert not transaction.staging_path.exists()


def test_transaction_preserves_competing_destination_at_final_gate(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_compact_evaluation(monkeypatch)
    request = _evaluation_request(fit_artifact)
    destination = tmp_path / "evaluation"
    original_entry_exists = health_artifacts.entry_exists_at
    calls = 0

    def inject_competing_destination(parent_fd: int, name: str) -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.mkdir()
            (destination / "sentinel").write_text("competitor", encoding="utf-8")
            return True
        return original_entry_exists(parent_fd, name)

    monkeypatch.setattr(
        health_artifacts,
        "entry_exists_at",
        inject_competing_destination,
    )

    with HealthEvaluationArtifactTransaction(
        destination,
        fit_artifact=fit_artifact,
        git_metadata_dirs=(),
    ) as transaction:
        transaction.append_condition(_compact_batch(fit_artifact))
        with pytest.raises(FileExistsError, match="destination already exists"):
            transaction.finalize(validation=request.validation, run=request.run)

    assert (destination / "sentinel").read_text(encoding="utf-8") == "competitor"
    assert tuple(path.name for path in tmp_path.iterdir()) == ("evaluation",)


def test_transaction_reauthentication_detects_stat_and_identity_drift(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_path = _copy_artifact(fit_artifact.path, tmp_path, "fit-copy")
    copied_fit = load_health_fit_artifact(copied_path)
    stat_transaction = HealthEvaluationArtifactTransaction(
        tmp_path / "stat-evaluation",
        fit_artifact=copied_fit,
        git_metadata_dirs=(),
    )
    fit_stat = copied_path.stat()
    os.utime(
        copied_path,
        ns=(fit_stat.st_atime_ns, fit_stat.st_mtime_ns + 1_000_000_000),
    )
    with pytest.raises(ArtifactValidationError, match="fit source changed"):
        stat_transaction._reauthenticate_fit_source()
    stat_transaction.abort()

    identity_transaction = HealthEvaluationArtifactTransaction(
        tmp_path / "identity-evaluation",
        fit_artifact=fit_artifact,
        git_metadata_dirs=(),
    )
    monkeypatch.setattr(
        health_artifacts,
        "_authenticate_fit_handle",
        lambda loaded: replace(loaded, artifact_sha256="e" * 64),
    )
    with pytest.raises(ArtifactValidationError, match="fit source changed"):
        identity_transaction._reauthenticate_fit_source()
    identity_transaction.abort()


def test_transaction_abort_surfaces_cleanup_failure_after_closing_descriptors(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transaction = HealthEvaluationArtifactTransaction(
        tmp_path / "evaluation",
        fit_artifact=fit_artifact,
        git_metadata_dirs=(),
    )
    staging_path = transaction.staging_path

    monkeypatch.setattr(
        health_artifacts,
        "_safe_cleanup_staging_at",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        transaction.abort()

    assert transaction._closed
    assert transaction._parent_fd == -1
    assert transaction._staging_fd == -1
    shutil.rmtree(staging_path)


def test_evaluation_loader_normalizes_non_contract_failures(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        health_artifacts,
        "_authenticate_fit_handle",
        lambda _fit: (_ for _ in ()).throw(ValueError("bad handle")),
    )

    with pytest.raises(ArtifactValidationError, match="invalid M4 health evaluation artifact"):
        load_health_evaluation_artifact(
            tmp_path / "missing",
            fit_artifact=fit_artifact,
        )


def test_evaluation_loader_checks_indexed_exact_count_before_row_parsing(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_compact_evaluation(monkeypatch)
    evaluation = write_health_evaluation_artifact(
        _evaluation_request(fit_artifact),
        tmp_path / "evaluation",
        fit_artifact=fit_artifact,
        git_metadata_dirs=(),
    )
    index_path = evaluation.path / HEALTH_PAYLOAD_INDEX_FILE
    payload_index = HealthPayloadIndexV1.model_validate_json(index_path.read_bytes())
    payload_index = payload_index.model_copy(
        update={
            "files": tuple(
                entry.model_copy(update={"record_count": 3})
                if entry.path == HEALTH_SEQUENCE_LOSSES_FILE
                else entry
                for entry in payload_index.files
            )
        }
    )
    index_path.write_bytes(canonical_json_bytes(payload_index))
    _rebind_artifact(
        evaluation.path,
        artifact_contract=HEALTH_EVAL_ARTIFACT_CONTRACT,
    )
    monkeypatch.setattr(
        health_artifacts,
        "_iter_ordered_ndjson",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rows parsed")),
    )

    with pytest.raises(ArtifactValidationError, match="frozen matrix"):
        load_health_evaluation_artifact(
            evaluation.path,
            fit_artifact=fit_artifact,
        )


def test_evaluation_loader_returns_metadata_light_handle(
    fit_artifact: LoadedHealthFitArtifact,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_compact_evaluation(monkeypatch)
    materialized = write_health_evaluation_artifact(
        _evaluation_request(fit_artifact),
        tmp_path / "evaluation",
        fit_artifact=fit_artifact,
        git_metadata_dirs=(),
    )

    loaded = load_health_evaluation_artifact(
        materialized.path,
        fit_artifact=fit_artifact,
    )

    assert materialized.sequence_rows_materialized
    assert not loaded.sequence_rows_materialized
    assert loaded.sequence_losses == ()
    assert loaded.sequence_contrasts == ()
    assert loaded.sequence_events == ()
    assert loaded.aggregates == materialized.aggregates


def _ndjson_member() -> Any:
    aggregate = _fit_request().summary
    line = canonical_json_bytes(aggregate)
    return health_artifacts._CanonicalNdjsonMember(
        records=(aggregate,),
        byte_length=len(line),
        sha256=hashlib.sha256(line).hexdigest(),
        record_count=1,
    )


def test_canonical_ndjson_writer_enforces_line_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = _ndjson_member()
    monkeypatch.setattr(health_artifacts, "HEALTH_MAX_RECORD_BYTES", 1)
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(ArtifactValidationError, match="line cap"):
            health_artifacts._write_canonical_ndjson_at(
                directory_fd,
                "records.ndjson",
                member,
            )
    finally:
        os.close(directory_fd)


def test_canonical_ndjson_writer_rejects_short_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = _ndjson_member()
    monkeypatch.setattr(health_artifacts.os, "write", lambda *_args: 0)
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(OSError, match="short write"):
            health_artifacts._write_canonical_ndjson_at(
                directory_fd,
                "records.ndjson",
                member,
            )
    finally:
        os.close(directory_fd)


def test_canonical_ndjson_writer_rejects_non_regular_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = _ndjson_member()
    monkeypatch.setattr(
        health_artifacts.os,
        "fstat",
        lambda _fd: SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_nlink=1,
            st_size=member.byte_length,
        ),
    )
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(ArtifactValidationError, match="private regular file"):
            health_artifacts._write_canonical_ndjson_at(
                directory_fd,
                "records.ndjson",
                member,
            )
    finally:
        os.close(directory_fd)


def test_canonical_ndjson_writer_rejects_commitment_mismatch(tmp_path: Path) -> None:
    member = _ndjson_member()
    wrong_member = replace(member, sha256="f" * 64)
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(ArtifactValidationError, match="disagrees with its preparation"):
            health_artifacts._write_canonical_ndjson_at(
                directory_fd,
                "records.ndjson",
                wrong_member,
            )
    finally:
        os.close(directory_fd)
