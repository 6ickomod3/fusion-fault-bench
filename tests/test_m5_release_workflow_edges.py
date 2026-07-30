from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fusion_fault_bench import replay_release_candidate
from fusion_fault_bench import replay_release_workflow as workflow

_REVISION = "a" * 40


def test_review_authority_rejects_symlink_mismatch_and_noncanonical_attestation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / workflow._IMPLEMENTATION_REVIEW_PATH
    attestation_path = tmp_path / workflow._IMPLEMENTATION_ATTESTATION_PATH
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(b"report\n")
    attestation_path.symlink_to(report_path)
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="absent, stale, or blocking"):
        workflow._read_review_authority(tmp_path, SimpleNamespace())

    attestation_path.unlink()
    attestation_path.write_bytes(b"attestation\n")
    monkeypatch.setattr(workflow, "_tracked_head_blob", lambda _root, _path: b"different\n")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="absent, stale, or blocking"):
        workflow._read_review_authority(tmp_path, SimpleNamespace())

    monkeypatch.setattr(
        workflow,
        "_tracked_head_blob",
        lambda _root, path: (
            report_path.read_bytes()
            if path == workflow._IMPLEMENTATION_REVIEW_PATH
            else attestation_path.read_bytes()
        ),
    )
    monkeypatch.setattr(
        workflow,
        "load_implementation_review_attestation",
        lambda *_args, **_kwargs: {"canonical": "different"},
    )
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="noncanonical"):
        workflow._read_review_authority(tmp_path, SimpleNamespace())


def test_upstream_sync_requires_named_ref_at_exact_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def configure(*, tracking_revision: str = _REVISION) -> None:
        answers = {
            ("symbolic-ref", "--quiet", "--short", "HEAD"): "main",
            ("config", "--get", "branch.main.remote"): "origin",
            ("config", "--get", "branch.main.merge"): "refs/heads/main",
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "origin/main",
            ("rev-parse", "--symbolic-full-name", "@{upstream}"): ("refs/remotes/origin/main"),
            ("rev-parse", "@{upstream}"): tracking_revision,
            ("remote", "get-url", "origin"): "git@example.invalid:repo.git",
        }
        monkeypatch.setattr(
            workflow,
            "_git_text",
            lambda _root, *arguments: answers[arguments],
        )
        monkeypatch.setattr(
            workflow,
            "_live_remote_bytes",
            lambda *_args, **_kwargs: f"{_REVISION}\trefs/heads/main\n".encode("ascii"),
        )

    configure()
    assert workflow._require_upstream_sync(Path("."), _REVISION) == "origin/main"

    monkeypatch.setattr(workflow, "_git_text", lambda *_args: "")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="upstream"):
        workflow._require_upstream_sync(Path("."), _REVISION)

    configure(tracking_revision="b" * 40)
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="upstream"):
        workflow._require_upstream_sync(Path("."), _REVISION)


def test_normalized_paths_reject_parent_component() -> None:
    output = Path(f"reports/generated/../m5-replay-primary-{_REVISION}-r1")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="not normalized"):
        workflow._normalized_run_paths(
            run_label="primary",
            revision=_REVISION,
            output_dir=output,
            time_l_output=output.with_name(f"{output.name}.time-l.txt"),
        )


@pytest.mark.parametrize(("discovered", "message"), ((None, "unavailable"), ("other", "outside")))
def test_execution_authority_rejects_missing_or_foreign_ffb(
    discovered: str | None,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean = SimpleNamespace(
        source_root=tmp_path,
        git_revision=_REVISION,
        lockfile_sha256="b" * 64,
        package_version="0.1.0",
    )
    expected = tmp_path / ".venv/bin/ffb"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"ffb")
    monkeypatch.setattr(
        workflow,
        "_clean_authority",
        lambda _root: (clean, SimpleNamespace(sha256="c" * 64)),
    )
    monkeypatch.setattr(
        workflow,
        "_read_review_authority",
        lambda *_args: (b"report", b"attestation"),
    )
    monkeypatch.setattr(
        workflow,
        "_software_verification_authority",
        lambda *_args: ("reports/generated/software.json", "d" * 64),
    )
    roots = iter(((tmp_path.parent / "dataset", (1, 2)), (tmp_path.parent / "cache", (3, 4))))
    monkeypatch.setattr(
        workflow,
        "_authenticated_input_directory",
        lambda *_args, **_kwargs: next(roots),
    )
    monkeypatch.setattr(
        workflow,
        "collect_runtime_environment",
        lambda: SimpleNamespace(os_name="Darwin"),
    )
    monkeypatch.setattr(workflow.shutil, "which", lambda _name: discovered)

    def open_foreign_executable(
        _path: Path,
        **_kwargs: object,
    ) -> tuple[str, object, int, bytes]:
        return (
            str(tmp_path.parent / "foreign-ffb"),
            SimpleNamespace(),
            os.open(expected, os.O_RDONLY),
            b"foreign ffb",
        )

    monkeypatch.setattr(
        workflow,
        "_open_authenticated_executable",
        open_foreign_executable,
    )
    for name in workflow._THREAD_ENVIRONMENT_KEYS:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("NUSCENES_ROOT", "/dataset")
    monkeypatch.setenv("UV_CACHE_DIR", "/cache")
    output = Path(f"reports/generated/m5-replay-primary-{_REVISION}-r1")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match=message):
        workflow._execution_authority(
            source_root=tmp_path,
            run_label="primary",
            output_dir=output,
            time_l_output=output.with_name(f"{output.name}.time-l.txt"),
        )


def test_candidate_prepare_rejects_postflight_source_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean = SimpleNamespace(source_root=tmp_path, git_revision=_REVISION)
    changed = SimpleNamespace(source_root=tmp_path, git_revision="b" * 40)
    implementation = object()
    authorities = iter(((clean, implementation), (changed, implementation)))
    monkeypatch.setattr(workflow, "_clean_authority", lambda _root: next(authorities))
    monkeypatch.setattr(
        workflow,
        "_read_review_authority",
        lambda *_args: (b"report", b"attestation"),
    )
    monkeypatch.setattr(workflow, "_authenticate_completed_replays", lambda **_kwargs: object())
    monkeypatch.setattr(workflow, "_require_frozen_candidate_inputs", lambda **_kwargs: None)
    monkeypatch.setattr(
        replay_release_candidate,
        "prepare_review_candidate",
        lambda **_kwargs: object(),
    )
    output = Path(f"reports/generated/m5-review-candidate-{_REVISION}")
    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="changed"):
        workflow.prepare_review_candidate(
            primary_artifact=Path("primary"),
            repeat_artifact=Path("repeat"),
            primary_time_l=Path("primary-time"),
            repeat_time_l=Path("repeat-time"),
            software_verification=Path("software"),
            output_dir=output,
            source_root=tmp_path,
        )
