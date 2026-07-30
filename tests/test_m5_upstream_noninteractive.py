from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fusion_fault_bench import replay_release_workflow as workflow

_REVISION = "a" * 40
_MERGE_REF = "refs/heads/main"


def _git_authority(monkeypatch: pytest.MonkeyPatch, remote_url: str) -> None:
    answers = {
        ("symbolic-ref", "--quiet", "--short", "HEAD"): "main",
        ("config", "--get", "branch.main.remote"): "origin",
        ("config", "--get", "branch.main.merge"): _MERGE_REF,
        (
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ): "origin/main",
        ("rev-parse", "--symbolic-full-name", "@{upstream}"): "refs/remotes/origin/main",
        ("rev-parse", "@{upstream}"): _REVISION,
        ("remote", "get-url", "origin"): remote_url,
    }

    def fake_git_text(_source_root: Path, *arguments: str) -> str:
        return answers[arguments]

    monkeypatch.setattr(workflow, "_git_text", fake_git_text)


@pytest.mark.parametrize(
    ("remote_url", "expected_ssh_command"),
    (
        ("git@example.test:owner/project.git", workflow._STRICT_SSH_COMMAND),
        ("ssh://git@example.test/owner/project.git", workflow._STRICT_SSH_COMMAND),
        ("https://example.test/owner/project.git", None),
    ),
)
def test_live_upstream_is_noninteractive_and_bounded(
    remote_url: str,
    expected_ssh_command: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git_authority(monkeypatch, remote_url)
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=f"{_REVISION}\t{_MERGE_REF}\n".encode("ascii"),
        )

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)

    assert workflow._require_upstream_sync(Path("/source"), _REVISION) == "origin/main"
    assert len(calls) == 1
    command, options = calls[0]
    assert command == (
        "git",
        "-C",
        "/source",
        "ls-remote",
        "--exit-code",
        "origin",
        _MERGE_REF,
    )
    environment = options["env"]
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment.get("GIT_SSH_COMMAND") == expected_ssh_command
    assert options["timeout"] == workflow._LIVE_UPSTREAM_TIMEOUT_SECONDS
    assert 0 < options["timeout"] <= 30
    assert options["capture_output"] is True
    assert options["check"] is False


@pytest.mark.parametrize("failure", ("returncode", "timeout", "oserror"))
def test_live_upstream_failure_is_sanitized(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_url = "https://user:secret@example.test/owner/project.git"
    _git_authority(monkeypatch, "git@example.test:owner/project.git")

    def fake_run(command: tuple[str, ...], **_kwargs: Any) -> SimpleNamespace:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1, stderr=secret_url.encode())
        if failure == "oserror":
            raise OSError(secret_url)
        return SimpleNamespace(returncode=128, stdout=b"", stderr=secret_url.encode())

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)

    with pytest.raises(
        workflow.ReplayReleaseWorkflowError,
        match="could not authenticate the upstream revision",
    ) as caught:
        workflow._require_upstream_sync(Path("/source"), _REVISION)

    assert secret_url not in str(caught.value)
    assert caught.value.__cause__ is None


def test_live_upstream_still_requires_exact_remote_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git_authority(monkeypatch, "https://example.test/owner/project.git")
    monkeypatch.setattr(
        workflow.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"{'b' * 40}\t{_MERGE_REF}\n".encode("ascii"),
        ),
    )

    with pytest.raises(workflow.ReplayReleaseWorkflowError, match="live upstream revision"):
        workflow._require_upstream_sync(Path("/source"), _REVISION)


def test_live_remote_does_not_mutate_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git_authority(monkeypatch, "git@example.test:owner/project.git")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "1")
    monkeypatch.setenv("GIT_SSH_COMMAND", "unsafe-ssh-wrapper")
    monkeypatch.setattr(
        workflow.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=f"{_REVISION}\t{_MERGE_REF}\n".encode("ascii"),
        ),
    )

    workflow._require_upstream_sync(Path("/source"), _REVISION)

    assert os.environ["GIT_TERMINAL_PROMPT"] == "1"
    assert os.environ["GIT_SSH_COMMAND"] == "unsafe-ssh-wrapper"
