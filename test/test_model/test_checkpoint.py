"""Tests for versioned checkpoint artifacts and inspection helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import spine.model.checkpoint as checkpoint_mod
from spine.model import CheckpointManifest, inspect_checkpoint, verify_checkpoint
from spine.utils.conditional import TORCH_AVAILABLE, torch
from spine.utils.torch import runtime


def test_checkpoint_manifest_is_plain_serializable_provenance(monkeypatch):
    """Manifest construction should record versions and optional source state."""
    monkeypatch.setattr(
        checkpoint_mod,
        "_discover_git_state",
        lambda: ("abc123", True),
    )

    manifest = CheckpointManifest.create(
        world_size=4,
        contents=("state_dict", "optimizer"),
    ).to_dict()

    assert manifest["spine_version"]
    assert manifest["created_at"].endswith("Z")
    assert manifest["world_size"] == 4
    assert manifest["git_revision"] == "abc123"
    assert manifest["git_dirty"] is True
    assert manifest["contents"] == ("state_dict", "optimizer")
    assert (
        CheckpointManifest.from_dict({**manifest, "future": True}).to_dict() == manifest
    )


def test_checkpoint_git_state_prefers_environment(monkeypatch):
    """Build-provided revisions should not require a source checkout."""
    monkeypatch.setenv("SPINE_GIT_REVISION", "build-sha")

    assert checkpoint_mod._discover_git_state() == ("build-sha", None)


def test_checkpoint_git_state_discovers_checkout_and_handles_failure(monkeypatch):
    """Checkout discovery should report tracked dirt and degrade safely."""
    for key in ("SPINE_GIT_REVISION", "GIT_COMMIT", "CI_COMMIT_SHA"):
        monkeypatch.delenv(key, raising=False)

    class GitMarker:
        @staticmethod
        def exists():
            return True

    class Repository:
        def __truediv__(self, _name):
            return GitMarker()

        def __str__(self):
            return "/repository"

    class ResolvedPath:
        parents = [None, None, None, Repository()]

    class Path:
        def __init__(self, _value):
            pass

        @staticmethod
        def resolve():
            return ResolvedPath()

    monkeypatch.setattr(checkpoint_mod, "Path", Path)
    results = iter(
        [SimpleNamespace(stdout="revision\n"), SimpleNamespace(stdout=" M file\n")]
    )
    monkeypatch.setattr(
        checkpoint_mod.sp, "run", lambda *_args, **_kwargs: next(results)
    )

    assert checkpoint_mod._discover_git_state() == ("revision", True)

    monkeypatch.setattr(
        checkpoint_mod.sp,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git missing")),
    )
    assert checkpoint_mod._discover_git_state() == (None, None)


def test_checkpoint_git_state_allows_installed_package_without_checkout(monkeypatch):
    """Installed wheels should simply omit unavailable source-control state."""
    for key in ("SPINE_GIT_REVISION", "GIT_COMMIT", "CI_COMMIT_SHA"):
        monkeypatch.delenv(key, raising=False)

    class MissingRepository:
        def __truediv__(self, _name):
            return self

        @staticmethod
        def exists():
            return False

    class ResolvedPath:
        parents = [None, None, None, MissingRepository()]

    class Path:
        def __init__(self, _value):
            pass

        @staticmethod
        def resolve():
            return ResolvedPath()

    monkeypatch.setattr(checkpoint_mod, "Path", Path)
    assert checkpoint_mod._discover_git_state() == (None, None)


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_checkpoint_atomic_save_checksum_and_inspection(tmp_path, monkeypatch):
    """Saved artifacts should be verifiable and inspectable without a model."""
    monkeypatch.setattr(
        checkpoint_mod,
        "_discover_git_state",
        lambda: (None, None),
    )
    path = tmp_path / "snapshot-3.ckpt"
    checkpoint = {
        "format_version": 2,
        "manifest": CheckpointManifest.create().to_dict(),
        "config": {"model": {"name": "test"}},
        "datasets": {"train": {"files": ["train.root"]}},
        "global_step": 3,
        "global_epoch": 1.5,
        "state_dict": {"weight": torch.ones(1)},
        "optimizer": {"state": {}},
        "lr_scheduler": {"last_epoch": 3},
        "runtime_state": {
            "world_size": 1,
            "ranks": [{"rank": 0, "rng": runtime.capture_rng_state(), "io": None}],
        },
    }

    digest = checkpoint_mod.save_checkpoint(checkpoint, path)
    info = inspect_checkpoint(path, verify=True)

    assert verify_checkpoint(path)
    assert (tmp_path / "snapshot-3.ckpt.sha256").exists()
    assert info["sha256"] == digest
    assert info["format_version"] == 2
    assert info["config"] == checkpoint["config"]
    assert info["datasets"] == checkpoint["datasets"]
    assert info["has_optimizer"]
    assert info["has_lr_scheduler"]
    assert info["has_runtime_state"]
    assert "state_dict" not in info


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_checkpoint_inspection_accepts_legacy_payload(tmp_path):
    """Metadata inspection should tolerate checkpoints predating manifests."""
    path = tmp_path / "legacy.ckpt"
    torch.save(
        {
            "state_dict": {"weight": torch.ones(1)},
            "optimizer": {},
            "global_step": 2,
            "global_epoch": 0.5,
        },
        path,
    )

    info = inspect_checkpoint(path)

    assert info["format_version"] == 1
    assert info["global_step"] == 2
    assert "manifest" not in info
    assert not info["has_runtime_state"]


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_checkpoint_verification_detects_modified_artifact(tmp_path):
    """Checksum verification should reject bytes changed after serialization."""
    path = tmp_path / "snapshot.ckpt"
    checkpoint_mod.save_checkpoint({"state_dict": {}}, path)
    with open(path, "ab") as checkpoint_file:
        checkpoint_file.write(b"changed")

    assert not verify_checkpoint(path)
    with pytest.raises(ValueError, match="checksum"):
        inspect_checkpoint(path, verify=True)


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_checkpoint_promotion_copies_artifact_and_checksum(tmp_path):
    """Best-checkpoint promotion should publish a loadable stable artifact."""
    source = tmp_path / "snapshot-4.ckpt"
    destination = tmp_path / "snapshot-best.ckpt"
    checkpoint_mod.save_checkpoint({"state_dict": {"x": torch.ones(1)}}, source)

    digest = checkpoint_mod.promote_checkpoint(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert checkpoint_mod.checkpoint_sha256(destination) == digest
    assert verify_checkpoint(destination)


def test_checkpoint_promotion_cleans_temporary_copy_on_failure(tmp_path, monkeypatch):
    """Failed best-checkpoint copies should not leave temporary artifacts."""
    source = tmp_path / "snapshot-4.ckpt"
    source.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        checkpoint_mod.shutil,
        "copyfile",
        lambda *_args: (_ for _ in ()).throw(OSError("copy failed")),
    )

    with pytest.raises(OSError, match="copy failed"):
        checkpoint_mod.promote_checkpoint(source, tmp_path / "snapshot-best.ckpt")

    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_checkpoint_verification_rejects_malformed_sidecars(tmp_path):
    """Checksum sidecars should identify both a digest and matching filename."""
    path = tmp_path / "snapshot.ckpt"
    path.write_bytes(b"checkpoint")
    sidecar = tmp_path / "snapshot.ckpt.sha256"
    sidecar.write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        verify_checkpoint(path)

    sidecar.write_text(f"{'0' * 64}  another.ckpt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="another.ckpt"):
        verify_checkpoint(path)


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_checkpoint_save_cleans_temporary_files_on_failures(tmp_path, monkeypatch):
    """Failed serialization and sidecar publication should remove temporaries."""
    path = tmp_path / "snapshot.ckpt"
    real_save = checkpoint_mod.torch.save
    monkeypatch.setattr(
        checkpoint_mod.torch,
        "save",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("save failed")),
    )
    with pytest.raises(OSError, match="save failed"):
        checkpoint_mod.save_checkpoint({}, path)
    assert not list(tmp_path.glob("*.tmp"))

    monkeypatch.setattr(checkpoint_mod.torch, "save", real_save)
    real_replace = checkpoint_mod.os.replace
    replace_calls = 0

    def replace(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("sidecar failed")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint_mod.os, "replace", replace)
    with pytest.raises(OSError, match="sidecar failed"):
        checkpoint_mod.save_checkpoint({}, path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_helpers_require_torch(monkeypatch, tmp_path):
    """Artifact serialization and inspection should fail clearly without Torch."""
    monkeypatch.setattr(checkpoint_mod, "TORCH_AVAILABLE", False)
    with pytest.raises(ImportError, match="save"):
        checkpoint_mod.save_checkpoint({}, tmp_path / "snapshot.ckpt")
    with pytest.raises(ImportError, match="inspect"):
        inspect_checkpoint(tmp_path / "snapshot.ckpt")


@pytest.mark.model
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required.")
def test_checkpoint_inspection_supports_legacy_torch_api(monkeypatch, tmp_path):
    """Inspection should retry Torch versions predating ``weights_only``."""
    path = tmp_path / "legacy.ckpt"
    path.touch()
    calls = []

    def load(_path, **kwargs):
        calls.append(kwargs)
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return {"state_dict": {}, "global_step": 1}

    monkeypatch.setattr(checkpoint_mod.torch, "load", load)
    assert inspect_checkpoint(path)["global_step"] == 1
    assert len(calls) == 2

    monkeypatch.setattr(
        checkpoint_mod.torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("unrelated")),
    )
    with pytest.raises(TypeError, match="unrelated"):
        inspect_checkpoint(path)
