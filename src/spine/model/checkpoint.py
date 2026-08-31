"""Versioned model-checkpoint metadata and artifact utilities."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import socket
import subprocess as sp
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from spine.utils.conditional import TORCH_AVAILABLE, torch
from spine.version import __version__

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "CheckpointManifest",
    "checkpoint_sha256",
    "inspect_checkpoint",
    "promote_checkpoint",
    "save_checkpoint",
    "verify_checkpoint",
]

CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class CheckpointManifest:
    """Portable provenance recorded with every SPINE checkpoint.

    The dataclass is converted to a plain dictionary before serialization so
    checkpoints remain compatible with PyTorch's restricted ``weights_only``
    loader and do not require this class to be importable when inspected.

    Attributes
    ----------
    created_at : str
        UTC creation timestamp in ISO-8601 form.
    spine_version : str
        SPINE package version which produced the checkpoint.
    python_version : str
        Python runtime version.
    torch_version : str or None
        PyTorch version, when PyTorch is available.
    cuda_version : str or None
        CUDA build version reported by PyTorch, when available.
    hostname : str
        Host on which rank zero serialized the artifact.
    world_size : int
        Number of participating training processes.
    git_revision : str or None
        Source revision, when discoverable from the environment or checkout.
    git_dirty : bool or None
        Whether tracked files differed from the discovered revision.
    contents : tuple[str, ...]
        Top-level checkpoint components present in the artifact.
    """

    created_at: str
    spine_version: str
    python_version: str
    torch_version: str | None
    cuda_version: str | None
    hostname: str
    world_size: int
    git_revision: str | None = None
    git_dirty: bool | None = None
    contents: tuple[str, ...] = ()

    @classmethod
    def create(
        cls,
        world_size: int = 1,
        contents: tuple[str, ...] = (),
    ) -> "CheckpointManifest":
        """Collect checkpoint provenance from the current runtime.

        Parameters
        ----------
        world_size : int, default 1
            Number of processes participating in training.
        contents : tuple[str, ...], optional
            Top-level checkpoint components present in the artifact.

        Returns
        -------
        CheckpointManifest
            Newly collected, serialization-ready provenance.
        """
        revision, dirty = _discover_git_state()
        torch_version = str(torch.__version__) if TORCH_AVAILABLE else None
        cuda_version = None
        if TORCH_AVAILABLE:
            cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

        return cls(
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            spine_version=__version__,
            python_version=platform.python_version(),
            torch_version=torch_version,
            cuda_version=cuda_version,
            hostname=socket.gethostname(),
            world_size=int(world_size),
            git_revision=revision,
            git_dirty=dirty,
            contents=tuple(contents),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a plain serialization-safe dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointManifest":
        """Reconstruct a manifest from checkpoint data.

        Parameters
        ----------
        value : mapping
            Serialized manifest mapping.
        """
        field_names = {field.name for field in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in field_names})


def _discover_git_state() -> tuple[str | None, bool | None]:
    """Return an optional source revision and tracked-file dirty flag."""
    for key in ("SPINE_GIT_REVISION", "GIT_COMMIT", "CI_COMMIT_SHA"):
        revision = os.environ.get(key)
        if revision:
            return revision, None

    repository = Path(__file__).resolve().parents[3]
    if not (repository / ".git").exists():
        return None, None

    try:
        revision = sp.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout.strip()
        status = sp.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout
        return revision or None, bool(status.strip())
    except (OSError, sp.SubprocessError):
        return None, None


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    """Compute the SHA-256 digest of a checkpoint artifact.

    Parameters
    ----------
    path : path-like
        Checkpoint file to hash.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(checkpoint: Mapping[str, Any], path: str | os.PathLike[str]) -> str:
    """Atomically serialize a checkpoint and write its SHA-256 sidecar.

    The checkpoint is first written in the destination directory and then
    atomically moved into place. The checksum follows the conventional
    ``<digest>  <filename>`` format in ``<path>.sha256``.

    Parameters
    ----------
    checkpoint : mapping
        Complete checkpoint payload.
    path : path-like
        Destination checkpoint path.

    Returns
    -------
    str
        SHA-256 digest of the final checkpoint bytes.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to save a model checkpoint.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Publish the artifact only after serialization completes successfully.
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
        torch.save(dict(checkpoint), temporary_path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:  # pragma: no cover - external cleanup race
                pass

    # Hash the published bytes, then atomically publish the matching sidecar.
    digest = checkpoint_sha256(path)
    _write_checksum_sidecar(path, digest)

    return digest


def promote_checkpoint(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> str:
    """Atomically copy a checkpoint into a stable best-checkpoint path.

    Parameters
    ----------
    source : path-like
        Existing checkpoint artifact to promote.
    destination : path-like
        Stable destination path, typically ending in ``-best.ckpt``.

    Returns
    -------
    str
        SHA-256 digest shared by the source and promoted artifact.
    """
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Copy beside the destination so publication remains an atomic rename.
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
        shutil.copyfile(source, temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:  # pragma: no cover - external cleanup race
                pass

    digest = checkpoint_sha256(destination)
    _write_checksum_sidecar(destination, digest)
    return digest


def _write_checksum_sidecar(path: Path, digest: str) -> None:
    """Atomically publish a checksum sidecar for ``path``."""
    sidecar = Path(f"{path}.sha256")
    sidecar_temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=sidecar.parent,
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            sidecar_temporary = temporary.name
            temporary.write(f"{digest}  {path.name}\n")
        os.replace(sidecar_temporary, sidecar)
        sidecar_temporary = None
    finally:
        if sidecar_temporary is not None:
            try:
                os.unlink(sidecar_temporary)
            except FileNotFoundError:  # pragma: no cover - external cleanup race
                pass


def verify_checkpoint(path: str | os.PathLike[str]) -> bool:
    """Verify a checkpoint against its adjacent SHA-256 sidecar.

    Parameters
    ----------
    path : str or os.PathLike
        Checkpoint artifact whose checksum should be verified.

    Raises
    ------
    FileNotFoundError
        If the checkpoint or sidecar does not exist.
    ValueError
        If the sidecar is malformed or names another artifact.
    """
    path = Path(path)
    sidecar = Path(f"{path}.sha256")
    sidecar_fields = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)
    if len(sidecar_fields) != 2:
        raise ValueError(f"Malformed checkpoint checksum sidecar: {sidecar}")
    expected, filename = sidecar_fields
    filename = filename.lstrip(" *")
    if filename != path.name:
        raise ValueError(
            f"Checksum sidecar names `{filename}`, expected `{path.name}`."
        )
    return checkpoint_sha256(path) == expected


def inspect_checkpoint(
    path: str | os.PathLike[str],
    *,
    verify: bool = False,
) -> dict[str, Any]:
    """Read checkpoint metadata without constructing a model.

    Model and optimizer tensors are omitted from the returned mapping. Legacy
    checkpoints are accepted and reported with format version 1.

    Parameters
    ----------
    path : path-like
        Checkpoint artifact to inspect.
    verify : bool, default False
        Whether to require and verify the SHA-256 sidecar first.
    """
    if verify and not verify_checkpoint(path):
        raise ValueError(f"Checkpoint checksum does not match: {path}")
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to inspect a model checkpoint.")

    # Prefer restricted loading while retaining support for older Torch APIs.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as err:
        if "weights_only" not in str(err):
            raise
        checkpoint = torch.load(path, map_location="cpu")

    # Expose provenance and availability without returning heavyweight state.
    keys = (
        "manifest",
        "config",
        "datasets",
        "global_step",
        "global_epoch",
        "validation",
    )
    result = {
        "format_version": int(checkpoint.get("format_version", 1)),
        **{key: checkpoint[key] for key in keys if key in checkpoint},
    }
    result["has_optimizer"] = "optimizer" in checkpoint
    result["has_lr_scheduler"] = "lr_scheduler" in checkpoint
    result["has_runtime_state"] = "runtime_state" in checkpoint
    result["path"] = os.fspath(path)
    if verify:
        result["sha256"] = checkpoint_sha256(path)
    return result
