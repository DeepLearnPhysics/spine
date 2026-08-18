"""PyTorch runtime utilities for tensor operations and memory management.

This module provides conditional PyTorch utilities that gracefully handle
PyTorch unavailability with sensible fallbacks or clear error messages.
"""

from __future__ import annotations

import random
from importlib import import_module
from typing import Any, Protocol, TypeGuard

import numpy as np

from ..conditional import TORCH_AVAILABLE, torch

__all__ = [
    "cdist_fast",
    "manual_seed",
    "cuda_is_available",
    "cuda_mem_info",
    "cuda_max_memory_allocated",
    "is_tensor",
    "distributed_barrier",
    "distributed_all_gather_object",
    "capture_rng_state",
    "restore_rng_state",
    "require_torch",
    "create_summary_writer",
]


def cdist_fast(v1: Any, v2: Any, metric: str = "euclidean") -> Any:
    """Compute pairwise distances without relying on ``torch.cdist``.

    PyTorch's matrix-multiplication implementation can lose substantial
    precision for nearby points, while its direct implementation is often
    slower than broadcasting for the small point sets used throughout SPINE.

    Parameters
    ----------
    v1, v2 : torch.Tensor
        ``(N, D)`` and ``(M, D)`` tensors of coordinates.
    metric : str, default "euclidean"
        One of ``"euclidean"``, ``"cityblock"`` or ``"chebyshev"``.

    Returns
    -------
    torch.Tensor
        ``(N, M)`` pairwise distance matrix.
    """
    require_torch("pairwise tensor distance calculation")

    differences = v1[:, None, :] - v2[None, :, :]
    if metric == "euclidean":
        return torch.sqrt(torch.sum(differences**2, dim=2))
    if metric == "cityblock":
        return torch.sum(torch.abs(differences), dim=2)
    if metric == "chebyshev":
        return torch.amax(torch.abs(differences), dim=2)

    raise ValueError(f"Unsupported distance metric: `{metric}`.")


class _TensorLike(Protocol):
    """Tensor interface required by scalar-output conversion."""

    def numel(self) -> int:
        """Return the number of tensor elements."""
        ...  # pragma: no cover - typing protocol

    def detach(self) -> _TensorLike:
        """Return a tensor detached from its computation graph."""
        ...  # pragma: no cover - typing protocol

    def item(self) -> Any:
        """Return the contained Python scalar."""
        ...  # pragma: no cover - typing protocol


def _serialize_numpy_state(state):
    """Convert a NumPy RNG tuple to restricted-loader-safe primitives."""
    name, keys, position, has_gaussian, cached_gaussian = state
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": int(position),
        "has_gaussian": int(has_gaussian),
        "cached_gaussian": float(cached_gaussian),
    }


def _deserialize_numpy_state(state):
    """Reconstruct a NumPy RNG tuple from serialized primitives."""
    return (
        state["name"],
        np.asarray(state["keys"], dtype=np.uint32),
        int(state["position"]),
        int(state["has_gaussian"]),
        float(state["cached_gaussian"]),
    )


def capture_rng_state():
    """Capture process-local Python, NumPy, Torch and CUDA RNG state.

    Returns
    -------
    dict
        State composed only of primitives and tensors accepted by PyTorch's
        restricted ``weights_only`` checkpoint loader.
    """
    state = {
        "python": random.getstate(),
        "numpy": _serialize_numpy_state(np.random.get_state()),
    }

    # Torch stores RNG state as restricted-loader-safe byte tensors.
    if TORCH_AVAILABLE:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state):
    """Restore process-local RNG state captured by :func:`capture_rng_state`.

    Parameters
    ----------
    state : mapping
        Serialized RNG state for the current process/rank.
    """
    random.setstate(tuple(state["python"]))
    np.random.set_state(_deserialize_numpy_state(state["numpy"]))

    # CUDA state is process-local, hence each distributed rank restores its own.
    if TORCH_AVAILABLE and "torch" in state:
        torch.set_rng_state(state["torch"])
    if TORCH_AVAILABLE and "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"])


def manual_seed(seed):
    """Set torch manual seeds if torch is available.

    Parameters
    ----------
    seed : int
        Random number generator seed

    Returns
    -------
    None
        This function does not return anything
    """
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def cuda_is_available():
    """Check if CUDA is available."""
    return TORCH_AVAILABLE and torch.cuda.is_available()


def cuda_mem_info():
    """Get CUDA memory info if available."""
    if TORCH_AVAILABLE:
        return torch.cuda.mem_get_info()
    return (0, 0)  # Return (used, total) as 0,0 if not available


def cuda_max_memory_allocated():
    """Get max CUDA memory allocated if available."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0


def is_tensor(obj: Any) -> TypeGuard[_TensorLike]:
    """Check whether an object is a tensor and narrow its static type.

    Parameters
    ----------
    obj : object
        Candidate tensor object.

    Returns
    -------
    bool
        Whether PyTorch is available and the object is a tensor.
    """
    return TORCH_AVAILABLE and torch.is_tensor(obj)


def distributed_barrier():
    """Call distributed barrier if available."""
    if (
        TORCH_AVAILABLE
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()


def distributed_all_gather_object(obj):
    """Gather a Python object from every distributed rank.

    Parameters
    ----------
    obj : object
        Python object to gather from the local rank.

    Returns
    -------
    list[object]
        Gathered objects from all ranks. In non-distributed execution, this
        simply returns ``[obj]``.
    """
    if (
        TORCH_AVAILABLE
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        objects = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(objects, obj)
        return objects

    return [obj]


def require_torch(operation="this operation"):
    """Raise informative error when torch is required but not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"PyTorch is required for {operation}. "
            "Use the released SPINE container or install a compatible "
            "PyTorch ecosystem manually."
        )


def create_summary_writer(log_dir, **kwargs):
    """Create a TensorBoard summary writer.

    Parameters
    ----------
    log_dir : str
        Output directory for TensorBoard event files.
    **kwargs
        Additional keyword arguments forwarded to
        ``torch.utils.tensorboard.SummaryWriter``.

    Returns
    -------
    object
        TensorBoard summary writer instance.
    """
    require_torch("TensorBoard logging")
    try:
        summary_writer_cls = import_module("torch.utils.tensorboard").SummaryWriter
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "TensorBoard logging requested but torch.utils.tensorboard is "
            "unavailable. Install the `tensorboard` package."
        ) from exc

    return summary_writer_cls(log_dir=log_dir, **kwargs)
