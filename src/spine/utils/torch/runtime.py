"""PyTorch runtime utilities for tensor operations and memory management.

This module provides conditional PyTorch utilities that gracefully handle
PyTorch unavailability with sensible fallbacks or clear error messages.
"""

from importlib import import_module

from ..conditional import TORCH_AVAILABLE, torch

__all__ = [
    "manual_seed",
    "cuda_is_available",
    "cuda_mem_info",
    "cuda_max_memory_allocated",
    "is_tensor",
    "distributed_barrier",
    "distributed_all_gather_object",
    "require_torch",
    "create_summary_writer",
]


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


def is_tensor(obj):
    """Check if object is a torch tensor."""
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
            "Install with: pip install spine[model]"
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
