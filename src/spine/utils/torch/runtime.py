"""PyTorch runtime utilities for tensor operations and memory management.

This module provides conditional PyTorch utilities that gracefully handle
PyTorch unavailability with sensible fallbacks or clear error messages.
"""

from ..conditional import TORCH_AVAILABLE, torch

__all__ = [
    "manual_seed",
    "cuda_is_available",
    "cuda_mem_info",
    "cuda_max_memory_allocated",
    "is_tensor",
    "distributed_barrier",
    "require_torch",
]


def manual_seed(seed):
    """Set torch manual seed if torch is available."""
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


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
    if TORCH_AVAILABLE and torch.distributed.is_available():
        torch.distributed.barrier()


def require_torch(operation="this operation"):
    """Raise informative error when torch is required but not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"PyTorch is required for {operation}. "
            "Install with: pip install spine-ml[model]"
        )
