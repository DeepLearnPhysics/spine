"""PyTorch utilities organized by functionality.

Submodules:
- `runtime`: optional PyTorch, tensor, RNG and distributed runtime adapters
- `devices`: process-visible CUDA device selection

Usage:
    from spine.utils.torch.runtime import create_summary_writer, manual_seed
    from spine.utils.torch.devices import set_visible_devices

All utilities gracefully handle PyTorch unavailability.
"""
