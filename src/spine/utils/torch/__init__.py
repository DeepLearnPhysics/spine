"""PyTorch utilities organized by functionality.

Submodules:
- `runtime`: Tensor operations, CUDA memory, distributed operations
- `devices`: CUDA device configuration and distributed training setup
- `training`: Optimizer and learning rate scheduler factories
- `scripts`: Utility functions and PyTorch extensions (e.g., cdist_fast)
- `adabound`: Custom AdaBound optimizers (when PyTorch available)

Usage:
    from spine.utils.torch.runtime import manual_seed
    from spine.utils.torch.devices import set_visible_devices
    from spine.utils.torch.training import optim_factory
    from spine.utils.torch.scripts import cdist_fast

All utilities gracefully handle PyTorch unavailability.
"""
