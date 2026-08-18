"""Shared configuration parsing for sparse convolutional networks."""

from __future__ import annotations

from typing import Any

from spine.config.factory import Config


def setup_cnn_configuration(
    self: Any,
    reps: int,
    depth: int,
    filters: int,
    input_kernel: int = 3,
    data_dim: int = 3,
    num_input: int = 1,
    allow_bias: bool = False,
    activation: Config = "lrelu",
    norm_layer: Config = "batch_norm",
    spatial_size: int | None = None,
) -> None:
    """Store and validate parameters shared by CNN-based models.

    The function stores canonical attribute names used by all CNN backbones:
    ``num_filters``, ``num_planes``, ``dimension``, ``act_cfg`` and
    ``norm_cfg``. It mutates ``self`` and returns nothing.

    Parameters
    ----------
    reps : int
        Number of residual or convolutional blocks at each depth.
    depth : int
        Number of feature resolutions in the encoder.
    filters : int
        Number of channels at the highest-resolution feature plane. Plane
        widths increase linearly with depth.
    input_kernel : int, default 3
        Kernel size of the initial sparse convolution.
    data_dim : int, default 3
        Number of spatial coordinate dimensions.
    num_input : int, default 1
        Number of input feature channels per active site.
    allow_bias : bool, default False
        Whether convolutional and linear layers may include bias terms.
    activation : str or mapping, default "lrelu"
        Activation configuration accepted by :func:`act_factory`.
    norm_layer : str or mapping, default "batch_norm"
        Normalization configuration accepted by :func:`norm_factory`.
    spatial_size : int, optional
        Input extent in voxels along each spatial axis. Required by operations
        that derive a final pooling kernel or normalize coordinates.

    Raises
    ------
    ValueError
        If a size, count or dimensionality argument is not positive.
    """
    if reps < 1:
        raise ValueError(f"`reps` must be positive, got {reps}.")
    if depth < 1:
        raise ValueError(f"`depth` must be positive, got {depth}.")
    if filters < 1:
        raise ValueError(f"`filters` must be positive, got {filters}.")
    if input_kernel < 1:
        raise ValueError(f"`input_kernel` must be positive, got {input_kernel}.")
    if data_dim < 1:
        raise ValueError(f"`data_dim` must be positive, got {data_dim}.")
    if num_input < 1:
        raise ValueError(f"`num_input` must be positive, got {num_input}.")
    if spatial_size is not None and spatial_size < 1:
        raise ValueError(f"`spatial_size` must be positive, got {spatial_size}.")

    self.reps = reps
    self.depth = depth
    self.num_filters = filters
    self.input_kernel = input_kernel
    self.dimension = data_dim
    self.num_input = num_input
    self.allow_bias = allow_bias
    self.spatial_size = spatial_size

    self.num_planes = [level * self.num_filters for level in range(1, self.depth + 1)]
    self.act_cfg = activation
    self.norm_cfg = norm_layer
