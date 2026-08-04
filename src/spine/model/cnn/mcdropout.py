"""Monte Carlo dropout variants of the UResNet encoder and decoder."""

from __future__ import annotations

from typing import Any

import torch

from spine.model import sparse

from .act_norm import act_factory, norm_factory
from .blocks import DropoutBlock, ResNetBlock
from .configuration import setup_cnn_configuration
from .uresnet_layers import EncoderOutput

__all__ = ["MCDropoutEncoder", "MCDropoutDecoder"]


def _dropout_layers(
    layers: list[int] | tuple[int, ...] | set[int] | None,
    depth: int,
) -> frozenset[int]:
    """Validate and normalize levels at which dropout is enabled.

    Parameters
    ----------
    layers : list, tuple or set of int, optional
        Zero-based depth indices. If omitted, dropout is enabled in the deeper
        half of the network.
    depth : int
        Total number of encoder feature levels.

    Returns
    -------
    frozenset of int
        Validated, immutable set of dropout levels.

    Raises
    ------
    ValueError
        If an index lies outside ``[0, depth)``.
    """
    if layers is None:
        layers = list(range(depth // 2, depth))

    result = frozenset(layers)
    invalid = sorted(index for index in result if index < 0 or index >= depth)
    if invalid:
        raise ValueError(
            f"`dropout_layers` contains levels outside [0, {depth}): {invalid}."
        )
    return result


class MCDropoutEncoder(sparse.Network):
    """Sparse residual encoder with dropout at configurable depth levels.

    Parameters not specific to dropout follow
    :func:`setup_cnn_configuration`. The encoder can either return a pooled
    image representation through :meth:`forward` or expose its sparse feature
    pyramid through :meth:`encode`.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        dropout_p: float = 0.5,
        dropout_layers: list[int] | tuple[int, ...] | set[int] | None = None,
        coord_conv: bool = False,
        pool_mode: str = "avg",
        feature_size: int = 512,
        add_classifier: bool = True,
    ) -> None:
        """Initialize the Monte Carlo dropout encoder.

        Parameters
        ----------
        cfg : dict
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.
        dropout_p : float, default 0.5
            Probability of dropping each feature during training or stochastic
            inference.
        dropout_layers : list, tuple or set of int, optional
            Encoder depth indices at which dropout blocks are used. Defaults
            to the deeper half of the network.
        coord_conv : bool, default False
            Append normalized spatial coordinates to input features.
        pool_mode : {"avg", "sum", "max", "conv", "none"}, default "avg"
            Operation used to reduce the deepest sparse plane.
        feature_size : int, default 512
            Width of the pooled output representation.
        add_classifier : bool, default True
            Apply the final sparse linear projection. If false, return pooled
            backbone features directly.

        Raises
        ------
        ValueError
            If probabilities, dimensions, layer indices or pooling parameters
            are invalid.
        """
        super().__init__(cfg.get("data_dim", 3))
        setup_cnn_configuration(self, **cfg)

        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"`dropout_p` must be in [0, 1), got {dropout_p}.")
        if feature_size < 1:
            raise ValueError(f"`feature_size` must be positive, got {feature_size}.")
        if coord_conv and self.spatial_size is None:
            raise ValueError("`coord_conv` requires `spatial_size`.")

        self.dropout_p = dropout_p
        self.dropout_layers = _dropout_layers(dropout_layers, self.depth)
        self.coord_conv = coord_conv
        self.pool_mode = pool_mode
        self.feature_size = feature_size
        # Retained for the standalone Bayesian particle classifiers.
        self.latent_size = feature_size
        self.add_classifier = add_classifier

        input_features = self.num_input + (self.dimension if coord_conv else 0)
        self.input_layer = sparse.Convolution(
            in_channels=input_features,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel,
            stride=1,
            dimension=self.dimension,
            bias=self.allow_bias,
        )

        encoding_blocks = []
        encoding_convolutions = []
        for level, num_features in enumerate(self.num_planes):
            block_type = DropoutBlock if level in self.dropout_layers else ResNetBlock
            blocks = [
                (
                    block_type(
                        num_features,
                        num_features,
                        dimension=self.dimension,
                        p=self.dropout_p,
                        activation=self.act_cfg,
                        normalization=self.norm_cfg,
                        bias=self.allow_bias,
                    )
                    if block_type is DropoutBlock
                    else block_type(
                        num_features,
                        num_features,
                        dimension=self.dimension,
                        activation=self.act_cfg,
                        normalization=self.norm_cfg,
                        bias=self.allow_bias,
                    )
                )
                for _ in range(self.reps)
            ]
            encoding_blocks.append(torch.nn.Sequential(*blocks))

            downsample: list[torch.nn.Module] = []
            if level < self.depth - 1:
                downsample.extend(
                    [
                        norm_factory(self.norm_cfg, num_features),
                        act_factory(self.act_cfg),
                        sparse.Convolution(
                            in_channels=self.num_planes[level],
                            out_channels=self.num_planes[level + 1],
                            kernel_size=2,
                            stride=2,
                            dimension=self.dimension,
                            bias=self.allow_bias,
                        ),
                    ]
                )
                if level in self.dropout_layers:
                    downsample.append(sparse.Dropout(p=self.dropout_p))
            encoding_convolutions.append(torch.nn.Sequential(*downsample))

        self.encoding_block = torch.nn.Sequential(*encoding_blocks)
        self.encoding_conv = torch.nn.Sequential(*encoding_convolutions)
        self.pool = self._make_pool(pool_mode)

        if add_classifier:
            self.linear = torch.nn.Sequential(
                sparse.ReLU(),
                sparse.Linear(self.num_planes[-1], feature_size),
            )
        else:
            self.linear = torch.nn.Identity()

    def _make_pool(self, pool_mode: str) -> torch.nn.Module:
        """Build the final sparse pooling operation.

        Parameters
        ----------
        pool_mode : str
            Requested pooling mode.

        Returns
        -------
        torch.nn.Module
            Sparse pooling module or convolutional reduction sequence.

        Raises
        ------
        ValueError
            If the mode is unknown or convolutional pooling cannot derive a
            valid kernel from ``spatial_size`` and ``depth``.
        """
        if pool_mode in {"avg", "global_average"}:
            return sparse.GlobalAvgPooling()
        if pool_mode == "sum":
            return sparse.GlobalSumPooling()
        if pool_mode == "max":
            return sparse.GlobalMaxPooling()
        if pool_mode in {"none", "no_pooling"}:
            return torch.nn.Identity()
        if pool_mode == "conv":
            if self.spatial_size is None:
                raise ValueError("`pool_mode: conv` requires `spatial_size`.")
            final_size = self.spatial_size // (2 ** (self.depth - 1))
            if final_size < 1:
                raise ValueError(
                    "`spatial_size` is too small for the configured depth."
                )
            return torch.nn.Sequential(
                sparse.Convolution(
                    in_channels=self.num_planes[-1],
                    out_channels=self.num_planes[-1],
                    kernel_size=final_size,
                    stride=final_size,
                    dimension=self.dimension,
                    bias=self.allow_bias,
                ),
                sparse.Dropout(p=self.dropout_p),
            )
        raise ValueError(
            f"Unknown pooling mode '{pool_mode}'. Expected 'avg', 'sum', "
            "'max', 'conv' or 'none'."
        )

    def encode(self, x: sparse.SparseTensor) -> EncoderOutput:
        """Encode a sparse tensor without applying final pooling.

        Parameters
        ----------
        x : sparse.SparseTensor
            Sparse input with the configured feature width.

        Returns
        -------
        EncoderOutput
            Encoder feature planes and deepest representation.
        """
        x = self.input_layer(x)
        encoder_tensors = [x]
        for block, downsample in zip(
            self.encoding_block, self.encoding_conv, strict=True
        ):
            x = block(x)
            encoder_tensors.append(x)
            x = downsample(x)

        return {"encoder_tensors": encoder_tensors, "final_tensor": x}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a coordinate-feature table into image representations.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, 1 + D + C)`` table containing batch IDs, coordinates and
            input features.

        Returns
        -------
        torch.Tensor
            Pooled feature matrix. With global pooling its shape is
            ``(B, feature_size)`` when ``add_classifier`` is true.

        Raises
        ------
        RuntimeError
            If coordinate convolution is enabled without a spatial size.
        """
        coords = x[:, : self.dimension + 1]
        features = x[:, self.dimension + 1 :]
        if self.coord_conv:
            spatial_size = self.spatial_size
            if spatial_size is None:  # Constructor validation narrows at runtime.
                raise RuntimeError("`coord_conv` requires `spatial_size`.")
            normalized = coords[:, 1:] / spatial_size
            features = torch.cat((normalized, features), dim=1)

        sparse_input = sparse.SparseTensor(
            coordinates=coords.int(),
            features=features,
        )
        encoded = self.encode(sparse_input)
        latent = self.linear(self.pool(encoded["final_tensor"]))
        return latent.features


class MCDropoutDecoder(sparse.Network):
    """Decode UResNet features with dropout at configurable depth levels.

    Each decoder level upsamples the preceding representation, concatenates
    the matching encoder skip plane, and applies residual or dropout-residual
    blocks. This supports stochastic segmentation inference using the same
    feature contract as :class:`UResNetDecoder`.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        dropout_p: float = 0.5,
        dropout_layers: list[int] | tuple[int, ...] | set[int] | None = None,
        encoder_filters: int | None = None,
    ) -> None:
        """Initialize the Monte Carlo dropout decoder.

        Parameters
        ----------
        cfg : dict
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.
        dropout_p : float, default 0.5
            Probability of dropping each feature.
        dropout_layers : list, tuple or set of int, optional
            Decoder depth indices at which dropout is enabled.
        encoder_filters : int, optional
            Base encoder width when it differs from the decoder width.

        Raises
        ------
        ValueError
            If ``dropout_p``, ``dropout_layers`` or ``encoder_filters`` is
            invalid.
        """
        super().__init__(cfg.get("data_dim", 3))
        setup_cnn_configuration(self, **cfg)

        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"`dropout_p` must be in [0, 1), got {dropout_p}.")
        if encoder_filters is not None and encoder_filters < 1:
            raise ValueError(
                f"`encoder_filters` must be positive, got {encoder_filters}."
            )

        self.dropout_p = dropout_p
        self.dropout_layers = _dropout_layers(dropout_layers, self.depth)
        encoder_planes = [
            level * (encoder_filters or self.num_filters)
            for level in range(1, self.depth + 1)
        ]
        self.num_planes[-1] = encoder_planes[-1]

        decoding_convolutions = []
        decoding_blocks = []
        for level in range(self.depth - 2, -1, -1):
            upsample: list[torch.nn.Module] = [
                norm_factory(self.norm_cfg, self.num_planes[level + 1]),
                act_factory(self.act_cfg),
                sparse.ConvolutionTranspose(
                    in_channels=self.num_planes[level + 1],
                    out_channels=self.num_planes[level],
                    kernel_size=2,
                    stride=2,
                    dimension=self.dimension,
                    bias=self.allow_bias,
                ),
            ]
            if level in self.dropout_layers:
                upsample.append(sparse.Dropout(p=self.dropout_p))
            decoding_convolutions.append(torch.nn.Sequential(*upsample))

            block_type = DropoutBlock if level in self.dropout_layers else ResNetBlock
            blocks = []
            for repetition in range(self.reps):
                kwargs = {
                    "dimension": self.dimension,
                    "activation": self.act_cfg,
                    "normalization": self.norm_cfg,
                    "bias": self.allow_bias,
                }
                if block_type is DropoutBlock:
                    kwargs["p"] = self.dropout_p
                blocks.append(
                    block_type(
                        self.num_planes[level] * (2 if repetition == 0 else 1),
                        self.num_planes[level],
                        **kwargs,
                    )
                )
            decoding_blocks.append(torch.nn.Sequential(*blocks))

        self.decoding_conv = torch.nn.Sequential(*decoding_convolutions)
        self.decoding_block = torch.nn.Sequential(*decoding_blocks)

    def forward(
        self,
        final: sparse.SparseTensor,
        encoder_tensors: list[sparse.SparseTensor],
    ) -> list[sparse.SparseTensor]:
        """Decode the deepest representation using encoder skip features.

        Parameters
        ----------
        final : sparse.SparseTensor
            Deepest encoder representation.
        encoder_tensors : list of sparse.SparseTensor
            Encoder feature planes, ordered from shallow to deep.

        Returns
        -------
        list of sparse.SparseTensor
            Decoder feature planes, ordered from deep to shallow.

        Raises
        ------
        ValueError
            If the number of encoder tensors does not match ``depth + 1``.
        """
        expected = self.depth + 1
        if len(encoder_tensors) != expected:
            raise ValueError(
                f"Expected {expected} encoder tensors, got " f"{len(encoder_tensors)}."
            )

        decoder_tensors = []
        x = final
        for index, (upsample, block) in enumerate(
            zip(self.decoding_conv, self.decoding_block, strict=True)
        ):
            x = upsample(x)
            x = sparse.cat(encoder_tensors[-index - 2], x)
            x = block(x)
            decoder_tensors.append(x)
        return decoder_tensors
