"""Sparse UResNeXt encoder-decoder backbone."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from spine.model import sparse

from .act_norm import act_factory, norm_factory
from .blocks import ResNeXtBlock
from .configuration import setup_cnn_configuration
from .uresnet_layers import EncoderOutput, UResNetOutput

__all__ = ["UResNeXt"]


class UResNeXt(sparse.Network):
    """Sparse U-shaped backbone built from grouped ResNeXt paths.

    This architecture follows the UResNet resolution schedule and skip
    topology while replacing residual blocks with :class:`ResNeXtBlock`.
    ``cardinality`` controls the number of parallel transformation paths.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        cardinality: int = 8,
        dilations: Sequence[int] | None = None,
    ) -> None:
        """Initialize the UResNeXt backbone.

        Parameters
        ----------
        cfg : dict
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.
        cardinality : int, default 8
            Number of parallel paths in each ResNeXt block.
        dilations : sequence of int, optional
            Dilation rate for each cardinal path. Defaults to rates increasing
            from one to four across the paths.

        Raises
        ------
        ValueError
            If ``cardinality`` is not positive, feature widths are not
            divisible by it, or the dilation count does not match it.
        """
        super().__init__(cfg.get("data_dim", 3))
        setup_cnn_configuration(self, **cfg)

        if cardinality < 1:
            raise ValueError(f"`cardinality` must be positive, got {cardinality}.")
        if any(plane % cardinality for plane in self.num_planes):
            raise ValueError(
                "Every feature-plane width must be divisible by `cardinality`."
            )
        if dilations is None:
            dilations = tuple(2 ** min(index // 2, 2) for index in range(cardinality))
        if len(dilations) != cardinality:
            raise ValueError("Expected `len(dilations) == cardinality`.")

        self.cardinality = cardinality
        self.dilations = tuple(dilations)
        self.input_layer = sparse.Convolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel,
            stride=1,
            dimension=self.dimension,
            bias=self.allow_bias,
        )

        encoding_blocks = []
        encoding_convolutions = []
        for level, num_features in enumerate(self.num_planes):
            encoding_blocks.append(
                torch.nn.Sequential(
                    *[
                        self._make_block(num_features, num_features)
                        for _ in range(self.reps)
                    ]
                )
            )
            downsample = []
            if level < self.depth - 1:
                downsample = [
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
            encoding_convolutions.append(torch.nn.Sequential(*downsample))
        self.encoding_block = torch.nn.Sequential(*encoding_blocks)
        self.encoding_conv = torch.nn.Sequential(*encoding_convolutions)

        decoding_blocks = []
        decoding_convolutions = []
        for level in range(self.depth - 2, -1, -1):
            decoding_convolutions.append(
                torch.nn.Sequential(
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
                )
            )
            blocks = []
            for repetition in range(self.reps):
                blocks.append(
                    self._make_block(
                        self.num_planes[level] * (2 if repetition == 0 else 1),
                        self.num_planes[level],
                    )
                )
            decoding_blocks.append(torch.nn.Sequential(*blocks))
        self.decoding_block = torch.nn.Sequential(*decoding_blocks)
        self.decoding_conv = torch.nn.Sequential(*decoding_convolutions)

    def _make_block(
        self,
        in_features: int,
        out_features: int,
    ) -> torch.nn.Module:
        """Build one configured ResNeXt block.

        Parameters
        ----------
        in_features : int
            Number of input feature channels.
        out_features : int
            Number of output feature channels.

        Returns
        -------
        torch.nn.Module
            Initialized grouped residual block.
        """
        return ResNeXtBlock(
            in_features,
            out_features,
            dimension=self.dimension,
            cardinality=self.cardinality,
            dilations=self.dilations,
            activation=self.act_cfg,
            normalization=self.norm_cfg,
        )

    def encode(self, x: sparse.SparseTensor) -> EncoderOutput:
        """Encode an existing sparse tensor.

        Parameters
        ----------
        x : sparse.SparseTensor
            Sparse input with ``num_input`` feature channels.

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

    def decode(
        self,
        final: sparse.SparseTensor,
        encoder_tensors: list[sparse.SparseTensor],
    ) -> list[sparse.SparseTensor]:
        """Decode a representation using concatenated encoder features.

        Parameters
        ----------
        final : sparse.SparseTensor
            Deepest encoder representation.
        encoder_tensors : list of sparse.SparseTensor
            Encoder feature planes ordered from shallow to deep.

        Returns
        -------
        list of sparse.SparseTensor
            Decoder feature planes ordered from deep to shallow.

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

    def forward(self, x: torch.Tensor) -> UResNetOutput:
        """Run UResNeXt on a coordinate-feature table.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, 1 + D + C)`` table containing batch IDs, coordinates and
            input features.

        Returns
        -------
        UResNetOutput
            Encoder, decoder and deepest sparse feature tensors.
        """
        coords = x[:, : self.dimension + 1].int()
        features = x[:, self.dimension + 1 :]
        sparse_input = sparse.SparseTensor(
            coordinates=coords,
            features=features,
        )
        encoded = self.encode(sparse_input)
        decoder_tensors = self.decode(
            encoded["final_tensor"], encoded["encoder_tensors"]
        )
        return {
            "encoder_tensors": encoded["encoder_tensors"],
            "decoder_tensors": decoder_tensors,
            "final_tensor": encoded["final_tensor"],
        }
