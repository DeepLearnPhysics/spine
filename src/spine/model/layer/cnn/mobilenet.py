"""Sparse MobileNet-style encoder and encoder-decoder backbones."""

from __future__ import annotations

from typing import Any

import torch

from spine.model import sparse
from spine.utils.factory import Config

from .act_norm import act_factory, norm_factory
from .blocks import MBResConv, MBResConvSE
from .configuration import setup_cnn_configuration
from .uresnet_layers import EncoderOutput, UResNetOutput

__all__ = ["MobileNetV3", "MB3Encoder"]


class MB3Encoder(sparse.Network):
    """Encode sparse images with mobile residual SE blocks.

    The encoder mirrors the UResNet downsampling schedule while replacing
    conventional residual blocks with :class:`MBResConvSE`. It exposes every
    feature plane for use by a decoder or downstream feature extractor.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialize the mobile encoder.

        Parameters
        ----------
        cfg : dict
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.
        """
        super().__init__(cfg.get("data_dim", 3))
        setup_cnn_configuration(self, **cfg)

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
                        self._mobile_block(
                            MBResConvSE,
                            num_features,
                            num_features,
                            dimension=self.dimension,
                            activation=self.act_cfg,
                            normalization=self.norm_cfg,
                            bias=self.allow_bias,
                        )
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

    @staticmethod
    def _mobile_block(
        block_type: type[MBResConv] | type[MBResConvSE],
        in_features: int,
        out_features: int,
        *,
        dimension: int,
        activation: Config,
        normalization: Config,
        bias: bool,
    ) -> torch.nn.Module:
        """Build one MobileNet residual block.

        Parameters
        ----------
        block_type : type
            Concrete mobile residual block class.
        in_features : int
            Number of input feature channels.
        out_features : int
            Number of output feature channels.
        dimension : int
            Number of spatial dimensions.
        activation : str or mapping
            Activation configuration.
        normalization : str or mapping
            Normalization configuration.
        bias : bool
            Whether convolutional and linear layers include bias terms.

        Returns
        -------
        torch.nn.Module
            Initialized mobile residual block.
        """
        return block_type(
            in_features,
            out_features,
            dimension=dimension,
            activation=activation,
            normalization=normalization,
            bias=bias,
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

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        """Encode a coordinate-feature table.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, 1 + D + C)`` table containing batch IDs, coordinates and
            input features.

        Returns
        -------
        EncoderOutput
            Encoder feature planes and deepest representation.
        """
        coords = x[:, : self.dimension + 1].int()
        features = x[:, self.dimension + 1 :]
        sparse_input = sparse.SparseTensor(
            coordinates=coords,
            features=features,
        )
        return self.encode(sparse_input)


class MobileNetV3(MB3Encoder):
    """Sparse MobileNet-style encoder-decoder network.

    The encoder uses squeeze-and-excitation mobile residual blocks. The
    decoder upsamples and concatenates encoder skips using mobile residual
    blocks without SE attention.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialize the mobile encoder and decoder.

        Parameters
        ----------
        cfg : dict
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.
        """
        super().__init__(cfg)

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
                    self._mobile_block(
                        MBResConv,
                        self.num_planes[level] * (2 if repetition == 0 else 1),
                        self.num_planes[level],
                        dimension=self.dimension,
                        activation=self.act_cfg,
                        normalization=self.norm_cfg,
                        bias=self.allow_bias,
                    )
                )
            decoding_blocks.append(torch.nn.Sequential(*blocks))

        self.decoding_conv = torch.nn.Sequential(*decoding_convolutions)
        self.decoding_block = torch.nn.Sequential(*decoding_blocks)

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

    def forward(self, x: torch.Tensor) -> UResNetOutput:
        """Run the encoder-decoder on a coordinate-feature table.

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
        encoded = super().forward(x)
        decoder_tensors = self.decode(
            encoded["final_tensor"], encoded["encoder_tensors"]
        )
        return {
            "encoder_tensors": encoded["encoder_tensors"],
            "decoder_tensors": decoder_tensors,
            "final_tensor": encoded["final_tensor"],
        }
