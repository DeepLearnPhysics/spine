"""Sparse feature pyramid network backbone."""

from __future__ import annotations

from typing import Any

import torch

from spine.model import sparse

from .act_norm import act_factory, norm_factory
from .blocks import ResNetBlock
from .configuration import setup_cnn_configuration
from .uresnet_layers import EncoderOutput, UResNetOutput

__all__ = ["FPN"]


class FPN(sparse.Network):
    """Feature pyramid network with additive lateral skip connections.

    The configuration follows :func:`setup_cnn_configuration`. Unlike UResNet,
    which concatenates encoder and decoder features, FPN projects each encoder
    feature plane and adds it to the corresponding decoder plane. The returned
    feature collections are ordered from shallow to deep in the encoder and
    from deep to shallow in the decoder.

    References
    ----------
    .. [1] Lin et al., "Feature Pyramid Networks for Object Detection," 2017.
       https://arxiv.org/abs/1612.03144
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialize the feature pyramid.

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
                        ResNetBlock(
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

        self.lateral = torch.nn.Sequential(
            *[
                sparse.Linear(num_features, num_features)
                for num_features in reversed(self.num_planes[:-1])
            ]
        )

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
            decoding_blocks.append(
                torch.nn.Sequential(
                    *[
                        ResNetBlock(
                            self.num_planes[level],
                            self.num_planes[level],
                            dimension=self.dimension,
                            activation=self.act_cfg,
                            normalization=self.norm_cfg,
                            bias=self.allow_bias,
                        )
                        for _ in range(self.reps)
                    ]
                )
            )

        self.decoding_conv = torch.nn.Sequential(*decoding_convolutions)
        self.decoding_block = torch.nn.Sequential(*decoding_blocks)

    def encode(self, x: sparse.SparseTensor) -> EncoderOutput:
        """Encode a sparse tensor into a multi-resolution feature pyramid.

        Parameters
        ----------
        x : sparse.SparseTensor
            Sparse input with ``num_input`` feature channels.

        Returns
        -------
        EncoderOutput
            Encoder feature planes and the deepest representation.
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
        """Decode a representation using additive lateral features.

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
        modules = zip(
            self.decoding_conv,
            self.lateral,
            self.decoding_block,
            strict=True,
        )
        for index, (upsample, lateral, block) in enumerate(modules):
            x = upsample(x)
            x += lateral(encoder_tensors[-index - 2])
            x = block(x)
            decoder_tensors.append(x)
        return decoder_tensors

    def forward(self, x: torch.Tensor) -> UResNetOutput:
        """Run the feature pyramid on a coordinate-feature table.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, 1 + D + C)`` table containing a batch column, ``D`` spatial
            coordinates and ``C`` input features.

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
