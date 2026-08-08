"""Module with all the backbone components of UResNet.

Contains the following components:
  - `UResNetEncoder`: Encoder component of UResNet
  - `UResNetDecoder`: Decoder component of UResNet
  - `UResNet`: Full encoder/decoder architecture of UResNet
"""

from __future__ import annotations

from typing import Any, TypedDict

import torch

from spine.model import sparse

from .act_norm import act_factory, norm_factory
from .blocks import ResNetBlock
from .configuration import setup_cnn_configuration

__all__ = [
    "EncoderOutput",
    "UResNetOutput",
    "UResNetEncoder",
    "UResNetDecoder",
    "UResNet",
]


class EncoderOutput(TypedDict):
    """Sparse feature planes produced by a CNN encoder.

    Attributes
    ----------
    encoder_tensors : list of sparse.SparseTensor
        Feature planes ordered from shallow to deep. The list includes the
        initial input projection and each residual stage.
    final_tensor : sparse.SparseTensor
        Deepest encoded representation.
    """

    encoder_tensors: list[sparse.SparseTensor]
    final_tensor: sparse.SparseTensor


class UResNetOutput(EncoderOutput):
    """Sparse feature planes produced by an encoder-decoder backbone.

    Attributes
    ----------
    decoder_tensors : list of sparse.SparseTensor
        Decoder feature planes ordered from deep to shallow.
    """

    decoder_tensors: list[sparse.SparseTensor]


class UResNetEncoder(sparse.Network):
    """Encode sparse images into a multi-resolution residual feature pyramid.

    Each level applies ``reps`` residual blocks before a stride-two sparse
    convolution moves to the next feature resolution. Feature widths increase
    linearly from ``filters`` to ``depth * filters``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialize the encoder.

        Parameters
        ----------
        cfg : dict
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.
        """
        # Initialize the parent class
        super().__init__(cfg.get("data_dim", 3))

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize the input layer
        self.input_layer = sparse.Convolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel,
            stride=1,
            dimension=self.dimension,
            bias=self.allow_bias,
        )

        # Initialize encoder
        encoding_convolutions = []
        encoding_blocks = []
        for level, num_features in enumerate(self.num_planes):
            blocks = []
            for _ in range(self.reps):
                blocks.append(
                    ResNetBlock(
                        num_features,
                        num_features,
                        dimension=self.dimension,
                        activation=self.act_cfg,
                        normalization=self.norm_cfg,
                        bias=self.allow_bias,
                    )
                )
            encoding_blocks.append(torch.nn.Sequential(*blocks))
            downsample = []
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
            encoding_convolutions.append(torch.nn.Sequential(*downsample))
        self.encoding_conv = torch.nn.Sequential(*encoding_convolutions)
        self.encoding_block = torch.nn.Sequential(*encoding_blocks)

    def forward(self, x: sparse.SparseTensor) -> EncoderOutput:
        """Encode a sparse tensor into multi-resolution features.

        Parameters
        ----------
        x : sparse.SparseTensor
            Sparse tensor with ``num_input`` feature channels.

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


class UResNetDecoder(sparse.Network):
    """Decode a sparse feature pyramid with concatenated skip connections.

    Each level upsamples the previous representation, concatenates the
    corresponding encoder feature plane, and fuses the result with residual
    blocks.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialize the decoder.

        Parameters
        ----------
        cfg : dict
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.
        """
        # Initialize the parent class
        super().__init__(cfg.get("data_dim", 3))

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize decoder
        decoding_blocks = []
        decoding_convolutions = []
        for level in range(self.depth - 2, -1, -1):
            upsample = torch.nn.Sequential(
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
            decoding_convolutions.append(upsample)
            blocks = []
            for repetition in range(self.reps):
                blocks.append(
                    ResNetBlock(
                        self.num_planes[level] * (2 if repetition == 0 else 1),
                        self.num_planes[level],
                        dimension=self.dimension,
                        activation=self.act_cfg,
                        normalization=self.norm_cfg,
                        bias=self.allow_bias,
                    )
                )
            decoding_blocks.append(torch.nn.Sequential(*blocks))
        self.decoding_block = torch.nn.Sequential(*decoding_blocks)
        self.decoding_conv = torch.nn.Sequential(*decoding_convolutions)

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
            encoder_tensor = encoder_tensors[-index - 2]
            x = upsample(x)
            x = sparse.cat(encoder_tensor, x)
            x = block(x)
            decoder_tensors.append(x)

        return decoder_tensors


class UResNet(sparse.Network):
    """Sparse UResNet backbone exposing all encoder and decoder feature planes.

    The backbone combines :class:`UResNetEncoder` and
    :class:`UResNetDecoder`. Unlike a task-specific segmentation model, it
    returns sparse intermediate representations without applying an output
    classifier.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialize the UResNet backbone.

        Parameters
        ----------
        cfg : dict
            Shared CNN configuration accepted by
            :func:`setup_cnn_configuration`.
        """
        # Initialize the parent class
        super().__init__(cfg.get("data_dim", 3))

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize the encoder/decoder blocks of the UResNet model
        self.encoder = UResNetEncoder(cfg)
        self.decoder = UResNetDecoder(cfg)

    def forward(
        self,
        x: torch.Tensor,
        batch_size: int | None = None,
    ) -> UResNetOutput:
        """Run the UResNet backbone on a coordinate-feature table.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, 1 + D + C)`` table containing batch IDs, coordinates and
            input features.
        batch_size : int, optional
            Explicit number of batch entries. This preserves trailing empty
            entries that cannot be inferred from coordinates.

        Returns
        -------
        UResNetOutput
            Encoder, decoder and deepest sparse feature tensors.
        """
        # Cast the input data to a sparse tensor
        coords = x[:, : self.dimension + 1].int()
        features = x[:, self.dimension + 1 :]
        sparse_input = sparse.SparseTensor(
            coordinates=coords,
            features=features,
            batch_size=batch_size,
        )

        # Pass it through the encoder
        encoder_output = self.encoder(sparse_input)
        encoder_tensors = encoder_output["encoder_tensors"]
        final_tensor = encoder_output["final_tensor"]

        # Pass it through the decoder
        decoder_tensors = self.decoder(final_tensor, encoder_tensors)

        return {
            "encoder_tensors": encoder_tensors,
            "decoder_tensors": decoder_tensors,
            "final_tensor": final_tensor,
        }
