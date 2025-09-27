"""Module with all the backbone components of UResNet.

Contains the following components:
  - `UResNetEncoder`: Encoder component of UResNet
  - `UResNetDecoder`: Decoder component of UResNet
  - `UResNet`: Full encoder/decoder architecture of UResNet
"""

from typing import List

import MinkowskiEngine as ME
import torch

from .act_norm import act_factory, norm_factory
from .blocks import ASPP, CascadeDilationBlock, ResNetBlock
from .configuration import setup_cnn_configuration

__all__ = ["UResNetEncoder", "UResNetDecoder", "UResNet"]


class UResNetEncoder(torch.nn.Module):
    """Vanilla UResNet encoder.

    See :func:`setup_cnn_configuration` for available parameters.
    """

    def __init__(self, cfg):
        """Initialize the encoder.

        Parameters
        ----------
        cfg : dict
            Encoder configuration block
        """
        # Initialize the parent class
        super().__init__()

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize the input layer
        self.input_layer = ME.MinkowskiConvolution(
            in_channels=self.num_input,
            out_channels=self.num_filters,
            kernel_size=self.input_kernel,
            stride=1,
            dimension=self.dim,
            bias=self.allow_bias,
        )

        # Initialize encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.num_planes):
            m = []
            for _ in range(self.reps):
                m.append(
                    ResNetBlock(
                        F,
                        F,
                        dimension=self.dim,
                        activation=self.act_cfg,
                        normalization=self.norm_cfg,
                        bias=self.allow_bias,
                    )
                )
            m = torch.nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth - 1:
                m.append(norm_factory(self.norm_cfg, F))
                m.append(act_factory(self.act_cfg))
                m.append(
                    ME.MinkowskiConvolution(
                        in_channels=self.num_planes[i],
                        out_channels=self.num_planes[i + 1],
                        kernel_size=2,
                        stride=2,
                        dimension=self.dim,
                        bias=self.allow_bias,
                    )
                )
            m = torch.nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = torch.nn.Sequential(*self.encoding_conv)
        self.encoding_block = torch.nn.Sequential(*self.encoding_block)

    def forward(self, x):
        """Pass a tensor through the encoder.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        encoder_tensors : List[ME.SparseTensor]
            List of intermediate tensors (taken between encoding block and
            convolution) from the encoder half
        final_tensor : ME.SparseTensor
            Feature tensor at deepest layer
        """
        x = self.input_layer(x)
        encoder_tensors = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoder_tensors.append(x)
            x = self.encoding_conv[i](x)

        result = {"encoder_tensors": encoder_tensors, "final_tensor": x}

        return result


class UResNetDecoder(torch.nn.Module):
    """Vanilla UResNet Decoder.

    See :func:`setup_cnn_configuration` for available parameters.
    """

    def __init__(self, cfg):
        """Initialize the decoder.

        Parameters
        ----------
        cfg : dict
            Decoder configuration block
        """
        # Initialize the parent class
        super().__init__()

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth - 2, -1, -1):
            m = []
            m.append(norm_factory(self.norm_cfg, self.num_planes[i + 1]))
            m.append(act_factory(self.act_cfg))
            m.append(
                ME.MinkowskiConvolutionTranspose(
                    in_channels=self.num_planes[i + 1],
                    out_channels=self.num_planes[i],
                    kernel_size=2,
                    stride=2,
                    dimension=self.dim,
                    bias=self.allow_bias,
                )
            )
            m = torch.nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(
                    ResNetBlock(
                        self.num_planes[i] * (2 if j == 0 else 1),
                        self.num_planes[i],
                        dimension=self.dim,
                        activation=self.act_cfg,
                        normalization=self.norm_cfg,
                        bias=self.allow_bias,
                    )
                )
            m = torch.nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = torch.nn.Sequential(*self.decoding_block)
        self.decoding_conv = torch.nn.Sequential(*self.decoding_conv)

    def forward(self, final, encoder_tensors):
        """Pass a tensor through the decoder.

        Parameters
        ----------
        final : ME.SparseTensor
            Output of the encoder
        encoder_tensors : List[ME.SparseTensor]
            List of tensors from each depth of the encoder

        Returns
        -------
        List[ME.SparseTensor]
            List of feature tensors in decoding path at each spatial resolution
        """
        decoder_tensors = []
        x = final
        for i, layer in enumerate(self.decoding_conv):
            encoder_tensor = encoder_tensors[-i - 2]
            x = layer(x)
            x = ME.cat(encoder_tensor, x)
            x = self.decoding_block[i](x)
            decoder_tensors.append(x)

        return decoder_tensors


class UResNet(torch.nn.Module):
    """Vanilla UResNet with access to intermediate feature planes.

    See :func:`setup_cnn_configuration` for available parameters.
    """

    def __init__(self, cfg):
        """Initialize the UResNet backbone.

        Parameters
        ----------
        cfg : dict
            Decoder configuration block
        """
        # Initialize the parent class
        super().__init__()

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize the encoder/decoder blocks of the UResNet model
        self.encoder = UResNetEncoder(cfg)
        self.decoder = UResNetDecoder(cfg)

    def forward(self, input_data):
        """Pass a tensor through the UResNet backbone.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        encoder_tensors : List[ME.SparseTensor]
            List of intermediate tensors (taken between encoding block and
            convolution) from the encoder half
        decoder_tensors : List[ME.SparseTensor]
            List of feature tensors in decoding path at each spatial resolution
        final_tensor : ME.SparseTensor
            Feature tensor at deepest layer
        """
        # Cast the input data to a sparse tensor
        coords = input_data[:, 0 : self.dim + 1].int()
        features = input_data[:, self.dim + 1 :]
        x = ME.SparseTensor(features, coordinates=coords)

        # Pass it through the encoder
        encoder_output = self.encoder(x)
        encoder_tensors = encoder_output["encoder_tensors"]
        final_tensor = encoder_output["final_tensor"]

        # Pass it through the decoder
        decoder_tensors = self.decoder(final_tensor, encoder_tensors)

        # Return
        res = {
            "encoder_tensors": encoder_tensors,
            "decoder_tensors": decoder_tensors,
            "final_tensor": final_tensor,
        }

        return res
