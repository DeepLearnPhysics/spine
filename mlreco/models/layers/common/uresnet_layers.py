import torch
import torch.nn as nn
import MinkowskiEngine as ME

from mlreco.models.layers.common.blocks import ResNetBlock, CascadeDilationBlock, ASPP
from mlreco.models.layers.common.activation_normalization_factories import activations_construct
from mlreco.models.layers.common.activation_normalization_factories import normalizations_construct
from mlreco.models.layers.common.configuration import setup_cnn_configuration


class UResNetEncoder(torch.nn.Module):
    '''
    Vanilla UResNet with access to intermediate feature planes.

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor
    input_kernel : int, default 3
        Receptive field size for very first convolution after input layer.

    Output
    ------
    encoderTensors: list of ME.SparseTensor
        list of intermediate tensors (taken between encoding block and convolution)
        from encoder half
    finalTensor: ME.SparseTensor
        feature tensor at deepest layer
    features_ppn: list of ME.SparseTensor
        list of intermediate tensors (right after encoding block + convolution)
    '''
    def __init__(self, cfg):
        '''
        Initialize the encoder

        Parameters
        ----------
        cfg : dict
            Encoder configuration block
        '''
        # Initialize the parent class
        super().__init__()

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize the input layer
        self.input_layer = ME.MinkowskiConvolution(
            in_channels = self.num_input, out_channels = self.num_filters,
            kernel_size = self.input_kernel, stride = 1, dimension = self.dim,
            bias = self.allow_bias)

        # Initialize encoder
        self.encoding_conv = []
        self.encoding_block = []
        for i, F in enumerate(self.nPlanes):
            m = []
            for _ in range(self.reps):
                m.append(ResNetBlock(F, F,
                    dimension = self.dim,
                    activation = self.activation_name,
                    activation_args = self.activation_args,
                    normalization = self.norm,
                    normalization_args = self.norm_args,
                    bias = self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_block.append(m)
            m = []
            if i < self.depth-1:
                m.append(normalizations_construct(self.norm,
                    F, **self.norm_args))
                m.append(activations_construct(
                    self.activation_name, **self.activation_args))
                m.append(ME.MinkowskiConvolution(in_channels = self.nPlanes[i],
                    out_channels=self.nPlanes[i+1],
                    kernel_size = 2, stride = 2, dimension = self.dim,
                    bias = self.allow_bias))
            m = nn.Sequential(*m)
            self.encoding_conv.append(m)
        self.encoding_conv = nn.Sequential(*self.encoding_conv)
        self.encoding_block = nn.Sequential(*self.encoding_block)


    def encoder(self, x):
        '''
        Vanilla UResNet Encoder.

        Parameters
        ----------
        x : MinkowskiEngine SparseTensor

        Returns
        -------
        dict
        '''
        # print('input' , self.input_layer)
        # for name, param in self.input_layer.named_parameters():
        #     print(name, param.shape, param)
        x = self.input_layer(x)
        encoderTensors = [x]
        features_ppn = [x]
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            encoderTensors.append(x)
            x = self.encoding_conv[i](x)
            features_ppn.append(x)

        result = {
            "encoderTensors": encoderTensors,
            "features_ppn": features_ppn,
            "finalTensor": x
        }
        return result


    def forward(self, input):

        encoderOutput = self.encoder(input)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']

        res = {
            'encoderTensors': encoderTensors,
            'finalTensor': finalTensor,
            'features_ppn': encoderOutput['features_ppn']
        }
        return res


class UResNetDecoder(torch.nn.Module):
    """
    Vanilla UResNet Decoder

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor

    Output
    ------
    list of ME.SparseTensor
    """
    def __init__(self, cfg, name='uresnet_decoder'):
        # Initialize the parent class
        super().__init__()

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize decoder
        self.decoding_block = []
        self.decoding_conv = []
        for i in range(self.depth-2, -1, -1):
            m = []
            m.append(normalizations_construct(self.norm, self.nPlanes[i+1], **self.norm_args))
            m.append(activations_construct(
                self.activation_name, **self.activation_args))
            m.append(ME.MinkowskiConvolutionTranspose(
                in_channels = self.nPlanes[i+1], out_channels = self.nPlanes[i],
                kernel_size = 2, stride = 2, dimension = self.dim,
                bias = self.allow_bias))
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)
            m = []
            for j in range(self.reps):
                m.append(ResNetBlock(self.nPlanes[i] * (2 if j == 0 else 1),
                                     self.nPlanes[i],
                                     dimension = self.dim,
                                     activation = self.activation_name,
                                     activation_args = self.activation_args,
                                     normalization = self.norm,
                                     normalization_args = self.norm_args,
                                     bias = self.allow_bias))
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)


    def decoder(self, final, encoderTensors):
        '''
        Vanilla UResNet Decoder

        Parameters
        ----------
        encoderTensors : list of SparseTensor
            output of encoder.

        Returns
        -------
        decoderTensors : list of SparseTensor
            list of feature tensors in decoding path at each spatial resolution.
        '''
        decoderTensors = []
        x = final
        for i, layer in enumerate(self.decoding_conv):
            eTensor = encoderTensors[-i-2]
            x = layer(x)
            x = ME.cat(eTensor, x)
            x = self.decoding_block[i](x)
            decoderTensors.append(x)
        return decoderTensors

    def forward(self, final, encoderTensors):
        return self.decoder(final, encoderTensors)


class UResNet(torch.nn.Module):
    '''
    Vanilla UResNet with access to intermediate feature planes.

    Configuration
    -------------
    data_dim: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth : int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters : int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps : int, default 2
        Convolution block repetition factor
    input_kernel : int, default 3
        Receptive field size for very first convolution after input layer.

    Output
    ------
    encoderTensors: list of ME.SparseTensor
        list of intermediate tensors (taken between encoding block and convolution)
        from encoder half
    decoderTensors: list of ME.SparseTensor
        list of intermediate tensors (taken between encoding block and convolution)
        from decoder half
    finalTensor: ME.SparseTensor
        feature tensor at deepest layer
    features_ppn: list of ME.SparseTensor
        list of intermediate tensors (right after encoding block + convolution)
    '''
    def __init__(self, cfg):
        # Initialize the parent class
        super().__init__()

        # Process the configuration
        setup_cnn_configuration(self, **cfg)

        # Initialize the encoder/decoder blocks of the UResNet model
        self.encoder = UResNetEncoder(cfg)
        self.decoder = UResNetDecoder(cfg)

    def forward(self, input):
        coords = input[:, 0:self.dim + 1].int()
        features = input[:, self.dim + 1:].float()

        x = ME.SparseTensor(features, coordinates=coords)
        encoderOutput = self.encoder(x)
        encoderTensors = encoderOutput['encoderTensors']
        finalTensor = encoderOutput['finalTensor']
        decoderTensors = self.decoder(finalTensor, encoderTensors)

        res = {
            'encoderTensors': encoderTensors,
            'decoderTensors': decoderTensors,
            'finalTensor': finalTensor,
            'features_ppn': encoderOutput['features_ppn']
        }

        return res
