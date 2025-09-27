from typing import Union

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn

from .act_norm import act_factory, norm_factory


class ConvolutionBlock(ME.MinkowskiNetwork):
    """Convolution block which operates a sequence of
    two (convolution + nomalization + activation) steps.
    """

    def __init__(
        self,
        in_features,
        out_features,
        stride=1,
        dilation=1,
        dimension=3,
        activation="relu",
        normalization="batch_norm",
        bias=False,
    ):
        """Initialize the convolution block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        stride : int, default 1
            Convolution kernel stride
        dilation : int, default 1
            Convolution kernel dilation
        dimension : int, default 3
            Dimension of the input image
        activation : union[str, dict], default 'relu'
            activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            normalization function configuration
        bias : bool, default False
            Whether to add a bias term to the kernel
        """
        # Initialize the parent class
        super().__init__(dimension)

        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            in_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
            bias=bias,
        )
        self.norm1 = norm_factory(normalization, out_features)
        self.act_fn1 = act_factory(activation)

        self.conv2 = ME.MinkowskiConvolution(
            out_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
            bias=bias,
        )
        self.norm2 = norm_factory(normalization, out_features)
        self.act_fn2 = act_factory(activation)

    def forward(self, x):
        """Pass a tensor through the convolution block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fn1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act_fn2(out)

        return out


class DropoutBlock(ME.MinkowskiNetwork):
    """Convolution block which operates a sequence of
    two (convolution + dropout + nomalization + activation) steps.
    """

    def __init__(
        self,
        in_features,
        out_features,
        stride=1,
        dilation=1,
        dimension=3,
        p=0.5,
        activation="relu",
        normalization="batch_norm",
        bias=False,
    ):
        """Initialize the dropout block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        stride : int, default 1
            Convolution kernel stride
        dilation : int, default 1
            Convolution kernel dilation
        p : float, default 0.5
            Dropout probability
        dimension : int, default 3
            Dimension of the input image
        activation : union[str, dict], default 'relu'
            activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            normalization function configuration
        bias : bool, default False
            Whether to add a bias term to the kernel
        """
        # Initialize the parent class
        super().__init__(dimension)

        assert dimension > 0

        self.conv1 = ME.MinkowskiConvolution(
            in_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
            bias=bias,
        )
        self.dropout1 = ME.MinkowskiDropout(p=p)
        self.norm1 = norm_factory(normalization, out_features)
        self.act_fn1 = act_factory(activation)

        self.conv2 = ME.MinkowskiConvolution(
            out_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
            bias=bias,
        )
        self.dropout2 = ME.MinkowskiDropout(p=p)
        self.norm2 = norm_factory(normalization, out_features)
        self.act_fn2 = act_factory(activation)

    def forward(self, x):
        """Pass a tensor through the dropout block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        out = self.conv1(x)
        out = self.dropout1(out)
        out = self.norm1(out)
        out = self.act_fn1(out)

        out = self.conv2(out)
        out = self.dropout2(out)
        out = self.norm2(out)
        out = self.act_fn2(out)

        return out


class ResNetBlock(ME.MinkowskiNetwork):
    """ResNet Block."""

    def __init__(
        self,
        in_features,
        out_features,
        stride=1,
        dilation=1,
        dimension=3,
        activation="relu",
        normalization="batch_norm",
        bias=False,
    ):
        """Initialize the ResNet block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        stride : int, default 1
            Convolution kernel stride
        dilation : int, default 1
            Convolution kernel dilation
        dimension : int, default 3
            Dimension of the input image
        activation : union[str, dict], default 'relu'
            activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            normalization function configuration
        bias : bool, default False
            Whether to add a bias term to the kernel
        """
        # Initialize the parent class
        super().__init__(dimension)

        assert dimension > 0

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features, bias=bias)
        else:
            self.residual = nn.Identity()

        self.conv1 = ME.MinkowskiConvolution(
            in_features,
            out_features,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
            bias=bias,
        )
        self.norm1 = norm_factory(normalization, in_features)
        self.act_fn1 = act_factory(activation)

        self.conv2 = ME.MinkowskiConvolution(
            out_features,
            out_features,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
            bias=bias,
        )
        self.norm2 = norm_factory(normalization, out_features)
        self.act_fn2 = act_factory(activation)

    def forward(self, x):
        """Pass a tensor through the ResNet block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        residual = self.residual(x)

        out = self.conv1(self.act_fn1(self.norm1(x)))
        out = self.conv2(self.act_fn2(self.norm2(out)))

        out += residual

        return out


class AtrousIIBlock(ME.MinkowskiNetwork):
    """ResNet-type block with atrous convolutions.

    Developed for the ACNN paper: "ACNN: a Full Resolution DCNN for Medical
    Image Segmentation"

    Original paper: https://arxiv.org/pdf/1901.09203.pdf
    """

    def __init__(
        self,
        in_features,
        out_features,
        dimension=3,
        activation="relu",
        normalization="batch_norm",
    ):
        """Initialize the AtrousII block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        dimension : int, default 3
            Dimension of the input image
        activation : union[str, dict], default 'relu'
            activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            normalization function configuration
        """
        # Initialize the parent class
        super().__init__(dimension)

        assert dimension > 0

        self.D = dimension

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = nn.Identity()

        self.conv1 = ME.MinkowskiConvolution(
            in_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=1,
            dimension=self.D,
        )
        self.norm1 = norm_factory(normalization, out_features)
        self.act_fn1 = act_factory(activation)

        self.conv2 = ME.MinkowskiConvolution(
            out_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=3,
            dimension=self.D,
        )
        self.norm2 = norm_factory(normalization, out_features)
        self.act_fn2 = act_factory(activation)

    def forward(self, x):
        """Pass a tensor through the AtrousII block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fn1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.act_fn2(out)

        return out


class ResNeXtBlock(ME.MinkowskiNetwork):
    """ResNeXt-type block with atrous convolutions.

    Notes
    -----
    For vanilla resnext blocks, set `dilation=1` and others to default.
    """

    def __init__(
        self,
        in_features,
        out_features,
        dimension=3,
        cardinality=4,
        depth=1,
        dilations=None,
        kernel_sizes=3,
        strides=1,
        activation="relu",
        normalization="batch_norm",
    ):
        """Initialize the ResNeXt block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        dimension : int, default 3
            Dimension of the input image
        cardinality : int, default 4
            Number of different paths, see ResNeXt paper
        depth : int, default 1
            Number of (convolutions + normalization + activation) layers
        dilations : int, optional
            Dilation rates for each convolution layer inside the cardinal paths
        kernel_sizes : int, default 3
            Kernel sizes for each convolution layer inside the cardinal paths
        strides : int, default 1
            Strides for each convolution layer inside the carndinal paths
        activation : union[str, dict], default 'relu'
            activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            normalization function configuration
        """
        # Initialize the parent class
        super().__init__(dimension)

        assert dimension > 0
        assert cardinality > 0
        assert in_features % cardinality == 0 and out_features % cardinality == 0

        self.D = dimension
        num_input = in_features // cardinality
        num_output = out_features // cardinality

        self.dilations = []
        if dilations is None:
            # Default
            self.dilations = [3**i for i in range(cardinality)]
        elif isinstance(dilations, int):
            self.dilations = [dilations for _ in range(cardinality)]
        elif isinstance(dilations, list):
            assert len(dilations) == cardinality
            self.dilations = dilations
        else:
            raise ValueError("Invalid type for input strides, must be int or list!")

        self.kernels = []
        if isinstance(kernel_sizes, int):
            self.kernels = [kernel_sizes for _ in range(cardinality)]
        elif isinstance(kernel_sizes, list):
            assert len(kernel_sizes) == cardinality
            self.kernels = kernel_sizes
        else:
            raise ValueError("Invalid type for input strides, must be int or list!")

        self.strides = []
        if isinstance(strides, int):
            self.strides = [strides for _ in range(cardinality)]
        elif isinstance(strides, list):
            assert len(strides) == cardinality
            self.strides = strides
        else:
            raise ValueError("Invalid type for input strides, must be int or list!")

        # For each path, generate sequentials
        self.paths = []
        for i in range(cardinality):
            m = []
            m.append(ME.MinkowskiLinear(in_features, num_input))
            for j in range(depth):
                in_C = num_input if j == 0 else num_output
                m.append(
                    ME.MinkowskiConvolution(
                        in_channels=in_C,
                        out_channels=num_output,
                        kernel_size=self.kernels[i],
                        stride=self.strides[i],
                        dilation=self.dilations[i],
                        dimension=self.D,
                    )
                )
                m.append(norm_factory(normalization, num_output))
                m.append(act_factory(activation))
            m = nn.Sequential(*m)
            self.paths.append(m)
        self.paths = nn.Sequential(*self.paths)
        self.linear = ME.MinkowskiLinear(out_features, out_features)

        # Skip Connection
        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """Pass a tensor through the ResNeXt block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        residual = self.residual(x)

        cat = tuple([layer(x) for layer in self.paths])

        out = ME.cat(cat)
        out = self.linear(out)
        out += residual

        return out


class SPP(ME.MinkowskiNetwork):
    """Spatial Pyramid Pooling.

    PSPNet (Pyramid Scene Parsing Network) uses vanilla SPPs, while
    DeeplabV3 and DeeplabV3+ uses ASPP (atrous versions).

    Default parameters will construct a global average pooling + unpooling
    layer which is done in ParseNet.
    """

    def __init__(
        self,
        in_features,
        out_features,
        kernel_sizes=None,
        dilations=None,
        mode="avg",
        dimension=3,
    ):
        """Initialize the SPP block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        kernel_sizes : int, optional
            Kernel sizes for each pooling operation
        dilations : int, optional
            Dilation rates for atrous convolutions. Note that
            `kernel_size == stride` for an SPP layer.
        mode : str, default 'avg'
            Pooling mode (one of 'avg', 'max' and 'sum'
        dimension : int, default 3
            Dimension of the input image
        """
        # Initialize the parent class
        super().__init__(dimension)

        if mode == "avg":
            self.pool_fn = ME.MinkowskiAvgPooling
        elif mode == "max":
            self.pool_fn = ME.MinkowskiMaxPooling
        elif mode == "sum":
            self.pool_fn = ME.MinkowskiSumPooling
        else:
            raise ValueError(
                "Invalid pooling mode, must be one of \
                'sum', 'max' or 'average'"
            )

        self.unpool_fn = ME.MinkowskiPoolingTranspose

        # Include global pooling as first modules.
        self.pool = [ME.MinkowskiGlobalPooling(dimension=dimension)]
        self.unpool = [ME.MinkowskiBroadcast(dimension=dimension)]
        multiplier = 1

        # Define subregion poolings
        self.spp = []
        if kernel_sizes is not None:
            if isinstance(dilations, int):
                dilations = [dilations for _ in range(len(kernel_sizes))]
            elif isinstance(dilations, list):
                assert len(kernel_sizes) == len(dilations)
            else:
                raise ValueError(
                    "Invalid input to dilations, must be either " "int or list of ints."
                )
            multiplier = len(kernel_sizes) + 1  # Additional 1 for globalPool
            for k, d in zip(kernel_sizes, dilations):
                pooling_layer = self.pool_fn(
                    kernel_size=k, dilation=d, stride=k, dimension=dimension
                )
                unpooling_layer = self.unpool_fn(
                    kernel_size=k, dilation=d, stride=k, dimension=dimension
                )
                self.pool.append(pooling_layer)
                self.unpool.append(unpooling_layer)
        self.pool = nn.Sequential(*self.pool)
        self.unpool = nn.Sequential(*self.unpool)
        self.linear = ME.MinkowskiLinear(in_features * multiplier, out_features)

    def forward(self, input):
        """Pass a tensor through the SPP block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        cat = []
        for i, pool in enumerate(self.pool):
            x = pool(input)
            # First item is Global Pooling
            if i == 0:
                x = self.unpool[i](input, x)
            else:
                x = self.unpool[i](x)
            cat.append(x)
        out = ME.cat(cat)
        out = self.linear(out)

        return out


class ASPP(ME.MinkowskiNetwork):
    """Atrous Spatial Pyramid Pooling."""

    def __init__(
        self,
        in_features,
        out_features,
        dimension=3,
        width=5,
        dilations=[2, 4, 6, 8, 12],
    ):
        """Initialize the ASPP block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        dimension : int, default 3
            Dimension of the input image
        dilations : list, default [2, 4, 6, 8, 12]
            Dilation rates for atrous convolutions
        width : int, default 5
            Width of atrous convolutions
        """
        # Initialize parent class
        super().__init__(dimension)

        assert len(dilations) == width

        m = []
        m.append(ME.MinkowskiLinear(in_features, out_features))
        for d in dilations:
            m.append(
                ME.MinkowskiConvolution(
                    in_features,
                    out_features,
                    kernel_size=3,
                    dilation=d,
                    dimension=self.D,
                )
            )
        self.net = nn.Sequential(*m)
        self.pool = ME.MinkowskiGlobalPooling(dimension=self.D)
        self.unpool = ME.MinkowskiBroadcast(dimension=self.D)
        self.out = nn.Sequential(
            ME.MinkowskiConvolution(
                in_features * (2 + width),
                out_features,
                kernel_size=3,
                dilation=1,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_features),
            ME.MinkowskiReLU(),
        )

    def forward(self, x):
        """Pass a tensor through the ASPP block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        cat = []
        for i, layer in enumerate(self.net):
            x_i = layer(x)
            cat.append(x_i)
        x_global = self.pool(x)
        x_global = self.unpool(x, x_global)
        cat.append(x_global)
        out = ME.cat(cat)
        return self.out(out)


class CascadeDilationBlock(ME.MinkowskiNetwork):
    """Cascaded Atrous Convolution Block."""

    def __init__(
        self,
        in_features,
        out_features,
        dimension=3,
        depth=6,
        dilations=[1, 2, 4, 8, 16, 32],
        activation="relu",
    ):
        """Initialize the Cascaded Atrous Convolution block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        dimension : int, default 3
            Dimension of the input image
        depth : int, default 1
            Number of atrous convolutions layers
        dilations : list, default [1, 2, 4, 8, 16, 32]
            Dilation rates for atrous convolutions
        activation : union[str, dict], default 'relu'
            activation function configuration
        """
        # Initialize parent class
        super().__init__(dimension)

        F = out_features
        m = []
        self.input_layer = ME.MinkowskiLinear(in_features, F)
        for i in range(depth):
            m.append(ResNetBlock(F, F, dilation=dilations[i], activation=activation))
        self.net = nn.Sequential(*m)

    def forward(self, x):
        """Pass a tensor through the Cascaded Atrous Convolution block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        x = self.input_layer(x)
        sumTensor = x
        for i, layer in enumerate(self.net):
            x = layer(x)
            sumTensor += x

        return sumTensor


class MBConv(ME.MinkowskiNetwork):
    """MBConv block."""

    def __init__(
        self,
        in_features,
        out_features,
        expand_ratio=2,
        dimension=3,
        dilation=1,
        kernel_size=3,
        stride=1,
        activation="relu",
        normalization="batch_norm",
        bias=False,
    ):
        """Initialize the MBConv block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        expand_ratio : int, default 2
            Multiplicative factor to apply to the input number of features
        dimension : int, default 3
            Dimension of the input image
        dilation : int, default 1
            Convolution kernel dilation
        kernel_size : int, default 3
            Convolution kernel size
        stride : int, default 1
            Convolution kernel stride
        activation : union[str, dict], default 'relu'
            activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            normalization function configuration
        bias : bool, default False
            Whether to add a bias term to the kernel
        """
        # Initialize the parent class
        super().__init__(dimension)

        self.D = dimension
        self.hidden_dim = int(expand_ratio * in_features)

        if expand_ratio == 1:
            self.m = nn.Sequential(
                norm_factory(normalization, in_features),
                act_factory(activation),
                ME.MinkowskiConvolution(
                    in_features,
                    out_features,
                    kernel_size=3,
                    stride=1,
                    dilation=1,
                    dimension=self.D,
                    bias=bias,
                ),
            )
        else:
            self.m = nn.Sequential(
                norm_factory(normalization, in_features),
                act_factory(activation),
                ME.MinkowskiLinear(in_features, self.hidden_dim),
                norm_factory(normalization, self.hidden_dim),
                act_factory(activation),
                ME.MinkowskiChannelwiseConvolution(
                    self.hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    dimension=self.D,
                ),
                norm_factory(normalization, self.hidden_dim),
                act_factory(activation),
                ME.MinkowskiLinear(self.hidden_dim, out_features),
            )

    def forward(self, x):
        """Pass a tensor through the MBConv block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        out = self.m(x)

        return out


class MBResConv(ME.MinkowskiNetwork):
    """MBResConv block."""

    def __init__(
        self,
        in_features,
        out_features,
        expand_ratio=2,
        dimension=3,
        dilation=1,
        kernel_size=3,
        stride=1,
        activation="relu",
        normalization="batch_norm",
        bias=False,
    ):
        """Initialize the MBResConv block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        expand_ratio : int, default 2
            Multiplicative factor to apply to the input number of features
        dimension : int, default 3
            Dimension of the input image
        dilation : int, default 1
            Convolution kernel dilation
        kernel_size : int, default 3
            Convolution kernel size
        stride : int, default 1
            Convolution kernel stride
        activation : union[str, dict], default 'relu'
            activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            normalization function configuration
        bias : bool, default False
            Whether to add a bias term to the kernel
        """
        # Initialize the parent class
        super().__init__(dimension)

        self.D = dimension
        self.m1 = MBConv(
            in_features,
            out_features,
            expand_ratio=expand_ratio,
            dimension=dimension,
            dilation=dilation,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            normalization=normalization,
            bias=bias,
        )
        self.m2 = MBConv(
            out_features,
            out_features,
            expand_ratio=expand_ratio,
            dimension=dimension,
            dilation=dilation,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            normalization=normalization,
            bias=bias,
        )
        if in_features == out_features:
            self.connection = nn.Identity()
        else:
            self.connection = nn.Sequential(
                norm_factory(normalization, in_features),
                act_factory(activation),
                ME.MinkowskiLinear(in_features, out_features),
            )

    def forward(self, x):
        """Pass a tensor through the MBResConv block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        x_add = self.connection(x)
        x = self.m1(x)
        x = self.m2(x)
        out = x_add + x

        return out


class SEBlock(ME.MinkowskiNetwork):
    """Squeeze and Excitation block."""

    def __init__(self, channels, ratio=8, dimension=3):
        """Initialize the SE block.

        Parameters
        ----------
        channel : int
            Number of input features
        ratio : int, default 8
            Squeezing ratio
        dimension : int, default 3
            Dimension of the input image
        """
        # Initialize the parent class
        super().__init__(dimension)

        assert channels // ratio > 0
        assert channels % ratio == 0
        self.linear1 = ME.MinkowskiLinear(channels, channels // ratio)
        self.relu = ME.MinkowskiReLU()
        self.linear2 = ME.MinkowskiLinear(channels // ratio, channels)
        self.sigmoid = ME.MinkowskiSigmoid()
        self.pool = ME.MinkowskiGlobalPooling()
        self.bcst = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        """Pass a tensor through the SE block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        g = self.pool(x)
        g = self.linear1(g)
        g = self.relu(g)
        g = self.linear2(g)
        g = self.sigmoid(g)
        out = self.bcst(x, g)

        return x


class SEResNetBlock(ME.MinkowskiNetwork):
    """Squeeze and Excitation ResNet block."""

    def __init__(
        self,
        in_features,
        out_features,
        se_ratio=8,
        stride=1,
        dilation=1,
        dimension=3,
        activation="relu",
        normalization="batch_norm",
    ):
        """Initialize the SEResNet block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        se_ratio : int, default 8
            Squeezing ratio
        stride : int, default 1
            Convolution kernel stride
        dilation : int, default 1
            Convolution kernel dilation
        dimension : int, default 3
            Dimension of the input image
        activation : union[str, dict], default 'relu'
            activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            normalization function configuration
        """
        # Initialize parent class
        super().__init__(dimension)

        assert dimension > 0

        if in_features != out_features:
            self.residual = ME.MinkowskiLinear(in_features, out_features)
        else:
            self.residual = nn.Identity()

        self.conv1 = ME.MinkowskiConvolution(
            in_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm1 = norm_factory(normalization, out_features)
        self.act_fn1 = act_factory(activation)

        self.conv2 = ME.MinkowskiConvolution(
            out_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm2 = norm_factory(normalization, out_features)
        self.act_fn2 = act_factory(activation)

        self.se_block = SEBlock(out_features, ratio=se_ratio, dimension=dimension)

    def forward(self, x):
        """Pass a tensor through the SEResNet block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        residual = self.residual(x)
        out = self.act_fn1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se_block(out)
        out += residual
        out = self.act_fn2(out)

        return out


class MBResConvSE(MBResConv):
    """MBResConvSE block."""

    def __init__(
        self,
        in_features,
        out_features,
        se_ratio=8,
        expand_ratio=2,
        dimension=3,
        dilation=1,
        kernel_size=3,
        stride=1,
        activation="relu",
        normalization="batch_norm",
        bias=False,
    ):
        """Initialize the MBResConvSE block.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        se_ratio : int, default 8
            Squeezing ratio
        expand_ratio : int, default 2
            Multiplicative factor to apply to the input number of features
        dimension : int, default 3
            Dimension of the input image
        dilation : int, default 1
            Convolution kernel dilation
        kernel_size : int, default 3
            Convolution kernel size
        stride : int, default 1
            Convolution kernel stride
        activation : union[str, dict], default 'relu'
            activation function configuration
        normalization : union[str, dict], default 'batch_norm'
            normalization function configuration
        bias : bool, default False
            Whether to add a bias term to the kernel
        """
        # Initialize the parent class
        super().__init__(
            in_features,
            out_features,
            expand_ratio=expand_ratio,
            dimension=dimension,
            dilation=dilation,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            normalization=normalization,
            bias=bias,
        )

        if in_features == out_features:
            self.connection = nn.Identity()
        else:
            self.connection = nn.Sequential(
                norm_factory(normalization, in_features),
                act_factory(activation),
                ME.MinkowskiLinear(in_features, out_features),
            )

        self.se = SEBlock(out_features, ratio=se_ratio)

    def forward(self, x):
        """Pass a tensor through the SEResNet block.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor

        Returns
        -------
        ME.SparseTensor
            Output sparse tensor
        """
        res = super().forward(x)
        attn = self.se(res)
        out = self.connection(x) + attn

        return out
