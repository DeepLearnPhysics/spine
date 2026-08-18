"""Reusable sparse convolutional network blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from spine.config.factory import Config
from spine.model import sparse

from .act_norm import act_factory, norm_factory


class ConvolutionBlock(sparse.Network):
    """Apply two sparse convolution-normalization-activation stages.

    The first convolution applies the requested stride and both convolutions
    use the same dilation. This block has no residual connection, so it may
    change both the coordinate resolution and feature width.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        dilation: int = 1,
        dimension: int = 3,
        activation: Config = "relu",
        normalization: Config = "batch_norm",
        bias: bool = False,
    ) -> None:
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

        Raises
        ------
        ValueError
            If ``dimension`` is not positive.
        """
        # Initialize the parent class
        super().__init__(dimension)

        if dimension <= 0:
            raise ValueError("Expected `dimension > 0`.")

        self.conv1 = sparse.Convolution(
            in_features,
            out_features,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
            bias=bias,
        )
        self.norm1 = norm_factory(normalization, out_features)
        self.act_fn1 = act_factory(activation)

        self.conv2 = sparse.Convolution(
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

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the convolution block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act_fn1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act_fn2(out)

        return out


class DropoutBlock(sparse.Network):
    """Apply two sparse convolution-dropout-normalization-activation stages.

    Dropout is applied independently after each convolution. The first
    convolution applies ``stride``; the second operates at the resulting
    resolution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        dilation: int = 1,
        dimension: int = 3,
        p: float = 0.5,
        activation: Config = "relu",
        normalization: Config = "batch_norm",
        bias: bool = False,
    ) -> None:
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

        Raises
        ------
        ValueError
            If ``dimension`` is not positive or ``p`` is outside ``[0, 1)``.
        """
        # Initialize the parent class
        super().__init__(dimension)

        if dimension <= 0:
            raise ValueError("Expected `dimension > 0`.")
        if not 0.0 <= p < 1.0:
            raise ValueError(f"`p` must be in [0, 1), got {p}.")

        self.conv1 = sparse.Convolution(
            in_features,
            out_features,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
            bias=bias,
        )
        self.dropout1 = sparse.Dropout(p=p)
        self.norm1 = norm_factory(normalization, out_features)
        self.act_fn1 = act_factory(activation)

        self.conv2 = sparse.Convolution(
            out_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            dimension=dimension,
            bias=bias,
        )
        self.dropout2 = sparse.Dropout(p=p)
        self.norm2 = norm_factory(normalization, out_features)
        self.act_fn2 = act_factory(activation)

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the dropout block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
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


class ResNetBlock(sparse.Network):
    """Apply a two-convolution pre-activation residual block.

    A linear projection adapts the residual when only the feature width
    changes. When ``stride`` changes the coordinate resolution, a stride-one
    kernel convolution projects and downsamples the residual path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stride: int = 1,
        dilation: int = 1,
        dimension: int = 3,
        activation: Config = "relu",
        normalization: Config = "batch_norm",
        bias: bool = False,
    ) -> None:
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

        Raises
        ------
        ValueError
            If ``dimension`` is not positive.
        """
        # Initialize the parent class
        super().__init__(dimension)

        if dimension <= 0:
            raise ValueError("Expected `dimension > 0`.")

        if stride != 1:
            self.residual = sparse.Convolution(
                in_features,
                out_features,
                kernel_size=1,
                stride=stride,
                dimension=dimension,
                bias=bias,
            )
        elif in_features != out_features:
            self.residual = sparse.Linear(in_features, out_features, bias=bias)
        else:
            self.residual = torch.nn.Identity()

        self.conv1 = sparse.Convolution(
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

        self.conv2 = sparse.Convolution(
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

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the ResNet block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        residual = self.residual(x)

        out = self.conv1(self.act_fn1(self.norm1(x)))
        out = self.conv2(self.act_fn2(self.norm2(out)))

        out += residual

        return out


class AtrousIIBlock(sparse.Network):
    """Apply the two-stage atrous residual block from ACNN.

    The two convolutions use dilation rates one and three, respectively, to
    enlarge the receptive field without reducing spatial resolution.

    References
    ----------
    .. [1] Zhou et al., "ACNN: a Full Resolution DCNN for Medical Image
       Segmentation," 2019. https://arxiv.org/abs/1901.09203
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dimension: int = 3,
        activation: Config = "relu",
        normalization: Config = "batch_norm",
    ) -> None:
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

        Raises
        ------
        ValueError
            If ``dimension`` is not positive.
        """
        # Initialize the parent class
        super().__init__(dimension)

        if dimension <= 0:
            raise ValueError("Expected `dimension > 0`.")

        if in_features != out_features:
            self.residual = sparse.Linear(in_features, out_features)
        else:
            self.residual = torch.nn.Identity()

        self.conv1 = sparse.Convolution(
            in_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=1,
            dimension=self.dimension,
        )
        self.norm1 = norm_factory(normalization, out_features)
        self.act_fn1 = act_factory(activation)

        self.conv2 = sparse.Convolution(
            out_features,
            out_features,
            kernel_size=3,
            stride=1,
            dilation=3,
            dimension=self.dimension,
        )
        self.norm2 = norm_factory(normalization, out_features)
        self.act_fn2 = act_factory(activation)

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the AtrousII block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
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


class ResNeXtBlock(sparse.Network):
    """Apply a grouped multi-path ResNeXt-style residual block.

    Each cardinal path first projects the full input to a fraction of the
    output width, then applies ``depth`` sparse convolutions. Path outputs are
    concatenated, projected, and added to a residual connection. Dilation and
    kernel size may vary by path, but all paths must use the same stride so
    their coordinate maps remain compatible.

    Notes
    -----
    Set ``dilations=1`` to recover a conventional ResNeXt-style block.

    References
    ----------
    .. [1] Xie et al., "Aggregated Residual Transformations for Deep Neural
       Networks," 2017. https://arxiv.org/abs/1611.05431
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dimension: int = 3,
        cardinality: int = 4,
        depth: int = 1,
        dilations: int | Sequence[int] | None = None,
        kernel_sizes: int | Sequence[int] = 3,
        strides: int | Sequence[int] = 1,
        activation: Config = "relu",
        normalization: Config = "batch_norm",
    ) -> None:
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

        Raises
        ------
        ValueError
            If dimensions are invalid, feature widths are not divisible by
            ``cardinality``, per-path sequences have the wrong length, or path
            strides disagree.
        """
        # Initialize the parent class
        super().__init__(dimension)

        if dimension <= 0:
            raise ValueError("Expected `dimension > 0`.")
        if cardinality <= 0:
            raise ValueError("Expected `cardinality > 0`.")
        if in_features % cardinality != 0 or out_features % cardinality != 0:
            raise ValueError(
                "Expected both input and output feature counts to be "
                "divisible by `cardinality`."
            )

        path_input_features = in_features // cardinality
        path_output_features = out_features // cardinality

        self.dilations = []
        if dilations is None:
            self.dilations = [3**path_index for path_index in range(cardinality)]
        elif isinstance(dilations, int):
            self.dilations = [dilations for _ in range(cardinality)]
        elif isinstance(dilations, Sequence):
            if len(dilations) != cardinality:
                raise ValueError("Expected `len(dilations) == cardinality`.")
            self.dilations = dilations
        else:
            raise ValueError("Invalid type for input strides, must be int or list!")

        self.kernel_sizes = []
        if isinstance(kernel_sizes, int):
            self.kernel_sizes = [kernel_sizes for _ in range(cardinality)]
        elif isinstance(kernel_sizes, Sequence):
            if len(kernel_sizes) != cardinality:
                raise ValueError("Expected `len(kernel_sizes) == cardinality`.")
            self.kernel_sizes = kernel_sizes
        else:
            raise ValueError("Invalid type for input strides, must be int or list!")

        self.strides = []
        if isinstance(strides, int):
            self.strides = [strides for _ in range(cardinality)]
        elif isinstance(strides, Sequence):
            if len(strides) != cardinality:
                raise ValueError("Expected `len(strides) == cardinality`.")
            self.strides = strides
        else:
            raise ValueError("Invalid type for input strides, must be int or list!")
        if len(set(self.strides)) != 1:
            raise ValueError(
                "All ResNeXt paths must use the same stride to concatenate."
            )

        # For each path, generate sequentials
        self.paths = []
        for path_index in range(cardinality):
            path_layers = [sparse.Linear(in_features, path_input_features)]
            for layer_index in range(depth):
                input_features = (
                    path_input_features if layer_index == 0 else path_output_features
                )
                path_layers.append(
                    sparse.Convolution(
                        in_channels=input_features,
                        out_channels=path_output_features,
                        kernel_size=self.kernel_sizes[path_index],
                        stride=(self.strides[path_index] if layer_index == 0 else 1),
                        dilation=self.dilations[path_index],
                        dimension=self.dimension,
                    )
                )
                path_layers.append(norm_factory(normalization, path_output_features))
                path_layers.append(act_factory(activation))
            self.paths.append(torch.nn.Sequential(*path_layers))
        self.paths = torch.nn.Sequential(*self.paths)
        self.linear = sparse.Linear(out_features, out_features)

        # Skip connection
        residual_stride = self.strides[0]
        if residual_stride != 1:
            self.residual = sparse.Convolution(
                in_features,
                out_features,
                kernel_size=1,
                stride=residual_stride,
                dimension=dimension,
            )
        elif in_features != out_features:
            self.residual = sparse.Linear(in_features, out_features)
        else:
            self.residual = torch.nn.Identity()

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the ResNeXt block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        residual = self.residual(x)

        path_outputs = tuple(layer(x) for layer in self.paths)
        out = sparse.cat(path_outputs)
        out = self.linear(out)
        out += residual

        return out


class SPP(sparse.Network):
    """Aggregate local and global context with spatial pyramid pooling.

    A global pooled branch is broadcast to every active coordinate. Optional
    local pooling branches use the requested kernels and dilations, are
    unpooled to the input coordinate map, and are concatenated before a final
    linear projection.

    Notes
    -----
    With no ``kernel_sizes``, the block contains only the global context
    branch, matching the pooling strategy used by ParseNet.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_sizes: Sequence[int] | None = None,
        dilations: int | Sequence[int] | None = None,
        mode: str = "avg",
        dimension: int = 3,
    ) -> None:
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

        Raises
        ------
        ValueError
            If ``mode`` is unknown or ``kernel_sizes`` and ``dilations`` have
            inconsistent lengths.
        """
        # Initialize the parent class
        super().__init__(dimension)

        if mode == "avg":
            self.pool_fn = sparse.AvgPooling
        elif mode == "max":
            self.pool_fn = sparse.MaxPooling
        elif mode == "sum":
            self.pool_fn = sparse.SumPooling
        else:
            raise ValueError("Invalid pooling mode, must be one of \
                'sum', 'max' or 'average'")

        self.unpool_fn = sparse.PoolingTranspose

        # Include global pooling as first modules.
        self.pool = [sparse.GlobalPooling()]
        self.unpool = [sparse.Broadcast()]
        multiplier = 1

        # Define subregion poolings
        if kernel_sizes is not None:
            if dilations is None:
                dilations = [1] * len(kernel_sizes)
            elif isinstance(dilations, int):
                dilations = [dilations for _ in range(len(kernel_sizes))]
            elif isinstance(dilations, Sequence):
                if len(kernel_sizes) != len(dilations):
                    raise ValueError("Expected `len(kernel_sizes) == len(dilations)`.")
            else:
                raise ValueError(
                    "Invalid input to dilations, must be either int or list of ints."
                )

            multiplier = len(kernel_sizes) + 1  # Additional 1 for globalPool
            for kernel_size, dilation in zip(
                kernel_sizes,
                dilations,
                strict=True,
            ):
                pooling_layer = self.pool_fn(
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=kernel_size,
                    dimension=dimension,
                )
                unpooling_layer = self.unpool_fn(
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=kernel_size,
                    dimension=dimension,
                )
                self.pool.append(pooling_layer)
                self.unpool.append(unpooling_layer)
        self.pool = torch.nn.Sequential(*self.pool)
        self.unpool = torch.nn.Sequential(*self.unpool)
        self.linear = sparse.Linear(in_features * multiplier, out_features)

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the SPP block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        pooled_outputs = []
        for branch_index, pool in enumerate(self.pool):
            pooled = pool(x)
            # First item is Global Pooling
            if branch_index == 0:
                pooled = self.unpool[branch_index](x, pooled)
            else:
                pooled = self.unpool[branch_index](pooled)
            pooled_outputs.append(pooled)
        out = sparse.cat(pooled_outputs)
        out = self.linear(out)

        return out


class ASPP(sparse.Network):
    """Aggregate multi-scale context with atrous spatial pyramid pooling.

    The block combines a pointwise projection, parallel dilated convolutions,
    and a projected global-context branch. Their features are concatenated and
    fused with a final sparse convolution.

    References
    ----------
    .. [1] Chen et al., "Rethinking Atrous Convolution for Semantic Image
       Segmentation," 2017. https://arxiv.org/abs/1706.05587
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dimension: int = 3,
        width: int = 5,
        dilations: Sequence[int] | None = None,
    ) -> None:
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

        Raises
        ------
        ValueError
            If the number of dilation rates does not equal ``width``.
        """
        # Initialize parent class
        super().__init__(dimension)

        if dilations is None:
            dilations = (2, 4, 6, 8, 12)
        if len(dilations) != width:
            raise ValueError("Expected `len(dilations) == width`.")

        branches = [sparse.Linear(in_features, out_features)]
        for dilation in dilations:
            branches.append(
                sparse.Convolution(
                    in_features,
                    out_features,
                    kernel_size=3,
                    dilation=dilation,
                    dimension=self.dimension,
                )
            )
        self.branches = torch.nn.Sequential(*branches)
        self.pool = sparse.GlobalPooling()
        self.global_linear = sparse.Linear(in_features, out_features)
        self.unpool = sparse.Broadcast()
        self.output_layer = torch.nn.Sequential(
            sparse.Convolution(
                out_features * (2 + width),
                out_features,
                kernel_size=3,
                dilation=1,
                dimension=self.dimension,
            ),
            sparse.BatchNorm(out_features),
            sparse.ReLU(),
        )

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the ASPP block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        branch_outputs = []
        for layer in self.branches:
            branch_outputs.append(layer(x))
        global_features = self.global_linear(self.pool(x))
        global_features = self.unpool(x, global_features)
        branch_outputs.append(global_features)
        out = sparse.cat(branch_outputs)
        return self.output_layer(out)


class CascadeDilationBlock(sparse.Network):
    """Accumulate features from a cascade of dilated residual blocks.

    The input is first projected to ``out_features``. Each residual block uses
    one dilation rate, consumes the preceding block's output, and contributes
    additively to the returned feature tensor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dimension: int = 3,
        depth: int = 6,
        dilations: Sequence[int] | None = None,
        activation: Config = "relu",
    ) -> None:
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

        Raises
        ------
        ValueError
            If the number of dilation rates does not equal ``depth``.
        """
        # Initialize parent class
        super().__init__(dimension)

        if dilations is None:
            dilations = (1, 2, 4, 8, 16, 32)
        if len(dilations) != depth:
            raise ValueError("Expected `len(dilations) == depth`.")

        num_features = out_features
        blocks = []
        self.input_layer = sparse.Linear(in_features, num_features)
        for layer_index in range(depth):
            blocks.append(
                ResNetBlock(
                    num_features,
                    num_features,
                    dimension=dimension,
                    dilation=dilations[layer_index],
                    activation=activation,
                )
            )
        self.blocks = torch.nn.Sequential(*blocks)

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the Cascaded Atrous Convolution block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        x = self.input_layer(x)
        summed = x
        for layer in self.blocks:
            x = layer(x)
            summed = summed + x

        return summed


class MBConv(sparse.Network):
    """Apply a sparse inverted-bottleneck mobile convolution.

    For expansion ratios greater than one, the block expands channels with a
    linear projection, applies a channel-wise sparse convolution, and projects
    to ``out_features``. An expansion ratio of one uses a direct sparse
    convolution.

    References
    ----------
    .. [1] Sandler et al., "MobileNetV2: Inverted Residuals and Linear
       Bottlenecks," 2018. https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        expand_ratio: int = 2,
        dimension: int = 3,
        dilation: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        activation: Config = "relu",
        normalization: Config = "batch_norm",
        bias: bool = False,
    ) -> None:
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

        Raises
        ------
        ValueError
            If ``expand_ratio`` is not positive.
        """
        # Initialize the parent class
        super().__init__(dimension)

        if expand_ratio < 1:
            raise ValueError(f"`expand_ratio` must be positive, got {expand_ratio}.")
        self.hidden_dim = int(expand_ratio * in_features)

        if expand_ratio == 1:
            self.layers = torch.nn.Sequential(
                norm_factory(normalization, in_features),
                act_factory(activation),
                sparse.Convolution(
                    in_features,
                    out_features,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dimension=self.dimension,
                    bias=bias,
                ),
            )
        else:
            self.layers = torch.nn.Sequential(
                norm_factory(normalization, in_features),
                act_factory(activation),
                sparse.Linear(in_features, self.hidden_dim),
                norm_factory(normalization, self.hidden_dim),
                act_factory(activation),
                sparse.ChannelwiseConvolution(
                    self.hidden_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    dimension=self.dimension,
                ),
                norm_factory(normalization, self.hidden_dim),
                act_factory(activation),
                sparse.Linear(self.hidden_dim, out_features),
            )

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the MBConv block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        out = self.layers(x)

        return out


class MBResConv(sparse.Network):
    """Apply two mobile convolutions with a residual connection.

    The first mobile convolution applies ``stride`` and the second remains at
    the resulting resolution. A projected residual path handles changes in
    feature width or coordinate stride.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        expand_ratio: int = 2,
        dimension: int = 3,
        dilation: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        activation: Config = "relu",
        normalization: Config = "batch_norm",
        bias: bool = False,
    ) -> None:
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

        self.first_block = MBConv(
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
        self.second_block = MBConv(
            out_features,
            out_features,
            expand_ratio=expand_ratio,
            dimension=dimension,
            dilation=dilation,
            kernel_size=kernel_size,
            stride=1,
            activation=activation,
            normalization=normalization,
            bias=bias,
        )
        if stride != 1:
            self.connection = torch.nn.Sequential(
                norm_factory(normalization, in_features),
                act_factory(activation),
                sparse.Convolution(
                    in_features,
                    out_features,
                    kernel_size=1,
                    stride=stride,
                    dimension=dimension,
                    bias=bias,
                ),
            )
        elif in_features == out_features:
            self.connection = torch.nn.Identity()
        else:
            self.connection = torch.nn.Sequential(
                norm_factory(normalization, in_features),
                act_factory(activation),
                sparse.Linear(in_features, out_features),
            )

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the MBResConv block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        residual = self.connection(x)
        x = self.first_block(x)
        x = self.second_block(x)
        out = residual + x

        return out


class SEBlock(sparse.Network):
    """Reweight sparse feature channels using squeeze-and-excitation.

    Global pooling produces one descriptor per batch entry. A two-layer
    bottleneck predicts sigmoid channel weights which are broadcast and
    multiplied into every active sparse site.

    References
    ----------
    .. [1] Hu et al., "Squeeze-and-Excitation Networks," 2018.
       https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        channels: int,
        ratio: int = 8,
        dimension: int = 3,
    ) -> None:
        """Initialize the SE block.

        Parameters
        ----------
        channels : int
            Number of input features
        ratio : int, default 8
            Squeezing ratio
        dimension : int, default 3
            Dimension of the input image

        Raises
        ------
        ValueError
            If ``ratio`` is not positive.
        """
        # Initialize the parent class
        super().__init__(dimension)

        if ratio < 1:
            raise ValueError(f"`ratio` must be positive, got {ratio}.")
        hidden_channels = max(1, channels // ratio)
        self.linear1 = sparse.Linear(channels, hidden_channels)
        self.relu = sparse.ReLU()
        self.linear2 = sparse.Linear(hidden_channels, channels)
        self.sigmoid = sparse.Sigmoid()
        self.pool = sparse.GlobalPooling()
        self.broadcast = sparse.BroadcastMultiplication()

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the SE block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        gate = self.pool(x)
        gate = self.linear1(gate)
        gate = self.relu(gate)
        gate = self.linear2(gate)
        gate = self.sigmoid(gate)
        out = self.broadcast(x, gate)

        return out


class SEResNetBlock(sparse.Network):
    """Apply a residual convolution block with channel-wise SE attention.

    Two sparse convolutions produce the residual update. Squeeze-and-excitation
    reweights that update before it is added to the projected identity path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        se_ratio: int = 8,
        stride: int = 1,
        dilation: int = 1,
        dimension: int = 3,
        activation: Config = "relu",
        normalization: Config = "batch_norm",
    ) -> None:
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

        Raises
        ------
        ValueError
            If ``dimension`` is not positive or ``se_ratio`` is invalid for
            the output feature width.
        """
        # Initialize parent class
        super().__init__(dimension)

        if dimension <= 0:
            raise ValueError("Expected `dimension > 0`.")

        if stride != 1:
            self.residual = sparse.Convolution(
                in_features,
                out_features,
                kernel_size=1,
                stride=stride,
                dimension=dimension,
            )
        elif in_features != out_features:
            self.residual = sparse.Linear(in_features, out_features)
        else:
            self.residual = torch.nn.Identity()

        self.conv1 = sparse.Convolution(
            in_features,
            out_features,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            dimension=dimension,
        )
        self.norm1 = norm_factory(normalization, out_features)
        self.act_fn1 = act_factory(activation)

        self.conv2 = sparse.Convolution(
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

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a tensor through the SEResNet block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
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
    """Apply a mobile residual block with squeeze-and-excitation.

    This variant inserts SE attention after the two mobile convolutions and
    before the residual addition.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        se_ratio: int = 8,
        expand_ratio: int = 2,
        dimension: int = 3,
        dilation: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        activation: Config = "relu",
        normalization: Config = "batch_norm",
        bias: bool = False,
    ) -> None:
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

        self.squeeze_excitation = SEBlock(
            out_features,
            ratio=se_ratio,
            dimension=dimension,
        )

    def forward(self, x: sparse.SparseTensor) -> sparse.SparseTensor:
        """Pass a sparse tensor through the mobile SE residual block.

        Parameters
        ----------
        x : sparse.SparseTensor
            Input sparse tensor

        Returns
        -------
        sparse.SparseTensor
            Output sparse tensor
        """
        residual = self.connection(x)
        out = self.second_block(self.first_block(x))
        out = residual + self.squeeze_excitation(out)

        return out
