"""Function which sets up the necessary configuration for all CNNs."""


def setup_cnn_configuration(
    self,
    reps,
    depth,
    filters,
    input_kernel=3,
    data_dim=3,
    num_input=1,
    allow_bias=False,
    activation="lrelu",
    norm_layer="batch_norm",
    spatial_size=None,
):
    """Base function for global network parameters (CNN-based models).

    This avoids repeating the same base configuration parsing everywhere.
    For example, typical usage would be:

    .. code-block:: python

        class UResNetEncoder(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                setup_cnn_configuration(self, **cfg)

    Parameters
    ----------
    reps : int
        Number of time convolutions are repeated at each depth
    depth : int
        Depth of the CNN (number of downsampling)
    filters : int
        Number of input filters
    input_kernel : int, default 3
        Input kernel size
    data_dim : int, default 3
        Dimension of the input image data
    num_input : int, default 1
        Number of features in the input image
    allow_bias : bool, default False
        Whether to allow biases in the convolution and linear layers
    activation : union[str, dict], default 'relu'
        Activation function configuration
    normalization : union[str, dict], default 'batch_norm'
        Normalization function configuration
    spatial_size : int, optional
        Size of the input image in number of voxels per data_dim. This is only
        necessary when passing the normalized coordinates as features.
    """
    # Store the base parameters
    self.reps = reps
    self.depth = depth
    self.num_filters = filters
    self.input_kernel = input_kernel
    self.dim = data_dim
    self.num_input = num_input
    self.allow_bias = allow_bias
    self.spatial_size = spatial_size

    # Convert the depth to a number of filters per plane
    self.num_planes = [i * self.num_filters for i in range(1, self.depth + 1)]

    # Store activation function configuration
    self.act_cfg = activation

    # Store the normalization function configuration
    self.norm_cfg = norm_layer
