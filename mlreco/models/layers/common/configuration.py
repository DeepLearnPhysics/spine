def setup_cnn_configuration(self, spatial_size, reps, depth, filters,
        input_kernel = 3, data_dim = 3, num_input = 1, allow_bias = False,
        activation = {'name': 'lrelu'}, norm_layer = {'name': 'batch_norm'}):
    '''
    Base function for global network parameters (CNN-based models).
    This avoids repeating everywhere the same base configuration.
    For example, typical usage would be:

    .. code-block:: python

        class UResNetEncoder(torch.nn.Module):
            def __init__(self, cfg):
                super().__init__()
                setup_cnn_configuration(self, **cfg)

    Defines the following default configuration:

    Parameters
    ----------
    spatial_size: int
        Size of the input image in number of voxels per    data_dim
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
    activation : dict, default {'name': 'lrelu'}
        Activation function parameters
    norm_layer : dict, default {'name': 'batch_norm'}
        Normalization function parameters
    '''
    # Store the base parameters
    self.spatial_size = spatial_size
    self.reps = reps
    self.depth = depth
    self.num_filters = filters
    self.input_kernel = input_kernel
    self.dim = data_dim
    self.num_input = num_input
    self.allow_bias = allow_bias

    # Convert the depth to a number of filters per plane
    self.nPlanes = [i * self.num_filters for i in range(1, self.depth + 1)]

    # Store activation function parameters
    setup_activation(self, **activation)

    # Store normalization function parameters
    setup_norm(self, **norm_layer)

def setup_activation(self, name, args={}):
    '''
    Setup the activation function name and parameters

    Parameters
    ----------
    name : str
        Name of the activation function under activation_normalization_factories
    args : dict, optional
        Keyword arguments to pass to the activation function
    '''
    self.activation_name = name
    self.activation_args = args

def setup_norm(self, name, args={}):
    '''
    Setup the normalization function name and parameters

    Parameters
    ----------
    name : str
        Name of the norm function under activation_normalization_factories
    args : dict, optional
        Keyword arguments to pass to the norm function
    '''
    self.norm = name
    self.norm_args = args
