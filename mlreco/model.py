"""Centralize all methods associated with a machine-learning model."""

from copy import deepcopy

from .models import model_factory

from .utils.logger import logger


class Model:
    """Groups all relevant functions to construct a model and its loss."""

    def __init__(self, name, modules, model_path=None, train=False,
                 distributed=False):
        """Process the model configuration.

        Parameters
        ----------
        name : str
            Name of the model as specified under mlreco.models.factories
        modules : dict
            Dictionary of modules that make up the model
        model_path : str, optional
            Path to global model weights to load
        train : bool, default False
            If True, enable autograd
        distributed : bool, default False
            Whether the model is part of a distributed training process
        """
        # Save parameters
        self.train = train
        self.distributed = distributed

        # Deepcopy the model configuration, remove the weight loading/freezing
        self.model_name = name
        self.model_cfg = deepcopy(modules)
        modules = self.clean_config(modules)

        # Initialize the model network and loss functions
        network_cls, loss_cls = model_factory(name)
        try:
            self.net = self.net_cls(**modules)
        except Exception as err:
            msg = f"Failed to instantiate {network_cls}"
            raise type(err)(f"{err}\n{msg}")

        try:
            self.loss_fn = self.loss_class(**modules)
        except Exception as err:
            msg = f"Failed to instantiate {loss_cls}"
            raise type(err)(f"{err}\n{msg}")

        # If requested, freeze some/all the model weights
        self.freeze_weights()

        # If requested, load the some/all the model weights
        self.load_model(model_path)

    def __call__(self, data):
        """Calls the forward function on the provided batch of data.

        Parameters
        ----------
        data : dict
            Dictionary of input data product keys which each map to its
            associated batched data product

        Returns
        -------
        dict
            Dictionary of model and loss outputs
        """
        return self.forward(data)

    def clean_config(self, config):
        """Remove model loading/freezing keys from all level of a dictionary.

        This is used to remove the weight loading/freezing from the input
        configuration before it is fed to the model/loss classes.

        Parameters
        ----------
        config : dict
            Dictionary to remove the keys from
        """
        keys = ['model_path', 'model_name', 'freeze_weights']
        if isinstance(config, dict):
            for k in keys:
                if k in config:
                    del config[k]
            for val in config.values():
                clean_config(val)

    def freeze_weights(self):
        """Freeze the weights of certain model components.

        Breadth-first search for `freeze_weights` parameters in the model
        configuration. If `freeze_weights` is `True` under a module block,
        `requires_grad` is set to `False` for its parameters. The batch
        normalization and dropout layers are set to evaluation mode.
        """
        # Loop over all the module blocks in the model configuration
        module_items = list(self.model_cfg.items())
        while len(module_items) > 0:
            # Get the module name and its configuration block
            module, config = module_items.pop()

            # If the module is to be frozen, apply
            if config.get('freeze_weights', False):
                # Fetch the module name to be found in the state dictionary
                model_name = config.get('model_name', module)

                # Set BN and DO layers to evaluation mode
                getattr(self.net, module).eval()

                # Freeze all the weights of this module
                count = 0
                for name, param in self.net.named_parameters():
                    if module in name:
                        key = name.replace(f'.{module}.', f'.{model_name}.')
                        if key in self.net.state_dict().keys():
                            param.requires_grad = False
                            count += 1

                # Throw if no weights were found to freeze
                assert count, (
                        f"Could not find any weights to freeze for {module}")

                logger.info("Froze %d weights in module %s", count, module)

            # Keep the BFS going by adding the nested blocks
            for key in config:
                if isinstance(config[key], dict):
                    module_items.append((key, config[key]))

    def load_weights(self, full_model_path):
        """Load the weights of certain model components.

        Breadth-first search for `model_path` parameters in the model
        configuration. If 'model_path' is found under a module block,
        the weights are loaded for its parameters.

        If a `model_path` is not found for a given module, load the overall
        weights from `model_path` under `trainval` for that module instead.

        Parameters
        ----------
        full_model_path : str
            Path to the weights for the full model
        """
        # If a general model path is provided, add it to the loading list first
        model_paths = []
        if full_model_path:
            model_paths = [(self.model_name, full_model_path, '')]

        # Find the list of sub-module weights to subsequently load
        module_items = list(self.model_cfg.items())
        while len(module_items) > 0:
            module, config = module_items.pop()
            if config.get('model_path', '') != '':
                model_name = config.get('model_name', module)
                model_paths.append((module, config['model_path'], model_name))
            for key in config:
                if isinstance(config[key], dict):
                    module_items.append((key, config[key]))

        # If no pre-trained weights are requested, nothing to do here
        self.start_iteration = 0
        if not model_paths:
            return

        # Loop over provided model paths
        for module, model_path, model_name in model_paths:
            # Check that the requested weight file can be found. If the path
            # points at > 1 file, skip for now (loaded in a loop later)
            if not os.path.isfile(model_path):
                if not self.train and glob.glob(model_path):
                    continue

                raise ValueError("Weight file not found for module "
                                f"{module}: {model_path}")

            # Load weight file into existing model
            logger.info("Restoring weights for module %s "
                        "from %s...", module, model_path)
            with open(model_path, 'rb') as f:
                # Read checkpoint. If loading weights to a non-distributed
                # model, remove leading keyword `module` from weight names.
                checkpoint = torch.load(f, map_location='cpu')
                state_dict = checkpoint['state_dict']
                if not self.distributed:
                    state_dict = {k.replace('module.', ''):v \
                            for k, v in state_dict.items()}

                # Check that all the needed weights are provided
                missing_keys = []
                if module == self.model_name:
                    for name in self.net.state_dict():
                        if not name in state_dict.keys():
                            missing_keys.append((name, name))
                else:
                    # Update the key names according to the name used to store
                    state_dict = {}
                    for name in self.net.state_dict():
                        if module in name:
                            key = name.replace(
                                    f'.{module}.', f'.{model_name}.')
                            if key in checkpoint['state_dict'].keys():
                                state_dict[name] = checkpoint['state_dict'][key]
                            else:
                                missing_keys.append((name, key))

                # If some necessary keys were not found, throw
                if missing_keys:
                    logger.critical(
                            "These necessary parameters could not be found:")
                    for name, key in missing_keys:
                        logger.critical(
                                "Parameter %s is missing for %s.", key, name)
                    raise ValueError("To be loaded, a set of weights "
                                     "must provide all necessary parameters.")

                # Load checkpoint. Check that all weights are used
                bad_keys = self.net.load_state_dict(state_dict, strict=False)
                if len(bad_keys.unexpected_keys) > 0:
                    logger.warning(
                            "This weight file contains parameters that could "
                            "not be loaded, indicating that the weight file "
                            "contains more than needed. This might be ok.")
                    logger.warning(
                            'Unexpected keys: %s', bad_keys.unexpected_keys)

                # Load the optimizer state from the main weight file only
                if (self.train and
                    module == self.model_name and
                    self.restore_optimizer):
                    self.optimizer.load_state_dict(checkpoint['optimizer'])

                # Get the latest iteration from the main weight file only
                if module == self.model_name:
                    self.start_iteration = checkpoint['global_step'] + 1

            logger.info('Done.')

    def prepare_data(self, data):
        """Fetches the necessary data products to form the input to the forward
        function and the input to the loss function.

        Parameters
        ----------
        data : dict
            Dictionary of input data product keys which each map to its
            associated batched data product

        Returns
        -------
        input_dict : dict
            Input to the forward pass of the model
        loss_dict : dict
            Labels to be used in the loss computation
        """
        # Fetch the requested data products
        device = self.rank if self.gpus else None
        input_dict, loss_dict = {}, {}
        with torch.set_grad_enabled(self.train):
            # Load the data products for the model forward
            input_dict = {}
            for param, name in self.input_dict.items():
                assert name in data, (
                        f"Must provide {name} in the dataloader schema to "
                         "input into the model forward")

                value = data[name]
                if isinstance(value, TensorBatch):
                    value = data[name].to_tensor(torch.float, device)
                input_dict[param] = value

            # Load the data products for the loss function
            loss_dict = {}
            if self.loss_dict is not None:
                for param, name in self.loss_dict.items():
                    assert name in data, (
                            f"Must provide {name} in the dataloader schema to "
                             "input into the loss function")

                    value = data[name]
                    if isinstance(value, TensorBatch):
                        value = data[name].to_tensor(torch.float, device)
                    loss_dict[param] = value

        return input_dict, loss_dict

    def forward(self, data, iteration=None):
        """Pass one minibatch of data through the network and the loss.

        Load one minibatch of data. pass it through the network forward
        function and the loss computation. Store the output.

        Parameters
        ----------
        data : dict
            Dictionary of input data product keys which each map to its
            associated batched data product

        Returns
        -------
        dict
            Dictionary of model and loss outputs
        """
        # Prepare the input to the forward and loss functions
        input_dict, loss_dict = self.prepare_data(data)

        # If in train mode, record the gradients for backward step
        with torch.set_grad_enabled(self.train):

            # Apply the model forward
            result = self.model(**input_dict)

            # Compute the loss if one is specified, append results
            self.loss = 0.
            if self.loss_dict:
                if not self.time_dependant:
                    result.update(self.loss_fn(**loss_dict, **result))
                else:
                    result.update(self.loss_fn(
                        iteration=iteration, **loss_dict, **result))

                if self.train:
                    self.loss = result['loss']

            # Filter and cast the output to numpy, if requested
            for key, value in result.items():
                # Skip keys that are not to be output
                if ((self.output_keys and key not in self.output_keys) or
                    (self.ignore_keys and key in self.ignore_keys)):
                    del result[key]

                # Convert to numpy, if requested
                if self.to_numpy:
                    if np.isscalar(value):
                        # Scalar
                        result[key] = value
                    elif (isinstance(value, torch.Tensor) and
                          value.numel() == 1):
                        # Scalar tensor
                        result[key] = value.item()
                    elif isinstance(
                            value, (TensorBatch, IndexBatch, EdgeIndexBatch)):
                        # Batch of data
                        result[key] = value.to_numpy()
                    elif (isinstance(value, list) and
                          len(value) and
                          isinstance(value[0], TensorBatch)):
                        # List of tensor batches
                        result[key] = [v.to_numpy() for v in value]
                    else:
                        raise ValueError(f"Cannot cast output {key} to numpy")

            return result
