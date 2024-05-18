"""Centralize all methods associated with a machine-learning model."""

import os
import glob
from copy import deepcopy

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .models import model_factory
from .models.experimental.bayes.calibration import (
        calibrator_factory, calibrator_loss_factory)

from .data import TensorBatch, IndexBatch, EdgeIndexBatch
from .utils.logger import logger


class Model:
    """Groups all relevant functions to construct a model and its loss."""

    def __init__(self, name, modules, network_input, model_path=None,
                 loss_input=None, calibration=None, train=False,
                 to_numpy=False, time_dependent=False, dtype=torch.float,
                 distributed=False, rank=None, detect_anomaly=False,
                 find_unused_parameters=False):
        """Process the model configuration.

        Parameters
        ----------
        name : str
            Name of the model as specified under mlreco.models.factories
        modules : dict
            Dictionary of modules that make up the model
        network_input : List[str]
            List of keys of parsed objects to input into the model forward
        model_path : str, optional
            Path to global model weights to load
        loss_input : List[str], optional
            List of keys of parsed objects to input into the loss forward
        calibration : dict, optional
            Model score calibration configuration
        to_numpy : int, default False
            Cast model output to numpy ndarray
        time_dependant : bool, default False
            Handles time-dependant loss, such as KL divergence annealing
        train : bool, default False
            If True, enable autograd
        dtype : torch.dtype
            Data type of the model parameters and input data 
        distributed : bool, default False
            Whether the model is part of a distributed training process
        rank : int, optional
            Process rank in a torch distributed process
        detect_anomaly : bool, default False
            Whether to attempt to detect a torch anomaly
        find_unused_parameters : bool, default False
            Attempts to detect unused model parameters in the forward pass
        """
        # Save parameters
        self.train = train
        self.to_numpy = to_numpy
        self.time_dependant = time_dependent
        self.dtype = dtype
        self.distributed = distributed
        self.rank = rank

        # If anomaly detection is requested, set it
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True, check_nan=True)

        # Deepcopy the model configuration, remove the weight loading/freezing
        self.model_name = name
        self.model_cfg = deepcopy(modules)
        self.clean_config(modules)

        # Initialize the model network and loss functions
        net_cls, loss_cls = model_factory(name)
        try:
            self.net = net_cls(**modules)
            print('RANK', rank)
            self.net.to(device=rank, dtype=dtype)
        except Exception as err:
            msg = f"Failed to instantiate {net_cls}"
            raise type(err)(f"{err}\n{msg}")

        try:
            self.loss_fn = loss_cls(**modules)
            self.loss_fn.to(device=rank, dtype=dtype)
        except Exception as err:
            msg = f"Failed to instantiate {loss_cls}"
            raise type(err)(f"{err}\n{msg}")

        # If the execution is distributed, wrap with DDP
        if self.distributed:
            self.model = DDP(
                    self.model, device_ids=[rank], output_device=self.rank,
                    find_unused_parameters=find_unused_parameters)

        # Put the model in evaluation mode if requested
        if train:
            self.net.train()
        else:
            self.net.eval()

        # If requested, put the model in calibration mode
        if calibration is not None:
            self.initialize_calibrator(**calibration)

        # Store the list of input keys to the forward/loss functions. These
        # should be specified as a dictionary mapping the name of the argument
        # in the forward/loss function to a data product name.
        self.input_dict = network_input
        self.loss_dict  = loss_input

        if not isinstance(network_input, dict):
            warn("Specify `network_input` as a dictionary, not a list.",
                 DeprecationWarning)
            fn   = self.model_class.forward
            keys = list(signature(fn).parameters.keys())[1:] # Skip `self`
            num_input = len(network_input)
            self.input_dict = {
                    keys[i]:network_input[i] for i in range(num_input)}

        if loss_input is not None and not isinstance(loss_input, dict):
            warn("Specify `loss_input` as a dictionary, not a list.",
                 DeprecationWarning)
            fn   = self.loss_class.forward
            keys = list(signature(fn).parameters.keys())[1:] # Skip `self`
            num_input = len(loss_input)
            self.loss_dict = {keys[i]:loss_input[i] for i in range(num_input)}

        # If requested, freeze some/all the model weights
        self.freeze_weights()

        # If requested, load the some/all the model weights
        self.load_weights(model_path)

    def initialize_calibrator(self, calibrator, calibrator_loss):
        """Switch model to calibration mode.

        Allows to calibrate logits to respond linearly to probability,
        for instance.

        Parameters
        ----------
        calibrator : dict
            Calibrator configuration dictionary
        calibrator_loss : dict
            Calibrator loss configuration dictionary
        """
        # Switch to calibration mode
        self.net = calibrator_factory(model=self.net, **calibrator)
        self.loss_fn = calibrator_loss_factory(**calibrator_loss)

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
                self.clean_config(val)

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
        input_dict, loss_dict = {}, {}
        with torch.set_grad_enabled(self.train):
            # Load the data products for the model forward
            input_dict = {}
            for param, name in self.input_dict.items():
                assert name in data, (
                        f"Must provide `{name}` in the dataloader schema to "
                         "input into the model forward.")

                value = data[name]
                if isinstance(value, TensorBatch):
                    value = data[name].to_tensor(
                            device=self.rank, dtype=self.dtype)
                input_dict[param] = value

            # Load the data products for the loss function
            loss_dict = {}
            if self.loss_dict is not None:
                for param, name in self.loss_dict.items():
                    assert name in data, (
                            f"Must provide `{name}` in the dataloader schema to "
                             "input into the loss function.")

                    value = data[name]
                    if isinstance(value, TensorBatch):
                        value = data[name].to_tensor(
                            device=self.rank, dtype=self.dtype)
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
            result = self.net(**input_dict)

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

            # Convert to numpy, if requested
            if self.to_numpy:
                for key, value in result.items():
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

    def save_state(self, iteration):
        """Save the model state.

        Save three things from the model:
        - global_step (iteration)
        - state_dict (model parameter values)
        - optimizer (optimizer parameter values)

        Parameters
        ----------
        iteration : int
            Iteration step index
        """
        # Make sure that the weight prefix is valid
        assert self.weight_prefix, (
                "Must provide a weight prefix to store them.")

        filename = f'{self.weight_prefix}-{iteration:d}.ckpt'
        torch.save({
            'global_step': iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filename)
