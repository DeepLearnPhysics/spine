"""Centralize all methods associated with a machine-learning model."""

import glob
import os
from copy import deepcopy

import numpy as np

from spine.data import EdgeIndexBatch, IndexBatch, TensorBatch
from spine.utils.conditional import TORCH_AVAILABLE, torch
from spine.utils.logger import logger
from spine.utils.stopwatch import StopwatchManager
from spine.utils.torch.training import lr_sched_factory, optim_factory

from .factories import model_factory


class ModelManager:
    """Groups all relevant functions to construct a model and its loss."""

    def __init__(
        self,
        name,
        modules,
        network_input,
        loss_input=None,
        weight_path=None,
        train=None,
        save_step=None,
        optimizer=None,
        restore_optimizer=False,
        lr_scheduler=None,
        to_numpy=False,
        time_dependent_loss=False,
        dtype="float32",
        distributed=False,
        rank=None,
        detect_anomaly=False,
        find_unused_parameters=False,
    ):
        """Process the model configuration.

        Parameters
        ----------
        name : str
            Name of the model as specified under spine.model.factories
        modules : dict
            Dictionary of modules that make up the model
        network_input : List[str]
            List of keys of parsed objects to input into the model forward
        loss_input : List[str], optional
            List of keys of parsed objects to input into the loss forward
        weight_path : str, optional
            Path to global model weights to load
        to_numpy : int, default False
            Cast model output to numpy ndarray
        time_dependant_loss : bool, default False
            Handles time-dependant loss, such as KL divergence annealing
        train : dict, default None
            Training regimen configuration
        dtype : str, default 'float32'
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
        self.time_dependant = time_dependent_loss
        self.dtype = getattr(torch, dtype)
        self.distributed = distributed
        self.rank = rank
        self.device = "cpu" if self.rank is None else f"cuda:{self.rank}"
        self.main_process = rank is None or rank == 0

        # Initialize the timers and the configuration dictionary
        self.watch = StopwatchManager()
        self.watch.initialize("forward")
        if train:
            self.watch.initialize(["backward", "save"])

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
            self.net.to(device=rank, dtype=self.dtype)
        except Exception as err:
            msg = f"Failed to instantiate {net_cls}"
            raise type(err)(f"{err}\n{msg}")

        try:
            self.loss_fn = loss_cls(**modules)
            self.loss_fn.to(device=rank, dtype=self.dtype)
        except Exception as err:
            msg = f"Failed to instantiate {loss_cls}"
            raise type(err)(f"{err}\n{msg}")

        # If requested, initialize the training process
        if train is not None:
            self.initialize_train(**train)
        else:
            self.train = False
            self.net.eval()

        # If requested, freeze some/all the model weights
        self.freeze_weights()

        # If requested, load the some/all the model weights
        self.weight_path = weight_path
        self.load_weights(weight_path)

        # If the execution is distributed, wrap with DDP
        if self.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=find_unused_parameters,
            )

        # Store the list of input keys to the forward/loss functions. These
        # should be specified as a dictionary mapping the name of the argument
        # in the forward/loss function to a data product name.
        self.input_dict = network_input
        self.loss_dict = loss_input
        assert isinstance(network_input, dict), (
            "Must specify `network_input` as a dictionary mapping model "
            "input keys onto data loader product keys."
        )
        assert loss_input is None or isinstance(loss_input, dict), (
            "Must specify `loss_input` as a dictionary mapping loss "
            "input keys onto data loader product keys."
        )

    def initialize_train(
        self,
        optimizer,
        weight_prefix="snapshot",
        restore_optimizer=False,
        save_step=-1,
        lr_scheduler=None,
    ):
        """Initialize the training regimen.

        Parameters
        ----------
        optimizer : dict
            Configuration of the optimizer
        weight_prefix : str, default 'snapshot'
            Path + name of the weight file prefix
        save_step : int, default -1
            Number of iterations before recording the model weights (-1: never)
        restore_optimizer : bool, default False
            Whether to load the  opimizer state from the torch checkpoint
        lr_scheduler : dict, optional
            Configuration of the learning rate scheduler
        """
        # Turn train on
        self.train = True
        self.net.train()

        # Store parameters
        self.weight_prefix = weight_prefix
        self.save_step = save_step
        self.restore_optimizer = restore_optimizer

        # Make a directory for the weight files, if need be
        save_dir = os.path.dirname(weight_prefix)
        if save_dir and not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Initiliaze the optimizer
        self.optimizer = optim_factory(optimizer, self.net.parameters())

        # Initialize the learning rate scheduler
        self.lr_scheduler = None
        if lr_scheduler is not None:
            self.lr_scheduler = lr_sched_factory(lr_scheduler, self.optimizer)

    def __call__(self, data, iteration=None):
        """Calls the forward (and backward) function on a batch of data.

        Parameters
        ----------
        data : dict
            Dictionary of input data product keys which each map to its
            associated batched data product
        iteration : int, optional
            Iteration number (relevant for time-dependant losses)

        Returns
        -------
        dict
            Dictionary of model and loss outputs
        """
        # Reset the gradient accumulation, free memory
        if self.train:
            self.optimizer.zero_grad(set_to_none=True)

        # Run the model forward
        self.watch.start("forward")
        result = self.forward(data, iteration)
        self.watch.stop("forward")

        # If traning run the backward pass and update the weigths
        if self.train:
            assert (
                "loss" in result
            ), "Every model must return a `loss` value to be trained."
            self.watch.start("backward")
            self.backward(result["loss"])
            self.watch.stop("backward")

        # If training and at an appropriate iteration, save model state
        if self.train:
            self.watch.start("save")
            if iteration is not None:
                save = ((iteration + 1) % self.save_step) == 0
                if save and self.main_process:
                    self.save_state(iteration)
            self.watch.stop("save")

        # If requested, cast the result dictionary to numpy
        if self.to_numpy:
            self.cast_to_numpy(result)

        return result

    def clean_config(self, config):
        """Remove model loading/freezing keys from all level of a dictionary.

        This is used to remove the weight loading/freezing from the input
        configuration before it is fed to the model/loss classes.

        Parameters
        ----------
        config : dict
            Dictionary to remove the keys from
        """
        keys = ["model_name", "weight_path", "freeze_weights"]
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
            if config.get("freeze_weights", False):
                # Fetch the module name to be found in the state dictionary
                model_name = config.get("model_name", module)

                # Set BN and DO layers to evaluation mode
                getattr(self.net, module).eval()

                # Freeze all the weights of this module
                count = 0
                for name, param in self.net.named_parameters():
                    if module in name:
                        key = name.replace(f".{module}.", f".{model_name}.")
                        if key in self.net.state_dict().keys():
                            param.requires_grad = False
                            count += 1

                # Throw if no weights were found to freeze
                assert count, f"Could not find any weights to freeze for {module}"

                logger.info("Froze %d weights in module %s", count, module)

            # Keep the BFS going by adding the nested blocks
            for key in config:
                if isinstance(config[key], dict):
                    module_items.append((key, config[key]))

    def load_weights(self, full_weight_path):
        """Load the weights of certain model components.

        Breadth-first search for `weight_path` parameters in the model
        configuration. If 'weight_path' is found under a module block,
        the weights are loaded for its parameters.

        If a `weight_path` is not found for a given module, load the overall
        weights from `weight_path` under `trainval` for that module instead.

        Parameters
        ----------
        full_weight_path : str
            Path to the weights for the full model
        """
        # If a general model path is provided, add it to the loading list first
        weight_paths = []
        if full_weight_path:
            weight_paths = [(self.model_name, full_weight_path, "")]

        # Find the list of sub-module weights to subsequently load
        module_items = list(self.model_cfg.items())
        while len(module_items) > 0:
            module, config = module_items.pop()
            if config.get("weight_path", "") != "":
                model_name = config.get("model_name", module)
                weight_paths.append((module, config["weight_path"], model_name))
            for key in config:
                if isinstance(config[key], dict):
                    module_items.append((key, config[key]))

        # If no pre-trained weights are requested, nothing to do here
        self.start_iteration = 0
        if not weight_paths:
            return

        # Loop over provided model paths
        for module, weight_path, model_name in weight_paths:
            # Check that the requested weight file can be found. If the path
            # points at > 1 file, skip for now (loaded in a loop later)
            if not os.path.isfile(weight_path):
                if not self.train and glob.glob(weight_path):
                    continue

                raise ValueError(
                    "Weight file not found for module " f"{module}: {weight_path}"
                )

            # Load weight file into existing model
            logger.info(
                "Restoring weights for module %s " "from %s...", module, weight_path
            )
            with open(weight_path, "rb") as f:
                # Read checkpoint
                checkpoint = torch.load(f, map_location=self.device)
                state_dict = checkpoint["state_dict"]

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
                        if f"{module}." in name:
                            suffix = "." if len(model_name) else ""
                            key = name.replace(f"{module}.", f"{model_name}{suffix}")
                            if key in checkpoint["state_dict"].keys():
                                state_dict[name] = checkpoint["state_dict"][key]
                            else:
                                missing_keys.append((name, key))

                # If some necessary keys were not found, throw
                if missing_keys:
                    logger.critical("These necessary parameters could not be found:")
                    for name, key in missing_keys:
                        logger.critical("Parameter %s is missing for %s.", key, name)
                    raise ValueError(
                        "To be loaded, a set of weights "
                        "must provide all necessary parameters."
                    )

                # Load checkpoint. Check that all weights are used
                bad_keys = self.net.load_state_dict(state_dict, strict=False)
                if len(bad_keys.unexpected_keys) > 0:
                    logger.warning(
                        "This weight file contains parameters that could "
                        "not be loaded, indicating that the weight file "
                        "contains more than needed. This might be ok."
                    )
                    logger.warning("Unexpected keys: %s", bad_keys.unexpected_keys)

                # Load the optimizer state from the main weight file only
                if self.train and module == self.model_name and self.restore_optimizer:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])

                # Get the latest iteration from the main weight file only
                if module == self.model_name:
                    self.start_iteration = checkpoint["global_step"] + 1

            logger.info("Done.")

    def prepare_data(self, data):
        """Fetches the necessary data products to form the input to the forward
        function and the input to the loss function.

        Parameters
        ----------
        data : dict
            Dictionary of input data product keys, each of which maps to its
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
                    "input into the model forward."
                )

                value = data[name]
                if isinstance(value, TensorBatch):
                    value = data[name].to_tensor(device=self.rank, dtype=self.dtype)
                input_dict[param] = value

            # Load the data products for the loss function
            loss_dict = {}
            if self.loss_dict is not None:
                for param, name in self.loss_dict.items():
                    assert name in data, (
                        f"Must provide `{name}` in the dataloader schema "
                        "to input into the loss function."
                    )

                    value = data[name]
                    if isinstance(value, TensorBatch):
                        value = data[name].to_tensor(device=self.rank, dtype=self.dtype)
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
        iteration : int, optional
            Iteration number (relevant for time-dependant losses)

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
            if self.loss_dict:
                if not self.time_dependant:
                    result.update(self.loss_fn(**loss_dict, **result))
                else:
                    result.update(
                        self.loss_fn(iteration=iteration, **loss_dict, **result)
                    )

        return result

    def backward(self, loss):
        """Run the backward step on the model.

        Parameters
        ----------
        loss : torch.tensor
            Scalar loss value to step the model weights
        """
        # Run the model backward
        loss.backward()

        # Step the optimizer
        self.optimizer.step()

        # Step the learning rate scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # If the model has a buffer that needs to be updated, do it after
        # the trainable parameter update
        if hasattr(self.net, "update_buffers"):
            logger.info("Updating buffers")
            self.net.update_buffers()

    def cast_to_numpy(self, result):
        """Casts the model output data products to numpy object in place.

        Parameters
        ----------
        result : dict
            Dictionary of model and loss outputs
        """
        # Loop over the key, value pairs in the result dictionary
        for key, value in result.items():
            # Cast to numpy or python scalars
            if np.isscalar(value):
                # Scalar
                result[key] = value

            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                # Scalar tensor
                result[key] = value.item()

            elif isinstance(value, (TensorBatch, IndexBatch, EdgeIndexBatch)):
                # Batch of data
                result[key] = value.to_numpy()

            elif (
                isinstance(value, list)
                and len(value)
                and isinstance(value[0], TensorBatch)
            ):
                # List of tensor batches
                result[key] = [v.to_numpy() for v in value]

            else:
                dtype = type(value)
                raise ValueError(f"Cannot cast output {key} of type {dtype} to numpy.")

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
        assert self.weight_prefix, "Must provide a weight prefix to store them."

        filename = f"{self.weight_prefix}-{iteration:d}.ckpt"
        model = self.net if not self.distributed else self.net.module
        torch.save(
            {
                "global_step": iteration,
                "state_dict": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filename,
        )
