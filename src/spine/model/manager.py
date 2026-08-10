"""Centralize all methods associated with a machine-learning model."""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from spine.data import (
    ClusterLabelBatch,
    EdgeIndexBatch,
    IndexBatch,
    TensorBatch,
    TensorBatchConvertible,
)
from spine.utils.conditional import TORCH_AVAILABLE, torch
from spine.utils.logger import logger
from spine.utils.stopwatch import StopwatchManager
from spine.utils.torch.training import lr_sched_factory, optim_factory

from .factories import model_factory


class ModelManager:
    """Groups all relevant functions to construct a model and its loss."""

    def __init__(
        self,
        name: str,
        modules: Mapping[str, Any],
        network_input: Mapping[str, str],
        loss_input: Mapping[str, str] | None = None,
        weight_path: str | None = None,
        weight_list: str | None = None,
        train: Mapping[str, Any] | None = None,
        to_numpy: bool = False,
        time_dependent_loss: bool = False,
        dtype: str = "float32",
        distributed: bool = False,
        rank: int | None = None,
        detect_anomaly: bool = False,
        find_unused_parameters: bool = False,
        iter_per_epoch: int | None = None,
    ) -> None:
        """Process the model configuration.

        Parameters
        ----------
        name : str
            Name of the model as specified under spine.model.factories
        modules : dict
            Dictionary of modules that make up the model. Top-level blocks
            ending in ``_loss`` are passed only to the loss constructor.
        network_input : List[str]
            List of keys of parsed objects to input into the model forward
        loss_input : List[str], optional
            List of keys of parsed objects to input into the loss forward
        weight_path : str, optional
            Path to global model weights to load
        weight_list : str, optional
            Path to a text file containing a list of weight file paths to load
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
        iter_per_epoch : int, optional
            Number of iterations per epoch (relevant for training)
        """
        # Check that torch is available for model operations
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required to use the model manager. "
                "Install with: pip install spine[model]"
            )

        if not isinstance(modules, Mapping):
            raise TypeError(
                "`modules` must be a mapping of model configuration blocks."
            )
        if not isinstance(network_input, Mapping):
            raise TypeError(
                "`network_input` must map model argument names to data product keys."
            )
        if loss_input is not None and not isinstance(loss_input, Mapping):
            raise TypeError(
                "`loss_input` must map loss argument names to data product keys."
            )
        if train is not None and not loss_input:
            raise ValueError("Training requires a non-empty `loss_input` mapping.")

        # Save parameters
        self.train: bool = train is not None
        self.to_numpy = to_numpy
        self.time_dependent = time_dependent_loss
        try:
            self.dtype = getattr(torch, dtype)
        except AttributeError as err:
            raise ValueError(f"Unknown PyTorch dtype `{dtype}`.") from err
        self.distributed = distributed
        self.rank = rank  # Global rank (process ID in distributed group)
        self.main_process = rank is None or rank == 0
        self.checkpoint_validation: dict[str, Any] | None = None

        # Determine device: use current_device() which setup_ddp() already configured
        if self.rank is None:
            self.device = "cpu"
            self.device_id = None
        else:
            # In distributed mode, setup_ddp() already called torch.cuda.set_device(local_rank)
            # This ensures we use the correct local GPU index, not global rank
            self.device_id = torch.cuda.current_device()
            self.device = f"cuda:{self.device_id}"

        # Initialize the timers and the configuration dictionary
        self.watch = StopwatchManager()
        self.watch.initialize("forward")
        if self.train:
            self.watch.initialize(["backward", "save"])

        # If anomaly detection is requested, set it
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True, check_nan=True)

        # Preserve the complete configuration for checkpoint loading/freezing,
        # and pass a sanitized copy to the model implementations.
        self.model_name = name
        self.model_cfg = deepcopy(modules)
        model_modules = self.clean_config(modules)
        network_modules = self.select_network_modules(model_modules)

        # Initialize the model network and loss functions
        net_cls, loss_cls = model_factory(name)
        try:
            self.net = net_cls(**network_modules)
            self.net.to(device=self.device, dtype=self.dtype)
        except Exception as err:
            msg = f"Failed to instantiate {net_cls}"
            raise type(err)(f"{err}\n{msg}")

        self.loss_fn = None
        if loss_input is not None:
            if loss_cls is None:
                raise ValueError(f"Model `{name}` does not define a loss.")
            try:
                self.loss_fn = loss_cls(**model_modules)
                self.loss_fn.to(device=self.device, dtype=self.dtype)
            except Exception as err:
                msg = f"Failed to instantiate {loss_cls}"
                raise type(err)(f"{err}\n{msg}")

        # If requested, initialize the training process
        if train is not None:
            self.initialize_train(**train, iter_per_epoch=iter_per_epoch)
        else:
            self.net.eval()

        # If requested, freeze some/all the model weights
        self.freeze_weights()
        self._validate_trainable_parameters()

        # Parse the list of weight files to consider for loading
        self.weight_path = weight_path
        if weight_path is not None:
            # If a path is provided, check if it is an simple path or a wildcard pattern
            if weight_list is not None:
                raise ValueError("Cannot specify both `weight_path` and `weight_list`.")
            if not os.path.isfile(weight_path):
                if self.train or not glob.glob(weight_path):
                    raise ValueError(f"Weight file not found: {weight_path}")
                self.weight_path = glob.glob(weight_path)

        elif weight_list is not None:
            with open(weight_list, "r", encoding="utf-8") as f:
                self.weight_path = [line.strip() for line in f if line.strip()]
                if not self.weight_path:
                    raise ValueError(f"No weight paths found in {weight_list}.")

        # Load the weights only if a single weight file is provided. If multiple weight
        # files are provided, the loading will be handled in a loop in the main driver.
        if self.weight_path is None or isinstance(self.weight_path, str):
            self.load_weights(self.weight_path)

        # If the execution is distributed, wrap with DDP
        if self.distributed:
            self.net = torch.nn.parallel.DistributedDataParallel(
                self.net,
                device_ids=[self.device_id],
                output_device=self.device_id,
                find_unused_parameters=find_unused_parameters,
            )

        # Store independent copies of the input mappings.
        self.input_dict = dict(network_input)
        self.loss_dict = None if loss_input is None else dict(loss_input)

    def initialize_train(
        self,
        optimizer: Mapping[str, Any],
        weight_prefix: str = "snapshot",
        restore_optimizer: bool = False,
        save_step: int | None = None,
        save_epoch: float | None = None,
        lr_scheduler: Mapping[str, Any] | None = None,
        iter_per_epoch: int | None = None,
    ) -> None:
        """Initialize the training regimen.

        Parameters
        ----------
        optimizer : dict
            Configuration of the optimizer
        weight_prefix : str, default 'snapshot'
            Path + name of the weight file prefix
        save_step : int, optional
            Number of iterations before recording the model weights
        save_epoch : float, optional
            Fraction of epoch to train on before recording the model weights
        restore_optimizer : bool, default False
            Whether to load the  opimizer state from the torch checkpoint
        lr_scheduler : dict, optional
            Configuration of the learning rate scheduler
        iter_per_epoch : int, optional
            Number of iterations per epoch (relevant for training)
        """
        # Turn train on
        self.train = True
        self.net.train()

        # Store parameters
        self.weight_prefix = weight_prefix
        self.restore_optimizer = restore_optimizer

        # Store the saving parameters
        if save_step is not None and save_epoch is not None:
            raise ValueError("Cannot specify both `save_step` and `save_epoch`.")

        self.save_step = save_step
        if save_epoch is not None:
            # Convert the save epoch to a save step
            if iter_per_epoch is None:
                raise ValueError("`save_epoch` requires `iter_per_epoch`.")
            self.save_step = max(1, int(save_epoch * iter_per_epoch))

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

    def __call__(
        self,
        data: Mapping[str, Any],
        iteration: int | None = None,
        epoch: float | None = None,
    ) -> dict[str, Any]:
        """Calls the forward (and backward) function on a batch of data.

        Parameters
        ----------
        data : dict
            Dictionary of input data product keys which each map to its
            associated batched data product
        iteration : int, optional
            Iteration number (relevant for training)
        epoch : float, optional
            Epoch fractional count (relevant for training)

        Returns
        -------
        dict
            Dictionary of model and loss outputs
        """
        # Reset active stopwatches
        self.watch.reset_if_active()

        # Validate training-owned inputs before changing optimizer state
        if self.train:
            if iteration is None:
                raise ValueError(
                    "Must provide iteration information when training a model."
                )
            self.optimizer.zero_grad(set_to_none=True)

        # Run the model forward
        self.watch.start("forward")
        result = self.forward(data, iteration)
        self.watch.stop("forward")

        # If training, run the backward pass and update the weights
        if self.train:
            if "loss" not in result:
                raise RuntimeError("Every trainable model must return a `loss` value.")
            self.watch.start("backward")
            self.backward(result["loss"])
            self.watch.stop("backward")

        # The driver owns checkpoint boundaries so it can validate these
        # weights before timing and serializing the associated checkpoint.

        # If requested, cast the result dictionary to numpy
        if self.to_numpy:
            self.cast_to_numpy(result)

        return result

    def evaluate(
        self,
        data: Mapping[str, Any],
        iteration: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate one batch without gradients or parameter updates.

        The manager temporarily switches the network and loss to evaluation
        mode, then restores their training state before returning. This allows
        validation to reuse the exact in-memory model, including its DDP
        wrapper, without constructing or reloading a second model.

        Parameters
        ----------
        data : mapping
            Validation batch containing the configured network and loss inputs.
        iteration : int, optional
            Current training iteration, forwarded to time-dependent losses.

        Returns
        -------
        dict
            Model and loss outputs for the validation batch.
        """
        # Record manager/module modes before temporarily entering evaluation
        was_training = self.train
        net_training = self.net.training
        loss_training = self.loss_fn.training if self.loss_fn is not None else None

        # Disable training behavior in the manager, network and loss module
        self.train = False
        self.net.eval()
        if self.loss_fn is not None:
            self.loss_fn.eval()

        # Run without gradients, restoring every mode even if forwarding fails
        try:
            with torch.no_grad():
                result = self.forward(data, iteration)
        finally:
            self.train = was_training
            self.net.train(net_training)
            if self.loss_fn is not None and loss_training is not None:
                self.loss_fn.train(loss_training)

        # Apply the ordinary output conversion after leaving evaluation mode
        if self.to_numpy:
            self.cast_to_numpy(result)

        return result

    def should_save(self, iteration: int) -> bool:
        """Return whether an iteration is a configured checkpoint boundary.

        Parameters
        ----------
        iteration : int
            Zero-based training iteration.

        Returns
        -------
        bool
            Whether the iteration completes one configured save period.
        """
        return self.save_step is not None and ((iteration + 1) % self.save_step) == 0

    @classmethod
    def clean_config(cls, config: Any) -> Any:
        """Remove model loading/freezing keys from all level of a dictionary.

        This is used to remove the weight loading/freezing from the input
        configuration before it is fed to the model/loss classes.

        Parameters
        ----------
        config : Mapping
            Dictionary to copy and sanitize

        Returns
        -------
        object
            Deep copy of the configuration without manager-only keys
        """
        config = deepcopy(config)
        keys = ["model_name", "weight_path", "freeze_weights"]
        if isinstance(config, dict):
            for k in keys:
                config.pop(k, None)
            for key, val in config.items():
                config[key] = cls.clean_config(val)
        elif isinstance(config, list):
            config = [cls.clean_config(val) for val in config]

        return config

    @staticmethod
    def select_network_modules(config: Mapping[str, Any]) -> dict[str, Any]:
        """Exclude top-level loss blocks from network configuration.

        Loss constructors receive the complete module configuration because
        they may depend on both model structure and loss-specific settings.
        Network constructors receive only blocks that do not end in
        ``"_loss"``.
        """
        return {
            module_name: module_cfg
            for module_name, module_cfg in config.items()
            if not module_name.endswith("_loss")
        }

    def freeze_weights(self) -> None:
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
                if not count:
                    raise ValueError(
                        f"Could not find any weights to freeze for {module}"
                    )

                logger.info("Froze %d weights in module %s", count, module)

            # Keep the BFS going by adding the nested blocks
            for key in config:
                if isinstance(config[key], dict):
                    module_items.append((key, config[key]))

    def _validate_trainable_parameters(self) -> None:
        """Ensure training has at least one parameter eligible for updates.

        Parameter ``requires_grad`` flags are authoritative here because this
        check runs after every configured module has been frozen. Inspecting
        the configuration or optimizer instead could include weights that were
        subsequently frozen or miss parameters frozen directly by a model.

        Raises
        ------
        ValueError
            If training is enabled but every network parameter is frozen.
        """
        if self.train and not any(
            param.requires_grad for param in self.net.parameters()
        ):
            raise ValueError(
                "Training requires at least one model parameter with "
                "`requires_grad=True`, but all model weights are frozen. "
                "Use inference mode or unfreeze at least one model component."
            )

    def load_weights(self, full_weight_path: str | None) -> None:
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

        # DDP checkpoints use the unwrapped network state. This also permits
        # loading another checkpoint after an inference model has been wrapped.
        net = (
            getattr(self.net, "module", self.net)
            if getattr(self, "distributed", False)
            else self.net
        )

        # Loop over provided model paths
        for module, weight_path, model_name in weight_paths:
            # Module-level weight paths must resolve to a single checkpoint.
            if not os.path.isfile(weight_path):
                raise ValueError(
                    "Weight file not found for module " f"{module}: {weight_path}"
                )

            # Load weight file into existing model
            logger.info(
                "Restoring weights for module %s from %s...", module, weight_path
            )
            with open(weight_path, "rb") as f:
                # Read checkpoint
                try:
                    checkpoint = torch.load(
                        f, map_location=self.device, weights_only=True
                    )
                except TypeError as err:
                    if "weights_only" not in str(err):
                        raise
                    f.seek(0)
                    checkpoint = torch.load(f, map_location=self.device)
                state_dict = checkpoint["state_dict"]

                # Check that all the needed weights are provided
                missing_keys = []
                if module == self.model_name:
                    for name in net.state_dict():
                        if not name in state_dict.keys():
                            missing_keys.append((name, name))

                else:
                    # Update the key names according to the name used to store
                    state_dict = {}
                    for name in net.state_dict():
                        if f"{module}." in name:
                            suffix = "." if len(model_name) > 0 else ""
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
                bad_keys = net.load_state_dict(state_dict, strict=False)
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
                    validation = checkpoint.get("validation")
                    if validation is not None:
                        self.checkpoint_validation = deepcopy(validation)

            logger.info("Done.")

    def prepare_data(
        self, data: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
                if name not in data:
                    raise ValueError(
                        f"Must provide `{name}` in the dataloader schema to "
                        "input into the model forward."
                    )

                value = data[name]
                if isinstance(value, (TensorBatch, ClusterLabelBatch)):
                    value = data[name].to_tensor(device=self.device, dtype=self.dtype)
                input_dict[param] = value

            # Load the data products for the loss function
            loss_dict = {}
            if self.loss_dict is not None:
                for param, name in self.loss_dict.items():
                    if name not in data:
                        raise ValueError(
                            f"Must provide `{name}` in the dataloader schema "
                            "to input into the loss function."
                        )

                    value = data[name]
                    if isinstance(value, (TensorBatch, ClusterLabelBatch)):
                        value = data[name].to_tensor(
                            device=self.device, dtype=self.dtype
                        )
                    loss_dict[param] = value

        return input_dict, loss_dict

    def forward(
        self, data: Mapping[str, Any], iteration: int | None = None
    ) -> dict[str, Any]:
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
                # Configured loss inputs guarantee construction of a loss module.
                assert self.loss_fn is not None
                if not self.time_dependent:
                    result.update(self.loss_fn(**loss_dict, **result))
                else:
                    result.update(
                        self.loss_fn(iteration=iteration, **loss_dict, **result)
                    )

        return result

    def backward(self, loss: Any) -> None:
        """Run the backward step on the model.

        Parameters
        ----------
        loss : torch.tensor
            Scalar loss value to step the model weights
        """
        # Fail with model-level context instead of PyTorch's opaque message
        # when a configured objective is detached from the trainable graph.
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
            raise RuntimeError(
                "Cannot run backward because the loss does not require gradients. "
                "Ensure it depends on at least one trainable model parameter."
            )

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

    def cast_to_numpy(self, result: dict[str, Any]) -> None:
        """Casts the model output data products to numpy object in place.

        Parameters
        ----------
        result : dict
            Dictionary of model and loss outputs
        """
        # Loop over the key, value pairs in the result dictionary
        for key, value in result.items():
            if isinstance(value, ClusterLabelBatch):
                result[key] = value.to_numpy()
                continue
            if isinstance(value, TensorBatchConvertible):
                value = value.to_tensor_batch()

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
                and len(value) > 0
                and isinstance(value[0], (TensorBatch, TensorBatchConvertible))
            ):
                # List of tensor batches
                result[key] = [
                    (
                        v.to_tensor_batch().to_numpy()
                        if isinstance(v, TensorBatchConvertible)
                        else v.to_numpy()
                    )
                    for v in value
                ]

            else:
                dtype = type(value)
                raise ValueError(f"Cannot cast output {key} of type {dtype} to numpy.")

    def save_state(
        self,
        iteration: int,
        epoch: float | None,
        validation: Mapping[str, Any] | None = None,
    ) -> None:
        """Save the model state.

        Save the training state associated with this checkpoint:
        - global_step (iteration)
        - global_epoch (epoch progress)
        - state_dict (model parameter values)
        - optimizer (optimizer parameter values)
        - validation (optional metrics and early-stopping progress)

        Parameters
        ----------
        iteration : int
            Iteration step index
        epoch : float, optional
            Epoch progress associated with the checkpoint.
        validation : mapping, optional
            Validation metrics and early-stopping state associated with these
            exact weights.
        """
        # Make sure that the weight prefix is valid
        if not self.weight_prefix:
            raise ValueError("Must provide a weight prefix to store model state.")

        filename = f"{self.weight_prefix}-{iteration:d}.ckpt"
        model = self.net if not self.distributed else self.net.module
        checkpoint = {
            "global_step": iteration,
            "global_epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if validation is not None:
            checkpoint["validation"] = dict(validation)

        torch.save(checkpoint, filename)
