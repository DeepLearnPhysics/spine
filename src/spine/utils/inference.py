"""Module with helper functions to run inference on a model configuration."""

from copy import deepcopy

import yaml


def get_inference_cfg(
    cfg, file_keys=None, weight_path=None, batch_size=None, num_workers=None, cpu=False
):
    """Turns a training configuration into an inference configuration.

    This script does the following:
    - Turn `train` to `False`
    - Set sequential sampling
    - Load the specified validation file_keys, if requested
    - Load the specified set of weight_path, if requested
    - Reset the batch_size to a different value, if requested
    - Sets num_workers to a different value, if requested
    - Make the model run in CPU mode, if requested

    Parameters
    ----------
    cfg : Union[str, dict]
        Configuration file or Path to the configuration file
    file_keys : str, optional
        Path to the dataset to use for inference
    weight_path : str, optional
        Path to the weigths to use for inference
    batch_size : int, optional
        Number of data samples per batch
    num_workers : int, optional
        Number of workers that load data
    cpu : bool, default False
        Whether or not to execute the inference on CPU

    Returns
    -------
    dict
        Dictionary of parameters to initialize handlers
    """
    # Fetch the training configuration
    if isinstance(cfg, dict):
        cfg = deepcopy(cfg)
    else:
        cfg = open(cfg, "r", encoding="utf-8")
        cfg = yaml.safe_load(cfg)

    # Turn train to False
    if "train" in cfg["base"]:
        del cfg["base"]["train"]

    # Turn on unwrapper
    cfg["base"]["unwrap"] = True

    # Convert mode output to numpy
    cfg["model"]["to_numpy"] = True

    # Get rid of random sampler
    if "sampler" in cfg["io"]["loader"]:
        del cfg["io"]["loader"]["sampler"]

    # Change the batch_size, if requested
    if batch_size is not None:
        cfg["io"]["loader"]["batch_size"] = batch_size

    # Change dataset, if requested
    if file_keys is not None:
        cfg["io"]["loader"]["dataset"]["file_keys"] = file_keys

    # Set the number of workers, if requested
    if num_workers is not None:
        cfg["io"]["loader"]["num_workers"] = num_workers

    # Change weights, if requested
    if weight_path is not None:
        cfg["model"]["weight_path"] = weight_path

    # Put the network in CPU mode, if requested
    if cpu:
        cfg["base"]["world_size"] = 0

    return cfg
