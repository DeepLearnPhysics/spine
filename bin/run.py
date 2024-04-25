#!/usr/bin/env python3
"""Main driver for training, validation, inference and analysis."""

import os
import sys
import argparse

import yaml

# Add parent lartpc_mlreco3d directory to the python path
current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.main_funcs import run


def main(config, source, output, detect_anomaly):
    """Main driver for training/validation/inference/analysis.

    Performs these basic functions:
    - Update the configuration with the command-line arguments
    - Run the appropriate piece of code

    Parameters
    ----------
    source : Union[str, List[str]]
        Path or list of paths to the input files
    output : str
        Path to the output file
    detect_anomaly : bool
        Whether to turn on anomaly detection in torch
    """
    # Try to find configuration file using the absolute path or under
    # the 'config' directory of the parent lartpc_mlreco3d
    cfg_file = config
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', config)
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"{config} not found")

    # Load the configuration file
    with open(cfg_file, 'r', encoding='utf-8') as cfg_yaml:
        cfg = yaml.safe_load(cfg_yaml)

    # Override the input/output command-line information into the configuration
    if source is not None:
        cfg['iotool']['dataset']['file_keys'] = source
    if output is not None and 'writer' in cfg['iotool']:
        cfg['iotool']['writer']['file_name'] = output

    # Update the configuration of the train/validation process
    if detect_anomaly is not None:
        cfg['trainval']['detect_anomaly'] = detect_anomaly
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None \
            and cfg['trainval']['gpus'] == '-1':
        cfg['trainval']['gpus'] = os.environ.get('CUDA_VISIBLE_DEVICES')

    # Execute train/validation process
    run(cfg)


if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
            description="Runs the training/validation/inference/analysis")
    parser.add_argument('--config', '-c',
                        help='Path to the configuration file',
                        type=str, nargs=1)
    parser.add_argument('--source', '-s', '-S',
                        help='Path or list of paths to data files',
                        type=str, nargs='+')
    parser.add_argument('--output', '-o',
                        help='Path to the output file',
                        type=str, nargs='?')
    parser.add_argument('--detect_anomaly',
                        help='Turns on autograd.detect_anomaly for debugging',
                        type=bool, action='store_const', const=True)
    args = parser.parse_args()

    # Execute the main function
    main(args.config, args.source, args.output, args.detect_anomaly)
