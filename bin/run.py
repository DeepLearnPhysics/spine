#!/usr/bin/env python3
import os
import sys
import yaml
import argparse

# Add parent lartpc_mlreco3d directory to the python path
current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

from mlreco.main_funcs import run


def main(config, data_keys, outfile, detect_anomaly):
    # Try to find configuration file using the absolute path or under
    # the 'config' directory of the parent lartpc_mlreco3d
    cfg_file = config
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', config)
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"{config} not found")

    # Load the configuration file
    with open(cfg_file, 'r') as cfg_yaml:
        cfg = yaml.safe_load(cfg_yaml)

    # Override the input/output command-line information into the configuration
    if data_keys is not None:
        cfg['iotool']['dataset']['data_keys'] = data_keys
    if outfile is not None and 'writer' in cfg['iotool']:
        cfg['iotool']['writer']['file_name'] = outfile

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
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--data_keys', '-s', '-S',
                        help = 'Specify path(s) to data files',
                        nargs = '+')
    parser.add_argument('--outfile', '-o',
                        help = 'Specify path to the output file',
                        nargs = '?')
    parser.add_argument('--detect_anomaly',
                        help = 'Turns on autograd.detect_anomaly for debugging',
                        action = 'store_const', const = True)
    args = parser.parse_args()

    # Execute the main function
    main(args.config, args.data_keys, args.outfile, args.detect_anomaly)
