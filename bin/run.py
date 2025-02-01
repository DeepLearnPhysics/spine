#!/usr/bin/env python3
"""Main driver for training, validation, inference and analysis."""

import os
import sys
import argparse

import pathlib
import yaml

# Add parent SPINE base directory to the python path
current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)

# Have to do this here because the import order matters. Will
# look into a cleaner way to do this later (TODO)
if os.getenv('FMATCH_BUILDDIR') is not None:
    print('Setting up OpT0Finder...')
    sys.path.append(os.path.join(os.getenv('FMATCH_BASEDIR'), 'python'))
    import flashmatch
    from flashmatch import flashmatch, geoalgo
    print('... done.')

from spine.main import run


def main(config, source, source_list, output, n, nskip, detect_anomaly,
         log_dir, weight_prefix, weight_path):
    """Main driver for training/validation/inference/analysis.

    Performs these basic functions:
    - Update the configuration with the command-line arguments
    - Run the appropriate piece of code

    Parameters
    ----------
    config : str
        Path to the configuration file
    source : List[str]
        List of paths to the input files
    source_list : str
        Path to a text file containing a list of data file paths
    output : str
        Path to the output file
    n : int
        Number of iterations to run
    nskip : int
        Number of iterations to skip
    detect_anomaly : bool
        Whether to turn on anomaly detection in torch
    log_dir : str
        Path to the directory for storing the training log
    weight_prefix : str
        Path to the directory for storing the training weights
    weight_path : str
        Path string a weight file or pattern for multiple weight files to load
        the model weights
    """
    # Try to find configuration file using the absolute path or under
    # the 'config' directory of the parent SPINE repository
    cfg_file = config
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', config)
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"Configuration not found: {config}")

    # Load the configuration file
    with open(cfg_file, 'r', encoding='utf-8') as cfg_yaml:
        cfg = yaml.safe_load(cfg_yaml)

    # If there is no base block, build one
    if 'base' not in cfg:
        cfg['base'] = {}

    # Propagate the configuration parent directory to inable relative paths
    parent_path = str(pathlib.Path(cfg_file).parent)
    cfg['base']['parent_path'] = parent_path

    # The configuration must minimally contain an IO block
    assert 'io' in cfg, (
            "Must provide an `io` block in the configuration.")

    # Override the input/output command-line information into the configuration
    if source is not None or source_list is not None:
        if 'reader' in cfg['io']:
            cfg['io']['reader']['file_keys'] = source or source_list
        elif 'loader' in cfg['io']:
            cfg['io']['loader']['dataset']['file_keys'] = source or source_list
        else:
            raise KeyError("Must specify `loader` or `reader` in the `io` block.")

    if n is not None:
        if 'reader' in cfg['io']:
            cfg['io']['reader']['n_entry'] = n
        elif 'loader' in cfg['io']:
            cfg['io']['loader']['dataset']['n_entry'] = n
        else:
            raise KeyError("Must specify `loader` or `reader` in the `io` block.")

    if nskip is not None:
        if 'reader' in cfg['io']:
            cfg['io']['reader']['n_skip'] = nskip
        elif 'loader' in cfg['io']:
            cfg['io']['loader']['dataset']['n_skip'] = nskip
        else:
            raise KeyError("Must specify `loader` or `reader` in the `io` block.")

    if output is not None and 'writer' in cfg['io']:
        cfg['io']['writer']['file_name'] = output

    if log_dir is not None:
        cfg['base']['log_dir'] = log_dir

    if weight_prefix is not None:
        if not 'train' in cfg['base']:
            raise KeyError("--weight_prefix flag provided: must specify "
                           "`train` in the `base` block.")
        cfg['base']['train']['weight_prefix']=weight_prefix

    if weight_path is not None:
        cfg['model']['weight_path']=weight_path

    # Turn on PyTorch anomaly detection, if requested
    if detect_anomaly is not None:
        assert 'model' in cfg, (
                "There is no model to detect anomalies for, add `model` block.")
        cfg['model']['detect_anomaly'] = detect_anomaly

    # If the -1 option for GPUs is selected, expose the process to all GPUs
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None \
            and cfg['base'].get('gpus', '') == '-1':
        cfg['base']['gpus'] = os.environ.get('CUDA_VISIBLE_DEVICES')

    # Execute train/validation process
    run(cfg)


if __name__ == '__main__':
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(
            description="Runs the training/validation/inference/analysis")

    parser.add_argument('--config', '-c',
                        help='Path to the configuration file',
                        type=str, required=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--source', '-s',
                       help='Path or list of paths to data files',
                       type=str, nargs="+")
    group.add_argument('--source-list', '-S',
                       help='Path to a text file of data file paths',
                       type=str)

    parser.add_argument('--output', '-o',
                        help='Path to the output file',
                        type=str)

    parser.add_argument('-n',
                        help='Number of iterations to process',
                        type=int)

    parser.add_argument('--nskip',
                        help='Number of iterations to skip',
                        type=int)

    parser.add_argument('--detect-anomaly',
                        help='Turns on autograd.detect_anomaly for debugging',
                        action='store_const', const=True)

    parser.add_argument('--log_dir',
                        help='Log directory',
                        type=str, default=None)

    parser.add_argument('--weight_prefix',
                        help='Prefix for weight files',
                        type=str, default=None)

    parser.add_argument('--weight_path',
                        help='Path to a weight file (or pattern to multiple weight files)',
                        type=str, default=None)

    args = parser.parse_args()

    # Execute the main function
    main(args.config, args.source, args.source_list, args.output, args.n,
         args.nskip, args.detect_anomaly, args.log_dir, args.weight_prefix,
         args.weight_path)
