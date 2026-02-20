#!/usr/bin/env python3
"""Comprehensive CLI entry point that handles torch imports conditionally."""

import argparse
import os
import pathlib
import sys
from typing import List

from spine.config import load_config_file
from spine.config.loader import resolve_config_path
from spine.config.operations import parse_value, set_nested_value


def main(
    config: str,
    source: List[str],
    source_list: str,
    output: str,
    n: int,
    nskip: int,
    entry_list: str,
    skip_entry_list: str,
    log_dir: str,
    weight_prefix: str,
    weight_path: str,
    config_overrides: List[str],
):
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
    entry_list : str
        Path to a text file containing a list of entries to process
    skip_entry_list : str
        Path to a text file containing a list of entries to skip
    log_dir : str
        Path to the directory for storing the training log
    weight_prefix : str
        Path to the directory for storing the training weights
    weight_path : str
        Path to a weight file or pattern for multiple weight files to load
        the model weights
    config_overrides : List[str]
        List of config overrides in the form "key.path=value"
    """
    # Load the configuration tools to find the appropriate config file
    cfg_file = resolve_config_path(config, current_dir=os.getcwd())

    # Load the configuration file using the advanced loader
    cfg = load_config_file(cfg_file)

    # If there is no base block, build one
    if "base" not in cfg:
        cfg["base"] = {}

    # Propagate the configuration parent directory to enable relative paths
    parent_path = str(pathlib.Path(cfg_file).parent)
    cfg["base"]["parent_path"] = parent_path

    # The configuration must minimally contain an IO block
    if "io" not in cfg:
        raise KeyError("Configuration file must contain an `io` block.")

    # Override the input command-line information into the configuration
    io_mapping = {
        "file_keys": source,
        "file_list": source_list,
        "n_entry": n,
        "n_skip": nskip,
        "entry_list": entry_list,
        "skip_entry_list": skip_entry_list,
    }
    for io_key, io_value in io_mapping.items():
        if io_value is not None:
            if "reader" in cfg["io"] and cfg["io"]["reader"] is not None:
                cfg["io"]["reader"][io_key] = io_value
            elif "loader" in cfg["io"] and cfg["io"]["loader"] is not None:
                assert (
                    "dataset" in cfg["io"]["loader"]
                ), "Missing `dataset` block in `io.loader` for input configuration."
                cfg["io"]["loader"]["dataset"][io_key] = io_value
            else:
                raise KeyError("Must specify `loader` or `reader` in the `io` block.")

    # Override the output path if provided
    if output is not None and "writer" in cfg["io"]:
        cfg["io"]["writer"]["file_name"] = output

    # Override logging and weight storage paths if provided
    if log_dir is not None:
        cfg["base"]["log_dir"] = log_dir

    if weight_prefix is not None:
        if not "train" in cfg["base"]:
            raise KeyError(
                "--weight_prefix flag provided: must specify "
                "`train` in the `base` block."
            )
        cfg["base"]["train"]["weight_prefix"] = weight_prefix

    # Override the weight loading path if provided
    if weight_path is not None:
        cfg["model"]["weight_path"] = weight_path

    # Apply any generic config overrides from --set arguments
    if config_overrides:
        for override in config_overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid --set format: '{override}'. "
                    f"Expected format: 'key.path=value'"
                )

            key_path, value_str = override.split("=", 1)
            key_path = key_path.strip()
            value_str = value_str.strip()

            # Parse the value (handles strings, numbers, booleans, lists, etc.)
            value = parse_value(value_str)

            # Set the nested value (returns tuple of (config, applied))
            cfg, _ = set_nested_value(cfg, key_path, value)

    # For actual training/inference, we need the main functionality
    from spine.main import run

    # Run the main function
    run(cfg)


def cli():
    """Main CLI entry point with conditional torch imports."""
    parser = argparse.ArgumentParser(
        description="SPINE - Scalable Particle Imaging with Neural Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spine --version                                Show version information
  spine --info                                  Show system and dependency info
  spine -c config.cfg                           Run ML training/inference with config file
  spine -c config.cfg --set io.loader.batch_size=8    Override config parameters
  spine -c config.cfg --set base.iterations=1000 --set io.loader.batch_size=16
  spine -c config.cfg --set model.detect_anomaly=true Debug PyTorch issues
  spine --help                                  Show this help message

For ML training/inference functionality, ensure PyTorch is installed:
  pip install spine[model]
""",
    )

    # Add a version command
    parser.add_argument(
        "--version", "-v", action="version", version=f"SPINE {get_version()}"
    )

    # Add basic info command
    parser.add_argument(
        "--info",
        "-i",
        action="store_true",
        help="Show system and dependency information",
    )

    # Add config file argument (-c/--config only)
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the configuration file (requires torch dependencies)",
    )

    # Add mutually exclusive group for source input
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-s", "--source", nargs="+", type=str, help="List of paths to the input files"
    )
    group.add_argument(
        "-S",
        "--source-list",
        help="Path to a text file containing a list of data file paths",
    )

    # Add output argument
    parser.add_argument("-o", "--output", help="Path to the output file")

    # Add entry and skip arguments
    parser.add_argument(
        "-n", "--iterations", type=int, help="Number of iterations to run"
    )

    parser.add_argument("--nskip", type=int, help="Number of iterations to skip")

    parser.add_argument(
        "--entry-list",
        help="Path to a text file containing a list of entries to process",
    )

    parser.add_argument(
        "--skip-entry-list",
        help="Path to a text file containing a list of entries to skip",
    )

    # Add logging and weight storage arguments
    parser.add_argument(
        "--log-dir", help="Path to the directory for storing the training log"
    )

    parser.add_argument(
        "--weight-prefix", help="Path to the directory for storing the training weights"
    )

    # Add path to weight file or pattern for loading model weights
    parser.add_argument(
        "--weight-path",
        help="Path string a weight file or pattern for multiple weight "
        "files to load model weights",
    )

    # Add option to dynamically override any config parameter using dot notation
    # (e.g., --set io.loader.batch_size=8)
    parser.add_argument(
        "--set",
        action="append",
        dest="config_overrides",
        metavar="KEY=VALUE",
        help="Override any config parameter using dot notation "
        "(e.g., --set io.loader.batch_size=8). "
        "Can be used multiple times for multiple overrides.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # If no arguments provided and no config, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # Handle version and info without torch
    if args.info:
        show_info()
        return

    config_file = args.config

    # For actual training/inference, call the main function
    main(
        config=config_file,
        source=args.source,
        source_list=args.source_list,
        output=args.output,
        n=args.iterations,
        nskip=args.nskip,
        entry_list=args.entry_list,
        skip_entry_list=args.skip_entry_list,
        log_dir=args.log_dir,
        weight_prefix=args.weight_prefix,
        weight_path=args.weight_path,
        config_overrides=args.config_overrides,
    )


def get_version():
    """Get SPINE version without importing heavy dependencies."""
    try:
        from spine.version import __version__

        return __version__
    except ImportError:
        return "unknown"


def show_info():
    """Show comprehensive package and system information."""
    print(f"SPINE (Scalable Particle Imaging with Neural Embeddings) v{get_version()}")
    print("https://github.com/DeepLearnPhysics/spine")
    print()

    # Check and display dependency status
    deps = check_dependencies()

    print("Dependency Status:")
    print("-" * 40)

    for name, version in deps.items():
        status = f"✓ {version}" if version else "✗ Not available"
        print(f"{name:15}: {status}")

    print(f"\nPython: {sys.version}")
    print()

    print("Available functionality:")
    print("  Core: Mathematical operations, data handling, I/O")

    if deps["torch"]:
        print(f"  Model: Neural networks available (PyTorch {deps['torch']})")
    else:
        print("  Model: Not available (install with: pip install spine[model])")

    if deps["plotly"]:
        print(f"  Visualization: Available (Plotly {deps['plotly']})")
    else:
        print("  Visualization: Not available (install with: pip install spine[viz])")

    if deps["torch"] is None:
        print("\n" + "=" * 50)
        print("NOTICE: PyTorch not found!")
        print("For full ML functionality, install with:")
        print("  pip install spine[model]")
        print("=" * 50)


def check_dependencies():
    """Check what optional dependencies are available."""
    deps = {}

    # Check PyTorch
    try:
        import torch

        deps["torch"] = torch.__version__
    except ImportError:
        deps["torch"] = None

    # Check visualization dependencies
    try:
        import matplotlib

        deps["matplotlib"] = matplotlib.__version__
    except ImportError:
        deps["matplotlib"] = None

    try:
        import plotly

        deps["plotly"] = plotly.__version__
    except ImportError:
        deps["plotly"] = None

    try:
        import seaborn

        deps["seaborn"] = seaborn.__version__
    except ImportError:
        deps["seaborn"] = None

    # Check ML dependencies
    try:
        import MinkowskiEngine

        deps["minkowski"] = MinkowskiEngine.__version__
    except ImportError:
        deps["minkowski"] = None

    return deps


if __name__ == "__main__":
    cli()
