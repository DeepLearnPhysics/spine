#!/usr/bin/env python3
"""Comprehensive CLI entry point that handles torch imports conditionally."""

import argparse
import os
import pathlib
import sys

import yaml


def main(
    config,
    source,
    source_list,
    output,
    n,
    nskip,
    detect_anomaly,
    log_dir,
    weight_prefix,
    weight_path,
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
    # the 'config' directory relative to the current working directory
    cfg_file = config
    if not os.path.isfile(cfg_file):
        # Try to find it in a config subdirectory
        cfg_file = os.path.join("config", config)
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"Configuration not found: {config}")

    # Load the configuration file
    with open(cfg_file, "r", encoding="utf-8") as cfg_yaml:
        cfg = yaml.safe_load(cfg_yaml)

    # If there is no base block, build one
    if "base" not in cfg:
        cfg["base"] = {}

    # Propagate the configuration parent directory to enable relative paths
    parent_path = str(pathlib.Path(cfg_file).parent)
    cfg["base"]["parent_path"] = parent_path

    # The configuration must minimally contain an IO block
    assert "io" in cfg, "Must provide an `io` block in the configuration."

    # Override the input/output command-line information into the configuration
    if (source is not None and len(source) > 0) or source_list is not None:
        if "reader" in cfg["io"]:
            cfg["io"]["reader"]["file_keys"] = source or source_list
        elif "loader" in cfg["io"]:
            cfg["io"]["loader"]["dataset"]["file_keys"] = source or source_list
        else:
            raise KeyError("Must specify `loader` or `reader` in the `io` block.")

    if n is not None:
        if "reader" in cfg["io"]:
            cfg["io"]["reader"]["n_entry"] = n
        elif "loader" in cfg["io"]:
            cfg["io"]["loader"]["dataset"]["n_entry"] = n
        else:
            raise KeyError("Must specify `loader` or `reader` in the `io` block.")

    if nskip is not None:
        if "reader" in cfg["io"]:
            cfg["io"]["reader"]["n_skip"] = nskip
        elif "loader" in cfg["io"]:
            cfg["io"]["loader"]["dataset"]["n_skip"] = nskip
        else:
            raise KeyError("Must specify `loader` or `reader` in the `io` block.")

    if output is not None and "writer" in cfg["io"]:
        cfg["io"]["writer"]["file_name"] = output

    if log_dir is not None:
        cfg["base"]["log_dir"] = log_dir

    if weight_prefix is not None:
        if not "train" in cfg["base"]:
            raise KeyError(
                "--weight_prefix flag provided: must specify "
                "`train` in the `base` block."
            )
        cfg["base"]["train"]["weight_prefix"] = weight_prefix

    if weight_path is not None:
        cfg["model"]["weight_path"] = weight_path

    # Turn on PyTorch anomaly detection, if requested
    if detect_anomaly is not None:
        assert (
            "model" in cfg
        ), "There is no model to detect anomalies for, add `model` block."
        cfg["model"]["detect_anomaly"] = detect_anomaly

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
  spine --version         Show version information
  spine --info           Show system and dependency info
  spine config.cfg       Run ML training/inference with config file
  spine --help           Show this help message

For ML training/inference functionality, ensure PyTorch is installed:
  pip install spine-ml[model]
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

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-s", "--source", nargs="+", type=str, help="List of paths to the input files"
    )
    group.add_argument(
        "-S",
        "--source-list",
        help="Path to a text file containing a list of data file paths",
    )

    parser.add_argument("-o", "--output", help="Path to the output file")

    parser.add_argument(
        "-n", "--iterations", type=int, help="Number of iterations to run"
    )

    parser.add_argument("--nskip", type=int, help="Number of iterations to skip")

    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Whether to turn on anomaly detection in torch",
    )

    parser.add_argument(
        "--log-dir", help="Path to the directory for storing the training log"
    )

    parser.add_argument(
        "--weight-prefix", help="Path to the directory for storing the training weights"
    )

    parser.add_argument(
        "--weight-path",
        help="Path string a weight file or pattern for multiple weight files to load model weights",
    )

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
        source=args.source or [],
        source_list=args.source_list,
        output=args.output,
        n=args.iterations,
        nskip=args.nskip or 0,
        detect_anomaly=args.detect_anomaly,
        log_dir=args.log_dir,
        weight_prefix=args.weight_prefix,
        weight_path=args.weight_path,
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
        print("  Model: Not available (install with: pip install spine-ml[model])")

    if deps["plotly"]:
        print(f"  Visualization: Available (Plotly {deps['plotly']})")
    else:
        print(
            "  Visualization: Not available (install with: pip install spine-ml[viz])"
        )

    if deps["torch"] is None:
        print("\n" + "=" * 50)
        print("NOTICE: PyTorch not found!")
        print("For full ML functionality, install with:")
        print("  pip install spine-ml[model]")
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
