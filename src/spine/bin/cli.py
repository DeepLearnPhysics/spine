#!/usr/bin/env python3
"""Comprehensive CLI entry point that handles torch imports conditionally."""

import argparse
import os
import pathlib
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from warnings import warn

from spine.banner import format_banner
from spine.config import load_config_file, to_inference_config
from spine.config.loader import resolve_config_path
from spine.config.operations import parse_value, set_nested_value


def main(
    config: str,
    source: list[str] | None,
    source_list: str | None,
    output: str | None,
    output_dir: str | None,
    output_suffix: str | None,
    n: int | None,
    nskip: int | None,
    entry_list: str | None,
    skip_entry_list: str | None,
    log_dir: str | None,
    weight_prefix: str | None,
    weight_path: str | None,
    weight_list: str | None,
    config_overrides: list[str] | None,
    val_source: list[str] | None = None,
    val_source_list: str | None = None,
    world_size: int | None = None,
    batch_size: int | None = None,
    minibatch_size: int | None = None,
    num_workers: int | None = None,
    epochs: float | None = None,
    iterations: int | None = None,
    tensorboard: bool | None = None,
    tensorboard_dir: str | None = None,
    resume: bool | None = None,
    inference: bool = False,
) -> None:
    """Main driver for training/validation/inference/analysis.

    Performs these basic functions:
    - Update the configuration with the command-line arguments
    - Run the appropriate piece of code

    Parameters
    ----------
    config : str
        Path to the configuration file
    source : list[str], optional
        List of paths to the input files
    source_list : str, optional
        Path to a text file containing a list of data file paths
    output : str, optional
        Path to the output file
    output_dir : str, optional
        Path to the output directory
    output_suffix : str, optional
        Suffix to append to generated output file names
    n : int, optional
        Number of iterations to run
    nskip : int, optional
        Number of dataset entries to skip
    entry_list : str, optional
        Path to a text file containing a list of entries to process
    skip_entry_list : str, optional
        Path to a text file containing a list of entries to skip
    log_dir : str, optional
        Path to the directory for storing the training log
    weight_prefix : str, optional
        Path to the directory for storing the training weights
    weight_path : str, optional
        Path to a weight file or pattern for multiple weight files to load
        the model weights
    weight_list : str, optional
        Path to a text file containing a list of weight file paths to load
        the model weights
    config_overrides : list[str], optional
        List of config overrides in the form "key.path=value"
    val_source : list[str], optional
        List of paths to validation input files
    val_source_list : str, optional
        Path to a text file containing validation data file paths
    world_size : int, optional
        Number of local processes/devices to use
    batch_size : int, optional
        Global loader batch size
    minibatch_size : int, optional
        Per-process loader batch size
    num_workers : int, optional
        Number of data-loader worker processes
    epochs : float, optional
        Number of training epochs
    iterations : int, optional
        Number of driver iterations to run
    tensorboard : bool, optional
        Whether to enable TensorBoard scalar logging
    tensorboard_dir : str, optional
        TensorBoard output directory, relative to ``log_dir`` when not absolute
    resume : bool, optional
        Command-line override for complete training-state restoration. ``None``
        leaves resume selection to the configuration and automatic defaults.
    inference : bool, default False
        Convert a training configuration to deterministic inference before
        applying command-line overrides.
    """
    # Identify the application before configuration loading, which may itself
    # trigger download/cache messages or validation warnings.
    print(format_banner(get_version()), end="")
    print("\nStartup\n-------")
    print(f"Configuration: {config}")

    # Load the configuration tools to find the appropriate config file
    cfg_file = resolve_config_path(config, current_dir=os.getcwd())

    # Load the configuration file using the advanced loader
    cfg = load_config_file(cfg_file)

    # Resource-resolution messages belong to the startup section. Close it
    # explicitly before runtime initialization begins.
    print()

    # If there is no base block, build one
    if "base" not in cfg:
        cfg["base"] = {}

    # Propagate the configuration parent directory to enable relative paths
    parent_path = str(pathlib.Path(cfg_file).parent)
    cfg["base"]["parent_path"] = parent_path

    if val_source is not None and val_source_list is not None:
        raise ValueError("--val-source and --val-source-list are mutually exclusive.")
    if batch_size is not None and minibatch_size is not None:
        raise ValueError("--batch-size and --minibatch-size are mutually exclusive.")
    if epochs is not None and iterations is not None:
        raise ValueError("--epochs and --iterations are mutually exclusive.")
    if tensorboard is False and tensorboard_dir is not None:
        raise ValueError("--tensorboard-dir cannot be used with --no-tensorboard.")
    if inference and (val_source is not None or val_source_list is not None):
        raise ValueError(
            "--val-source and --val-source-list cannot be used with --inference."
        )

    # Convert the loaded training configuration before applying explicit CLI
    # overrides, so command-line arguments remain authoritative.
    if inference:
        cfg = to_inference_config(cfg)

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
    source_override = source is not None or source_list is not None
    for io_key, io_value in io_mapping.items():
        if io_value is not None or (
            source_override and io_key in ("file_keys", "file_list")
        ):
            if "reader" in cfg["io"] and cfg["io"]["reader"] is not None:
                cfg["io"]["reader"][io_key] = io_value
            elif "loader" in cfg["io"] and cfg["io"]["loader"] is not None:
                assert (
                    "dataset" in cfg["io"]["loader"]
                ), "Missing `dataset` block in `io.loader` for input configuration."
                cfg["io"]["loader"]["dataset"][io_key] = io_value
            else:
                raise KeyError("Must specify `loader` or `reader` in the `io` block.")

    # Override validation sources independently of the training input. Remove
    # the alternate selector because validation requires exactly one of them.
    if val_source is not None or val_source_list is not None:
        validation = cfg.setdefault("validation", {})
        if not isinstance(validation, dict):
            raise TypeError("The `validation` block must be a mapping.")
        if val_source is not None:
            validation["file_keys"] = val_source
            validation.pop("file_list", None)
        else:
            validation["file_list"] = val_source_list
            validation.pop("file_keys", None)

    # Override runtime and loader resource settings. Batch shape and worker
    # count are loader properties, while process count and duration belong to
    # the driver base configuration.
    if world_size is not None:
        cfg["base"]["world_size"] = world_size
    if epochs is not None:
        cfg["base"]["epochs"] = epochs
        cfg["base"].pop("iterations", None)
    elif iterations is not None:
        cfg["base"]["iterations"] = iterations
        cfg["base"].pop("epochs", None)

    loader_overrides = {
        "batch_size": batch_size,
        "minibatch_size": minibatch_size,
        "num_workers": num_workers,
    }
    if any(value is not None for value in loader_overrides.values()):
        loader = cfg["io"].get("loader")
        if loader is None:
            raise KeyError(
                "--batch-size, --minibatch-size, and --num-workers require "
                "an `io.loader` block."
            )
        if batch_size is not None:
            loader["batch_size"] = batch_size
            loader.pop("minibatch_size", None)
        elif minibatch_size is not None:
            loader["minibatch_size"] = minibatch_size
            loader.pop("batch_size", None)
        if num_workers is not None:
            loader["num_workers"] = num_workers

    # Override the output configuration if provided
    writer = cfg["io"].get("writer")
    if writer is not None:
        if output is not None:
            writer["file_name"] = output
        if output_dir is not None:
            writer["directory"] = output_dir
        if output_suffix is not None:
            writer["suffix"] = output_suffix
    elif output is not None or output_dir is not None or output_suffix is not None:
        warn(
            "No `io.writer` is configured; output options are ignored.",
            stacklevel=2,
        )

    # Override logging and weight storage paths if provided
    if log_dir is not None:
        cfg["base"]["log_dir"] = log_dir

    if tensorboard is not None:
        if tensorboard is False:
            cfg["base"]["tensorboard"] = False
        elif not isinstance(cfg["base"].get("tensorboard"), dict):
            cfg["base"]["tensorboard"] = True
    if tensorboard_dir is not None:
        tensorboard_cfg = cfg["base"].get("tensorboard")
        if not isinstance(tensorboard_cfg, dict):
            tensorboard_cfg = {}
            cfg["base"]["tensorboard"] = tensorboard_cfg
        tensorboard_cfg["log_dir"] = tensorboard_dir

    if weight_prefix is not None:
        train_cfg = cfg.get("train", cfg["base"].get("train"))
        if train_cfg is None:
            raise KeyError(
                "--weight_prefix flag provided: must specify a `train` block."
            )
        train_cfg["weight_prefix"] = weight_prefix

    # Override the weight loading path if provided
    if weight_path is not None:
        cfg["model"]["weight_path"] = weight_path
    if weight_list is not None:
        cfg["model"]["weight_list"] = weight_list

    # Apply an explicit resume override to either supported train location.
    if resume is not None:
        train_cfg = cfg.get("train", cfg["base"].get("train"))
        if train_cfg is None:
            raise KeyError("--resume/--no-resume requires a `train` block.")
        train_cfg["resume"] = resume

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

    # Override distributed settings from environment variables (SLURM/torchrun)
    # This handles multi-node training where each process sees 1 GPU but is part
    # of a larger distributed group
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        launcher_world_size = int(os.environ["WORLD_SIZE"])
        if world_size is not None and world_size != launcher_world_size:
            raise ValueError(
                f"--world-size={world_size} conflicts with launcher "
                f"WORLD_SIZE={launcher_world_size}."
            )
        cfg["base"]["world_size"] = launcher_world_size
        cfg["base"]["distributed"] = True

    # For actual training/inference, we need the main functionality
    from spine.main import run

    # Run the main function
    run(cfg)


def cli() -> None:
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

For ML training/inference, use the released SPINE container or install a
compatible PyTorch, PyG, and sparse-convolution ecosystem manually.
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
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "-s", "--source", nargs="+", type=str, help="List of paths to the input files"
    )
    source_group.add_argument(
        "-S",
        "--source-list",
        help="Path to a text file containing a list of data file paths",
    )

    # Add mutually exclusive validation source inputs
    val_source_group = parser.add_mutually_exclusive_group()
    val_source_group.add_argument(
        "--val-source",
        nargs="+",
        type=str,
        help="List of paths to validation input files",
    )
    val_source_group.add_argument(
        "--val-source-list",
        help="Path to a text file containing validation data file paths",
    )

    # Add output arguments
    parser.add_argument("-o", "--output", help="Path to the output file")
    parser.add_argument("--output-dir", help="Path to the output directory")
    parser.add_argument(
        "--output-suffix", help="Suffix to append to generated output file names"
    )

    # Add dataset entry and skip arguments
    parser.add_argument(
        "-n",
        "--num-entries",
        dest="num_entries",
        type=int,
        help="Number of dataset entries to load",
    )

    parser.add_argument(
        "--skip-entries",
        "--nskip",
        dest="nskip",
        type=int,
        help="Number of dataset entries to skip",
    )

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

    # Add launch-resource and training-duration arguments
    parser.add_argument(
        "--world-size",
        type=int,
        help="Number of local processes/devices (multi-node launchers set WORLD_SIZE)",
    )
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch-size", type=int, help="Global data-loader batch size"
    )
    batch_group.add_argument(
        "--minibatch-size", type=int, help="Per-process/GPU data-loader batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, help="Number of data-loader worker processes"
    )
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument(
        "--epochs", type=float, help="Number of training epochs"
    )
    duration_group.add_argument(
        "--iterations", type=int, help="Number of driver iterations to run"
    )

    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable TensorBoard scalar logging; use --no-tensorboard to disable",
    )
    parser.add_argument(
        "--tensorboard-dir",
        help="TensorBoard directory (defaults to <log-dir>/tensorboard)",
    )

    parser.add_argument(
        "--weight-prefix", help="Path to the directory for storing the training weights"
    )

    # Add path to weight file or pattern for loading model weights
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument(
        "--weight-path",
        help="Path string a weight file or pattern for multiple weight "
        "files to load model weights",
    )
    weight_group.add_argument(
        "--weight-list",
        help="Path to a text file containing a list of weight file paths",
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume all available training state from the configured checkpoint; "
        "use --no-resume for weights-only initialization",
    )

    parser.add_argument(
        "--inference",
        action="store_true",
        help="Convert a training configuration to inference before running it",
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
        output_dir=args.output_dir,
        output_suffix=args.output_suffix,
        n=args.num_entries,
        nskip=args.nskip,
        entry_list=args.entry_list,
        skip_entry_list=args.skip_entry_list,
        log_dir=args.log_dir,
        weight_prefix=args.weight_prefix,
        weight_path=args.weight_path,
        weight_list=args.weight_list,
        config_overrides=args.config_overrides,
        val_source=args.val_source,
        val_source_list=args.val_source_list,
        world_size=args.world_size,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        iterations=args.iterations,
        tensorboard=args.tensorboard,
        tensorboard_dir=args.tensorboard_dir,
        resume=args.resume,
        inference=args.inference,
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

    model_deps = (
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torch-cluster",
        "MinkowskiEngine",
    )
    missing_model_deps = [name for name in model_deps if not deps[name]]
    if not missing_model_deps:
        print("  Model stack: Available")
    else:
        print(f"  Model stack: Incomplete (missing: {', '.join(missing_model_deps)})")

    if deps["plotly"]:
        print(f"  Visualization: Available (Plotly {deps['plotly']})")
    else:
        print("  Visualization: Not available (install with: pip install spine[viz])")

    if deps["torch"] is None:
        print("\n" + "=" * 50)
        print("NOTICE: PyTorch not found!")
        print("For full ML functionality, use the released SPINE container")
        print("or install the compatible ML ecosystem manually.")
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

    # Check the compiled packages needed by the complete model stack without
    # importing them, as imports may initialize CUDA or compiled extensions.
    for distribution in (
        "torch-geometric",
        "torch-scatter",
        "torch-cluster",
        "MinkowskiEngine",
    ):
        try:
            deps[distribution] = package_version(distribution)
        except PackageNotFoundError:
            deps[distribution] = None

    return deps


if __name__ == "__main__":
    cli()
