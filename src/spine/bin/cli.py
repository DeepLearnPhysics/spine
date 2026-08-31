#!/usr/bin/env python3
"""Comprehensive CLI entry point that handles torch imports conditionally."""

import argparse
import os
import pathlib
import sys
from textwrap import dedent
from warnings import warn

from spine.banner import format_banner
from spine.bin.info import get_version, show_info
from spine.bin.source import (
    apply_source_overrides,
    apply_validation_source_overrides,
    get_input_config,
)
from spine.bin.weight import apply_module_weight_overrides
from spine.config import load_config_file, to_inference_config
from spine.config.loader import resolve_config_path
from spine.config.operations import parse_value, set_nested_value


def main(
    config: str,
    source: list[str] | None,
    source_list: str | list[str] | None,
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
    module_weight: list[str] | None = None,
    val_source: list[str] | None = None,
    val_source_list: str | list[str] | None = None,
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
    export_weights: str | None = None,
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
        Input paths, optionally written as ``target=path`` for a composite
        dataset source
    source_list : str or list[str], optional
        Path to a text file containing data file paths, optionally qualified by
        a composite-dataset source name. A list supports multiple targets.
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
    module_weight : list[str], optional
        Module checkpoint overrides in the form ``MODULE=PATH``
    val_source : list[str], optional
        Validation paths, optionally written as ``target=path`` for a
        composite dataset source
    val_source_list : str or list[str], optional
        Validation file-list paths, optionally qualified by composite-dataset
        source names
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
    export_weights : str, optional
        Compose the configured model checkpoints into one CPU inference
        checkpoint and exit without initializing data I/O.
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

    # Apply source selectors separately because composite datasets route them
    # into named source blocks rather than the dataset root.
    apply_source_overrides(cfg["io"], source, source_list)

    # Override the remaining input information into the configuration.
    io_mapping = {
        "n_entry": n,
        "n_skip": nskip,
        "entry_list": entry_list,
        "skip_entry_list": skip_entry_list,
    }
    input_cfg = None
    for io_key, io_value in io_mapping.items():
        if io_value is not None:
            if input_cfg is None:
                input_cfg, _ = get_input_config(cfg["io"])
            input_cfg[io_key] = io_value

    # Validation source selectors follow the training dataset topology while
    # remaining independent of the training paths themselves.
    if val_source is not None or val_source_list is not None:
        validation = cfg.setdefault("validation", {})
        if not isinstance(validation, dict):
            raise TypeError("The `validation` block must be a mapping.")
        apply_validation_source_overrides(
            validation,
            cfg["io"],
            val_source,
            val_source_list,
        )

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

    # Module checkpoints are independent of the optional global checkpoint.
    if module_weight:
        if "model" not in cfg:
            raise KeyError("--module-weight requires a `model` block.")
        apply_module_weight_overrides(cfg["model"], module_weight)

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

    # Weight composition is a terminal model-only operation. It deliberately
    # bypasses driver, data-loader and distributed runtime initialization.
    if export_weights is not None:
        from spine.model import export_model_weights

        digest = export_model_weights(cfg, export_weights)
        print(f"Exported composed weights: {export_weights}")
        print(f"SHA-256: {digest}")
        return

    # For actual training/inference, we need the main functionality
    from spine.main import run

    # Run the main function
    run(cfg)


def cli() -> None:
    """Main CLI entry point with conditional torch imports."""
    parser = argparse.ArgumentParser(
        description="SPINE - Scalable Particle Imaging with Neural Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              spine --version
                  Show version information.
              spine --info
                  Show system and dependency information.
              spine -c config.yaml
                  Run training or inference from a configuration file.
              spine -c config.yaml --set io.loader.batch_size=8
                  Override a configuration parameter.
              spine -c config.yaml --set base.iterations=1000 \\
                --set io.loader.batch_size=16
                  Override multiple configuration parameters.
              spine -c config.yaml --set model.detect_anomaly=true
                  Enable PyTorch anomaly detection.
              spine -c config.yaml \\
                --source larcv=raw.root hdf5=cache.h5
                  Override the sources of a composite dataset.
              spine -c config.yaml --weight-path full-chain.ckpt \\
                --module-weight uresnet_ppn=uresnet.ckpt
                  Override one module after loading a global checkpoint.
              spine -c config.yaml \\
                --module-weight uresnet_ppn=uresnet.ckpt \\
                --module-weight graph_spice=graph-spice.ckpt \\
                --export-weights full-chain.ckpt
                  Compose component checkpoints into one inference artifact.
              spine --help
                  Show this help message.

            For ML training/inference, use the released SPINE container or
            install a compatible PyTorch, PyG, and sparse-convolution ecosystem
            manually.
            """),
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

    # Source options may coexist when they address different composite inputs.
    parser.add_argument(
        "-s",
        "--source",
        nargs="+",
        type=str,
        metavar="[TARGET=]PATH",
        help="Input paths, optionally qualified by a composite source name",
    )
    parser.add_argument(
        "-S",
        "--source-list",
        nargs="+",
        metavar="[TARGET=]LIST",
        help="Input file lists, optionally qualified by composite source names",
    )

    # Validation source options use the same target-aware contract as input.
    parser.add_argument(
        "--val-source",
        nargs="+",
        type=str,
        metavar="[TARGET=]PATH",
        help="Validation paths, optionally qualified by a composite source name",
    )
    parser.add_argument(
        "--val-source-list",
        nargs="+",
        metavar="[TARGET=]LIST",
        help="Validation file lists, optionally qualified by composite source names",
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
        "--module-weight",
        action="extend",
        nargs="+",
        metavar="MODULE=PATH",
        help="Checkpoint for a configured model module; accepts one or more "
        "MODULE=PATH assignments",
    )
    parser.add_argument(
        "--export-weights",
        metavar="PATH",
        help="Compose configured checkpoints into one CPU inference artifact",
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
        module_weight=args.module_weight,
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
        export_weights=args.export_weights,
    )


if __name__ == "__main__":
    cli()
