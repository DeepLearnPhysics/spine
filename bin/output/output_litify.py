#!/usr/bin/env python3
"""Create a compact output directly from a V2 SPINE HDF5 file."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import yaml

from spine.io.transform import DEFAULT_LITE_KEYS, litify_hdf5


def load_keys(config_path: str) -> list[str]:
    """Load product keys from a small or full SPINE YAML configuration."""
    with open(config_path, encoding="utf-8") as config_file:
        config: Any = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise TypeError("Litify configuration must contain a mapping.")

    keys = config.get("keys")
    if keys is None:
        writer = config.get("io", {}).get("writer", {})
        keys = writer.get("keys")
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise ValueError(
            "Configuration must define a string list under `keys` or "
            "`io.writer.keys`."
        )
    return keys


def resolve_keys(
    cli_keys: Sequence[str] | None,
    config_path: str | None,
) -> tuple[str, ...]:
    """Resolve CLI/config product selection with an explicit precedence."""
    if cli_keys is not None and config_path is not None:
        raise ValueError("Use either `--keys` or `--config`, not both.")
    if cli_keys is not None:
        return tuple(cli_keys)
    if config_path is not None:
        return tuple(load_keys(config_path))
    return DEFAULT_LITE_KEYS


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Structurally litify a format-V2 SPINE HDF5 output without "
            "deserializing events or rebuilding classes."
        )
    )
    parser.add_argument("source", help="Input V2 HDF5 file.")
    parser.add_argument("target", help="Output lite HDF5 file.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--keys",
        nargs="+",
        metavar="PRODUCT",
        help="Products to retain. Administrative index products are automatic.",
    )
    selection.add_argument(
        "--config",
        help=(
            "YAML file containing `keys`, or an existing SPINE configuration "
            "containing `io.writer.keys`."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("lite", "fixed_only"),
        default="lite",
        help="Retain the standard lite fields or fixed fields only.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4096,
        help="Object rows processed per fixed-table block.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output atomically.",
    )
    return parser


def main() -> None:
    """Parse arguments and run structural litification."""
    args = build_parser().parse_args()
    keys = resolve_keys(args.keys, args.config)
    litify_hdf5(
        args.source,
        args.target,
        keys=keys,
        mode=args.mode,
        overwrite=args.overwrite,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
