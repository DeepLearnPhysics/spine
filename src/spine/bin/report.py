"""Standalone command-line interface for reducing SPINE metric CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from spine.vis.metric.report import build_report


def build_parser() -> argparse.ArgumentParser:
    """Build the metric reporting argument parser."""
    parser = argparse.ArgumentParser(
        description="Reduce completed SPINE metric CSV shards into a report."
    )
    parser.add_argument("--config", required=True, help="Report YAML configuration.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing completed analyzer CSV shards.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which to write summary.json and plots.",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Run the standalone SPINE metric reporter.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments to parse. If omitted, arguments are read from
        :data:`sys.argv`.

    Returns
    -------
    int
        Process exit status. A successful report returns zero.
    """
    args = build_parser().parse_args(argv)
    summary = build_report(args.config, args.input_dir, args.output_dir)
    summary_path = Path(args.output_dir) / "summary.json"
    print(
        f"Wrote {len(summary['metrics'])} metric summaries to "
        f"{summary_path.resolve()}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli())
