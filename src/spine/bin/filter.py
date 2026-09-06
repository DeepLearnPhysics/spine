"""Standalone command-line interface for file-aware entry filtering."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from spine.io.filter import build_manifest, scan_sources


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive source and source-list arguments."""
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--source",
        nargs="+",
        help="Input paths or glob expressions to inspect.",
    )
    sources.add_argument(
        "--source-list",
        help="Text file containing input paths or glob expressions.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        help="Directory containing reusable per-source counter records.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``spine-filter`` command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect source entries and build a reusable file-aware eligibility "
            "manifest. max_count is exclusive: count < max_count is accepted."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    scan = commands.add_parser(
        "scan", help="Create or validate persistent per-source counter records."
    )
    scan.add_argument("--config", required=True, help="Filter YAML configuration.")
    _add_source_arguments(scan)
    scan.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of source files to inspect concurrently (default: 1).",
    )
    scan.add_argument(
        "--force",
        "--replace",
        action="store_true",
        help="Rescan sources even when a compatible counter record exists.",
    )

    build = commands.add_parser(
        "build", help="Validate scan records and build a consolidated manifest."
    )
    build.add_argument("--config", required=True, help="Filter YAML configuration.")
    _add_source_arguments(build)
    build.add_argument(
        "--output", required=True, help="Destination entry-filter manifest."
    )
    build.add_argument(
        "--output-source-list",
        required=True,
        help="Destination list of sources containing accepted entries.",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Run the standalone entry-filter scanner or manifest builder.

    Parameters
    ----------
    argv : sequence[str], optional
        Arguments to parse.  When omitted, use :data:`sys.argv`.

    Returns
    -------
    int
        Zero after a successful operation.
    """
    args = build_parser().parse_args(argv)
    common = {
        "config": args.config,
        "sources": args.source,
        "source_list": args.source_list,
        "cache_dir": args.cache_dir,
    }
    if args.command == "scan":
        records = scan_sources(
            **common,
            workers=args.workers,
            force=args.force,
        )
        reused = sum(record["reused"] for record in records)
        print(
            f"Prepared {len(records)} scan records: {reused} reused, "
            f"{len(records) - reused} scanned."
        )
        return 0

    manifest = build_manifest(
        **common,
        output=args.output,
        output_source_list=args.output_source_list,
    )
    print(
        f"Wrote {args.output}: {manifest['accepted_entries']} accepted, "
        f"{manifest['rejected_entries']} rejected."
    )
    return 0


if __name__ == "__main__":
    sys.exit(cli())
