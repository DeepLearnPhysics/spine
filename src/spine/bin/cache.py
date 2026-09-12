"""Command-line maintenance for manifest-backed SPINE caches."""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Sequence

from spine.io.cache import CacheRepository, collect_garbage


def build_parser() -> argparse.ArgumentParser:
    """Build the ``spine-cache`` maintenance command parser."""
    parser = argparse.ArgumentParser(
        description="Inspect and maintain manifest-backed SPINE cache repositories."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    begin = commands.add_parser(
        "begin", help="Register and fence one parallel publication attempt."
    )
    begin.add_argument("path", help="Cache repository to initialize or update.")
    begin.add_argument("stage", help="Logical cache stage to publish.")
    begin.add_argument(
        "--publication-id",
        help="Shared attempt ID (default: generate a UUID).",
    )

    garbage = commands.add_parser(
        "gc", help="Remove stale files unreachable from the current manifest."
    )
    garbage.add_argument("path", help="Cache repository to inspect.")
    garbage.add_argument(
        "--dry-run",
        action="store_true",
        help="Report eligible files without deleting them.",
    )
    garbage.add_argument(
        "--min-age",
        type=float,
        default=86400.0,
        metavar="SECONDS",
        help="Minimum inactivity age in seconds (default: 86400).",
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    """Run a cache maintenance command.

    Parameters
    ----------
    argv : sequence[str], optional
        Arguments to parse. When omitted, use :data:`sys.argv`.

    Returns
    -------
    int
        Zero after successful maintenance.
    """
    args = build_parser().parse_args(argv)
    if args.command == "begin":
        publication_id = args.publication_id or uuid.uuid4().hex
        repository = CacheRepository(args.path, create=True)
        repository.begin_publication(args.stage, publication_id)
        print(publication_id)
        return 0

    repository = CacheRepository(args.path)
    report = collect_garbage(
        repository,
        dry_run=args.dry_run,
        min_age_seconds=args.min_age,
    )

    action = "Would remove" if report.dry_run else "Removed"
    count = len(report.shard_generations) + len(report.pending_transactions)
    print(f"{action} {count} cache generations ({report.total_bytes} bytes).")
    for path in (*report.shard_generations, *report.pending_transactions):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
