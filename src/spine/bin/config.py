"""Command line tools for inspecting resolved SPINE configurations."""

import argparse
import difflib
import sys
from typing import List, Optional, TextIO

import yaml

from spine.config import load_config_file


def resolved_config_yaml(cfg_path: str) -> str:
    """Load a SPINE config file and render the resolved content as YAML."""
    cfg = load_config_file(cfg_path)
    return yaml.safe_dump(
        cfg,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def dump_config(cfg_path: str, output: Optional[str] = None) -> None:
    """Dump a resolved config to stdout or a file."""
    content = resolved_config_yaml(cfg_path)

    if output is None:
        sys.stdout.write(content)
        return

    with open(output, "w", encoding="utf-8") as f:
        f.write(content)


def diff_configs(cfg_path: str, ref_path: str, context: int = 3) -> str:
    """Return a unified diff between two resolved config files."""
    cfg_yaml = resolved_config_yaml(cfg_path)
    ref_yaml = resolved_config_yaml(ref_path)

    return "".join(
        difflib.unified_diff(
            cfg_yaml.splitlines(keepends=True),
            ref_yaml.splitlines(keepends=True),
            fromfile=cfg_path,
            tofile=ref_path,
            n=context,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the config inspection CLI parser."""
    parser = argparse.ArgumentParser(
        description="Inspect fully resolved SPINE configuration files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spine-config dump config.yaml
  spine-config dump config.yaml -o resolved.yaml
  spine-config diff base.yaml changed.yaml
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser(
        "dump", help="Load a SPINE config and dump the resolved YAML."
    )
    dump_parser.add_argument("config", help="Path to the configuration file.")
    dump_parser.add_argument(
        "-o",
        "--output",
        help="Path to write the resolved YAML. Defaults to stdout.",
    )

    diff_parser = subparsers.add_parser(
        "diff", help="Compare two resolved SPINE configs as YAML."
    )
    diff_parser.add_argument("config", help="Path to the first configuration file.")
    diff_parser.add_argument("reference", help="Path to the second configuration file.")
    diff_parser.add_argument(
        "-U",
        "--context",
        type=int,
        default=3,
        help="Number of context lines in the unified diff.",
    )

    return parser


def cli(argv: Optional[List[str]] = None, stream: Optional[TextIO] = None) -> int:
    """Run the config inspection CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = stream or sys.stdout

    if args.command == "dump":
        dump_config(args.config, args.output)
        return 0

    if args.command == "diff":
        diff = diff_configs(args.config, args.reference, args.context)
        stream.write(diff)
        return 1 if diff else 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(cli())
