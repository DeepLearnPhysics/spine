"""Command line tools for inspecting resolved SPINE configurations."""

import argparse
import difflib
import sys
from typing import List, Optional, TextIO

import yaml

from spine.config import load_config_file
from spine.config.loader import DownloadTag


def represent_download(dumper: yaml.SafeDumper, value: DownloadTag) -> yaml.Node:
    """Represent an unresolved download tag in dumped YAML.

    Parameters
    ----------
    dumper : yaml.SafeDumper
        YAML dumper instance
    value : DownloadTag
        Tagged download value to render

    Returns
    -------
    yaml.Node
        YAML node tagged with `!download`
    """
    if isinstance(value.value, dict):
        return dumper.represent_mapping("!download", value.value)

    return dumper.represent_scalar("!download", str(value.value))


yaml.SafeDumper.add_representer(DownloadTag, represent_download)


def resolved_config_yaml(cfg_path: str, download: bool = False) -> str:
    """Load a SPINE config file and render the resolved content as YAML.

    Parameters
    ----------
    cfg_path : str
        Path to the configuration file
    download : bool, default False
        If `True`, resolve `!download` tags by downloading files. If `False`,
        preserve `!download` tags in the dumped YAML.

    Returns
    -------
    str
        Resolved YAML content
    """
    cfg = load_config_file(cfg_path, download=download)
    return yaml.safe_dump(
        cfg,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def dump_config(
    cfg_path: str, output: Optional[str] = None, download: bool = False
) -> None:
    """Dump a resolved config to stdout or a file.

    Parameters
    ----------
    cfg_path : str
        Path to the configuration file
    output : str, optional
        Path to write the rendered YAML
    download : bool, default False
        If `True`, resolve `!download` tags by downloading files. If `False`,
        preserve `!download` tags in the dumped YAML.

    Returns
    -------
    None
        This function does not return anything
    """
    content = resolved_config_yaml(cfg_path, download)

    if output is None:
        sys.stdout.write(content)
        return

    with open(output, "w", encoding="utf-8") as f:
        f.write(content)


def diff_configs(
    cfg_path: str, ref_path: str, context: int = 3, download: bool = False
) -> str:
    """Return a unified diff between two resolved config files.

    Parameters
    ----------
    cfg_path : str
        Path to the first configuration file
    ref_path : str
        Path to the second configuration file
    context : int, default 3
        Number of context lines to include in the diff
    download : bool, default False
        If `True`, resolve `!download` tags by downloading files. If `False`,
        preserve `!download` tags in the dumped YAML.

    Returns
    -------
    str
        Unified diff between the resolved YAML files
    """
    cfg_yaml = resolved_config_yaml(cfg_path, download)
    ref_yaml = resolved_config_yaml(ref_path, download)

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
    dump_parser.add_argument(
        "--download",
        action="store_true",
        help="Resolve !download tags by downloading files. Defaults to preserving them.",
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
    diff_parser.add_argument(
        "--download",
        action="store_true",
        help="Resolve !download tags by downloading files. Defaults to preserving them.",
    )

    return parser


def cli(argv: Optional[List[str]] = None, stream: Optional[TextIO] = None) -> int:
    """Run the config inspection CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = stream or sys.stdout

    if args.command == "dump":
        dump_config(args.config, args.output, args.download)
        return 0

    if args.command == "diff":
        diff = diff_configs(args.config, args.reference, args.context, args.download)
        stream.write(diff)
        return 1 if diff else 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(cli())
