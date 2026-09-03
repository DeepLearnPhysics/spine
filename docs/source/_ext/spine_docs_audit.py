"""Validate production documentation examples and repository references."""

from __future__ import annotations

import importlib
import re
import shlex
import textwrap
from pathlib import Path
from typing import Any

import yaml
from sphinx.errors import ExtensionError

REQUIRED_GUIDES = {
    "configuration.rst",
    "operations.rst",
    "support.rst",
    "troubleshooting.rst",
    "workflows.rst",
}


class DocumentationLoader(yaml.SafeLoader):
    """Safe YAML loader that preserves arbitrary documentation tags."""


def _construct_tagged(
    loader: DocumentationLoader, _suffix: str, node: yaml.Node
) -> Any:
    """Construct a tagged scalar, sequence, or mapping without side effects."""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


DocumentationLoader.add_multi_constructor("!", _construct_tagged)


def _code_blocks(path: Path, language: str) -> list[tuple[int, str]]:
    """Extract dedented code blocks of one language from an RST document."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[int, str]] = []
    directive = re.compile(rf"^(\s*)\.\. code-block::\s+{re.escape(language)}\s*$")
    index = 0
    while index < len(lines):
        match = directive.match(lines[index])
        if not match:
            index += 1
            continue
        directive_indent = len(match.group(1))
        start = index + 1
        index += 1
        while index < len(lines) and (
            not lines[index].strip() or lines[index].lstrip().startswith(":")
        ):
            index += 1
        body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                body.append("")
                index += 1
                continue
            indentation = len(line) - len(line.lstrip())
            if indentation <= directive_indent:
                break
            body.append(line)
            index += 1
        while body and not body[-1]:
            body.pop()
        if body:
            blocks.append((start + 1, textwrap.dedent("\n".join(body))))
    return blocks


def _validate_yaml(path: Path, content: str, location: str, problems: list[str]) -> Any:
    """Parse YAML content and record a useful error without resolving resources."""
    try:
        return yaml.load(content, Loader=DocumentationLoader)
    except yaml.YAMLError as error:
        problems.append(f"{path}:{location}: invalid YAML: {error}")
        return None


def _long_cli_options(module_name: str) -> set[str]:
    """Collect long options from an executable's authoritative parser."""
    module = importlib.import_module(module_name)
    parsers = [module.build_parser()]
    options: set[str] = set()
    while parsers:
        parser = parsers.pop()
        for action in parser._actions:
            options.update(
                option for option in action.option_strings if option.startswith("--")
            )

            # Subcommand options live on their own parsers rather than the
            # executable's root parser.
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                parsers.extend(
                    choice for choice in choices.values() if hasattr(choice, "_actions")
                )

    return options


def _validate_cli_example(
    path: Path,
    line_number: int,
    block: str,
    options: dict[str, set[str]],
    problems: list[str],
) -> None:
    """Reject unknown long options in documented SPINE shell commands."""
    commands = re.sub(r"\\\s*\n", " ", block).splitlines()
    for command in commands:
        try:
            tokens = shlex.split(command, comments=True)
        except ValueError as error:
            problems.append(f"{path}:{line_number}: invalid shell example: {error}")
            continue
        for executable, accepted in options.items():
            for position, token in enumerate(tokens):
                if token != executable:
                    continue
                for argument in tokens[position + 1 :]:
                    if argument in options:
                        break
                    option = argument.split("=", 1)[0]
                    if option.startswith("--") and option not in accepted:
                        problems.append(
                            f"{path}:{line_number}: unknown {executable} option {option}"
                        )


def audit_production_docs(app, config) -> None:
    """Validate guides, literal examples, and maintained configuration syntax."""
    source_dir = Path(app.srcdir)
    repository = source_dir.parents[1]
    problems: list[str] = []

    missing_guides = sorted(
        guide for guide in REQUIRED_GUIDES if not (source_dir / guide).is_file()
    )
    if missing_guides:
        problems.append("missing production guides: " + ", ".join(missing_guides))

    rst_paths = sorted(
        path
        for path in source_dir.rglob("*.rst")
        if "generated" not in path.relative_to(source_dir).parts
    )
    cli_options = {
        "spine": _long_cli_options("spine.bin.cli"),
        "spine-config": _long_cli_options("spine.bin.config"),
    }
    exact_config_reference = re.compile(r"(?<![\w/{])config/[\w./-]+\.ya?ml")
    for path in rst_paths:
        text = path.read_text(encoding="utf-8")
        for line_number, block in _code_blocks(path, "yaml"):
            _validate_yaml(path, block, str(line_number), problems)
        for line_number, block in _code_blocks(path, "python"):
            try:
                compile(block, f"{path}:{line_number}", "exec")
            except SyntaxError as error:
                problems.append(
                    f"{path}:{line_number}: invalid Python example: {error.msg}"
                )
        for line_number, block in _code_blocks(path, "bash"):
            _validate_cli_example(path, line_number, block, cli_options, problems)
        for match in exact_config_reference.finditer(text):
            target = repository / match.group(0)
            if not target.is_file():
                line = text.count("\n", 0, match.start()) + 1
                problems.append(
                    f"{path}:{line}: missing referenced config {match.group(0)}"
                )

    config_dir = repository / "config"
    for path in sorted((*config_dir.rglob("*.yaml"), *config_dir.rglob("*.yml"))):
        content = path.read_text(encoding="utf-8")
        loaded = _validate_yaml(path, content, "1", problems)
        if not isinstance(loaded, dict):
            problems.append(f"{path}: maintained configuration must be a mapping")
            continue
        includes = loaded.get("include", [])
        if isinstance(includes, str):
            includes = [includes]
        if isinstance(includes, list):
            for include in includes:
                if not isinstance(include, str) or "://" in include:
                    continue
                include_path = Path(include)
                if not include_path.is_absolute():
                    include_path = path.parent / include_path
                if not include_path.is_file():
                    problems.append(f"{path}: missing included config {include}")

    if problems:
        raise ExtensionError(
            "SPINE production documentation audit failed:\n  " + "\n  ".join(problems)
        )


def setup(app):
    """Register the production-documentation audit."""
    app.connect("config-inited", audit_production_docs)
    return {"parallel_read_safe": True}
