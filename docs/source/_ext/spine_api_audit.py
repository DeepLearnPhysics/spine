"""Fail documentation builds when public source modules disappear from the API."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from sphinx.errors import ExtensionError

MANUAL_MODULES = {
    "spine.model.uresnet.ppn.model": "api/uresnet_ppn.rst",
    "spine.model.uresnet.ppn.ppn": "api/uresnet_ppn.rst",
    "spine.model.uresnet.ppn.vertex": "api/uresnet_ppn.rst",
}

# These are thin, typed adapters over the selected sparse backend. Their
# signatures are the contract; parameter semantics belong to that backend.
PARAMETER_DOC_EXEMPT_MODULES = {"spine.model.sparse.modules"}


def _module_name(source_root: Path, path: Path) -> str:
    """Convert a source file path to its importable module name."""
    return ".".join(("spine", *path.relative_to(source_root).with_suffix("").parts))


def _is_overload(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function declaration is a typing overload."""
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "overload":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "overload":
            return True
    return False


def _resolve_target(target: str, modules: set[str]) -> str | None:
    """Resolve an API target to the longest source-module prefix."""
    choices = [
        module
        for module in modules
        if target == module or target.startswith(f"{module}.")
    ]
    return max(choices, key=len) if choices else None


def _documented_parameters(*docstrings: str | None) -> set[str]:
    """Extract NumPy-style parameter names from one or more docstrings."""
    names: set[str] = set()
    for docstring in docstrings:
        if not docstring:
            continue
        lines = docstring.splitlines()
        for index, line in enumerate(lines[:-1]):
            if line.strip() != "Parameters":
                continue
            if not re.fullmatch(r"-+", lines[index + 1].strip()):
                continue
            for position in range(index + 2, len(lines)):
                if (
                    position + 1 < len(lines)
                    and lines[position].strip()
                    and re.fullmatch(r"-+", lines[position + 1].strip())
                ):
                    break
                match = re.match(r"^([*A-Za-z_][*A-Za-z0-9_, ]*)\s*:", lines[position])
                if match:
                    names.update(
                        name.strip().lstrip("*") for name in match.group(1).split(",")
                    )
            break
    return names


def _documented_modules(source_dir: Path, modules: set[str]) -> set[str]:
    """Collect source modules referenced by hand-written API sources."""
    documented: set[str] = set()
    rst_paths = [
        path
        for path in source_dir.rglob("*.rst")
        if "generated" not in path.relative_to(source_dir).parts
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in rst_paths)
    for module in modules:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(module)}(?:\.|\b)", combined):
            documented.add(module)

    for path in rst_paths:
        current_module: str | None = None
        in_autosummary = False
        for line in path.read_text(encoding="utf-8").splitlines():
            current_match = re.match(r"\.\. currentmodule::\s+(\S+)", line)
            if current_match:
                current_module = current_match.group(1)

            directive = re.match(r"\.\. auto(?:module|class|function)::\s+(\S+)", line)
            if directive:
                target = directive.group(1)
                if not target.startswith("spine.") and current_module:
                    target = f"{current_module}.{target}"
                resolved = _resolve_target(target, modules)
                if resolved:
                    documented.add(resolved)

            if line.startswith(".. autosummary::"):
                in_autosummary = True
                continue
            if not in_autosummary:
                continue
            if not line.strip():
                continue
            if not line.startswith(" "):
                in_autosummary = False
                continue
            item = line.strip()
            if item.startswith(":") or item.startswith(".."):
                continue
            target = item
            if not target.startswith("spine.") and current_module:
                target = f"{current_module}.{target}"
            resolved = _resolve_target(target, modules)
            if resolved:
                documented.add(resolved)

    return documented


def audit_api_docs(app, config) -> None:
    """Validate module reachability and baseline public-docstring coverage."""
    source_dir = Path(app.srcdir)
    repository = source_dir.parents[1]
    source_root = repository / "src" / "spine"
    paths = sorted(
        path for path in source_root.rglob("*.py") if path.name != "__init__.py"
    )
    modules = {_module_name(source_root, path) for path in paths}
    documented = _documented_modules(source_dir, modules)

    missing_manual_pages = [
        f"{module} ({page})"
        for module, page in MANUAL_MODULES.items()
        if not (source_dir / page).is_file()
    ]
    uncovered = sorted(modules - documented - MANUAL_MODULES.keys())

    missing_docstrings: list[str] = []
    missing_parameters: list[str] = []
    mismatched_function_parameters: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = _module_name(source_root, path)
        if ast.get_docstring(tree) is None:
            missing_docstrings.append(module)
        for node in tree.body:
            if not isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            if node.name.startswith("_"):
                continue
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and _is_overload(node):
                continue
            if ast.get_docstring(node) is None:
                missing_docstrings.append(f"{module}.{node.name}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _is_overload(node):
                    continue
                arguments = [
                    argument.arg
                    for argument in (
                        node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    )
                ]
                if node.args.vararg:
                    arguments.append(node.args.vararg.arg)
                if node.args.kwarg:
                    arguments.append(node.args.kwarg.arg)
                documented_parameters = _documented_parameters(ast.get_docstring(node))
                absent = [
                    argument
                    for argument in arguments
                    if argument not in documented_parameters
                ]
                surplus = sorted(documented_parameters - set(arguments))
                details = []
                if absent:
                    details.append(f"missing {', '.join(absent)}")
                if surplus:
                    details.append(f"unknown {', '.join(surplus)}")
                if details:
                    mismatched_function_parameters.append(
                        f"{module}.{node.name}: {'; '.join(details)}"
                    )
            if not isinstance(node, ast.ClassDef):
                continue
            initializer = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == "__init__"
                ),
                None,
            )
            if initializer is None or module in PARAMETER_DOC_EXEMPT_MODULES:
                continue
            arguments = [
                argument.arg
                for argument in (
                    initializer.args.posonlyargs
                    + initializer.args.args
                    + initializer.args.kwonlyargs
                )
                if argument.arg not in {"self", "cls"}
            ]
            if initializer.args.vararg:
                arguments.append(initializer.args.vararg.arg)
            if initializer.args.kwarg:
                arguments.append(initializer.args.kwarg.arg)
            documented_parameters = _documented_parameters(
                ast.get_docstring(node), ast.get_docstring(initializer)
            )
            absent = [
                argument
                for argument in arguments
                if argument not in documented_parameters
            ]
            if absent:
                missing_parameters.append(f"{module}.{node.name}: {', '.join(absent)}")

    problems = []
    if uncovered:
        problems.append(
            "source modules absent from the API:\n  " + "\n  ".join(uncovered)
        )
    if missing_manual_pages:
        problems.append(
            "manual optional-runtime pages are missing:\n  "
            + "\n  ".join(missing_manual_pages)
        )
    if missing_docstrings:
        problems.append(
            "public modules or top-level symbols lack docstrings:\n  "
            + "\n  ".join(missing_docstrings)
        )
    if missing_parameters:
        problems.append(
            "public constructor parameters lack descriptions:\n  "
            + "\n  ".join(missing_parameters)
        )
    if mismatched_function_parameters:
        problems.append(
            "public function parameter documentation does not match signatures:\n  "
            + "\n  ".join(mismatched_function_parameters)
        )
    if problems:
        raise ExtensionError("SPINE API coverage audit failed:\n" + "\n".join(problems))


def setup(app):
    """Register the API coverage audit."""
    app.connect("config-inited", audit_api_docs)
    return {"parallel_read_safe": True}
