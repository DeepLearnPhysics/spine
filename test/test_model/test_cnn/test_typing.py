"""Static contract tests for CNN sparse tensor interfaces."""

import ast
import inspect
import re
from pathlib import Path

CNN_DIR = (
    Path(__file__).resolve().parents[4] / "src" / "spine" / "model" / "layer" / "cnn"
)


def cnn_sources():
    """Yield all maintained CNN source files."""
    yield from sorted(CNN_DIR.glob("*.py"))


def public_definitions(tree):
    """Yield public classes and callables from a parsed module."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("_") or node.name == "__init__":
            yield node


def doc_sections(docstring):
    """Split a NumPy-style docstring into named sections."""
    lines = inspect.cleandoc(docstring).splitlines()
    sections = {}
    for index in range(len(lines) - 1):
        title = lines[index].strip()
        underline = lines[index + 1].strip()
        if title and len(underline) >= 3 and set(underline) == {"-"}:
            sections[title] = index + 2
    return lines, sections


def test_cnn_code_does_not_use_backend_style_tensor_properties():
    for path in cnn_sources():
        tree = ast.parse(path.read_text())
        legacy_attributes = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in {"F", "C"}
        }
        assert not legacy_attributes, f"{path} uses {legacy_attributes}"


def test_cnn_code_does_not_use_ambiguous_batch_data_properties():
    """Require explicit narrowing when extracting data from a TensorBatch."""
    for path in cnn_sources():
        tree = ast.parse(path.read_text())
        ambiguous_attributes = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr == "data":
                ambiguous_attributes.append((node.lineno, node.attr))
            elif node.attr == "tensor":
                is_torch_constructor = (
                    isinstance(node.value, ast.Name) and node.value.id == "torch"
                )
                if not is_torch_constructor:
                    ambiguous_attributes.append((node.lineno, node.attr))

        assert not ambiguous_attributes, (
            f"{path} uses ambiguous batch properties {ambiguous_attributes}; "
            "use `torch_tensor()` or `numpy_tensor()`"
        )


def test_cnn_data_path_methods_do_not_use_any_annotations():
    method_names = {"forward", "encode", "decode"}

    for path in cnn_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in method_names:
                continue

            annotations = [
                argument.annotation
                for argument in (*node.args.posonlyargs, *node.args.args)
                if argument.arg != "self" and argument.annotation is not None
            ]
            if node.args.vararg is not None:
                annotations.append(node.args.vararg.annotation)
            if node.args.kwarg is not None:
                annotations.append(node.args.kwarg.annotation)
            annotations.append(node.returns)

            rendered = {
                ast.unparse(annotation)
                for annotation in annotations
                if annotation is not None
            }
            assert "Any" not in rendered, f"{path}:{node.lineno} uses Any"


def test_cnn_local_variables_use_clear_snake_case_names():
    """Reject mixed-case and unexplained single-letter local variables."""
    allowed_single_letters = {"_", "p", "x"}
    forbidden_attributes = {"bcst", "m", "m1", "m2", "net"}

    for path in cnn_sources():
        tree = ast.parse(path.read_text())
        for function in ast.walk(tree):
            if not isinstance(
                function,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                continue

            violations = []
            for node in ast.walk(function):
                if not isinstance(node, ast.Name) or not isinstance(
                    node.ctx,
                    ast.Store,
                ):
                    continue

                name = node.id
                unclear_single_letter = (
                    len(name) == 1 and name not in allowed_single_letters
                )
                mixed_case = (
                    re.search(r"[A-Z]", name) is not None and not name.isupper()
                )
                if unclear_single_letter or mixed_case:
                    violations.append((node.lineno, name))

            assert not violations, (
                f"{path}:{function.lineno} uses unclear local names " f"{violations}"
            )

        attribute_violations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if not isinstance(node.value, ast.Name) or node.value.id != "self":
                continue

            name = node.attr
            mixed_case = re.search(r"[A-Z]", name) is not None and not name.isupper()
            if len(name) == 1 or name in forbidden_attributes or mixed_case:
                attribute_violations.append((node.lineno, name))

        assert not attribute_violations, (
            f"{path} uses unclear instance attributes " f"{attribute_violations}"
        )


def test_cnn_public_interfaces_have_numpy_style_docstrings():
    """Require documented public APIs with canonical NumPy-style sections."""
    canonical_sections = {
        "Parameters",
        "Other Parameters",
        "Returns",
        "Yields",
        "Receives",
        "Raises",
        "Warns",
        "Warnings",
        "Attributes",
        "Methods",
        "See Also",
        "Notes",
        "References",
        "Examples",
    }

    for path in cnn_sources():
        tree = ast.parse(path.read_text())
        for node in public_definitions(tree):
            docstring = ast.get_docstring(node)
            assert docstring, f"{path}:{node.lineno} has no docstring"

            _, sections = doc_sections(docstring)
            invalid_sections = set(sections) - canonical_sections
            assert not invalid_sections, (
                f"{path}:{node.lineno} uses nonstandard docstring sections "
                f"{sorted(invalid_sections)}"
            )


def test_cnn_callable_parameters_are_documented():
    """Require public callable arguments in NumPy-style parameter sections."""
    for path in cnn_sources():
        tree = ast.parse(path.read_text())
        for node in public_definitions(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            arguments = [
                argument.arg
                for argument in (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
                if argument.arg not in {"self", "cls"}
            ]
            if node.args.vararg is not None:
                arguments.append(node.args.vararg.arg)
            if node.args.kwarg is not None:
                arguments.append(node.args.kwarg.arg)
            if not arguments:
                continue

            lines, sections = doc_sections(ast.get_docstring(node))
            documented = set()
            section_names = ("Parameters", "Other Parameters")
            section_starts = sorted(sections.values())
            for section_name in section_names:
                if section_name not in sections:
                    continue
                start = sections[section_name]
                stop = next(
                    (position - 2 for position in section_starts if position > start),
                    len(lines),
                )
                for line in lines[start:stop]:
                    declaration, separator, _ = line.partition(":")
                    if not separator or line[:1].isspace():
                        continue
                    documented.update(
                        name.strip().lstrip("*") for name in declaration.split(",")
                    )

            missing = set(arguments) - documented
            assert not missing, (
                f"{path}:{node.lineno} does not document parameters "
                f"{sorted(missing)}"
            )
