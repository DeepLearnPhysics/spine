"""Sphinx directive for readable, source-driven dataclass API pages."""

from __future__ import annotations

import inspect
import math
import pydoc
from collections import OrderedDict
from dataclasses import MISSING, Field, fields, is_dataclass
from types import NoneType, UnionType
from typing import Any, get_args, get_origin

from docutils import nodes
from docutils.statemachine import StringList
from numpydoc.docscrape import NumpyDocString
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles


def _load_object(fullname: str) -> type:
    """Resolve and validate a fully qualified dataclass name."""
    obj = pydoc.locate(fullname)
    if not isinstance(obj, type):
        raise TypeError(f"{fullname!r} does not resolve to a class.")
    if not is_dataclass(obj):
        raise TypeError(f"{fullname!r} is not a dataclass.")
    return obj


def _numpy_doc(obj: Any) -> NumpyDocString:
    """Parse an object's own NumPy-style docstring."""
    docstring = obj.__dict__.get("__doc__", "") if isinstance(obj, type) else None
    if docstring is None:
        docstring = inspect.getdoc(obj) or ""
    return NumpyDocString(inspect.cleandoc(docstring or ""))


def _attribute_docs(cls: type) -> dict[str, str]:
    """Return field descriptions declared by a class docstring."""
    descriptions: dict[str, str] = {}
    parsed = _numpy_doc(cls)
    for section in ("Attributes", "Parameters"):
        for names, _type_name, description in parsed[section]:
            for name in names.split(","):
                # Tolerate historical ``name: type`` spelling in docstrings.
                field_name = name.strip().split(":", maxsplit=1)[0]
                descriptions.setdefault(
                    field_name, " ".join(line.strip() for line in description)
                )
    return descriptions


def _summary(obj: Any) -> str:
    """Return a compact summary without parameter/attribute sections."""
    parsed = _numpy_doc(obj)
    lines = [*parsed["Summary"], *parsed["Extended Summary"]]
    return " ".join(line.strip() for line in lines if line.strip())


def _type_name(annotation: Any) -> str:
    """Render an annotation as a compact, stable type name."""
    if annotation is Any:
        result = "Any"
    elif annotation is None or annotation is NoneType:
        result = "None"
    elif (origin := get_origin(annotation)) is not None:
        arguments = get_args(annotation)
        if origin in (UnionType,):
            result = " | ".join(_type_name(argument) for argument in arguments)
        else:
            origin_name = getattr(
                origin, "__name__", str(origin).replace("typing.", "")
            )
            result = f"{origin_name}[{', '.join(_type_name(arg) for arg in arguments)}]"
    elif isinstance(annotation, UnionType):
        result = " | ".join(_type_name(argument) for argument in get_args(annotation))
    elif isinstance(annotation, str):
        result = annotation
    elif hasattr(annotation, "__module__") and hasattr(annotation, "__qualname__"):
        module = annotation.__module__
        prefix = "" if module in ("builtins", "numpy") else f"{module}."
        numpy_prefix = "np." if module == "numpy" else prefix
        result = f"{numpy_prefix}{annotation.__qualname__}"
    else:
        result = str(annotation).replace("typing.", "")

    return result


def _default(field: Field[Any]) -> str:
    """Describe a field default without evaluating default factories."""
    if field.default is not MISSING:
        value = field.default
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return repr(value)
    if field.default_factory is not MISSING:
        return "factory"
    return "required"


def _metadata(field: Field[Any]) -> str:
    """Format the user-relevant parts of SPINE field metadata."""
    metadata = dict(field.metadata)
    parts: list[str] = []

    if "length" in metadata:
        parts.append(f"length {metadata['length']}")
    if "dtype" in metadata:
        parts.append(f"dtype {_type_name(metadata['dtype'])}")
    if "units" in metadata:
        parts.append(f"units {metadata['units']}")
    if "enum" in metadata:
        parts.append(f"enum {_type_name(metadata['enum'])}")
    if "reference" in metadata:
        reference = str(metadata["reference"])
        space = metadata.get("reference_space")
        parts.append(f"{space + ' ' if space else ''}{reference} reference")

    labels = {
        "index": "index",
        "position": "position",
        "vector": "vector",
        "categorical": "categorical",
        "pointwise": "point-wise",
        "cat": "concatenated",
        "skip": "not serialized",
        "lite_skip": "omitted from lite output",
    }
    parts.extend(label for key, label in labels.items() if metadata.get(key))
    return "; ".join(parts)


def _field_groups(cls: type) -> list[tuple[type, list[Field[Any]]]]:
    """Group effective dataclass fields by their nearest declaring class."""
    grouped: OrderedDict[type, list[Field[Any]]] = OrderedDict(
        (owner, []) for owner in cls.__mro__ if owner is not object
    )
    for field in fields(cls):
        owner = next(
            candidate
            for candidate in cls.__mro__
            if field.name in candidate.__dict__.get("__annotations__", {})
        )
        grouped[owner].append(field)
    return [(owner, values) for owner, values in grouped.items() if values]


def _property_groups(cls: type) -> list[tuple[type, list[tuple[str, property]]]]:
    """Group effective public properties by their nearest declaring class."""
    groups: list[tuple[type, list[tuple[str, property]]]] = []
    seen: set[str] = set()
    field_names = {field.name for field in fields(cls)}
    for owner in cls.__mro__:
        values: list[tuple[str, property]] = []
        for name, value in owner.__dict__.items():
            if (
                name not in seen
                and name not in field_names
                and not name.startswith("_")
                and isinstance(value, property)
            ):
                values.append((name, value))
                seen.add(name)
        if values:
            groups.append((owner, values))
    return groups


def _method_groups(cls: type) -> list[tuple[type, list[tuple[str, Any]]]]:
    """Group effective public methods by their nearest declaring class."""
    groups: list[tuple[type, list[tuple[str, Any]]]] = []
    seen: set[str] = set()
    for owner in cls.__mro__:
        values: list[tuple[str, Any]] = []
        for name, value in owner.__dict__.items():
            member = (
                value.__func__
                if isinstance(value, (classmethod, staticmethod))
                else value
            )
            if name not in seen and not name.startswith("_") and callable(member):
                values.append((name, member))
                seen.add(name)
        if values:
            groups.append((owner, values))
    return groups


def _owner_label(root: type, owner: type) -> str:
    """Describe whether a member is declared locally or inherited."""
    relation = "Declared by" if owner is root else "Inherited from"
    return f"**{relation}** ``{owner.__name__}``"


def _append_table(
    lines: list[str], headers: tuple[str, ...], rows: list[tuple[str, ...]]
) -> None:
    """Append a list-table to generated reStructuredText."""
    widths = " ".join("1" for _ in headers)
    lines.extend(
        [
            ".. list-table::",
            f"   :widths: {widths}",
            "   :header-rows: 1",
            "   :class: dataclass-members",
            "",
        ]
    )
    lines.append(f"   * - {headers[0]}")
    for header in headers[1:]:
        lines.append(f"     - {header}")
    for row in rows:
        lines.append(f"   * - {row[0]}")
        for value in row[1:]:
            lines.append(f"     - {value}")
    lines.append("")


def _stored_field_lines(cls: type) -> list[str]:
    """Build the stored-field section for a dataclass."""
    lines = [
        "Stored fields",
        "-------------",
        "",
        "Fields are grouped by the class that declares the effective "
        "annotation. Defaults named ``factory`` are created independently "
        "for each instance.",
        "",
    ]
    for owner, owner_fields in _field_groups(cls):
        descriptions = _attribute_docs(owner)
        lines.extend([_owner_label(cls, owner), ""])
        rows: list[tuple[str, str, str]] = []
        for field in owner_fields:
            details = descriptions.get(field.name, "—")
            metadata = _metadata(field)
            if metadata:
                details = f"{details} **Metadata:** {metadata}."
            type_default = (
                f"``{_type_name(field.type)}``; default ``{_default(field)}``"
            )
            rows.append((f"``{field.name}``", type_default, details))
        _append_table(lines, ("Field", "Type and default", "Description"), rows)

    return lines


def _property_lines(cls: type) -> list[str]:
    """Build the computed-property section for a dataclass."""
    groups = _property_groups(cls)
    if not groups:
        return []

    lines = ["Computed properties", "-------------------", ""]
    for owner, properties in groups:
        lines.extend([_owner_label(cls, owner), ""])
        rows: list[tuple[str, str, str]] = []
        for name, value in properties:
            annotation = inspect.signature(value.fget).return_annotation
            if annotation is inspect.Signature.empty:
                annotation = Any
            rows.append(
                (
                    f"``{name}``",
                    f"``{_type_name(annotation)}``",
                    _summary(value.fget) or "—",
                )
            )
        _append_table(lines, ("Property", "Type", "Description"), rows)

    return lines


def _method_lines(cls: type, class_name: str) -> list[str]:
    """Build the public-method section for a dataclass."""
    groups = _method_groups(cls)
    if not groups:
        return []

    lines = ["Methods", "-------", ""]
    for owner, methods in groups:
        lines.extend([_owner_label(cls, owner), ""])
        for name, _method in methods:
            lines.extend([f".. automethod:: {class_name}.{name}", ""])

    return lines


class SpineDataclassDirective(SphinxDirective):
    """Render dataclass fields, properties and methods by declaration owner."""

    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        """Generate and parse a structured dataclass reference page."""
        fullname = self.arguments[0].strip()
        cls = _load_object(fullname)
        module_name, class_name = fullname.rsplit(".", maxsplit=1)
        lines = [
            f".. currentmodule:: {module_name}",
            "",
            f".. py:class:: {class_name}",
            "",
        ]

        summary = _summary(cls)
        if summary:
            lines.extend([f"   {summary}", ""])

        hierarchy = [base.__name__ for base in cls.__mro__ if base is not object]
        if len(hierarchy) > 1:
            lines.extend(
                [f"**Method resolution order:** ``{' → '.join(hierarchy)}``", ""]
            )

        lines.extend(_stored_field_lines(cls))
        lines.extend(_property_lines(cls))
        lines.extend(_method_lines(cls, class_name))

        content = StringList(lines, source=self.get_source_info()[0])
        container = nodes.section()
        nested_parse_with_titles(self.state, content, container)
        return list(container.children)


def setup(app):
    """Register the SPINE dataclass directive."""
    app.add_directive("spine-dataclass", SpineDataclassDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
