"""Hovertext and attribute-formatting helpers for output-object drawers."""

from __future__ import annotations

from typing import Any

import numpy as np

import spine.data.out

__all__ = [
    "dep_tostr",
    "enum_name",
    "format_hover_value",
    "is_depositions",
    "is_long_form",
    "is_sources",
    "src_tostr",
    "tostr",
]


def is_depositions(attr: str) -> bool:
    """Check whether an attribute stores one deposition value per point.

    Parameters
    ----------
    attr : str
        Name of the attribute to inspect.

    Returns
    -------
    bool
        ``True`` if the attribute corresponds to point-wise depositions.
    """
    return attr.startswith("depositions") and not attr.endswith("sum")


def is_sources(attr: str) -> bool:
    """Check whether an attribute stores one source value per point.

    Parameters
    ----------
    attr : str
        Name of the attribute to inspect.

    Returns
    -------
    bool
        ``True`` if the attribute corresponds to point-wise sources.
    """
    return attr.startswith("sources")


def is_long_form(attr: str) -> bool:
    """Check whether an attribute is point-wise rather than object-wise.

    Parameters
    ----------
    attr : str
        Name of the attribute to inspect.

    Returns
    -------
    bool
        ``True`` if the attribute expands to one value per point.
    """
    return is_depositions(attr) or is_sources(attr)


def tostr(attr_name: str, value: Any) -> str:
    """Format a scalar attribute value for Plotly hovertext.

    Parameters
    ----------
    attr_name : str
        Human-readable attribute label.
    value : Any
        Scalar value to render in the hover string.

    Returns
    -------
    str
        HTML fragment suitable for Plotly hovertext.
    """
    return f"<br>{attr_name}: {value}"


def enum_name(obj: Any, attr: str, value: Any) -> str | None:
    """Resolve an enumerated attribute value to its symbolic name.

    Parameters
    ----------
    obj : Any
        Output object that owns the attribute.
    attr : str
        Name of the enumerated attribute.
    value : Any
        Raw encoded value of the attribute.

    Returns
    -------
    Optional[str]
        Enum label if one can be resolved, otherwise ``None``.
    """
    # Truth interactions expose their enum accessors differently from the
    # generic enum_values mapping used by the other output objects.
    if isinstance(obj, spine.data.out.TruthInteraction) and attr in (
        "interaction_type",
        "interaction_mode",
    ):
        enum_value = getattr(obj, f"{attr}_enum")
        if enum_value is not None:
            return enum_value.name
        return None

    if not hasattr(obj, "enum_values") or attr not in obj.enum_values:
        return None

    return obj.enum_values[attr].get(value)


def format_hover_value(obj: Any, attr: str, value: Any) -> Any:
    """Append enum names and documented units to one hover value.

    Parameters
    ----------
    obj : Any
        Output object that owns the attribute.
    attr : str
        Name of the attribute being rendered.
    value : Any
        Raw attribute value.

    Returns
    -------
    Any
        Display-ready hover value, potentially enriched with enum labels and
        units.
    """
    label = enum_name(obj, attr, value)
    if label is not None:
        value = f"{label} ({value})"

    units = obj.field_units.get(attr) if hasattr(obj, "field_units") else None
    if units is not None:
        value = f"{value} {units}"

    return value


def dep_tostr(value: float) -> str:
    """Format one deposition value for point-wise hovertext.

    Parameters
    ----------
    value : float
        Deposition value to display.

    Returns
    -------
    str
        HTML fragment suitable for Plotly hovertext.
    """
    return f"<br>Deposition: {value:0.3f}"


def src_tostr(value: np.ndarray) -> str:
    """Format one source pair for point-wise hovertext.

    Parameters
    ----------
    value : np.ndarray
        ``(2,)`` module and TPC identifier pair.

    Returns
    -------
    str
        HTML fragment suitable for Plotly hovertext.
    """
    return f"<br>Module, TPC: {value[0]:d}, {value[1]:d}"
