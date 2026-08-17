"""Color and hovertext builders for output-object visualization."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import out as out_data
from spine.geo import Geometry

from ...layout import HIGH_CONTRAST_COLORS, PLOTLY_COLORS_WGRAY
from .formatting import (
    dep_tostr,
    format_hover_value,
    is_depositions,
    is_long_form,
    is_sources,
    src_tostr,
    tostr,
)

__all__ = ["build_object_colors", "colorable_attributes", "object_color_kind"]


def object_color_kind(obj_cls: type, attr: str) -> str | None:
    """Return the schema-driven color strategy for an object attribute.

    Parameters
    ----------
    obj_cls : type
        SPINE data class which owns the attribute.
    attr : str
        Attribute name to classify.

    Returns
    -------
    str, optional
        ``"continuous"`` or ``"discrete"`` when the attribute can color
        rendered points, otherwise ``None``.
    """
    metadata = obj_cls.attr_metadata(attr)
    attr_type = obj_cls.attr_type(attr)

    # Positions and vectors cannot supply one scalar color per displayed point.
    if metadata.position or metadata.vector:
        return None

    # Point-wise fields use their declared storage and reference semantics.
    if metadata.pointwise:
        if metadata.reference or metadata.categorical or metadata.enum:
            return "discrete"
        return "continuous" if metadata.dtype is not None else None

    # Object-level colors are limited to scalar numeric and boolean fields.
    if attr_type not in (int, float, bool):
        return None
    if attr_type is bool or metadata.enum or metadata.reference:
        return "discrete"
    if metadata.categorical:
        return "discrete"
    return "continuous"


def colorable_attributes(obj_cls: type) -> tuple[str, ...]:
    """Return attributes which can provide one color value per drawn point.

    Parameters
    ----------
    obj_cls : type
        SPINE data class whose schema should be inspected.

    Returns
    -------
    tuple[str, ...]
        Attribute names with a continuous or discrete color strategy.
    """
    return tuple(
        attr for attr in obj_cls.attr_names() if object_color_kind(obj_cls, attr)
    )


def _object_class(objects: Any, obj_name: str) -> type:
    """Resolve the element class of an object collection.

    Parameters
    ----------
    objects : sequence
        Populated or typed object collection.
    obj_name : str
        Collection name used to infer a class when the collection is empty.

    Returns
    -------
    type
        Concrete output-object class represented by the collection.
    """
    if len(objects) > 0:
        return type(objects[0])
    prefix, obj_type = obj_name.split("_", 1)
    cls_name = f"{prefix.capitalize()}{obj_type[:-1].capitalize()}"
    return getattr(out_data, cls_name)


def _continuous_range(values: Any) -> tuple[float, float]:
    """Return a finite continuous color range for scalar values.

    Parameters
    ----------
    values : array-like
        Values from which to determine the finite range.

    Returns
    -------
    tuple[float, float]
        Non-degenerate lower and upper color limits.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if len(finite) == 0:
        return 0.0, 1.0
    cmin, cmax = float(np.min(finite)), float(np.max(finite))
    if cmin == cmax:
        cmax = cmin + 1.0
    return cmin, cmax


def _discrete_colors(
    values: Any,
) -> tuple[np.ndarray, list[str] | None, int, int]:
    """Encode categorical values as consecutive color indexes.

    Parameters
    ----------
    values : array-like
        Categorical values to encode.

    Returns
    -------
    np.ndarray
        Integer color index for each input value.
    list[str] or None
        Discrete color scale, or ``None`` for an empty input.
    int
        Lower color-scale limit.
    int
        Upper color-scale limit.
    """
    # Map arbitrary values onto compact indexes before selecting palette entries.
    unique, encoded = np.unique(values, return_inverse=True)
    count = len(unique)
    colorscale = HIGH_CONTRAST_COLORS

    # Plotly requires special handling for empty and single-valued scales.
    if count == 0:
        colorscale = None
    elif count == 1:
        colorscale = [colorscale[0]] * 2
    elif count <= len(colorscale):
        colorscale = colorscale[:count]
    else:
        repeat = (count - 1) // len(colorscale) + 1
        colorscale = np.tile(colorscale, repeat)[:count].tolist()
    return encoded, colorscale, 0, max(count - 1, 0)


def build_object_colors(
    *,
    data: dict[str, Any],
    obj_name: str,
    attrs: list[str],
    color_attr: str | None,
    split_traces: bool,
    geo: Geometry | None,
    lite: bool,
    truth_point_key: str,
    truth_point_mode: str,
    dep_modes: dict[str, str],
) -> dict[str, Any]:
    """Provide colors, colorscale, and hovertext for one object collection.

    Parameters
    ----------
    data : Dict[str, Any]
        Full event dictionary containing the object collection and supporting
        point, deposition, or source arrays.
    obj_name : str
        Name of the object collection to visualize.
    attrs : List[str]
        Object attributes requested for hovertext generation.
    color_attr : str, optional
        Attribute used to define the marker or cluster colors.
    split_traces : bool
        If ``True``, produce per-object trace labels rather than one shared
        group label.
    geo : Geometry, optional
        Geometry helper used to decode TPC source labels.
    lite : bool
        If ``True``, object point clouds are not available and long-form
        attributes are therefore invalid.
    truth_point_key : str
        Data key used to retrieve the truth point cloud currently being drawn.
    truth_point_mode : str
        Truth point mode used to validate compatible deposition and source
        attributes.
    dep_modes : Dict[str, str]
        Mapping from truth deposition display modes to the backing deposition
        arrays in ``data``.

    Returns
    -------
    Dict[str, Any]
        Color configuration ready to forward to the trace-building helpers.
    """
    # Use singular legend labels when one trace is emitted per object.
    name = " ".join(obj_name.split("_")).capitalize()
    if split_traces:
        name = name[:-1]

    # Keep the color dimension among the hover attributes when it is requested
    # explicitly, especially for long-form attributes.
    if color_attr is not None and color_attr not in attrs:
        raise ValueError(
            "The attribute used to define the color scale must be "
            "included in the list of hovertext attributes."
        )

    # Source-valued attributes need geometry to decode module/TPC information.
    needs_geo = any(is_sources(attr) for attr in attrs or []) or (
        color_attr is not None and is_sources(color_attr)
    )
    if needs_geo and geo is None:
        raise ValueError(
            "Provide detector name/geometry if the TPC sources are to be displayed."
        )
    source_geo = geo if needs_geo else None

    # Start from one hover label per object.
    obj_type = obj_name.split("_")[-1][:-1].capitalize()
    count = len(data[obj_name])
    hovertext = [[f"{obj_type} {i}"] for i in range(count)]
    has_long_form = False
    if attrs is not None and any(is_long_form(attr) for attr in attrs):
        # Long-form attributes contribute one value per point, so expand the
        # base object labels to the displayed point cloud length.
        if lite:
            raise ValueError(
                "Long-form attributes are not available when drawing "
                "lite-version objects."
            )
        point_key = truth_point_key if "truth" in obj_name else "points"
        hovertext = [ht * len(data[point_key]) for ht in hovertext]
        has_long_form = True

    # Default to coloring by object id when no attribute is specified.
    color = np.arange(len(data[obj_name]))
    if color_attr is None:
        color_attr = "id"

    if attrs is not None:
        for attr in attrs:
            attr_name = " ".join(attr.split("_")).capitalize()

            # Keep truth point, deposition, and source modes aligned with the
            # point cloud that is actually being drawn.
            if "truth" in obj_name:
                if is_depositions(attr):
                    prefix = truth_point_mode.replace("points", "depositions")
                    if not attr.startswith(prefix):
                        raise ValueError(
                            f"Points mode {truth_point_mode} and deposition "
                            f"mode {attr} are incompatible."
                        )
                if is_sources(attr):
                    ref_name = truth_point_mode.replace("points", "sources")
                    if attr != ref_name:
                        raise ValueError(
                            f"Points mode {truth_point_mode} and source "
                            f"mode {attr} are incompatible."
                        )

            values = [getattr(obj, attr) for obj in data[obj_name]]
            if is_sources(attr):
                assert source_geo is not None
                values = [source_geo.get_sources(value) for value in values]

            # Reuse the same attribute walk when the hover attribute also sets
            # the color dimension.
            if attr == color_attr:
                if not is_sources(attr):
                    color = values
                else:
                    assert source_geo is not None
                    color = [source_geo.get_chambers(value) for value in values]

            if is_depositions(attr):
                for i, ht in enumerate(hovertext):
                    hovertext[i] = [
                        ht[j] + dep_tostr(value) for j, value in enumerate(values[i])
                    ]
            elif is_sources(attr):
                for i, ht in enumerate(hovertext):
                    hovertext[i] = [
                        ht[j] + src_tostr(value) for j, value in enumerate(values[i])
                    ]
            else:
                for i, ht in enumerate(hovertext):
                    val_str = tostr(
                        attr_name,
                        format_hover_value(data[obj_name][i], attr, values[i]),
                    )
                    hovertext[i] = [ht[j] + val_str for j in range(len(hovertext[i]))]

    if not has_long_form:
        hovertext = [ht[0] for ht in hovertext]

    obj_cls = _object_class(data[obj_name], obj_name)
    color_kind = object_color_kind(obj_cls, color_attr)
    if color_kind is None:
        raise ValueError(f"Color attribute not supported: {color_attr}.")

    # Choose the colorscale strategy from the declared field schema.
    if is_depositions(color_attr):
        # Depositions use a continuous colorscale.
        dep_mode = dep_modes[color_attr] if "truth" in obj_name else "depositions"
        colorscale = "Inferno"
        cmin = 0.0
        cmax = 2 * np.median(data[dep_mode])

    elif is_sources(color_attr):
        # Sources use one discrete color per TPC chamber.
        assert source_geo is not None
        count = source_geo.tpc.num_chambers
        colorscale = HIGH_CONTRAST_COLORS
        if count == 0:
            colorscale = None
        elif count == 1:
            colorscale = [colorscale[0]] * 2
        elif count <= len(colorscale):
            colorscale = colorscale[:count]
        else:
            repeat = (count - 1) // len(colorscale) + 1
            colorscale = np.tile(colorscale, repeat)[:count]
        cmin = 0
        cmax = count - 1

    elif obj_cls.attr_metadata(color_attr).enum is not None:
        # Shape and PID use a discrete colorscale with known cardinality.
        enum = obj_cls.attr_metadata(color_attr).enum
        num_classes = len(enum)
        colorscale = PLOTLY_COLORS_WGRAY[: num_classes + 1]
        cmin = -1
        cmax = num_classes - 1

    elif obj_cls.attr_type(color_attr) is bool:
        # Boolean attributes use a two-color discrete colorscale.
        color = np.array(color, dtype=np.int32)
        colorscale = PLOTLY_COLORS_WGRAY[1:3]
        cmin = 0
        cmax = 1

    elif color_kind == "continuous":
        # Numeric scalar attributes use a continuous colorscale.
        colorscale = "Inferno"
        cmin, cmax = _continuous_range(color)

    elif color_kind == "discrete":
        color, colorscale, cmin, cmax = _discrete_colors(color)

    else:  # pragma: no cover
        # Defensive guard for future color strategies; current classification
        # returns only the two handled values after the ``None`` check above.
        raise RuntimeError(f"Unknown color strategy: {color_kind}.")

    return {
        "color": color,
        "hovertext": hovertext,
        "name": name,
        "colorscale": colorscale,
        "cmin": cmin,
        "cmax": cmax,
    }
