"""Color palettes and helpers shared across visualization modules."""

from __future__ import annotations

from copy import deepcopy

from plotly import colors

# Colorscale definitions
PLOTLY_COLORS = list(colors.qualitative.Plotly)
PLOTLY_COLORS_TUPLE = colors.convert_colors_to_same_type(
    deepcopy(PLOTLY_COLORS), "tuple"
)[0]
PLOTLY_COLORS_WGRAY = ["#808080"] + PLOTLY_COLORS
HIGH_CONTRAST_COLORS = list(colors.qualitative.Dark24) + list(
    colors.qualitative.Light24
)

__all__ = [
    "PLOTLY_COLORS",
    "PLOTLY_COLORS_TUPLE",
    "PLOTLY_COLORS_WGRAY",
    "HIGH_CONTRAST_COLORS",
    "color_rgba",
]


def color_rgba(color: tuple[int, int, int], alpha: float) -> str:
    """Convert an RGB triplet into a Plotly-compatible RGBA string.

    Parameters
    ----------
    color : Tuple[int, int, int]
        RGB values.
    alpha : float
        Opacity in ``[0, 1]``.

    Returns
    -------
    str
        CSS ``rgba`` color string.
    """
    # Plotly accepts CSS rgba strings rather than raw tuples.
    return f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})"
