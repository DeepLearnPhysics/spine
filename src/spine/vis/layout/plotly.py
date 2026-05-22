"""Plotly layout helpers for 3D detector and comparison displays."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from plotly import graph_objs as go
from plotly.subplots import make_subplots

from spine.data import ImageMeta3D
from spine.geo import GeoManager, Geometry

__all__ = ["dual_figure3d", "layout3d"]


def layout3d(
    ranges: np.ndarray | None = None,
    meta: ImageMeta3D | None = None,
    geo: Geometry | None = None,
    use_geo: bool = False,
    show_optical: bool = False,
    show_crt: bool = False,
    detector_padding: float = 0.1,
    titles: list[str] | None = None,
    detector_coords: bool = False,
    backgroundcolor: str = "white",
    gridcolor: str = "lightgray",
    width: int = 800,
    height: int = 800,
    showlegend: bool = True,
    camera: dict[str, Any] | None = None,
    aspectmode: str = "manual",
    aspectratio: dict[str, float] | None = None,
    dark: bool = False,
    margin: dict[str, int] | None = None,
    hoverlabel: dict[str, Any] | None = None,
    **kwargs: Any,
) -> go.Layout:
    """Build a Plotly 3D layout from point ranges, metadata, or geometry.

    Parameters
    ----------
    ranges : np.ndarray, optional
        Either explicit ``(3, 2)`` axis ranges or a point cloud with shape
        ``(N, 3)`` used to infer those ranges.
    meta : Any, optional
        Metadata object used to infer image bounds or convert geometry bounds
        into pixel coordinates.
    geo : Any, optional
        Geometry object used to infer detector boundaries.
    use_geo : bool, default False
        If ``True``, use the globally configured geometry manager when ``geo``
        is not provided explicitly.
    show_optical : bool, default False
        If ``True``, include optical detector extents in the geometry bounds.
    show_crt : bool, default False
        If ``True``, include CRT extents in the geometry bounds.
    detector_padding : float, default 0.1
        Fractional padding added around detector geometry bounds.
    titles : List[str], optional
        Axis titles in ``x``, ``y``, ``z`` order.
    detector_coords : bool, default False
        If ``True``, treat all coordinates as detector coordinates rather than
        pixel coordinates.
    backgroundcolor : str, default ``"white"``
        Plot background color.
    gridcolor : str, default ``"lightgray"``
        Axis grid color.
    width : int, default 800
        Figure width in pixels.
    height : int, default 800
        Figure height in pixels.
    showlegend : bool, default True
        Whether to display the legend.
    camera : Dict[str, Any], optional
        Explicit Plotly camera specification.
    aspectmode : str, default ``"manual"``
        Plotly aspect mode.
    aspectratio : Dict[str, float], optional
        Explicit aspect ratio for the three axes.
    dark : bool, default False
        If ``True``, use a dark Plotly theme.
    margin : Dict[str, int], optional
        Plot margins.
    hoverlabel : Dict[str, Any], optional
        Hoverlabel style dictionary.
    **kwargs : Any
        Additional keyword arguments forwarded to :class:`plotly.graph_objs.Layout`.

    Returns
    -------
    go.Layout
        Plotly layout object ready to attach to a figure.
    """
    # Resolve the displayed coordinate range either from explicit bounds or by
    # inferring it from a supplied point cloud.
    if ranges is None:
        ranges = np.asarray([None, None, None])
    else:
        if ranges.shape != (3, 2):
            if len(ranges.shape) != 2 or ranges.shape[1] != 3:
                raise ValueError(
                    "If ranges is not of shape (3, 2), it must be of shape (N, 3)."
                )
            ranges = np.vstack([np.min(ranges, axis=0), np.max(ranges, axis=0)]).T

        if not np.all(ranges[:, 1] >= ranges[:, 0]):
            raise ValueError(
                "Each range upper bound must be greater than its lower bound."
            )

    if use_geo or geo is not None:
        # Geometry-driven bounds take precedence over explicit point clouds.
        if ranges is not None and None not in ranges:
            raise ValueError(
                "Should not pass geo or ask to `use_geo=True` along with `ranges`."
            )
        if geo is None:
            geo = GeoManager.get_instance()
        ranges = geo.get_boundaries(with_optical=show_optical, with_crt=show_crt)
        lengths = ranges[:, 1] - ranges[:, 0]
        ranges[:, 0] -= lengths * detector_padding
        ranges[:, 1] += lengths * detector_padding

        if detector_coords is False:
            if meta is None:
                raise ValueError(
                    "Must provide metadata information to convert the detector "
                    "coordinates to pixel coordinates."
                )
            ranges = meta.to_px(ranges.T).T

    elif meta is not None:
        # Metadata without geometry falls back to full image extents.
        if ranges is not None and None not in ranges:
            raise ValueError("Should not specify both `ranges` and `meta` parameters.")
        if detector_coords:
            ranges = np.vstack([meta.lower, meta.upper]).T
        else:
            ranges = np.vstack(
                [[0, 0, 0], np.round((meta.upper - meta.lower) / meta.size)]
            ).T

    if camera is None:
        # Use the detector-style default camera unless the caller overrides it.
        camera = {
            "eye": {"x": -2.0, "y": 1.0, "z": -0.01},
            "up": {"x": 0.0, "y": 1.0, "z": 0.0},
            "center": {"x": 0.0, "y": -0.1, "z": -0.01},
        }

    if aspectmode == "manual" and aspectratio is None:
        # Infer aspect ratios from the displayed range to avoid distorted
        # detector views when the caller does not specify one explicitly.
        axes = ["x", "y", "z"]
        ratios = np.ones(len(ranges))
        if ranges is not None and ranges[0] is not None:
            ranges_arr = np.array(ranges)
            max_range = np.max(ranges_arr[:, 1] - ranges_arr[:, 0])
            ratios = (ranges_arr[:, 1] - ranges_arr[:, 0]) / max_range
        aspectratio = {axes[i]: value for i, value in enumerate(ratios)}

    if titles is not None and len(titles) != 3:
        raise ValueError("Must specify one title per axis")
    if titles is None:
        unit = "cm" if detector_coords else "pixel"
        titles = [f"x [{unit}]", f"y [{unit}]", f"z [{unit}]"]

    if "legend" not in kwargs:
        kwargs["legend"] = {
            "title": "Legend",
            "tracegroupgap": 1,
            "itemsizing": "constant",
        }

    if dark:
        kwargs["template"] = "plotly_dark"
        kwargs["paper_bgcolor"] = "black"
        backgroundcolor = "black"

    if margin is None:
        margin = {"b": 0, "t": 0, "l": 0, "r": 0}

    if hoverlabel is None:
        hoverlabel = {"font_family": "Droid sans, monospace"}

    axis_base = {
        "nticks": 10,
        "showticklabels": True,
        "tickfont": {"size": 14},
        "backgroundcolor": backgroundcolor,
        "gridcolor": gridcolor,
        "showbackground": True,
    }

    scene = {"aspectmode": aspectmode, "aspectratio": aspectratio, "camera": camera}
    for i, axis in enumerate(("x", "y", "z")):
        scene[f"{axis}axis"] = {
            "title": {"text": titles[i], "font": {"size": 20}},
            "range": ranges[i],
            **axis_base,
        }

    return go.Layout(
        showlegend=showlegend,
        width=width,
        height=height,
        margin=margin,
        scene1=scene,
        scene2=deepcopy(scene),
        scene3=deepcopy(scene),
        hoverlabel=hoverlabel,
        **kwargs,
    )


def dual_figure3d(
    traces_left: list[object],
    traces_right: list[object],
    layout: go.Layout | None = None,
    titles: list[str] | None = None,
    width: int = 1500,
    height: int = 750,
    synchronize: bool = False,
    margin: dict[str, int] | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Create a side-by-side pair of 3D Plotly scenes.

    Parameters
    ----------
    traces_left : List[object]
        Plotly traces to draw in the left subplot.
    traces_right : List[object]
        Plotly traces to draw in the right subplot.
    layout : go.Layout, optional
        Base layout to apply to the figure.
    titles : List[str], optional
        Subplot titles for the left and right scenes.
    width : int, default 1500
        Figure width in pixels.
    height : int, default 750
        Figure height in pixels.
    synchronize : bool, default False
        If ``True``, keep the two subplot cameras synchronized.
    margin : Dict[str, int], optional
        Figure margins.
    **kwargs : Any
        Additional keyword arguments used only when a default layout must be
        created internally.

    Returns
    -------
    go.Figure
        Plotly figure containing the two 3D scenes.
    """
    # Check that the titles are not provided or there is two of them.
    if titles is not None and len(titles) != 2:
        raise ValueError("Must specify two titles, one for each subplot.")

    # Build the two-scene subplot shell and add the caller-provided traces.
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=titles,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        horizontal_spacing=0.05,
        vertical_spacing=0.04,
    )

    num_left, num_right = len(traces_left), len(traces_right)
    fig.add_traces(traces_left, rows=[1] * num_left, cols=[1] * num_left)
    fig.add_traces(traces_right, rows=[1] * num_right, cols=[2] * num_right)

    if margin is None:
        margin = {"b": 20, "t": 20, "l": 20, "r": 20}

    if layout is None:
        layout = layout3d(width=width, height=height, margin=margin, **kwargs)
    else:
        layout.update({"width": width, "height": height, "margin": margin})
    fig.layout.update(layout)

    if synchronize:
        fig = go.FigureWidget(fig)
        syncing = [False]

        # Guard against infinite ping-pong updates by tracking whether the
        # synchronization callback is already applying a camera change.
        def cam_change_left(_: Any, camera: dict[str, Any]) -> None:
            if not syncing[0]:
                syncing[0] = True
                fig.layout.scene2.camera = camera
                syncing[0] = False

        def cam_change_right(_: Any, camera: dict[str, Any]) -> None:
            if not syncing[0]:
                syncing[0] = True
                fig.layout.scene1.camera = camera
                syncing[0] = False

        fig.layout.scene1.on_change(cam_change_left, "camera")
        fig.layout.scene2.on_change(cam_change_right, "camera")

    return fig
