"""Plotly backend for renderer-neutral scenes."""

from __future__ import annotations

from typing import Any

import numpy as np
from plotly import graph_objs as go

from ..layout import dual_figure3d, layout3d
from .model import PointLayer, Scene

__all__ = ["PlotlyBackend"]


class PlotlyBackend:
    """Convert renderer-neutral point layers into Plotly figures."""

    def render(
        self,
        scene: Scene,
        *,
        split_objects: bool = False,
        synchronize: bool = False,
        layout: go.Layout | None = None,
        **layout_kwargs: Any,
    ) -> go.Figure:
        """Render a scene, optionally materializing one trace per object.

        Parameters
        ----------
        scene : Scene
            Renderer-neutral scene to convert.
        split_objects : bool, default False
            Whether to emit one Plotly trace per domain object.
        synchronize : bool, default False
            Whether to synchronize cameras when rendering two views.
        layout : go.Layout, optional
            Preconfigured Plotly layout. If omitted, build a 3D layout from
            ``layout_kwargs``.
        **layout_kwargs : dict, optional
            Options forwarded to :func:`spine.vis.layout.layout3d`.

        Returns
        -------
        go.Figure
            Plotly representation of the scene.

        Raises
        ------
        ValueError
            If the scene contains more than two independent views.
        """
        # Return a valid empty figure for scenes without views
        if not scene.views:
            return go.Figure(layout=layout or layout3d(**layout_kwargs))

        # Plotly's current subplot helper supports the event-display use case
        if len(scene.views) > 2:
            raise ValueError("The Plotly backend currently supports at most two views.")

        # Convert layers to plain dictionaries to avoid double Plotly validation
        trace_groups = [
            [
                trace
                for layer in view.layers
                for trace in self._point_traces(layer, split_objects)
            ]
            for view in scene.views
        ]

        # Build a common layout unless the caller provided one explicitly
        layout = layout or layout3d(**layout_kwargs)

        # Render truth and reconstruction as linked side-by-side scenes
        if len(trace_groups) == 2:
            return dual_figure3d(
                trace_groups[0],
                trace_groups[1],
                layout=layout,
                synchronize=synchronize,
                titles=[view.name for view in scene.views],
            )

        # Emit all layers into one 3D scene for the common case
        return go.Figure(data=trace_groups[0], layout=layout)

    def _point_traces(
        self, layer: PointLayer, split_objects: bool
    ) -> list[dict[str, Any]]:
        """Build unvalidated trace dictionaries for one point layer.

        Returning dictionaries avoids constructing validated ``Scatter3d``
        objects only for ``Figure`` to validate and copy them a second time.

        Parameters
        ----------
        layer : PointLayer
            Point layer to convert.
        split_objects : bool
            Whether to slice the layer at its stored object boundaries.

        Returns
        -------
        list[dict]
            Unvalidated Plotly trace dictionaries.
        """
        # Preserve one contiguous trace unless object splitting is explicit
        if not split_objects or layer.object_offsets is None:
            return [self._point_trace(layer, slice(None), layer.name)]

        # Materialize object slices only at the backend boundary
        traces = []
        for index, (start, stop) in enumerate(
            zip(layer.object_offsets[:-1], layer.object_offsets[1:])
        ):
            name = f"{layer.name} {index}" if layer.name else str(index)
            traces.append(self._point_trace(layer, slice(start, stop), name))
        return traces

    @staticmethod
    def _point_trace(
        layer: PointLayer, selection: slice, name: str | None
    ) -> dict[str, Any]:
        """Convert one point-layer slice into a Plotly trace dictionary.

        Parameters
        ----------
        layer : PointLayer
            Source point layer.
        selection : slice
            Point range included in the trace.
        name : str, optional
            Trace legend label.

        Returns
        -------
        dict
            Unvalidated ``scatter3d`` trace specification.
        """
        # Slice the shared coordinate buffer without copying it
        points = layer.positions[selection]

        # Slice per-point values while preserving shared scalar values
        values = layer.values
        if values is not None and not np.isscalar(values):
            values = values[selection]

        # Apply the same selection to optional per-point hover labels
        hovertext = layer.hovertext
        if hovertext is not None and not isinstance(hovertext, str):
            hovertext = hovertext[selection]

        # Translate common point-style hints to Plotly marker properties
        marker = {
            "size": layer.style.size,
            "color": values,
            "opacity": layer.style.opacity,
            "colorscale": layer.style.colorscale,
            "cmin": layer.style.cmin,
            "cmax": layer.style.cmax,
        }

        # Keep the trace unvalidated until the final Figure construction
        return {
            "type": "scatter3d",
            "mode": "markers",
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "name": name,
            "text": hovertext,
            "hovertemplate": "x: %{x}<br>y: %{y}<br>z: %{z}<br>%{text}<extra></extra>",
            "marker": marker,
        }
