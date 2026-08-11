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
        """Render a scene, optionally materializing one trace per object."""
        if not scene.views:
            return go.Figure(layout=layout or layout3d(**layout_kwargs))
        if len(scene.views) > 2:
            raise ValueError("The Plotly backend currently supports at most two views.")

        trace_groups = [
            [
                trace
                for layer in view.layers
                for trace in self._point_traces(layer, split_objects)
            ]
            for view in scene.views
        ]
        layout = layout or layout3d(**layout_kwargs)
        if len(trace_groups) == 2:
            return dual_figure3d(
                trace_groups[0],
                trace_groups[1],
                layout=layout,
                synchronize=synchronize,
                titles=[view.name for view in scene.views],
            )
        return go.Figure(data=trace_groups[0], layout=layout)

    def _point_traces(
        self, layer: PointLayer, split_objects: bool
    ) -> list[dict[str, Any]]:
        """Build unvalidated trace dictionaries for one point layer.

        Returning dictionaries avoids constructing validated ``Scatter3d``
        objects only for ``Figure`` to validate and copy them a second time.
        """
        if not split_objects or layer.object_offsets is None:
            return [self._point_trace(layer, slice(None), layer.name)]

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
        points = layer.positions[selection]
        values = layer.values
        if values is not None and not np.isscalar(values):
            values = values[selection]
        hovertext = layer.hovertext
        if hovertext is not None and not isinstance(hovertext, str):
            hovertext = hovertext[selection]
        marker = {
            "size": layer.style.size,
            "color": values,
            "opacity": layer.style.opacity,
            "colorscale": layer.style.colorscale,
            "cmin": layer.style.cmin,
            "cmax": layer.style.cmax,
        }
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
