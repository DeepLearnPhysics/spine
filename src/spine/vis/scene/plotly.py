"""Plotly backend for renderer-neutral scenes."""

from __future__ import annotations

from typing import Any

import numpy as np
from plotly import graph_objs as go

from ..layout import dual_figure3d, layout3d
from ..trace.arrow import scatter_arrows
from .model import (
    BoxLayer,
    LineLayer,
    LineStyle,
    MarkerLayer,
    MeshLayer,
    PointLayer,
    Scene,
    VectorLayer,
)

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
        # Reconstruct the established detector layout from portable metadata.
        # Explicit backend arguments and preconfigured layouts take precedence.
        if layout is None:
            metadata = scene.metadata
            layout_kwargs = dict(layout_kwargs)
            bounds = metadata.get("layout_bounds", metadata.get("bounds"))
            if bounds is not None:
                layout_kwargs.setdefault("ranges", np.asarray(bounds))
            if metadata.get("detector_coords") is not None:
                layout_kwargs.setdefault("detector_coords", metadata["detector_coords"])
            if metadata.get("up_dir") is not None:
                layout_kwargs.setdefault("up_dir", np.asarray(metadata["up_dir"]))
            layout = layout3d(**layout_kwargs)

        # Return a valid empty figure for scenes without views
        if not scene.views:
            return go.Figure(layout=layout)

        # Plotly's current subplot helper supports the event-display use case
        if len(scene.views) > 2:
            raise ValueError("The Plotly backend currently supports at most two views.")

        # Convert layers to plain dictionaries to avoid double Plotly validation
        trace_groups = [
            [
                trace
                for layer in view.layers
                for trace in self._layer_traces(layer, split_objects)
            ]
            for view in scene.views
        ]

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

    def _layer_traces(self, layer: Any, split_objects: bool) -> list[dict[str, Any]]:
        """Convert one neutral layer to Plotly trace dictionaries.

        Parameters
        ----------
        layer : Any
            Supported renderer-neutral layer.
        split_objects : bool
            Whether point layers should be split at object boundaries.

        Returns
        -------
        list[dict]
            Unvalidated Plotly trace dictionaries.

        Raises
        ------
        TypeError
            If the layer type is unsupported.
        """
        if isinstance(layer, PointLayer):
            return self._point_traces(layer, split_objects)
        if isinstance(layer, LineLayer):
            return [self._line_trace(layer)]
        if isinstance(layer, VectorLayer):
            return self._vector_traces(layer)
        if isinstance(layer, MeshLayer):
            if layer.style.wireframe:
                return [self._mesh_wireframe_trace(layer)]
            return [self._mesh_trace(layer)]
        if isinstance(layer, BoxLayer):
            return [self._box_trace(layer)]
        raise TypeError(f"Unsupported scene layer: {type(layer).__name__}.")

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
        if isinstance(layer, MarkerLayer):
            marker["symbol"] = layer.symbol

        # Keep the trace unvalidated until the final Figure construction
        return {
            "type": "scatter3d",
            "mode": "markers",
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "name": name,
            "meta": layer.metadata,
            "text": hovertext,
            "hovertemplate": "x: %{x}<br>y: %{y}<br>z: %{z}<br>%{text}<extra></extra>",
            "marker": marker,
        }

    @staticmethod
    def _line_trace(layer: LineLayer) -> dict[str, Any]:
        """Convert independent segments to one NaN-separated Plotly line.

        Parameters
        ----------
        layer : LineLayer
            Neutral line segments and display style.

        Returns
        -------
        dict
            Unvalidated ``scatter3d`` line specification.

        Raises
        ------
        ValueError
            If hover labels or values do not align with the segments.
        """
        # Plotly represents disconnected lines with non-finite separators.
        separators = np.full((len(layer.segments), 1, 3), np.nan, dtype=np.float32)
        points = np.concatenate((layer.segments, separators), axis=1).reshape(-1, 3)

        hovertext = layer.hovertext
        if hovertext is not None:
            if np.isscalar(hovertext):
                hovertext = np.full(len(layer.segments), hovertext, dtype=object)
            else:
                hovertext = np.asarray(hovertext, dtype=object)
                if hovertext.ndim == 0:
                    hovertext = np.full(
                        len(layer.segments), hovertext.item(), dtype=object
                    )
                elif hovertext.shape != (len(layer.segments),):
                    raise ValueError(
                        "Line hover text must provide one label per segment."
                    )
            hovertext = np.column_stack(
                (
                    hovertext,
                    hovertext,
                    np.full(len(hovertext), None, dtype=object),
                )
            ).reshape(-1)

        # Repeat segment values at both endpoints and leave separators uncolored.
        values = layer.values
        if values is not None and not np.isscalar(values):
            values = np.asarray(values)
            if values.shape != (len(layer.segments),):
                raise ValueError("Line values must provide one scalar per segment.")
            values = np.column_stack(
                (values, values, np.full(len(values), np.nan))
            ).reshape(-1)
        return {
            "type": "scatter3d",
            "mode": "lines",
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "name": layer.name,
            "meta": layer.metadata,
            "hovertext": hovertext,
            "line": {
                "width": layer.style.width,
                "color": (
                    layer.style.color if layer.style.color is not None else values
                ),
                "colorscale": layer.style.colorscale,
                "cmin": layer.style.cmin,
                "cmax": layer.style.cmax,
            },
            "opacity": layer.style.opacity,
        }

    @staticmethod
    def _vector_traces(layer: VectorLayer) -> list[dict[str, Any]]:
        """Convert vectors while preserving optional per-vector colors.

        Parameters
        ----------
        layer : VectorLayer
            Neutral vector glyphs and display style.

        Returns
        -------
        list[dict]
            Plotly cone or arrow trace dictionaries.
        """
        if layer.values is None:
            return [PlotlyBackend._vector_trace(layer)]

        traces = scatter_arrows(
            layer.origins,
            layer.vectors,
            length=layer.scale,
            tip_ratio=layer.head_size,
            color=layer.values,
            hovertext=layer.hovertext,
            linewidth=layer.style.width,
            colorscale=layer.style.colorscale,
            cmin=layer.style.cmin,
            cmax=layer.style.cmax,
            name=layer.name,
        )
        return [trace.to_plotly_json() for trace in traces]

    @staticmethod
    def _vector_trace(layer: VectorLayer) -> dict[str, Any]:
        """Convert vectors to Plotly cones rooted at their origins.

        Parameters
        ----------
        layer : VectorLayer
            Neutral vectors sharing one display style.

        Returns
        -------
        dict
            Unvalidated Plotly ``cone`` trace specification.
        """
        vectors = layer.vectors * layer.scale
        return {
            "type": "cone",
            "x": layer.origins[:, 0],
            "y": layer.origins[:, 1],
            "z": layer.origins[:, 2],
            "u": vectors[:, 0],
            "v": vectors[:, 1],
            "w": vectors[:, 2],
            "name": layer.name,
            "meta": layer.metadata,
            "hovertext": layer.hovertext,
            "anchor": "tail",
            "sizemode": "absolute",
            "sizeref": 1.0,
            "showscale": False,
            "colorscale": (
                [[0.0, layer.style.color], [1.0, layer.style.color]]
                if layer.style.color is not None
                else None
            ),
            "opacity": layer.style.opacity,
        }

    @staticmethod
    def _mesh_trace(layer: MeshLayer) -> dict[str, Any]:
        """Convert an indexed neutral mesh to Plotly Mesh3d.

        Parameters
        ----------
        layer : MeshLayer
            Neutral mesh vertices, faces and display style.

        Returns
        -------
        dict
            Unvalidated Plotly ``mesh3d`` trace specification.
        """
        intensity = layer.values
        if intensity is not None and np.isscalar(intensity):
            intensity = np.full(len(layer.vertices), intensity)
        return {
            "type": "mesh3d",
            "x": layer.vertices[:, 0],
            "y": layer.vertices[:, 1],
            "z": layer.vertices[:, 2],
            "i": layer.faces[:, 0],
            "j": layer.faces[:, 1],
            "k": layer.faces[:, 2],
            "name": layer.name,
            "meta": layer.metadata,
            "hovertext": layer.hovertext,
            "color": layer.style.color,
            "intensity": intensity,
            "opacity": layer.style.opacity,
            "colorscale": layer.style.colorscale,
            "cmin": layer.style.cmin,
            "cmax": layer.style.cmax,
            "flatshading": True,
        }

    @classmethod
    def _mesh_wireframe_trace(cls, layer: MeshLayer) -> dict[str, Any]:
        """Convert mesh triangle edges to a NaN-separated Plotly line.

        Parameters
        ----------
        layer : MeshLayer
            Neutral mesh to render as wireframe edges.

        Returns
        -------
        dict
            Unvalidated Plotly ``scatter3d`` line specification.
        """
        segments = []
        for face in layer.faces:
            a, b, c = layer.vertices[face]
            segments.extend(((a, b), (b, c), (c, a)))
        lines = LineLayer(
            np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3),
            name=layer.name,
            hovertext=layer.hovertext,
            style=LineStyle(
                color=layer.style.color,
                opacity=layer.style.opacity,
                colorscale=layer.style.colorscale,
                cmin=layer.style.cmin,
                cmax=layer.style.cmax,
            ),
            metadata=layer.metadata,
        )
        return cls._line_trace(lines)

    @classmethod
    def _box_trace(cls, layer: BoxLayer) -> dict[str, Any]:
        """Convert axis-aligned boxes to a line or triangle trace.

        Parameters
        ----------
        layer : BoxLayer
            Neutral boxes and face or wireframe display configuration.

        Returns
        -------
        dict
            Unvalidated Plotly mesh or line trace specification.
        """
        # Templates use the conventional eight-corner box ordering and are
        # translated and scaled independently for each requested box.
        vertices, faces, segments = [], [], []
        corners = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.int32,
        )
        face_template = np.asarray(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.int32,
        )
        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )
        for bounds in layer.bounds:
            base = len(vertices)
            box_vertices = bounds[0] + corners * (bounds[1] - bounds[0])
            vertices.extend(box_vertices)
            faces.extend(face_template + base)
            segments.extend([[box_vertices[a], box_vertices[b]] for a, b in edges])
        if layer.draw_faces:
            values = layer.values
            if values is not None and not np.isscalar(values):
                values = np.repeat(values, 8)
            hovertext = layer.hovertext
            if hovertext is not None and not np.isscalar(hovertext):
                hovertext = np.repeat(hovertext, 8)
            mesh = MeshLayer(
                np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
                np.asarray(faces, dtype=np.int32).reshape(-1, 3),
                name=layer.name,
                values=values,
                hovertext=hovertext,
                style=layer.mesh_style,
                metadata=layer.metadata,
            )
            return cls._mesh_trace(mesh)
        values = layer.values
        if values is not None and not np.isscalar(values):
            values = np.repeat(values, 12)
        hovertext = layer.hovertext
        if hovertext is not None and not np.isscalar(hovertext):
            hovertext = np.repeat(hovertext, 12)
        lines = LineLayer(
            np.asarray(segments, dtype=np.float32).reshape(-1, 2, 3),
            name=layer.name,
            values=values,
            hovertext=hovertext,
            style=layer.line_style,
            metadata=layer.metadata,
        )
        return cls._line_trace(lines)
