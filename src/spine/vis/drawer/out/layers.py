"""Neutral layer construction for output-object scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spine.constants import TRACK_SHP

from ...scene import (
    LineStyle,
    MarkerLayer,
    PointLayer,
    PointStyle,
    VectorLayer,
    plotly_trace_to_layer,
)
from .colors import build_object_colors
from .formatting import is_long_form

if TYPE_CHECKING:  # pragma: no cover
    from .drawer import Drawer

__all__ = ["SceneLayerBuilder"]


class SceneLayerBuilder:
    """Construct renderer-neutral buffers from configured output objects.

    Parameters
    ----------
    drawer : Drawer
        Configured output drawer which owns the event data and drawing modes.
    """

    def __init__(self, drawer: Drawer) -> None:
        """Initialize the layer builder.

        Parameters
        ----------
        drawer : Drawer
            Configured output drawer which supplies data and display settings.
        """
        self.drawer = drawer

    @staticmethod
    def neutralize_traces(traces: list[Any], **metadata: Any) -> list[Any]:
        """Convert established 3D traces into neutral layer primitives.

        Parameters
        ----------
        traces : list
            Plotly traces to convert.
        **metadata : Any
            Semantic metadata attached to each converted layer.

        Returns
        -------
        list
            Renderer-neutral layers in the same order as ``traces``.
        """
        return [plotly_trace_to_layer(trace, **metadata) for trace in traces]

    def marker_layer(
        self,
        obj_name: str,
        point_attr: str,
        color: str = "black",
        colors: dict[str, Any] | None = None,
    ) -> MarkerLayer:
        """Build discrete object markers with object membership.

        Parameters
        ----------
        obj_name : str
            Name of the truth or reconstruction object collection.
        point_attr : str
            Point-valued object attribute to draw.
        color : str, default ``"black"``
            Shared fallback color used when ``colors`` is not provided.
        colors : dict, optional
            Parent-object color values and color-scale configuration.

        Returns
        -------
        MarkerLayer
            Marker positions, labels, object IDs and display style.
        """
        # Skip undefined end points and empty truth objects while retaining the
        # source-object indexes needed to align optional parent colors.
        positions, object_ids, hovertext, indices = [], [], [], []
        obj_type = obj_name.split("_")[-1][:-1].capitalize()
        for index, obj in enumerate(self.drawer.data[obj_name]):
            if point_attr == "end_point" and obj.shape != TRACK_SHP:
                continue
            if obj.is_truth and len(getattr(obj, self.drawer.truth_index_mode)) == 0:
                continue
            positions.append(getattr(obj, point_attr))
            object_ids.append(index)
            indices.append(index)
            hovertext.append(f"{obj_type} {index} " + " ".join(point_attr.split("_")))

        # Match the established Plotly marker sizes and symbols.
        marker_size = 10.0 if point_attr == "vertex" else 7.0
        symbol = {
            "start_point": "circle",
            "end_point": "circle-open",
            "vertex": "diamond",
        }[point_attr]

        # Use one fixed color unless object-level colors were requested.
        values = color
        style = PointStyle(size=marker_size)
        if colors is not None:
            values = np.asarray(colors["color"])[indices]
            style = PointStyle(
                size=marker_size,
                colorscale=colors["colorscale"],
                cmin=colors["cmin"],
                cmax=colors["cmax"],
            )

        return MarkerLayer(
            np.asarray(positions, dtype=np.float32).reshape(-1, 3),
            name=" ".join(obj_name.split("_")).capitalize()[:-1]
            + " "
            + " ".join(point_attr.split("_")),
            values=values,
            hovertext=np.asarray(hovertext, dtype=object),
            object_ids=np.asarray(object_ids, dtype=np.int32),
            style=style,
            symbol=symbol,
            metadata={"kind": point_attr, "object_name": obj_name},
        )

    def vector_layer(
        self, obj_name: str, colors: dict[str, Any] | None = None
    ) -> VectorLayer:
        """Build start-direction vectors with object membership.

        Parameters
        ----------
        obj_name : str
            Name of the truth or reconstruction object collection.
        colors : dict, optional
            Parent-object color values and color-scale configuration.

        Returns
        -------
        VectorLayer
            Direction origins, vectors, labels, object IDs and display style.
        """
        # Empty truth objects do not have a meaningful displayed direction.
        origins, vectors, object_ids, hovertext, indices = [], [], [], [], []
        obj_type = obj_name.split("_")[-1][:-1].capitalize()
        for index, obj in enumerate(self.drawer.data[obj_name]):
            if obj.is_truth and len(getattr(obj, self.drawer.truth_index_mode)) == 0:
                continue
            origins.append(obj.start_point)
            vectors.append(obj.start_dir)
            object_ids.append(index)
            indices.append(index)
            hovertext.append(f"{obj_type} {index} direction")

        # Fall back to the legacy black arrows when colors are not matched.
        values = None
        style = LineStyle(width=5.0, color="black")
        if colors is not None:
            values = np.asarray(colors["color"])[indices]
            style = LineStyle(
                width=5.0,
                colorscale=colors["colorscale"],
                cmin=colors["cmin"],
                cmax=colors["cmax"],
            )

        return VectorLayer(
            np.asarray(origins, dtype=np.float32).reshape(-1, 3),
            np.asarray(vectors, dtype=np.float32).reshape(-1, 3),
            name=" ".join(obj_name.split("_")).capitalize()[:-1] + " directions",
            values=values,
            hovertext=np.asarray(hovertext, dtype=object),
            object_ids=np.asarray(object_ids, dtype=np.int32),
            scale=10.0,
            head_size=0.25,
            style=style,
            metadata={"kind": "directions", "object_name": obj_name},
        )

    def object_layer(
        self,
        obj_name: str,
        attrs: list[str],
        color_attr: str | None,
    ) -> PointLayer:
        """Build one contiguous renderer-neutral object point layer.

        Parameters
        ----------
        obj_name : str
            Name of the truth or reconstruction object collection.
        attrs : list[str]
            Requested hover and numeric point attributes.
        color_attr : str, optional
            Attribute used to define point colors.

        Returns
        -------
        PointLayer
            Contiguous point layer retaining domain-object boundaries.

        Raises
        ------
        ValueError
            If the backing point cloud is absent or attribute values cannot be
            aligned with the selected points.
        """
        # Select the point cloud appropriate for truth or reconstruction
        point_key = self.drawer.truth_point_key if "truth" in obj_name else "points"
        if point_key not in self.drawer.data:
            raise ValueError(
                f"The `{point_key}` attribute must be provided if the full "
                f"version of the `{obj_name}` objects is to be drawn."
            )

        # Resolve object indices and encode their boundaries as prefix offsets
        objects = self.drawer.data[obj_name]
        points = self.drawer.data[point_key]
        indices = [
            np.asarray(self.drawer.get_index(obj), dtype=np.int64) for obj in objects
        ]
        counts = np.asarray([len(index) for index in indices], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts)))

        # Gather domain-object points into one contiguous renderer buffer
        positions = (
            np.concatenate(
                [points[index] for index in indices if len(index) > 0], axis=0
            )
            if np.sum(counts)
            else np.empty((0, 3), dtype=np.float32)
        )

        # Repeat stable domain IDs for client-side picking and highlighting
        object_ids = (
            np.concatenate(
                [
                    np.full(len(index), int(obj.id), dtype=np.int32)
                    for obj, index in zip(objects, indices)
                ]
            )
            if np.sum(counts)
            else np.empty(0, dtype=np.int32)
        )

        # Reuse the established color and hover semantics from the Plotly drawer
        color_dict = build_object_colors(
            data=self.drawer.data,
            obj_name=obj_name,
            attrs=attrs,
            color_attr=color_attr,
            split_traces=False,
            geo=self.drawer.geo,
            lite=False,
            truth_point_key=self.drawer.truth_point_key,
            truth_point_mode=self.drawer.truth_point_mode,
            dep_modes=self.drawer.dep_modes,
        )
        values = self.expand_object_values(color_dict["color"], indices, len(points))
        hovertext = self.expand_object_values(
            color_dict["hovertext"], indices, len(points), dtype=object
        )

        # Preserve requested numeric dimensions for client-side filtering
        layer_attrs = {"object_id": object_ids}
        for name in dict.fromkeys([*attrs, color_attr]):
            if name is None:
                continue
            if name == "id":
                layer_attrs[name] = object_ids
                continue
            raw_values = [getattr(obj, name) for obj in objects]
            try:
                expanded = self.expand_object_values(raw_values, indices, len(points))
            except (TypeError, ValueError):
                continue
            if expanded is not None and np.asarray(expanded).dtype.kind in "biuf":
                layer_attrs[name] = expanded

        # Store the exact numeric mappings for every colorable hover attribute.
        # This lets a browser renderer recolor an uploaded buffer without asking
        # Python to rebuild the scene and preserves categorical source mappings.
        attribute_styles = {}
        for name in attrs:
            try:
                attribute_colors = build_object_colors(
                    data=self.drawer.data,
                    obj_name=obj_name,
                    attrs=[name],
                    color_attr=name,
                    split_traces=False,
                    geo=self.drawer.geo,
                    lite=False,
                    truth_point_key=self.drawer.truth_point_key,
                    truth_point_mode=self.drawer.truth_point_mode,
                    dep_modes=self.drawer.dep_modes,
                )
                attribute_values = self.expand_object_values(
                    attribute_colors["color"], indices, len(points)
                )
            except (TypeError, ValueError):
                continue
            attribute_values = np.asarray(attribute_values)
            if attribute_values.ndim != 1 or len(attribute_values) != len(positions):
                continue
            if attribute_values.dtype.kind not in "biuf":
                continue
            layer_attrs[name] = attribute_values
            attribute_styles[name] = {
                "colorscale": attribute_colors["colorscale"],
                "cmin": attribute_colors["cmin"],
                "cmax": attribute_colors["cmax"],
            }

        # Package arrays and common display hints without constructing Plotly objects
        return PointLayer(
            positions=positions,
            name=color_dict["name"],
            values=values,
            hovertext=hovertext,
            object_ids=object_ids,
            object_offsets=offsets,
            attributes=layer_attrs,
            style=PointStyle(
                colorscale=color_dict["colorscale"],
                cmin=color_dict["cmin"],
                cmax=color_dict["cmax"],
            ),
            metadata={
                "object_name": obj_name,
                "color_attribute": color_attr,
                "attribute_styles": attribute_styles,
                "long_form_attributes": [name for name in attrs if is_long_form(name)],
            },
        )

    def raw_layer(self, prefix: str) -> PointLayer:
        """Build a renderer-neutral raw-deposition layer.

        Parameters
        ----------
        prefix : str
            Object declination, one of ``"reco"`` or ``"truth"``.

        Returns
        -------
        PointLayer
            Raw input points colored by deposition value.

        Raises
        ------
        ValueError
            If the required point or deposition arrays are absent.
        """
        # Select reconstruction or truth input arrays consistently with Drawer
        if prefix == "reco":
            point_key, dep_key = "points", "depositions"
        else:
            point_key, dep_key = self.drawer.truth_point_key, self.drawer.truth_dep_key
        if point_key not in self.drawer.data or dep_key not in self.drawer.data:
            raise ValueError(
                f"Must provide `{point_key}` and `{dep_key}` to draw raw input."
            )

        # Match the legacy Plotly color range for raw depositions
        points = self.drawer.data[point_key]
        deps = np.asarray(self.drawer.data[dep_key])
        cmax = float(2 * np.median(deps)) if len(deps) > 0 else 1.0
        return PointLayer(
            positions=points,
            name="Raw input",
            values=deps,
            attributes={"depositions": deps},
            style=PointStyle(colorscale="Inferno", cmin=0.0, cmax=cmax),
            metadata={"kind": "raw", "prefix": prefix},
        )

    @staticmethod
    def expand_object_values(
        values: Any,
        indices: list[np.ndarray],
        source_count: int,
        dtype: Any | None = None,
    ) -> Any:
        """Expand scalar, per-object, or per-source values to displayed points.

        Parameters
        ----------
        values : Any
            Shared scalar, source-aligned sequence or per-object values.
        indices : list[np.ndarray]
            Source indices selected by each domain object.
        source_count : int
            Number of points in the source point cloud.
        dtype : data-type, optional
            Output data type used when concatenating values.

        Returns
        -------
        Any
            Shared scalar or one contiguous value array per displayed point.

        Raises
        ------
        ValueError
            If values cannot be aligned with the selected object points.
        """
        # Shared values require no point-wise expansion
        if values is None or np.isscalar(values):
            return values

        # Gather arrays already aligned with the source point cloud
        if len(values) == source_count and len(values) != len(indices):
            array = np.asarray(values)
            parts = [array[index] for index in indices if len(index) > 0]
        elif len(values) == len(indices):
            # Expand scalar or point-wise values supplied per domain object
            parts = []
            for value, index in zip(values, indices):
                if len(index) == 0:
                    continue
                if np.isscalar(value):
                    parts.append(np.full(len(index), value, dtype=dtype))
                    continue
                array = np.asarray(value)
                if len(array) < len(index):
                    raise ValueError(
                        "Per-object values are shorter than object points."
                    )
                parts.append(array[: len(index)])
        else:
            # Ambiguous lengths cannot be mapped to the selected point buffer
            raise ValueError(
                "Values must be scalar, per source point, or per displayed object."
            )

        # Preserve a well-defined dtype for empty scenes and requested text arrays
        if not parts:
            return np.empty(0, dtype=dtype or np.float32)

        # Concatenate once after all object-level slices have been collected
        return (
            np.concatenate(parts).astype(dtype, copy=False)
            if dtype
            else np.concatenate(parts)
        )
