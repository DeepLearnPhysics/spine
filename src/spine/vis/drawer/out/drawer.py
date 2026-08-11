"""Drawer class for reconstructed and truth output objects."""

from __future__ import annotations

from typing import Any

import numpy as np
from plotly import graph_objs as go

import spine.data.out
from spine.geo import GeoManager

from ...layout import dual_figure3d, layout3d
from ...scene import PointLayer, PointStyle, Scene, SceneView
from ...trace.cluster import scatter_clusters
from ..geo import GeoDrawer
from ..lite import scatter_lite
from .colors import build_object_colors
from .traces import (
    build_crt_trace,
    build_direction_trace,
    build_end_point_trace,
    build_flash_hypothesis_trace,
    build_flash_trace,
    build_raw_trace,
    build_start_point_trace,
    build_vertex_trace,
    get_flash_hypothesis_pe,
    get_flash_pe,
)

__all__ = ["Drawer"]


class Drawer:
    """Handle drawing of reconstructed and truth output objects.

    This class owns the public visualization API for reconstructed and truth
    fragments, particles, and interactions. It validates the requested draw
    mode and attributes, dispatches to the lower-level trace builders, and
    assembles the final Plotly figure.
    """

    # List of supported object families
    _obj_types = ("fragments", "particles", "interactions")

    # Supported draw modes
    _draw_modes = ("reco", "truth", "both", "all")

    # Supported truth point, deposition and source modes and their corresponding backing data keys
    _point_modes = (
        ("points", "points_label"),
        ("points_adapt", "points"),
        ("points_g4", "points_g4"),
    )
    _dep_modes = (
        ("depositions", "depositions_label"),
        ("depositions_q", "depositions_q_label"),
        ("depositions_adapt", "depositions_label_adapt"),
        ("depositions_adapt_q", "depositions"),
        ("depositions_g4", "depositions_g4"),
    )
    _source_modes = (("sources", "sources_label"), ("sources_adapt", "sources"))

    def __init__(
        self,
        data: dict[str, Any],
        draw_mode: str = "both",
        truth_point_mode: str = "points",
        truth_dep_mode: str = "depositions",
        split_scene: bool = True,
        geo: Any | None = None,
        detector_coords: bool = True,
        lite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the drawer configuration.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing output objects and their supporting arrays.
        draw_mode : str, default ``"both"``
            Drawing mode, one of ``"reco"``, ``"truth"``, ``"both"``, or
            ``"all"``.
        truth_point_mode : str, default ``"points"``
            Truth-object point attribute to use when drawing truth objects.
        truth_dep_mode : str, default ``"depositions"``
            Truth-object deposition attribute to use when drawing truth raw
            inputs.
        split_scene : bool, default True
            If ``True`` and both truth and reconstruction are drawn, split the
            two views into separate scenes.
        geo : Any, optional
            Geometry object used to draw detector overlays.
        detector_coords : bool, default True
            If ``True``, interpret object coordinates as detector coordinates.
        lite : bool, default False
            If ``True``, draw the lite object representation without per-point
            clouds.
        **kwargs : Any
            Additional layout options forwarded to
            :func:`spine.vis.layout.layout3d`.
        """
        # Store the data products to be visualized for use by the trace builders
        self.data = data

        # Validate the requested draw mode and determine which object families to draw
        if draw_mode not in self._draw_modes:
            raise ValueError(
                f"`mode` not recognized: {draw_mode}. Must be one of {self._draw_modes}."
            )

        # Determine which object families to draw based on the requested draw mode
        self.prefixes: list[str] = []
        if draw_mode != "truth":
            self.prefixes.append("reco")
        if draw_mode != "reco":
            self.prefixes.append("truth")

        self.supported_objs = []
        for mode in ["reco", "truth"]:
            for obj_type in self._obj_types:
                self.supported_objs.append(f"{mode}_{obj_type}")

        # Validate the requested truth point and deposition modes
        if truth_point_mode not in self.point_modes:
            raise ValueError(
                "The `truth_point_mode` argument must be one of "
                f"{self.point_modes.keys()}. Got `{truth_point_mode}` instead."
            )
        if truth_dep_mode not in self.dep_modes:
            raise ValueError(
                "The `truth_dep_mode` argument must be one of "
                f"{self.dep_modes.keys()}. Got `{truth_dep_mode}` instead."
            )
        self.truth_point_mode = truth_point_mode
        self.truth_point_key = self.point_modes[truth_point_mode]
        self.truth_dep_key = self.dep_modes[truth_dep_mode]
        self.truth_index_mode = truth_point_mode.replace("points", "index")

        # Initialize the geometry drawer if a geometry object is provided or can be
        # fetched from the GeoManager singleton
        self.geo, self.geo_drawer = None, None
        self.meta = data.get("meta", None)
        if geo is not None or GeoManager.is_initialized():
            self.geo = geo if geo is not None else GeoManager.get_instance()
            self.geo_drawer = GeoDrawer(geo=self.geo, detector_coords=detector_coords)
            self.geo = self.geo_drawer.geo

        # Store the remaining configuration options for use by the trace builders and layout
        self.lite = lite
        self.split_scene = split_scene
        self.detector_coords = detector_coords
        self.meta = self.meta if self.geo is None else None
        self.layout_kwargs = kwargs

    @property
    def point_modes(self) -> dict[str, str]:
        """Map truth point-mode names to the backing point arrays.

        Returns
        -------
        Dict[str, str]
            Mapping from truth point mode names to the corresponding event-data
            keys.
        """
        return dict(self._point_modes)

    @property
    def dep_modes(self) -> dict[str, str]:
        """Map truth deposition-mode names to the backing deposition arrays.

        Returns
        -------
        Dict[str, str]
            Mapping from truth deposition mode names to the corresponding
            event-data keys.
        """
        return dict(self._dep_modes)

    @property
    def source_modes(self) -> dict[str, str]:
        """Map truth source-mode names to the backing source arrays.

        Returns
        -------
        Dict[str, str]
            Mapping from truth source mode names to the corresponding event-data
            keys.
        """
        return dict(self._source_modes)

    def get_index(self, obj: Any) -> np.ndarray:
        """Fetch the index array used to draw one object.

        Parameters
        ----------
        obj : Any
            Reconstructed or truth output object.

        Returns
        -------
        np.ndarray
            Point indices used to draw the object.
        """
        if not obj.is_truth:
            return obj.index
        return getattr(obj, self.truth_index_mode)

    def get_scene(
        self,
        obj_type: str,
        attr: str | list[str] | None = None,
        color_attr: str | None = None,
        draw_raw: bool = False,
    ) -> Scene:
        """Build a renderer-neutral scene for the dominant point-cloud path.

        Unlike :meth:`get`, this method does not instantiate Plotly objects.
        It preserves object boundaries and per-point attributes in contiguous
        arrays so a backend may upload one buffer or choose to split objects.

        Parameters
        ----------
        obj_type : str
            Object family to draw.
        attr : Union[str, List[str]], optional
            Object attributes included in hover labels and numeric layer data.
        color_attr : str, optional
            Attribute used to define point colors.
        draw_raw : bool, default False
            If ``True``, include the raw deposition cloud as its own layer.

        Returns
        -------
        Scene
            Renderer-neutral scene with one view or split reco/truth views.

        Notes
        -----
        Geometry, optical, CRT, and auxiliary glyphs remain on the established
        Plotly path for now. They can be added as new neutral layer types
        without changing this API.
        """
        if self.lite:
            raise NotImplementedError(
                "Renderer-neutral lite-object scenes are not implemented yet."
            )
        attrs = self._validate_request(obj_type, attr)
        layers = {}
        for prefix in self.prefixes:
            obj_name = f"{prefix}_{obj_type}"
            if obj_name not in self.data:
                raise ValueError(
                    f"Must provide `{obj_name}` in the data products to draw them."
                )
            object_layer = self._object_layer(
                obj_name, attrs[prefix], color_attr=color_attr
            )
            layers[prefix] = [object_layer]
            if draw_raw:
                layers[prefix].insert(0, self._raw_layer(prefix))

        if len(self.prefixes) > 1 and self.split_scene:
            views = [
                SceneView(
                    name=f"{prefix.capitalize()} {obj_type}",
                    layers=layers[prefix],
                    metadata={"prefix": prefix},
                )
                for prefix in self.prefixes
            ]
        else:
            views = [
                SceneView(
                    name=obj_type.capitalize(),
                    layers=[
                        layer for prefix in self.prefixes for layer in layers[prefix]
                    ],
                )
            ]
        return Scene(
            views=views,
            metadata={
                "object_type": obj_type,
                "split_scene": self.split_scene,
                "detector_coords": self.detector_coords,
            },
        )

    def _validate_request(
        self, obj_type: str, attr: str | list[str] | None
    ) -> dict[str, list[str]]:
        """Validate an object request and normalize attributes per prefix."""
        if obj_type not in self._obj_types:
            raise ValueError(
                f"Object type not recognized: {obj_type}. Must be one of "
                f"{self._obj_types}."
            )
        req_attrs = [attr] if isinstance(attr, str) else attr
        req_attrs = list(dict.fromkeys(req_attrs)) if req_attrs is not None else []
        req_attr_set = set(req_attrs)
        found_attrs = set()
        attrs = {prefix: [] for prefix in self.prefixes}
        for prefix in self.prefixes:
            class_name = f"{prefix.capitalize()}{obj_type[:-1].capitalize()}"
            class_obj = getattr(spine.data.out, class_name)()
            valid_attrs = set(class_obj.attr_names())
            attrs[prefix] = [name for name in req_attrs if name in valid_attrs]
            found_attrs.update(attrs[prefix])
        if req_attr_set != found_attrs:
            missing_attrs = req_attr_set.difference(found_attrs)
            raise ValueError(
                "The following requested attributes are not available for "
                f"any of the drawn objects: {missing_attrs}."
            )
        return attrs

    def get(
        self,
        obj_type: str,
        attr: str | list[str] | None = None,
        color_attr: str | None = None,
        draw_raw: bool = False,
        draw_end_points: bool = False,
        draw_directions: bool = False,
        draw_vertices: bool = False,
        draw_flashes: bool = False,
        draw_flash_hypotheses: bool = False,
        matched_flash_only: bool = True,
        optical_size_by_pe: bool = False,
        flash_hypothesis_key: str = "flash_hypotheses",
        draw_crthits: bool = False,
        matched_crthit_only: bool = True,
        synchronize: bool = False,
        titles: list[str] | None = None,
        split_traces: bool = False,
    ) -> go.Figure:
        """Draw the requested reconstructed and/or truth object family.

        Parameters
        ----------
        obj_type : str
            Object family to draw, one of ``"fragments"``, ``"particles"``, or
            ``"interactions"``.
        attr : Union[str, List[str]], optional
            Object attribute or attributes to include in the hovertext.
        color_attr : str, optional
            Attribute used to determine trace colors.
        draw_raw : bool, default False
            If ``True``, prepend a raw-input deposition trace.
        draw_end_points : bool, default False
            If ``True``, draw fragment or particle start and end points.
        draw_directions : bool, default False
            If ``True``, draw fragment or particle start-direction arrows.
        draw_vertices : bool, default False
            If ``True``, draw interaction vertices.
        draw_flashes : bool, default False
            If ``True``, draw optical flashes.
        draw_flash_hypotheses : bool, default False
            If ``True``, draw stored optical predictions for the selected
            interactions.
        matched_flash_only : bool, default True
            If ``True``, only draw flashes matched to the selected interactions.
        optical_size_by_pe : bool, default False
            If ``True``, scale both measured-flash and hypothesis detector
            glyphs by PE using one shared normalization.
        flash_hypothesis_key : str, default 'flash_hypotheses'
            Event-data key containing optical hypothesis objects.
        draw_crthits : bool, default False
            If ``True``, draw CRT hits and hit planes.
        matched_crthit_only : bool, default True
            If ``True``, only draw CRT hits matched to the selected objects.
        synchronize : bool, default False
            If ``True`` and two scenes are drawn, synchronize their cameras.
        titles : List[str], optional
            Titles for the two scenes when ``split_scene`` is enabled.
        split_traces : bool, default False
            If ``True``, emit one trace per object instead of one shared trace.

        Returns
        -------
        go.Figure
            Plotly figure containing all requested object and detector traces.
        """
        attrs = self._validate_request(obj_type, attr)

        traces: dict[str, list] = {}
        for prefix in self.prefixes:
            obj_name = f"{prefix}_{obj_type}"
            if obj_name not in self.data:
                raise ValueError(
                    f"Must provide `{obj_name}` in the data products to draw them."
                )
            traces[prefix] = self._object_traces(
                obj_name, attrs[prefix], color_attr, split_traces
            )

        # Draw raw inputs as a separate trace, if requested
        if draw_raw:
            for prefix in self.prefixes:
                traces[prefix] = (
                    build_raw_trace(
                        data=self.data,
                        prefix=prefix,
                        prefixes=self.prefixes,
                        truth_point_key=self.truth_point_key,
                        truth_dep_key=self.truth_dep_key,
                        lite=self.lite,
                    )
                    + traces[prefix]
                )

        # Draw start and end points as separate traces, if requested
        if draw_end_points:
            if obj_type == "interactions":
                raise ValueError("Interactions do not have end point attributes.")
            for prefix in self.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                traces[prefix] += build_start_point_trace(
                    data=self.data,
                    obj_name=obj_name,
                    split_traces=split_traces,
                    truth_index_mode=self.truth_index_mode,
                )
                traces[prefix] += build_end_point_trace(
                    data=self.data,
                    obj_name=obj_name,
                    split_traces=split_traces,
                    truth_index_mode=self.truth_index_mode,
                )

        # Draw directions as a separate trace, if requested
        if draw_directions:
            if obj_type == "interactions":
                raise ValueError("Interactions do not have direction attributes.")
            for prefix in self.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                traces[prefix] += build_direction_trace(
                    data=self.data,
                    obj_name=obj_name,
                    split_traces=split_traces,
                    truth_index_mode=self.truth_index_mode,
                )

        # Draw vertices as separate traces, if requested
        if draw_vertices:
            for prefix in self.prefixes:
                obj_name = f"{prefix}_interactions"
                if obj_name not in self.data:
                    raise ValueError(
                        "Must provide interactions to draw their vertices."
                    )
                traces[prefix] += build_vertex_trace(
                    data=self.data,
                    obj_name=obj_name,
                    split_traces=split_traces,
                    truth_index_mode=self.truth_index_mode,
                )

        # Draw flashes as a separate trace, if requested
        show_optical = draw_flashes or draw_flash_hypotheses
        if draw_flashes and "flashes" not in self.data:
            raise ValueError("Must provide the `flashes` objects to draw them.")
        if draw_flash_hypotheses and flash_hypothesis_key not in self.data:
            raise ValueError(
                f"Must provide the `{flash_hypothesis_key}` objects to draw "
                "hypotheses."
            )

        if show_optical:
            for prefix in self.prefixes:
                obj_name = f"{prefix}_interactions"
                if obj_name not in self.data:
                    raise ValueError(
                        "Must provide interactions to draw matched flashes or "
                        "optical hypotheses."
                    )

                flash_pe = None
                if draw_flashes:
                    flash_pe = get_flash_pe(
                        self.data, obj_name, matched_flash_only, self.geo
                    )
                hypothesis_pe = None
                if draw_flash_hypotheses:
                    hypothesis_pe = get_flash_hypothesis_pe(
                        self.data, obj_name, flash_hypothesis_key, self.geo
                    )

                pe_arrays = [pe for pe in (flash_pe, hypothesis_pe) if pe is not None]
                pe_max = max(
                    (float(np.max(pe, initial=0.0)) for pe in pe_arrays),
                    default=0.0,
                )
                cmax = pe_max if pe_max > 0.0 else 1.0

                if draw_flashes:
                    traces[prefix] += build_flash_trace(
                        data=self.data,
                        obj_name=obj_name,
                        matched_only=matched_flash_only,
                        geo=self.geo,
                        geo_drawer=self.geo_drawer,
                        meta=self.meta,
                        size_by_pe=optical_size_by_pe,
                        pe_max=pe_max,
                        pe_per_detector=flash_pe,
                        cmin=0.0,
                        cmax=cmax,
                        opacity=0.55 if draw_flash_hypotheses else 1.0,
                    )
                if draw_flash_hypotheses:
                    traces[prefix] += build_flash_hypothesis_trace(
                        data=self.data,
                        obj_name=obj_name,
                        hypothesis_key=flash_hypothesis_key,
                        geo=self.geo,
                        geo_drawer=self.geo_drawer,
                        meta=self.meta,
                        size_by_pe=optical_size_by_pe,
                        pe_max=pe_max,
                        pe_per_detector=hypothesis_pe,
                        cmin=0.0,
                        cmax=cmax,
                        opacity=0.8,
                    )

        # Draw CRT hits as a separate trace, if requested
        show_crt = False
        if draw_crthits:
            if "crthits" not in self.data:
                raise ValueError("Must provide the `crthits` objects to draw them.")
            show_crt = True
            for prefix in self.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                traces[prefix] += build_crt_trace(
                    data=self.data,
                    obj_name=obj_name,
                    matched_only=matched_crthit_only,
                    geo=self.geo,
                    geo_drawer=self.geo_drawer,
                    meta=self.meta,
                )

        # Draw the geometry overlay if a GeoDrawer is available
        if self.geo_drawer is not None:
            if self.prefixes and self.split_scene:
                for prefix in self.prefixes:
                    traces[prefix] += self.geo_drawer.tpc_traces(meta=self.meta)
            else:
                traces[self.prefixes[-1]] += self.geo_drawer.tpc_traces(meta=self.meta)

        # Build the layout with or without separate scenes based on the configuration and
        # assemble the final figure with all requested traces and the geometry overlay
        layout = layout3d(
            geo=self.geo,
            use_geo=self.geo is not None,
            meta=self.meta,
            detector_coords=self.detector_coords,
            show_optical=show_optical,
            show_crt=show_crt,
            **self.layout_kwargs,
        )

        # If both truth and reconstruction are drawn and split_scene is enabled, draw
        # them in separate scenes with linked cameras. Otherwise, draw all traces
        # together in a single scene.
        if len(self.prefixes) > 1 and self.split_scene:
            if titles is None:
                titles = [f"Reconstructed {obj_type}", f"Truth {obj_type}"]
            return dual_figure3d(
                traces["reco"],
                traces["truth"],
                layout=layout,
                synchronize=synchronize,
                titles=titles,
            )

        # If not splitting into separate scenes, emit all traces together with
        # the shared layout.
        if titles is not None:
            raise ValueError(
                "Providing titles does not do anything when split_scene is False."
            )
        all_traces = []
        for trace_group in traces.values():
            all_traces += trace_group
        return go.Figure(all_traces, layout=layout)

    def _object_layer(
        self,
        obj_name: str,
        attrs: list[str],
        color_attr: str | None,
    ) -> PointLayer:
        """Build one contiguous renderer-neutral object point layer."""
        point_key = self.truth_point_key if "truth" in obj_name else "points"
        if point_key not in self.data:
            raise ValueError(
                f"The `{point_key}` attribute must be provided if the full "
                f"version of the `{obj_name}` objects is to be drawn."
            )

        objects = self.data[obj_name]
        points = self.data[point_key]
        indices = [np.asarray(self.get_index(obj), dtype=np.int64) for obj in objects]
        counts = np.asarray([len(index) for index in indices], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        positions = (
            np.concatenate([points[index] for index in indices if len(index)], axis=0)
            if np.sum(counts)
            else np.empty((0, 3), dtype=np.float32)
        )
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

        color_dict = build_object_colors(
            data=self.data,
            obj_name=obj_name,
            attrs=attrs,
            color_attr=color_attr,
            split_traces=False,
            geo=self.geo,
            lite=False,
            truth_point_key=self.truth_point_key,
            truth_point_mode=self.truth_point_mode,
            dep_modes=self.dep_modes,
        )
        values = self._expand_object_values(color_dict["color"], indices, len(points))
        hovertext = self._expand_object_values(
            color_dict["hovertext"], indices, len(points), dtype=object
        )

        layer_attrs = {"object_id": object_ids}
        for name in dict.fromkeys([*attrs, color_attr]):
            if name is None or name == "id":
                continue
            raw_values = [getattr(obj, name) for obj in objects]
            try:
                expanded = self._expand_object_values(raw_values, indices, len(points))
            except (TypeError, ValueError):
                continue
            if expanded is not None and np.asarray(expanded).dtype.kind in "biuf":
                layer_attrs[name] = expanded

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
            metadata={"object_name": obj_name},
        )

    def _raw_layer(self, prefix: str) -> PointLayer:
        """Build a renderer-neutral raw-deposition layer."""
        if prefix == "reco":
            point_key, dep_key = "points", "depositions"
        else:
            point_key, dep_key = self.truth_point_key, self.truth_dep_key
        if point_key not in self.data or dep_key not in self.data:
            raise ValueError(
                f"Must provide `{point_key}` and `{dep_key}` to draw raw input."
            )
        points = self.data[point_key]
        deps = np.asarray(self.data[dep_key])
        cmax = float(2 * np.median(deps)) if len(deps) else 1.0
        return PointLayer(
            positions=points,
            name="Raw input",
            values=deps,
            attributes={"depositions": deps},
            style=PointStyle(colorscale="Inferno", cmin=0.0, cmax=cmax),
            metadata={"kind": "raw", "prefix": prefix},
        )

    @staticmethod
    def _expand_object_values(
        values: Any,
        indices: list[np.ndarray],
        source_count: int,
        dtype: Any | None = None,
    ) -> Any:
        """Expand scalar, per-object, or per-source values to displayed points."""
        if values is None or np.isscalar(values):
            return values
        if len(values) == source_count and len(values) != len(indices):
            array = np.asarray(values)
            parts = [array[index] for index in indices if len(index)]
        elif len(values) == len(indices):
            parts = []
            for value, index in zip(values, indices):
                if not len(index):
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
            raise ValueError(
                "Values must be scalar, per source point, or per displayed object."
            )
        if not parts:
            return np.empty(0, dtype=dtype or np.float32)
        return (
            np.concatenate(parts).astype(dtype, copy=False)
            if dtype
            else np.concatenate(parts)
        )

    def _object_traces(
        self,
        obj_name: str,
        attr: list[str],
        color_attr: str | None,
        split_traces: bool,
    ) -> list:
        """Build object traces for one reconstructed or truth collection.

        Parameters
        ----------
        obj_name : str
            Name of the object collection to visualize.
        attr : List[str]
            Object attributes requested for hovertext generation.
        color_attr : str, optional
            Attribute used to define the marker or cluster colors.
        split_traces : bool
            If ``True``, emit one trace per object.

        Returns
        -------
        list
            Trace list representing the requested objects.
        """
        # Build the color mapping for the object collection based on the requested
        # color attribute and the semantics of its name if it is not explicitly requested.
        color_dict = build_object_colors(
            data=self.data,
            obj_name=obj_name,
            attrs=attr,
            color_attr=color_attr,
            split_traces=split_traces,
            geo=self.geo,
            lite=self.lite,
            truth_point_key=self.truth_point_key,
            truth_point_mode=self.truth_point_mode,
            dep_modes=self.dep_modes,
        )

        # If the full version of the objects is available, use it to draw clustered
        # traces. Otherwise, fall back to drawing the lite representation with one
        # point per object.
        if not self.lite:
            # Full objects are rendered by indexing into the backing point cloud.
            point_key = self.truth_point_key if "truth" in obj_name else "points"
            if point_key not in self.data:
                raise ValueError(
                    f"The `{point_key}` attribute must be provided if the full "
                    f"version of the `{obj_name}` objects is to be drawn."
                )

            points = self.data[point_key]
            clusts = [self.get_index(obj) for obj in self.data[obj_name]]
            return scatter_clusters(
                points,
                clusts,
                single_trace=not split_traces,
                shared_legend=not split_traces,
                **color_dict,
            )

        # Lite objects carry their own minimal geometric representation.
        return scatter_lite(
            self.data[obj_name],
            shared_legend=not split_traces,
            **color_dict,
        )
