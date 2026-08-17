"""Drawer class for reconstructed and truth output objects."""

from __future__ import annotations

from typing import Any

import numpy as np
from plotly import graph_objs as go

import spine.data.out
from spine.geo import GeoManager

from ...layout import dual_figure3d, layout3d
from ...scene import Scene
from ...trace.cluster import scatter_clusters
from ..geo import GeoDrawer
from ..lite import scatter_lite
from .colors import build_object_colors, colorable_attributes, object_color_kind
from .formatting import is_long_form
from .scene import SceneBuilder
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

__all__ = ["Drawer", "colorable_attributes", "object_color_kind"]


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

    # Supported truth point, deposition and source modes and their backing keys
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
        match_aux_colors: bool = True,
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
        match_aux_colors : bool, default True
            If ``True``, match end points, directions, and vertices to their
            parent object colors. If ``False``, use the legacy black end-point
            and direction colors and green interaction vertices.
        **kwargs : Any
            Additional layout options forwarded to
            :func:`spine.vis.layout.layout3d`.
        """
        # Store the data products to be visualized for use by the trace builders
        self.data = data

        # Validate the requested draw mode
        if draw_mode not in self._draw_modes:
            raise ValueError(
                f"`mode` not recognized: {draw_mode}. Must be one of "
                f"{self._draw_modes}."
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

        # Store the remaining trace-builder and layout configuration
        self.lite = lite
        self.match_aux_colors = match_aux_colors
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
    ) -> Scene:
        """Build a renderer-neutral scene for the requested event display.

        Parameters
        ----------
        obj_type : str
            Object family to draw.
        attr : str or list[str], optional
            Object attributes included in hover labels and numeric layer data.
        color_attr : str, optional
            Attribute used to define point colors.
        draw_raw : bool, default False
            If ``True``, include the raw deposition cloud.
        draw_end_points : bool, default False
            If ``True``, include object start and end markers.
        draw_directions : bool, default False
            If ``True``, include object start-direction vectors.
        draw_vertices : bool, default False
            If ``True``, include interaction vertex markers.
        draw_flashes : bool, default False
            If ``True``, include measured optical responses.
        draw_flash_hypotheses : bool, default False
            If ``True``, include predicted optical responses.
        matched_flash_only : bool, default True
            If ``True``, restrict measured flashes to object matches.
        optical_size_by_pe : bool, default False
            If ``True``, scale optical glyphs by photoelectron count.
        flash_hypothesis_key : str, default ``"flash_hypotheses"``
            Event-data key containing optical hypotheses.
        draw_crthits : bool, default False
            If ``True``, include CRT hits and hit planes.
        matched_crthit_only : bool, default True
            If ``True``, restrict CRT hits to object matches.

        Returns
        -------
        Scene
            Renderer-neutral scene with one combined or two split views.

        Raises
        ------
        ValueError
            If a requested collection or required backing array is absent.
        RuntimeError
            If raw input is requested for lite objects.
        """
        # Keep the public entry point small; neutral assembly belongs to the
        # renderer-independent scene builder.
        return SceneBuilder(self).build(
            obj_type=obj_type,
            attr=attr,
            color_attr=color_attr,
            draw_raw=draw_raw,
            draw_end_points=draw_end_points,
            draw_directions=draw_directions,
            draw_vertices=draw_vertices,
            draw_flashes=draw_flashes,
            draw_flash_hypotheses=draw_flash_hypotheses,
            matched_flash_only=matched_flash_only,
            optical_size_by_pe=optical_size_by_pe,
            flash_hypothesis_key=flash_hypothesis_key,
            draw_crthits=draw_crthits,
            matched_crthit_only=matched_crthit_only,
        )

    def build_geometry_traces(self) -> list[Any]:
        """Build detector outlines with contrast for the configured theme.

        Returns
        -------
        list
            Plotly traces describing the configured TPC geometry.

        Raises
        ------
        RuntimeError
            If this drawer was initialized without a geometry drawer.
        """
        if self.geo_drawer is None:
            raise RuntimeError(
                "A geometry drawer is required to build geometry traces."
            )

        dark = self.layout_kwargs.get("dark", False)
        color = "rgba(255,255,255,0.400)" if dark else "rgba(0,0,0,0.200)"
        traces = self.geo_drawer.tpc_traces(meta=self.meta, color=color)
        for trace in traces:
            trace.update(meta={"kind": "geometry"})
        return traces

    def resolve_aux_colors(
        self, obj_name: str, attrs: list[str], color_attr: str | None
    ) -> dict[str, Any] | None:
        """Build parent-object colors for auxiliary markers and vectors.

        Long-form attributes carry one value per point and therefore cannot
        define one unambiguous color for an object-level glyph. In that case,
        auxiliary glyphs fall back to the same object-ID palette used by the
        default object display.

        Parameters
        ----------
        obj_name : str
            Name of the parent object collection.
        attrs : list[str]
            Attributes included in the parent hover labels.
        color_attr : str, optional
            Attribute used to color the parent objects.

        Returns
        -------
        dict or None
            Parent color mapping, or ``None`` when matching is disabled.
        """
        if not self.match_aux_colors:
            return None

        aux_attr = color_attr
        aux_attrs = attrs
        if aux_attr is not None and is_long_form(aux_attr):
            aux_attr = None
            aux_attrs = []

        return build_object_colors(
            data=self.data,
            obj_name=obj_name,
            attrs=aux_attrs,
            color_attr=aux_attr,
            split_traces=False,
            geo=self.geo,
            lite=self.lite,
            truth_point_key=self.truth_point_key,
            truth_point_mode=self.truth_point_mode,
            dep_modes=self.dep_modes,
        )

    @staticmethod
    def _aux_color_kwargs(colors: dict[str, Any] | None) -> dict[str, Any]:
        """Return trace keyword arguments for a resolved color mapping.

        Parameters
        ----------
        colors : dict, optional
            Resolved parent-object color configuration.

        Returns
        -------
        dict
            Color, color-scale and range arguments accepted by trace helpers.
        """
        if colors is None:
            return {}
        return {
            "color": colors["color"],
            "colorscale": colors["colorscale"],
            "cmin": colors["cmin"],
            "cmax": colors["cmax"],
        }

    def validate_request(
        self, obj_type: str, attr: str | list[str] | None
    ) -> dict[str, list[str]]:
        """Validate an object request and normalize attributes per prefix.

        Parameters
        ----------
        obj_type : str
            Object family to validate.
        attr : str or list[str], optional
            Requested hover or point attributes.

        Returns
        -------
        dict[str, list[str]]
            Valid attributes grouped by truth/reconstruction prefix.

        Raises
        ------
        ValueError
            If the object family or any requested attribute is unsupported.
        """
        # Validate the supported domain-object family
        if obj_type not in self._obj_types:
            raise ValueError(
                f"Object type not recognized: {obj_type}. Must be one of "
                f"{self._obj_types}."
            )

        # Normalize scalar and repeated attributes while preserving order
        req_attrs = [attr] if isinstance(attr, str) else attr
        req_attrs = list(dict.fromkeys(req_attrs)) if req_attrs is not None else []
        req_attr_set = set(req_attrs)
        found_attrs = set()
        attrs = {prefix: [] for prefix in self.prefixes}

        # Keep only attributes exposed by each truth/reconstruction class
        for prefix in self.prefixes:
            class_name = f"{prefix.capitalize()}{obj_type[:-1].capitalize()}"
            class_obj = getattr(spine.data.out, class_name)()
            valid_attrs = set(class_obj.attr_names())
            attrs[prefix] = [name for name in req_attrs if name in valid_attrs]
            found_attrs.update(attrs[prefix])

        # Reject attributes unsupported by every requested object declination
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
        attrs = self.validate_request(obj_type, attr)

        traces: dict[str, list] = {}
        for prefix in self.prefixes:
            obj_name = f"{prefix}_{obj_type}"
            if obj_name not in self.data:
                raise ValueError(
                    f"Must provide `{obj_name}` in the data products to draw them."
                )
            traces[prefix] = self.build_object_traces(
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
                colors = self.resolve_aux_colors(obj_name, attrs[prefix], color_attr)
                color_kwargs = self._aux_color_kwargs(colors)
                traces[prefix] += build_start_point_trace(
                    data=self.data,
                    obj_name=obj_name,
                    split_traces=split_traces,
                    truth_index_mode=self.truth_index_mode,
                    **color_kwargs,
                )
                traces[prefix] += build_end_point_trace(
                    data=self.data,
                    obj_name=obj_name,
                    split_traces=split_traces,
                    truth_index_mode=self.truth_index_mode,
                    **color_kwargs,
                )

        # Draw directions as a separate trace, if requested
        if draw_directions:
            if obj_type == "interactions":
                raise ValueError("Interactions do not have direction attributes.")
            for prefix in self.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                colors = self.resolve_aux_colors(obj_name, attrs[prefix], color_attr)
                traces[prefix] += build_direction_trace(
                    data=self.data,
                    obj_name=obj_name,
                    split_traces=split_traces,
                    truth_index_mode=self.truth_index_mode,
                    **self._aux_color_kwargs(colors),
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
                    **self._aux_color_kwargs(
                        self.resolve_aux_colors(
                            obj_name,
                            attrs[prefix] if obj_type == "interactions" else [],
                            color_attr if obj_type == "interactions" else None,
                        )
                    ),
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
            geo_traces = self.build_geometry_traces()
            if self.prefixes and self.split_scene:
                for prefix in self.prefixes:
                    traces[prefix] += geo_traces
            else:
                traces[self.prefixes[-1]] += geo_traces

        # Build the layout and assemble the requested traces and geometry overlay
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

    def build_object_traces(
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
        # Build the color mapping from the requested or inferred color attribute
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
