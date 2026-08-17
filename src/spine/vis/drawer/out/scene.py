"""Renderer-neutral scene construction for output objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...scene import Scene, SceneView
from .layers import SceneLayerBuilder
from .traces import (
    build_crt_trace,
    build_flash_hypothesis_trace,
    build_flash_trace,
    get_flash_hypothesis_pe,
    get_flash_pe,
)

if TYPE_CHECKING:
    from .drawer import Drawer

__all__ = ["SceneBuilder"]


class SceneBuilder:
    """Build renderer-neutral scenes from a configured output drawer.

    The builder owns neutral layer assembly while :class:`Drawer` remains
    the public configuration and Plotly compatibility façade.

    Parameters
    ----------
    drawer : Drawer
        Configured output drawer which supplies event data and display modes.
    """

    def __init__(self, drawer: Drawer) -> None:
        """Initialize the scene and layer builders.

        Parameters
        ----------
        drawer : Drawer
            Configured output drawer which owns the scene inputs.
        """
        self.drawer = drawer
        self.layers = SceneLayerBuilder(drawer)

    def build(
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
        """Build a renderer-neutral scene for all event-display primitives.

        Unlike :meth:`Drawer.get`, this method does not instantiate Plotly objects.
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
        draw_end_points : bool, default False
            If ``True``, draw object start and end markers.
        draw_directions : bool, default False
            If ``True``, draw object start-direction vectors.
        draw_vertices : bool, default False
            If ``True``, draw interaction vertex markers.
        draw_flashes : bool, default False
            If ``True``, draw measured optical responses.
        draw_flash_hypotheses : bool, default False
            If ``True``, draw predicted optical responses.
        matched_flash_only : bool, default True
            If ``True``, restrict measured flashes to object matches.
        optical_size_by_pe : bool, default False
            If ``True``, scale optical glyphs by photoelectron count.
        flash_hypothesis_key : str, default ``"flash_hypotheses"``
            Event-data key containing optical hypotheses.
        draw_crthits : bool, default False
            If ``True``, draw CRT hits and planes.
        matched_crthit_only : bool, default True
            If ``True``, restrict CRT hits to object matches.

        Returns
        -------
        Scene
            Renderer-neutral scene with one view or split reco/truth views.

        Raises
        ------
        ValueError
            If the requested object collection or backing point data is absent.
        RuntimeError
            If raw input is requested for lite objects.
        """
        # Normalize the request using the same validation as the Plotly path
        attrs = self.drawer.validate_request(obj_type, attr)

        # Build one compact object layer, plus optional raw input, per prefix.
        # Lite representations are adapted into the same neutral primitives.
        layers = {}
        for prefix in self.drawer.prefixes:
            obj_name = f"{prefix}_{obj_type}"
            if obj_name not in self.drawer.data:
                raise ValueError(
                    f"Must provide `{obj_name}` in the data products to draw them."
                )
            if self.drawer.lite:
                traces = self.drawer.build_object_traces(
                    obj_name, attrs[prefix], color_attr, split_traces=False
                )
                layers[prefix] = self.layers.neutralize_traces(
                    traces, kind="objects", object_name=obj_name
                )
            else:
                layers[prefix] = [
                    self.layers.object_layer(
                        obj_name, attrs[prefix], color_attr=color_attr
                    )
                ]
            if draw_raw:
                if self.drawer.lite:
                    raise RuntimeError("Cannot draw raw input in lite mode.")
                layers[prefix].insert(0, self.layers.raw_layer(prefix))

        # Add discrete object markers and vector glyphs.
        if draw_end_points:
            if obj_type == "interactions":
                raise ValueError("Interactions do not have end point attributes.")
            for prefix in self.drawer.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                colors = self.drawer.resolve_aux_colors(
                    obj_name, attrs[prefix], color_attr
                )
                layers[prefix] += [
                    self.layers.marker_layer(obj_name, "start_point", colors=colors),
                    self.layers.marker_layer(obj_name, "end_point", colors=colors),
                ]

        if draw_directions:
            if obj_type == "interactions":
                raise ValueError("Interactions do not have direction attributes.")
            for prefix in self.drawer.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                colors = self.drawer.resolve_aux_colors(
                    obj_name, attrs[prefix], color_attr
                )
                layers[prefix].append(self.layers.vector_layer(obj_name, colors=colors))

        if draw_vertices:
            for prefix in self.drawer.prefixes:
                obj_name = f"{prefix}_interactions"
                if obj_name not in self.drawer.data:
                    raise ValueError(
                        "Must provide interactions to draw their vertices."
                    )
                layers[prefix].append(
                    self.layers.marker_layer(
                        obj_name,
                        "vertex",
                        color="green",
                        colors=self.drawer.resolve_aux_colors(
                            obj_name,
                            attrs[prefix] if obj_type == "interactions" else [],
                            color_attr if obj_type == "interactions" else None,
                        ),
                    )
                )

        # Optical and CRT helpers already encode detector-specific glyph
        # geometry; adapt their meshes and markers at the neutral boundary.
        show_optical = draw_flashes or draw_flash_hypotheses
        if draw_flashes and "flashes" not in self.drawer.data:
            raise ValueError("Must provide the `flashes` objects to draw them.")
        if draw_flash_hypotheses and flash_hypothesis_key not in self.drawer.data:
            raise ValueError(
                f"Must provide the `{flash_hypothesis_key}` objects to draw "
                "hypotheses."
            )
        if show_optical:
            for prefix in self.drawer.prefixes:
                obj_name = f"{prefix}_interactions"
                if obj_name not in self.drawer.data:
                    raise ValueError(
                        "Must provide interactions to draw matched flashes or "
                        "optical hypotheses."
                    )
                flash_pe = (
                    get_flash_pe(
                        self.drawer.data, obj_name, matched_flash_only, self.drawer.geo
                    )
                    if draw_flashes
                    else None
                )
                hypothesis_pe = (
                    get_flash_hypothesis_pe(
                        self.drawer.data,
                        obj_name,
                        flash_hypothesis_key,
                        self.drawer.geo,
                    )
                    if draw_flash_hypotheses
                    else None
                )

                # Normalize measured and predicted PE against the same maximum
                # so their colors and optional detector sizes remain comparable.
                pe_arrays = [pe for pe in (flash_pe, hypothesis_pe) if pe is not None]
                pe_max = max(
                    (float(np.max(pe, initial=0.0)) for pe in pe_arrays),
                    default=0.0,
                )
                cmax = pe_max if pe_max > 0.0 else 1.0
                traces = []
                if draw_flashes:
                    traces += build_flash_trace(
                        data=self.drawer.data,
                        obj_name=obj_name,
                        matched_only=matched_flash_only,
                        geo=self.drawer.geo,
                        geo_drawer=self.drawer.geo_drawer,
                        meta=self.drawer.meta,
                        size_by_pe=optical_size_by_pe,
                        pe_max=pe_max,
                        pe_per_detector=flash_pe,
                        cmin=0.0,
                        cmax=cmax,
                        opacity=0.55 if draw_flash_hypotheses else 1.0,
                    )
                if draw_flash_hypotheses:
                    traces += build_flash_hypothesis_trace(
                        data=self.drawer.data,
                        obj_name=obj_name,
                        hypothesis_key=flash_hypothesis_key,
                        geo=self.drawer.geo,
                        geo_drawer=self.drawer.geo_drawer,
                        meta=self.drawer.meta,
                        size_by_pe=optical_size_by_pe,
                        pe_max=pe_max,
                        pe_per_detector=hypothesis_pe,
                        cmin=0.0,
                        cmax=cmax,
                        opacity=0.8,
                    )
                layers[prefix] += self.layers.neutralize_traces(
                    traces, kind="optical", object_name=obj_name
                )

        # CRT layers use the selected object family for match filtering.
        show_crt = False
        if draw_crthits:
            if "crthits" not in self.drawer.data:
                raise ValueError("Must provide the `crthits` objects to draw them.")
            show_crt = True
            for prefix in self.drawer.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                traces = build_crt_trace(
                    data=self.drawer.data,
                    obj_name=obj_name,
                    matched_only=matched_crthit_only,
                    geo=self.drawer.geo,
                    geo_drawer=self.drawer.geo_drawer,
                    meta=self.drawer.meta,
                )
                layers[prefix] += self.layers.neutralize_traces(
                    traces, kind="crt", object_name=obj_name
                )

        # Detector geometry is a set of neutral line and mesh layers. Match the
        # established behavior by repeating it in split views only.
        if self.drawer.geo_drawer is not None:
            geo_traces = self.drawer.build_geometry_traces()
            geo_layers = self.layers.neutralize_traces(geo_traces, kind="geometry")
            if self.drawer.prefixes and self.drawer.split_scene:
                for prefix in self.drawer.prefixes:
                    layers[prefix] += geo_layers
            else:
                layers[self.drawer.prefixes[-1]] += geo_layers

        # Preserve independent truth and reconstruction views when requested
        if len(self.drawer.prefixes) > 1 and self.drawer.split_scene:
            views = [
                SceneView(
                    name=f"{prefix.capitalize()} {obj_type}",
                    layers=layers[prefix],
                    metadata={"prefix": prefix},
                )
                for prefix in self.drawer.prefixes
            ]
        else:
            # Merge layer groups into one view without merging their buffers
            views = [
                SceneView(
                    name=obj_type.capitalize(),
                    layers=[
                        layer
                        for prefix in self.drawer.prefixes
                        for layer in layers[prefix]
                    ],
                )
            ]

        # Store physical bounds separately from padded Plotly layout bounds.
        bounds, layout_bounds = None, None
        if self.drawer.geo is not None:
            bounds_array = self.drawer.geo.get_boundaries(
                with_optical=show_optical, with_crt=show_crt
            )
            bounds = bounds_array.tolist()
            layout_bounds = np.asarray(bounds_array, dtype=np.float64).copy()
            padding = self.drawer.layout_kwargs.get("detector_padding", 0.1)
            lengths = layout_bounds[:, 1] - layout_bounds[:, 0]
            layout_bounds[:, 0] -= padding * lengths
            layout_bounds[:, 1] += padding * lengths
            layout_bounds = layout_bounds.tolist()
        elif self.drawer.meta is not None:
            if self.drawer.detector_coords:
                bounds_array = np.vstack(
                    (self.drawer.meta.lower, self.drawer.meta.upper)
                ).T
            else:
                bounds_array = np.vstack(
                    (
                        np.zeros(3),
                        np.round(
                            (self.drawer.meta.upper - self.drawer.meta.lower)
                            / self.drawer.meta.size
                        ),
                    )
                ).T
            bounds = bounds_array.tolist()
            layout_bounds = bounds

        # Attach domain context without introducing backend-specific objects
        return Scene(
            views=views,
            metadata={
                "object_type": obj_type,
                "split_scene": self.drawer.split_scene,
                "detector_coords": self.drawer.detector_coords,
                "bounds": bounds,
                "layout_bounds": layout_bounds,
                "up_dir": (
                    np.asarray(
                        getattr(self.drawer.geo, "up_dir", [0.0, 1.0, 0.0])
                    ).tolist()
                    if self.drawer.geo is not None
                    else [0.0, 1.0, 0.0]
                ),
            },
        )
