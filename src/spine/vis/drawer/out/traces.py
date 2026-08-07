"""Trace-building helpers for output-object drawers."""

from __future__ import annotations

from typing import Any

import numpy as np
from plotly import graph_objs as go

from spine.constants import TRACK_SHP

from ...trace.arrow import scatter_arrows
from ...trace.point import scatter_points_3d

__all__ = [
    "build_crt_trace",
    "build_direction_trace",
    "build_end_point_trace",
    "build_flash_trace",
    "build_flash_hypothesis_trace",
    "get_flash_pe",
    "get_flash_hypothesis_pe",
    "build_point_trace",
    "build_raw_trace",
    "build_start_point_trace",
    "build_vertex_trace",
]


def build_raw_trace(
    *,
    data: dict[str, Any],
    prefix: str,
    prefixes: list[str],
    truth_point_key: str,
    truth_dep_key: str,
    lite: bool,
) -> list[go.Scatter3d]:
    """Draw the raw reconstruction input for one prefix.

    Parameters
    ----------
    data : Dict[str, Any]
        Full event dictionary.
    prefix : str
        Object declination to draw, one of ``"reco"`` or ``"truth"``.
    prefixes : List[str]
        Allowed declinations configured by the parent drawer.
    truth_point_key : str
        Data key pointing at the truth point cloud currently being drawn.
    truth_dep_key : str
        Data key pointing at the truth deposition array currently being drawn.
    lite : bool
        If ``True``, raw point clouds are not available.

    Returns
    -------
    List[go.Scatter3d]
        Single raw-input trace wrapped in a list for consistency.
    """
    if lite:
        raise RuntimeError("Cannot draw raw input in lite mode.")

    if prefix == "reco":
        if "points" not in data or "depositions" not in data:
            raise ValueError(
                "Must provide `points` and `depositions` to draw the raw input."
            )
        points = data["points"]
        deps = data["depositions"]
    elif prefix == "truth":
        if truth_point_key not in data or truth_dep_key not in data:
            raise ValueError(
                f"Must provide `{truth_point_key}` and `{truth_dep_key}` to draw the raw input."
            )
        points = data[truth_point_key]
        deps = data[truth_dep_key]
    else:
        raise ValueError(f"Prefix not recognized: {prefix}. Must be one of {prefixes}.")

    # Match the legacy raw-input display scaling so the dynamic range remains
    # comparable to the pre-refactor plots.
    cmin = 0.0
    cmax = 2 * np.median(deps) if len(deps) else 1.0

    return scatter_points_3d(
        points,
        color=deps,
        cmin=cmin,
        cmax=cmax,
        colorscale="Inferno",
        name="Raw input",
    )


def build_start_point_trace(
    *,
    data: dict[str, Any],
    obj_name: str,
    split_traces: bool,
    truth_index_mode: str,
    color: str | np.ndarray = "black",
    markersize: float = 7,
    marker_symbol: str = "circle",
    **kwargs: Any,
) -> list[go.Scatter3d]:
    """Scatter object start points.

    Parameters
    ----------
    data : Dict[str, Any]
        Full event dictionary.
    obj_name : str
        Object collection name.
    split_traces : bool
        If ``True``, emit one trace per object.
    truth_index_mode : str
        Truth-object index attribute used to skip empty truth objects.
    color : Union[str, np.ndarray], default ``"black"``
        Marker color specification.
    markersize : float, default 7
        Marker size in pixels.
    marker_symbol : str, default ``"circle"``
        Plotly marker symbol.
    **kwargs : Any
        Additional trace keyword arguments forwarded to
        :func:`spine.vis.trace.point.scatter_points`.

    Returns
    -------
    List[go.Scatter3d]
        Start-point traces.
    """
    return build_point_trace(
        data=data,
        obj_name=obj_name,
        point_attr="start_point",
        split_traces=split_traces,
        truth_index_mode=truth_index_mode,
        color=color,
        markersize=markersize,
        marker_symbol=marker_symbol,
        **kwargs,
    )


def build_end_point_trace(
    *,
    data: dict[str, Any],
    obj_name: str,
    split_traces: bool,
    truth_index_mode: str,
    color: str | np.ndarray = "black",
    markersize: float = 7,
    marker_symbol: str = "circle-open",
    **kwargs: Any,
) -> list[go.Scatter3d]:
    """Scatter object end points.

    Parameters
    ----------
    data : Dict[str, Any]
        Full event dictionary.
    obj_name : str
        Object collection name.
    split_traces : bool
        If ``True``, emit one trace per object.
    truth_index_mode : str
        Truth-object index attribute used to skip empty truth objects.
    color : Union[str, np.ndarray], default ``"black"``
        Marker color specification.
    markersize : float, default 7
        Marker size in pixels.
    marker_symbol : str, default ``"circle-open"``
        Plotly marker symbol.
    **kwargs : Any
        Additional trace keyword arguments forwarded to
        :func:`spine.vis.trace.point.scatter_points`.

    Returns
    -------
    List[go.Scatter3d]
        End-point traces.
    """
    return build_point_trace(
        data=data,
        obj_name=obj_name,
        point_attr="end_point",
        split_traces=split_traces,
        truth_index_mode=truth_index_mode,
        color=color,
        markersize=markersize,
        marker_symbol=marker_symbol,
        **kwargs,
    )


def build_vertex_trace(
    *,
    data: dict[str, Any],
    obj_name: str,
    split_traces: bool,
    truth_index_mode: str,
    vertex_attr: str = "vertex",
    color: str | np.ndarray = "green",
    markersize: float = 10,
    marker_symbol: str = "diamond",
    **kwargs: Any,
) -> list[go.Scatter3d]:
    """Scatter interaction vertices.

    Parameters
    ----------
    data : Dict[str, Any]
        Full event dictionary.
    obj_name : str
        Object collection name.
    split_traces : bool
        If ``True``, emit one trace per object.
    truth_index_mode : str
        Truth-object index attribute used to skip empty truth objects.
    vertex_attr : str, default ``"vertex"``
        Attribute containing the interaction vertex.
    color : Union[str, np.ndarray], default ``"green"``
        Marker color specification.
    markersize : float, default 10
        Marker size in pixels.
    marker_symbol : str, default ``"diamond"``
        Plotly marker symbol.
    **kwargs : Any
        Additional trace keyword arguments forwarded to
        :func:`spine.vis.trace.point.scatter_points`.

    Returns
    -------
    List[go.Scatter3d]
        Vertex traces.
    """
    return build_point_trace(
        data=data,
        obj_name=obj_name,
        point_attr=vertex_attr,
        split_traces=split_traces,
        truth_index_mode=truth_index_mode,
        color=color,
        markersize=markersize,
        marker_symbol=marker_symbol,
        **kwargs,
    )


def build_point_trace(
    *,
    data: dict[str, Any],
    obj_name: str,
    point_attr: str,
    split_traces: bool,
    truth_index_mode: str,
    **kwargs: Any,
) -> list[go.Scatter3d]:
    """Scatter one point-valued attribute per object.

    Parameters
    ----------
    data : Dict[str, Any]
        Full event dictionary.
    obj_name : str
        Object collection name.
    point_attr : str
        Attribute containing the point coordinate to display.
    split_traces : bool
        If ``True``, emit one trace per object.
    truth_index_mode : str
        Truth-object index attribute used to skip empty truth objects.
    **kwargs : Any
        Additional trace keyword arguments forwarded to
        :func:`spine.vis.trace.point.scatter_points`.

    Returns
    -------
    List[go.Scatter3d]
        Point traces.
    """
    name = (
        " ".join(obj_name.split("_")).capitalize()[:-1]
        + " "
        + " ".join(point_attr.split("_"))
    )

    # Collect only the points that should actually be displayed. For example,
    # shower-like particles do not have meaningful end points.
    obj_type = obj_name.split("_")[-1][:-1].capitalize()
    point_list, hovertext, idxs = [], [], []
    for i, obj in enumerate(data[obj_name]):
        if point_attr == "end_point" and obj.shape != TRACK_SHP:
            continue
        if obj.is_truth and not len(getattr(obj, truth_index_mode)) > 0:
            continue

        point_list.append(getattr(obj, point_attr))
        hovertext.append(f"{obj_type} {i} " + " ".join(point_attr.split("_")))
        idxs.append(i)

    points = np.empty((0, 3))
    if point_list:
        points = np.vstack(point_list)

    # Preserve the legacy API shape: a single combined trace by default, with
    # an optional split path for per-object toggling in Plotly.
    if not split_traces:
        return scatter_points_3d(
            points, hovertext=np.array(hovertext), name=name, **kwargs
        )

    traces: list[go.Scatter3d] = []
    for i, point in enumerate(point_list):
        traces += scatter_points_3d(
            point[None, :],
            hovertext=hovertext[i],
            name=f"{name} {idxs[i]}",
            **kwargs,
        )

    return traces


def build_direction_trace(
    *,
    data: dict[str, Any],
    obj_name: str,
    split_traces: bool,
    truth_index_mode: str,
    color: str | np.ndarray = "black",
    **kwargs: Any,
) -> list[go.Scatter3d | go.Cone]:
    """Scatter or split start-direction arrows for one object collection.

    Parameters
    ----------
    data : Dict[str, Any]
        Full event dictionary.
    obj_name : str
        Object collection name.
    split_traces : bool
        If ``True``, emit one trace per object.
    truth_index_mode : str
        Truth-object index attribute used to skip empty truth objects.
    color : Union[str, np.ndarray], default ``"black"``
        Arrow color specification.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :func:`spine.vis.trace.arrow.scatter_arrows`.

    Returns
    -------
    List[go.Scatter3d | go.Cone]
        Direction-arrow traces.
    """
    name = " ".join(obj_name.split("_")).capitalize()[:-1] + " directions"

    obj_type = obj_name.split("_")[-1][:-1].capitalize()
    point_list, dir_list, hovertext, idxs = [], [], [], []
    for i, obj in enumerate(data[obj_name]):
        if obj.is_truth and not len(getattr(obj, truth_index_mode)) > 0:
            continue
        point_list.append(obj.start_point)
        dir_list.append(obj.start_dir)
        hovertext.append(f"{obj_type} {i} direction")
        idxs.append(i)

    points, dirs = np.empty((0, 3)), np.empty((0, 3))
    if point_list:
        points = np.vstack(point_list)
        dirs = np.vstack(dir_list)

    if not split_traces:
        return scatter_arrows(
            points,
            dirs,
            hovertext=np.array(hovertext),
            name=name,
            color=color,
            **kwargs,
        )

    traces: list[go.Scatter3d | go.Cone] = []
    for i, (point, start_dir) in enumerate(zip(point_list, dir_list)):
        traces += scatter_arrows(
            point[None, :],
            start_dir[None, :],
            color=color,
            hovertext=hovertext[i],
            name=f"{name} {idxs[i]}",
            **kwargs,
        )

    return traces


def _aggregate_optical_pe(objects: list[Any], geo: Any) -> np.ndarray:
    """Aggregate per-channel optical responses onto physical detectors."""
    if geo is None or geo.optical is None:
        raise RuntimeError("This geometry does not have optical detectors to draw.")

    pe = np.zeros(geo.optical.num_detectors, dtype=np.float32)
    for obj in objects:
        pe_per_ch = np.asarray(obj.pe_per_ch, dtype=np.float32)
        if geo.optical.global_index:
            index = np.arange(geo.optical.num_detectors)
            det_ids = geo.optical.det_ids
        else:
            index = geo.optical.volume_index(obj.volume_id)
            det_ids = geo.optical.volumes[obj.volume_id].det_ids

        if det_ids is not None:
            pe_per_ch = np.bincount(
                det_ids,
                weights=pe_per_ch,
                minlength=len(index),
            )
        if len(pe_per_ch) != len(index):
            raise ValueError(
                "Optical response length does not match the detector geometry "
                f"for volume {obj.volume_id}."
            )
        pe[index] += pe_per_ch

    return pe


def get_flash_pe(
    data: dict[str, Any], obj_name: str, matched_only: bool, geo: Any
) -> np.ndarray:
    """Return observed PE accumulated over the requested flashes."""
    if "flashes" not in data:
        raise ValueError("Must provide the `flashes` objects to draw them.")

    if matched_only:
        flash_ids = []
        for inter in data[obj_name]:
            if inter.is_flash_matched:
                flash_ids.extend(inter.flash_ids)
    else:
        flash_ids = np.arange(len(data["flashes"]))

    flashes = [data["flashes"][flash_id] for flash_id in flash_ids]
    return _aggregate_optical_pe(flashes, geo)


def get_flash_hypothesis_pe(
    data: dict[str, Any], obj_name: str, hypothesis_key: str, geo: Any
) -> np.ndarray:
    """Return predicted PE accumulated over hypotheses for selected interactions."""
    if hypothesis_key not in data:
        raise ValueError(
            f"Must provide the `{hypothesis_key}` objects to draw hypotheses."
        )

    hypothesis_ids = []
    for inter in data[obj_name]:
        for hypothesis_id in getattr(inter, "flash_hypothesis_ids", []):
            hypothesis = data[hypothesis_key][hypothesis_id]
            if not hypothesis.is_matched or np.all(
                np.isin(hypothesis.flash_ids, inter.flash_ids)
            ):
                hypothesis_ids.append(hypothesis_id)
    hypotheses = [data[hypothesis_key][hypo_id] for hypo_id in hypothesis_ids]
    return _aggregate_optical_pe(hypotheses, geo)


def _optical_size_scale(
    pe: np.ndarray, size_by_pe: bool, pe_max: float | None, base_scale: float
) -> float | np.ndarray:
    """Build bounded detector size scales from PE values."""
    if not size_by_pe:
        return base_scale

    if pe_max is None:
        pe_max = float(np.max(pe, initial=0.0))
    if pe_max <= 0.0:
        return np.full(len(pe), base_scale, dtype=np.float32)

    normalized = np.sqrt(np.clip(pe / pe_max, 0.0, 1.0))
    return base_scale * (0.25 + 0.75 * normalized)


def build_flash_trace(
    *,
    data: dict[str, Any],
    obj_name: str,
    matched_only: bool,
    geo: Any | None,
    geo_drawer: Any | None,
    meta: Any | None,
    size_by_pe: bool = False,
    pe_max: float | None = None,
    pe_per_detector: np.ndarray | None = None,
    **kwargs: Any,
) -> list:
    """Draw cumulative optical flash charge on detector elements.

    Parameters
    ----------
    data : Dict[str, Any]
        Full event dictionary.
    obj_name : str
        Interaction collection name used to identify matched flashes.
    matched_only : bool
        If ``True``, only flashes matched to the interactions are included.
    geo : Any, optional
        Detector geometry.
    geo_drawer : Any, optional
        Geometry drawer used to build optical detector traces.
    meta : Any, optional
        Metadata used for detector-to-pixel coordinate conversion.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :meth:`spine.vis.drawer.geo.GeoDrawer.optical_traces`.

    Returns
    -------
    list
        Optical-detector traces colored by accumulated PE counts.
    """
    if geo_drawer is None:
        raise RuntimeError(
            "Cannot draw optical detectors without geometry information."
        )
    if geo is None or geo.optical is None:
        raise RuntimeError("This geometry does not have optical detectors to draw.")
    name = " ".join(obj_name.split("_")).capitalize()[:-1] + " flashes"
    if pe_per_detector is None:
        pe_per_detector = get_flash_pe(data, obj_name, matched_only, geo)
    size_scale = _optical_size_scale(
        pe_per_detector, size_by_pe, pe_max, base_scale=1.0
    )

    return geo_drawer.optical_traces(
        meta=meta,
        color=pe_per_detector,
        size_scale=size_scale,
        zero_supress=True,
        colorscale="Inferno",
        name=name,
        **kwargs,
    )


def build_flash_hypothesis_trace(
    *,
    data: dict[str, Any],
    obj_name: str,
    hypothesis_key: str,
    geo: Any | None,
    geo_drawer: Any | None,
    meta: Any | None,
    size_by_pe: bool = False,
    pe_max: float | None = None,
    pe_per_detector: np.ndarray | None = None,
    **kwargs: Any,
) -> list:
    """Draw predicted optical charge for the selected interactions."""
    if geo_drawer is None:
        raise RuntimeError(
            "Cannot draw optical detectors without geometry information."
        )
    if geo is None or geo.optical is None:
        raise RuntimeError("This geometry does not have optical detectors to draw.")

    if pe_per_detector is None:
        pe_per_detector = get_flash_hypothesis_pe(data, obj_name, hypothesis_key, geo)
    size_scale = _optical_size_scale(
        pe_per_detector, size_by_pe, pe_max, base_scale=0.65
    )
    name = " ".join(obj_name.split("_")).capitalize()[:-1] + " hypotheses"
    return geo_drawer.optical_traces(
        meta=meta,
        color=pe_per_detector,
        size_scale=size_scale,
        zero_supress=True,
        colorscale="Viridis",
        name=name,
        **kwargs,
    )


def build_crt_trace(
    *,
    data: dict[str, Any],
    obj_name: str,
    matched_only: bool,
    geo: Any | None,
    geo_drawer: Any | None,
    meta: Any | None,
    **kwargs: Any,
) -> list:
    """Draw hit CRT planes and hit centers for one object collection.

    Parameters
    ----------
    data : Dict[str, Any]
        Full event dictionary.
    obj_name : str
        Object collection name used to identify matched CRT hits.
    matched_only : bool
        If ``True``, only hits matched to the objects are included.
    geo : Any, optional
        Detector geometry.
    geo_drawer : Any, optional
        Geometry drawer used to build CRT plane traces.
    meta : Any, optional
        Metadata used for detector-to-pixel coordinate conversion.
    **kwargs : Any
        Additional keyword arguments forwarded to
        :func:`spine.vis.trace.point.scatter_points`.

    Returns
    -------
    list
        CRT plane traces followed by hit-center point traces.
    """
    if geo is None or geo_drawer is None:
        raise RuntimeError("Cannot draw CRT detectors without geometry information.")
    if geo.crt is None:
        raise RuntimeError("This geometry does not have CRT planes to draw.")

    name_pl = " ".join(obj_name.split("_")).capitalize()[:-1] + " CRT planes"
    name_hits = " ".join(obj_name.split("_")).capitalize()[:-1] + " CRT hits"

    # When requested, collapse to the subset of hits actually referenced by
    # the selected output objects.
    crthits = data["crthits"]
    if matched_only:
        crt_ids = []
        for inter in data[obj_name]:
            if inter.is_crt_matched:
                crt_ids.extend(inter.crt_ids)
        crt_ids = np.unique(crt_ids)
        crthits = [crthits[idx] for idx in crt_ids]

    det_ids = [geo.crt.get_plane_id(hit.center, hit.plane) for hit in crthits]
    unique_det_ids = list(np.unique(det_ids))

    hovertext_pl = [f"CRT Plane {det_id}" for det_id in unique_det_ids]
    hovertext_hits = [
        f"CRT hit {hit.id}<br>CRT Plane ID: {det_ids[i]}"
        for i, hit in enumerate(crthits)
    ]

    traces = geo_drawer.crt_traces(
        meta=meta,
        draw_ids=unique_det_ids,
        hovertext=hovertext_pl,
        name=name_pl,
    )

    points = np.empty((0, 3))
    if crthits:
        points = np.vstack([hit.center for hit in crthits])

    traces += scatter_points_3d(
        points,
        color="gray",
        markersize=5,
        hovertext=hovertext_hits,
        name=name_hits,
        **kwargs,
    )

    return traces
