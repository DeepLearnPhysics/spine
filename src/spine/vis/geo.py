"""Draw detectors based on their geometry definition."""

import time
from typing import List, Optional, Union

import numpy as np
import plotly.graph_objs as go
from plotly.basedatatypes import BaseTraceType

from spine.data import Meta
from spine.geo import GeoManager, Geometry
from spine.vis.cylinder import cylinder_traces

from .box import box_traces
from .ellipsoid import ellipsoid_traces
from .layout import layout3d

__all__ = ["GeoDrawer"]


class GeoDrawer:
    """Handles drawing all things related to the detector geometry.

    This class is loads a :class:`Geometry` object once from a geometry file
    and uses it to represent all things related to the detector geometry:
    - TPC boundaries
    - Optical detectors
    - CRT detectors

    Attributes
    ----------
    geo : Geometry
        The underlying detector geometry
    detector_coords : bool
        Whether or not to use detector coordinates (True) or pixel indices
    """

    def __init__(self, geo: Optional[Geometry] = None, detector_coords: bool = True):
        """Initializes the underlying detector :class:`Geometry` object.

        Parameters
        ----------
        geo : Geometry, optional
            If provided, this Geometry instance is used.
            If None, the global GeoManager instance is used.
        detector_coords : bool, default False
            If False, the coordinates are converted to pixel indices
        """
        # Fetch the geometry instance, if need be
        if geo is None:
            self.geo = GeoManager.get_instance()
        else:
            assert isinstance(
                geo, Geometry
            ), "The `geo` parameter must be a Geometry instance."
            self.geo = geo

        # Store whether to use detector coordinates or not
        self.detector_coords = detector_coords

    def show(
        self,
        meta: Optional[Meta] = None,
        tpc: bool = True,
        optical: bool = True,
        crt: bool = True,
        **kwargs,
    ):
        """Displays the detector geometry in a 3D plotly figure.

        Parameters
        ----------
        meta : Meta, optional
            Metadata information (only needed if pixel_coordinates is True)
        tpc : bool, default True
            Whether or not to include TPC traces
        optical : bool, default True
            Whether or not to include optical detector traces
        crt : bool, default True
            Whether or not to include CRT detector traces
        **kwargs : dict, optional
            Additional arguments to pass to layout3d
        """

        # Get all the detector traces
        traces = self.traces(
            meta=meta,
            tpc=tpc,
            optical=optical,
            crt=crt,
        )

        # Initialize the layout
        layout = layout3d(
            geo=self.geo,
            use_geo=True,
            meta=meta,
            detector_coords=self.detector_coords,
            show_optical=optical and self.geo.optical is not None,
            show_crt=crt and self.geo.crt is not None,
            **kwargs,
        )

        # Build the 3D layout
        fig = go.Figure(data=traces, layout=layout)

        # Show the figure
        fig.show()

    def traces(
        self,
        meta: Optional[Meta] = None,
        tpc: bool = True,
        optical: bool = True,
        crt: bool = True,
    ) -> List[BaseTraceType]:
        """Returns all traces associated with the detector geometry.

        Parameters
        ----------
        meta : Meta, optional
            Metadata information (only needed if pixel_coordinates is True)
        tpc : bool, default True
            Whether or not to include TPC traces
        optical : bool, default True
            Whether or not to include optical detector traces
        crt : bool, default True
            Whether or not to include CRT detector traces

        Returns
        -------
        List[BaseTraceType]
            List of detector traces
        """
        traces = []
        if tpc:
            traces += self.tpc_traces(meta=meta)
        if optical and self.geo.optical is not None:
            traces += self.optical_traces(meta=meta)
        if crt and self.geo.crt is not None:
            traces += self.crt_traces(meta=meta)

        return traces

    def tpc_traces(
        self,
        meta: Optional[Meta] = None,
        draw_faces: bool = False,
        shared_legend: bool = True,
        name: str = "TPC",
        color: Union[int, str, np.ndarray] = "rgba(0,0,0,0.150)",
        linewidth: int = 5,
        **kwargs,
    ) -> List[BaseTraceType]:
        """Function which produces a list of traces which represent the TPCs in
        a 3D event display.

        Parameters
        ----------
        meta : Meta, optional
            Metadata information (only needed if pixel_coordinates is True)
        draw_faces : bool, default False
            Weather or not to draw the box faces, or only the edges
        shared_legend : bool, default True
            If True, the legend entry in plotly is shared between all the
            TPC volumes
        name : str, default 'TPC'
            Name of the TPC volumes
        color : Union[int, str, np.ndarray]
            Color of boxes or list of color of boxes
        linewidth : int, default 2
            Width of the box edge lines
        **kwargs : dict, optional
            List of additional arguments to pass to
            spine.viusalization.boxes.box_traces

        Returns
        -------
        List[BaseTraceType]
            List of detector traces (one per TPC)
        """
        # Load the list of TPC boundaries
        boundaries = np.stack([c.boundaries for c in self.geo.tpc.chambers])

        # If required, convert to pixel coordinates
        if not self.detector_coords:
            assert meta is not None, (
                "Must provide meta information to convert the TPC "
                "boundaries to pixel coordinates."
            )
            boundaries = meta.to_px(boundaries.transpose(0, 2, 1)).transpose(0, 2, 1)

        # Get a trace per TPC volume
        detectors = box_traces(
            boundaries[..., 0],
            boundaries[..., 1],
            draw_faces=draw_faces,
            color=color,
            linewidth=linewidth,
            shared_legend=shared_legend,
            name=name,
            **kwargs,
        )

        return detectors

    def optical_traces(
        self,
        meta: Optional[Meta] = None,
        shared_legend: bool = True,
        legendgroup: Optional[str] = None,
        name: str = "Optical",
        color: Union[int, str, np.ndarray] = "rgba(0,0,255,0.25)",
        hovertext: Optional[Union[str, List[str]]] = None,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        zero_supress: bool = False,
        volume_id: Optional[int] = None,
        **kwargs,
    ) -> List[BaseTraceType]:
        """Function which produces a list of traces which represent the optical
        detectors in a 3D event display.

        Parameters
        ----------
        meta : Meta, optional
            Metadata information (only needed if pixel_coordinates is True)
        shared_legend : bool, default True
            If True, the legend entry in plotly is shared between all the
            optical volumes
        legendgroup : str, optional
            Legend group to be shared between all boxes
        name : str, default 'Optical'
            Name of the optical volumes
        color : Union[int, str, np.ndarray]
            Color of optical detectors or list of color of optical detectors
        hovertext : Union[str, List[str]], optional
            Label or list of labels associated with each optical detector
        cmin : float, optional
            Minimum value along the color scale
        cmax : float, optional
            Maximum value along the color scale
        zero_supress : bool, default False
            If `True`, do not draw optical detectors that are not activated
        volume_id : int, optional
            Specifies which optical volume to represent. If not specified, all
            the optical volumes are drawn
        **kwargs : dict, optional
            List of additional arguments to pass to
            spine.vis.ellipsoid.ellipsoid_traces or spine.vis.box.box_traces

        Returns
        -------
        List[plotly.graph_objs.Mesh3D]
            List of optical detector traces (one per optical detector)
        """
        # Check that there is optical detectors to draw
        assert (
            self.geo.optical is not None
        ), "This geometry does not have optical detectors to draw."

        # Fetch the optical element positions and sizes
        if volume_id is None:
            positions = self.geo.optical.positions
        else:
            positions = self.geo.optical.volumes[volume_id].positions
        half_sizes = self.geo.optical.sizes / 2

        # If there is more than one detector shape, fetch shape IDs
        shape_ids = self.geo.optical.shape_ids

        # Convert the positions to pixel coordinates, if needed
        if not self.detector_coords:
            assert meta is not None, (
                "Must provide meta information to convert the optical "
                "element positions/sizes to pixel coordinates."
            )
            positions = meta.to_px(positions)
            half_sizes = half_sizes / meta.size

        # Check that the colors provided fix the appropriate range
        if color is not None and isinstance(color, (list, tuple, np.ndarray)):
            assert len(color) == len(
                positions
            ), "Must provide one value for each optical detector."

        # Build the hovertext vectors
        if hovertext is not None:
            if not isinstance(hovertext, (list, tuple, np.ndarray)):
                hovertext = [hovertext] * len(positions)
            elif len(hovertext) != len(positions):
                raise ValueError(
                    "The `hovertext` attribute should be provided as a scalar, "
                    "one value per point or one value per optical detector."
                )

        else:
            hovertext = [f"PD ID: {i}" for i in range(len(positions))]
            if isinstance(color, (list, tuple, np.ndarray)):
                for i, hc in enumerate(hovertext):
                    hovertext[i] = hc + f"<br>Value: {color[i]:.3f}"

        # If cmin/cmax are not provided, must build them so that all optical
        # detectors share the same colorscale range (not guaranteed otherwise)
        if isinstance(color, (list, tuple, np.ndarray)) and len(color) > 0:
            if cmin is None:
                cmin = np.min(color)
            if cmax is None:
                cmax = np.max(color)

        # If the legend is to be shared, make sure there is a common legend group
        if shared_legend and legendgroup is None:
            legendgroup = "group_" + str(time.time())

        # Draw each of the optical detectors
        traces = []
        for i, shape in enumerate(self.geo.optical.shape):
            # Restrict the positions to those of this shape, if needed
            if shape_ids is None:
                pos = positions
                col = color
                ht = hovertext
            else:
                index = np.where(np.asarray(shape_ids) == i)[0]
                pos = positions[index]
                if isinstance(color, (list, tuple, np.ndarray)):
                    col = color[index]
                else:
                    col = color
                ht = [hovertext[i] for i in index]

            # If zero-supression is requested, only draw the optical detectors
            # which record a non-zero signal
            if zero_supress and isinstance(col, (list, tuple, np.ndarray)):
                index = np.where(np.asarray(col) != 0)[0]
                pos = pos[index]
                col = col[index]
                ht = [ht[i] for i in index]

            # Dispatch the drawing based on the type of optical detector
            hd = half_sizes[i]
            if shape == "box":
                # Convert the positions/sizes to box lower/upper bounds
                lower, upper = pos - hd, pos + hd

                # Build boxes
                traces += box_traces(
                    lower,
                    upper,
                    shared_legend=shared_legend,
                    name=name,
                    color=col,
                    cmin=cmin,
                    cmax=cmax,
                    draw_faces=True,
                    hovertext=ht,
                    legendgroup=legendgroup,
                    **kwargs,
                )

            elif shape == "ellipsoid":
                # Convert the optical detector sizes to a covariance matrix
                covmat = np.diag(hd**2)

                # Build ellipsoids
                traces += ellipsoid_traces(
                    pos,
                    covmat,
                    shared_legend=shared_legend,
                    name=name,
                    color=col,
                    cmin=cmin,
                    cmax=cmax,
                    hovertext=ht,
                    legendgroup=legendgroup,
                    **kwargs,
                )

            elif shape == "disk":
                # Build disks as very flat cylinders
                axis = np.zeros(3, dtype=hd.dtype)
                axis[np.argmin(hd)] = 1.0
                length = 2.0 * hd[np.argmin(hd)]
                diameter = 2.0 * hd[np.argmax(hd)]

                # Build disks
                traces += cylinder_traces(
                    pos,
                    axis,
                    length,
                    diameter,
                    shared_legend=shared_legend,
                    name=name,
                    color=col,
                    cmin=cmin,
                    cmax=cmax,
                    hovertext=ht,
                    legendgroup=legendgroup,
                    **kwargs,
                )

            else:
                raise ValueError(
                    f"Optical detector shape '{shape}' not recognized. "
                    "Should be one of 'box', 'ellipsoid' or 'disk'."
                )

        # Set the legend display options
        if shape_ids is not None:
            # If the legend is shared, ensure that only the first trace shows the legend
            if shared_legend:
                for i, trace in enumerate(traces):
                    if i == 0:
                        trace.showlegend = True
                    else:
                        trace.showlegend = False
            else:
                # If the legend is not shared, ensure that the names are unique
                for i, trace in enumerate(traces):
                    trace.name = f"{name} {i}"

        return traces

    def crt_traces(
        self,
        meta: Optional[Meta] = None,
        draw_faces: bool = True,
        shared_legend: bool = True,
        name: str = "CRT",
        color: Union[int, str, np.ndarray] = "rgba(0,256,256,0.150)",
        draw_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> List[BaseTraceType]:
        """Function which produces a list of traces which represent the optical
        detectors in a 3D event display.

        Parameters
        ----------
        meta : Meta, optional
            Metadata information (only needed if pixel_coordinates is True)
        draw_faces : bool, default True
            Weather or not to draw the box faces, or only the edges
        shared_legend : bool, default True
            If True, the legend entry in plotly is shared between all the
            CRT volumes
        name : str, default 'CRT'
            Name of the CRT volumes
        color : Union[int, str, np.ndarray]
            Color of CRT detectors or list of color of CRT detectors
        draw_ids : List[int], optional
            If specified, only the requested CRT planes are drawn
        **kwargs : dict, optional
            List of additional arguments to pass to
            spine.vis.ellipsoid.ellipsoid_traces or spine.vis.box.box_traces

        Returns
        -------
        List[plotly.graph_objs.Mesh3D]
            List of CRT detector traces (one per CRT element)
        """
        # Check that there are CRT planes to draw
        assert (
            self.geo.crt is not None
        ), "This geometry does not have CRT planes to draw."

        # Load the list of CRT plane boundaries
        boundaries = np.stack([p.boundaries for p in self.geo.crt.planes])

        # If required, convert to pixel coordinates
        if not self.detector_coords:
            assert meta is not None, (
                "Must provide meta information to convert the CRT plane "
                "boundaries to pixel coordinates."
            )
            boundaries = meta.to_px(boundaries.transpose(0, 2, 1)).transpose(0, 2, 1)

        # Restrict the list of boundaries, if requested
        if draw_ids is not None:
            tmp = np.empty(
                (len(draw_ids), *boundaries.shape[1:]), dtype=boundaries.dtype
            )
            for i, idx in enumerate(draw_ids):
                tmp[i] = boundaries[idx]

            boundaries = tmp

        # Get a trace per CRT plane
        detectors = box_traces(
            boundaries[..., 0],
            boundaries[..., 1],
            draw_faces=draw_faces,
            color=color,
            shared_legend=shared_legend,
            name=name,
            **kwargs,
        )

        return detectors
