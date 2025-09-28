"""Draw detectors based on their geometry definition."""

import time
from functools import partial

import numpy as np

from spine.utils.geo import Geometry

from .box import box_traces
from .ellipsoid import ellipsoid_traces

__all__ = ["GeoDrawer"]


class GeoDrawer:
    """Handles drawing all things related to the detector geometry.

    This class is loads a :class:`Geometry` object once from a geometry file
    and uses it to represent all things related to the detector geometry:
    - TPC boundaries
    - Optical detectors
    - CRT detectors
    """

    def __init__(self, detector=None, file_path=None, detector_coords=True):
        """Initializes the underlying detector :class:`Geometry` object.

        Parameters
        ----------
        detector : str, optional
            Name of a recognized detector to the geometry from
        file_path : str, optional
            Path to a `.yaml` geometry configuration
        detector_coords : bool, default False
            If False, the coordinates are converted to pixel indices
        """
        # Initialize the detector geometry
        self.geo = Geometry(detector, file_path)

        # Store whether to use detector cooordinates or not
        self.detector_coords = detector_coords

    def tpc_traces(
        self,
        meta=None,
        draw_faces=False,
        shared_legend=True,
        name="TPC",
        color="rgba(0,0,0,0.150)",
        linewidth=5,
        **kwargs,
    ):
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
            detector volumes
        name : Union[str, List[str]], default 'Detector'
            Name(s) of the detector volumes
        color : Union[int, str, np.ndarray]
            Color of boxes or list of color of boxes
        linewidth : int, default 2
            Width of the box edge lines
        **kwargs : dict, optional
            List of additional arguments to pass to
            spine.viusalization.boxes.box_traces

        Returns
        -------
        List[Union[plotly.graph_objs.Scatter3D, plotly.graph_objs.Mesh3D]]
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

        # Get a trace per detector volume
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
        meta=None,
        shared_legend=True,
        legendgroup=None,
        name="Optical",
        color="rgba(0,0,255,0.25)",
        hovertext=None,
        cmin=None,
        cmax=None,
        zero_supress=False,
        volume_id=None,
        **kwargs,
    ):
        """Function which produces a list of traces which represent the optical
        detectors in a 3D event display.

        Parameters
        ----------
        meta : Meta, optional
            Metadata information (only needed if pixel_coordinates is True)
        shared_legend : bool, default True
            If True, the legend entry in plotly is shared between all the
            detector volumes
        legendgroup : str, optional
            Legend group to be shared between all boxes
        name : Union[str, List[str]], default 'Detector'
            Name(s) of the detector volumes
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

        # Fetch the optical element positions and dimensions
        if volume_id is None:
            positions = self.geo.optical.positions.reshape(-1, 3)
        else:
            positions = self.geo.optical.positions[volume_id]
        half_dimensions = self.geo.optical.dimensions / 2

        # If there is more than one detector shape, fetch shape IDs
        shape_ids = None
        if self.geo.optical.shape_ids is not None:
            shape_ids = self.geo.optical.shape_ids
            if volume_id is None:
                shape_ids = np.tile(shape_ids, self.geo.optical.num_volumes)

        # Convert the positions to pixel coordinates, if needed
        if not self.detector_coords:
            assert meta is not None, (
                "Must provide meta information to convert the optical "
                "element positions/dimensions to pixel coordinates."
            )
            positions = meta.to_px(positions)
            half_dimensions = half_dimensions / meta.size

        # Check that the colors provided fix the appropriate range
        if color is not None and not np.isscalar(color):
            assert len(color) == len(
                positions
            ), "Must provide one value for each optical detector."

        # Build the hovertext vectors
        if hovertext is not None:
            if np.isscalar(hovertext):
                hovertext = [hovertext] * len(positions)
            elif len(hovertext) != len(positions):
                raise ValueError(
                    "The `hovertext` attribute should be provided as a scalar, "
                    "one value per point or one value per optical detector."
                )

        else:
            hovertext = [f"PD ID: {i}" for i in range(len(positions))]
            if color is not None and not np.isscalar(color):
                for i, hc in enumerate(hovertext):
                    hovertext[i] = hc + f"<br>Value: {color[i]:.3f}"

        # If cmin/cmax are not provided, must build them so that all optical
        # detectors share the same colorscale range (not guaranteed otherwise)
        if color is not None and not np.isscalar(color) and len(color) > 0:
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
                if color is not None and not np.isscalar(color):
                    col = color[index]
                else:
                    col = color
                ht = [hovertext[i] for i in index]

            # If zero-supression is requested, only draw the optical detectors
            # which record a non-zero signal
            if zero_supress and color is not None and not np.isscalar(color):
                index = np.where(np.asarray(col) != 0)[0]
                pos = pos[index]
                col = col[index]
                ht = [ht[i] for i in index]

            # Determine wheter to show legends or not
            showlegend = not shared_legend or i == 0

            # Dispatch the drawing based on the type of optical detector
            hd = half_dimensions[i]
            if shape == "box":
                # Convert the positions/dimensions to box lower/upper bounds
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
                    showlegend=showlegend,
                    **kwargs,
                )

            else:
                # Convert the optical detector dimensions to a covariance matrix
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
                    showlegend=showlegend,
                    **kwargs,
                )

        return traces

    def crt_traces(
        self,
        meta=None,
        draw_faces=True,
        shared_legend=True,
        name="CRT",
        color="rgba(0,256,256,0.150)",
        draw_ids=None,
        **kwargs,
    ):
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
            detector volumes
        name : Union[str, List[str]], default 'Detector'
            Name(s) of the detector volumes
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

        # Get a trace per detector volume
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
