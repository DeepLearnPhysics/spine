"""Draw detectors based on their geometry definition."""

from spine.utils.geo import Geometry

from .box import box_traces


def detector_traces(detector=None, boundaries=None, meta=None,
                    detector_coords=True, draw_faces=False, shared_legend=True,
                    name='Detector', color='rgba(0,0,0,0.150)',
                    linewidth=2, **kwargs):
    """Function which takes loads a file with detector boundaries and
    produces a list of traces which represent them in a 3D event display.

    The detector boundary file is a `.npy` or `.npz` file which contains
    a single tensor of shape (N, 3, 2), with N the number of detector
    volumes. The first column for each volume represents the lower boundary
    and the second the upper boundary. The boundaries must be ordered.

    Parameters
    ----------
    detector : str, optional
        Name of a recognized detector to the geometry from
    boundaries : str, optional
        Name of a recognized detector to get the geometry from or path
        to a `.npy` boundary file to load the boundaries from.
    meta : Meta, optional
        Metadata information (only needed if pixel_coordinates is True)
    detector_coords : bool, default False
        If False, the coordinates are converted to pixel indices
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
    # Load the list of boundaries
    boundaries = Geometry(detector, boundaries).tpcs

    # If required, convert to pixel coordinates
    if not detector_coords:
        assert meta is not None, (
                "Must provide meta information to convert the detector "
                "boundaries to pixel coordinates.")
        boundaries = meta.to_px(
                boundaries.transpose(0,2,1)).transpose(0,2,1)

    # Get a trace per detector volume
    detectors = box_traces(
            boundaries[..., 0], boundaries[..., 1], draw_faces=draw_faces,
            color=color, linewidth=linewidth, shared_legend=shared_legend,
            name=name, **kwargs)

    return detectors
