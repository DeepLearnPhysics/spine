"""Draw reconstruction output-level objects"""

from collections import defaultdict

import numpy as np
from plotly import graph_objs as go

from spine.utils.globals import COORD_COLS, PID_LABELS, SHAPE_LABELS, TRACK_SHP

from .point import scatter_points
from .cluster import scatter_clusters
from .detector import detector_traces
from .layout import (
        layout3d, dual_figure3d, PLOTLY_COLORS_WGRAY, HIGH_CONTRAST_COLORS)


class Drawer:
    """Class dedicated to drawing the true/reconstructed output.

    This class is given the entire input/output dictionary from one entry and
    provides functions to represent the output.
    """
    # List of recognized object types
    _obj_types = ('fragments', 'particles', 'interactions')

    # List of recognized drawing modes
    _draw_modes = ('reco', 'truth', 'both', 'all')

    # List of known point modes
    _point_modes = ('points', 'points_adapt', 'points_g4')

    # Map between attribute and underlying point objects
    _point_map = {'points': 'points_label', 'points_adapt': 'points', 
                  'points_g4': 'points_g4'}

    def __init__(self, data, draw_mode='both', truth_point_mode='points',
                 split_scene=True, detector=None, detector_coords=True,
                 **kwargs):
        """Initialize the drawer attributes

        Parameters
        ----------
        data : dict
            Dictionary of data products
        draw_mode : str, default 'both'
            Drawing mode, one of 'reco', 'truth' or 'both'
        truth_point_mode : str, optional
            If specified, tells which attribute of the :class:`TruthFragment`,
            :class:`TruthParticle` or :class:`TruthInteraction` object to use
            to fetch its point coordinates
        split_scene : bool, default True
            If True and when drawing both reconstructed and truth information,
            split the traces between two separate scenes
        detector : str, optional
            Name of the detector to be drawn
        detector_coords : bool, default True
            Whether the object coordinates are expressed in detector coordinates
        **kwargs : dict, optional
            Additional arguments to pass to the :func:`layout3d` function
        """
        # Store the data used to draw the reconstruction output
        self.data = data

        # Set up the list of prefixes to access
        assert draw_mode in self._draw_modes, (
                f"`mode` not recognized: {draw_mode}. Must be one of "
                 "{self._draw_modes}.")

        self.prefixes = []
        if draw_mode != 'truth':
            self.prefixes.append('reco')
        if draw_mode != 'reco':
            self.prefixes.append('truth')

        # Build a list of supported objects
        self.supported_objs = []
        for mode in ['reco', 'truth']:
            for obj_type in self._obj_types:
                self.supported_objs.append(f'{mode}_{obj_type}')

        # Set up the truth point mode
        assert truth_point_mode in self._point_modes, (
                 "The `truth_point_mode` argument must be one of "
                f"{self._point_modes}. Got `{truth_point_mode}` instead.")
        self.truth_point_mode = truth_point_mode
        self.truth_index_mode = truth_point_mode.replace('points', 'index')

        # Save the detector properties
        self.meta = data.get('meta', None)
        self.detector = detector
        self.detector_coords = detector_coords

        # Initialize the layout
        self.split_scene = split_scene
        meta = self.meta if detector is None else None
        self.layout = layout3d(
                detector=self.detector, meta=meta,
                detector_coords=self.detector_coords, **kwargs)

    def get(self, obj_type, attr=None, draw_end_points=False,
            draw_vertices=False, synchronize=False, titles=None,
            split_traces=False):
        """Draw the requested object type with the requested mode.

        Parameters
        ----------
        obj_type : str
            Name of the object type to draw (one of 'fragment', 'particle' or
            'interaction'
        attr : Union[str, List[str]]
            Name of list of names of attributes to draw
        draw_end_points : bool, default False
            If `True`, draw the fragment or particle end points
        draw_vertices : bool, default False
            If `True`, draw the interaction vertices
        synchronize : bool, default False
            If `True`, matches the camera position/angle of one plot to the other
        titles : List[str], optional
            Titles of the two scenes (only relevant for split_scene True
        split_traces : bool, default False
            If `True`, one trace is produced for each object

        Returns
        -------
        go.Figure
            Figure containing all the necessary information to draw
        """
        # Check that what is to be drawn is a known object type that is provided
        assert obj_type in self._obj_types, (
                f"Object type not recognized: {obj_type}. Must be one of "
                f"{self._obj_types}.")

        # Fetch the objects
        traces = {}
        for prefix in self.prefixes:
            obj_name = f'{prefix}_{obj_type}'
            assert obj_name in self.data, (
                    f"Must provide `{obj_name}` in the data products to draw "
                     "them.")
            traces[prefix] = self._object_traces(obj_name, attr, split_traces)

        # Fetch the end points, if requested
        if draw_end_points:
            assert obj_name != 'interactions', (
                    "Interactions do not have end point attributes.")
            for prefix in self.prefixes:
                obj_name = f'{prefix}_{obj_type}'
                traces[prefix] += self._start_point_trace(obj_name)
                traces[prefix] += self._end_point_trace(obj_name)

        # Fetch the vertex, if requested
        if draw_vertices:
            for prefix in self.prefixes:
                obj_name = f'{prefix}_interactions'
                assert obj_name in self.data, (
                        "Must provide interactions to draw their vertices.")
                traces[prefix] += self._vertex_trace(obj_name)

        # Add the detector traces, if available
        if self.detector is not None:
            if len(self.prefixes) and self.split_scene:
                for prefix in self.prefixes:
                    traces[prefix] += detector_traces(
                        detector=self.detector, meta=self.meta,
                        detector_coords=self.detector_coords)
            else:
                traces[self.prefixes[-1]] += detector_traces(
                        detector=self.detector, meta=self.meta,
                        detector_coords=self.detector_coords)

        # Initialize the figure, return
        if len(self.prefixes) > 1 and self.split_scene:
            if titles is None:
                titles = [f'Reconstructed {obj_type}', f'Truth {obj_type}']
            figure = dual_figure3d(
                    traces['reco'], traces['truth'], layout=self.layout,
                    synchronize=synchronize, titles=titles)

        else:
            assert titles is None, (
                    "Providing titles does not do anything when split_scene "
                    "is False.")
            all_traces = []
            for trace_group in traces.values():
                all_traces += trace_group
            figure = go.Figure(all_traces, layout=self.layout)

        return figure

    def _object_traces(self, obj_name, attr=None, split_traces=False):
        """Draw a specific object.

        Parameters
        ----------
        obj_name : str
            Name of the objects to be represented
        attr : str
            Attribute name used to set the color
        split_traces : bool, default False
            If `True`, one trace is produced for each object

        Returns
        -------
        List[plotly.graph_objs.Scatter3d]
            List of traces, one per object being drawn up
        """
        # Fetch the clusters and the points
        if 'reco' in obj_name:
            points = self.data['points']
            index_mode = 'index'

        else:
            points = self.data[self._point_map[self.truth_point_mode]]
            index_mode = self.truth_index_mode

        clusts = [getattr(obj, index_mode) for obj in self.data[obj_name]]

        # Get the colors
        color_dict = self._object_colors(obj_name, attr, split_traces)

        # Return
        return scatter_clusters(
                points, clusts, single_trace=not split_traces,
                shared_legend=not split_traces, **color_dict)

    def _object_colors(self, obj_name, attr, split_traces=False):
        """Provides an appropriate colorscale and range for a given attribute.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        attr : str
            Object attribute to draw
        split_traces : bool, default False
            If `True`, one trace is produced for each object

        Returns
        -------
        dict
            Dictionary of color parameters (colorscale, cmin, cmax)
        """
        # Define the name of the trace group
        name = ' '.join(obj_name.split('_')).capitalize()
        if split_traces:
            name = name[:-1]

        # Initialize hovertext per object
        obj_type = obj_name.split('_')[-1][:-1].capitalize()
        count = len(self.data[obj_name])
        hovertext = [f'{obj_type} {i}' for i in range(count)]

        # Fetch the list of color values and attributes
        if attr is None:
            attr = 'id'
            color = np.arange(len(self.data[obj_name]))

        else:
            attr_name = ' '.join(attr.split('_')).capitalize()
            color = [getattr(obj, attr) for obj in self.data[obj_name]]
            if not attr.startswith('depositions'):
                for i, hc in enumerate(hovertext):
                    hovertext[i] = hc + f'<br>{attr_name}: {color[i]}'
            else:
                for i, hc in enumerate(hovertext):
                    hovertext[i] = [
                            hc + f'<br>Value: {v:0.3f}' for v in color[i]]

        # Set up the appropriate color scheme
        if attr.startswith('depositions'):
            # Continuous values shared between particles
            colorscale = 'Inferno'
            cmin = 0.
            cmax = 2*np.median(self.data[attr])

        elif attr.startswith('is_'):
            # Boolean
            color = np.array(color, dtype=np.int32)
            colorscale = PLOTLY_COLORS_WGRAY[1:3]
            cmin = 0
            cmax = 1

        elif attr == 'shape' or attr == 'pid':
            # Fixed length values, with potentially invalid values
            ref = SHAPE_LABELS if attr == 'shape' else PID_LABELS
            num_classes = len(ref)
            colorscale = PLOTLY_COLORS_WGRAY[:num_classes + 1]
            cmin = -1
            cmax = num_classes - 1

        elif attr.endswith('id'):
            # Variable-lengh discrete values
            color = np.unique(color, return_inverse=True)[-1]
            colorscale = HIGH_CONTRAST_COLORS
            count = len(color)
            if count == 0:
                colorscale = None
            elif count == 1:
                colorscale = [colorscale[0]] * 2 # Avoid length 0 colorscale
            elif count <= len(colorscale):
                colorscale = colorscale[:count]
            else:
                repeat = (count - 1)//len(colorscale) + 1
                self._colorscale = np.repeat(colorscale, repeat)[:count]

            cmin = 0
            cmax = count# - 1

        else:
            raise KeyError(
                    f"Drawing the `{attr}` attribute of each object in the "
                    f"{name.lower()} is not supported.")

        return {'color': color, 'hovertext': hovertext, 'name': name,
                'colorscale': colorscale, 'cmin': cmin, 'cmax': cmax}

    def _start_point_trace(self, obj_name, color='black', markersize=5,
                            marker_symbol='circle', **kwargs):
        """Scatters the start points of the requested object type.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        color : Union[str, np.ndarray], optional
            Color of markers/lines or (N) list of color of markers/lines
        markersize : float, default 5
            Marker size
        marker_symbol : float, default 'circle'
            Marker style
        **kwargs : dict, optional
            Additional parameters to pass

        Returns
        -------
        dict
            Dictionary of color parameters (colorscale, cmin, cmax)
        """
        return self._point_trace(
                obj_name, 'start_point', color=color, markersize=markersize,
                marker_symbol=marker_symbol, **kwargs)

    def _end_point_trace(self, obj_name, color='black', markersize=5,
                          marker_symbol='circle-open', **kwargs):
        """Scatters the end points of the requested object type.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        color : Union[str, np.ndarray], optional
            Color of markers/lines or (N) list of color of markers/lines
        markersize : float, default 5
            Marker size
        marker_symbol : float, default 'circle-open'
            Marker style
        **kwargs : dict, optional
            Additional parameters to pass

        Returns
        -------
        dict
            Dictionary of color parameters (colorscale, cmin, cmax)
        """
        return self._point_trace(
                obj_name, 'end_point', color=color, markersize=markersize,
                marker_symbol=marker_symbol, **kwargs)

    def _vertex_trace(self, obj_name, vertex_attr='vertex', color='green',
                      markersize=10, marker_symbol='diamond', **kwargs):
        """Scatters the vertex of the requested object type.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        color : Union[str, np.ndarray], optional
            Color of markers/lines or (N) list of color of markers/lines
        markersize : float, default 10
            Marker size
        marker_symbol : float, default 'circle-open'
            Marker style
        **kwargs : dict, optional
            Additional parameters to pass

        Returns
        -------
        dict
            Dictionary of color parameters (colorscale, cmin, cmax)
        """
        return self._point_trace(
                obj_name, vertex_attr, color=color, markersize=markersize,
                marker_symbol=marker_symbol, **kwargs)

    def _point_trace(self, obj_name, point_attr, **kwargs):
        """Scatters a set of discrete points per object instance.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        point_attr : str
            Name of the attribute specifying end point to draw
        **kwargs : dict, optional
            List of additional arguments to pass to :func:`scatter_points`

        Returns
        -------
        dict
            Dictionary of color parameters (colorscale, cmin, cmax)
        """
        # Define the name of the trace
        name = (' '.join(obj_name.split('_')).capitalize()[:-1] + ' ' +
                ' '.join(point_attr.split('_')))

        # Fetch the particular end point of each object
        obj_type = obj_name.split('_')[-1][:-1].capitalize()
        point_list, hovertext = [], []
        for i, obj in enumerate(self.data[obj_name]):
            # If it is an end point, skip if the object is not a track
            if point_attr == 'end_point' and obj.shape != TRACK_SHP:
                continue

            # Skip empty true objects
            if obj.is_truth and not len(getattr(obj, self.truth_index_mode)):
                continue

            # Append the particular end point of this object and the label
            point_list.append(getattr(obj, point_attr))
            hovertext.append(f'{obj_type} {i} ' + ' '.join(point_attr.split('_')))

        points = np.empty((0, 3), dtype=self.data['points'].dtype)
        if len(point_list):
            points = np.vstack(point_list)

        return scatter_points(
                points, hovertext=np.array(hovertext), name=name, **kwargs)
