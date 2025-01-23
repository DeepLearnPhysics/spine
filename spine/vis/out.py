"""Draw reconstruction output-level objects"""

from collections import defaultdict
from warnings import warn

import numpy as np
from plotly import graph_objs as go

from spine.utils.globals import COORD_COLS, PID_LABELS, SHAPE_LABELS, TRACK_SHP

from .geo import GeoDrawer
from .point import scatter_points
from .cluster import scatter_clusters
from .layout import (
        layout3d, dual_figure3d, PLOTLY_COLORS_WGRAY, HIGH_CONTRAST_COLORS)


class Drawer:
    """Handles drawing the true/reconstructed output.

    This class is given the entire input/output dictionary from one entry and
    provides functions to represent the output.
    """
    # List of recognized object types
    _obj_types = ('fragments', 'particles', 'interactions')

    # List of recognized drawing modes
    _draw_modes = ('reco', 'truth', 'both', 'all')

    # List of known point modes for true objects and their corresponding keys
    _point_modes = (
            ('points', 'points_label'),
            ('points_adapt', 'points'),
            ('points_g4', 'points_g4')
    )

    # List of known deposition modes for true particles and their corresponding keys
    _dep_modes = (
            ('depositions', 'depositions_label'),
            ('depositions_q', 'depositions_q_label'),
            ('depositions_adapt', 'depositions_label_adapt'),
            ('depositions_adapt_q', 'depositions'),
            ('depositions_g4', 'depositions_g4')
    )

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
                f"{self._draw_modes}.")

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
        assert truth_point_mode in self.point_modes, (
                 "The `truth_point_mode` argument must be one of "
                f"{self.point_modes.keys()}. Got `{truth_point_mode}` instead.")
        self.truth_point_mode = truth_point_mode
        self.truth_point_key = self.point_modes[self.truth_point_mode]
        self.truth_index_mode = truth_point_mode.replace('points', 'index')

        # If detector information is provided, initialie the geometry drawer
        self.geo_drawer = None
        self.meta = data.get('meta', None)
        if detector is not None:
            self.geo_drawer = GeoDrawer(
                    detector=detector, detector_coords=detector_coords)

        # Initialize the layout
        self.split_scene = split_scene
        meta = self.meta if detector is None else None
        self.layout = layout3d(
                detector=detector, meta=meta, detector_coords=detector_coords,
                **kwargs)

    @property
    def point_modes(self):
        """Dictionary which makes the correspondance between the name of a true
        object point attribute with the underlying point tensor it points to.

        Returns
        -------
        Dict[str, str]
            Dictionary of (attribute, key) mapping for point coordinates
        """
        return dict(self._point_modes)

    @property
    def dep_modes(self):
        """Dictionary which makes the correspondance between the name of a true
        object deposition attribute with the underlying deposition array it points to.

        Returns
        -------
        Dict[str, str]
            Dictionary of (attribute, key) mapping for point depositions
        """
        return dict(self._dep_modes)

    def get_index(self, obj):
        """Get a certain pre-defined index attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects index are obtained using the
        `truth_index_mode` attribute of the class.

        Parameters
        ----------
        obj : Union[FragmentBase, ParticleBase, InteractionBase]
            Fragment, Particle or Interaction object

        Results
        -------
        np.ndarray
           (N) Object index
        """
        if not obj.is_truth:
            return obj.index
        else:
            return getattr(obj, self.truth_index_mode)

    def get(self, obj_type, attr=None, color_attr=None, draw_raw=False,
            draw_end_points=False, draw_vertices=False, draw_flashes=False,
            synchronize=False, titles=None, split_traces=False,
            matched_flash_only=True):
        """Draw the requested object type with the requested mode.

        Parameters
        ----------
        obj_type : str
            Name of the object type to draw (one of 'fragment', 'particle' or
            'interaction'
        attr : Union[str, List[str]], optional
            Name of list of names of attributes to draw
        color_attr : str, optional
            Name of the attribute to use to determine the color
        draw_raw : bool, default False
            If `True`, add a trace which corresponds to the raw depositions
        draw_end_points : bool, default False
            If `True`, draw the fragment or particle end points
        draw_vertices : bool, default False
            If `True`, draw the interaction vertices
        draw_flashes : bool, default False
            If `True`, draw flashes that have been matched to interactions
        synchronize : bool, default False
            If `True`, matches the camera position/angle of one plot to the other
        titles : List[str], optional
            Titles of the two scenes (only relevant for split_scene True
        split_traces : bool, default False
            If `True`, one trace is produced for each object
        matched_flash_only : bool, default True
            If `True`, only flashes matched to interactions are drawn

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
            traces[prefix] = self._object_traces(
                    obj_name, attr, color_attr, split_traces)

        # Fetch the raw depositions, if requested
        if draw_raw:
            for prefix in self.prefixes:
                traces[prefix] = self._raw_trace() + traces[prefix]

        # Fetch the end points, if requested
        if draw_end_points:
            assert obj_name != 'interactions', (
                    "Interactions do not have end point attributes.")
            for prefix in self.prefixes:
                obj_name = f'{prefix}_{obj_type}'
                traces[prefix] += self._start_point_trace(obj_name)
                traces[prefix] += self._end_point_trace(obj_name)

        # Fetch the vertices, if requested
        if draw_vertices:
            for prefix in self.prefixes:
                obj_name = f'{prefix}_interactions'
                assert obj_name in self.data, (
                        "Must provide interactions to draw their vertices.")
                traces[prefix] += self._vertex_trace(obj_name)

        # Fetch the flashes, if requested
        if draw_flashes:
            assert 'flashes' in self.data, (
                    "Must provide the `flashes` objects to draw them.")
            for prefix in self.prefixes:
                obj_name = f'{prefix}_interactions'
                assert obj_name in self.data, (
                        "Must provide interactions to draw matched flashes.")
                traces[prefix] += self._flash_trace(obj_name, matched_flash_only)

        # Add the TPC traces, if available
        if self.geo_drawer is not None:
            if len(self.prefixes) and self.split_scene:
                for prefix in self.prefixes:
                    traces[prefix] += self.geo_drawer.tpc_traces(meta=self.meta)
            else:
                traces[self.prefixes[-1]] += self.geo_drawer.tpc_traces(meta=self.meta)

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

    def _object_traces(self, obj_name, attr, color_attr, split_traces):
        """Draw a specific object.

        Parameters
        ----------
        obj_name : str
            Name of the objects to be represented
        attr : Union[str, List[str]]
            Attribute name(s) used to set the color/hovertext
        color_attr : str
            Name of the attribute to use to determine the color
        split_traces : bool
            If `True`, one trace is produced for each object

        Returns
        -------
        List[plotly.graph_objs.Scatter3d]
            List of traces, one per object being drawn up
        """
        # Fetch the clusters and the points
        point_key = self.truth_point_key if 'truth' in obj_name else 'points'
        points = self.data[point_key]
        clusts = [self.get_index(obj) for obj in self.data[obj_name]]

        # Get the colors
        color_dict = self._object_colors(
                obj_name, attr, color_attr, split_traces)

        # Return
        return scatter_clusters(
                points, clusts, single_trace=not split_traces,
                shared_legend=not split_traces, **color_dict)

    def _object_colors(self, obj_name, attr, color_attr, split_traces):
        """Provides an appropriate colorscale and range for a given attribute.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        attr : Union[str, List[str]]
            Attribute name(s) used to set the color/hovertext
        color_attr : str
            Name of the attribute to use to determine the color
        split_traces : bool
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

        # Make sure that the color attribute is requested
        if color_attr is not None:
            if isinstance(attr, str):
                assert attr == color_attr, (
                    "The attribute used to define the color scale cannot differ "
                    "from the attribute used for the hovertext information.")
            else:
                assert color_attr in attr, (
                    "The attribute used to define the color scale must be "
                    "included in the list of hovertext attributes.")

        # Initialize hovertext per object
        obj_type = obj_name.split('_')[-1][:-1].capitalize()
        count = len(self.data[obj_name])
        hovertext = [f'{obj_type} {i}' for i in range(count)]

        # Fetch the list of color values and attributes
        if attr is None:
            attr = 'id'
            color_attr = 'id'
            color = np.arange(len(self.data[obj_name]))

        else:
            # Fetch hover information for each of the requested attributes
            single_attr = isinstance(attr, str)
            attrs = [attr] if single_attr else attr
            for attr in attrs:
                # If it is a true deposition attribute, check that it matches
                # the point mode that is being used to draw the true objects
                if 'truth' in obj_name and attr.startswith('depositions'):
                    prefix = self.truth_point_mode.replace('points', 'depositions')
                    assert attr.startswith(prefix), (
                            f"Points mode {self.truth_point_mode} and deposition "
                            f"mode {attr} are incompatible.")

                # Get the value, color and hovertext
                attr_name = ' '.join(attr.split('_')).capitalize()
                values = [getattr(obj, attr) for obj in self.data[obj_name]]
                if single_attr or attr == color_attr:
                    color = values
                if not attr.startswith('depositions'):
                    for i, hc in enumerate(hovertext):
                        if isinstance(hc, str):
                            hovertext[i] = hc + f'<br>{attr_name}: {values[i]}'
                        else:
                            hovertext[i] = [
                                    hcj + f'<br>{attr_name}: {values[i]}' for hcj in hc]

                else:
                    for i, hc in enumerate(hovertext):
                        if isinstance(hc, str):
                            hovertext[i] = [
                                    hc + f'<br>Value: {v:0.3f}' for v in values[i]]
                        else:
                            hovertext[i] = [
                                    hc[i] + f'<br>Value: {v:0.3f}' for i, v in enumerate(values[i])]

            # Determine which attribute to define the colorscale
            if color_attr is None:
                if single_attr:
                    color_attr = attrs[0]
                else:
                    color_attr = 'id'
                    color = np.arange(len(self.data[obj_name]))
                
        # Set up the appropriate color scheme
        if color_attr.startswith('depositions'):
            # Continuous values shared between objects
            dep_mode = self.dep_modes[color_attr] if 'truth' in obj_name else 'depositions'
            colorscale = 'Inferno'
            cmin = 0.
            cmax = 2*np.median(self.data[dep_mode])

        elif color_attr.startswith('is_'):
            # Boolean
            color = np.array(color, dtype=np.int32)
            colorscale = PLOTLY_COLORS_WGRAY[1:3]
            cmin = 0
            cmax = 1

        elif color_attr == 'shape' or color_attr == 'pid':
            # Fixed length values, with potentially invalid values
            ref = SHAPE_LABELS if color_attr == 'shape' else PID_LABELS
            num_classes = len(ref)
            colorscale = PLOTLY_COLORS_WGRAY[:num_classes + 1]
            cmin = -1
            cmax = num_classes - 1

        elif color_attr.endswith('id'):
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
            raise ValueError(
                    f"Color attribute not supported: {color_attr}.")

        return {'color': color, 'hovertext': hovertext, 'name': name,
                'colorscale': colorscale, 'cmin': cmin, 'cmax': cmax}

    def _raw_trace(self):
        """Draws the raw input image (pre-reconstruction).

        Returns
        -------
        List[plotly.graph_objs.Scatter3d]
            List of one trace containing the input to the reconstruction
        """
        # Fetch the input attributes
        points = self.data['points']
        deps = self.data['depositions']

        # Fetch the colorscale limits
        cmin = 0.
        cmax = 2*np.median(deps) if len(deps) else 1.

        return scatter_points(
                points, color=deps, cmin=cmin, cmax=cmax, colorscale='Inferno',
                name='Raw input')

    def _start_point_trace(self, obj_name, color='black', markersize=7,
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
        list
            List of start point traces
        """
        return self._point_trace(
                obj_name, 'start_point', color=color, markersize=markersize,
                marker_symbol=marker_symbol, **kwargs)

    def _end_point_trace(self, obj_name, color='black', markersize=7,
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
        list
            List of end point traces
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
        list
            List of vertex point traces
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
        list
            List of point traces
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

    def _flash_trace(self, obj_name, matched_only, **kwargs):
        """Draw the cumlative PEs of flashes that have been matched to
        interactions specified by `obj_name`.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        matched_only : bool
            If `True`, only flashes matched to interactions are drawn
        **kwargs : dict, optional
            List of additional arguments to pass to :func:`optical_traces`

        Returns
        -------
        list
            List of optical detector traces
        """
        # If there was no geometry provided by the user, nothing to do here
        assert self.geo_drawer is not None, (
                "Cannot draw optical detectors without geometry information.")

        # Define the name of the trace
        name = ' '.join(obj_name.split('_')).capitalize()[:-1] + ' flashes'

        # Find the list of flash IDs to draw
        if matched_only:
            flash_ids = []
            for inter in self.data[obj_name]:
                if inter.is_flash_matched:
                    flash_ids.extend(inter.flash_ids)
        else:
            flash_ids = np.arange(len(self.data['flashes']))

        # Sum values from each flash to build a a global color scale
        color = np.zeros(self.geo_drawer.geo.optical.num_detectors)
        opt_det_ids = self.geo_drawer.geo.optical.det_ids
        for flash_id in flash_ids:
            flash = self.data['flashes'][flash_id]
            index = self.geo_drawer.geo.optical.volume_index(flash.volume_id)
            pe_per_ch = flash.pe_per_ch
            if opt_det_ids is not None:
                pe_per_ch = np.bincount(opt_det_ids, weights=pe_per_ch)
            color[index] += pe_per_ch

        # Return the set of optical detectors with a color scale
        return self.geo_drawer.optical_traces(
                meta=self.meta, color=color, zero_supress=True,
                colorscale='Inferno', name=name)
