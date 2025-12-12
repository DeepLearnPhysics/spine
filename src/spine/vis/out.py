"""Draw reconstruction output-level objects"""

import numpy as np
from plotly import graph_objs as go

from spine.geo import GeoManager
from spine.utils.globals import PID_LABELS, SHAPE_LABELS, TRACK_SHP

from .arrow import scatter_arrows
from .cluster import scatter_clusters
from .geo import GeoDrawer
from .layout import HIGH_CONTRAST_COLORS, PLOTLY_COLORS_WGRAY, dual_figure3d, layout3d
from .lite import scatter_lite
from .point import scatter_points

__all__ = ["Drawer"]


class Drawer:
    """Handles drawing the true/reconstructed output.

    This class is given the entire input/output dictionary from one entry and
    provides functions to represent the output.
    """

    # List of recognized object types
    _obj_types = ("fragments", "particles", "interactions")

    # List of recognized drawing modes
    _draw_modes = ("reco", "truth", "both", "all")

    # List of known point modes for true objects and their corresponding keys
    _point_modes = (
        ("points", "points_label"),
        ("points_adapt", "points"),
        ("points_g4", "points_g4"),
    )

    # List of known deposition modes for true particles and their corresponding keys
    _dep_modes = (
        ("depositions", "depositions_label"),
        ("depositions_q", "depositions_q_label"),
        ("depositions_adapt", "depositions_label_adapt"),
        ("depositions_adapt_q", "depositions"),
        ("depositions_g4", "depositions_g4"),
    )

    # List of known source modes for true particles and their corresponding keys
    _source_modes = (("sources", "sources_label"), ("sources_adapt", "sources"))

    def __init__(
        self,
        data,
        draw_mode="both",
        truth_point_mode="points",
        split_scene=True,
        geo=None,
        detector_coords=True,
        lite=False,
        **kwargs,
    ):
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
        geo : Geometry, optional
            Geometry object used to draw detector elements
        detector_coords : bool, default True
            Whether the object coordinates are expressed in detector coordinates
        lite : bool, default False
            If `True`, load the objects in their lite version (no space points)
        **kwargs : dict, optional
            Additional arguments to pass to the :func:`layout3d` function
        """
        # Store the data used to draw the reconstruction output
        self.data = data

        # Set up the list of prefixes to access
        assert draw_mode in self._draw_modes, (
            f"`mode` not recognized: {draw_mode}. Must be one of "
            f"{self._draw_modes}."
        )

        self.prefixes = []
        if draw_mode != "truth":
            self.prefixes.append("reco")
        if draw_mode != "reco":
            self.prefixes.append("truth")

        # Build a list of supported objects
        self.supported_objs = []
        for mode in ["reco", "truth"]:
            for obj_type in self._obj_types:
                self.supported_objs.append(f"{mode}_{obj_type}")

        # Set up the truth point mode
        assert truth_point_mode in self.point_modes, (
            "The `truth_point_mode` argument must be one of "
            f"{self.point_modes.keys()}. Got `{truth_point_mode}` instead."
        )
        self.truth_point_mode = truth_point_mode
        self.truth_point_key = self.point_modes[self.truth_point_mode]
        self.truth_index_mode = truth_point_mode.replace("points", "index")

        # If detector information is provided, initialize the geometry drawer
        self.geo, self.geo_drawer = None, None
        self.meta = data.get("meta", None)
        if geo is not None or GeoManager.is_initialized():
            self.geo = geo if geo is not None else GeoManager.get_instance()
            self.geo_drawer = GeoDrawer(geo=self.geo, detector_coords=detector_coords)
            self.geo = self.geo_drawer.geo

        # Store whether to draw in lite mode
        self.lite = lite

        # Store the layout information
        self.split_scene = split_scene
        self.detector_coords = detector_coords
        self.meta = self.meta if self.geo is None else None
        self.layout_kwargs = kwargs

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

    @property
    def source_modes(self):
        """Dictionary which makes the correspondance between the name of a true
        object source attribute with the underlying source array it points to.

        Returns
        -------
        Dict[str, str]
            Dictionary of (attribute, key) mapping for point source
        """
        return dict(self._source_modes)

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

    def get(
        self,
        obj_type,
        attr=None,
        color_attr=None,
        draw_raw=False,
        draw_end_points=False,
        draw_directions=False,
        draw_vertices=False,
        draw_flashes=False,
        matched_flash_only=True,
        draw_crthits=False,
        matched_crthit_only=True,
        synchronize=False,
        titles=None,
        split_traces=False,
    ):
        """Draw the requested object type with the requested mode.

        Parameters
        ----------
        obj_type : str
            Name of the object type to draw (one of 'fragments', 'particles' or
            'interactions'
        attr : Union[str, List[str]], optional
            Name of list of names of attributes to draw
        color_attr : str, optional
            Name of the attribute to use to determine the color
        draw_raw : bool, default False
            If `True`, add a trace which corresponds to the raw depositions
        draw_end_points : bool, default False
            If `True`, draw the fragment or particle end points
        draw_directions : bool, default False
            If `True`, draw the fragment or particle start directions
        draw_vertices : bool, default False
            If `True`, draw the interaction vertices
        draw_flashes : bool, default False
            If `True`, draw the flashes
        matched_flash_only : bool, default True
            If `True`, only flashes matched to interactions are drawn
        draw_crthits : bool, default False
            If `True`, draw the CRT hits
        matched_crthit_only : bool, default True
            If `True`, only CRT hits matched to interactions are drawn
        synchronize : bool, default False
            If `True`, matches the camera position/angle of one plot to the other
        titles : List[str], optional
            Titles of the two scenes (only relevant for split_scene=True)
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
            f"{self._obj_types}."
        )

        # Fetch the objects
        traces = {}
        for prefix in self.prefixes:
            obj_name = f"{prefix}_{obj_type}"
            assert obj_name in self.data, (
                f"Must provide `{obj_name}` in the data products to draw " "them."
            )
            traces[prefix] = self._object_traces(
                obj_name, attr, color_attr, split_traces
            )

        # Fetch the raw depositions, if requested
        if draw_raw:
            for prefix in self.prefixes:
                traces[prefix] = self._raw_trace() + traces[prefix]

        # Fetch the end points, if requested
        if draw_end_points:
            assert (
                obj_type != "interactions"
            ), "Interactions do not have end point attributes."
            for prefix in self.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                traces[prefix] += self._start_point_trace(obj_name, split_traces)
                traces[prefix] += self._end_point_trace(obj_name, split_traces)

        # Fetch the directions, if requested
        if draw_directions:
            assert (
                obj_type != "interactions"
            ), "Interactions do not have direction attributes."
            for prefix in self.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                traces[prefix] += self._direction_trace(obj_name, split_traces)

        # Fetch the vertices, if requested
        if draw_vertices:
            for prefix in self.prefixes:
                obj_name = f"{prefix}_interactions"
                assert (
                    obj_name in self.data
                ), "Must provide interactions to draw their vertices."
                traces[prefix] += self._vertex_trace(obj_name, split_traces)

        # Fetch the flashes, if requested
        show_optical = False
        if draw_flashes:
            assert (
                "flashes" in self.data
            ), "Must provide the `flashes` objects to draw them."
            show_optical = True
            for prefix in self.prefixes:
                obj_name = f"{prefix}_interactions"
                assert (
                    obj_name in self.data
                ), "Must provide interactions to draw matched flashes."
                traces[prefix] += self._flash_trace(obj_name, matched_flash_only)

        # Fetch the CRT hits, if requested
        show_crt = False
        if draw_crthits:
            assert (
                "crthits" in self.data
            ), "Must provide the `crthits` objects to draw them."
            show_crt = True
            for prefix in self.prefixes:
                obj_name = f"{prefix}_{obj_type}"
                traces[prefix] += self._crt_trace(obj_name, matched_crthit_only)

        # Add the TPC traces, if available
        if self.geo_drawer is not None:
            if self.prefixes and self.split_scene:
                for prefix in self.prefixes:
                    traces[prefix] += self.geo_drawer.tpc_traces(meta=self.meta)
            else:
                traces[self.prefixes[-1]] += self.geo_drawer.tpc_traces(meta=self.meta)

        # Initialize the layout based on what needs to be shown
        layout = layout3d(
            geo=self.geo,
            use_geo=self.geo is not None,
            meta=self.meta,
            detector_coords=self.detector_coords,
            show_optical=show_optical,
            show_crt=show_crt,
            **self.layout_kwargs,
        )

        # Initialize the figure, return
        if len(self.prefixes) > 1 and self.split_scene:
            if titles is None:
                titles = [f"Reconstructed {obj_type}", f"Truth {obj_type}"]
            figure = dual_figure3d(
                traces["reco"],
                traces["truth"],
                layout=layout,
                synchronize=synchronize,
                titles=titles,
            )

        else:
            assert titles is None, (
                "Providing titles does not do anything when split_scene " "is False."
            )
            all_traces = []
            for trace_group in traces.values():
                all_traces += trace_group
            figure = go.Figure(all_traces, layout=layout)

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
        # Get the colors
        if attr is not None and isinstance(attr, str):
            attr = [attr]
        color_dict = self._object_colors(obj_name, attr, color_attr, split_traces)

        # Fetch object representations
        if not self.lite:
            # Check that the object has points to be drawn
            point_key = self.truth_point_key if "truth" in obj_name else "points"
            assert point_key in self.data, (
                f"The `{point_key}` attribute must be provided if the full "
                f"version of the `{obj_name}` objects is to be drawn."
            )

            # Fetch the points and the clusters
            points = self.data[point_key]
            clusts = [self.get_index(obj) for obj in self.data[obj_name]]

            # Initialize traces
            traces = scatter_clusters(
                points,
                clusts,
                single_trace=not split_traces,
                shared_legend=not split_traces,
                **color_dict,
            )

        else:
            # Initialize traces
            traces = scatter_lite(
                self.data[obj_name],
                shared_legend=not split_traces,
                **color_dict,
            )

        return traces

    def _object_colors(self, obj_name, attrs, color_attr, split_traces):
        """Provides an appropriate colorscale and range for a given attribute.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        attrs : Optional[List[str]]
            Attribute name(s) used to set the color/hovertext
        color_attr : Optional[str]
            Name of the attribute to use to determine the color
        split_traces : bool
            If `True`, one trace is produced for each object

        Returns
        -------
        dict
            Dictionary of color parameters (colorscale, cmin, cmax)
        """
        # Define the name of the trace group
        name = " ".join(obj_name.split("_")).capitalize()
        if split_traces:
            name = name[:-1]

        # Make sure the color attribute is included in the hovertext attributes
        if color_attr is not None:
            assert color_attr in attrs, (
                "The attribute used to define the color scale must be "
                "included in the list of hovertext attributes."
            )

        # Make sure that the geometry is provided if needed
        if any(self._is_sources(attr) for attr in attrs or []):
            assert self.geo is not None, (
                "Provide detector name/geometry if the TPC "
                "sources are to be displayed."
            )

        # Initialize hovertext per object
        obj_type = obj_name.split("_")[-1][:-1].capitalize()
        count = len(self.data[obj_name])
        hovertext = [[f"{obj_type} {i}"] for i in range(count)]
        has_long_form = False
        if attrs is not None and any(self._is_long_form(attr) for attr in attrs):
            if self.lite:
                raise ValueError(
                    "Long-form attributes are not available when drawing "
                    "lite-version objects."
                )
            point_key = self.truth_point_key if "truth" in obj_name else "points"
            hovertext = [ht * len(self.data[point_key]) for ht in hovertext]
            has_long_form = True

        # Fetch the list of colors and hovertexts
        color = np.arange(len(self.data[obj_name]))
        if color_attr is None:
            color_attr = "id"

        if attrs is not None:
            # Fetch hover information for each of the requested attributes
            for attr in attrs:
                # Get the atribute name in display format
                attr_name = " ".join(attr.split("_")).capitalize()

                # If it is a true deposition/source attribute, check that it matches
                # the point mode that is being used to draw the true objects
                if "truth" in obj_name:
                    if self._is_depositions(attr):
                        prefix = self.truth_point_mode.replace("points", "depositions")
                        assert attr.startswith(prefix), (
                            f"Points mode {self.truth_point_mode} and deposition "
                            f"mode {attr} are incompatible."
                        )
                    if self._is_sources(attr):
                        ref_name = self.truth_point_mode.replace("points", "sources")
                        assert attr == ref_name, (
                            f"Points mode {self.truth_point_mode} and source "
                            f"mode {attr} are incompatible."
                        )

                # Get the list of values
                values = [getattr(obj, attr) for obj in self.data[obj_name]]
                if self._is_sources(attr):
                    assert self.geo is not None  # For the linter's sake
                    values = [self.geo.get_sources(v) for v in values]

                # Get the colors, if needed
                if len(attrs) == 1 or attr == color_attr:
                    if not self._is_sources(attr):
                        color = values
                    else:
                        assert self.geo is not None, (
                            "Provide detector name/geometry if the TPC "
                            "sources are to be displayed."
                        )
                        color = [self.geo.get_chambers(v) for v in values]

                # Get the hovertext for this attribute
                if self._is_depositions(attr):
                    for i, ht in enumerate(hovertext):
                        hovertext[i] = [
                            ht[j] + self._dep_tostr(v) for j, v in enumerate(values[i])
                        ]
                elif self._is_sources(attr):
                    for i, ht in enumerate(hovertext):
                        hovertext[i] = [
                            ht[j] + self._src_tostr(v) for j, v in enumerate(values[i])
                        ]
                else:
                    for i, ht in enumerate(hovertext):
                        val_str = self._tostr(attr_name, values[i])
                        hovertext[i] = [
                            ht[j] + val_str for j in range(len(hovertext[i]))
                        ]

            # If there is only one attribute and no color attribute specified, use it
            if color_attr is None and len(attrs) == 1:
                color_attr = attrs[0]

        # Flatten hovertext if no long-form attributes are used
        if not has_long_form:
            hovertext = [ht[0] for ht in hovertext]

        # Set up the appropriate color scheme
        if self._is_depositions(color_attr):
            # Continuous values shared between objects
            dep_mode = (
                self.dep_modes[color_attr] if "truth" in obj_name else "depositions"
            )
            colorscale = "Inferno"
            cmin = 0.0
            cmax = 2 * np.median(self.data[dep_mode])

        elif self._is_sources(color_attr):
            # Variable-length descrete values
            assert self.geo is not None  # For the linter's sake
            count = self.geo.tpc.num_chambers
            colorscale = HIGH_CONTRAST_COLORS
            if count == 0:
                colorscale = None
            elif count == 1:
                colorscale = [colorscale[0]] * 2  # Avoid length 0 colorscale
            elif count <= len(colorscale):
                colorscale = colorscale[:count]
            else:
                repeat = (count - 1) // len(colorscale) + 1
                colorscale = np.tile(colorscale, repeat)[:count]

            cmin = 0
            cmax = count - 1

        elif color_attr.startswith("is_"):
            # Boolean
            color = np.array(color, dtype=np.int32)
            colorscale = PLOTLY_COLORS_WGRAY[1:3]
            cmin = 0
            cmax = 1

        elif color_attr == "shape" or color_attr == "pid":
            # Fixed length values, with potentially invalid values
            ref = SHAPE_LABELS if color_attr == "shape" else PID_LABELS
            num_classes = len(ref)
            colorscale = PLOTLY_COLORS_WGRAY[: num_classes + 1]
            cmin = -1
            cmax = num_classes - 1

        elif color_attr.endswith("id"):
            # Variable-lengh discrete values
            unique, color = np.unique(color, return_inverse=True)
            colorscale = HIGH_CONTRAST_COLORS
            count = len(unique)
            if count == 0:
                colorscale = None
            elif count == 1:
                colorscale = [colorscale[0]] * 2  # Avoid length 0 colorscale
            elif count <= len(colorscale):
                colorscale = colorscale[:count]
            else:
                repeat = (count - 1) // len(colorscale) + 1
                colorscale = np.tile(colorscale, repeat)[:count]

            cmin = 0
            cmax = count - 1

        else:
            raise ValueError(f"Color attribute not supported: {color_attr}.")

        return {
            "color": color,
            "hovertext": hovertext,
            "name": name,
            "colorscale": colorscale,
            "cmin": cmin,
            "cmax": cmax,
        }

    def _raw_trace(self):
        """Draws the raw input image (pre-reconstruction).

        Returns
        -------
        List[plotly.graph_objs.Scatter3d]
            List of one trace containing the input to the reconstruction
        """
        # Throw if drawing in lite mode
        if self.lite:
            raise RuntimeError("Cannot draw raw input in lite mode.")

        # Fetch the input attributes
        points = self.data["points"]
        deps = self.data["depositions"]

        # Fetch the colorscale limits
        cmin = 0.0
        cmax = 2 * np.median(deps) if len(deps) else 1.0

        return scatter_points(
            points,
            color=deps,
            cmin=cmin,
            cmax=cmax,
            colorscale="Inferno",
            name="Raw input",
        )

    def _start_point_trace(
        self,
        obj_name,
        split_traces,
        color="black",
        markersize=7,
        marker_symbol="circle",
        **kwargs,
    ):
        """Scatters the start points of the requested object type.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        split_traces : bool
            If `True`, one trace is produced for each object
        color : Union[str, np.ndarray], default 'black'
            Color of markers/lines or (N) list of color of markers/lines
        markersize : float, default 7
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
            obj_name,
            "start_point",
            split_traces,
            color=color,
            markersize=markersize,
            marker_symbol=marker_symbol,
            **kwargs,
        )

    def _end_point_trace(
        self,
        obj_name,
        split_traces,
        color="black",
        markersize=7,
        marker_symbol="circle-open",
        **kwargs,
    ):
        """Scatters the end points of the requested object type.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        split_traces : bool
            If `True`, one trace is produced for each object
        color : Union[str, np.ndarray], default 'black'
            Color of markers/lines or (N) list of color of markers/lines
        markersize : float, default 7
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
            obj_name,
            "end_point",
            split_traces,
            color=color,
            markersize=markersize,
            marker_symbol=marker_symbol,
            **kwargs,
        )

    def _vertex_trace(
        self,
        obj_name,
        split_traces,
        vertex_attr="vertex",
        color="green",
        markersize=10,
        marker_symbol="diamond",
        **kwargs,
    ):
        """Scatters the vertex of the requested object type.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        split_traces : bool
            If `True`, one trace is produced for each object
        color : Union[str, np.ndarray], default 'green'
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
            obj_name,
            vertex_attr,
            split_traces,
            color=color,
            markersize=markersize,
            marker_symbol=marker_symbol,
            **kwargs,
        )

    def _point_trace(self, obj_name, point_attr, split_traces, **kwargs):
        """Scatters a set of discrete points per object instance.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        point_attr : str
            Name of the attribute specifying end point to draw
        split_traces : bool
            If `True`, one trace is produced for each object
        **kwargs : dict, optional
            List of additional arguments to pass to :func:`scatter_points`

        Returns
        -------
        list
            List of point traces
        """
        # Define the name of the trace
        name = (
            " ".join(obj_name.split("_")).capitalize()[:-1]
            + " "
            + " ".join(point_attr.split("_"))
        )

        # Fetch the particular end point of each object
        obj_type = obj_name.split("_")[-1][:-1].capitalize()
        point_list, hovertext, idxs = [], [], []
        for i, obj in enumerate(self.data[obj_name]):
            # If it is an end point, skip if the object is not a track
            if point_attr == "end_point" and obj.shape != TRACK_SHP:
                continue

            # Skip empty true objects
            if obj.is_truth and not len(getattr(obj, self.truth_index_mode)) > 0:
                continue

            # Append the particular end point of this object and the label
            point_list.append(getattr(obj, point_attr))
            hovertext.append(f"{obj_type} {i} " + " ".join(point_attr.split("_")))
            idxs.append(i)

        points = np.empty((0, 3))
        if point_list:
            points = np.vstack(point_list)

        if not split_traces:
            traces = scatter_points(
                points, hovertext=np.array(hovertext), name=name, **kwargs
            )

        else:
            traces = []
            for i, point in enumerate(point_list):
                traces += scatter_points(
                    point[None, :],
                    hovertext=hovertext[i],
                    name=f"{name} {idxs[i]}",
                    **kwargs,
                )

        return traces

    def _direction_trace(self, obj_name, split_traces, color="black", **kwargs):
        """Scatters a set of discrete points per object instance.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        split_traces : bool
            If `True`, one trace is produced for each object
        color : Union[str, np.ndarray], default 'black'
            Color of markers/lines or (N) list of color of markers/lines
        **kwargs : dict, optional
            List of additional arguments to pass to :func:`scatter_arrows`

        Returns
        -------
        list
            List of point traces
        """
        # Define the name of the trace
        name = " ".join(obj_name.split("_")).capitalize()[:-1] + " directions"

        # Fetch the direction of each object
        obj_type = obj_name.split("_")[-1][:-1].capitalize()
        point_list, dir_list, hovertext, idxs = [], [], [], []
        for i, obj in enumerate(self.data[obj_name]):
            # Skip empty true objects
            if obj.is_truth and not len(getattr(obj, self.truth_index_mode)) > 0:
                continue

            # Append the direction of this object and the label
            point_list.append(obj.start_point)
            dir_list.append(obj.start_dir)
            hovertext.append(f"{obj_type} {i} direction")
            idxs.append(i)

        points, dirs = np.empty((0, 3)), np.empty((0, 3))
        if point_list:
            points = np.vstack(point_list)
            dirs = np.vstack(dir_list)

        if not split_traces:
            traces = scatter_arrows(
                points,
                dirs,
                hovertext=np.array(hovertext),
                name=name,
                color=color,
                **kwargs,
            )

        else:
            traces = []
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

    def _flash_trace(self, obj_name, matched_only, **kwargs):
        """Draw the cumulative PEs of flashes.

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
        assert (
            self.geo_drawer is not None
        ), "Cannot draw optical detectors without geometry information."

        # Check that there are optical detectors to draw
        assert (
            self.geo is not None and self.geo.optical is not None
        ), "This geometry does not have optical detectors to draw."

        # Define the name of the trace
        name = " ".join(obj_name.split("_")).capitalize()[:-1] + " flashes"

        # Find the list of flash IDs to draw
        if matched_only:
            flash_ids = []
            for inter in self.data[obj_name]:
                if inter.is_flash_matched:
                    flash_ids.extend(inter.flash_ids)
        else:
            flash_ids = np.arange(len(self.data["flashes"]))

        # Sum values from each flash to build a a global color scale
        color = np.zeros(self.geo.optical.num_detectors)
        op_det_ids = self.geo.optical.det_ids
        for flash_id in flash_ids:
            flash = self.data["flashes"][flash_id]
            pe_per_ch = flash.pe_per_ch
            if not self.geo.optical.global_index:
                op_det_ids = self.geo.optical.volumes[flash.volume_id].det_ids
            if op_det_ids is not None:
                pe_per_ch = np.bincount(op_det_ids, weights=pe_per_ch)
            index = self.geo.optical.volume_index(flash.volume_id)
            color[index] += pe_per_ch

        # Return the set of optical detectors with a color scale
        return self.geo_drawer.optical_traces(
            meta=self.meta,
            color=color,
            zero_supress=True,
            colorscale="Inferno",
            name=name,
            **kwargs,
        )

    def _crt_trace(self, obj_name, matched_only, **kwargs):
        """Draw the CRT planes and the hits.

        Parameters
        ----------
        obj_name : str
            Name of the object to draw
        matched_only : bool
            If `True`, only CRT hits matched to interactions are drawn
        **kwargs : dict, optional
            List of additional arguments to pass to :func:`optical_traces`

        Returns
        -------
        list
            List of optical detector traces
        """
        # If there was no geometry provided by the user, nothing to do here
        assert (
            self.geo is not None and self.geo_drawer is not None
        ), "Cannot draw CRT detectors without geometry information."

        # Check that there are CRT planes to draw
        assert (
            self.geo.crt is not None
        ), "This geometry does not have CRT planes to draw."

        # Define the names of the traces
        name_pl = " ".join(obj_name.split("_")).capitalize()[:-1] + " CRT planes"
        name_hits = " ".join(obj_name.split("_")).capitalize()[:-1] + " CRT hits"

        # Fetch CRT hits. Restrict to matched hits, if requested
        crthits = self.data["crthits"]
        if matched_only:
            crt_ids = []
            for inter in self.data[obj_name]:
                if inter.is_crt_matched:
                    crt_ids.extend(inter.crt_ids)
            crt_ids = np.unique(crt_ids)
            crthits = [crthits[idx] for idx in crt_ids]

        # Identify which of the CRT planes were hit (to know what to draw)
        det_ids = [self.geo.crt.get_plane_id(hit.center, hit.plane) for hit in crthits]
        unique_det_ids = np.unique(det_ids)

        # Initialize the hovertext for the planes and hits
        hovertext_pl, hovertext_hits = [], []
        for i, det_id in enumerate(unique_det_ids):
            hovertext_pl.append(f"CRT Plane {det_id}")
        for i, hit in enumerate(crthits):
            hovertext_hits.append(f"CRT hit {hit.id}<br>CRT Plane ID: {det_ids[i]}")

        # Initialize the CRT plane traces
        traces = self.geo_drawer.crt_traces(
            meta=self.meta,
            draw_ids=unique_det_ids,
            hovertext=hovertext_pl,
            name=name_pl,
        )

        # Build a scatter plot of CRT hits
        points = np.empty((0, 3))
        if crthits:
            points = np.vstack([hit.center for hit in crthits])

        traces += scatter_points(
            points,
            color="gray",
            markersize=5,
            hovertext=hovertext_hits,
            name=name_hits,
            **kwargs,
        )

        return traces

    def _is_long_form(self, attr):
        """Check if an attribute is a long-form attribute (depositions or sources).

        Parameters
        ----------
        attr : str
            Object attribute to check

        Returns
        -------
        bool
            `True` is the attribute is a long-form attribute
        """
        return self._is_depositions(attr) or self._is_sources(attr)

    @staticmethod
    def _is_depositions(attr):
        """Check if an attribute represents one deposition value per point.

        Parameters
        ----------
        attr : str
            Object attribute to check

        Returns
        -------
        bool
            `True` is the attribute is a deposition attribute
        """
        return attr.startswith("depositions") and not attr.endswith("sum")

    @staticmethod
    def _is_sources(attr):
        """Check if an attribute represents one source value per point.

        Parameters
        ----------
        attr : str
            Object attribute to check

        Returns
        -------
        bool
            `True` is the attribute is a source attribute
        """
        return attr.startswith("sources")

    def _tostr(self, attr_name, value):
        """Convert a scalar attribute value to a human-readable string.

        Parameters
        ----------
        attr_name : str
            Name of the attribute
        value : Any
            Attribute value

        Returns
        -------
        str
            Human-readable string
        """
        return f"<br>{attr_name}: {value}"

    def _dep_tostr(self, value):
        """Convert a deposition value to a human-readable string.

        Parameters
        ----------
        value : str
            Deposition value

        Returns
        -------
        str
            Human-readable string
        """
        return f"<br>Deposition: {value:0.3f}"

    def _src_tostr(self, value):
        """Convert a source value to a human-readable string.

        Parameters
        ----------
        value : str
            Source value

        Returns
        -------
        str
            Human-readable string
        """
        return f"<br>Module, TPC: {value[0]:d}, {value[1]:d}"
