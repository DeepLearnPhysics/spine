"""Defines default plotly layouts."""

from copy import deepcopy

import numpy as np
from plotly import colors
from plotly import graph_objs as go
from plotly.subplots import make_subplots
import matplotlib as mpl
import seaborn as sns

from spine.utils.geo import Geometry

# Colorscale definitions
PLOTLY_COLORS = colors.qualitative.Plotly
PLOTLY_COLORS_TUPLE = colors.convert_colors_to_same_type(
        deepcopy(PLOTLY_COLORS), 'tuple')[0]
PLOTLY_COLORS_WGRAY = ['#808080'] + PLOTLY_COLORS
HIGH_CONTRAST_COLORS = np.concatenate(
        [colors.qualitative.Dark24, colors.qualitative.Light24])


def layout3d(ranges=None, meta=None, detector=None, titles=None,
             detector_coords=False, backgroundcolor='white',
             gridcolor='lightgray', width=800, height=800, showlegend=True,
             camera=None, aspectmode='manual', aspectratio=None, dark=False,
             margin=None, hoverlabel=None, **kwargs):
    """Produces plotly.graph_objs.Layout object for a certain format.

    Parameters
    ----------
    ranges : np.ndarray, optional
        (3, 2) or (N, 3) Array used to specify the plot region in (x,y,z)
        directions. If not specified (None), the range will be set to include
        all points. Alternatively can be an array of shape (3,2) specifying
        (x,y,z) axis (min,max) range for a display, or simply a list of points
        with shape (N,3+) where [:,0],[:,1],[:,2] correspond to (x,y,z) values
        and the plotting region is decided by measuring the min,max range in
        each coordinates. This last option is useful if one wants to define
        the region based on a set of points that is not same as what's plotted.
    meta : Meta, optional
        Metadata information used to infer the full image range
    detector : str
        Name of a recognized detector to get the geometry from
    titles : List[str], optional
        (3) Array of strings for (x,y,z) axis title respectively
    detector_coords : bool, default False
        Whether or not the coordinates being drawn are in detector_coordinates
        or pixel IDs
    backgroundcolor : Union[str, int], default 'white'
        Color of the layout background
    gridcolor : Union[str, int], default 'lightgray'
        Color of the grid
    width : int, default 900
        Width of the layout in pixels
    height : int, default 900
        Height of the layout in pixels
    showlegend : bool, default True
        Whether or not to show the image legend
    aspectmode : str, default manual
        Plotly aspect mode. If manual, will define it based on the ranges
    aspectratio : dict, optional
        Plotly dictionary which specifies the aspect ratio for x, y an d z
    dark : bool, default False
        Dark layout
    margin : dict, optional
        Specifies the margin in each subplot
    hoverlabel : dict, optional
        Specifies the style hovertext labels
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Layout

    Results
    -------
    plotly.graph_objs.Layout
        Object that can be given to plotly.graph_objs.Figure for visualization
        (together with traces)
    """
    # Figure out the drawing range
    if ranges is None:
        ranges = [None, None, None]
    else:
        # If the range is provided, just use it
        if ranges.shape != (3, 2):
            assert len(ranges.shape) == 2 and ranges.shape[1] == 3, (
                "If ranges is not of shape (3, 2), it must be of shape (N, 3).")
            ranges = np.vstack(
                    [np.min(ranges, axis=0), np.max(ranges, axis=0)]).T

        # Check that the range is sensible
        assert np.all(ranges[:, 1] >= ranges[:, 0])

    if detector is not None:
        # If detector geometry is provided, make the full detector the range
        assert ranges is None or None in ranges, (
                "Should not specify `detector` along with `ranges`.")
        geo = Geometry(detector)
        lengths = geo.tpc.dimensions
        ranges = geo.tpc.boundaries

        # Add some padding
        ranges[:, 0] -= lengths*0.1
        ranges[:, 1] += lengths*0.1

        # If pixel coordinates are requested, use meta to make the conversion
        if detector_coords is False:
            assert meta is not None, (
                    "Must provide metadata information to convert the detector "
                    "coordinates to pixel coordinates.")
            ranges = meta.to_px(ranges.T).T

    elif meta is not None:
        # If meta information is provided, make the full image the range
        assert ranges is None or None in ranges, (
                "Should not specify both `ranges` and `meta` parameters.")
        if detector_coords:
            ranges = np.vstack([meta.lower, meta.upper]).T
        else:
            ranges = np.vstack([[0, 0, 0],
                np.round((meta.upper - meta.lower)/meta.size)]).T


    # Define detector-style camera, unless explicitely provided
    if camera is None:
        camera = {'eye':    {'x':-2., 'y': 1.,   'z': -0.01},
                  'up':     {'x':0.,  'y': 1.,   'z': 0.},
                  'center': {'x':0.,  'y': -0.1, 'z': -0.01}}

    # Infer the image width/height and aspect ratios, unless they are specified
    if aspectmode == 'manual':
        if aspectratio is None:
            axes = ['x', 'y', 'z']
            ratios = np.ones(len(ranges))
            if ranges[0] is not None:
                max_range = np.max(ranges[:, 1]-ranges[:, 0])
                ratios = (ranges[:, 1] - ranges[:, 0])/max_range
            aspectratio = {axes[i]: v for i, v in enumerate(ratios)}

    # Check on the axis titles, define default
    assert titles is None or len(titles) == 3, 'Must specify one title per axis'
    if titles is None:
        unit = 'cm' if detector_coords else 'pixel'
        titles = [f'x [{unit}]', f'y [{unit}]', f'z [{unit}]']

    # Initialize some default legend behavior
    if 'legend' not in kwargs:
        kwargs['legend'] = {
                'title':'Legend', 'tracegroupgap': 1, 'itemsizing': 'constant'}

    # If a dark layout is requested, set the theme and the background color
    # accordingly
    if dark:
        kwargs['template'] = 'plotly_dark'
        kwargs['paper_bgcolor'] = 'black'
        backgroundcolor = 'black'

    # If the margin is not provided, use 0 by default
    if margin is None:
        margin = {'b': 0, 't': 0, 'l': 0, 'r': 0}

    # Set hoverlabel font, if not provided
    if hoverlabel is None:
        hoverlabel = {'font_family': 'Droid sans, monospace'}

    # Initialize the general scene layout
    axis_base = {'nticks': 10, 'showticklabels': True, 'tickfont': {'size': 14},
                 'backgroundcolor': backgroundcolor, 'gridcolor': gridcolor,
                 'showbackground': True}

    scene = {'aspectmode': aspectmode, 'aspectratio': aspectratio,
             'camera': camera}
    for i, axis in enumerate(('x', 'y', 'z')):
        scene[f'{axis}axis'] = {
                'title': {'text': titles[i], 'font': {'size': 20}},
                'range': ranges[i], **axis_base}

    # Initialize layout
    layout = go.Layout(
            showlegend=showlegend, width=width, height=height, margin=margin,
            scene1=scene, scene2=deepcopy(scene), scene3=deepcopy(scene),
            hoverlabel=hoverlabel, **kwargs)

    return layout


def dual_figure3d(traces_left, traces_right, layout=None, titles=None,
                  width=1500, height=750, synchronize=False,
                  margin=None, **kwargs):
    """Function which returns a plotly.graph_objs.Figure with two set of traces
    side-by-side in separate subplots.

    Parameters
    ----------
    traces_left : List[object]
        List of plotly traces to draw in the left subplot
    traces_right : List[object]
        List of plotly traces to draw in the right subplot
    layout : plotly.Layout, optional
        Plotly layout
    titles : List[str], optional
        Titles of the two subplots
    width : int, default 1000
        Width of the layout in pixels
    height : int, default 500
        Height of the layout in pixels
    synchronize : bool, default False
        If True, matches the camera position/angle of one plot to the other
    margin : dict, optional
        Specifies the margin in each subplot
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Layout

    Returns
    -------
    plotly.graph_objs.Figure
        Plotly figure with the two subplots
    """
    # If no title is provided, ommit them both
    if titles is None:
        titles = [None, None]

    # Make subplot and add traces
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=(titles[0], titles[1]),
                        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                        horizontal_spacing=0.05, vertical_spacing=0.04)

    num_left, num_right = len(traces_left), len(traces_right)
    fig.add_traces(traces_left, rows=[1]*num_left, cols=[1]*num_left)
    fig.add_traces(traces_right, rows=[1]*num_right, cols=[2]*num_right)

    # If the margin is not provided, use 20 by default
    if margin is None:
        margin = {'b': 20, 't': 20, 'l': 20, 'r': 20}

    # Inialize and set layout
    if layout is None:
        layout = layout3d(width=width, height=height, margin=margin, **kwargs)
    else:
        layout.update({'width': width, 'height': height, 'margin': margin})
    fig.layout.update(layout)

    # If requested, synchronize the two cameras
    if synchronize:
        fig = go.FigureWidget(fig)

        def cam_change_left(scene, camera): # pylint: disable=W0613
            fig.layout.scene2.camera = camera

        def cam_change_right(scene, camera): # pylint: disable=W0613
            fig.layout.scene1.camera = camera

        fig.layout.scene1.on_change(cam_change_left,  'camera')
        fig.layout.scene2.on_change(cam_change_right, 'camera')

    return fig


def apply_latex_style():
    """Sets the necessary :mod:`matplotlib` and :mod:`seaborn` parameters
    to draw a plot using latex style.
    """
    sns.set(rc={'figure.figsize': set_latex_size(250),
                'text.usetex': True,
                'font.family': 'serif',
                'axes.labelsize': 8,
                'font.size': 8,
                'legend.fontsize': 8,
                'legend.labelspacing': 0.25,
                'legend.columnspacing': 0.25,
                'xtick.labelsize': 8,
                'ytick.labelsize': 8,}, context='paper')
    sns.set_style('white')
    sns.set_style(rc={'axes.grid': True, 'font.family': 'serif'})
    mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath,bm}"]


def set_latex_size(width, fraction=1):
    """Returns optimal figure dimension for a latex plot, depending on
    the requested width.

    Parameters
    ----------
    width : int
        Width of the page in points (pixels)
    fraction : float, default 1
        Fraction of the page width used by the figure

    Returns
    -------
    width : float
        Width of the figure in inches
    height : float
        Height of the figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt

    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in


def color_rgba(color, alpha):
    """Convert an RGB color array into an RGBA string.

    Parameters
    ----------
    color : List[int]
        (3) List of RGB values
    alpha : float
        Alpha value in [0, 1]

    Returns
    -------
    str
        RGBA string
    """
    r, g, b = color
    return f'rgba({r}, {g}, {b}, {alpha})'
