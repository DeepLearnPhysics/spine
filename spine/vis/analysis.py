"""Draw analysis-level objects (RecoParticle, TruthParticle)."""

import plotly
import plotly.graph_objs as go
import numpy as np

from spine.utils.globals import COORD_COLS, PID_LABELS, SHAPE_LABELS

class Scatter3D:

    def __init__(self):

        self._traces = []
        self._colors = {}

        self._color_bounds = [None, None]
        self._colorscale = None

    def clear_state(self):
        self._traces = []
        self._colors = {}
        self._color_bounds = [None, None]
        self._colorscale = None

    def scatter_start_points(self, particles, prefix=''):
        for p in particles:
            if p.start_point is not None and (np.abs(p.start_point) < 1e8).all():
                plot = go.Scatter3d(x=np.array([p.start_point[0]]),
                    y=np.array([p.start_point[1]]),
                    z=np.array([p.start_point[2]]),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='red',
                        # colorscale=colorscale,
                        opacity=0.6),
                        # hovertext=p.ppn_candidates[:, 4],
                    name='{} {} Startpoint'.format(type(p).__name__, p.id))
                self._traces.append(plot)

    def scatter_end_points(self, particles, prefix=''):
        for p in particles:
            if p.end_point is not None and (np.abs(p.end_point) < 1e8).all():
                plot = go.Scatter3d(x=np.array([p.end_point[0]]),
                    y=np.array([p.end_point[1]]),
                    z=np.array([p.end_point[2]]),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='cyan',
                        # line=dict(width=2, color='red'),
                        # cmin=cmin, cmax=cmax,
                        # colorscale=colorscale,
                        opacity=0.6),
                        # hovertext=p.ppn_candidates[:, 4],
                    name='{} {} Endpoint'.format(type(p).__name__, p.id))
                self._traces.append(plot)

    def scatter_vertices(self, interactions):
        for ia in interactions:
            if ia.vertex is not None and (np.abs(ia.vertex) < 1e8).all():
                plot = go.Scatter3d(x=np.array([ia.vertex[0]]),
                    y=np.array([ia.vertex[1]]),
                    z=np.array([ia.vertex[2]]),
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='cyan',
                        # line=dict(width=2, color='red'),
                        # cmin=cmin, cmax=cmax,
                        # colorscale=colorscale,
                        opacity=0.6),
                        # hovertext=p.ppn_candidates[:, 4],
                    name='{} {} Vertex'.format(type(ia).__name__, ia.id))
                self._traces.append(plot)

    def set_pixel_color(self, objects, color, colorscale=None, precmin=None, precmax=None, mode='points'):

        cmin, cmax = np.inf, -np.inf

        if 'depositions' not in color:
            for entry in objects:
                attribute = getattr(entry, color)
                assert np.isscalar(attribute)
                self._colors[int(entry.id)] = int(attribute) \
                    * np.ones(getattr(entry, mode).shape[0], dtype=np.int64)

                if int(attribute) < cmin:
                    cmin = int(attribute)
                if int(attribute) > cmax:
                    cmax = int(attribute)
        else:
            for entry in objects:
                depositions = getattr(entry, color)
                assert isinstance(depositions, np.ndarray)
                self._colors[int(entry.id)] = depositions
                dmin, dmax = depositions.min(), depositions.max()
                if dmin < cmin:
                    cmin = dmin
                if dmax > cmax:
                    cmax = dmax

        self._color_bounds = [cmin, cmax]

        # Define limits
        if color == 'pid':
            values = list(PID_LABELS.keys())
            self._color_bounds = [-1, max(values)]
        elif color == 'semantic_type':
            values = list(SHAPE_LABELS.keys())
            self._color_bounds = [-1, max(values)]
        elif color == 'is_primary':
            self._color_bounds = [-1, 1]
        elif 'depositions' in color:
            self._color_bounds = [0, cmax]

        # If manually specified, overrule
        if precmin is not None: self._color_bounds[0] = precmin
        if precmax is not None: self._color_bounds[1] = precmax

        # Define colorscale
        self._colorscale = colorscale
        if isinstance(colorscale, str) and hasattr(plotly.colors.qualitative, colorscale):
            self._colorscale = getattr(plotly.colors.qualitative, colorscale)
        if isinstance(colorscale, list) and isinstance(colorscale[0], str):
            count = np.round(self._color_bounds[1] - self._color_bounds[0]) + 1
            if count < len(colorscale):
                self._colorscale = colorscale[:count]
            if count > len(colorscale):
                repeat = int((count-1)/len(colorscale)) + 1
                self._colorscale = np.repeat(colorscale, repeat)[:count]


    def check_attribute_name(self, objects, color):

        attr_list = [att for att in dir(objects[0]) if att[0] != '_']
        if color not in attr_list:
            raise ValueError(f'"{color}" is not a valid attribute for object type {type(objects[0])}!')

    def __call__(self, objects, color='id', mode='points', colorscale='rainbow',
                 legend_name=None,
                 cmin=None, cmax=None, size=1, scatter_start_points=False,
                 scatter_end_points=False, scatter_vertices=False, **kwargs):

        if not len(objects):
            return []

        self.check_attribute_name(objects, color)
        self.clear_state()

        self.set_pixel_color(objects, color, colorscale, cmin, cmax, mode)

        for entry in objects:
            if getattr(entry, mode).shape[0] <= 0:
                continue
            c = self._colors[int(entry.id)].tolist()
            hovertext = [f'{color}: {ci}' for ci in c]

            if legend_name is None:
                name = type(entry).__name__
            else:
                name = legend_name

            plot = scatter_points(getattr(entry, mode)[:, :3],
                                  color = c, colorscale = self._colorscale,
                                  cmin = self._color_bounds[0], cmax = self._color_bounds[1],
                                  markersize = size, hovertext = hovertext,
                                  name = '{} {}'.format(name, entry.id), **kwargs)

            self._traces += plot

        if isinstance(objects[0], Particle) and scatter_start_points:
            self.scatter_start_points(objects)
        if isinstance(objects[0], Particle) and scatter_end_points:
            self.scatter_end_points(objects)
        if isinstance(objects[0], Interaction) and scatter_vertices:
            self.scatter_vertices(objects)

        return self._traces
