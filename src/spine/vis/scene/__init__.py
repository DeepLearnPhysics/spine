"""Renderer-neutral scenes and configurable rendering backends."""

from .adapter import plotly_trace_to_layer
from .backend import get_backend, register_backend, render_scene
from .model import (
    BoxLayer,
    LineLayer,
    LineStyle,
    MarkerLayer,
    MeshLayer,
    MeshStyle,
    PointLayer,
    PointStyle,
    Scene,
    SceneView,
    VectorLayer,
)
from .plotly import PlotlyBackend

__all__ = [
    "BoxLayer",
    "LineLayer",
    "LineStyle",
    "MarkerLayer",
    "MeshLayer",
    "MeshStyle",
    "PointLayer",
    "PointStyle",
    "PlotlyBackend",
    "Scene",
    "SceneView",
    "VectorLayer",
    "get_backend",
    "register_backend",
    "render_scene",
    "plotly_trace_to_layer",
]
