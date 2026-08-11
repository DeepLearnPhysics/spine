"""Renderer-neutral scenes and configurable rendering backends."""

from .backend import get_backend, register_backend, render_scene
from .model import PointLayer, PointStyle, Scene, SceneView
from .plotly import PlotlyBackend

__all__ = [
    "PointLayer",
    "PointStyle",
    "PlotlyBackend",
    "Scene",
    "SceneView",
    "get_backend",
    "register_backend",
    "render_scene",
]
