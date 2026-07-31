"""GNN input graph construction."""

from .bipartite import BipartiteGraph
from .complete import CompleteGraph
from .delaunay import DelaunayGraph
from .knn import KNNGraph
from .loop import LoopGraph
from .mst import MSTGraph

__all__ = [
    "BipartiteGraph",
    "CompleteGraph",
    "DelaunayGraph",
    "KNNGraph",
    "LoopGraph",
    "MSTGraph",
]
