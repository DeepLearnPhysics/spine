"""Shared cluster formation, measurement and graph operations.

These routines understand SPINE batch products and reconstruction clusters;
array-only numerical primitives remain under :mod:`spine.math`.
"""

# Explicit public re-exports intentionally mirror submodule ``__all__`` lists.
# pylint: disable=duplicate-code

from . import direction, features, formation, graph, label, topology
from .direction import (
    cluster_dedx,
    cluster_dedx_dir,
    cluster_direction,
    get_cluster_dedxs,
    get_cluster_dedxs_batch,
    get_cluster_directions,
    get_cluster_directions_batch,
)
from .features import (
    get_cluster_centers,
    get_cluster_energies,
    get_cluster_features,
    get_cluster_features_base,
    get_cluster_features_batch,
    get_cluster_features_extended,
    get_cluster_sizes,
)
from .formation import break_clusters, form_clusters, form_clusters_batch
from .label import (
    get_cluster_closest_label_batch,
    get_cluster_closest_primary_label_batch,
    get_cluster_label,
    get_cluster_label_batch,
    get_cluster_points_label,
    get_cluster_points_label_batch,
    get_cluster_primary_label_batch,
)

__all__ = [
    "break_clusters",
    "cluster_dedx",
    "cluster_dedx_dir",
    "cluster_direction",
    "direction",
    "features",
    "formation",
    "form_clusters",
    "form_clusters_batch",
    "get_cluster_centers",
    "get_cluster_closest_label_batch",
    "get_cluster_closest_primary_label_batch",
    "get_cluster_dedxs",
    "get_cluster_dedxs_batch",
    "get_cluster_directions",
    "get_cluster_directions_batch",
    "get_cluster_energies",
    "get_cluster_features",
    "get_cluster_features_base",
    "get_cluster_features_batch",
    "get_cluster_features_extended",
    "get_cluster_label",
    "get_cluster_label_batch",
    "get_cluster_points_label",
    "get_cluster_points_label_batch",
    "get_cluster_primary_label_batch",
    "get_cluster_sizes",
    "graph",
    "label",
    "topology",
]
