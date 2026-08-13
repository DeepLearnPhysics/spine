"""SPICE spatial-embedding clustering model and objective."""

from __future__ import annotations

from typing import Any

from spine.data import TensorBatch

from ..registry import ModelSpec
from .cluster import SPICEClusterer
from .embedder import SPICEClusterOutput, SPICEEmbedder
from .loss import SPICELoss as _SPICELoss

__all__ = ["SPICE", "SPICELoss"]


class SPICE(SPICEEmbedder):
    """Spatial embeddings with optional proposal-free fragment production."""

    def __init__(self, spice: dict[str, Any]) -> None:
        """Initialize the SPICE model.

        Parameters
        ----------
        spice : dict
            SPICE configuration containing the embedder parameters.
        """
        config = dict(spice)
        cluster_config = config.pop("clusterer", None)
        self.make_clusters = bool(config.pop("make_clusters", False))
        if self.make_clusters and cluster_config is None:
            cluster_config = {}
        if cluster_config is not None and not isinstance(cluster_config, dict):
            raise TypeError("`spice.clusterer` must be a mapping.")

        # The embedder owns shape filtering; the clusterer consumes precisely
        # that filtered row domain and therefore shares the same shape list.
        super().__init__(**config)
        self.clusterer = (
            SPICEClusterer(self.shapes, **cluster_config)
            if cluster_config is not None
            else None
        )

    def forward(self, data: TensorBatch, seg_label: TensorBatch) -> SPICEClusterOutput:
        """Predict embeddings and, when requested, fragment clusters."""
        result: SPICEClusterOutput = super().forward(data, seg_label)
        if self.clusterer is None or not self.make_clusters:
            return result

        index = result["filter_index"].index
        filtered_shapes = TensorBatch(
            seg_label.torch_tensor()[index],
            result["embeddings"].counts,
        )
        clusts, clust_shapes = self.clusterer(
            result["embeddings"],
            result["margins"],
            result["seediness"],
            filtered_shapes,
        )
        result["clusts"] = clusts
        result["clust_shapes"] = clust_shapes
        return result


class SPICELoss(_SPICELoss):
    """Top-level SPICE loss exposed through the model registry."""


MODEL_SPEC = ModelSpec("spice", SPICE, SPICELoss)
