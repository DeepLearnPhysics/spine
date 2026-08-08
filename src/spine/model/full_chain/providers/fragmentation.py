"""Dense-clustering providers for the full reconstruction chain."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.dbscan import DBSCAN
from spine.model.graph_spice import GraphSPICE, GraphSPICELoss
from spine.utils.gnn.cluster import form_clusters_batch, get_cluster_label_batch

from ..registry import ProviderSpec, register_provider
from ..stage import ChainLossStage, ChainStage
from ..state import ChainState, StageResult


class FragmentationStage(ChainStage):
    """Construct particle fragments with labels, DBSCAN, or Graph-SPICE.

    Classical and learned fragmenters may own disjoint semantic shapes and
    publish one combined cluster list in the canonical voxel-index namespace.
    Native Graph-SPICE diagnostics remain namespaced public outputs.
    """

    requires = frozenset({"point_data", "seg_pred"})
    optional = frozenset({"clust_label", "coord_label"})
    provides = frozenset({"fragment_clusts", "fragment_shapes"})

    def __init__(
        self,
        name: str,
        mode: str,
        dbscan: DBSCAN | None,
        graph_spice: GraphSPICE | None,
    ) -> None:
        """Initialize the selected fragment implementations.

        Parameters
        ----------
        name : str
            Stage name.
        mode : {"dbscan", "graph_spice", "dbscan_graph_spice", "label"}
            Fragmentation implementation selection.
        dbscan : DBSCAN, optional
            Classical fragmenter.
        graph_spice : GraphSPICE, optional
            Learned fragmenter.
        """
        super().__init__(name)
        valid_modes = {"dbscan", "graph_spice", "dbscan_graph_spice", "label"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown fragmentation mode `{mode}`. Choose from "
                f"{sorted(valid_modes)}."
            )
        self.mode = mode
        self.dbscan = dbscan
        self.graph_spice = graph_spice

    @staticmethod
    def _empty(data: TensorBatch) -> tuple[IndexBatch, TensorBatch]:
        """Create empty fragment products using the input batch layout.

        Parameters
        ----------
        data : TensorBatch
            Canonical voxel tensor defining batch spans.

        Returns
        -------
        IndexBatch
            Empty cluster collection.
        TensorBatch
            Empty semantic-shape collection.
        """
        counts = np.zeros(data.batch_size, dtype=np.int64)
        clusts = IndexBatch([], data.counts, counts, [], default=np.empty(0, int))
        shapes = TensorBatch(np.empty(0, dtype=np.int64), counts)
        return clusts, shapes

    @staticmethod
    def _restore_graph_indexes(
        clusts: IndexBatch,
        filter_index: IndexBatch,
        spans: Any,
    ) -> IndexBatch:
        """Map Graph-SPICE indexes back to the canonical voxel set.

        Parameters
        ----------
        clusts : IndexBatch
            Clusters indexed into the Graph-SPICE filtered tensor.
        filter_index : IndexBatch
            Map from filtered rows to canonical voxel rows.
        spans : array-like
            Canonical event spans.

        Returns
        -------
        IndexBatch
            Clusters expressed in canonical voxel indexes.
        """
        index = filter_index.index
        restored = [index[cluster] for cluster in clusts.index_list]
        return IndexBatch(
            restored,
            spans=spans,
            counts=clusts.counts,
            single_counts=clusts.single_counts,
            default=index[:0],
        )

    def forward(self, state: ChainState) -> StageResult:
        """Build fragments and expose native Graph-SPICE diagnostics.

        Parameters
        ----------
        state : ChainState
            State containing voxel data, semantic predictions, and optional
            truth products.

        Returns
        -------
        StageResult
            Canonical fragment clusters/shapes and native learned outputs.
        """
        data: TensorBatch = state.require("point_data", self.name).data
        seg_pred: TensorBatch = state.require("seg_pred", self.name)
        clust_label: ClusterLabelBatch | None = state.get("clust_label")
        coord_label: TensorBatch | None = state.get("coord_label")
        fragments, fragment_shapes = self._empty(data)
        outputs: dict[str, Any] = {}

        # Append classical connected components when requested.
        if self.dbscan is not None:
            ppn_output = {
                key: value
                for key, value in state.outputs.items()
                if key.startswith("ppn_")
            }
            dbscan_clusts, dbscan_shapes = self.dbscan(
                data,
                seg_pred,
                coord_label=coord_label,
                **ppn_output,
            )
            fragments = fragments.merge(dbscan_clusts)
            fragment_shapes = fragment_shapes.merge(dbscan_shapes)

        # Graph-SPICE operates on a shape-filtered tensor. Restore its output
        # indexes before combining them with clusters from another provider.
        if self.graph_spice is not None:
            semantic_rows = TensorBatch(
                seg_pred.torch_tensor().reshape(-1, 1),
                data.counts,
            )
            graph_result = self.graph_spice(data, semantic_rows)
            outputs.update(
                {f"graph_spice_{key}": value for key, value in graph_result.items()}
            )
            if "clusts" in graph_result:
                graph_clusts = self._restore_graph_indexes(
                    graph_result["clusts"],
                    graph_result["filter_index"],
                    data.counts,
                )
                fragments = fragments.merge(graph_clusts.to_numpy())
                fragment_shapes = fragment_shapes.merge(graph_result["clust_shapes"])

        # Truth-defined fragmentation remains useful for isolated downstream
        # training and deterministic pipeline validation.
        if self.mode == "label":
            if clust_label is None:
                raise ValueError("Label fragmentation requires `clust_label`.")
            fragments = form_clusters_batch(clust_label.to_numpy(), column="cluster")
            fragment_shapes = get_cluster_label_batch(
                clust_label,
                fragments,
                column="shape",
            )

        products = {
            "fragment_clusts": fragments,
            "fragment_shapes": fragment_shapes,
        }
        outputs.update(products)
        return StageResult(products, outputs)


class GraphSPICELossStage(ChainLossStage):
    """Route namespaced Graph-SPICE outputs to its native objective.

    The adapter selects adapted cluster truth when available and reconstructs
    the semantic input expected by the standalone Graph-SPICE loss.
    """

    def __init__(self, name: str, loss: GraphSPICELoss) -> None:
        """Initialize the Graph-SPICE loss adapter.

        Parameters
        ----------
        name : str
            Stage name.
        loss : GraphSPICELoss
            Native edge objective.
        """
        super().__init__(name)
        self.loss = loss

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate Graph-SPICE supervision on adapted cluster labels.

        Parameters
        ----------
        data : dict
            Chain predictions and cluster truth.

        Returns
        -------
        dict
            Native Graph-SPICE loss metrics.
        """
        clust_label = data.get("clust_label_adapt", data.get("clust_label"))
        if clust_label is None:
            raise ValueError("Graph-SPICE loss requires `clust_label`.")
        seg_pred = data.get("seg_pred")
        if seg_pred is None:
            raise ValueError("Graph-SPICE loss requires `seg_pred`.")

        # Remove the public namespace before calling the standalone objective.
        output = {
            key.removeprefix("graph_spice_"): value
            for key, value in data.items()
            if key.startswith("graph_spice_")
        }
        semantic_rows = TensorBatch(
            seg_pred.torch_tensor().reshape(-1, 1),
            clust_label.counts,
        )
        return self.loss(
            seg_label=semantic_rows,
            clust_label=clust_label,
            **output,
        )


def build_fragmentation_stage(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainStage:
    """Build and validate configured fragment implementations.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Fragmentation mode and native model blocks.
    owner : torch.nn.Module
        Full-chain model that owns native trainable modules.

    Returns
    -------
    ChainStage
        Configured fragmentation adapter.
    """
    mode = config.get("mode")
    if not isinstance(mode, str):
        raise ValueError("Fragmentation requires a string `mode`.")

    dbscan = None
    graph_spice = None
    shapes: list[int] = []
    if "dbscan" in mode:
        dbscan_config = config.get("dbscan")
        if not isinstance(dbscan_config, dict):
            raise ValueError("DBSCAN fragmentation requires a `dbscan` block.")
        dbscan = DBSCAN(**dbscan_config)
        owner.add_module("dbscan", dbscan)
        shapes.extend(dbscan.shapes)

    if "graph_spice" in mode:
        graph_config = config.get("graph_spice")
        if not isinstance(graph_config, dict):
            raise ValueError(
                "Graph-SPICE fragmentation requires a `graph_spice` block."
            )
        graph_spice = GraphSPICE(graph_config)
        owner.add_module("graph_spice", graph_spice)
        shapes.extend(graph_spice.shapes)

    # Learned/classical fragmenters must partition all four supported shapes
    # exactly once. This catches both omissions and ambiguous duplicate owners.
    if mode != "label":
        unique, counts = np.unique(shapes, return_counts=True)
        expected = np.arange(4, dtype=unique.dtype)
        if not np.array_equal(unique, expected) or not np.all(counts == 1):
            raise ValueError(
                "DBSCAN and Graph-SPICE must collectively own each of the four "
                "fragment shapes exactly once."
            )

    return FragmentationStage(name, mode, dbscan, graph_spice)


def build_fragmentation_loss(
    name: str,
    config: dict[str, Any],
    owner: Any,
) -> ChainLossStage | None:
    """Build the Graph-SPICE loss when configured.

    Parameters
    ----------
    name : str
        Stage name.
    config : dict
        Resolved Graph-SPICE model and loss blocks.
    owner : torch.nn.Module
        Full-chain loss module that owns the native objective.

    Returns
    -------
    ChainLossStage or None
        Graph-SPICE loss adapter, or ``None`` without supervision.
    """
    model_config = config.get("graph_spice")
    loss_config = config.get("loss")
    if model_config is None or loss_config is None:
        return None
    if not isinstance(model_config, dict) or not isinstance(loss_config, dict):
        raise TypeError("Graph-SPICE model and loss blocks must be mappings.")
    loss = GraphSPICELoss(model_config, loss_config)
    owner.add_module("graph_spice_loss", loss)
    return GraphSPICELossStage(name, loss)


PROVIDER_SPEC = register_provider(
    ProviderSpec("fragmentation", build_fragmentation_stage, build_fragmentation_loss)
)
