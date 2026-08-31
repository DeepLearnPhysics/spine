"""Dense-clustering providers for the full reconstruction chain."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.cluster.formation import form_clusters_batch
from spine.cluster.label import get_cluster_label_batch
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch
from spine.model.common.dbscan import DBSCAN
from spine.model.graph_spice import GraphSPICE, GraphSPICELoss
from spine.model.spice import SPICE, SPICELoss

from ..registry import ProviderSpec, register_provider
from ..stage import ChainLossStage, ChainStage
from ..state import ChainState, StageResult


class FragmentationStage(ChainStage):
    """Construct particle fragments with labels, DBSCAN, SPICE, or Graph-SPICE.

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
        spice: SPICE | None,
        graph_spice: GraphSPICE | None,
    ) -> None:
        """Initialize the selected fragment implementations.

        Parameters
        ----------
        name : str
            Stage name.
        mode : str
            Fragmentation implementation selection.
        dbscan : DBSCAN, optional
            Classical fragmenter.
        spice : SPICE, optional
            Spatial-embedding fragmenter.
        graph_spice : GraphSPICE, optional
            Learned fragmenter.
        """
        super().__init__(name)
        valid_modes = {
            "dbscan",
            "spice",
            "graph_spice",
            "dbscan_spice",
            "dbscan_graph_spice",
            "spice_graph_spice",
            "dbscan_spice_graph_spice",
            "label",
        }
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown fragmentation mode `{mode}`. Choose from "
                f"{sorted(valid_modes)}."
            )
        self.mode = mode
        self.dbscan = dbscan
        self.spice = spice
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
    def _restore_filtered_indexes(
        clusts: IndexBatch,
        filter_index: IndexBatch,
        spans: Any,
    ) -> IndexBatch:
        """Map filtered model indexes back to the canonical voxel set.

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

        # SPICE directly clusters its spatial embeddings. As with Graph-SPICE,
        # its cluster indexes first refer to a shape-filtered tensor.
        if self.spice is not None:
            semantic_rows = TensorBatch(
                seg_pred.torch_tensor().reshape(-1, 1),
                data.counts,
            )
            spice_result = self.spice(data, semantic_rows)
            outputs.update(
                {f"spice_{key}": value for key, value in spice_result.items()}
            )
            spice_clusts = self._restore_filtered_indexes(
                spice_result["clusts"],
                spice_result["filter_index"],
                data.counts,
            )
            fragments = fragments.merge(spice_clusts.to_numpy())
            fragment_shapes = fragment_shapes.merge(spice_result["clust_shapes"])

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
                graph_clusts = self._restore_filtered_indexes(
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


class SPICELossStage(ChainLossStage):
    """Route namespaced SPICE embedding outputs to its native objective."""

    def __init__(self, name: str, loss: SPICELoss) -> None:
        """Initialize the SPICE loss adapter.

        Parameters
        ----------
        name : str
            Stage name.
        loss : SPICELoss
            Native SPICE objective.
        """
        super().__init__(name)
        self.loss = loss

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate SPICE supervision on adapted cluster labels."""
        clust_label = data.get("clust_label_adapt", data.get("clust_label"))
        if clust_label is None:
            raise ValueError("SPICE loss requires `clust_label`.")
        output = {
            key.removeprefix("spice_"): value
            for key, value in data.items()
            if key.startswith("spice_")
        }
        return self.loss(clust_label=clust_label, **output)


class FragmentationLossStage(ChainLossStage):
    """Combine independent learned-fragmentation objectives."""

    def __init__(self, name: str, stages: dict[str, ChainLossStage]) -> None:
        """Initialize named native loss adapters.

        Parameters
        ----------
        name : str
            Stage name.
        stages : dict[str, ChainLossStage]
            Learned fragmentation objectives keyed by implementation name.
        """
        super().__init__(name)
        self.stages = stages

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        """Sum objectives and retain implementation-namespaced diagnostics."""
        result: dict[str, Any] = {"loss": 0.0, "accuracy": 1.0, "num_losses": 0}
        for implementation, stage in self.stages.items():
            native = stage(data)
            count = int(native.get("num_losses", 1))
            if count < 1:
                raise ValueError(
                    f"Fragmentation loss `{implementation}` reported no objectives."
                )
            previous = result["num_losses"]
            result["loss"] = result["loss"] + native["loss"]
            result["accuracy"] = (
                result["accuracy"] * previous
                + float(native.get("accuracy", 1.0)) * count
            ) / (previous + count)
            result["num_losses"] += count
            for key, value in native.items():
                if key not in {"loss", "accuracy", "num_losses"}:
                    result[f"{implementation}_{key}"] = value
        return result


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
    spice = None
    graph_spice = None
    shapes: list[int] = []
    if "dbscan" in mode:
        dbscan_config = config.get("dbscan")
        if not isinstance(dbscan_config, dict):
            raise ValueError("DBSCAN fragmentation requires a `dbscan` block.")
        dbscan = DBSCAN(**dbscan_config)
        owner.add_module("dbscan", dbscan)
        shapes.extend(dbscan.shapes)

    if mode in {
        "spice",
        "dbscan_spice",
        "spice_graph_spice",
        "dbscan_spice_graph_spice",
    }:
        spice_config = config.get("spice")
        if not isinstance(spice_config, dict):
            raise ValueError("SPICE fragmentation requires a `spice` block.")
        spice_config = dict(spice_config)
        spice_config["make_clusters"] = True
        spice_config.setdefault("clusterer", {})
        spice = SPICE(spice_config)
        owner.add_module("spice", spice)
        shapes.extend(spice.shapes)

    if mode in {
        "graph_spice",
        "dbscan_graph_spice",
        "spice_graph_spice",
        "dbscan_spice_graph_spice",
    }:
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
                "DBSCAN, SPICE and Graph-SPICE must collectively own each of the four "
                "fragment shapes exactly once."
            )

    return FragmentationStage(name, mode, dbscan, spice, graph_spice)


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
    spice_config = config.get("spice")
    graph_config = config.get("graph_spice")
    loss_config = config.get("loss")
    if loss_config is None:
        return None
    if not isinstance(loss_config, dict):
        raise TypeError("Fragmentation loss blocks must be mappings.")
    stages: dict[str, ChainLossStage] = {}
    if spice_config is not None:
        if not isinstance(spice_config, dict):
            raise TypeError("SPICE model configuration must be a mapping.")
        native_config = loss_config.get("spice", loss_config)
        if not isinstance(native_config, dict):
            raise TypeError("SPICE loss configuration must be a mapping.")
        loss = SPICELoss(spice_config, native_config)
        owner.add_module("spice_loss", loss)
        stages["spice"] = SPICELossStage(name, loss)
    if graph_config is not None:
        if not isinstance(graph_config, dict):
            raise TypeError("Graph-SPICE model and loss blocks must be mappings.")
        native_config = loss_config.get("graph_spice", loss_config)
        if not isinstance(native_config, dict):
            raise TypeError("Graph-SPICE loss configuration must be a mapping.")
        loss = GraphSPICELoss(graph_config, native_config)
        owner.add_module("graph_spice_loss", loss)
        stages["graph_spice"] = GraphSPICELossStage(name, loss)
    if len(stages) == 1:
        return next(iter(stages.values()))
    if stages:
        return FragmentationLossStage(name, stages)
    return None


PROVIDER_SPEC = register_provider(
    ProviderSpec("fragmentation", build_fragmentation_stage, build_fragmentation_loss)
)
