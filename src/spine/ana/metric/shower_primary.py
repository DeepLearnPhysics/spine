"""Node-aligned shower-primary reporting with training-compatible targets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from spine.ana.base import AnaBase
from spine.cluster.primary import shower_primary_target
from spine.constants import DELTA_SHP, MICHL_SHP, SHOWR_SHP
from spine.data import IndexBatch, TensorBatch
from spine.io.collate import CollateAll
from spine.model.common.quality import ClusterQualityFilter

__all__ = ["ShowerPrimaryAna"]


class ShowerPrimaryAna(AnaBase):
    """Export one row per reconstructed shower node, without object matching.

    Configure ``use_closest``, ``high_purity``, ``use_group_pred`` and overlap
    thresholds identically to the shower-primary objective. Targets and masks
    use the same shared kernels as that objective. ``prediction`` is the raw
    node-logit argmax used by training, rather than a downstream primary choice.
    """

    name = "shower_primary_eval"

    def __init__(
        self,
        use_closest: bool = False,
        high_purity: bool = False,
        use_group_pred: bool = False,
        min_iou: float | Sequence[float] | None = None,
        min_purity: float | Sequence[float] | None = None,
        min_efficiency: float | Sequence[float] | None = None,
        match_target: str = "group",
        label_key: str = "clust_label_adapt",
        shapes: Sequence[int] = (SHOWR_SHP, MICHL_SHP, DELTA_SHP),
        **kwargs: Any,
    ) -> None:
        """Initialize target policy; options mirror ``shower_primary`` loss.

        ``label_key`` must reference structured truth on the same reconstructed
        point domain as ``fragment_clusts``.

        Parameters
        ----------
        use_closest : bool, default False
            Among fragments of the true primary particle, select the one
            closest to the shower start. Requires ``coord_label``.
        high_purity : bool, default False
            Accept only groups containing exactly one primary target.
        use_group_pred : bool, default False
            Apply purity cuts to predicted groups rather than truth groups.
            Only used when ``high_purity`` is enabled.
        min_iou : float or sequence of float, optional
            Minimum truth-instance overlap IoU, shared or specified separately
            for secondary and primary target classes.
        min_purity : float or sequence of float, optional
            Minimum fragment purity under the shared overlap-quality policy.
        min_efficiency : float or sequence of float, optional
            Minimum fragment efficiency under the shared overlap-quality policy.
        match_target : str, default 'group'
            Truth-instance field used for overlap-quality matching.
        label_key : str, default 'clust_label_adapt'
            Structured truth-label product on the reconstructed point domain.
        shapes : sequence of int, default (0, 2, 3)
            Semantic IDs modeled by the shower network: shower, Michel and
            delta by default. Unmodeled logit sentinels are always excluded.
        **kwargs : dict, optional
            Additional options forwarded to ``AnaBase``, including log paths,
            output prefixes and CSV buffering.
        """
        super().__init__(**kwargs)
        self.use_closest = use_closest
        self.high_purity = high_purity
        self.use_group_pred = use_group_pred
        self.label_key = label_key
        self.shapes = tuple(shapes)
        self.quality = ClusterQualityFilter(
            min_iou,
            min_purity,
            min_efficiency,
            match_target=match_target,
            num_classes=2,
        )
        keys = {
            label_key: True,
            "fragment_clusts": True,
            "fragment_shapes": True,
            "fragment_node_pred": True,
            "coord_label": use_closest,
            "fragment_group_pred": high_purity and use_group_pred,
        }
        self.update_keys(keys)
        self.initialize_writer("nodes")

    def process(self, data: Mapping[str, Any]) -> None:
        """Write explicit target, prediction and validity for each shower node."""
        logits = np.asarray(data["fragment_node_pred"])
        # Unmodeled fragments carry -inf sentinels in the merged full-chain
        # products. They must not influence closest targets or group purity.
        node_ids = np.flatnonzero(
            np.isin(data["fragment_shapes"], self.shapes)
            & np.isfinite(logits).all(axis=1)
        )
        if not len(node_ids):
            return
        # Rebatch one unwrapped event without losing its structured schemas.
        sample = {"labels": data[self.label_key]}
        if self.use_closest:
            sample["coordinates"] = data["coord_label"]
        batch = CollateAll(data_keys=tuple(sample))([sample])
        labels = batch["labels"]
        clusters = IndexBatch(
            [data["fragment_clusts"][i] for i in node_ids],
            spans=labels.data.counts,
            counts=[len(node_ids)],
            single_counts=[len(data["fragment_clusts"][i]) for i in node_ids],
        )
        groups = None
        if self.high_purity and self.use_group_pred:
            groups = TensorBatch(
                np.asarray(data["fragment_group_pred"])[node_ids], [len(node_ids)]
            )
        target, valid = shower_primary_target(
            labels,
            clusters,
            batch.get("coordinates"),
            use_closest=self.use_closest,
            high_purity=self.high_purity,
            groups=groups,
        )
        if self.quality.active:
            valid &= self.quality.node_mask(labels, clusters, target.data)
        predictions = logits[node_ids].argmax(axis=1)
        for i, node_id in enumerate(node_ids):
            self.append(
                "nodes",
                fragment_id=int(node_id),
                target=int(target.data[i]),
                prediction=int(predictions[i]),
                valid=bool(valid[i]),
                shape=int(data["fragment_shapes"][node_id]),
                target_definition="closest" if self.use_closest else "group_primary",
            )
