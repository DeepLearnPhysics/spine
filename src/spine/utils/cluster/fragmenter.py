"""Classes to process the output of dense clustering algorithms into a set
of fragments."""

import numpy as np
import torch
import torch.nn as nn

from spine.utils.cluster.dense_cluster import fit_predict, gaussian_kernel_cuda
from spine.utils.gnn.cluster import form_clusters, get_cluster_label

__all__ = ["SPICEFragmentManager"]


class SPICEFragmentManager:
    """
    Full chain model fragment mananger for SPICE Clustering
    """

    def __init__(self, frag_cfg: dict, **kwargs):
        super(SPICEFragmentManager, self).__init__(frag_cfg, **kwargs)
        self._s_thresholds = frag_cfg.get("s_thresholds", [0.0, 0.0, 0.0, 0.0])
        self._p_thresholds = frag_cfg.get("p_thresholds", [0.5, 0.5, 0.5, 0.5])
        self._spice_classes = frag_cfg.get("cluster_classes", [])
        self._spice_min_voxels = frag_cfg.get("min_voxels", 2)

    def forward(self, input, cnn_result, semantic_labels=None):
        """
        Inputs:
            - input (torch.Tensor): N x 6 (coords, edep, semantic_labels)
            - cnn_result: dict of List[torch.Tensor], containing:
                - segmentation
                - embeddings
                - seediness
                - margins

        Returns:
            - fragments
            - frag_batch_ids
            - frag_seg

        """
        if self._use_segmentation_prediction:
            assert semantic_labels is None
            semantic_labels = torch.argmax(
                cnn_result["segmentation"][0], dim=1
            ).flatten()

        batch_labels = input[:, self._batch_column]
        fragments, frag_batch_ids = [], []
        for batch_id in batch_labels.unique():
            for s in self._spice_classes:
                mask = torch.nonzero(
                    (batch_labels == batch_id) & (semantic_labels == s), as_tuple=True
                )[0]
                if len(cnn_result["embeddings"][0][mask]) < self._spice_min_voxels:
                    continue

                pred_labels = fit_predict(
                    embeddings=cnn_result["embeddings"][0][mask],
                    seediness=cnn_result["seediness"][0][mask],
                    margins=cnn_result["margins"][0][mask],
                    fitfunc=gaussian_kernel_cuda,
                    s_threshold=self._s_thresholds[s],
                    p_threshold=self._p_thresholds[s],
                )

                for c in pred_labels.unique():
                    if c < 0:
                        continue
                    fragments.append(mask[pred_labels == c])
                    frag_batch_ids.append(int(batch_id))

        fragments_np = np.empty(len(fragments), dtype=object)
        fragments_np[:] = fragments
        frag_batch_ids = np.array(frag_batch_ids)
        frag_seg = np.empty(len(fragments), dtype=np.int32)
        for i, f in enumerate(fragments):
            vals, cnts = semantic_labels[f].unique(return_counts=True)
            assert len(vals) == 1
            frag_seg[i] = vals[torch.argmax(cnts)].item()

        return fragments_np, frag_batch_ids, frag_seg
