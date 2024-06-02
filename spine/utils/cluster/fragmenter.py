"""Classes to process the output of dense clustering algorithms into a set
of fragments."""

from spine.utils.gnn.cluster import form_clusters, get_cluster_label

import torch
import torch.nn as nn
import numpy as np

from spine.utils.cluster.dense_cluster import fit_predict, gaussian_kernel_cuda

__all__ = ['SPICEFragmentManager', 'GraphSPICEFragmentManager']


class SPICEFragmentManager:
    """
    Full chain model fragment mananger for SPICE Clustering
    """
    def __init__(self, frag_cfg : dict, **kwargs):
        super(SPICEFragmentManager, self).__init__(frag_cfg, **kwargs)
        self._s_thresholds     = frag_cfg.get('s_thresholds'   , [0.0, 0.0, 0.0, 0.0])
        self._p_thresholds     = frag_cfg.get('p_thresholds'   , [0.5, 0.5, 0.5, 0.5])
        self._spice_classes    = frag_cfg.get('cluster_classes', []                  )
        self._spice_min_voxels = frag_cfg.get('min_voxels'     , 2                   )

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
            semantic_labels = torch.argmax(cnn_result['segmentation'][0],
                                           dim=1).flatten()

        batch_labels = input[:, self._batch_column]
        fragments, frag_batch_ids = [], []
        for batch_id in batch_labels.unique():
            for s in self._spice_classes:
                mask = torch.nonzero((batch_labels == batch_id) &
                                     (semantic_labels == s), as_tuple=True)[0]
                if len(cnn_result['embeddings'][0][mask]) < self._spice_min_voxels:
                    continue

                pred_labels = fit_predict(embeddings  = cnn_result['embeddings'][0][mask],
                                          seediness   = cnn_result['seediness'][0][mask],
                                          margins     = cnn_result['margins'][0][mask],
                                          fitfunc     = gaussian_kernel_cuda,
                                          s_threshold = self._s_thresholds[s],
                                          p_threshold = self._p_thresholds[s])

                for c in pred_labels.unique():
                    if c < 0:
                        continue
                    fragments.append(mask[pred_labels == c])
                    frag_batch_ids.append(int(batch_id))

        fragments_np    = np.empty(len(fragments), dtype=object)
        fragments_np[:] = fragments
        frag_batch_ids = np.array(frag_batch_ids)
        frag_seg = np.empty(len(fragments), dtype=np.int32)
        for i, f in enumerate(fragments):
            vals, cnts = semantic_labels[f].unique(return_counts=True)
            assert len(vals) == 1
            frag_seg[i] = vals[torch.argmax(cnts)].item()

        return fragments_np, frag_batch_ids, frag_seg
            

class GraphSPICEFragmentManager:
    """Builds fragments from edge Graph-SPICE edge predictions."""

    def __init__(self, frag_cfg : dict, **kwargs):
        super(GraphSPICEFragmentManager, self).__init__(frag_cfg, **kwargs)


    def process(self, filtered_input, n, filtered_semantic, offset=0):
        
        fragments = form_clusters(filtered_input, column=-1)
        fragments = [f.int().detach().cpu().numpy() for f in fragments]

        if len(fragments) > 0:
            frag_batch_ids = get_cluster_batch(filtered_input.detach().cpu().numpy(),\
                                            fragments)
            fragments_seg = get_cluster_label(filtered_input, fragments, column=-2)
            fragments_id = get_cluster_label(filtered_input, fragments, column=-1)
        else:
            frag_batch_ids = np.empty((0,))
            fragments_seg = np.empty((0,))
            fragments_id = np.empty((0,))
        
        fragments = [np.arange(n)[filtered_semantic.detach().cpu().numpy()][clust]+offset \
                     for clust in fragments]

        return fragments, frag_batch_ids, fragments_seg, fragments_id

    def forward(self, filtered_input, original_input, filtered_semantic):
        """
        Inputs:
            - input (torch.Tensor): N x 6 (coords, edep, semantic_labels)
                for GraphSPICE, we skip clustering for some labels
                (namely michel, delta, and low E)
            - cnn_result: dict of List[torch.Tensor], containing:
                - segmentation
                - graph
                - graph_info
            - semantic_labels:
            - gs_manager: ClusterGraphManager object for GraphSPICE handling

        Returns:
            - fragments
            - frag_batch_ids
            - frag_seg

        """
        all_fragments, all_frag_batch_ids, all_fragments_seg = [], [], []
        all_fragments_id = []
        for b in filtered_input[:, self._batch_column].unique():
            mask = filtered_input[:, self._batch_column] == b
            original_mask = original_input[:, self._batch_column] == b
        
            # How many voxels belong to that batch
            n = torch.count_nonzero(original_mask)
            # The index start of the batch in original data
            # - note: we cannot simply accumulate the values
            # of n, as this will fail if a batch is missing
            # from the original data (eg no track in that batch).
            offset = torch.nonzero(original_mask).min().item()
            
            fragments, frag_batch_ids, fragments_seg, fragments_id = self.process(filtered_input[mask], 
                                                                    n.item(), 
                                                                    filtered_semantic[original_mask].cpu(),
                                                                    offset=offset)
            
            all_fragments.extend(fragments)
            all_frag_batch_ids.extend(frag_batch_ids)
            all_fragments_seg.extend(fragments_seg)
            all_fragments_id.extend(fragments_id)
        return all_fragments, all_frag_batch_ids, all_fragments_seg, all_fragments_id
