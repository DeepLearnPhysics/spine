import torch
import torch.nn as nn

from spine.model.layer.cluster import loss_factory
from spine.model.layer.cluster.embeddings import (
    SPICE as SPICE_base,  # TODO why does this live out of this module?
)


class SPICE(SPICE_base):

    MODULES = [
        "network_base",
        "uresnet_encoder",
        "embedding_decoder",
        "seediness_decoder",
    ]

    def __init__(self, cfg):
        super(SPICE, self).__init__(cfg)


class SPICELoss(nn.Module):
    """
    Loss function for Proposal-Free Mask Generators.
    """

    def __init__(self, cfg, name="spice_loss"):
        super(SPICELoss, self).__init__()

        self.model_config = cfg.get("spice", {})
        self.skip_classes = self.model_config.get("skip_classes", [2, 3, 4])

        self.loss_config = cfg.get(name, {})
        self.loss_func_name = self.loss_config.get("name", "se_lovasz_inter")
        self.loss_func = loss_factory(self.loss_func_name)
        self.loss_func = self.loss_func(cfg)
        # print(self.loss_func)

    def class_mask(self, cluster_label):
        """
        Filter classes according to segmentation label.
        """
        mask = torch.ones(len(cluster_label), dtype=bool, device=cluster_label.device)
        for c in self.skip_classes:
            mask &= cluster_label[:, -1] != c

        return mask

    def forward(self, result, cluster_label):
        mask = self.class_mask(cluster_label[0])
        segment_label = [cluster_label[0][mask][:, [0, 1, 2, 3, -1]]]
        group_label = [cluster_label[0][mask][:, [0, 1, 2, 3, 5]]]
        return self.loss_func(result, segment_label, group_label)
