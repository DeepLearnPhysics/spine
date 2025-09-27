from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from spine.model.layer.factories import loss_fn_factory, metric_fn_factory
from spine.utils.weighting import get_class_weights

from .misc import *

__all__ = ["NodeEdgeLoss", "EdgeLoss"]


class GraphSPICEEmbeddingLoss(nn.Module):
    """
    Loss function for Sparse Spatial Embeddings Model, with fixed
    centroids and symmetric gaussian kernels.
    """

    name = "embedding"

    def __init__(self, cfg, name="graph_spice_loss"):
        super(GraphSPICEEmbeddingLoss, self).__init__()
        self.loss_config = cfg  # [name]
        self.batch_column = self.loss_config.get("batch_column", 0)

        self.ft_interloss = self.loss_config.get("ft_interloss_margin", 1.5)
        self.sp_interloss = self.loss_config.get("sp_interloss_margin", 0.2)

        self.ft_intraloss = self.loss_config.get("ft_intraloss_margin", 1.0)
        self.sp_intraloss = self.loss_config.get("sp_intraloss_margin", 0.0)

        self.eps = self.loss_config.get("eps", 0.001)

        self.ft_loss_params = self.loss_config.get(
            "ft_loss_params", dict(inter=1.0, intra=1.0, reg=0.1)
        )
        self.sp_loss_params = self.loss_config.get(
            "sp_loss_params", dict(inter=1.0, intra=1.0)
        )

        self.kernel_lossfn_name = self.loss_config.get("kernel_lossfn", "BCE")
        if self.kernel_lossfn_name == "BCE":
            self.kernel_lossfn = nn.BCEWithLogitsLoss(reduction="mean")
        elif self.kernel_lossfn_name == "lovasz_hinge":
            self.kernel_lossfn = LovaszHingeLoss(reduction="none")
        else:
            self.kernel_lossfn = nn.BCEWithLogitsLoss(reduction="none")

        self.seg_lossfn_name = self.loss_config.get("seg_lossfn", "CE")
        if self.seg_lossfn_name == "CE":
            self.seg_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        elif self.seg_lossfn_name == "lovasz_softmax":
            self.seg_loss_fn = LovaszSoftmaxWithLogitsLoss(reduction="mean")
        else:
            self.seg_loss_fn = nn.CrossEntropyLoss(reduction="mean")

    def feature_embedding_loss(self, ft_emb, groups, ft_centroids):
        """
        Compute discriminative feature embedding loss.

        INPUTS:
            - ft_emb (N x F)
            - groups (N)
            - ft_centroids (N_c X F)
        """
        intercluster_loss = inter_cluster_loss(ft_centroids, margin=self.ft_interloss)
        intracluster_loss = intra_cluster_loss(
            ft_emb, ft_centroids, groups, margin=self.ft_intraloss
        )
        reg_loss = torch.mean(torch.norm(ft_centroids, dim=1))
        out = {}
        out["intercluster_loss"] = float(intercluster_loss)
        out["intracluster_loss"] = float(intracluster_loss)
        out["regularization_loss"] = float(reg_loss)
        out["loss"] = (
            self.ft_loss_params["inter"] * intercluster_loss
            + self.ft_loss_params["intra"] * intracluster_loss
            + self.ft_loss_params["reg"] * reg_loss
        )
        return out

    def spatial_embedding_loss(self, sp_emb, groups, sp_centroids):
        """
        Compute spatial centroid regression loss.

        INPUTS:
            - sp_emb (N x D)
            - groups (N)
            - ft_centroids (N_c X F)
        """
        out = {}
        intercluster_loss = inter_cluster_loss(sp_centroids, margin=self.sp_interloss)
        intracluster_loss = intra_cluster_loss(
            sp_emb, sp_centroids, groups, margin=self.sp_intraloss
        )
        out["intercluster_loss"] = float(intercluster_loss)
        out["intracluster_loss"] = float(intracluster_loss)
        out["loss"] = (
            self.sp_loss_params["inter"] * intercluster_loss
            + self.sp_loss_params["intra"] * intracluster_loss
        )

        return out

    def covariance_loss(
        self, sp_emb, ft_emb, cov, groups, sp_centroids, ft_centroids, eps=0.001
    ):

        logits, acc, targets = get_graphspice_logits(
            sp_emb, ft_emb, cov, groups, sp_centroids, ft_centroids, eps
        )
        # Compute kernel score loss
        cov_loss = self.kernel_lossfn(logits, targets)
        return cov_loss, acc

    def occupancy_loss(self, occ, groups):
        """
        INPUTS:
            - occ (N x 1)
            - groups (N)
        """
        bincounts = torch.bincount(groups).float()
        bincounts[bincounts == 0] = 1
        occ_truth = torch.log(bincounts)
        occ_loss = torch.abs(
            torch.gather(occ - occ_truth[None, :], 1, groups.view(-1, 1))
        )
        if len(occ_loss.squeeze().size()) and len(groups.size()):
            occ_loss = scatter_mean(occ_loss.squeeze(), groups)
            # occ_loss = occ_loss[occ_loss > 0]
            return occ_loss.mean()
        else:
            # print(occ_loss.squeeze().size(), groups.size())
            return 0.0

    def combine_multiclass(
        self, sp_embeddings, ft_embeddings, covariance, occupancy, slabels, clabels
    ):
        """
        Wrapper function for combining different components of the loss,
        in particular when clustering must be done PER SEMANTIC CLASS.

        NOTE: When there are multiple semantic classes, we compute the DLoss
        by first masking out by each semantic segmentation (ground-truth/prediction)
        and then compute the clustering loss over each masked point cloud.

        INPUTS:
            features (torch.Tensor): pixel embeddings
            slabels (torch.Tensor): semantic labels
            clabels (torch.Tensor): group/instance/cluster labels

        OUTPUT:
            loss_segs (list): list of computed loss values for each semantic class.
            loss[i] = computed DLoss for semantic class <i>.
            acc_segs (list): list of computed clustering accuracy for each semantic class.
        """
        loss = defaultdict(list)
        loss["loss"] = []
        loss["ft_intra_loss"] = []
        loss["ft_inter_loss"] = []
        loss["ft_reg_loss"] = []
        loss["sp_intra_loss"] = []
        loss["sp_inter_loss"] = []
        loss["cov_loss"] = []
        loss["occ_loss"] = []

        accuracy = defaultdict(float)
        accuracy["accuracy"] = 0.0
        for i in range(5):
            accuracy[f"accuracy_{i}"] = 0.0

        semantic_classes = slabels.unique()
        counts = 0
        for sc in semantic_classes:
            if int(sc) == 4:
                continue
            index = slabels == sc
            sp_emb = sp_embeddings[index]
            ft_emb = ft_embeddings[index]
            cov = covariance[index]
            occ = occupancy[index]
            groups = clabels[index]

            # Remove groups labeled -1
            mask = groups != -1
            sp_emb = sp_emb[mask]
            ft_emb = ft_emb[mask]
            cov = cov[mask]
            occ = occ[mask]
            groups = groups[mask]

            groups_unique, _ = unique_label_torch(groups)
            # print(torch.unique(groups_unique, return_counts=True))
            if groups_unique.shape[0] < 2:
                continue
            sp_centroids = find_cluster_means(sp_emb, groups_unique)
            ft_centroids = find_cluster_means(ft_emb, groups_unique)
            # Get different loss components
            ft_out = self.feature_embedding_loss(ft_emb, groups_unique, ft_centroids)
            sp_out = self.spatial_embedding_loss(sp_emb, groups_unique, sp_centroids)
            cov_loss, acc = self.covariance_loss(
                sp_emb,
                ft_emb,
                cov,
                groups_unique,
                sp_centroids,
                ft_centroids,
                eps=self.eps,
            )
            occ_loss = self.occupancy_loss(occ, groups_unique)
            # TODO: Combine loss with weighting, keep track for logging
            loss["ft_intra_loss"].append(ft_out["intracluster_loss"])
            loss["ft_inter_loss"].append(ft_out["intercluster_loss"])
            loss["ft_reg_loss"].append(ft_out["regularization_loss"])
            loss["sp_intra_loss"].append(sp_out["intracluster_loss"])
            loss["sp_inter_loss"].append(sp_out["intercluster_loss"])
            loss["cov_loss"].append(float(cov_loss))
            loss["occ_loss"].append(float(occ_loss))
            loss["loss"].append(ft_out["loss"] + sp_out["loss"] + cov_loss + occ_loss)
            # TODO: Implement train-time accuracy estimation
            accuracy["accuracy_{}".format(int(sc))] = acc
            accuracy["accuracy"] += acc
            counts += 1

        if counts > 0:
            accuracy["accuracy"] /= counts
            for i in range(5):
                accuracy[f"accuracy_{i}"] /= counts

        return loss, accuracy

    def forward(self, out, segment_label, cluster_label):

        num_gpus = len(segment_label)
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        segmentationLayer = "segmentation" in out
        for i in range(num_gpus):
            slabels = segment_label[i][:, -1]
            # coords = segment_label[i][:, :3].float()
            # if torch.cuda.is_available():
            #    coords = coords.cuda()
            slabels = slabels.long()
            clabels = cluster_label[i][:, -1].long()
            # print(clabels)
            batch_idx = segment_label[i][:, self.batch_column]
            sp_embedding = out["spatial_embeddings"][i]
            ft_embedding = out["feature_embeddings"][i]
            covariance = out["covariance"][i]
            occupancy = out["occupancy"][i]
            if segmentationLayer:
                segmentation = out["segmentation"][i]
            # nbatch = batch_idx.unique().shape[0]

            for bidx in batch_idx.unique(sorted=True):
                batch_mask = batch_idx == bidx
                sp_embedding_batch = sp_embedding[batch_mask]
                ft_embedding_batch = ft_embedding[batch_mask]
                if segmentationLayer:
                    segmentation_batch = segmentation[batch_mask]
                slabels_batch = slabels[batch_mask]
                clabels_batch = clabels[batch_mask]
                covariance_batch = covariance[batch_mask]
                occupancy_batch = occupancy[batch_mask]

                if segmentationLayer:
                    loss_seg = self.seg_loss_fn(segmentation_batch, slabels_batch)
                    acc_seg = float(
                        torch.sum(
                            torch.argmax(segmentation_batch, dim=1) == slabels_batch
                        )
                    ) / float(segmentation_batch.shape[0])

                loss_class, acc_class = self.combine_multiclass(
                    sp_embedding_batch,
                    ft_embedding_batch,
                    covariance_batch,
                    occupancy_batch,
                    slabels_batch,
                    clabels_batch,
                )
                for key, val in loss_class.items():
                    loss[key].append(sum(val) / len(val) if len(val) else 0.0)
                for s, acc in acc_class.items():
                    accuracy[s].append(acc)

                if segmentationLayer:
                    loss["gs_loss_seg"].append(loss_seg)
                    accuracy["gs_acc_seg"].append(acc_seg)

        loss_avg = {}
        acc_avg = defaultdict(float)

        for key, val in loss.items():
            if sum(val) > 0:
                loss_avg[key] = sum(val) / len(val)
            else:
                loss_avg[key] = 0.0
        if segmentationLayer:
            loss_avg["loss"] += loss_avg["gs_loss_seg"]
        for key, val in accuracy.items():
            if sum(val) > 0:
                acc_avg[key] = sum(val) / len(val)
            else:
                acc_avg[key] = 1.0

        res = {}
        res.update(loss_avg)
        res.update(acc_avg)

        return res


class NodeEdgeLoss(torch.nn.modules.loss._Loss):
    """
    Combined Node + Edge Loss
    """

    name = "node_edge"

    def __init__(
        self,
        invert=True,
        edge_loss=None,
        use_cluster_labels=True,
        embedding_loss_weight=1.0,
        **kwargs,
    ):
        super(NodeEdgeHybridLoss, self).__init__()

        self.loss_fn = GraphSPICEEmbeddingLoss(kwargs)
        if edge_loss is None:
            edge_loss = {}
        self.edge_loss = edge_loss
        self.invert = invert
        self.edge_loss = WeightedEdgeLoss(invert=self.invert, **self.edge_loss)

        self.metric_fn = IoUScore()
        self.use_cluster_labels = use_cluster_labels
        self.embedding_loss_weight = embedding_loss_weight

    def forward(self, result, segment_label, cluster_label):

        group_label = [cluster_label[0][:, [0, 1, 2, 3, 5]]]

        res = self.loss_fn(result, segment_label, group_label)
        edge_score = result["gs_edge_attr"][0].squeeze()
        x = edge_score
        pred = x >= 0

        iou, edge_loss = 0, 0

        if self.use_cluster_labels:
            edge_truth = result["gs_edge_label"][0].squeeze()
            # print(edge_score.squeeze(), edge_truth, edge_score.shape, edge_truth.shape)
            edge_loss = self.edge_loss(edge_score, edge_truth.float())
            edge_loss = edge_loss.mean()

            if self.invert:
                edge_truth = torch.logical_not(edge_truth.long())

            iou = self.metric_fn(pred, edge_truth)
        # iou2 = self.metric_fn(~pred, ~edge_truth)

        res["edge_accuracy"] = iou
        if "loss" in res:
            res["loss"] = res["loss"] * self.embedding_loss_weight + edge_loss
        else:
            res["loss"] = edge_loss
        res["edge_loss"] = float(edge_loss)

        return res


class EdgeLoss(torch.nn.modules.loss._Loss):
    """Loss applied to edge scores produced by a GraphSPICE.

    This loss simply treats the edge score as logit predictions and compares
    them to edge labels (simple classification task).
    """

    name = "edge"

    def __init__(
        self,
        loss="log_dice",
        invert=True,
        balance_loss=False,
        equal_sampling=False,
        min_sample_edges=1000,
        metric="iou",
    ):
        """Initialize the loss function.

        Parameters
        ----------
        loss : str, default 'log_dice'
            Loss functional used to train edge scores
        invert : bool, default True
            If `True`, on edges are labeled as 0 and off edges as 1
        balance_loss : bool, default True
            Whether to weight the loss to account for class imbalance
        equal_sampling : bool, default False
            If `True`, sample the same number of edges from each label class
        min_sample_edges : int, default 1000
            If sampling evenly, minimum number of edges to sample with replacement
        """
        # Initialize the parent class
        super().__init__()

        # Store parameters
        self.invert = invert
        self.balance_loss = balance_loss
        self.equal_sampling = equal_sampling
        self.min_sample_edges = min_sample_edges

        # Intialize the loss function
        self.loss_fn = loss_fn_factory(loss, reduction="none")

        # Intitliaze the extra metric, if required
        self.metric_fn = None
        if metric is not None:
            self.metric_fn = metric_fn_factory(metric)

    def sample_edges(self, edge_score, edge_truth):
        """Subsample a set number of edges from each edge label class.

        Parameters
        ----------
        edge_scores : torch.Tensor
            Edge score predictions
        edge_label : torch.Tensor
            Edge labels

        Returns
        -------
        edge_scores : torch.Tensor
            Subsampled edge predictions
        edge_label : torch.Tensor
            Subsampled edge labels
        """
        with torch.no_grad():
            num_gaps, num_bridges = torch.bincount(edge_truth)
            n = max(num_gaps, self.min_sampled_edges)
            perm = torch.randperm(n).cuda()
            index = torch.arange(edge_truth.shape[0], dtype=torch.long).cuda()

        if num_gaps == 0:
            mask = index[perm]
            sampled_score, sampled_truth = edge_score[mask], edge_truth[mask]
        else:
            gap_indices = index[edge_truth == 0]
            brg_indices = index[edge_truth == 1]

            sampled_brgs = brg_indices[perm]
            index = torch.cat([gap_indices, brg_indices])

            sampled_score, sampled_truth = edge_score[index], edge_truth[index]

        return sampled_score, sampled_truth

    def forward(self, clust_label, edge_attr, edge_label, **kwargs):
        """Applies the edge loss to a batch of data.

        Parameters
        ----------
        clust_label : TensorBatch
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        edge_attr : TensorBatch
            (E) Edge scores
        edge_label : TensorBatch
            (E) Edge binary labels
        **kwargs : dict, optional
            Additional upstream model outputs not used in this loss

        Returns
        -------
        loss : torch.Tensor
            Value of the loss
        accuracy : float
            Value of the edge-wise classification accuracy
        iou : float
            IoU accuracy metric
        count : int
            Number of edges the loss was applied to
        """
        # Extract the raw tensors from the batches
        edge_attr = edge_attr.tensor.flatten()
        edge_pred = (edge_attr > 0.0).long()
        edge_label = edge_label.tensor

        # If requested, extract an equal number of samples from scores/labels
        if self.equal_sampling:
            edge_attr, edge_label = self.sample_edges(edge_attr, edge_label)

        # If requested, make off connections the positive label
        if self.invert:
            edge_label = torch.logical_not(edge_label).long()

        # Apply the loss function
        loss = self.loss_fn(edge_attr, edge_label)

        # If requested, apply the weights
        if self.balance_loss:
            weight = get_class_weights(edge_label, 2, per_class=False)
            loss *= weight

        loss = loss.mean()

        # Evaluate the accuracy
        num_edges = len(edge_pred)
        accuracy = 1.0
        if num_edges > 0:
            accuracy = (edge_pred == edge_label).sum() / num_edges

        metric = {}
        if self.metric_fn is not None:
            metric = {self.metric_fn.name: self.metric_fn(edge_pred, edge_label)}

        # Prepare and return the result dictionary
        return {"loss": loss, "accuracy": accuracy, **metric}
