from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from spine.constants import BATCH_COL, INTER_COL, NU_COL, VTX_COLS
from spine.model import sparse
from spine.model.layer.cnn.vertex_ppn import VertexPPN, VertexPPNLoss
from spine.model.layer.pointcloud import PointNetEncoder
from spine.model.uresnet import SegmentationLoss, UResNetSegmentation
from spine.utils.gnn.cluster import form_clusters, get_cluster_label


class VertexPPNChain(torch.nn.Module):
    """
    Experimental model for PPN-like vertex prediction
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super(VertexPPNChain, self).__init__()
        self.model_config = cfg
        self.backbone = UResNetSegmentation(cfg)
        self.vertex_ppn = VertexPPN(cfg)
        self.num_classes = self.backbone.num_classes
        self.num_filters = self.backbone.F
        self.segmentation = sparse.Linear(self.num_filters, self.num_classes)

    def forward(self, input: Any) -> dict[str, Any]:

        primary_labels = None
        if self.training:
            if len(input) != 2:
                raise ValueError("Expected `len(input) == 2`.")
            primary_labels = input[1][:, -2]
            segment_labels = input[1][:, -1]

        input_tensors = [input[0][:, :5]]

        out = defaultdict(list)

        for igpu, x in enumerate(input_tensors):
            # input_data = x[:, :5]
            res = self.backbone([x])
            input_sparse_tensor = res["encoderTensors"][0][0]
            segmentation = self.segmentation(res["decoderTensors"][igpu][-1])
            res_vertex = self.vertex_ppn(
                res["finalTensor"][igpu],
                res["decoderTensors"][igpu],
                input_sparse_tensor=input_sparse_tensor,
                primary_labels=primary_labels,
                segment_labels=segment_labels,
            )
            out["segmentation"].append(segmentation.F)
            out.update(res_vertex)
        return out


class UResNetVertexLoss(torch.nn.Module):
    """
    See Also
    --------
    spine.model.uresnet.SegmentationLoss, spine.model.layer.common.ppn.PPNLonelyLoss
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super(UResNetVertexLoss, self).__init__()
        self.vertex_loss = VertexPPNLoss(cfg)
        self.segmentation_loss = SegmentationLoss(cfg)

    def forward(
        self,
        outputs: dict[str, Any],
        kinematics_label: Any,
    ) -> dict[str, Any]:

        res_segmentation = self.segmentation_loss(outputs, kinematics_label)

        res_vertex = self.vertex_loss(outputs, kinematics_label)

        res = {
            "loss": res_segmentation["loss"] + res_vertex["vertex_loss"],
            "accuracy": (res_segmentation["accuracy"] + res_vertex["vertex_acc"]) / 2.0,
            "reg_loss": res_vertex["vertex_reg_loss"],
        }
        return res


class VertexPointNet(torch.nn.Module):

    def __init__(
        self,
        cfg: dict[str, Any],
        name: str = "vertex_pointnet",
    ) -> None:
        super(VertexPointNet, self).__init__()
        self.encoder = PointNetEncoder(cfg)
        self.D = cfg[name].get("D", 3)
        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.latent_size, self.D), torch.nn.Softplus()
        )

    def split_input(self, point_cloud: Any, clusts: Any | None = None) -> Any:
        point_cloud_cpu = point_cloud.detach().cpu().numpy()
        batches, bcounts = np.unique(point_cloud_cpu[:, BATCH_COL], return_counts=True)
        if clusts is None:
            clusts = form_clusters(point_cloud_cpu, column=INTER_COL)
        if len(clusts) == 0:
            return Batch()

        data_list = []
        for i, c in enumerate(clusts):
            x = point_cloud[c, 4].view(-1, 1)
            pos = point_cloud[c, 1:4]
            data = Data(x=x, pos=pos)
            data_list.append(data)

        split_data = Batch.from_data_list(data_list)
        return split_data, clusts

    def forward(
        self,
        input: Any,
        clusts: Any | None = None,
    ) -> dict[str, Any]:
        res = {}
        (point_cloud,) = input
        batch, clusts = self.split_input(point_cloud, clusts)

        interactions = torch.unique(batch.batch)
        centroids = torch.vstack(
            [batch.pos[batch.batch == b].mean(dim=0) for b in interactions]
        )

        out = self.encoder(batch)
        out = self.final_layer(out)
        res["clusts"] = [clusts]
        res["vertex_pred"] = [centroids + out]
        return res


class VertexPointNetLoss(torch.nn.Module):

    def __init__(
        self,
        cfg: dict[str, Any],
        name: str = "vertex_pointnet_loss",
    ) -> None:
        super(VertexPointNetLoss, self).__init__()
        self.spatial_size = cfg[name].get("spatial_size", 6144)
        self.loss_fn = torch.nn.MSELoss(reduction="none")

    def forward(
        self,
        res: dict[str, Any],
        cluster_label: Any,
    ) -> dict[str, Any]:

        clusts = res["clusts"][0]
        vertex_pred = res["vertex_pred"][0]

        device = cluster_label[0].device

        vtx_x = get_cluster_label(cluster_label[0], clusts, column=VTX_COLS[0])
        vtx_y = get_cluster_label(cluster_label[0], clusts, column=VTX_COLS[1])
        vtx_z = get_cluster_label(cluster_label[0], clusts, column=VTX_COLS[2])

        nu_label = get_cluster_label(cluster_label[0], clusts, column=NU_COL)
        nu_mask = torch.Tensor(nu_label == 1).bool().to(device)

        vtx_label = torch.cat(
            [
                torch.Tensor(vtx_x.reshape(-1, 1)).to(device),
                torch.Tensor(vtx_y.reshape(-1, 1)).to(device),
                torch.Tensor(vtx_z.reshape(-1, 1)).to(device),
            ],
            dim=1,
        )

        mask = (
            nu_mask
            & (vtx_label >= 0).all(dim=1)
            & (vtx_label < self.spatial_size).all(dim=1)
        )
        loss = self.loss_fn(vertex_pred[mask], vtx_label[mask]).sum(dim=1).mean()

        result = {"loss": loss, "accuracy": loss}

        return result
