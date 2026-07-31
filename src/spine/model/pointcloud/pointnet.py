"""PointNet++ feature encoder built on PyTorch Geometric."""

from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius

__all__ = ["PointNet", "PointNetEncoder"]

# From Pytorch Geometric Examples for PointNet:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py


class SAModule(torch.nn.Module):
    """Downsample points and aggregate features within a fixed radius."""

    def __init__(
        self,
        ratio: float,
        radius_value: float,
        local_network: torch.nn.Module,
    ) -> None:
        """Initialize a set-abstraction stage.

        Parameters
        ----------
        ratio : float
            Fraction of points retained by farthest-point sampling.
        radius_value : float
            Radius used to collect neighbors around sampled points.
        local_network : torch.nn.Module
            Network applied by the PointNet convolution.
        """
        super().__init__()
        self.ratio = ratio
        self.radius = radius_value
        self.conv = PointNetConv(local_network, add_self_loops=False)

    def forward(
        self,
        x: torch.Tensor | None,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply sampling and local feature aggregation.

        Parameters
        ----------
        x : torch.Tensor, optional
            Point features with shape ``(N, F)``.
        pos : torch.Tensor
            Point positions with shape ``(N, 3)``.
        batch : torch.Tensor
            Batch ID of each point.

        Returns
        -------
        tuple of torch.Tensor
            Aggregated features, sampled positions, and sampled batch IDs.
        """
        sampled_index = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos,
            pos[sampled_index],
            self.radius,
            batch,
            batch[sampled_index],
            max_num_neighbors=64,
        )
        edge_index = torch.stack([col, row], dim=0)
        destination_features = None if x is None else x[sampled_index]
        x = self.conv(
            (x, destination_features),
            (pos, pos[sampled_index]),
            edge_index,
        )
        pos = pos[sampled_index]
        batch = batch[sampled_index]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    """Aggregate all remaining points into one feature vector per entry."""

    def __init__(self, network: torch.nn.Module) -> None:
        """Initialize the global set-abstraction stage.

        Parameters
        ----------
        network : torch.nn.Module
            Network applied before global max pooling.
        """
        super().__init__()
        self.network = network

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool point features globally.

        Parameters
        ----------
        x : torch.Tensor
            Point features with shape ``(N, F)``.
        pos : torch.Tensor
            Point positions with shape ``(N, 3)``.
        batch : torch.Tensor
            Batch ID of each point.

        Returns
        -------
        tuple of torch.Tensor
            Global features, placeholder positions, and global batch IDs.
        """
        x = self.network(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet(torch.nn.Module):
    """Encode point clouds with hierarchical PointNet++ abstraction."""

    def __init__(
        self,
        cfg: dict[str, Any],
        name: str = "pointnet",
    ) -> None:
        """Initialize the PointNet++ encoder.

        Parameters
        ----------
        cfg : dict
            Configuration mapping containing the block selected by ``name``.
        name : str, default "pointnet"
            Name of the PointNet configuration block.

        Raises
        ------
        ValueError
            If the depth, sampling ratios, radii, or MLP specifications are
            inconsistent.
        """
        super().__init__()

        if name not in cfg:
            raise ValueError(f"PointNet configuration is missing `{name}`.")
        self.model_config = cfg[name]

        self.depth = self.model_config.get("depth", 2)
        if self.depth < 1:
            raise ValueError(f"`depth` must be positive, got {self.depth}.")

        sampling_ratio = self.model_config.get("sampling_ratio", 0.5)
        if isinstance(sampling_ratio, (int, float)):
            self.sampling_ratios = [float(sampling_ratio)] * self.depth
        elif isinstance(sampling_ratio, list):
            if len(sampling_ratio) != self.depth:
                raise ValueError("Expected one sampling ratio per PointNet depth.")
            self.sampling_ratios = [float(value) for value in sampling_ratio]
        else:
            raise ValueError("Sampling ratio must be a number or a list of numbers.")
        if any(not 0.0 < value <= 1.0 for value in self.sampling_ratios):
            raise ValueError("Sampling ratios must lie in `(0, 1]`.")

        neighbor_radius = self.model_config.get("neighbor_radius", 3.0)
        if isinstance(neighbor_radius, (int, float)):
            self.neighbor_radii = [float(neighbor_radius)] * self.depth
        elif isinstance(neighbor_radius, list):
            if len(neighbor_radius) != self.depth:
                raise ValueError("Expected one neighbor radius per PointNet depth.")
            self.neighbor_radii = [float(value) for value in neighbor_radius]
        else:
            raise ValueError("Neighbor radius must be a number or a list of numbers.")
        if any(value <= 0.0 for value in self.neighbor_radii):
            raise ValueError("Neighbor radii must be positive.")

        self.mlp_specs = []
        self.sa_modules = torch.nn.ModuleList()

        for depth_index in range(self.depth):
            spec_name = f"mlp_specs_{depth_index}"
            if spec_name not in self.model_config:
                raise ValueError(f"PointNet configuration is missing `{spec_name}`.")
            mlp_specs = self.model_config[spec_name]
            self.sa_modules.append(
                SAModule(
                    self.sampling_ratios[depth_index],
                    self.neighbor_radii[depth_index],
                    MLP(mlp_specs),
                )
            )
            self.mlp_specs.append(mlp_specs)

        self.mlp_specs_glob = self.model_config.get(
            "mlp_specs_glob", [256 + 3, 256, 512, 1024]
        )
        self.mlp_specs_final = self.model_config.get(
            "mlp_specs_final", [1024, 512, 256, 128]
        )
        self.dropout = self.model_config.get("dropout", 0.5)
        self.latent_size = self.mlp_specs_final[-1]

        self.sa3_module = GlobalSAModule(MLP(self.mlp_specs_glob))
        self.mlp = MLP(self.mlp_specs_final, dropout=self.dropout, norm=None)

    def forward(self, data: Data | Batch) -> torch.Tensor:
        """Encode a batched point cloud.

        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            Point-cloud features, positions, and batch assignments.

        Returns
        -------
        torch.Tensor
            One latent feature vector per point-cloud entry.
        """
        if data.batch is None:
            raise ValueError("PointNet input must define point batch IDs.")
        sa0_out = (data.x, data.pos, data.batch)

        out = sa0_out

        for abstraction_module in self.sa_modules:
            out = abstraction_module(*out)

        sa3_out = self.sa3_module(*out)
        x, _, _ = sa3_out

        return self.mlp(x)


class PointNetEncoder(torch.nn.Module):
    """Thin encoder wrapper exposing PointNet's latent feature size."""

    def __init__(
        self,
        cfg: dict[str, Any],
        model_name: str = "pointnet",
    ) -> None:
        """Initialize the PointNet encoder.

        Parameters
        ----------
        cfg : dict
            PointNet configuration mapping.
        model_name : str, default "pointnet"
            Name of the PointNet configuration block.
        """
        super().__init__()
        self.net = PointNet(cfg, name=model_name)
        self.latent_size = self.net.latent_size
        self.feature_size = self.latent_size

    def forward(self, batch: Batch) -> torch.Tensor:
        """Encode a PyTorch Geometric point-cloud batch.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batched point-cloud input.

        Returns
        -------
        torch.Tensor
            One latent feature vector per point-cloud entry.
        """
        return self.net(batch)
