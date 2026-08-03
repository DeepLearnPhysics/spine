"""Sparse spatial embeddings used by the SPICE clustering model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import torch

from spine.constants.factory import enum_factory
from spine.data import IndexBatch, TensorBatch
from spine.model import sparse
from spine.model.cnn.act_norm import norm_factory
from spine.model.cnn.uresnet_layers import UResNetDecoder, UResNetEncoder

__all__ = ["SPICEEmbedder", "SPICEOutput"]


class SPICEOutput(TypedDict):
    """Batched outputs produced by :class:`SPICEEmbedder`."""

    embeddings: TensorBatch
    margins: TensorBatch
    seediness: TensorBatch
    filter_index: IndexBatch


class SPICEEmbedder(torch.nn.Module):
    """Produce spatial embeddings, cluster margins, and seediness scores.

    SPICE uses one sparse UResNet encoder and two independent decoders. The
    embedding branch predicts a coordinate offset and a positive cluster
    margin for each voxel. The seediness branch predicts how representative
    each voxel is of its target cluster.
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        skip_classes: Sequence[int | str] = (2, 3, 4),
        coord_conv: bool = True,
        margin_dim: int = 1,
        seediness_dim: int = 1,
        seed_freeze: bool = False,
    ) -> None:
        """Initialize the SPICE embedding network.

        Parameters
        ----------
        uresnet : dict
            Shared UResNet encoder and decoder configuration.
        skip_classes : sequence of int or str, default (2, 3, 4)
            Semantic classes excluded before embedding.
        coord_conv : bool, default True
            Append normalized coordinates to the input charge feature.
        margin_dim : int, default 1
            Number of predicted cluster-margin values per voxel.
        seediness_dim : int, default 1
            Number of predicted seediness values per voxel.
        seed_freeze : bool, default False
            Freeze the seediness decoder and output projection.

        Raises
        ------
        ValueError
            If the UResNet input contract or an output dimension is invalid.
        """
        # Initialize the parent class
        super().__init__()

        # Validate and store the SPICE-specific configuration
        self.dimension = int(uresnet.get("data_dim", 3))
        spatial_size = uresnet.get("spatial_size")
        if spatial_size is None:
            raise ValueError("SPICE requires `uresnet.spatial_size`.")
        if spatial_size < 1:
            raise ValueError("`uresnet.spatial_size` must be positive.")
        if margin_dim != 1:
            raise ValueError("SPICE currently supports exactly one cluster margin.")
        if seediness_dim != 1:
            raise ValueError("SPICE currently supports exactly one seediness score.")

        self.spatial_size = int(spatial_size)
        self.coord_conv = coord_conv
        self.margin_dim = margin_dim
        self.seediness_dim = seediness_dim
        self.seed_freeze = seed_freeze
        self.skip_classes = self._parse_shapes(skip_classes)

        # Check that the UResNet input width matches the frontend features
        expected_inputs = 1 + (self.dimension if coord_conv else 0)
        configured_inputs = int(uresnet.get("num_input", 1))
        if configured_inputs != expected_inputs:
            raise ValueError(
                "SPICE UResNet expects one charge feature plus normalized "
                f"coordinates when `coord_conv` is enabled: expected "
                f"`num_input={expected_inputs}`, got {configured_inputs}."
            )

        # Initialize the shared encoder and task-specific decoders
        self.encoder = UResNetEncoder(uresnet)
        self.embedding_decoder = UResNetDecoder(uresnet)
        self.seediness_decoder = UResNetDecoder(uresnet)
        num_filters = self.encoder.num_filters

        # Initialize the embedding, margin, and seediness projections
        self.embedding_output = torch.nn.Sequential(
            norm_factory(self.encoder.norm_cfg, num_filters),
            sparse.Linear(
                num_filters,
                self.dimension + self.margin_dim,
                bias=False,
            ),
        )
        self.seediness_output = torch.nn.Sequential(
            norm_factory(self.encoder.norm_cfg, num_filters),
            sparse.Linear(num_filters, self.seediness_dim, bias=False),
        )

        # Optionally freeze the complete seediness branch
        if self.seed_freeze:
            for parameter in self.seediness_decoder.parameters():
                parameter.requires_grad = False
            for parameter in self.seediness_output.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _parse_shapes(shapes: Sequence[int | str]) -> tuple[int, ...]:
        """Normalize semantic class identifiers to integer values."""
        # Resolve named and integer shape identifiers to one representation
        parsed = []
        for shape in shapes:
            parsed.append(
                enum_factory("shape", shape) if isinstance(shape, str) else int(shape)
            )
        return tuple(parsed)

    def filter_class(
        self,
        data: TensorBatch,
        seg_label: TensorBatch,
    ) -> tuple[TensorBatch, IndexBatch]:
        """Remove semantic classes excluded from SPICE clustering.

        Parameters
        ----------
        data : TensorBatch
            ``(N, 1 + D + F)`` sparse input table.
        seg_label : TensorBatch
            ``(N, 1 + D + 1)`` voxel-wise semantic labels.

        Returns
        -------
        tuple
            Filtered input data and an index back into the original batch.
        """
        # Validate that data and labels describe the same voxel rows
        data_tensor = data.torch_tensor()
        label_tensor = seg_label.values.torch_tensor()
        if len(data_tensor) != len(label_tensor):
            raise ValueError(
                "Input data and cluster labels must have the same length, got "
                f"{len(data_tensor)} and {len(label_tensor)}."
            )

        # Build the semantic-class selection mask
        mask = torch.ones(len(label_tensor), dtype=torch.bool, device=data.device)
        if self.skip_classes:
            excluded = torch.tensor(
                self.skip_classes,
                dtype=label_tensor.dtype,
                device=label_tensor.device,
            )
            mask = ~torch.isin(label_tensor, excluded)

        # Narrow the input while preserving event counts and source indexes
        index = torch.where(mask)[0]
        data_batch = data.select(mask)
        filter_index = IndexBatch(index, spans=data.counts, counts=data_batch.counts)
        return data_batch, filter_index

    def forward(
        self,
        data: TensorBatch,
        seg_label: TensorBatch,
    ) -> SPICEOutput:
        """Embed a batch of sparse voxels.

        Parameters
        ----------
        data : TensorBatch
            ``(N, 1 + D + F)`` sparse input table.
        seg_label : TensorBatch
            ``(N, 1 + D + 1)`` voxel-wise semantic labels used to exclude
            unsupported classes.

        Returns
        -------
        dict
            Spatial embeddings, positive margins, seediness scores, and the
            index relating filtered voxels to the original input.
        """
        # Remove semantic classes that SPICE does not cluster
        data, filter_index = self.filter_class(data, seg_label)
        data_tensor = data.torch_tensor()

        # Split coordinates and charge, then build coordinate-convolution inputs
        coordinates = data_tensor[:, : self.dimension + 1]
        raw_features = data_tensor[:, self.dimension + 1 :]
        if raw_features.shape[1] < 1:
            raise ValueError("SPICE requires at least one input feature per voxel.")
        charge = raw_features[:, :1]

        half_size = self.spatial_size / 2.0
        normalized_coordinates = (coordinates[:, 1:] - half_size) / half_size
        features = charge
        if self.coord_conv:
            features = torch.cat((normalized_coordinates, charge), dim=1)

        # Encode the sparse image once for both prediction branches
        sparse_input = sparse.SparseTensor(
            features=features,
            coordinates=coordinates.int(),
            batch_size=data.batch_size,
        )
        encoder_output = self.encoder(sparse_input)
        encoder_tensors = encoder_output["encoder_tensors"]
        final_tensor = encoder_output["final_tensor"]

        # Decode embedding and seediness features independently
        embedding_features = self.embedding_decoder(
            final_tensor,
            encoder_tensors,
        )[-1]
        seediness_features = self.seediness_decoder(
            final_tensor,
            encoder_tensors,
        )[-1]

        embedding_output = self.embedding_output(embedding_features).aligned_features()
        seediness_output = self.seediness_output(seediness_features).aligned_features()

        # Transform raw projections to their physical output domains
        spatial_offsets = torch.tanh(embedding_output[:, : self.dimension])
        embeddings = spatial_offsets + normalized_coordinates
        margins = 2.0 * torch.sigmoid(embedding_output[:, self.dimension :])
        seediness = torch.sigmoid(seediness_output)

        # Restore the filtered event batching on every dense output
        return {
            "embeddings": TensorBatch(embeddings, data.counts),
            "margins": TensorBatch(margins, data.counts),
            "seediness": TensorBatch(seediness, data.counts),
            "filter_index": filter_index,
        }
