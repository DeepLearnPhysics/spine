"""Vertex proposal decoder built on sparse UResNet feature planes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NoReturn, TypeAlias

import torch

from spine.data import TensorBatch, TensorSchema
from spine.model import sparse
from spine.model.cnn.act_norm import act_factory, norm_factory
from spine.model.cnn.blocks import ResNetBlock
from spine.model.cnn.configuration import setup_cnn_configuration

from .ppn import ExpandAs

VertexPPNOutput: TypeAlias = dict[str, TensorBatch | list[TensorBatch]]

__all__ = ["VertexPPN", "VertexPPNLoss", "VertexPPNOutput"]


class VertexPPN(sparse.Network):
    """Predict a vertex offset and vertex score at the input resolution.

    This decoder consumes the deepest UResNet tensor and its decoder feature
    planes. Intermediate masks softly gate features in the same way as the
    point-proposal network.
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        vertex_ppn: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the vertex decoder.

        Parameters
        ----------
        uresnet : dict
            Shared configuration of the UResNet feature backbone.
        vertex_ppn : dict, optional
            Vertex-head configuration. ``score_threshold`` controls the
            probability threshold retained for future hard masking modes.

        Raises
        ------
        ValueError
            If ``score_threshold`` is outside ``[0, 1]``.
        """
        # Initialize the sparse-network base and shared CNN configuration
        super().__init__(uresnet.get("data_dim", 3))
        setup_cnn_configuration(self, **uresnet)

        # Validate the vertex-specific masking configuration
        config = {} if vertex_ppn is None else vertex_ppn
        self.score_threshold = float(config.get("score_threshold", 0.5))
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("`score_threshold` must be between zero and one.")

        # Build the multiscale decoding, fusion, and score-prediction stages
        decoding_blocks = []
        decoding_convolutions = []
        self.vertex_pred = torch.nn.ModuleList()
        for level in range(self.depth - 2, -1, -1):
            decoding_convolutions.append(
                torch.nn.Sequential(
                    norm_factory(self.norm_cfg, self.num_planes[level + 1]),
                    act_factory(self.act_cfg),
                    sparse.ConvolutionTranspose(
                        in_channels=self.num_planes[level + 1],
                        out_channels=self.num_planes[level],
                        kernel_size=2,
                        stride=2,
                        dimension=self.dimension,
                        bias=self.allow_bias,
                    ),
                )
            )
            blocks = []
            for repetition in range(self.reps):
                blocks.append(
                    ResNetBlock(
                        self.num_planes[level] * (2 if repetition == 0 else 1),
                        self.num_planes[level],
                        dimension=self.dimension,
                        activation=self.act_cfg,
                        normalization=self.norm_cfg,
                        bias=self.allow_bias,
                    )
                )
            decoding_blocks.append(torch.nn.Sequential(*blocks))
            self.vertex_pred.append(sparse.Linear(self.num_planes[level], 2))

        # Register the scale-wise decoder and soft-mask expansion modules
        self.decoding_conv = torch.nn.Sequential(*decoding_convolutions)
        self.decoding_block = torch.nn.Sequential(*decoding_blocks)
        self.expand_as = ExpandAs()

        # Initialize the final full-resolution regression and score heads
        num_output = self.num_planes[0]
        self.final_block = ResNetBlock(
            num_output,
            num_output,
            dimension=self.dimension,
            activation=self.act_cfg,
            normalization=self.norm_cfg,
            bias=self.allow_bias,
        )
        self.vertex_regression = sparse.Convolution(
            num_output,
            self.dimension,
            kernel_size=3,
            stride=1,
            dimension=self.dimension,
            bias=self.allow_bias,
        )
        self.vertexness_score = sparse.Convolution(
            num_output,
            2,
            kernel_size=3,
            stride=1,
            dimension=self.dimension,
            bias=self.allow_bias,
        )

    def forward(
        self,
        final_tensor: sparse.SparseTensor,
        decoder_tensors: Sequence[sparse.SparseTensor],
    ) -> VertexPPNOutput:
        """Predict vertex quantities from a UResNet feature pyramid.

        Parameters
        ----------
        final_tensor : sparse.SparseTensor
            Deepest UResNet encoder representation.
        decoder_tensors : sequence of sparse.SparseTensor
            UResNet decoder feature planes ordered from deep to shallow.

        Returns
        -------
        VertexPPNOutput
            Per-site vertex offsets and logits, intermediate score layers,
            coordinates and row-aligned predictions.

        Raises
        ------
        ValueError
            If the decoder feature count does not match ``depth - 1``.
        """
        # Validate the supplied UResNet feature pyramid
        expected = self.depth - 1
        if len(decoder_tensors) != expected:
            raise ValueError(
                f"Expected {expected} decoder tensors, got " f"{len(decoder_tensors)}."
            )

        # Decode every scale while collecting intermediate score products
        vertex_layers = []
        vertex_coords = []
        x = final_tensor
        modules = zip(
            self.decoding_conv,
            self.decoding_block,
            self.vertex_pred,
            strict=True,
        )
        for index, (upsample, block, predictor) in enumerate(modules):
            x = upsample(x)
            x = sparse.cat(decoder_tensors[index], x)
            x = block(x)
            scores = predictor(x)
            probabilities = sparse.softmax(scores, dim=1)
            counts = x.counts
            vertex_layers.append(TensorBatch(scores.features, counts))
            vertex_coords.append(
                TensorBatch(
                    scores.coordinates,
                    counts,
                    has_batch_col=True,
                    coord_cols=tuple(range(1, self.dimension + 1)),
                    schema=TensorSchema(
                        coordinate_groups={"points": tuple(range(self.dimension))}
                    ),
                )
            )
            expanded = self.expand_as(
                probabilities,
                x.features.shape,
                use_binary_mask=False,
                score_threshold=self.score_threshold,
            )
            x = x * expanded.detach()

        # Predict the final full-resolution vertex offsets and logits
        x = self.final_block(x)
        offsets = self.vertex_regression(x)
        scores = self.vertexness_score(x)
        points = x.replace_features(
            torch.cat((offsets.features, scores.features), dim=1)
        )

        # Return both input-aligned and unique sparse-site representations
        return {
            "vertex_points": points.to_tensor_batch(
                include_coordinates=False,
                restore=True,
            ),
            "vertex_points_unique": points.to_tensor_batch(
                include_coordinates=False,
            ),
            "vertex_layers": vertex_layers,
            "vertex_coords": vertex_coords,
            "vertex_output_coordinates": TensorBatch(
                x.coordinates,
                x.counts,
                has_batch_col=True,
                coord_cols=tuple(range(1, self.dimension + 1)),
                schema=TensorSchema(
                    coordinate_groups={"points": tuple(range(self.dimension))}
                ),
            ),
        }


class VertexPPNLoss(torch.nn.Module):
    """Represent the currently unsupported vertex proposal loss.

    The historical implementation never completed its heatmap construction
    and depended on removed backend-specific coordinate maps. A new loss must
    define its accepted vertex-label schema before training can be supported.
    """

    def __init__(self, **_: object) -> None:
        """Initialize the unsupported loss component.

        Other Parameters
        ----------------
        **_ : object
            Reserved configuration values, accepted so construction can fail
            at the point where training is attempted.
        """
        super().__init__()

    def forward(self, **_: object) -> NoReturn:
        """Reject training until a vertex-label contract is implemented.

        Other Parameters
        ----------------
        **_ : object
            Model outputs and labels that would be consumed by a future loss.

        Raises
        ------
        NotImplementedError
            Always raised because no supported vertex target schema exists.
        """
        raise NotImplementedError(
            "VertexPPNLoss is not implemented. Define a vertex-label schema "
            "and heatmap target contract before enabling VertexPPN training."
        )
