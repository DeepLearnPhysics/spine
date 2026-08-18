"""Vertex proposal network and supervised objective."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from spine.data import TensorBatch, TensorSchema
from spine.model import sparse
from spine.model.cnn.blocks import ResNetBlock
from spine.model.common.weighting import get_class_weights

from .ppn import PointProposalDecoder, PPNLoss, ProposalTask

VertexPPNOutput: TypeAlias = dict[str, TensorBatch | list[TensorBatch]]
VertexPPNLossOutput: TypeAlias = dict[
    str,
    torch.Tensor | float | list[TensorBatch],
]

__all__ = [
    "VertexPPN",
    "VertexPPNLoss",
    "VertexPPNOutput",
    "VertexPPNLossOutput",
]


@dataclass(frozen=True)
class _VertexConfig:
    """Validated vertex-head configuration."""

    mask_score_threshold: float


@dataclass(frozen=True)
class _VertexLossConfig:
    """Validated vertex-loss configuration."""

    resolution: float
    balance_mask_loss: bool
    mask_weighting_mode: str
    reg_loss_weight: float
    mask_loss_weight: float
    return_mask_labels: bool


def vertex_raw_schema(dimension: int) -> TensorSchema:
    """Return the schema of raw vertex-head predictions.

    Parameters
    ----------
    dimension : int
        Number of spatial offset coordinates.

    Returns
    -------
    TensorSchema
        Schema containing offset and binary vertex-logit fields.
    """
    return TensorSchema(
        feature_fields={
            "offsets": tuple(range(dimension)),
            "vertex_logits": (dimension, dimension + 1),
        },
        feats_only=True,
    )


class VertexPPN(PointProposalDecoder[VertexPPNOutput]):
    """Predict interaction vertices from sparse UResNet feature planes.

    The vertexer uses the same generic multiscale proposal architecture as
    particle-point PPN, but owns independent foreground, offset and vertexness
    heads. It can therefore run by itself or serve as the independent vertex
    path in a model that also predicts per-particle points.
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        vertex: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the vertex proposal decoder.

        Parameters
        ----------
        uresnet : dict
            Shared UResNet configuration.
        vertex : dict, optional
            Vertex-head configuration. ``mask_score_threshold`` controls the
            foreground decision and defaults to ``0.5``. The historical
            ``score_threshold`` spelling remains accepted.

        Raises
        ------
        ValueError
            If the score threshold lies outside ``[0, 1]``.
        TypeError
            If the vertex block contains an unknown option.
        """
        config = self._parse_model_config(vertex)
        backbone = dict(uresnet)
        backbone.pop("num_classes", None)
        ghost = bool(backbone.pop("ghost", False))

        super().__init__(
            backbone,
            [
                ProposalTask(
                    "vertex",
                    "vertex_pred",
                    config.mask_score_threshold,
                )
            ],
            ghost=ghost,
            legacy_layers=False,
        )
        self.mask_score_threshold = config.mask_score_threshold

        # The final feature refinement and predictions are vertex-specific.
        num_output = self.num_planes[0]
        self.final_block = ResNetBlock(
            num_output,
            num_output,
            dimension=self.dimension,
            activation=self.act_cfg,
        )
        self.vertex_regression = sparse.Convolution(
            num_output,
            self.dimension,
            kernel_size=3,
            stride=1,
            dimension=self.dimension,
            bias=self.allow_bias,
        )

    @staticmethod
    def _parse_model_config(vertex: dict[str, Any] | None) -> _VertexConfig:
        """Validate and normalize a vertex-head configuration.

        Parameters
        ----------
        vertex : dict, optional
            User-provided vertex options.

        Returns
        -------
        _VertexConfig
            Validated score threshold.

        Raises
        ------
        TypeError
            If an unknown option is present.
        ValueError
            If the threshold lies outside ``[0, 1]``.
        """
        config = {} if vertex is None else dict(vertex)
        threshold = float(
            config.pop(
                "mask_score_threshold",
                config.pop("score_threshold", 0.5),
            )
        )
        if config:
            unexpected = ", ".join(sorted(config))
            raise TypeError(f"Unexpected vertex configuration: {unexpected}.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("`mask_score_threshold` must be between zero and one.")
        return _VertexConfig(threshold)

    def forward(
        self,
        final_tensor: sparse.SparseTensor,
        decoder_tensors: Sequence[sparse.SparseTensor],
        ghost: sparse.SparseTensor | None = None,
        seg_label: TensorBatch | None = None,
    ) -> VertexPPNOutput:
        """Predict vertex quantities from a UResNet feature pyramid.

        Parameters
        ----------
        final_tensor : sparse.SparseTensor
            Deepest UResNet encoder representation.
        decoder_tensors : sequence of sparse.SparseTensor
            UResNet decoder planes ordered from deep to shallow.
        ghost : sparse.SparseTensor, optional
            Ghost logits required when the shared backbone predicts ghosts.
        seg_label : TensorBatch, optional
            Accepted for interface symmetry with the generic decoder. The
            standalone vertexer normally uses predicted ghost logits.

        Returns
        -------
        VertexPPNOutput
            Row-aligned and unique predictions, multiscale foreground products
            and final-resolution coordinates.
        """
        x, proposal_outputs = self.decode(
            final_tensor,
            decoder_tensors,
            ghost,
            seg_label,
        )
        vertex_outputs = proposal_outputs["vertex"]
        output_coords = TensorBatch(
            x.coordinates,
            x.counts,
            has_batch_col=True,
            coord_cols=tuple(range(1, self.dimension + 1)),
        )

        # Regress an offset from each voxel center and classify whether that
        # site belongs to a vertex neighborhood.
        x = self.final_block(x)
        offsets = self.vertex_regression(x)
        points = x.replace_features(
            torch.cat(
                (
                    offsets.features,
                    vertex_outputs["layers"][-1].torch_tensor(),
                ),
                dim=1,
            )
        )
        points_unique = points.to_tensor_batch(include_coordinates=False)
        points_aligned = points.to_tensor_batch(
            include_coordinates=False,
            restore=True,
        )
        schema = vertex_raw_schema(self.dimension)
        points_unique.schema = schema
        points_aligned.schema = schema

        return {
            "vertex_points": points_aligned,
            "vertex_points_unique": points_unique,
            "vertex_masks": vertex_outputs["masks"],
            "vertex_layers": vertex_outputs["layers"],
            "vertex_coords": vertex_outputs["coords"],
            "vertex_output_coords": output_coords,
        }


class VertexPPNLoss(torch.nn.Module):
    """Train multiscale vertex foreground and position predictions.

    Vertex labels follow the :class:`LArCVVertexPointParser` contract: their
    coordinates are the true interaction vertices and the optional feature is
    an interaction identifier. Sites within ``resolution`` voxels of a target
    are foreground. At those sites, the predicted offset from the voxel center
    is trained to recover the closest target vertex.
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        vertex_loss: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the vertex proposal objective.

        Parameters
        ----------
        uresnet : dict
            UResNet configuration supplying ``depth`` and ``data_dim``.
        vertex_loss : dict, optional
            Foreground and regression loss configuration.
        """
        super().__init__()
        try:
            self.depth = int(uresnet["depth"])
        except KeyError as err:
            raise ValueError("The UResNet configuration must define `depth`.") from err
        if self.depth < 2:
            raise ValueError(
                "Vertex PPN loss requires a UResNet depth of at least two."
            )
        self.dimension = int(uresnet.get("data_dim", 3))

        config = self._parse_loss_config(vertex_loss)
        self.resolution = config.resolution
        self.balance_mask_loss = config.balance_mask_loss
        self.mask_weighting_mode = config.mask_weighting_mode
        self.reg_loss_weight = config.reg_loss_weight
        self.mask_loss_weight = config.mask_loss_weight
        self.return_mask_labels = config.return_mask_labels

    @staticmethod
    def _parse_loss_config(
        vertex_loss: dict[str, Any] | None,
    ) -> _VertexLossConfig:
        """Validate and normalize vertex-loss options.

        Parameters
        ----------
        vertex_loss : dict, optional
            User-provided objective configuration.

        Returns
        -------
        _VertexLossConfig
            Validated foreground and regression settings.

        Raises
        ------
        TypeError
            If an unknown option is present.
        ValueError
            If the loss name, resolution or component weights are invalid.
        """
        config = {} if vertex_loss is None else dict(vertex_loss)
        mask_loss = str(config.pop("mask_loss", "CE"))
        resolution = float(config.pop("resolution", 5.0))
        balance_mask_loss = bool(config.pop("balance_mask_loss", True))
        mask_weighting_mode = str(config.pop("mask_weighting_mode", "const"))
        reg_loss_weight = float(config.pop("reg_loss_weight", 1.0))
        mask_loss_weight = float(config.pop("mask_loss_weight", 1.0))
        return_mask_labels = bool(config.pop("return_mask_labels", False))
        if config:
            unexpected = ", ".join(sorted(config))
            raise TypeError(f"Unexpected vertex-loss configuration: {unexpected}.")
        if mask_loss != "CE":
            raise ValueError(f"Mask loss name not recognized: {mask_loss}")
        if resolution <= 0.0:
            raise ValueError(f"`resolution` must be positive, got {resolution}.")
        if reg_loss_weight < 0.0 or mask_loss_weight < 0.0:
            raise ValueError("Vertex loss weights must be nonnegative.")
        return _VertexLossConfig(
            resolution,
            balance_mask_loss,
            mask_weighting_mode,
            reg_loss_weight,
            mask_loss_weight,
            return_mask_labels,
        )

    def forward(
        self,
        vertex_label: TensorBatch,
        vertex_points: TensorBatch,
        vertex_masks: Sequence[TensorBatch],
        vertex_layers: Sequence[TensorBatch],
        vertex_coords: Sequence[TensorBatch],
        vertex_output_coords: TensorBatch,
        vertex_points_unique: TensorBatch | None = None,
        **_: object,
    ) -> VertexPPNLossOutput:
        """Compute vertex foreground and offset losses.

        Parameters
        ----------
        vertex_label : TensorBatch
            Parsed interaction-vertex coordinates.
        vertex_points : TensorBatch
            Row-aligned final vertex predictions.
        vertex_masks : sequence of TensorBatch
            Thresholded foreground masks at every proposal resolution.
        vertex_layers : sequence of TensorBatch
            Foreground logits at every proposal resolution.
        vertex_coords : sequence of TensorBatch
            Sparse coordinates corresponding to each foreground layer.
        vertex_output_coords : TensorBatch
            Coordinates of the final proposal feature plane.
        vertex_points_unique : TensorBatch, optional
            Predictions on unique sparse sites, preferred for supervision.
        **_ : object
            Other model outputs ignored by this loss.

        Returns
        -------
        VertexPPNLossOutput
            Combined loss, component metrics and optional mask labels.
        """
        expected = self.depth - 1
        layer_counts = {
            "vertex_masks": len(vertex_masks),
            "vertex_layers": len(vertex_layers),
            "vertex_coords": len(vertex_coords),
        }
        invalid = {
            name: count for name, count in layer_counts.items() if count != expected
        }
        if invalid:
            details = ", ".join(f"{name}={count}" for name, count in invalid.items())
            raise ValueError(f"Expected {expected} vertex layers, got {details}.")

        coords_final = vertex_coords[-1]
        if not torch.equal(
            coords_final.torch_tensor(), vertex_output_coords.torch_tensor()
        ):
            raise ValueError(
                "`vertex_output_coords` must match the final `vertex_coords` tensor."
            )
        loss_points = (
            vertex_points if vertex_points_unique is None else vertex_points_unique
        )
        if loss_points.shape[0] != coords_final.shape[0]:
            raise ValueError(
                "Vertex predictions and final coordinates must have matching rows."
            )

        label_points = vertex_label.coords.torch_tensor()
        positive_list, closest_list = [], []
        offset = 0
        for batch_index in range(vertex_label.batch_size):
            points = vertex_label.coords[batch_index]
            count = coords_final.counts[batch_index]
            if len(points) == 0:
                positive_list.append(
                    torch.zeros(count, dtype=torch.bool, device=coords_final.device)
                )
                closest_list.append(
                    torch.full(
                        (count,),
                        -1,
                        dtype=torch.long,
                        device=coords_final.device,
                    )
                )
                continue

            anchors = coords_final.coords[batch_index] + 0.5
            positives, closest = PPNLoss.get_ppn_positives(
                anchors,
                points,
                self.resolution,
                offset,
            )
            positive_list.append(positives)
            closest_list.append(closest)
            offset += len(points)

        positives = torch.cat(positive_list).long()
        closest = torch.cat(closest_list)
        mask_tensor = sparse.SparseTensor(
            positives[:, None].float(),
            coordinates=coords_final.batch_coordinates,
        )

        # Construct coarser targets by max pooling the final vertex heatmap.
        downsample = sparse.MaxPooling(2, 2, dimension=self.dimension)
        dtype = vertex_layers[-1].dtype
        device = vertex_layers[-1].device
        mask_losses = torch.zeros(expected, dtype=dtype, device=device)
        mask_accuracies = torch.zeros(expected, dtype=dtype, device=device)
        mask_labels_output = []
        for reverse_index in range(expected):
            layer_index = expected - 1 - reverse_index
            coords = vertex_coords[layer_index]
            logits = vertex_layers[layer_index].torch_tensor()
            mask_features = PPNLoss.align_coordinate_values(
                mask_tensor.coordinates,
                mask_tensor.features,
                coords.batch_coordinates,
                f"vertex mask at layer {layer_index}",
                missing_value=0,
            )
            labels = mask_features.flatten().long()
            if vertex_masks[layer_index].shape[0] != len(logits):
                raise ValueError(
                    f"Vertex mask and score rows differ at layer {layer_index}."
                )
            if self.return_mask_labels:
                mask_labels_output.append(TensorBatch(labels[:, None], coords.counts))

            if len(logits) > 0:
                weights = None
                if self.balance_mask_loss:
                    weights = get_class_weights(
                        labels,
                        2,
                        self.mask_weighting_mode,
                    )
                mask_losses[layer_index] = torch.nn.functional.cross_entropy(
                    logits,
                    labels,
                    weight=weights,
                )
                with torch.no_grad():
                    mask_accuracies[layer_index] = (
                        (torch.argmax(logits, dim=1) == labels).float().mean()
                    )
            else:
                mask_losses[layer_index] = logits.sum() * 0.0

            if layer_index != 0:
                mask_tensor = downsample(mask_tensor)

        # Offset regression uses ground-truth-positive sites so an initially
        # weak foreground classifier cannot starve the regression head.
        positive_indices = torch.where(positives)[0]
        reg_loss = mask_losses.sum() * 0.0
        reg_accuracy = torch.tensor(1.0, dtype=dtype, device=device)
        if len(positive_indices) > 0:
            anchors = coords_final.coords.torch_tensor() + 0.5
            predictions = (loss_points.feature("offsets").torch_tensor() + anchors)[
                positive_indices
            ]
            targets = label_points[closest[positive_indices]]
            reg_loss = torch.nn.functional.mse_loss(predictions, targets)
            with torch.no_grad():
                reg_accuracy = (
                    (
                        torch.linalg.vector_norm(predictions - targets, dim=1)
                        < self.resolution
                    )
                    .float()
                    .mean()
                )

        mask_loss = mask_losses.mean()
        mask_accuracy = mask_accuracies.mean()
        loss = self.mask_loss_weight * mask_loss + self.reg_loss_weight * reg_loss
        accuracy = (mask_accuracy + reg_accuracy) / 2
        result: VertexPPNLossOutput = {
            "loss": loss,
            "accuracy": accuracy.item(),
            "mask_loss": mask_loss.item(),
            "mask_accuracy": mask_accuracy.item(),
            "reg_loss": reg_loss.item(),
            "reg_accuracy": reg_accuracy.item(),
        }
        for layer_index in range(expected):
            result[f"mask_loss_layer_{layer_index}"] = mask_losses[layer_index]
            result[f"mask_accuracy_layer_{layer_index}"] = mask_accuracies[layer_index]
        if self.return_mask_labels:
            result["mask_labels"] = mask_labels_output[::-1]
        return result
