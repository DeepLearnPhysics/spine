from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from spine.constants import (
    COORD_COLS,
    GHOST_SHP,
    PART_COL,
    PPN_LENDP_COL,
    PPN_LPART_COL,
    PPN_LTYPE_COL,
    PPN_ROFF_COLS,
    PPN_RTYPE_COLS,
    SHAPE_COL,
    TRACK_SHP,
    VALUE_COL,
)
from spine.data import TensorBatch
from spine.model import sparse
from spine.model.cnn.act_norm import act_factory
from spine.model.cnn.blocks import ResNetBlock
from spine.model.cnn.configuration import setup_cnn_configuration
from spine.utils.logger import logger
from spine.utils.torch.scripts import cdist_fast
from spine.utils.weighting import get_class_weights

PPNOutput: TypeAlias = dict[str, TensorBatch | list[TensorBatch]]
PPNLossOutput: TypeAlias = dict[
    str,
    torch.Tensor | float | list[TensorBatch],
]

__all__ = ["PPN", "PPNLoss", "PPNOutput", "PPNLossOutput"]


@dataclass(frozen=True)
class _PPNConfig:
    """Validated PPN head configuration."""

    mask_score_threshold: float
    classify_endpoints: bool
    propagate_all: bool
    use_binary_mask: bool
    use_true_ghost_mask: bool


@dataclass(frozen=True)
class _PPNModules:
    """Modules constructed for a PPN head."""

    decoding_block: torch.nn.Sequential
    decoding_conv: torch.nn.Sequential
    ppn_masks: torch.nn.ModuleList
    expand_as: ExpandAs
    final_block: torch.nn.Module
    ppn_pixel_pos: torch.nn.Module
    ppn_type: torch.nn.Module
    ppn_endpoint: torch.nn.Module | None
    masker: torch.nn.Module | None
    merge_concat: torch.nn.Module | None
    ghost_mask: torch.nn.Module | None


@dataclass(frozen=True)
class _PPNLossConfig:
    """Validated PPN loss configuration."""

    resolution: float
    point_classes: tuple[int, ...] | None
    balance_mask_loss: bool
    mask_weighting_mode: str
    balance_type_loss: bool
    type_weighting_mode: str
    reg_loss_weight: float
    type_loss_weight: float
    mask_loss_weight: float
    endpoint_loss_weight: float
    return_mask_labels: bool
    restrict_to_clusters: bool


class PPN(sparse.Network):
    """Predict sparse points of interest from UResNet feature planes.

    PPN follows the UResNet decoder resolution schedule. At each level it
    predicts foreground logits and uses their probabilities to gate features
    propagated to the next level. The final plane predicts a sub-voxel offset,
    semantic point type, foreground logits, and optionally track endpoint
    class. When ghost masking is enabled, decoder features are pruned before
    they enter the proposal path.

    Notes
    -----
    The returned ``ppn_points`` is restored to the original input row order;
    ``ppn_points_unique`` contains one prediction per active sparse site.

    See Also
    --------
    PPNLoss
        Supervised objective for PPN outputs.
    spine.model.uresnet.ppn.UResNetPPN
        Task model that combines the UResNet segmentation backbone and PPN.
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        ppn: dict[str, Any],
    ) -> None:
        """Initialize the point proposal decoder.

        Parameters
        ----------
        uresnet : dict
            UResNet configuration. PPN consumes ``num_classes``, ``ghost`` and
            the shared CNN parameters.
        ppn : dict
            Proposal-head configuration.
        """
        # Initialize the parent class
        super().__init__(uresnet.get("data_dim", 3))

        # Initialize the shared backbone configuration. Ghost handling belongs
        # to the UResNet configuration so the PPN block cannot override it.
        backbone = dict(uresnet)
        try:
            self.num_classes = int(backbone.pop("num_classes"))
        except KeyError as err:
            raise ValueError(
                "The UResNet configuration must define `num_classes`."
            ) from err
        self.ghost = bool(backbone.pop("ghost", False))
        setup_cnn_configuration(self, **backbone)
        if self.depth < 2:
            raise ValueError("PPN requires a UResNet depth of at least two.")

        # Store the PPN-specific configuration.
        config = self._parse_model_config(**ppn)
        self.mask_score_threshold = config.mask_score_threshold
        self.classify_endpoints = config.classify_endpoints
        self.propagate_all = config.propagate_all
        self.use_binary_mask = config.use_binary_mask
        self.use_true_ghost_mask = config.use_true_ghost_mask
        if self.use_true_ghost_mask and not self.ghost:
            raise ValueError("`use_true_ghost_mask` requires UResNet `ghost: true`.")

        # Construct and register the PPN modules.
        modules = self._build_modules()
        self.decoding_block = modules.decoding_block
        self.decoding_conv = modules.decoding_conv
        self.ppn_masks = modules.ppn_masks
        self.expand_as = modules.expand_as
        self.final_block = modules.final_block
        self.ppn_pixel_pos = modules.ppn_pixel_pos
        self.ppn_type = modules.ppn_type
        self.ppn_endpoint = modules.ppn_endpoint
        self.masker = modules.masker
        self.merge_concat = modules.merge_concat
        self.ghost_mask = modules.ghost_mask

    @staticmethod
    def _parse_model_config(
        mask_score_threshold: float = 0.5,
        classify_endpoints: bool = False,
        propagate_all: bool = False,
        use_binary_mask: bool = False,
        use_true_ghost_mask: bool = False,
    ) -> _PPNConfig:
        """Validate and normalize the PPN-specific configuration.

        Parameters
        ----------
        mask_score_threshold : float, default 0.5
            Foreground probability above which a site is marked positive.
        classify_endpoints : bool, default False
            Predict start-versus-end logits for track points.
        propagate_all : bool, default False
            Replace foreground probabilities with ones so every feature is
            propagated.
        use_binary_mask : bool, default False
            Threshold foreground probabilities before feature gating.
        use_true_ghost_mask : bool, default False
            Use segmentation labels rather than predicted ghost logits.

        Returns
        -------
        _PPNConfig
            Normalized PPN configuration.

        Raises
        ------
        ValueError
            If ``mask_score_threshold`` is outside ``[0, 1]``.
        """
        if not 0.0 <= mask_score_threshold <= 1.0:
            raise ValueError("`mask_score_threshold` must be between zero and one.")
        return _PPNConfig(
            mask_score_threshold=mask_score_threshold,
            classify_endpoints=classify_endpoints,
            propagate_all=propagate_all,
            use_binary_mask=use_binary_mask,
            use_true_ghost_mask=use_true_ghost_mask,
        )

    def _build_modules(self) -> _PPNModules:
        """Construct the proposal decoder and its output heads."""
        # Initialize the decoding blocks
        decoding_block = []
        decoding_conv = []
        ppn_masks = torch.nn.ModuleList()
        for level in range(self.depth - 2, -1, -1):
            upsample_layers = [
                sparse.BatchNorm(self.num_planes[level + 1]),
                act_factory(self.act_cfg),
                sparse.ConvolutionTranspose(
                    in_channels=self.num_planes[level + 1],
                    out_channels=self.num_planes[level],
                    kernel_size=2,
                    stride=2,
                    dimension=self.dimension,
                ),
            ]
            decoding_conv.append(torch.nn.Sequential(*upsample_layers))

            decoding_blocks = []
            for repetition in range(self.reps):
                decoding_blocks.append(
                    ResNetBlock(
                        self.num_planes[level] * (2 if repetition == 0 else 1),
                        self.num_planes[level],
                        dimension=self.dimension,
                        activation=self.act_cfg,
                    )
                )
            decoding_block.append(torch.nn.Sequential(*decoding_blocks))
            ppn_masks.append(sparse.Linear(self.num_planes[level], 2))

        decoding_block_module = torch.nn.Sequential(*decoding_block)
        decoding_conv_module = torch.nn.Sequential(*decoding_conv)

        # Expands the scores to the appropriate feature shape
        expand_as = ExpandAs()

        # Final ResNet block at the original image size
        num_output = self.num_planes[0]
        final_block = ResNetBlock(
            num_output, num_output, dimension=self.dimension, activation=self.act_cfg
        )

        # Final linear layer for positional regression (dimension size)
        ppn_pixel_pos = sparse.Linear(num_output, self.dimension)

        # Final convolution layer for type classification
        ppn_type = sparse.Linear(num_output, self.num_classes)

        # Final convolution layer for endpoint prediction
        if self.classify_endpoints:
            ppn_endpoint = sparse.Linear(num_output, 2)
        else:
            ppn_endpoint = None

        # Ghost point removal tools
        masker = None
        merge_concat = None
        ghost_mask = None
        if self.ghost:
            logger.debug("Ghost masking is enabled for PPN.")
            masker = AttentionMask()
            merge_concat = MergeConcat()
            ghost_mask = GhostMask(self.dimension)

        return _PPNModules(
            decoding_block=decoding_block_module,
            decoding_conv=decoding_conv_module,
            ppn_masks=ppn_masks,
            expand_as=expand_as,
            final_block=final_block,
            ppn_pixel_pos=ppn_pixel_pos,
            ppn_type=ppn_type,
            ppn_endpoint=ppn_endpoint,
            masker=masker,
            merge_concat=merge_concat,
            ghost_mask=ghost_mask,
        )

    def forward(
        self,
        final_tensor: sparse.SparseTensor,
        decoder_tensors: Sequence[sparse.SparseTensor],
        ghost: sparse.SparseTensor | None = None,
        seg_label: TensorBatch | None = None,
    ) -> PPNOutput:
        """Predict points of interest from UResNet feature tensors.

        Parameters
        ----------
        final_tensor : sparse.SparseTensor
            Deepest UResNet encoder representation.
        decoder_tensors : sequence of sparse.SparseTensor
            UResNet decoder feature planes ordered from deep to shallow.
        ghost : sparse.SparseTensor, optional
            Predicted ghost logits on the input coordinate map.
        seg_label : TensorBatch, optional
            Segmentation labels used when ``use_true_ghost_mask`` is enabled.

        Returns
        -------
        PPNOutput
            Point predictions, foreground masks, logits and coordinates at
            each proposal resolution.

        Raises
        ------
        ValueError
            If required ghost inputs are missing, labels do not align with the
            input reference, or final proposal coordinates do not align with
            the highest-resolution decoder plane.
        """
        # Get the list of decoder feature maps
        decoder_feature_maps = []
        decoder_ghost_masks = []
        if self.ghost:
            ghost_mask_layer = self.ghost_mask
            masker = self.masker
            if ghost_mask_layer is None or masker is None:
                raise RuntimeError(
                    "Ghost-enabled PPN is missing its ghost-processing modules."
                )

            # If there are ghosts, must downsample the ghost label/prediction
            # and apply it to each decoder feature map
            with torch.no_grad():
                if self.use_true_ghost_mask:
                    # If using the true ghost mask, use the label tensor
                    if seg_label is None:
                        raise ValueError(
                            "If `use_true_ghost_mask` is set to `True`, must "
                            "provide the `seg_label` tensor."
                        )

                    labels = seg_label.torch_tensor()
                    if labels.shape[0] != decoder_tensors[-1].reference_size:
                        raise ValueError(
                            "The label tensor length must match that "
                            "of the last UResNet layer"
                        )

                    ghost_coords = labels[:, :VALUE_COL]
                    ghost_mask_tensor = labels[:, SHAPE_COL] < GHOST_SHP
                else:
                    # If using predictions, convert the ghost scores to a mask
                    if ghost is None:
                        raise ValueError(
                            "Ghost masking requires ghost prediction logits."
                        )
                    ghost_coords = ghost.coordinates
                    ghost_mask_tensor = 1.0 - torch.argmax(
                        ghost.features, dim=1, keepdim=True
                    )

                ghost_mask_tensor = ghost_mask_tensor.to(
                    dtype=decoder_tensors[-1].features.dtype
                )
                ghost_mask = sparse.SparseTensor(
                    ghost_mask_tensor,
                    coordinates=ghost_coords,
                    coordinate_manager=decoder_tensors[-1].coordinate_manager,
                    batch_size=decoder_tensors[-1].batch_size,
                )

            # Downsample stride 1 ghost mask to all intermediate decoder layers
            for decoder_tensor in reversed(decoder_tensors):
                scaled_ghost_mask = ghost_mask_layer(
                    ghost_mask,
                    decoder_tensor,
                )
                nonghost_tensor = masker(
                    decoder_tensor,
                    scaled_ghost_mask,
                )
                decoder_feature_maps.append(nonghost_tensor)
                decoder_ghost_masks.append(scaled_ghost_mask)

            decoder_feature_maps = decoder_feature_maps[::-1]
            decoder_ghost_masks = decoder_ghost_masks[::-1]

        else:
            decoder_feature_maps = list(decoder_tensors)

        expected_layers = self.depth - 1
        if len(decoder_feature_maps) != expected_layers:
            raise ValueError(
                f"Expected {expected_layers} decoder tensors, got "
                f"{len(decoder_feature_maps)}."
            )

        # Loop over the PPN decoding path
        ppn_masks, ppn_layers, ppn_coords = [], [], []
        x = final_tensor
        for level, layer in enumerate(self.decoding_conv):
            # Pass the previous features through the decoding convolution
            x = layer(x)

            # Merge with the UesNet decoding features
            decoder_tensor = decoder_feature_maps[level]
            if self.ghost:
                if self.merge_concat is None or self.masker is None:
                    raise RuntimeError("Ghost-enabled PPN is missing its merge module.")
                x = self.masker(x, decoder_ghost_masks[level])
                x = self.merge_concat(decoder_tensor, x)
            else:
                x = sparse.cat(decoder_tensor, x)

            # Apply the decoding block, linear layer and sigmoid function
            x = self.decoding_block[level](x)
            scores = self.ppn_masks[level](x)
            probabilities = sparse.softmax(scores, dim=1)
            foreground_mask = probabilities.features[:, 1:] > self.mask_score_threshold

            # Store the coordinates, raw score logits and score mask
            counts = scores.counts
            ppn_coords.append(
                TensorBatch(
                    scores.coordinates,
                    counts,
                    has_batch_col=True,
                    coord_cols=COORD_COLS,
                )
            )
            ppn_layers.append(TensorBatch(scores.features, counts))
            ppn_masks.append(TensorBatch(foreground_mask, counts))

            # Expand the score mask
            expanded_scores = self.expand_as(
                probabilities,
                x.features.shape,
                propagate_all=self.propagate_all,
                use_binary_mask=self.use_binary_mask,
                score_threshold=self.mask_score_threshold,
            )

            # Scale the out of this layer using the score mask
            x = x * expanded_scores.detach()

        # Output set of coordinates (should match the last decoder tensor)
        if x.coordinates.shape[0] != decoder_feature_maps[-1].shape[0]:
            raise ValueError(
                "The output of the last PPN layer should be consistent "
                "with the length of the last UResNet decoder layer"
            )
        if not torch.equal(x.coordinates, decoder_feature_maps[-1].coordinates):
            raise ValueError(
                "The final PPN coordinates must match the highest-resolution "
                "decoder coordinates."
            )
        final_counts = x.counts
        ppn_output_coords = TensorBatch(
            x.coordinates, final_counts, has_batch_col=True, coord_cols=COORD_COLS
        )

        # Pass the final PPN tensor through the individual predictions, combine
        x = self.final_block(x)
        pixel_pos = self.ppn_pixel_pos(x)
        ppn_type = self.ppn_type(x)
        ppn_endpoint = None
        if self.ppn_endpoint is not None:
            ppn_endpoint = self.ppn_endpoint(x)

        # X, Y, Z, logits, and prob score
        point_features = x.replace_features(
            torch.cat(
                [
                    pixel_pos.features,
                    ppn_type.features,
                    ppn_layers[-1].torch_tensor(),
                ],
                dim=1,
            )
        )
        ppn_points_unique = point_features.to_tensor_batch(
            include_coordinates=False,
        )
        ppn_points = point_features.to_tensor_batch(
            include_coordinates=False, restore=True
        )

        result: PPNOutput = {
            "ppn_points": ppn_points,
            "ppn_points_unique": ppn_points_unique,
            "ppn_masks": ppn_masks,
            "ppn_layers": ppn_layers,
            "ppn_coords": ppn_coords,
            "ppn_output_coords": ppn_output_coords,
        }
        if ppn_endpoint is not None:
            result["ppn_classify_endpoints_unique"] = ppn_endpoint.to_tensor_batch(
                include_coordinates=False,
            )
            result["ppn_classify_endpoints"] = ppn_endpoint.to_tensor_batch(
                include_coordinates=False, restore=True
            )

        return result


class PPNLoss(torch.nn.Module):
    """Compute foreground, position, type and endpoint objectives for PPN.

    Foreground cross-entropy is evaluated at every proposal resolution.
    Position and semantic type losses are evaluated at final-resolution sites
    within ``resolution`` of a target point. Endpoint classification is
    included when the model supplies endpoint logits.

    See Also
    --------
    PPN
        Network that produces the predictions consumed by this loss.
    """

    def __init__(
        self,
        uresnet: dict[str, Any],
        ppn_loss: dict[str, Any],
    ) -> None:
        """Initialize the point proposal objective.

        Parameters
        ----------
        uresnet : dict
            UResNet configuration supplying ``depth`` and ``data_dim``.
        ppn_loss : dict
            PPN loss configuration.
        """
        # Initialize the parent class
        super().__init__()

        # Store the backbone parameters required to construct proposal targets.
        try:
            self.depth = int(uresnet["depth"])
        except KeyError as err:
            raise ValueError("The UResNet configuration must define `depth`.") from err
        if self.depth < 2:
            raise ValueError("PPN loss requires a UResNet depth of at least two.")
        self.dimension = int(uresnet.get("data_dim", 3))

        # Store the normalized loss configuration.
        config = self._parse_loss_config(**ppn_loss)
        self.resolution = config.resolution
        self.point_classes = config.point_classes
        self.balance_mask_loss = config.balance_mask_loss
        self.mask_weighting_mode = config.mask_weighting_mode
        self.balance_type_loss = config.balance_type_loss
        self.type_weighting_mode = config.type_weighting_mode
        self.reg_loss_weight = config.reg_loss_weight
        self.type_loss_weight = config.type_loss_weight
        self.mask_loss_weight = config.mask_loss_weight
        self.endpoint_loss_weight = config.endpoint_loss_weight
        self.return_mask_labels = config.return_mask_labels
        self.restrict_to_clusters = config.restrict_to_clusters

        # Instantiate the component loss functions.
        self.reg_loss_fn = torch.nn.MSELoss(reduction="mean")
        self.type_loss_fn = torch.nn.functional.cross_entropy
        self.end_loss_fn = torch.nn.functional.cross_entropy
        self.mask_loss = "CE"
        self.mask_loss_fn = torch.nn.functional.cross_entropy

    @staticmethod
    def _parse_loss_config(
        mask_loss: str = "CE",
        resolution: float = 5.0,
        point_classes: int | Sequence[int] | None = None,
        balance_mask_loss: bool = True,
        mask_weighting_mode: str = "const",
        balance_type_loss: bool = True,
        type_weighting_mode: str = "const",
        reg_loss_weight: float = 1.0,
        type_loss_weight: float = 1.0,
        mask_loss_weight: float = 1.0,
        endpoint_loss_weight: float = 1.0,
        return_mask_labels: bool = False,
        restrict_to_clusters: bool = False,
    ) -> _PPNLossConfig:
        """Validate and normalize the loss function parameters.

        Parameters
        ----------
        mask_loss : str, default "CE"
            Foreground loss name. Currently only cross-entropy is supported.
        resolution : float, default 5.0
            Maximum voxel-space distance between a site and target point for
            that site to be positive.
        point_classes : int or sequence of int, optional
            Restrict supervision to target points with these semantic classes.
        balance_mask_loss : bool, default True
            Apply class weights to foreground loss.
        mask_weighting_mode : str, default "const"
            Class-weighting strategy for foreground loss.
        balance_type_loss : bool, default True
            Apply class weights to semantic type loss.
        type_weighting_mode : str, default "const"
            Class-weighting strategy for semantic type loss.
        reg_loss_weight : float, default 1.0
            Relative weight of the position regression loss.
        type_loss_weight : float, default 1.0
            Relative weight of the semantic type loss.
        mask_loss_weight : float, default 1.0
            Relative weight of the foreground loss.
        endpoint_loss_weight : float, default 1.0
            Relative weight of endpoint classification loss.
        return_mask_labels : bool, default False
            Include generated foreground targets in the result.
        restrict_to_clusters : bool, default False
            Associate sites only with target points from the same particle.

        Returns
        -------
        _PPNLossConfig
            Normalized PPN loss configuration.

        Raises
        ------
        ValueError
            If ``mask_loss`` is unsupported.
        """
        if point_classes is None:
            normalized_point_classes = None
        elif isinstance(point_classes, int):
            normalized_point_classes = (point_classes,)
        else:
            normalized_point_classes = tuple(point_classes)

        if mask_loss != "CE":
            raise ValueError(f"Mask loss name not recognized: {mask_loss}")
        if resolution <= 0.0:
            raise ValueError(f"`resolution` must be positive, got {resolution}.")
        loss_weights = {
            "reg_loss_weight": reg_loss_weight,
            "type_loss_weight": type_loss_weight,
            "mask_loss_weight": mask_loss_weight,
            "endpoint_loss_weight": endpoint_loss_weight,
        }
        for name, value in loss_weights.items():
            if value < 0.0:
                raise ValueError(f"`{name}` must be nonnegative, got {value}.")
        return _PPNLossConfig(
            resolution=resolution,
            point_classes=normalized_point_classes,
            balance_mask_loss=balance_mask_loss,
            mask_weighting_mode=mask_weighting_mode,
            balance_type_loss=balance_type_loss,
            type_weighting_mode=type_weighting_mode,
            reg_loss_weight=reg_loss_weight,
            type_loss_weight=type_loss_weight,
            mask_loss_weight=mask_loss_weight,
            endpoint_loss_weight=endpoint_loss_weight,
            return_mask_labels=return_mask_labels,
            restrict_to_clusters=restrict_to_clusters,
        )

    @staticmethod
    def align_coordinate_values(
        source_coords: torch.Tensor,
        source_values: torch.Tensor,
        target_coords: torch.Tensor,
        value_name: str,
    ) -> torch.Tensor:
        """Align coordinate-associated values to a target sparse row order.

        The common case uses an exact coordinate-order match and returns
        immediately. The fallback coalesces duplicate source coordinates,
        checks that their values agree, and gathers them in target order.

        Parameters
        ----------
        source_coords : torch.Tensor
            ``(N, D + 1)`` source coordinates including the batch column.
        source_values : torch.Tensor
            ``(N, ...)`` values associated with the source coordinates.
        target_coords : torch.Tensor
            ``(M, D + 1)`` coordinates defining the desired row order.
        value_name : str
            Human-readable value name used in validation errors.

        Returns
        -------
        torch.Tensor
            ``(M, ...)`` values aligned to ``target_coords``.

        Raises
        ------
        ValueError
            If shapes are inconsistent, duplicate values conflict, or target
            coordinates are absent from the source.
        """
        if len(source_coords) != len(source_values):
            raise ValueError(
                f"The {value_name} coordinates and values must have matching "
                f"lengths, got {len(source_coords)} and {len(source_values)}."
            )
        if source_coords.shape[1] != target_coords.shape[1]:
            raise ValueError(
                f"The source and target coordinates for {value_name} must have "
                "the same width."
            )

        target_coords = target_coords.to(
            device=source_coords.device,
            dtype=source_coords.dtype,
        )
        if len(source_coords) == len(target_coords) and torch.equal(
            source_coords,
            target_coords,
        ):
            return source_values

        combined_coords = torch.cat((source_coords, target_coords), dim=0)
        _, inverse = torch.unique(combined_coords, dim=0, return_inverse=True)
        source_ids = inverse[: len(source_coords)]
        target_ids = inverse[len(source_coords) :]
        num_coords = int(inverse.max().item()) + 1 if len(inverse) > 0 else 0

        representatives = source_values.new_empty(
            (num_coords, *source_values.shape[1:])
        )
        representatives[source_ids] = source_values
        if len(source_ids) > 0 and torch.any(
            representatives[source_ids] != source_values
        ):
            raise ValueError(
                f"Duplicate coordinates carry conflicting {value_name} values."
            )

        present = torch.zeros(
            num_coords,
            dtype=torch.bool,
            device=source_coords.device,
        )
        present[source_ids] = True
        if len(target_ids) > 0 and not bool(torch.all(present[target_ids])):
            raise ValueError(
                f"Target coordinates are missing from the {value_name} source."
            )

        return representatives[target_ids]

    @staticmethod
    def get_ppn_positives(
        coords: torch.Tensor,
        ppn_labels: torch.Tensor,
        resolution: float,
        offset: int,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign foreground sites to their closest valid PPN target.

        If voxel ``labels`` are provided, they restrict the mask
        applied to voxels within some distance of label points associated with
        the correct particle instance, not any particle instance.

        Parameters
        ----------
        coords : torch.Tensor
            ``(N, D)`` voxel coordinates.
        ppn_labels : torch.Tensor
            ``(P, 1 + D + L)`` point-label table.
        resolution : float
            Maximum distance at which a site is foreground.
        offset : int
            Offset that converts within-entry target indices to indices in the
            concatenated label tensor.
        labels : torch.Tensor, optional
            ``(N,)`` particle ID for each voxel.

        Returns
        -------
        tuple of torch.Tensor
            Boolean foreground mask with shape ``(N,)`` and closest global
            target index with shape ``(N,)``.

        Raises
        ------
        RuntimeError
            If pairwise distance computation does not return a matrix.
        """
        # Detach this process from the computation graph (mask not learnable)
        with torch.no_grad():
            # Compute the distance from the PPN labels to all the image points
            distance_matrix = cdist_fast(
                ppn_labels[:, COORD_COLS],
                coords,
            )
            if distance_matrix is None:
                raise RuntimeError("Failed to compute the PPN distance matrix.")

            # Mask out particle voxels for which the particle ID disagrees
            if labels is not None:
                invalid_particle_mask = ppn_labels[:, [PPN_LPART_COL]] != labels
                distance_matrix[invalid_particle_mask] = torch.inf

            # Generate a positive mask for all particle voxels within some
            # distance of their label points
            positives = (distance_matrix < resolution).any(dim=0)

            # Assign the closest label point to each postive particle voxel
            positive_indices = torch.where(positives)[0]
            closest_indices = torch.full(
                (len(coords),), -1, dtype=torch.long, device=coords.device
            )
            closest_indices[positive_indices] = offset + torch.argmin(
                distance_matrix[:, positive_indices],
                dim=0,
            )

            return positives, closest_indices

    def forward(
        self,
        ppn_label: TensorBatch,
        ppn_points: TensorBatch,
        ppn_masks: Sequence[TensorBatch],
        ppn_layers: Sequence[TensorBatch],
        ppn_coords: Sequence[TensorBatch],
        ppn_output_coords: TensorBatch,
        ppn_classify_endpoints: TensorBatch | None = None,
        ppn_points_unique: TensorBatch | None = None,
        ppn_classify_endpoints_unique: TensorBatch | None = None,
        clust_label: TensorBatch | None = None,
        **kwargs: object,
    ) -> PPNLossOutput:
        """Compute PPN losses and accuracy metrics.

        Parameters
        ----------
        ppn_label : TensorBatch
            ``(P, 1 + D + L)`` point-label table.
        ppn_points : TensorBatch
            Row-aligned final-resolution point predictions.
        ppn_masks : sequence of TensorBatch
            Predicted binary masks at each proposal resolution.
        ppn_layers : sequence of TensorBatch
            Foreground logits at each proposal resolution.
        ppn_coords : sequence of TensorBatch
            Batched coordinates at each proposal resolution.
        ppn_output_coords : TensorBatch
            Coordinates at the final proposal resolution.
        ppn_classify_endpoints : TensorBatch, optional
            Row-aligned endpoint classification logits.
        ppn_points_unique : TensorBatch, optional
            PPN predictions on unique sparse sites. When provided, these are
            used for voxel-wise losses while ``ppn_points`` remains aligned
            with the original input rows for downstream consumers.
        ppn_classify_endpoints_unique : TensorBatch, optional
            Endpoint logits on unique sparse sites.
        clust_label : TensorBatch, optional
            ``(N, 1 + D + C)`` cluster-label table used to restrict particle
            associations.
        **kwargs : object
            Other upstream outputs ignored by this loss.

        Returns
        -------
        PPNLossOutput
            Combined loss, component losses, accuracies, per-layer foreground
            metrics and optionally generated mask labels.
        """
        # Initialize the basics
        num_layers = len(ppn_layers)
        expected_layers = self.depth - 1
        layer_counts = {
            "ppn_masks": len(ppn_masks),
            "ppn_layers": num_layers,
            "ppn_coords": len(ppn_coords),
        }
        invalid_counts = {
            name: count
            for name, count in layer_counts.items()
            if count != expected_layers
        }
        if invalid_counts:
            details = ", ".join(
                f"{name}={count}" for name, count in invalid_counts.items()
            )
            raise ValueError(f"Expected {expected_layers} PPN layers, got {details}.")
        batch_size = ppn_label.batch_size
        loss_points = ppn_points if ppn_points_unique is None else ppn_points_unique
        loss_endpoints = (
            ppn_classify_endpoints
            if ppn_classify_endpoints_unique is None
            else ppn_classify_endpoints_unique
        )

        # If requested, narrow down the list of label points
        if self.point_classes is not None:
            if len(self.point_classes) == 0:
                raise ValueError(
                    "Should provide at least one class to include in the loss"
                )
            ppn_label_list = []
            for label_tensor in ppn_label.split():
                labels = label_tensor[:, PPN_LTYPE_COL]
                mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
                for point_class in self.point_classes:
                    mask |= labels == point_class
                valid_index = torch.where(mask)[0]
                ppn_label_list.append(label_tensor[valid_index])

            ppn_label = TensorBatch.from_list(ppn_label_list)

        # Compute the label mask for the final PPN layer. Record which
        # label point is closest to each image voxel (defines label for it)
        coords_final = ppn_coords[-1]
        coords_final_tensor = coords_final.torch_tensor()
        output_coords_tensor = ppn_output_coords.torch_tensor()
        if not torch.equal(coords_final_tensor, output_coords_tensor):
            raise ValueError(
                "`ppn_output_coords` must match the final `ppn_coords` tensor."
            )
        if loss_points.shape[0] != coords_final.shape[0]:
            raise ValueError(
                "The PPN point predictions and final coordinates must have "
                f"matching rows, got {loss_points.shape[0]} and "
                f"{coords_final.shape[0]}."
            )
        if (
            loss_endpoints is not None
            and loss_endpoints.shape[0] != coords_final.shape[0]
        ):
            raise ValueError(
                "The endpoint predictions and final coordinates must have "
                f"matching rows, got {loss_endpoints.shape[0]} and "
                f"{coords_final.shape[0]}."
            )
        ppn_label_tensor = ppn_label.torch_tensor()

        # Align particle IDs to the unique final-resolution sparse sites. This
        # is a no-op for the usual one-row-per-coordinate input and resolves
        # duplicate input rows when cluster-restricted supervision is enabled.
        aligned_part_labels = None
        if self.restrict_to_clusters:
            if clust_label is None:
                raise ValueError(
                    "When using 'restrict_to_clusters', must provide "
                    "'clust_label' to the PPN loss."
                )
            clust_label_tensor = clust_label.torch_tensor()
            aligned_part_labels = self.align_coordinate_values(
                clust_label_tensor[:, :VALUE_COL],
                clust_label_tensor[:, PART_COL],
                coords_final_tensor[:, :VALUE_COL],
                "particle label",
            )
            aligned_part_labels = TensorBatch(
                aligned_part_labels,
                coords_final.counts,
            )

        closest_list, positive_list = [], []
        offset = 0
        for batch_index in range(batch_size):
            # If there are no label points, there are no positive points
            points_label = ppn_label[batch_index]
            if len(points_label) == 0:
                positive = torch.zeros(
                    coords_final.counts[batch_index],
                    dtype=torch.bool,
                    device=coords_final.device,
                )
                closest = torch.empty_like(positive, dtype=torch.long)
                positive_list.append(positive)
                closest_list.append(closest)
                continue

            # If needed, find which particle instance voxels belong to
            part_labels = None
            if aligned_part_labels is not None:
                part_labels = aligned_part_labels[batch_index]

            # Assign positive/negative labels to each voxel in the image
            points_entry = coords_final[batch_index][:, COORD_COLS] + 0.5
            positive, closest = self.get_ppn_positives(
                points_entry,
                points_label,
                resolution=self.resolution,
                offset=offset,
                labels=part_labels,
            )

            # Append
            positive_list.append(positive)
            closest_list.append(closest)
            offset += len(points_label)

        closest_indices = torch.cat(closest_list, dim=0)
        positives = torch.cat(positive_list, dim=0).long()

        # Downsample the final mask to each PPN layer, apply mask loss
        downsample = sparse.MaxPooling(2, 2, dimension=self.dimension)
        mask_tensor = sparse.SparseTensor(
            positives[:, None].float(),
            coordinates=coords_final_tensor[:, :VALUE_COL],
        )

        dtype, device = ppn_label_tensor.dtype, ppn_label_tensor.device
        mask_losses = torch.zeros(num_layers, dtype=dtype, device=device)
        mask_accuracies = torch.zeros(
            num_layers,
            dtype=dtype,
            device=device,
        )
        mask_label_list = []
        for reverse_index in range(num_layers):
            # Narrow down outputs to this specific layer
            layer_index = num_layers - 1 - reverse_index
            coords_layer = ppn_coords[layer_index]
            scores_layer = ppn_layers[layer_index]
            scores_layer_tensor = scores_layer.torch_tensor()
            coords_layer_tensor = coords_layer.torch_tensor()
            mask_features = self.align_coordinate_values(
                mask_tensor.coordinates,
                mask_tensor.features,
                coords_layer_tensor[:, :VALUE_COL],
                f"PPN mask at layer {layer_index}",
            )
            mask_labels = mask_features.flatten().long()

            if ppn_masks[layer_index].shape[0] != scores_layer.shape[0]:
                raise ValueError(
                    f"PPN mask and score rows differ at layer {layer_index}: "
                    f"{ppn_masks[layer_index].shape[0]} != "
                    f"{scores_layer.shape[0]}."
                )

            # If requested, store the label features in a list
            if self.return_mask_labels:
                mask_label_list.append(
                    TensorBatch(mask_labels[:, None], coords_layer.counts)
                )

            # Compute the mask weights over the whole batch, if requested
            num_points = len(scores_layer_tensor)
            if num_points == 0:
                mask_losses[layer_index] = scores_layer_tensor.sum() * 0.0
            else:
                mask_weight = None
                if self.balance_mask_loss:
                    mask_weight = get_class_weights(
                        mask_labels, 2, self.mask_weighting_mode
                    )

                # Compute the mask loss for this layer, increment
                mask_losses[layer_index] = self.mask_loss_fn(
                    scores_layer_tensor,
                    mask_labels,
                    weight=mask_weight,
                    reduction="mean",
                )

            # Compute the mask accuracy for this layer/batch, increment
            with torch.no_grad():
                mask_predictions = torch.argmax(
                    scores_layer_tensor,
                    dim=1,
                )
                if num_points > 0:
                    mask_accuracies[layer_index] = (
                        mask_predictions == mask_labels
                    ).sum() / num_points

            # Update the mask label for the next iteration
            if layer_index != 0:
                mask_tensor = downsample(mask_tensor)

        # Apply the other losses to the last layer only
        zero = torch.tensor(0.0, dtype=dtype, device=device)
        one = torch.tensor(1.0, dtype=dtype, device=device)
        type_loss, reg_loss, end_loss = zero, zero, zero
        type_acc, end_acc = one, one
        pos_mask = torch.where(positives)[0]
        if len(pos_mask) > 0:
            # Supervise the regression and classification heads at
            # ground-truth-positive sites. Using thresholded predictions here
            # would make their supervision depend on a non-differentiable mask
            # decision and could starve these heads early in training.

            # Closest ppn point label (index) to given positive point
            closest_indices = closest_indices[pos_mask]

            anchors = coords_final_tensor[:, COORD_COLS] + 0.5
            loss_points_tensor = loss_points.torch_tensor()
            pixel_pos = loss_points_tensor[:, PPN_ROFF_COLS] + anchors
            pixel_logits = loss_points_tensor[:, PPN_RTYPE_COLS]

            pixel_pos = pixel_pos[pos_mask]
            pixel_logits = pixel_logits[pos_mask]

            # Type loss
            # Compute type weights over the whole batch, if requested
            type_labels = ppn_label_tensor[:, PPN_LTYPE_COL].long()
            type_weight = None
            if self.balance_type_loss:
                num_classes = pixel_logits.shape[1]
                type_weight = get_class_weights(
                    type_labels, num_classes, self.type_weighting_mode
                )

            # The closest target selected above defines both the regression
            # target and its semantic type. This keeps the two heads aligned
            # when several point labels fall within the positive radius.
            closest_type_labels = type_labels[closest_indices]
            type_loss = self.type_loss_fn(
                pixel_logits, closest_type_labels, weight=type_weight
            )

            # Compute the type accuracy
            with torch.no_grad():
                num_points = len(closest_type_labels)
                type_predictions = torch.argmax(pixel_logits, dim=1)
                type_acc = (type_predictions == closest_type_labels).sum() / num_points

            # Regression loss
            # Compute the regression loss. The offset should point to
            # the closest label point from that voxel
            point_labels = ppn_label_tensor[:, COORD_COLS]
            closest_point_labels = point_labels[closest_indices]
            reg_loss = self.reg_loss_fn(pixel_pos, closest_point_labels)

            # Endpoint loss
            # If the upstream models produced endpoint predictions, apply loss.
            # Narrow the problem down to predictions closest to track points
            track_index = torch.where(closest_type_labels == TRACK_SHP)[0]
            if loss_endpoints is not None and len(track_index) > 0:
                # Get the end point predictions
                end_logits = loss_endpoints.torch_tensor()[pos_mask]
                end_logits = end_logits[track_index]

                # The endpoint class belongs to the same closest track target
                # used by the regression and semantic-type objectives.
                end_labels = ppn_label_tensor[:, PPN_LENDP_COL].long()
                closest_end_labels = end_labels[closest_indices]
                closest_end_labels = closest_end_labels[track_index]
                end_loss = self.end_loss_fn(end_logits, closest_end_labels)

                # Compute the end point classification accuracy
                with torch.no_grad():
                    num_points = len(closest_end_labels)
                    endpoint_predictions = torch.argmax(
                        end_logits,
                        dim=1,
                    )
                    end_acc = (
                        endpoint_predictions == closest_end_labels
                    ).sum() / num_points

        # Combine the losses and accuracies
        mask_loss = torch.mean(mask_losses)
        mask_acc = torch.mean(mask_accuracies)

        loss = (
            self.mask_loss_weight * mask_loss
            + self.type_loss_weight * type_loss
            + self.reg_loss_weight * reg_loss
        )
        accuracy = (mask_acc + type_acc) / 2

        if loss_endpoints is not None:
            loss += self.endpoint_loss_weight * end_loss
            accuracy = (mask_acc + type_acc + end_acc) / 3

        # Prepare the result dictionary
        result: PPNLossOutput = {
            "loss": loss,
            "accuracy": accuracy.item(),
            "mask_loss": mask_loss.item(),
            "mask_accuracy": mask_acc.item(),
            "type_loss": type_loss.item(),
            "type_accuracy": type_acc.item(),
            "reg_loss": reg_loss.item(),
        }

        # Add the endpoint metrics if present
        if loss_endpoints is not None:
            result["classify_endpoints_loss"] = end_loss.item()
            result["classify_endpoints_accuracy"] = end_acc.item()

        # Add the mask loss at each layer
        for layer in range(num_layers):
            result[f"mask_loss_layer_{layer}"] = mask_losses[layer]
            result[f"mask_accuracy_layer_{layer}"] = mask_accuracies[layer]

        # If needed, return the mask labels
        if self.return_mask_labels:
            result["mask_labels"] = mask_label_list[::-1]

        return result


class ExpandAs(torch.nn.Module):
    """Expand foreground scores across a sparse feature width.

    The input must contain two class scores per active site. The foreground
    score in column one is selected, optionally replaced or thresholded, and
    expanded without copying across the requested feature dimension.
    """

    def forward(
        self,
        x: sparse.SparseTensor,
        shape: Sequence[int],
        propagate_all: bool = False,
        use_binary_mask: bool = False,
        score_threshold: float = 0.5,
    ) -> sparse.SparseTensor:
        """Expand foreground scores to a target feature shape.

        Parameters
        ----------
        x : sparse.SparseTensor
            Sparse tensor with a ``(N, 2)`` score matrix.
        shape : sequence of int
            Target feature shape, typically ``(N, C)``.
        propagate_all : bool, default False
            Replace every foreground score with one.
        use_binary_mask : bool, default False
            Convert foreground scores to a boolean mask before expansion.
        score_threshold : float, default 0.5
            Foreground threshold used when ``use_binary_mask`` is true.

        Returns
        -------
        sparse.SparseTensor
            Tensor on the same coordinate map with expanded foreground
            features.

        Raises
        ------
        ValueError
            If the input does not contain exactly two score channels.
        """
        # If all features must be propagated, set all scores to 1.0
        if x.features.shape[1] != 2:
            raise ValueError("Expects a two-score tensor")
        features = x.features[:, 1:]
        if propagate_all:
            features = torch.ones_like(features)

        # Expand the features to the right dimension
        if use_binary_mask:
            features = (features > score_threshold).expand(*shape)
        else:
            features = features.expand(*shape)

        return x.replace_features(features)


class AttentionMask(torch.nn.Module):
    """Align a sparse score mask to a feature tensor and prune rejected sites.

    Sparse addition is used to form the union coordinate map before pruning,
    allowing ``x`` and ``mask`` to contain different active coordinates at the
    same tensor stride.
    """

    def __init__(self, score_threshold: float = 0.5) -> None:
        """Initialize the attention mask.

        Parameters
        ----------
        score_threshold : float, default 0.5
            Mask score above which a sparse site is retained.

        Raises
        ------
        ValueError
            If ``score_threshold`` is outside ``[0, 1]``.
        """
        # Initialize parent class
        super().__init__()

        # Pruning layer
        self.prune = sparse.Pruning()

        # Store parameters
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("`score_threshold` must be between zero and one.")
        self.score_threshold = score_threshold

    def forward(
        self,
        x: sparse.SparseTensor,
        mask: sparse.SparseTensor,
    ) -> sparse.SparseTensor:
        """Prune features using a possibly non-aligned sparse mask.

        Parameters
        ----------
        x : sparse.SparseTensor
            Sparse feature tensor to prune.
        mask : sparse.SparseTensor
            Sparse scalar score mask at the same tensor stride.

        Returns
        -------
        sparse.SparseTensor
            Feature tensor restricted to sites above ``score_threshold``.

        Raises
        ------
        ValueError
            If the feature and mask tensor strides differ.
        """
        if x.tensor_stride != mask.tensor_stride:
            raise ValueError("Expected `x.tensor_stride == mask.tensor_stride`.")

        # Create a mask sparse tensor in x-coordinates
        mask_placeholder = sparse.SparseTensor(
            coordinates=x.coordinates,
            features=x.features.new_zeros(
                (x.features.shape[0], mask.features.shape[1])
            ),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
            source=x,
        )

        aligned_mask = mask_placeholder + mask

        input_placeholder = sparse.SparseTensor(
            coordinates=aligned_mask.coordinates,
            features=x.features.new_zeros(
                (aligned_mask.features.shape[0], x.features.shape[1])
            ),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
            source=x,
        )

        aligned_input = input_placeholder + x

        keep_mask = aligned_mask.features.squeeze(dim=1) > self.score_threshold
        return self.prune(aligned_input, keep_mask)


class MergeConcat(torch.nn.Module):
    """Concatenate sparse tensors after aligning their coordinate unions."""

    def forward(
        self,
        x: sparse.SparseTensor,
        other: sparse.SparseTensor,
    ) -> sparse.SparseTensor:
        """Align and concatenate two sparse tensors.

        Parameters
        ----------
        x : sparse.SparseTensor
            First sparse feature tensor.
        other : sparse.SparseTensor
            Second sparse feature tensor at the same tensor stride.

        Returns
        -------
        sparse.SparseTensor
            Tensor defined on the coordinate union with concatenated features.

        Raises
        ------
        ValueError
            If the tensor strides differ.
        """
        if x.tensor_stride != other.tensor_stride:
            raise ValueError("Expected `x.tensor_stride == other.tensor_stride`.")
        if torch.equal(x.coordinates, other.coordinates):
            return x.replace_features(torch.cat((other.features, x.features), dim=1))

        # Create a placeholder tensor with x.coordinates coordinates
        other_placeholder = sparse.SparseTensor(
            coordinates=x.coordinates,
            features=x.features.new_zeros(
                (x.features.shape[0], other.features.shape[1])
            ),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
            source=x,
        )

        # Set placeholder values with other.features features by performing
        # sparse tensor addition.
        aligned_other = other_placeholder + other

        # Same procedure, but with other
        input_placeholder = sparse.SparseTensor(
            coordinates=aligned_other.coordinates,
            features=x.features.new_zeros(
                (
                    aligned_other.features.shape[0],
                    x.features.shape[1],
                )
            ),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
            source=x,
        )

        aligned_input = input_placeholder + x

        # Now x and other share the same coordinates and shape. Keep one
        # coordinate-map key and concatenate only the aligned feature arrays;
        # independently created union maps are not interchangeable in sparse
        # backends such as MinkowskiEngine.
        if not torch.equal(aligned_other.coordinates, aligned_input.coordinates):
            raise RuntimeError("Failed to align sparse tensors for concatenation.")
        return aligned_input.replace_features(
            torch.cat((aligned_other.features, aligned_input.features), dim=1)
        )


class GhostMask(sparse.Network):
    """Downsample a ghost mask to match a sparse feature resolution.

    Repeated stride-two max pooling preserves a site as non-ghost when any
    contributing finer-resolution site is non-ghost.
    """

    def __init__(self, dimension: int = 3) -> None:
        """Initialize the mask downsampler.

        Parameters
        ----------
        dimension : int, default 3
            Number of spatial dimensions.

        Raises
        ------
        ValueError
            If ``dimension`` is not positive.
        """
        # Initialize parent class
        super().__init__(dimension)

        # Initialize the downsampler
        if dimension < 1:
            raise ValueError(f"`dimension` must be positive, got {dimension}.")
        self.downsample = sparse.MaxPooling(2, 2, dimension=dimension)

        # Set the layer in evaluation mode (no gradients)
        self.eval()

    def forward(
        self,
        ghost_mask: sparse.SparseTensor,
        premask_tensor: sparse.SparseTensor,
    ) -> sparse.SparseTensor:
        """Downsample a ghost mask to the target tensor stride.

        Parameters
        ----------
        ghost_mask : sparse.SparseTensor
            Mask defined at tensor stride one.
        premask_tensor : sparse.SparseTensor
            Feature tensor whose stride determines the target resolution.

        Returns
        -------
        sparse.SparseTensor
            Ghost mask downsampled to ``premask_tensor.tensor_stride``.

        Raises
        ------
        ValueError
            If the target stride is not a positive power of two.
        """
        with torch.no_grad():
            factor = premask_tensor.tensor_stride[0]
            if factor < 1 or factor & (factor - 1):
                raise ValueError(
                    "Ghost-mask downsampling requires a positive power-of-two "
                    f"tensor stride, got {factor}."
                )
            while factor > 1:
                ghost_mask = self.downsample(ghost_mask)
                factor //= 2

            return ghost_mask
