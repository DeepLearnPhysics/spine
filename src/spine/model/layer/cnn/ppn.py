from collections import Counter

import MinkowskiEngine as ME
import MinkowskiFunctional as MF
import numpy as np
import torch
import torch.nn as nn

from spine.data import TensorBatch
from spine.utils.globals import (
    COORD_COLS,
    GHOST_SHP,
    PART_COL,
    PPN_LENDP_COL,
    PPN_LPART_COL,
    PPN_LTYPE_COL,
    PPN_ROFF_COLS,
    PPN_RPOS_COLS,
    PPN_RTYPE_COLS,
    SHAPE_COL,
    TRACK_SHP,
    VALUE_COL,
)
from spine.utils.logger import logger
from spine.utils.torch.scripts import cdist_fast
from spine.utils.weighting import get_class_weights

from .act_norm import act_factory
from .blocks import ASPP, SPP, ResNetBlock
from .configuration import setup_cnn_configuration

__all__ = ["PPN", "PPNLoss"]


class PPN(torch.nn.Module):
    """Point Proposal Network (PPN).

    It requires a UResNet network as a backbone. Typical configuration:

    .. code-block:: yaml

        model:
          name: uresnet_ppn_chain
          modules:
            uresnet:
              # Your uresnet config here
            ppn:
              # Your ppn config here

    Configuration
    -------------
    dimension: int, default 3
    num_input: int, default 1
    allow_bias: bool, default False
    spatial_size: int, default 512
    leakiness: float, default 0.33
    activation: dict
        For activation function, defaults to `{'name': 'lrelu', 'args': {}}`
    norm_layer: dict
        For normalization function, defaults to `{'name': 'batch_norm', 'args': {}}`

    depth: int, default 5
        Depth of UResNet, also corresponds to how many times we down/upsample.
    filters: int, default 16
        Number of filters in the first convolution of UResNet.
        Will increase linearly with depth.
    reps: int, default 2
        Convolution block repetition factor
    input_kernel: int, default 3
        Receptive field size for very first convolution after input layer.
    num_classes: int, default 5
    mask_score_threshold: float, default 0.5
    classify_endpoints: bool, default False
        Enable classification of points into start vs end points.
    ppn_resolution: float, default 1.0
    ghost: bool, default False
    use_true_ghost_mask: bool, default False
    mask_loss_name: str, default 'BCE'
        Can be 'BCE' or 'LogDice'

    Output
    ------
    ppn_points: torch.Tensor
        Contains  X, Y, Z predictions, semantic class prediction logits, and prob score
    ppn_masks: list of torch.Tensor
        Binary masks at various spatial scales of PPN predictions (voxel-wise score > some threshold)
    ppn_coords: list of torch.Tensor
        List of XYZ coordinates at various spatial scales.
    ppn_layers: list of torch.Tensor
        List of score features at various spatial scales.
    ppn_output_coords: torch.Tensor
        XYZ coordinates tensor at the very last layer of PPN (initial spatial scale)
    ppn_classify_endpoints: torch.Tensor
        Logits for end/start point classification.

    See Also
    --------
    :class:`PPNLoss`, :class:`spine.model.uresnet_ppn_chain`
    """

    def __init__(self, uresnet, ppn, uresnet_loss=None, ppn_loss=None):
        """Initializes the standalone PPN network.

        Parameters
        ----------
        uresnet : dict
            Dictionary of the backbone uresnet configuration
        ppn : dict
            Model configuration
        uresnet_loss : dict, optional
            UResNet loss configuration
        ppn_loss : dict, optional
            PPN loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize the backbone configuration
        self.process_backbone_config(**uresnet)

        # Process the PPN specific parameters
        self.process_model_config(**ppn)

        # Inialize the PPN model
        self.initialize_model()

    def process_backbone_config(self, num_classes, ghost=False, **backbone):
        """Initialize the underlying UResNet model configuration.

        Parameters
        ----------
        num_classes : int
            Number of classes to classify the voxels as
        ghost : bool, default False
            Whether to add a deghosting step in the classification model
        **backbone : dict
            UResNet backbone configuration
        """
        # Store the semantic segmentation configuration
        self.num_classes = num_classes
        self.ghost = ghost

        # Initialize the UResNet backbone, store the relevant parameters
        setup_cnn_configuration(self, **backbone)

    def process_model_config(
        self,
        mask_score_threshold=0.5,
        classify_endpoints=False,
        propagate_all=False,
        use_binary_mask=False,
        ghost=False,
        use_true_ghost_mask=False,
    ):
        """Process the PPN-specific parameters.

        Parameters
        ----------
        mask_score_threshold : float, default 0.5
            Predicted score above which a pixel is considered positive
        classify_endpoints : bool, default False
            Whether or not to predict which point is the start/end for a track
        propagate_all : bool, default False
            If `True`, the mask will not be applied at every PPN layer
        use_binary_mask : bool, default False
            If `True`, converts the features to a binary mask based on score
        ghost : bool, default False
            Whether or not the input contains ghosts
        use_true_ghost_mask : bool, default False
            If `True`, ghost labels to deghost the tensor
        """
        # Register parameters
        self.mask_score_threshold = mask_score_threshold
        self.classify_endpoints = classify_endpoints
        self.propagate_all = propagate_all
        self.use_binary_mask = use_binary_mask
        self.ghost = ghost
        self.use_true_ghost_mask = use_true_ghost_mask

    def initialize_model(self):
        """Initializes the PPN-specific decoder."""
        # Initialize the decoding blocks
        self.decoding_block = []
        self.decoding_conv = []
        self.ppn_masks = nn.ModuleList()
        for i in range(self.depth - 2, -1, -1):
            m = []
            m.append(ME.MinkowskiBatchNorm(self.num_planes[i + 1]))
            m.append(act_factory(self.act_cfg))
            m.append(
                ME.MinkowskiConvolutionTranspose(
                    in_channels=self.num_planes[i + 1],
                    out_channels=self.num_planes[i],
                    kernel_size=2,
                    stride=2,
                    dimension=self.dim,
                )
            )
            m = nn.Sequential(*m)
            self.decoding_conv.append(m)

            m = []
            for j in range(self.reps):
                m.append(
                    ResNetBlock(
                        self.num_planes[i] * (2 if j == 0 else 1),
                        self.num_planes[i],
                        dimension=self.dim,
                        activation=self.act_cfg,
                    )
                )
            m = nn.Sequential(*m)
            self.decoding_block.append(m)
            self.ppn_masks.append(ME.MinkowskiLinear(self.num_planes[i], 2))

        self.decoding_block = nn.Sequential(*self.decoding_block)
        self.decoding_conv = nn.Sequential(*self.decoding_conv)

        # Expands the scores to the appropriate feature shape
        self.expand_as = ExpandAs()

        # Final ResNet block at the original image size
        num_output = self.num_planes[0]
        self.final_block = ResNetBlock(
            num_output, num_output, dimension=self.dim, activation=self.act_cfg
        )

        # Final linear layer for positional regression (dimension size)
        self.ppn_pixel_pos = ME.MinkowskiLinear(num_output, self.dim)

        # Final convolution layer for type classification
        self.ppn_type = ME.MinkowskiLinear(num_output, self.num_classes)

        # Final convolution layer for endpoint prediction
        if self.classify_endpoints:
            self.ppn_endpoint = ME.MinkowskiLinear(num_output, 2)

        # Ghost point removal tools
        if self.ghost:
            logger.debug("Ghost Masking is enabled for MinkPPN.")
            self.masker = AttentionMask()
            self.merge_concat = MergeConcat()
            self.ghost_mask = MinkGhostMask(self.dim)

    def forward(self, final_tensor, decoder_tensors, ghost=None, seg_label=None):
        """Compute the PPN loss for a batch of data.

        The PPN loss comprises three components:
        - Regression loss: position of the point of interest within a pixel
        - Type: type of point of interest
        - Mask: whether or not a pixel is within some distance of a point

        Parameters
        ----------
        final_tensor : TensorBatch
            Feature tensors at the deepest layer of the backbone UResNet
        decoder_tensors : List[TensorBatch]
            Feature tensors of each of the decoding blocks
        ghost : TensorBatch, optional
            Logits of the ghost predictions of the backbone UResNet
        seg_label : TensorBatch, optional
            Segmentation label tensor

        Returns
        -------
        dict
             Dictionary of outputs
        """
        # Get the list of decoder feature maps
        decoder_feature_maps = []
        if self.ghost:
            # If there are ghosts, must downsample the ghost label/prediction
            # and apply it to each decoder feature map
            with torch.no_grad():
                if self.use_true_ghost_mask:
                    # If using the true ghost mask, use the label tensor
                    assert seg_label is not None, (
                        "If `use_true_ghost_mask` is set to `True`, must "
                        "provide the `seg_label` tensor."
                    )

                    labels = seg_label.tensor
                    assert labels.shape[0] == decoder_tensors[-1].tensor.shape[0], (
                        "The label tensor length must match that "
                        "of the last UResNet layer"
                    )

                    ghost_coords = labels[:, :VALUE_COL]
                    ghost_mask_tensor = labels[:, SHAPE_COL] < GHOST_SHP
                else:
                    # If using predictions, convert the ghost scores to a mask
                    ghost_coords = ghost.tensor.C
                    ghost_mask_tensor = 1.0 - torch.argmax(
                        ghost.tensor.F, dim=1, keepdim=True
                    )

                ghost_mask = ME.SparseTensor(
                    ghost_mask_tensor, coordinates=ghost_coords
                )

            # Downsample stride 1 ghost mask to all intermediate decoder layers
            for t in decoder_tensors[::-1]:
                scaled_ghost_mask = self.ghost_mask(ghost_mask, t.tensor)
                nonghost_tensor = self.masker(t.tensor, scaled_ghost_mask)
                decoder_feature_maps.append(nonghost_tensor)

            decoder_feature_maps = decoder_feature_maps[::-1]

        else:
            decoder_feature_maps = [t.tensor for t in decoder_tensors]

        # Loop over the PPN decoding path
        ppn_masks, ppn_layers, ppn_coords = [], [], []
        x = final_tensor.tensor
        for i, layer in enumerate(self.decoding_conv):
            # Pass the previous features through the decoding convolution
            x = layer(x)

            # Merge with the UesNet decoding features
            decoder_tensor = decoder_feature_maps[i]
            if self.ghost:
                x = self.merge_concat(decoder_tensor, x)
            else:
                x = ME.cat(decoder_tensor, x)

            # Apply the decoding block, linear layer and sigmoid function
            x = self.decoding_block[i](x)
            scores = self.ppn_masks[i](x)
            softmax = MF.softmax(scores, dim=1)
            mask = softmax.F[:, 1:] > self.mask_score_threshold

            # Store the coordinates, raw score logits and score mask
            counts = decoder_tensors[i].counts
            ppn_coords.append(
                TensorBatch(scores.C, counts, has_batch_col=True, coord_cols=COORD_COLS)
            )
            ppn_layers.append(TensorBatch(scores.F, counts))
            ppn_masks.append(TensorBatch(mask, counts))

            # Expand the score mask
            s_expanded = self.expand_as(
                softmax,
                x.F.shape,
                propagate_all=self.propagate_all,
                use_binary_mask=self.use_binary_mask,
            )

            # Scale the out of this layer using the score mask
            x = x * s_expanded.detach()

        # Output set of coordinates (should match the last decoder tensor)
        assert x.C.shape[0] == decoder_tensors[-1].tensor.shape[0], (
            "The output of the last PPN layer should be consistent "
            "with the length of the last UResNet decoder layer"
        )
        final_counts = decoder_tensors[-1].counts
        ppn_output_coords = TensorBatch(
            x.C, final_counts, has_batch_col=True, coord_cols=COORD_COLS
        )

        # Pass the final PPN tensor through the individual predictions, combine
        x = self.final_block(x)
        pixel_pos = self.ppn_pixel_pos(x)
        ppn_type = self.ppn_type(x)
        if self.classify_endpoints:
            ppn_endpoint = self.ppn_endpoint(x)

        # X, Y, Z, logits, and prob score
        ppn_points = TensorBatch(
            torch.cat([pixel_pos.F, ppn_type.F, ppn_layers[-1].tensor], dim=1),
            final_counts,
        )

        result = {
            "ppn_points": ppn_points,
            "ppn_masks": ppn_masks,
            "ppn_layers": ppn_layers,
            "ppn_coords": ppn_coords,
            "ppn_output_coords": ppn_output_coords,
        }
        if self.classify_endpoints:
            result["ppn_classify_endpoints"] = TensorBatch(ppn_endpoint.F, final_counts)

        return result


class PPNLoss(torch.nn.modules.loss._Loss):
    """Loss function for PPN.

    Output
    ------
    reg_loss : float
        Distance loss
    mask_loss : float
        Binary voxel-wise prediction loss (is there an object of interest or not)
    classify_endpoints_loss : float
        Endpoint classification loss
    type_loss : float
        Semantic prediction loss
    output_mask_accuracy: float
        Binary voxel-wise prediction accuracy in the last layer
    type_accuracy : float
        Semantic prediction accuracy
    classify_endpoints_accuracy : float
        Endpoint classification accuracy

    See Also
    --------
    PPN, spine.model.uresnet_ppn_chain
    """

    def __init__(self, uresnet, ppn, ppn_loss, uresnet_loss=None):
        """Initializes the standalone PPN loss.

        Parameters
        ----------
        uresnet : dict
            Dictionary of the backbone uresnet configuration
        ppn : dict
            Model configuration
        uresnet_loss : dict, optional
            UResNet loss configuration
        ppn_loss : dict, optional
            PPN loss configuration
        """
        # Initialize the parent class
        super().__init__()

        # Initialize what we need from the backbone configuration
        self.process_backbone_config(**uresnet)

        # Initialize the loss configuration
        self.process_loss_config(**ppn_loss)

    def process_backbone_config(self, depth, **kwargs):
        """Process the parameters of the backbone model needed for in the loss.

        Parameters
        ----------
        depth : int
            Depth of the UResNet
        **kwargs : dict, optional
            Leftover model configuration (no need in the loss)
        """
        # Store the semantic segmentation configuration
        self.depth = depth

    def process_loss_config(
        self,
        mask_loss="CE",
        resolution=5.0,
        point_classes=None,
        balance_mask_loss=True,
        mask_weighting_mode="const",
        balance_type_loss=True,
        type_weighting_mode="const",
        reg_loss_weight=1.0,
        type_loss_weight=1.0,
        mask_loss_weight=1.0,
        endpoint_loss_weight=1.0,
        return_mask_labels=False,
        restrict_to_clusters=False,
    ):
        """Process the loss function parameters.

        Parameters
        ----------
        resolution : float, default 5.
            Distance from a label point in pixels within which a voxel is
            considered positive (pixel of interest)
        mask_loss : str, default 'CE'
            Name of the loss function to use
        point_classes : Union[int, list], optional
            If provided, restricts the loss to points of (a) certain shape(s)
        balance_mask_loss : bool, default True
            Apply class-weights to the mask loss
        mask_weighting_mode : str, default 'const'
            Method for class-weighting the mask loss
        balance_type_loss : bool, default True
            Apply class-weights to the type loss
        type_weighting_method : str, default 'const'
            Method class-weighting the type loss
        reg_loss_weight : float, default 1.
            Relative weight to apply to the regression loss
        type_loss_weight : float, default 1.
            Relative weight to apply to the point type loss
        mask_loss_weight : float, default 1.
            Relative weight to apply to the mask loss
        endpoint_loss_weight : float, default 1.
            Relative weight to apply to the endpoint classification
        return_mask_labels : bool, default False
            If `True`, returns the masks used to compute the mask loss
        restrict_to_clusters : bool, default False
            If `True`, when computing the positive labels for PPN, it will only
            look for points that are close to a given PPN label point with the
            same particle id.
        """
        # Store the loss parameters
        self.resolution = resolution
        self.balance_mask_loss = balance_mask_loss
        self.mask_weighting_mode = mask_weighting_mode
        self.balance_type_loss = balance_type_loss
        self.type_weighting_mode = type_weighting_mode
        self.reg_loss_weight = reg_loss_weight
        self.type_loss_weight = type_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.endpoint_loss_weight = endpoint_loss_weight
        self.return_mask_labels = return_mask_labels
        self.restrict_to_clusters = restrict_to_clusters
        self.point_classes = point_classes
        if point_classes is not None and isinstance(point_classes, int):
            self.point_classes = [point_classes]

        # Instantiate the regression loss function
        self.reg_loss_fn = torch.nn.MSELoss(reduction="mean")

        # Instantiate the point type loss function
        self.type_loss_fn = torch.nn.functional.cross_entropy

        # Instantiate the end point tyepe loss function
        self.end_loss_fn = torch.nn.functional.cross_entropy

        # Instantiate the mask loss function
        self.mask_loss = mask_loss
        if mask_loss == "CE":
            self.mask_loss_fn = torch.nn.functional.cross_entropy
        else:
            raise ValueError(f"Mask loss name not recognized: {mask_loss}")

    @staticmethod
    def get_ppn_positives(
        coords: torch.Tensor,
        ppn_labels: torch.Tensor,
        resolution: float,
        offset: int,
        labels: torch.Tensor = None,
    ):
        """Get ppn positive label mask.

        If the voxel `labels` are provided, they are used to restrict the mask
        applied to voxels within some distance of label points associated with
        the correct particle instance, not any particle instance.

        Parameters
        ----------
        coords : torch.Tensor
            (N, 3) 3D coordinates of the image voxels
        ppn_label : torch.Tensor
            (N, 1 + D + N_l) Tensor of PPN labels for the batch
        resolution : float
            Distance from a label point in pixels within which a voxel is
            considered positive (pixel of interest)
        offset : int
            The index offset needed to transform within-batch index to
            global (image) index.
        labels : torch.Tensor, optional
            (N) tensor of the particle id label for each voxel

        Returns
        -------
        positives : torch.Tensor
            (N) tensor of the positive label mask
        closests : torch.Tensor
            (N) tensor of the closest label point index
        """
        # Detach this process from the computation graph (mask not learnable)
        with torch.no_grad():
            # Compute the distance from the PPN labels to all the image points
            dist_mat = cdist_fast(ppn_labels[:, COORD_COLS], coords)

            # Mask out particle voxels for which the particle ID disagrees
            if labels is not None:
                bad_mask = ppn_labels[:, [PPN_LPART_COL]] != labels
                dist_mat[bad_mask] = torch.inf

            # Generate a positive mask for all particle voxels within some
            # distance of their label points
            positives = (dist_mat < resolution).any(dim=0)

            # Assign the closest label point to each postive particle voxel
            pos_index = torch.where(positives)[0]
            closests = torch.full(
                (len(coords),), -1, dtype=torch.long, device=coords.device
            )
            closests[pos_index] = offset + torch.argmin(dist_mat[:, pos_index], dim=0)

            return positives, closests

    def forward(
        self,
        ppn_label,
        ppn_points,
        ppn_masks,
        ppn_layers,
        ppn_coords,
        ppn_output_coords,
        ppn_classify_endpoints=None,
        clust_label=None,
        **kwargs,
    ):
        """Computes the three PPN losses.

        Parameters
        ----------
        ppn_label : TensorBatch
            (N, 1 + D + N_l) Tensor of PPN labels for the batch
        ppn_points : TensorBatch
            Complete PPN predictions at the last layer
        ppn_masks : List[TensorBatch]
            Binary mask at each layer of the PPN
        ppn_layers : List[TensorBatch]
            Output logits at each layer of the PPN
        ppn_coords : List[TensorBatch]
            Set of coordinates at each layer of the PPN
        ppn_output_coords : TensorBatch
            Set of coordinates at the very last layer of the PPN
        ppn_classify_endpoins : TensorBatch, optional
            Set of logits associated with end point classification
        clust_label : TensorBatch, optional
            (N, 1 + D + N_c) Tensor of cluster labels
            - N_c is is the number of cluster labels
        **kwargs : dict, optional
            Other outputs of the upstream model which are not relevant here

        Returns
        -------
        dict
            Dictionary of accuracies and losses
        """
        # Initialize the basics
        num_layers = len(ppn_layers)
        batch_size = ppn_label.batch_size

        # If requested, narrow down the list of label points
        if self.point_classes is not None:
            assert len(
                self.point_classes
            ), "Should provide at least one class to include in the loss"
            ppn_label_list = []
            for b, label_tensor in enumerate(ppn_label.split()):
                labels = label_tensor[:, PPN_LTYPE_COL]
                mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
                for c in self.point_classes:
                    mask |= labels == c
                valid_index = torch.where(mask)[0]
                ppn_label_list[b] = label_tensor[valid_index]

            ppn_label = TensorBatch.from_list(ppn_label_list)

        # Compute the label mask for the final PPN layer. Record which
        # label point is closest to each image voxel (defines label for it)
        coords_final = ppn_coords[-1]
        closest_list, positive_list = [], []
        part_labels = None
        offset = 0
        for b in range(batch_size):
            # If there are no label points, there are no positive points
            points_label = ppn_label[b]
            if not len(points_label):
                positive = torch.zeros(
                    coords_final.counts[b], dtype=torch.bool, device=coords_final.device
                )
                closest = torch.empty_like(positive, dtype=torch.long)
                positive_list.append(positive)
                closest_list.append(closest)
                continue

            # If needed, find which particle instance voxels belong to
            if self.restrict_to_clusters:
                assert clust_label is not None, (
                    "When using 'restrict_to_clusters', must provide "
                    "'clust_label' to the PPN loss."
                )
                part_labels = clust_label[b][:, PART_COL]

            # Assign positive/negative labels to each voxel in the image
            points_entry = coords_final[b][:, COORD_COLS] + 0.5
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

        closests = torch.cat(closest_list, dim=0)
        positives = torch.cat(positive_list, dim=0).long()

        # Downsample the final mask to each PPN layer, apply mask loss
        downsample = ME.MinkowskiMaxPooling(2, 2, dimension=3)  # TODO
        mask_tensor = ME.SparseTensor(
            positives[:, None].float(), coordinates=coords_final.tensor[:, :VALUE_COL]
        )

        dtype, device = ppn_label.tensor.dtype, ppn_label.tensor.device
        mask_losses = torch.zeros(num_layers, dtype=dtype, device=device)
        mask_accs = torch.zeros(num_layers, dtype=dtype, device=device)
        mask_label_list = []
        for i in range(0, num_layers):
            # Narrow down outputs to this specific layer
            layer = num_layers - 1 - i
            coords_layer = ppn_coords[layer]
            scores_layer = ppn_layers[layer]
            mask_labels = mask_tensor.F.flatten().long()

            # If requested, store the label features in a list
            if self.return_mask_labels:
                mask_label_list.append(
                    TensorBatch(mask_labels[:, None], coords_layer.counts)
                )

            # Compute the mask weights over the whole batch, if requested
            mask_weight = None
            if self.balance_mask_loss:
                mask_weight = get_class_weights(
                    mask_labels, 2, self.mask_weighting_mode
                )

            # Compute the mask loss for this layer, increment
            mask_losses[layer] = self.mask_loss_fn(
                scores_layer.tensor, mask_labels, weight=mask_weight, reduction="mean"
            )

            # Compute the mask accuracy for this layer/batch, increment
            with torch.no_grad():
                num_points = len(scores_layer.tensor)
                mask_pred = torch.argmax(scores_layer.tensor, dim=1)
                mask_accs[layer] = (mask_pred == mask_labels).sum() / num_points

            # Update the mask label for the next iteration
            if layer != 0:
                mask_tensor = downsample(mask_tensor)

        # Apply the other losses to the last layer only
        zero = torch.tensor(0.0, dtype=ppn_label.dtype, device=ppn_label.device)
        one = torch.tensor(1.0, dtype=ppn_label.dtype, device=ppn_label.device)
        type_loss, reg_loss, end_loss = zero, zero, zero
        type_acc, end_acc = one, one
        pos_mask = torch.where(positives)[0]
        if len(pos_mask):
            # Narrow the loss down to the true positive pixels
            # TODO: should this be predicted positive pixels?

            # Closest ppn point label (index) to given positive point
            closests = closests[pos_mask]

            anchors = coords_final.tensor[:, COORD_COLS] + 0.5
            pixel_pos = ppn_points.tensor[:, PPN_ROFF_COLS] + anchors
            pixel_scores = ppn_points.tensor[:, PPN_RPOS_COLS]
            pixel_logits = ppn_points.tensor[:, PPN_RTYPE_COLS]

            pixel_pos = pixel_pos[pos_mask]
            pixel_scores = pixel_scores[pos_mask]
            pixel_logits = pixel_logits[pos_mask]

            ########## Type loss ##########
            # Compute type weights over the whole batch, if requested
            type_labels = ppn_label.tensor[:, PPN_LTYPE_COL].long()
            type_weight = None
            if self.balance_type_loss:
                num_classes = pixel_logits.shape[1]
                type_weight = get_class_weights(
                    type_labels, num_classes, self.type_weighting_mode
                )

            # Compute the type loss
            # TODO: deal with having multiple valid labels (track/shower/etc.)
            closest_type_labels = type_labels[closests]
            type_loss = self.type_loss_fn(
                pixel_logits, closest_type_labels, weight=type_weight
            )

            # Compute the type accuracy
            with torch.no_grad():
                num_points = len(closest_type_labels)
                type_pred = torch.argmax(pixel_logits, dim=1)
                type_acc = (type_pred == closest_type_labels).sum() / num_points

            ########## Regression loss ##########
            # Compute the regression loss. The offset should point to
            # the closest label point from that voxel
            point_labels = ppn_label.tensor[:, COORD_COLS]
            closest_point_labels = point_labels[closests]
            reg_loss = self.reg_loss_fn(pixel_pos, closest_point_labels)

            ########## End points loss ##########
            # If the upstream models produced endpoint predictions, apply loss.
            # Narrow the problem down to predictions closest to track points
            track_index = torch.where(closest_type_labels == TRACK_SHP)[0]
            if ppn_classify_endpoints and len(track_index):
                # Get the end point predictions
                end_logits = ppn_classify_endpoints.tensor[pos_mask]
                end_logits = end_logits[track_index]

                # Compute the end point classification loss
                # TODO: deal with having multiple valid labels (start/end)
                end_labels = ppn_label.tensor[:, PPN_LENDP_COL].long()
                closest_end_labels = end_labels[closests]
                closest_end_labels = closest_end_labels[track_index]
                end_loss = self.end_loss_fn(end_logits, closest_end_labels)

                # Compute the end point classification accuracy
                with torch.no_grad():
                    num_points = len(closest_end_labels)
                    end_pred = torch.argmax(end_logits, dim=1)
                    end_acc = (end_pred == closest_end_labels).sum() / num_points

        # Combine the losses and accuracies
        mask_loss = torch.mean(mask_losses)
        mask_acc = torch.mean(mask_accs)

        loss = (
            self.mask_loss_weight * mask_loss
            + self.type_loss_weight * type_loss
            + self.reg_loss_weight * reg_loss
        )
        accuracy = (mask_acc + type_acc) / 2

        if ppn_classify_endpoints is not None:
            loss += self.endpoint_loss_weight * end_loss
            accurary = accuracy * 2.0 + end_acc

        # Prepare the result dictionary
        result = {
            "loss": loss,
            "accuracy": accuracy.item(),
            "mask_loss": mask_loss.item(),
            "mask_accuracy": mask_acc.item(),
            "type_loss": type_loss.item(),
            "type_accuracy": type_acc.item(),
            "reg_loss": reg_loss.item(),
        }

        # Add the endpoint metrics if present
        if ppn_classify_endpoints is not None:
            result["classify_endpoints_loss"] = end_loss.item()
            result["classify_endpoints_accuracy"] = end_acc.item()

        # Add the mask loss at each layer
        for layer in range(num_layers):
            result[f"mask_loss_layer_{layer}"] = mask_losses[layer]
            result[f"mask_accuracy_layer_{layer}"] = mask_accs[layer]

        # If needed, return the mask labels
        if self.return_mask_labels:
            result["mask_labels"] = mask_label_list[::-1]

        return result


class ExpandAs(nn.Module):
    """Expands a one dimensional feature tensor to a higher dimension.

    Given a sparse tensor with one dimensional features, expand the
    feature map to a given shape and return a newly constructed
    ME.SparseTensor. This is used to expand a score array and apply
    it to the entire feature tensor of the the input.
    """

    def forward(
        self, x, shape, propagate_all=False, use_binary_mask=False, score_threshold=0.5
    ):
        """Expand a tensor to the appropriate shape

        Parameters
        ----------
        x : torch.Tensor
            (N, 2) Input tensor
        shape : ntuple
            (N, X) Shape to expand the mask to
        propagate_all : bool, default False
            If `True`, sets all features to 1.
        use_binary_mask : bool, default False
            If `True`, sets all features to either 0 or 1
        scrore_threshold : float, default 0.5
            If `use_binary_mask == True`, sets the threshold above which
            the feature is 1.0 and below which it is 0.0
        """
        # If all features must be propagated, set all scores to 1.0
        assert x.F.shape[1] == 2, "Expects a two-score tensor"
        features = x.F[:, 1:]
        if propagate_all:
            features[...] = 1.0

        # Expand the features to the right dimension
        if use_binary_mask:
            features = (features > score_threshold).expand(*shape)
        else:
            features = features.expand(*shape)

        return ME.SparseTensor(
            features=features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )


class AttentionMask(torch.nn.Module):
    """Returns a masked tensor of x according to mask, where the number of
    coordinates between x and mask differ.
    """

    def __init__(self, score_threshold=0.5):
        """Initialize attention mask.

        Parameters
        ----------
        score_threshold : float, default 0.5
            Score above which a voxel is considered positive
        """
        # Initialize parent class
        super().__init__()

        # Pruning layer
        self.prune = ME.MinkowskiPruning()

        # Store parameters
        self.score_threshold = score_threshold

    def forward(self, x, mask):
        """Prune the input data.

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor
        mask : ME.SparseTensor
            Mask to apply
        """
        assert x.tensor_stride == mask.tensor_stride

        device = x.F.device
        # Create a mask sparse tensor in x-coordinates
        x0 = ME.SparseTensor(
            coordinates=x.C,
            features=torch.zeros(x.F.shape[0], mask.F.shape[1]).to(device),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
        )

        mask_in_xcoords = x0 + mask

        x_expanded = ME.SparseTensor(
            coordinates=mask_in_xcoords.C,
            features=torch.zeros(mask_in_xcoords.F.shape[0], x.F.shape[1]).to(device),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
        )

        x_expanded = x_expanded + x

        target = mask_in_xcoords.F.int().bool().squeeze()
        x_pruned = self.prune(x_expanded, target)
        return x_pruned


class MergeConcat(torch.nn.Module):
    """Merge one sparse tensor with another."""

    def forward(self, x, other):
        """Merge two sparse tensors

        Parameters
        ----------
        x : ME.SparseTensor
            Input sparse tensor
        other : ME.SparseTensor
            Other sparse tensor to merge

        Returns
        -------
        ME.SparseTensor
            Concatenated sparse tensor
        """
        assert x.tensor_stride == other.tensor_stride
        device = x.F.device

        # Create a placeholder tensor with x.C coordinates
        x0 = ME.SparseTensor(
            coordinates=x.C,
            features=torch.zeros(x.F.shape[0], other.F.shape[1]).to(device),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
        )

        # Set placeholder values with other.F features by performing
        # sparse tensor addition.
        x1 = x0 + other

        # Same procedure, but with other
        x_expanded = ME.SparseTensor(
            coordinates=x1.C,
            features=torch.zeros(x1.F.shape[0], x.F.shape[1]).to(device),
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride,
        )

        x2 = x_expanded + x

        # Now x and other share the same coordinates and shape
        concated = ME.cat(x1, x2)
        return concated


class MinkGhostMask(torch.nn.Module):
    """Ghost mask downsampler.

    Downsamples the ghost mask and prunes a tensor with current
    ghost mask to obtain nonghost tensor and the new ghost mask.
    """

    def __init__(self):
        """Initialize the mask downsampler."""
        # Initialize parent class
        super().__init__()

        # Initialize the downsampler
        self.downsample = ME.MinkowskiMaxPooling(2, 2, dimension=3)

        # Set the layer in evaluation mode (no gradients)
        self.eval()

    def forward(self, ghost_mask, premask_tensor):
        """Applies mask and downsamples it for the next layer.

        Parameters
        ----------
        ghost_mask : ME.SparseTensor
            Current resolution ghost mask
        premask_tensor : ME.SparseTensor
            Current resolution feature map to be pruned

        Returns
        -------
        downsampled_mask : ME.SparseTensor)
            2x2 downsampled ghost mask
        downsampled_tensor : ME.SparseTensor
            2x2 downsampled feature map
        """
        # assert ghost_mask.shape[0] == premask_tensor.shape[0]
        with torch.no_grad():
            factor = premask_tensor.tensor_stride[0]
            for i in range(np.log2(factor).astype(int)):
                ghost_mask = self.downsample(ghost_mask)

            return ghost_mask
