"""Task losses for modular whole-image and object-image predictions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch

from spine.constants import PID_MASSES
from spine.data import ClusterLabelBatch, IndexBatch, TensorBatch

from ..common.factories import loss_fn_factory
from ..common.quality import ClusterQualityFilter

__all__ = [
    "ImageTaskLoss",
    "ImageClassificationLoss",
    "ImageRegressionLoss",
    "ImageLoss",
]


class ImageTaskLoss(torch.nn.Module, ABC):
    """Common interface for one named image prediction objective."""

    label: str
    target: str | int | None
    target_reduction: str

    @abstractmethod
    def forward(
        self,
        labels: ClusterLabelBatch | TensorBatch | Sequence[Any] | torch.Tensor,
        objects: IndexBatch,
        prediction: TensorBatch,
    ) -> dict[str, Any]:
        """Evaluate one image task."""


def _head_sizes(heads: Mapping[str, Any]) -> dict[str, int]:
    """Extract configured output widths without constructing model heads."""
    # Normalize the shorthand and full head configurations to output widths
    result = {}
    for label, config in heads.items():
        if isinstance(config, int):
            result[label] = config
        elif isinstance(config, Mapping) and "out_channels" in config:
            result[label] = int(config["out_channels"])
        else:
            raise ValueError(f"Cannot determine output width of image head `{label}`.")
    return result


def _object_targets(
    labels: ClusterLabelBatch | TensorBatch | Sequence[Any] | torch.Tensor,
    objects: IndexBatch,
    target: str | int | None,
    target_reduction: str,
    device: torch.device,
) -> torch.Tensor:
    """Normalize direct or voxel-level labels to one target per object.

    ``target_reduction='ancestor'`` restricts each object to rows belonging to
    its root ancestor before reading the target. The virtual
    ``kinetic_energy`` target is derived from that particle's initial momentum
    and PID-dependent rest mass.
    """
    num_objects = len(objects.index_list)

    # Tensor batches may contain either direct object or voxel-level labels
    if isinstance(labels, ClusterLabelBatch):
        if target is None:
            raise ValueError("Structured cluster labels require a named target.")
        if not isinstance(target, str):
            raise TypeError("Structured cluster-label targets must be named fields.")
        kinetic_energy = target == "kinetic_energy"
        if target_reduction == "ancestor":
            if not kinetic_energy and target != "pid":
                raise ValueError(
                    "Ancestor target reduction currently supports only `pid` "
                    "and `kinetic_energy`."
                )
            pid_values = labels.ancestor_pids.torch_tensor()
            target_values = (
                labels.ancestor_momenta.torch_tensor() if kinetic_energy else pid_values
            )
        else:
            pid_values = labels.pids.torch_tensor()
            target_values = (
                labels.momenta.torch_tensor()
                if kinetic_energy
                else labels.voxel_field(target).torch_tensor()
            )

        values = []
        for object_index in objects.to_numpy().index_list:
            index = torch.as_tensor(
                object_index, dtype=torch.long, device=target_values.device
            )
            object_values = target_values[index]
            object_pids = pid_values[index].long()
            unique_pid = torch.unique(object_pids)
            particle_id = int(unique_pid[0].item()) if len(unique_pid) == 1 else -1
            unique_values = torch.unique(object_values)
            if kinetic_energy:
                if (
                    particle_id not in PID_MASSES
                    or len(unique_values) != 1
                    or not torch.isfinite(unique_values[0]).item()
                    or (unique_values[0] < 0).item()
                ):
                    values.append(target_values.new_tensor(-1.0))
                    continue
                momentum = unique_values[0]
                mass = PID_MASSES[particle_id]
                values.append(torch.sqrt(momentum.square() + mass**2) - mass)
            else:
                unique, counts = torch.unique(object_values, return_counts=True)
                values.append(unique[counts.argmax()])
        targets = torch.stack(values) if values else target_values.new_empty((0,))

    elif isinstance(labels, TensorBatch):
        label_tensor = labels.torch_tensor()
        if target is not None:
            raise TypeError(
                "Voxel-level particle targets require ClusterLabelBatch; "
                "TensorBatch is reserved for direct object labels."
            )
        if target_reduction != "mode":
            raise ValueError("Direct image labels do not support `target_reduction`.")
        if len(label_tensor) != num_objects:
            raise ValueError(
                "Direct TensorBatch labels must contain one row per image object."
            )
        targets = label_tensor

    # Plain tensors and sequences already represent one target per object
    else:
        if isinstance(labels, (str, bytes)):
            raise TypeError("Image labels must be numeric values, not text.")
        targets = torch.as_tensor(labels, device=device)
        if len(targets) != num_objects:
            raise ValueError("Direct labels must contain one value per image object.")

    return targets.to(device=device)


class ImageClassificationLoss(ImageTaskLoss):
    """Supervise one image-model classification head."""

    name = "class"
    aliases = ("classification",)

    def __init__(
        self,
        out_channels: int,
        label: str = "labels",
        target: str | int | None = None,
        target_reduction: str = "mode",
        loss: str | dict[str, Any] = "ce",
        balance_loss: bool = False,
        class_weights: Sequence[float] | None = None,
        ignore_index: int = -1,
        *,
        min_iou: float | Sequence[float] | None = None,
        min_purity: float | Sequence[float] | None = None,
        min_efficiency: float | Sequence[float] | None = None,
        match_target: str | None = None,
    ) -> None:
        """Initialize a classification objective.

        Parameters
        ----------
        out_channels : int
            Number of classes predicted by the corresponding head.
        label : str, default "labels"
            Loss-input key containing direct or voxel-level labels.
        target : str or int, optional
            Cluster-label column used to reduce voxel labels to objects.
        target_reduction : {"mode", "ancestor"}, default "mode"
            Strategy used to select an object target. ``ancestor`` reads the
            target from the root particle of an ancestor-defined object.
        loss : str or dict, default "ce"
            Classification loss configuration.
        balance_loss : bool, default False
            Derive inverse-frequency weights from each minibatch.
        class_weights : sequence of float, optional
            Fixed class weights.
        ignore_index : int, default -1
            Label value excluded from supervision and metrics.
        min_iou : float or sequence of float, optional
            Minimum truth-instance IoU, shared or specified per target class.
        min_purity : float or sequence of float, optional
            Minimum predicted-instance purity, shared or specified per class.
        min_efficiency : float or sequence of float, optional
            Minimum truth-instance efficiency, shared or specified per class.
        match_target : str, optional
            Truth-instance field used to evaluate overlap quality. Defaults to
            ``ancestor`` for ancestor reduction and ``group`` otherwise.
        """
        # Initialize the parent class
        super().__init__()

        # Validate and store the task definition
        if out_channels < 2:
            raise ValueError("Image classification requires at least two classes.")
        if balance_loss and class_weights is not None:
            raise ValueError("Cannot combine fixed and dynamic class weights.")
        if class_weights is not None and len(class_weights) != out_channels:
            raise ValueError("Provide exactly one weight per image class.")
        if target_reduction not in {"mode", "ancestor"}:
            raise ValueError("Image target reduction must be `mode` or `ancestor`.")

        self.num_classes = out_channels
        self.label = label
        self.target = target
        self.target_reduction = target_reduction
        self.balance_loss = balance_loss
        self.class_weights = class_weights
        self.ignore_index = ignore_index

        # Ancestor-reduced objects must be compared with ancestor instances;
        # ordinary image objects use the particle-group instance definition.
        match_target = match_target or (
            "ancestor" if target_reduction == "ancestor" else "group"
        )
        self.quality_filter = ClusterQualityFilter(
            min_iou,
            min_purity,
            min_efficiency,
            match_target=match_target,
            num_classes=out_channels,
        )
        self.loss_fn: Any = loss_fn_factory(loss, reduction="none")

    def forward(
        self,
        labels: ClusterLabelBatch | TensorBatch | Sequence[Any] | torch.Tensor,
        objects: IndexBatch,
        prediction: TensorBatch,
    ) -> dict[str, Any]:
        """Compute the image-object classification objective.

        Parameters
        ----------
        labels : ClusterLabelBatch, TensorBatch, sequence or torch.Tensor
            Direct object labels or structured voxel labels from which the
            configured target can be reduced.
        objects : IndexBatch
            Voxel indexes defining the image objects represented by the
            prediction rows.
        prediction : TensorBatch
            Classification logits with shape ``(N, C)``.

        Returns
        -------
        dict
            Mean loss, global accuracy, supervised-object count and one
            accuracy entry per target class. When quality filtering is active,
            also includes the number of otherwise valid rejected objects.

        Raises
        ------
        TypeError
            If overlap filtering is requested without structured voxel labels.
        ValueError
            If target labels are not scalar class IDs or fall outside the
            configured output range.
        """
        # Normalize labels and identify valid class targets
        logits = prediction.torch_tensor()
        targets = _object_targets(
            labels,
            objects,
            self.target,
            self.target_reduction,
            logits.device,
        )
        if targets.ndim > 1:
            if targets.shape[1] != 1:
                raise ValueError("Classification targets must be scalar class IDs.")
            targets = targets.flatten()
        targets = targets.long()

        # Validate ordinary classification eligibility before overlap quality.
        valid = targets != self.ignore_index
        invalid = valid & ((targets < 0) | (targets >= self.num_classes))
        if torch.any(invalid).item():
            raise ValueError(
                f"Classification labels must lie in [0, {self.num_classes})."
            )

        # Overlap thresholds require voxel-level truth and reconstructed object
        # indexes; direct image labels do not define an instance match.
        count_rejected = 0
        if self.quality_filter.active:
            if not isinstance(labels, ClusterLabelBatch):
                raise TypeError(
                    "Image overlap thresholds require `ClusterLabelBatch` labels."
                )
            classes = targets.detach().cpu().numpy()
            quality_mask = self.quality_filter.node_mask(labels, objects, classes)
            quality_mask_tensor = torch.as_tensor(
                quality_mask,
                dtype=torch.bool,
                device=targets.device,
            )
            count_rejected = int(
                torch.count_nonzero(valid & ~quality_mask_tensor).item()
            )

            # Reuse the configured ignore index so quality rejection follows
            # precisely the same downstream path as an absent target.
            targets = targets.clone()
            targets[~quality_mask_tensor] = self.ignore_index
            valid &= quality_mask_tensor

        # Return a differentiable zero when no supervised object is available
        valid_index = torch.nonzero(valid).flatten()
        if len(valid_index) == 0:
            result: dict[str, Any] = {
                "loss": logits.sum() * 0.0,
                "accuracy": 1.0,
                "count": 0,
            }
            if self.quality_filter.active:
                result["count_rejected"] = count_rejected
            for class_id in range(self.num_classes):
                result[f"accuracy_class_{class_id}"] = 1.0
            return result

        # Evaluate the per-object loss on valid targets only
        logits = logits[valid_index]
        targets = targets[valid_index]
        losses = self.loss_fn(logits, targets)
        sample_weights = None
        counts = torch.bincount(targets, minlength=self.num_classes)

        # Build optional fixed or minibatch-derived class weights
        if self.balance_loss:
            class_weights = torch.ones(
                self.num_classes,
                dtype=logits.dtype,
                device=logits.device,
            )
            present = counts > 0
            class_weights[present] = len(targets) / self.num_classes / counts[present]
            sample_weights = class_weights[targets]
        elif self.class_weights is not None:
            class_weights = torch.as_tensor(
                self.class_weights,
                dtype=logits.dtype,
                device=logits.device,
            )
            sample_weights = class_weights[targets]

        # Reduce weighted and unweighted objectives consistently
        if sample_weights is None:
            loss = losses.mean()
        else:
            loss = (losses * sample_weights).sum() / sample_weights.sum()

        # Report global and per-class assignment accuracy
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()
        result = {
            "loss": loss,
            "accuracy": accuracy,
            "count": len(valid_index),
        }
        if self.quality_filter.active:
            result["count_rejected"] = count_rejected
        for class_id in range(self.num_classes):
            class_mask = targets == class_id
            result[f"accuracy_class_{class_id}"] = (
                (predictions[class_mask] == class_id).float().mean().item()
                if torch.any(class_mask).item()
                else 1.0
            )
        return result


class ImageRegressionLoss(ImageTaskLoss):
    """Supervise one scalar or vector image-model regression head."""

    name = "reg"
    aliases = ("regression",)

    def __init__(
        self,
        out_channels: int,
        label: str = "labels",
        target: str | int | None = None,
        target_reduction: str = "mode",
        loss: str | dict[str, Any] = "mse",
        ignore_value: float | None = -1.0,
        *,
        min_iou: float | Sequence[float] | None = None,
        min_purity: float | Sequence[float] | None = None,
        min_efficiency: float | Sequence[float] | None = None,
        match_target: str | None = None,
        quality_target: str = "pid",
        quality_num_classes: int | None = None,
    ) -> None:
        """Initialize a regression objective.

        Parameters
        ----------
        out_channels : int
            Number of values predicted per image object.
        label : str, default "labels"
            Loss-input key containing direct or voxel-level targets.
        target : str or int, optional
            Cluster-label column used to reduce voxel labels to objects.
            ``kinetic_energy`` derives initial kinetic energy from momentum
            and the PID-dependent rest mass.
        target_reduction : {"mode", "ancestor"}, default "mode"
            Strategy used to select an object target. ``ancestor`` reads the
            target from the root particle of an ancestor-defined object.
        loss : str or dict, default "mse"
            Regression loss configuration, such as ``mse`` or ``huber``.
        ignore_value : float, optional
            Target value excluded from supervision. Set to ``null`` to retain
            every finite target.
        min_iou : float or sequence of float, optional
            Minimum truth-instance IoU, shared or specified per quality class.
        min_purity : float or sequence of float, optional
            Minimum predicted-instance purity, shared or specified per class.
        min_efficiency : float or sequence of float, optional
            Minimum truth-instance efficiency, shared or specified per class.
        match_target : str, optional
            Truth-instance field used to evaluate overlap quality. Defaults to
            ``ancestor`` for ancestor reduction and ``group`` otherwise.
        quality_target : str, default 'pid'
            Categorical field used to select class-dependent thresholds.
        quality_num_classes : int, optional
            Number of quality classes. Required for threshold sequences.
        """
        # Initialize the parent class
        super().__init__()

        # Validate and store the task definition
        if out_channels < 1:
            raise ValueError("Image regression output width must be positive.")
        if target_reduction not in {"mode", "ancestor"}:
            raise ValueError("Image target reduction must be `mode` or `ancestor`.")
        self.out_channels = out_channels
        self.label = label
        self.target = target
        self.target_reduction = target_reduction
        self.ignore_value = ignore_value
        self.quality_target = quality_target

        # Match against the same instance definition used to construct the
        # image objects unless the caller explicitly selects another field.
        match_target = match_target or (
            "ancestor" if target_reduction == "ancestor" else "group"
        )
        self.quality_filter = ClusterQualityFilter(
            min_iou,
            min_purity,
            min_efficiency,
            match_target=match_target,
            num_classes=quality_num_classes,
            require_num_classes=True,
        )
        self.loss_fn: Any = loss_fn_factory(loss, reduction="none")

    def forward(
        self,
        labels: ClusterLabelBatch | TensorBatch | Sequence[Any] | torch.Tensor,
        objects: IndexBatch,
        prediction: TensorBatch,
    ) -> dict[str, Any]:
        """Compute the image-object regression objective and residual metrics.

        Parameters
        ----------
        labels : ClusterLabelBatch, TensorBatch, sequence or torch.Tensor
            Direct object targets or structured voxel labels from which the
            configured regression target can be reduced.
        objects : IndexBatch
            Voxel indexes defining the image objects represented by the
            prediction rows.
        prediction : TensorBatch
            Scalar or vector predictions with shape ``(N, out_channels)``.

        Returns
        -------
        dict
            Mean loss, residual bias, mean absolute error, root mean squared
            error and the number of supervised objects. When quality filtering
            is active, also includes the number of otherwise valid rejected
            objects.

        Raises
        ------
        TypeError
            If overlap filtering is requested without structured voxel labels.
        ValueError
            If prediction and target shapes differ, or class-dependent quality
            thresholds cannot be reduced to scalar class IDs.
        """
        # Normalize targets and enforce the configured output shape
        predictions = prediction.torch_tensor()
        targets = _object_targets(
            labels,
            objects,
            self.target,
            self.target_reduction,
            predictions.device,
        )
        if targets.ndim == 1:
            targets = targets.view(-1, 1)
        if targets.shape != predictions.shape:
            raise ValueError(
                "Regression prediction and target shapes do not match: "
                f"{tuple(predictions.shape)} != {tuple(targets.shape)}."
            )
        targets = targets.to(dtype=predictions.dtype)

        # Exclude non-finite and explicitly ignored targets
        valid = torch.isfinite(targets).all(dim=1)
        if self.ignore_value is not None:
            valid &= torch.all(targets != self.ignore_value, dim=1)

        # Apply the same instance-quality policy used by classification heads.
        count_rejected = 0
        if self.quality_filter.active:
            if not isinstance(labels, ClusterLabelBatch):
                raise TypeError(
                    "Image overlap thresholds require `ClusterLabelBatch` labels."
                )
            classes = None
            if self.quality_filter.class_dependent:
                # Regression targets are continuous, so obtain a separate
                # categorical label to select class-dependent requirements.
                quality_values = labels.voxel_field(self.quality_target).data
                if quality_values.ndim > 1 and quality_values.shape[1] != 1:
                    raise ValueError("Overlap quality classes must be scalar IDs.")
                classes = _object_targets(
                    labels,
                    objects,
                    self.quality_target,
                    self.target_reduction,
                    predictions.device,
                )
                classes = classes.flatten().long().detach().cpu().numpy()
            quality_mask = self.quality_filter.node_mask(labels, objects, classes)
            quality_mask_tensor = torch.as_tensor(
                quality_mask,
                dtype=torch.bool,
                device=valid.device,
            )
            count_rejected = int(
                torch.count_nonzero(valid & ~quality_mask_tensor).item()
            )

            # Keep a conventional ignored value for callers which inspect the
            # filtered targets while excluding it from all objective metrics.
            targets = targets.clone()
            targets[~quality_mask_tensor] = -1
            valid &= quality_mask_tensor

        valid_index = torch.nonzero(valid).flatten()

        # Return a differentiable zero when no supervised object is available
        if len(valid_index) == 0:
            zero = predictions.sum() * 0.0
            result = {
                "loss": zero,
                "bias": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "count": 0,
            }
            if self.quality_filter.active:
                result["count_rejected"] = count_rejected
            return result

        # Evaluate the objective and physical residual metrics
        predictions = predictions[valid_index]
        targets = targets[valid_index]
        losses = self.loss_fn(predictions, targets)
        loss = losses.reshape(len(valid_index), -1).mean(dim=1).mean()
        residuals = predictions - targets
        result = {
            "loss": loss,
            "bias": residuals.mean().item(),
            "mae": residuals.abs().mean().item(),
            "rmse": residuals.square().mean().sqrt().item(),
            "count": len(valid_index),
        }
        if self.quality_filter.active:
            result["count_rejected"] = count_rejected

        return result


class ImageLoss(torch.nn.Module):
    """Apply independently configured objectives to named image heads."""

    def __init__(
        self,
        image: dict[str, Any],
        image_loss: dict[str, dict[str, Any]],
    ) -> None:
        """Initialize the image-task loss orchestrator.

        Parameters
        ----------
        image : dict
            Upstream image-model configuration supplying named head widths.
        image_loss : dict
            One classification or regression loss configuration per head.
        """
        # Initialize the parent class
        super().__init__()

        # Match every loss task to an existing prediction head
        try:
            sizes = _head_sizes(image["heads"])
        except KeyError as err:
            raise ValueError(
                "Image loss requires model `heads` configuration."
            ) from err
        if set(image_loss) != set(sizes):
            raise ValueError(
                "Image loss tasks must exactly match prediction heads: "
                f"{sorted(image_loss)} != {sorted(sizes)}."
            )

        # Initialize the independently weighted task objectives
        self.tasks = torch.nn.ModuleDict()
        self.task_weights: dict[str, float] = {}
        task_types = {
            "class": ImageClassificationLoss,
            "classification": ImageClassificationLoss,
            "reg": ImageRegressionLoss,
            "regression": ImageRegressionLoss,
        }
        for label, task_config in image_loss.items():
            config = dict(task_config)
            try:
                name = config.pop("name")
            except KeyError as err:
                raise ValueError(f"Image loss `{label}` requires `name`.") from err
            weight = float(config.pop("weight", 1.0))
            if weight <= 0.0:
                raise ValueError("Image task weights must be positive.")
            try:
                task_class = task_types[name]
            except KeyError as err:
                valid = ", ".join(sorted(task_types))
                raise ValueError(
                    f"Unknown image task `{name}`. Choose from {valid}."
                ) from err
            self.tasks[label] = task_class(out_channels=sizes[label], **config)
            self.task_weights[label] = weight

    def forward(self, objects: IndexBatch, **data: Any) -> dict[str, Any]:
        """Evaluate every configured task and aggregate its weighted loss."""
        # Evaluate and namespace the metrics produced by each head objective
        result: dict[str, Any] = {}
        total_loss: torch.Tensor | None = None
        classification_accuracies = []
        for label, task_module in self.tasks.items():
            task = cast(ImageTaskLoss, task_module)
            prediction_key = f"{label}_pred"
            if prediction_key not in data:
                raise ValueError(f"Image output is missing `{prediction_key}`.")
            label_key = task.label
            if label_key not in data:
                raise ValueError(f"Image loss input is missing `{label_key}`.")

            task_result = task(data[label_key], objects, data[prediction_key])
            weighted_loss = self.task_weights[label] * task_result["loss"]
            total_loss = (
                weighted_loss if total_loss is None else total_loss + weighted_loss
            )
            for key, value in task_result.items():
                result[f"{label}_{key}"] = value
            if "accuracy" in task_result:
                classification_accuracies.append(float(task_result["accuracy"]))

        # Combine task losses and classification-only summary metrics
        assert total_loss is not None
        result["loss"] = total_loss
        if classification_accuracies:
            result["accuracy"] = sum(classification_accuracies) / len(
                classification_accuracies
            )
        return result
