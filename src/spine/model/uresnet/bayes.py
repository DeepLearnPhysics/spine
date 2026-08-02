"""Bayesian UResNet segmentation with dropout or evidential uncertainty."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Literal

import numpy as np
import torch

from spine.constants import VALUE_COL
from spine.data import TensorBatch
from spine.model import sparse

from ..cnn.mcdropout import MCDropoutDecoder, MCDropoutEncoder
from ..cnn.uresnet_layers import EncoderOutput
from ..common.evidential import EVDLoss
from ..common.factories import loss_fn_factory
from ..registry import ModelSpec

__all__ = [
    "BayesianUResNet",
    "BayesianSegmentationLoss",
    "MODEL_SPEC",
]

BayesianMode = Literal["standard", "mc_dropout", "evidential"]


class BayesianBackboneOutput(EncoderOutput):
    """Sparse feature planes produced by one Bayesian UResNet pass."""

    decoder_tensors: list[sparse.SparseTensor]


class BayesianUResNet(torch.nn.Module):
    """Segment sparse images while exposing predictive uncertainty.

    The model uses dropout-enabled UResNet encoder and decoder blocks. During
    ordinary training, ``mode='standard'`` performs one stochastic pass and
    returns logits. ``mode='mc_dropout'`` keeps only dropout layers active at
    inference and averages multiple stochastic passes. ``mode='evidential'``
    predicts nonnegative Dirichlet evidence in one pass.

    Notes
    -----
    ``standard`` and ``mc_dropout`` have identical trainable parameters, so a
    checkpoint trained with the former can be evaluated with the latter.
    """

    def __init__(self, uresnet_bayes: dict[str, Any]) -> None:
        """Initialize the Bayesian segmentation network.

        Parameters
        ----------
        uresnet_bayes : dict
            UResNet backbone configuration plus ``num_classes``, ``mode``,
            ``num_samples``, ``dropout_p`` and optional ``dropout_layers``.

        Raises
        ------
        ValueError
            If the class count, mode, or number of stochastic samples is
            invalid.
        """
        # Initialize the parent class
        super().__init__()

        # Extract and validate the Bayesian prediction contract
        cfg = dict(uresnet_bayes)
        try:
            self.num_classes = int(cfg.pop("num_classes"))
        except KeyError as err:
            raise ValueError("Bayesian UResNet requires `num_classes`.") from err
        if self.num_classes < 2:
            raise ValueError("Bayesian UResNet requires at least two classes.")

        mode = cfg.pop("mode", "standard")
        if mode == "evd":
            mode = "evidential"
        if mode not in {"standard", "mc_dropout", "evidential"}:
            raise ValueError(
                "Unknown Bayesian UResNet mode "
                f"`{mode}`. Choose standard, mc_dropout or evidential."
            )
        self.mode: BayesianMode = mode

        self.num_samples = int(cfg.pop("num_samples", 20))
        if self.num_samples < 1:
            raise ValueError("`num_samples` must be positive.")

        # Initialize the dropout-enabled UResNet backbone
        dropout_p = float(cfg.pop("dropout_p", 0.5))
        dropout_layers = cfg.pop("dropout_layers", None)
        self.encoder = MCDropoutEncoder(
            cfg,
            dropout_p=dropout_p,
            dropout_layers=dropout_layers,
            add_classifier=False,
        )
        self.decoder = MCDropoutDecoder(
            cfg,
            dropout_p=dropout_p,
            dropout_layers=dropout_layers,
        )

        # Build the segmentation projection for logits or positive evidence
        classifier: list[torch.nn.Module] = [
            sparse.Linear(self.encoder.num_filters, self.num_classes)
        ]
        if self.mode == "evidential":
            classifier.append(sparse.Softplus())
        self.classifier = torch.nn.Sequential(*classifier)

    def _forward_once(
        self,
        input_tensor: torch.Tensor,
        batch_size: int,
    ) -> tuple[sparse.SparseTensor, BayesianBackboneOutput]:
        """Run one stochastic UResNet pass.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Coordinate-feature table accepted by the sparse frontend.
        batch_size : int
            Number of entries represented by the input batch.

        Returns
        -------
        sparse.SparseTensor
            Per-voxel logits or evidence.
        BayesianBackboneOutput
            Encoder and decoder feature planes from this pass.
        """
        # Convert the dense coordinate-feature table to a sparse input
        dimension = self.encoder.dimension
        coordinates = input_tensor[:, : dimension + 1].int()
        features = input_tensor[:, dimension + 1 :]
        sparse_input = sparse.SparseTensor(
            coordinates=coordinates,
            features=features,
            batch_size=batch_size,
        )

        # Run the encoder, decoder, and final prediction projection
        encoder_output = self.encoder.encode(sparse_input)
        decoder_tensors = self.decoder(
            encoder_output["final_tensor"],
            encoder_output["encoder_tensors"],
        )
        output = self.classifier(decoder_tensors[-1])

        # Retain the feature pyramid for downstream model stages
        backbone_output: BayesianBackboneOutput = {
            "final_tensor": encoder_output["final_tensor"],
            "encoder_tensors": encoder_output["encoder_tensors"],
            "decoder_tensors": decoder_tensors,
        }
        return output, backbone_output

    @contextmanager
    def _stochastic_dropout(self) -> Iterator[None]:
        """Temporarily enable dropout without changing normalization layers."""
        # Capture the original state of every sparse dropout layer
        dropout_modules = [
            module for module in self.modules() if isinstance(module, sparse.Dropout)
        ]
        training_states = [module.training for module in dropout_modules]
        try:
            # Activate stochastic masks without putting the full model in training
            for module in dropout_modules:
                module.train()
            yield
        finally:
            # Restore caller-owned training states after stochastic inference
            for module, training in zip(dropout_modules, training_states, strict=True):
                module.train(training)

    @staticmethod
    def _feature_batch(
        output: sparse.SparseTensor,
        data: TensorBatch,
    ) -> TensorBatch:
        """Wrap sparse features with the input batch boundaries."""
        return TensorBatch(output.aligned_features(), data.counts)

    def forward(self, data: TensorBatch) -> dict[str, Any]:
        """Predict segmentation and uncertainty products for one batch.

        Parameters
        ----------
        data : TensorBatch
            ``(N, 1 + D + C)`` coordinate-feature table.

        Returns
        -------
        dict
            Always contains ``segmentation``. Monte Carlo mode also returns
            ``softmax``; evidential mode returns evidence, concentration,
            expected probability, and uncertainty. Single-pass modes expose
            the sparse UResNet feature planes.
        """
        # Narrow the input table to the columns consumed by the backbone
        data_tensor = data.torch_tensor()
        num_columns = 1 + self.encoder.dimension + self.encoder.num_input
        input_tensor = data_tensor[:, :num_columns]

        # Average repeated stochastic predictions in Monte Carlo mode
        if self.mode == "mc_dropout":
            logits_sum: torch.Tensor | None = None
            probability_sum: torch.Tensor | None = None
            with self._stochastic_dropout():
                for _ in range(self.num_samples):
                    output, _ = self._forward_once(input_tensor, data.batch_size)
                    logits = output.aligned_features()
                    probabilities = torch.softmax(logits, dim=1)
                    logits_sum = logits if logits_sum is None else logits_sum + logits
                    probability_sum = (
                        probabilities
                        if probability_sum is None
                        else probability_sum + probabilities
                    )

            if logits_sum is None or probability_sum is None:  # pragma: no cover
                raise RuntimeError("Monte Carlo inference produced no samples.")
            segmentation = logits_sum / self.num_samples
            softmax = probability_sum / self.num_samples
            return {
                "segmentation": TensorBatch(segmentation, data.counts),
                "softmax": TensorBatch(softmax, data.counts),
            }

        # Run the standard single-pass prediction path
        output, backbone_output = self._forward_once(input_tensor, data.batch_size)
        prediction = output.aligned_features()
        result: dict[str, Any] = {
            "segmentation": TensorBatch(prediction, data.counts),
            **backbone_output,
        }

        # Derive Dirichlet uncertainty products from positive evidence
        if self.mode == "evidential":
            concentration = prediction + 1.0
            total_concentration = concentration.sum(dim=1, keepdim=True)
            result.update(
                {
                    "evidence": TensorBatch(prediction, data.counts),
                    "concentration": TensorBatch(concentration, data.counts),
                    "expected_probability": TensorBatch(
                        concentration / total_concentration,
                        data.counts,
                    ),
                    "uncertainty": TensorBatch(
                        self.num_classes / total_concentration,
                        data.counts,
                    ),
                }
            )
        return result


class BayesianSegmentationLoss(torch.nn.Module):
    """Optimize Bayesian semantic segmentation predictions.

    Cross entropy supervises standard and Monte Carlo dropout logits.
    Evidential mode instead applies an annealed Dirichlet objective directly
    to the nonnegative evidence returned by :class:`BayesianUResNet`.
    """

    def __init__(
        self,
        uresnet_bayes: dict[str, Any],
        uresnet_bayes_loss: dict[str, Any],
    ) -> None:
        """Initialize the Bayesian segmentation objective.

        Parameters
        ----------
        uresnet_bayes : dict
            Upstream model configuration supplying ``num_classes`` and mode.
        uresnet_bayes_loss : dict
            Loss name, class-balancing option, and optional evidential-loss
            arguments.

        Raises
        ------
        ValueError
            If the configured loss family is inconsistent with the model mode.
        """
        # Initialize the parent class
        super().__init__()

        # Extract the model properties that determine the objective family
        self.num_classes = int(uresnet_bayes["num_classes"])
        mode = uresnet_bayes.get("mode", "standard")
        self.mode = "evidential" if mode == "evd" else mode

        cfg = dict(uresnet_bayes_loss)
        default_loss = "edl_sumsq" if self.mode == "evidential" else "ce"
        loss_name = cfg.pop("loss", default_loss)
        self.balance_loss = bool(cfg.pop("balance_loss", False))

        # Initialize the evidential or conventional segmentation objective
        if self.mode == "evidential":
            if not loss_name.startswith("edl_"):
                raise ValueError("Evidential mode requires an `edl_*` loss.")
            self.loss_fn = EVDLoss(
                loss_name,
                reduction="none",
                num_classes=self.num_classes,
                mode="evidence",
                **cfg,
            )
        else:
            if loss_name.startswith("edl_"):
                raise ValueError("A dropout mode requires a classification loss.")
            if cfg:
                unknown = ", ".join(sorted(cfg))
                raise ValueError(
                    f"Unexpected loss options for `{loss_name}`: {unknown}."
                )
            self.loss_fn = loss_fn_factory(loss_name, reduction="none")

    def forward(
        self,
        seg_label: TensorBatch,
        segmentation: TensorBatch,
        weights: TensorBatch | None = None,
        iteration: int = 0,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Compute loss and voxel-classification accuracy.

        Parameters
        ----------
        seg_label : TensorBatch
            ``(N, 1 + D + 1)`` semantic labels.
        segmentation : TensorBatch
            ``(N, C)`` logits or evidence from the network.
        weights : TensorBatch, optional
            Explicit nonnegative per-voxel weights.
        iteration : int, default 0
            Training iteration used to anneal the evidential KL term.
        **_kwargs : dict, optional
            Other network products ignored by this loss.

        Returns
        -------
        dict
            Scalar loss, global accuracy, classwise accuracies, and effective
            weights when weighting is active.
        """
        # Validate and normalize semantic labels and predictions
        labels = seg_label.torch_tensor()[:, VALUE_COL].long()
        predictions = segmentation.torch_tensor()
        if len(labels) != len(predictions):
            raise ValueError(
                "The segmentation prediction and label lengths do not match: "
                f"{len(predictions)} != {len(labels)}."
            )
        if len(labels) == 0:
            raise ValueError("Bayesian segmentation loss requires nonempty labels.")
        if torch.any((labels < 0) | (labels >= self.num_classes)).item():
            raise ValueError(
                f"Segmentation labels must lie in [0, {self.num_classes})."
            )

        # Validate optional externally supplied voxel weights
        effective_weights = weights.torch_tensor() if weights is not None else None
        if effective_weights is not None:
            effective_weights = effective_weights.flatten()
            if len(effective_weights) != len(labels):
                raise ValueError("The weight and label lengths do not match.")
            if torch.any(effective_weights < 0).item():
                raise ValueError("Segmentation weights must be nonnegative.")

        # Derive and combine optional minibatch class-balancing weights
        counts = torch.bincount(labels, minlength=self.num_classes)
        if self.balance_loss:
            class_weights = torch.ones(
                self.num_classes,
                dtype=predictions.dtype,
                device=predictions.device,
            )
            present = counts > 0
            class_weights[present] = len(labels) / self.num_classes / counts[present]
            balance_weights = class_weights[labels]
            effective_weights = (
                balance_weights
                if effective_weights is None
                else effective_weights * balance_weights
            )

        # Evaluate the mode-specific per-voxel objective
        if self.mode == "evidential":
            losses = self.loss_fn(predictions, labels, iteration=iteration)
        else:
            losses = self.loss_fn(predictions, labels)
        if losses.ndim > 1:
            losses = losses.reshape(len(labels), -1).mean(dim=1)

        # Reduce the objective with the effective voxel weights
        if effective_weights is None:
            loss = losses.mean()
        else:
            weight_sum = effective_weights.sum()
            if weight_sum.item() <= 0:
                raise ValueError("The sum of segmentation weights must be positive.")
            loss = (losses * effective_weights).sum() / weight_sum

        # Compute global and per-class segmentation accuracy
        with torch.no_grad():
            predictions_class = predictions.argmax(dim=1)
            accuracy = (predictions_class == labels).float().mean().item()
            class_accuracy = np.ones(self.num_classes, dtype=np.float32)
            for class_id in range(self.num_classes):
                if counts[class_id].item() > 0:
                    class_mask = labels == class_id
                    class_accuracy[class_id] = (
                        (predictions_class[class_mask] == class_id)
                        .float()
                        .mean()
                        .item()
                    )

        # Package scalar metrics and reusable effective weights
        result: dict[str, Any] = {"loss": loss, "accuracy": accuracy}
        for class_id, value in enumerate(class_accuracy):
            result[f"accuracy_class_{class_id}"] = value
        if effective_weights is not None:
            result["weights"] = TensorBatch(effective_weights, seg_label.counts)
        return result


MODEL_SPEC = ModelSpec(
    "uresnet_bayes",
    BayesianUResNet,
    BayesianSegmentationLoss,
)
