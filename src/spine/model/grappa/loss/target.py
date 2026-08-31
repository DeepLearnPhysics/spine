"""Utilities for validating and materializing cached GrapPA supervision.

Cached supervision must preserve the item ordering and per-event partitioning
of the model objects it describes. The helpers in this module centralize the
usual prediction-aligned invariants and keep device conversion out of the
individual loss classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import TensorBatch
from spine.utils.conditional import torch

__all__ = [
    "prepare_cached_target",
    "prepare_cached_validity",
    "target_tensor",
    "validity_batch",
]


def _counts_numpy(batch: TensorBatch) -> np.ndarray:
    """Return a batch's event counts as a CPU NumPy array.

    Only the small metadata array is transferred when counts are stored as a
    tensor; the potentially large target payload is left untouched.

    Parameters
    ----------
    batch : TensorBatch
        Batch whose per-event counts should be normalized.

    Returns
    -------
    np.ndarray
        Number of entries associated with each event in the batch.
    """
    counts = batch.counts
    if isinstance(counts, np.ndarray):
        return counts
    return counts.detach().cpu().numpy()


def validity_batch(mask: np.ndarray, reference: TensorBatch) -> TensorBatch:
    """Wrap a validity array using the partitioning of a reference batch.

    Parameters
    ----------
    mask : np.ndarray
        One boolean-compatible value per node or edge.
    reference : TensorBatch
        Target batch whose per-event counts define the mask partitioning.

    Returns
    -------
    TensorBatch
        Boolean validity mask aligned and partitioned like ``reference``.
    """
    # Validity is persisted as a compact one-dimensional boolean payload.
    return TensorBatch(np.asarray(mask, dtype=bool), _counts_numpy(reference))


def prepare_cached_target(
    labels: TensorBatch,
    valid_mask: TensorBatch,
    prediction: TensorBatch,
    kind: str,
) -> np.ndarray:
    """Validate cached supervision and return its NumPy validity mask.

    Parameters
    ----------
    labels : TensorBatch
        Cached labels aligned with the prediction axis.
    valid_mask : TensorBatch
        One boolean-compatible value per prediction.
    prediction : TensorBatch
        Prediction batch which defines the required counts and length.
    kind : str
        Human-readable objective axis used in error messages.

    Returns
    -------
    np.ndarray
        One-dimensional Boolean validity mask on CPU.

    Raises
    ------
    TypeError
        If either cached input is not represented as a ``TensorBatch``.
    ValueError
        If labels or validity do not match the prediction length and event
        partitioning, or if the validity payload is not one-dimensional.
    """
    # Validate that the cached labels and validity mask are TensorBatch instances.
    if not isinstance(labels, TensorBatch) or not isinstance(valid_mask, TensorBatch):
        raise TypeError(f"Cached {kind} labels and validity mask must be TensorBatch.")

    # Equal total length is insufficient: event boundaries must agree as well.
    prediction_counts = _counts_numpy(prediction)
    if len(labels) != len(prediction) or not np.array_equal(
        _counts_numpy(labels), prediction_counts
    ):
        raise ValueError(f"Cached {kind} labels must align with predictions.")
    return prepare_cached_validity(valid_mask, prediction, kind)


def prepare_cached_validity(
    valid_mask: TensorBatch,
    prediction: TensorBatch,
    kind: str,
) -> np.ndarray:
    """Validate and materialize a prediction-aligned cached validity mask.

    This is separate from :func:`prepare_cached_target` for objectives whose
    stable cached target lives on a different axis from their predictions,
    such as forest edge supervision backed by node group IDs.

    Parameters
    ----------
    valid_mask : TensorBatch
        One boolean-compatible value per prediction.
    prediction : TensorBatch
        Prediction batch which defines the required counts and length.
    kind : str
        Human-readable objective axis used in error messages.

    Returns
    -------
    np.ndarray
        One-dimensional Boolean validity mask on CPU.
    """
    if not isinstance(valid_mask, TensorBatch):
        raise TypeError(f"Cached {kind} validity mask must be TensorBatch.")

    prediction_counts = _counts_numpy(prediction)
    if len(valid_mask) != len(prediction) or not np.array_equal(
        _counts_numpy(valid_mask), prediction_counts
    ):
        raise ValueError(f"Cached {kind} validity mask must align with predictions.")

    # Validation happens on CPU because masks are consumed as NumPy indexes.
    mask = valid_mask.to_numpy().data
    if mask.ndim != 1:
        raise ValueError(f"Cached {kind} validity mask must be one-dimensional.")
    return mask.astype(bool, copy=False)


def target_tensor(
    labels: TensorBatch,
    prediction: TensorBatch,
    dtype: Any = None,
) -> Any:
    """Materialize cached or live targets beside their predictions.

    Parameters
    ----------
    labels : TensorBatch
        Target values to present to the loss function.
    prediction : TensorBatch
        Prediction batch defining the target device.
    dtype : torch.dtype, optional
        Explicit target type required by the loss, such as ``torch.long`` for
        classification or the prediction dtype for regression.

    Returns
    -------
    torch.Tensor
        Target payload on the prediction device with the requested dtype.
    """
    # ``as_tensor`` avoids a copy when labels already have the right placement.
    return torch.as_tensor(labels.data, dtype=dtype, device=prediction.device)
