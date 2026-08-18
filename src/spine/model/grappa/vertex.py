"""Decode GrapPA particle-level interaction vertex predictions."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from spine.data import Meta, TensorBatch

__all__ = ["decode_vertex_positions", "vertex_position_scales"]


def vertex_position_scales(
    prediction: TensorBatch,
    meta: Sequence[Meta],
) -> torch.Tensor:
    """Build one image-size scale vector per particle prediction.

    Parameters
    ----------
    prediction : TensorBatch
        Particle-aligned three-dimensional regression values.
    meta : sequence of Meta
        Image metadata for every batch entry.

    Returns
    -------
    torch.Tensor
        ``(P, 3)`` image dimensions aligned with ``prediction``.

    Raises
    ------
    ValueError
        If metadata and prediction batch sizes differ.
    """
    if len(meta) != prediction.batch_size:
        raise ValueError(
            "Expected one metadata entry per batch entry, but received "
            f"{len(meta)} metadata entries for a batch size of "
            f"{prediction.batch_size}."
        )

    values = prediction.torch_tensor()
    scales = torch.empty_like(values)
    for batch_id, image_meta in enumerate(meta):
        lower, upper = prediction.edges[batch_id : batch_id + 2]
        scales[lower:upper] = torch.as_tensor(
            image_meta.count,
            dtype=values.dtype,
            device=values.device,
        )
    return scales


def decode_vertex_positions(
    prediction: TensorBatch,
    *,
    start_points: TensorBatch | None = None,
    end_points: TensorBatch | None = None,
    meta: Sequence[Meta] | None = None,
    normalize_positions: bool = False,
    use_anchor_points: bool = False,
    restore_absolute: bool = False,
    position_scales: torch.Tensor | None = None,
) -> TensorBatch:
    """Decode raw GrapPA regression values into vertex positions.

    GrapPA can learn absolute positions or offsets from the closest particle
    endpoint, optionally in image-normalized coordinates. This helper is used
    by both training and full-chain inference so those transformations cannot
    diverge.

    Parameters
    ----------
    prediction : TensorBatch
        ``(P, 3)`` raw regression output aligned with particle nodes.
    start_points : TensorBatch, optional
        Particle start coordinates required by anchor mode.
    end_points : TensorBatch, optional
        Particle end coordinates required by anchor mode.
    meta : sequence of Meta, optional
        Image metadata required to construct normalization scales.
    normalize_positions : bool, default False
        Interpret predictions in coordinates normalized by image dimensions.
    use_anchor_points : bool, default False
        Interpret predictions as offsets from the closest particle endpoint.
    restore_absolute : bool, default False
        Convert a normalized decoded position back to image coordinates.
    position_scales : torch.Tensor, optional
        Precomputed ``(P, 3)`` image scales. Supplying these avoids rebuilding
        scales when the caller also transforms target positions.

    Returns
    -------
    TensorBatch
        Decoded positions with the same particle counts as ``prediction``.

    Raises
    ------
    ValueError
        If normalization metadata, anchor points or aligned row counts are
        missing.
    """
    values = prediction.torch_tensor()
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Vertex regression predictions must have shape (P, 3).")

    scales = position_scales
    if normalize_positions and scales is None:
        if meta is None:
            raise ValueError("Normalized vertex prediction requires `meta`.")
        scales = vertex_position_scales(prediction, meta)
    if scales is not None and scales.shape != values.shape:
        raise ValueError("Vertex position scales must match prediction shape.")

    decoded = values
    if use_anchor_points:
        if start_points is None or end_points is None:
            raise ValueError(
                "Anchored vertex prediction requires particle end points "
                "(`start_points` and `end_points`)."
            )
        starts = start_points.torch_tensor()
        ends = end_points.torch_tensor()
        if starts.shape != values.shape or ends.shape != values.shape:
            raise ValueError("Particle endpoint rows must match vertex predictions.")

        endpoints = torch.stack((starts, ends), dim=1)
        if normalize_positions:
            assert scales is not None
            endpoints = endpoints / scales[:, None, :]

        # The raw prediction is an offset from whichever endpoint places the
        # decoded position closest to the network's proposed location.
        distances = torch.linalg.vector_norm(
            values[:, None, :] - endpoints,
            dim=2,
        )
        endpoint_ids = torch.argmin(distances, dim=1)
        row_ids = torch.arange(len(values), device=values.device)
        decoded = endpoints[row_ids, endpoint_ids] + values

    if normalize_positions and restore_absolute:
        assert scales is not None
        decoded = decoded * scales

    return TensorBatch(
        decoded,
        prediction.counts,
        coord_cols=(0, 1, 2),
        meta=prediction.meta,
    )
