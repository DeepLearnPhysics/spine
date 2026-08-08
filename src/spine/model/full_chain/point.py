"""Aligned point-data representations used by the full reconstruction chain."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch

from spine.data import IndexBatch, TensorBatch

__all__ = ["PointBatch"]


@dataclass(frozen=True)
class PointBatch:
    """Keep charge and calibrated point-data representations row-aligned.

    ``data`` is the representation consumed by subsequent model stages.
    ``data_q`` preserves the same rows in input-charge units, while
    ``data_calib`` stores calibrated energy and coordinates when available.
    Any row selection is applied to every representation and aligned auxiliary
    product together.

    The class deliberately models a point-row domain rather than a particular
    coordinate system. The rows may therefore describe voxel centers, detector
    hits, or continuous point-cloud samples, provided every attached product
    has identical per-event counts.

    Attributes
    ----------
    data : TensorBatch
        Active point representation consumed by the next model stage. This is
        identical to ``data_q`` before calibration and to ``data_calib`` after
        calibration.
    data_q : TensorBatch
        Point representation in the input charge units. This view is retained
        even when calibrated values become active.
    data_calib : TensorBatch, optional
        Calibrated point representation, including corrected coordinates when
        a calibrator updates positions.
    sources : TensorBatch, optional
        Detector-source identifiers aligned one-to-one with the point rows.
    orig_index : IndexBatch, optional
        Mapping from the current rows to the rows of the driver input tensor.
    adapted : bool, default False
        Whether the current row domain has been selected or filtered.
    """

    data: TensorBatch
    data_q: TensorBatch
    data_calib: TensorBatch | None = None
    sources: TensorBatch | None = None
    orig_index: IndexBatch | None = None
    adapted: bool = False

    @classmethod
    def from_input(
        cls,
        data: TensorBatch,
        sources: TensorBatch | None = None,
        orig_index: IndexBatch | None = None,
    ) -> "PointBatch":
        """Build an aligned family from the driver-provided point tensor.

        Parameters
        ----------
        data : TensorBatch
            Original point data supplied to the full chain.
        sources : TensorBatch, optional
            Detector-source identifiers aligned with ``data``.
        orig_index : IndexBatch, optional
            Existing mapping from ``data`` rows to an upstream row domain.

        Returns
        -------
        PointBatch
            Uncalibrated point family whose active and charge views both
            reference ``data``.

        Raises
        ------
        ValueError
            If an optional aligned product does not share the data row domain.
        """
        result = cls(data, data, sources=sources, orig_index=orig_index)
        result._validate()
        return result

    @staticmethod
    def _counts_array(value: TensorBatch | IndexBatch) -> np.ndarray:
        """Return structural event counts as a CPU NumPy array.

        Count comparisons are control-flow checks over small arrays. Moving
        them to the CPU avoids backend-specific equality behavior while point
        payloads remain on their original device.

        Parameters
        ----------
        value : TensorBatch or IndexBatch
            Batched product whose per-event counts are needed.

        Returns
        -------
        np.ndarray
            One row count for each event in the batch.
        """
        counts = value.counts
        if isinstance(counts, torch.Tensor):
            counts = counts.detach().cpu().numpy()
        return np.asarray(counts)

    def _validate(self) -> None:
        """Check that every stored representation describes the same rows.

        Raises
        ------
        ValueError
            If an attached product has a different total length or event
            partition, or if ``data`` is not the appropriate active view.
        """
        aligned = {
            "data_q": self.data_q,
            "data_calib": self.data_calib,
            "sources": self.sources,
            "orig_index": self.orig_index,
        }
        # Total length alone is insufficient: event boundaries must agree so
        # that later per-entry models and calibrators see matching rows.
        for name, value in aligned.items():
            if value is None:
                continue
            if len(value.data) != len(self.data.data):
                raise ValueError(f"Point product `{name}` has a different row count.")
            if not np.array_equal(
                self._counts_array(value),
                self._counts_array(self.data),
            ):
                raise ValueError(f"Point product `{name}` has different event counts.")

        # Keep the active view unambiguous. Downstream stages can consume
        # ``data`` without independently reasoning about calibration state.
        if self.data_calib is None:
            if self.data is not self.data_q:
                raise ValueError("Uncalibrated active point data must be `data_q`.")
        elif self.data is not self.data_calib:
            raise ValueError("Calibrated active point data must be `data_calib`.")

    @staticmethod
    def _copy_with_values(data: TensorBatch, values: Any) -> TensorBatch:
        """Copy a packed tensor and replace its primary logical feature.

        Parameters
        ----------
        data : TensorBatch
            Source point tensor, including coordinates and logical schema.
        values : array-like
            Replacement values aligned one-to-one with ``data`` rows.

        Returns
        -------
        TensorBatch
            Independent tensor with the same coordinates, batching, schema,
            and metadata as ``data``.
        """
        tensor = data.torch_tensor().clone()
        # Charge is the first logical feature, which need not be the first
        # packed column when batch and coordinate columns are present.
        column = int(data.feature_columns()[0])
        tensor[:, column] = torch.as_tensor(
            values,
            dtype=data.dtype,
            device=data.device,
        )
        return TensorBatch(
            tensor,
            data.counts,
            has_batch_col=data.has_batch_col,
            coord_cols=data.coord_cols,
            schema=data.schema,
            meta=data.meta,
        )

    def with_charge(self, values: Any) -> "PointBatch":
        """Replace charge values without changing the aligned row domain.

        Parameters
        ----------
        values : array-like
            New charge values, one per current point row.

        Returns
        -------
        PointBatch
            New uncalibrated family with an independent charge tensor.

        Raises
        ------
        ValueError
            If calibrated data are already attached. Changing their source
            charge would make the existing calibration stale.
        """
        if self.data_calib is not None:
            raise ValueError(
                "Charge cannot be changed after calibration without recalibrating."
            )
        data_q = self._copy_with_values(self.data_q, values)
        result = replace(self, data=data_q, data_q=data_q)
        result._validate()
        return result

    def with_calibration(self, data_calib: TensorBatch) -> "PointBatch":
        """Attach calibrated data and make it active for downstream models.

        Parameters
        ----------
        data_calib : TensorBatch
            Calibrated values and coordinates on the current row domain.

        Returns
        -------
        PointBatch
            New family retaining ``data_q`` beside the active calibrated view.

        Raises
        ------
        ValueError
            If the calibrated tensor is not aligned with the current rows.
        """
        result = replace(self, data=data_calib, data_calib=data_calib)
        result._validate()
        return result

    def align(self, tensor: TensorBatch) -> TensorBatch:
        """Align an original-domain tensor to the current point rows.

        Parameters
        ----------
        tensor : TensorBatch
            Tensor already on the current rows or on the original driver row
            domain represented by ``orig_index``.

        Returns
        -------
        TensorBatch
            ``tensor`` unchanged when already aligned, otherwise a selected
            tensor with the current event counts.

        Raises
        ------
        ValueError
            If row selection is necessary but no original-row mapping exists.
        """
        # Avoid an unnecessary gather for the common unfiltered case.
        if len(tensor.data) == len(self.data.data) and np.array_equal(
            self._counts_array(tensor), self._counts_array(self.data)
        ):
            return tensor
        if self.orig_index is None:
            raise ValueError(
                "Cannot align a tensor with different rows without `orig_index`."
            )

        return TensorBatch(
            tensor.data[self.orig_index.index],
            self.data.counts,
            has_batch_col=tensor.has_batch_col,
            coord_cols=tensor.coord_cols,
            schema=tensor.schema,
            meta=tensor.meta,
        )

    def select(self, mask: Any) -> "PointBatch":
        """Apply one row selection to every aligned point representation.

        Parameters
        ----------
        mask : array-like
            Global boolean mask or row indexes. Selected rows must remain
            grouped by event, as required by :meth:`TensorBatch.select`.

        Returns
        -------
        PointBatch
            New adapted family with the same selection applied to charge,
            calibration, sources, and original-row indexes.
        """
        # Select persistent views independently, then recover the active view
        # from calibration state rather than relying on object identity alone.
        data_q = self.data_q.select(mask)
        data_calib = None
        if self.data_calib is not None:
            data_calib = self.data_calib.select(mask)
        data = data_q if data_calib is None else data_calib
        sources = None if self.sources is None else self.sources.select(mask)

        # Compose the new selection with any mapping already supplied upstream.
        if self.orig_index is None:
            if isinstance(mask, torch.Tensor):
                positions = torch.arange(len(self.data.data), device=mask.device)[mask]
            else:
                positions = np.arange(len(self.data.data))[mask]
            spans = self.data.counts
        else:
            positions = self.orig_index.index[mask]
            spans = self.orig_index.spans
        orig_index = IndexBatch(positions, spans=spans, counts=data.counts)

        result = type(self)(
            data=data,
            data_q=data_q,
            data_calib=data_calib,
            sources=sources,
            orig_index=orig_index,
            adapted=True,
        )
        result._validate()
        return result

    def canonical_products(self) -> dict[str, Any]:
        """Return flat aliases for consumers not yet aware of ``PointBatch``.

        Returns
        -------
        dict
            Active ``data`` and any aligned ``sources`` or ``orig_index``.

        Notes
        -----
        These aliases are internal compatibility products. Stable model
        outputs are produced separately by :meth:`public_outputs`.
        """
        products: dict[str, Any] = {"data": self.data}
        if self.sources is not None:
            products["sources"] = self.sources
        if self.orig_index is not None:
            products["orig_index"] = self.orig_index
        return products

    def public_outputs(self) -> dict[str, Any]:
        """Materialize stable adapted and calibrated reconstruction outputs.

        Returns
        -------
        dict
            Adapted charge as ``data_adapt``, calibrated data as
            ``data_calib``, and adapted mapping/source products when present.
            Unmodified representations are omitted.
        """
        outputs: dict[str, Any] = {}
        if self.adapted:
            outputs["data_adapt"] = self.data_q
            if self.orig_index is not None:
                outputs["orig_index"] = self.orig_index
            if self.sources is not None:
                outputs["sources_adapt"] = self.sources
        if self.data_calib is not None:
            outputs["data_calib"] = self.data_calib
        return outputs
