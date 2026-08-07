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
        """Build an aligned family from the driver-provided point tensor."""
        result = cls(data, data, sources=sources, orig_index=orig_index)
        result._validate()
        return result

    @staticmethod
    def _counts_array(value: TensorBatch | IndexBatch) -> np.ndarray:
        """Return small structural counts as a CPU NumPy array."""
        counts = value.counts
        if isinstance(counts, torch.Tensor):
            counts = counts.detach().cpu().numpy()
        return np.asarray(counts)

    def _validate(self) -> None:
        """Check that every stored representation describes the same point rows."""
        aligned = {
            "data_q": self.data_q,
            "data_calib": self.data_calib,
            "sources": self.sources,
            "orig_index": self.orig_index,
        }
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

        if self.data_calib is None:
            if self.data is not self.data_q:
                raise ValueError("Uncalibrated active point data must be `data_q`.")
        elif self.data is not self.data_calib:
            raise ValueError("Calibrated active point data must be `data_calib`.")

    @staticmethod
    def _copy_with_values(data: TensorBatch, values: Any) -> TensorBatch:
        """Copy a packed tensor and replace its primary logical feature."""
        tensor = data.torch_tensor().clone()
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
        """Replace charge values without changing the aligned row domain."""
        if self.data_calib is not None:
            raise ValueError(
                "Charge cannot be changed after calibration without recalibrating."
            )
        data_q = self._copy_with_values(self.data_q, values)
        result = replace(self, data=data_q, data_q=data_q)
        result._validate()
        return result

    def with_calibration(self, data_calib: TensorBatch) -> "PointBatch":
        """Attach calibrated data and make it active for downstream models."""
        result = replace(self, data=data_calib, data_calib=data_calib)
        result._validate()
        return result

    def align(self, tensor: TensorBatch) -> TensorBatch:
        """Align an original-domain tensor to the current point rows."""
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
        """Apply one row selection to all aligned point representations."""
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
        """Return compatibility aliases for consumers not yet bundle-aware."""
        products: dict[str, Any] = {"data": self.data}
        if self.sources is not None:
            products["sources"] = self.sources
        if self.orig_index is not None:
            products["orig_index"] = self.orig_index
        return products

    def public_outputs(self) -> dict[str, Any]:
        """Materialize stable adapted and calibrated reconstruction outputs."""
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
