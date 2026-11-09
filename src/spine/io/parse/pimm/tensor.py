"""Aligned input tensors and semantic segmentation labels."""

from __future__ import annotations

from typing import Any

from spine.data import TensorData

from .base import PIMMParserBase

__all__ = ["PIMMTensorParser", "PIMMSegmentationParser"]


class PIMMTensorParser(PIMMParserBase):
    """Return voxelized detector hit charge."""

    name = "tensor"

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Return summed detector charge on the shared sparse grid.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by PIMMReader.

        Returns
        -------
        TensorData
            Sparse coordinates and summed charge in the source hit units.
        """
        # Reuse the same grid and overlap winners as the input tensor.
        event = self.prepare(trees)
        return TensorData(
            coords=event["coords"].astype(self.itype),
            features=event["energy"].astype(self.ftype)[:, None],
            meta=event["meta"],
        )


class PIMMSegmentationParser(PIMMParserBase):
    """Return semantic labels in exactly the input voxel order."""

    name = "segmentation"

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Return five-class labels; unlabelled hits use -1.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by PIMMReader.

        Returns
        -------
        TensorData
            Semantic labels sharing the input tensor coordinates and row order.
        """
        # Reuse the same grid and overlap winners as the input tensor.
        event = self.prepare(trees)
        return TensorData(
            coords=event["coords"].astype(self.itype),
            features=event["shape"].astype(self.ftype)[:, None],
            meta=event["meta"],
        )
