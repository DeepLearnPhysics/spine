"""Aligned input tensors and semantic segmentation labels."""

from __future__ import annotations

from typing import Any

from spine.data import TensorData

from .base import SuperaParserBase

__all__ = ["SuperaTensorParser", "SuperaSegmentationParser"]


class SuperaTensorParser(SuperaParserBase):
    """Return voxelized truth-deposit energy in MeV."""

    name = "tensor"

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Return summed truth energy in MeV on the shared sparse grid.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by SuperaReader.

        Returns
        -------
        TensorData
            Sparse coordinates and summed deposit energy in MeV.
        """
        # Reuse the same grid and overlap winners as the input tensor.
        event = self.prepare(trees)
        return TensorData(
            coords=event["coords"].astype(self.itype),
            features=event["energy"].astype(self.ftype)[:, None],
            meta=event["meta"],
        )


class SuperaSegmentationParser(SuperaParserBase):
    """Return semantic labels in exactly the input voxel order."""

    name = "segmentation"

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Return five-class labels; unknown point semantics are rejected.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by SuperaReader.

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
