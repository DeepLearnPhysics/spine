"""Compact fragmentation labels with event-local particle relationships."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import ClusterLabelData

from .base import SuperaParserBase

__all__ = ["SuperaClusterLabelParser"]


class SuperaClusterLabelParser(SuperaParserBase):
    """Return fragment IDs and the winning fragment's particle table row."""

    name = "cluster_label"

    def __call__(self, trees: dict[str, Any]) -> ClusterLabelData:
        """Build compact fragment supervision with instance and ancestor links.

        Parameters
        ----------
        trees : dict[str, Any]
            Raw event products returned by SuperaReader.

        Returns
        -------
        ClusterLabelData
            Compact fragment labels and a named particle-truth sidecar.
        """
        # Reuse the same grid and overlap winners as the input tensor.
        event = self.prepare(trees)
        # The winning fragment row is also the compact particle association.
        return ClusterLabelData(
            coords=event["coords"].astype(self.itype),
            features=np.column_stack(
                (event["energy"], event["fragment"], event["fragment"])
            ).astype(self.ftype),
            particles=event["labels"],
            meta=event["meta"],
            remove_duplicates=False,
        )
