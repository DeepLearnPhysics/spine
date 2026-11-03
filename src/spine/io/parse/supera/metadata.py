"""Shared image metadata and native event/particle provenance."""

from __future__ import annotations

from typing import Any

import numpy as np

from spine.data import Meta, TensorData

from ..base import ParserBase
from .base import SuperaParserBase

__all__ = ["SuperaMetaParser", "SuperaSourceIDsParser", "SuperaEventInfoParser"]


class SuperaMetaParser(SuperaParserBase):
    """Expose the shared image transform for truth construction in cm."""

    name = "meta"

    def __call__(self, trees: dict[str, Any]) -> Meta:
        """Return the metadata shared by all native spatial products.

        Parameters
        ----------
        trees : dict[str, Any]
            Projected native event products.

        Returns
        -------
        Meta
            Detector bounds and voxel dimensions in centimetres.
        """
        return self.prepare(trees)["meta"]


class SuperaSourceIDsParser(SuperaParserBase):
    """Preserve sparse source IDs separately from internal particle indexes."""

    name = "source_ids"

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Return one source identifier per particle-table row.

        Parameters
        ----------
        trees : dict[str, Any]
            Projected native event products.

        Returns
        -------
        TensorData
            Integer provenance IDs; batching never shifts these identifiers.
        """
        return TensorData(
            features=self.prepare(trees)["source_ids"][:, None],
            feature_fields={"source_id": (0,)},
        )


class SuperaEventInfoParser(ParserBase):
    """Preserve native event/job identities without fabricating run/subrun IDs."""

    name = "event_info"

    def __init__(self, dtype: str) -> None:
        """Request the small event-identity product.

        Parameters
        ----------
        dtype : str
            Dataset floating-point dtype; identities retain integer precision.
        """
        super().__init__(dtype, identity_event="events")

    def __call__(self, trees: dict[str, Any]) -> TensorData:
        """Return all stored identity fields as a named integer tensor.

        Parameters
        ----------
        trees : dict[str, Any]
            Projected native event products.

        Returns
        -------
        TensorData
            One row containing the file's event, input and job identities.
        """
        fields = sorted(trees["events"])
        return TensorData(
            features=np.array(
                [[trees["events"][key] for key in fields]], dtype=np.int64
            ),
            feature_fields={key: (column,) for column, key in enumerate(fields)},
        )
