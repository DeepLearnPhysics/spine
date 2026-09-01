"""In-memory schema state for one staged HDF5 product namespace."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..hdf5 import HDF5Writer

__all__ = ["StageState"]


@dataclass
class StageState:
    """Describe the serialization schema and progress of one stage.

    A regular :class:`HDF5Writer` maintains one flat schema for an output
    file. A staged cache instead owns an independent schema under each named
    stage, so the writer stores this state while switching between stages.

    Attributes
    ----------
    keys : set[str]
        Public and private product keys persisted by the stage.
    type_dict : dict[str, HDF5Writer.DataFormat]
        Physical HDF5 representation inferred for each product.
    object_dtypes : list[list[tuple[str, type]]]
        Structured object dtypes used by the inherited serialization backend.
    product_metadata : dict[str, dict[str, Any]]
        Typed-product reconstruction metadata.
    product_children : dict[str, tuple[str, str]]
        Private child products and their owning public product.
    event_dtype : numpy.dtype or list, optional
        Compound event-axis dtype created for this stage.
    entries_since_flush : int, default 0
        Entries appended since the most recent configured flush.
    """

    keys: set[str]
    type_dict: dict[str, HDF5Writer.DataFormat]
    object_dtypes: list[list[tuple[str, type]]]
    product_metadata: dict[str, dict[str, Any]]
    product_children: dict[str, tuple[str, str]]
    event_dtype: np.dtype | list[tuple[str, Any]] | None = None
    entries_since_flush: int = 0
