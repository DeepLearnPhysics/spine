"""Optical prediction data structures produced by SPINE."""

from dataclasses import dataclass, field

import numpy as np

from spine.data.base import DataBase
from spine.data.decorator import stored_property
from spine.data.field import FieldMetadata

__all__ = ["FlashHypothesis"]


@dataclass(eq=False, repr=False)
class FlashHypothesis(DataBase):
    """Predicted optical response for one interaction in one optical volume.

    Unlike :class:`spine.data.Flash`, this object represents a prediction and
    therefore does not carry measured timing or position information. A
    match-specific prediction references the observed flash or flashes through
    :attr:`flash_ids`; standalone predictions leave this array empty.

    Attributes
    ----------
    id : int
        Index of the hypothesis in the event-level hypothesis collection.
    interaction_id : int
        ID of the interaction which produced the predicted light.
    volume_id : int
        Optical volume in which the response is predicted.
    flash_ids : np.ndarray
        IDs of observed flashes associated with this prediction. This may
        contain multiple IDs when measured flashes were merged before matching.
    score : float
        Match score associated with this prediction, if applicable.
    is_truth : bool
        Whether the source interaction is a truth interaction.
    pe_per_ch : np.ndarray
        Predicted number of photoelectrons per optical channel.
    total_pe : float
        Total predicted number of photoelectrons.
    """

    id: int = field(default=-1, metadata=FieldMetadata(index=True))
    interaction_id: int = -1
    volume_id: int = -1
    score: float = np.nan
    is_truth: bool = False
    flash_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32),
        metadata=FieldMetadata(dtype=np.int32),
    )
    pe_per_ch: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32),
        metadata=FieldMetadata(dtype=np.float32),
    )

    @property
    @stored_property
    def total_pe(self) -> float:
        """Total predicted photoelectron count."""
        return float(np.sum(self.pe_per_ch))

    @property
    def is_matched(self) -> bool:
        """Whether this prediction is associated with an observed flash."""
        return len(self.flash_ids) > 0
