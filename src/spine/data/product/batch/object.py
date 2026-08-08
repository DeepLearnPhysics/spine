"""Batched object-list data product."""

from __future__ import annotations

from collections.abc import Iterable

from ..object import ObjectListData

__all__ = ["ObjectListBatch"]


class ObjectListBatch(list[ObjectListData]):
    """Sequence containing one self-describing object list per event.

    Subclassing :class:`list` preserves the historical consumer interface
    while making the event/batch relationship explicit in the type system.

    Parameters
    ----------
    entries : iterable[ObjectListData]
        One typed object list per event.
    """

    def __init__(self, entries: Iterable[ObjectListData]) -> None:
        """Initialize the batch from event-level object-list products.

        Raises
        ------
        TypeError
            If any entry is not an :class:`ObjectListData`.
        """
        entries = list(entries)
        if not all(isinstance(entry, ObjectListData) for entry in entries):
            raise TypeError("ObjectListBatch entries must be ObjectListData objects.")
        super().__init__(entries)

    @property
    def batch_size(self) -> int:
        """Return the number of events in the batch."""
        return len(self)

    def event(self, batch_id: int) -> ObjectListData:
        """Return one event-level object list.

        Parameters
        ----------
        batch_id : int
            Event position in the batch.
        """
        return self[batch_id]
