"""Lightweight parsers for cached HDF5 object products."""

from __future__ import annotations

from typing import Any

from spine.data import ObjectList

from ..base import ParserBase
from ..data import ParserObjectList

__all__ = ["HDF5ObjectParser", "HDF5ObjectListParser"]


class HDF5ObjectParser(ParserBase):
    """Return one cached HDF5 object as-is.

    The HDF5 reader already rebuilds stored SPINE classes when
    ``build_classes=True``. This parser simply forwards the reconstructed
    object into the dataset schema layer.
    """

    name = "object"
    returns = "object"

    def __call__(self, trees: dict[str, Any]) -> Any:
        """Parse one cached object entry.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        object
            Reconstructed cached object.
        """
        return self.process(**self.get_input_data(trees))

    def process(self, object_event: Any) -> Any:
        """Return one reconstructed cached object.

        Parameters
        ----------
        object_event : object
            Reconstructed object loaded by :class:`HDF5Reader`.

        Returns
        -------
        object
            Input object unchanged.
        """
        return object_event


class HDF5ObjectListParser(ParserBase):
    """Wrap one cached HDF5 object list into a :class:`ParserObjectList`.

    This parser expects the HDF5 reader to have already reconstructed each
    element of the list as a local SPINE data object. It preserves an incoming
    :class:`ObjectList` default when available and otherwise infers the default
    from the first element of the list.
    """

    name = "object_list"
    returns = "object_list"

    def __call__(self, trees: dict[str, Any]) -> ParserObjectList:
        """Parse one cached object-list entry.

        Parameters
        ----------
        trees : dict
            Mapping from configured HDF5 product names to cached entry values.

        Returns
        -------
        ParserObjectList
            Parsed object list with batching metadata support.
        """
        return self.process(**self.get_input_data(trees))

    def process(self, object_list_event: ObjectList | list[Any]) -> ParserObjectList:
        """Normalize one reconstructed cached object list.

        Parameters
        ----------
        object_list_event : ObjectList or list[object]
            Reconstructed cached objects for one entry.

        Returns
        -------
        ParserObjectList
            Parsed object list.

        Raises
        ------
        ValueError
            If the cached list is empty and does not carry a default object to
            preserve its intended type.
        """
        if isinstance(object_list_event, ObjectList):
            return ParserObjectList(list(object_list_event), object_list_event.default)

        if len(object_list_event):
            return ParserObjectList(
                list(object_list_event), type(object_list_event[0])()
            )

        raise ValueError(
            "Cannot infer the default type of an empty cached object list. "
            "Store object lists with preserved typing or ensure the list is "
            "non-empty."
        )
