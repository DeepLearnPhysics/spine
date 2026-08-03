"""Typed object-list containers and event-level data products."""

from typing import Any

from .base import DataProduct

__all__ = ["ObjectList", "ObjectListData"]


class ObjectList(list):
    """List carrying a representative object for empty-list typing.

    Python lists do not retain their intended element type when empty. The
    ``default`` object gives serialization and downstream processing a
    concrete representative without inserting it into the list itself.

    Unlike :class:`ObjectListData`, this runtime container does not claim that
    its members contain indexes requiring shifts during collation. It is used,
    for example, for final reconstructed particle lists whose identifiers are
    already expressed in their output namespace.

    Attributes
    ----------
    default : object
        Representative object used to type the list when it is empty.
    """

    def __init__(self, object_list: list[object], default: object) -> None:
        """Initialize the typed list.

        Parameters
        ----------
        object_list : list[object]
            Objects stored in the list.
        default : object
            Representative object used to preserve the element type.
        """
        # Initialize the underlying list, then retain its empty-list type hint
        super().__init__(object_list)
        self.default = default


class ObjectListData(ObjectList, DataProduct):
    """Objects for one event with index-shifting instructions.

    Unlike a runtime :class:`ObjectList`, this parser product explicitly
    describes how object index attributes must be shifted into the batched
    namespace during collation. A scalar shift applies uniformly, while a
    mapping can describe independent namespaces. ``default`` inherited from
    :class:`ObjectList` preserves the object type for empty events.

    Attributes
    ----------
    index_shifts : int or dict[str, int]
        Shift(s) to apply to object index attributes during collation.
    """

    product_type = "object_list"

    def __init__(
        self,
        object_list: list[Any],
        default: Any,
        index_shifts: int | dict[str, int] | None = None,
    ) -> None:
        """Initialize the list and the default value.

        Parameters
        ----------
        object_list : list[Any]
            Parsed objects associated with one event entry.
        default : Any
            Default object used to type an empty list.
        index_shifts : int or dict[str, int], optional
            Shift(s) to apply to object index attributes during batching.
        """
        # Initialize the underlying object list
        super().__init__(object_list, default)

        # Store the index shifts
        if index_shifts is not None:
            self.index_shifts = index_shifts
        else:
            self.index_shifts = len(object_list)

    @property
    def to_object_list(self) -> ObjectList:
        """Drop batching metadata and return a plain ObjectList.

        Returns
        -------
        ObjectList
            Underlying object list without ``index_shifts`` metadata.
        """
        return ObjectList(self, default=self.default)
