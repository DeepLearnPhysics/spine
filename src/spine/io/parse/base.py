"""Shared parser base classes and input-data plumbing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class ParserBase(ABC):
    """Abstract parent class of all parser classes.

    Provides basic functionality shared by all parsers:
    1. Defines a :meth:`__call__` function shared by all classes

    Attributes
    ----------
    name : str
        Name of the parser
    aliases : List[str]
        Aliases of the parser (allowed but disfavored names)
    data_map : dict[str, str]
        Maps function parameter names onto a file data product name
    tree_keys : List[str]
        List of file data product name
    """

    # Name of the parser (as specified in the configuration)
    name: ClassVar[str | None] = None

    # Alternative allowed names of the parser
    aliases: ClassVar[tuple[str, ...]] = ()

    # Type of object(s) returned by the parser
    returns: ClassVar[str | None] = None

    # Overlay method for the objects returned by the parser
    overlay: ClassVar[str | None] = None

    # List of recognized data type returns
    _data_types: ClassVar[tuple[str, ...]] = (
        "tensor",
        "object",
        "object_list",
        "scalar",
    )

    def __init__(self, dtype: str, **kwargs: Any) -> None:
        """Register parser configuration and input tree requirements.

        Parameters
        ----------
        dtype : str
            Floating-point dtype used by the parser outputs.
        **kwargs : dict, optional
            Parser configuration. Any key ending in ``_event`` or
            ``_event_list`` is interpreted as the name of one or more input
            tree products required by the parser.

        Notes
        -----
        Tree-product arguments must contain either the ``_event`` or
        ``_event_list`` suffix.
        """
        # Store the type in which the parsers should return their data
        self.ftype = dtype
        self.itype = dtype.replace("float", "int")

        # Do a self-consistency check on return data types
        assert self.returns in self._data_types, (
            f"Parser return data type not recognized for '{self.name}': "
            f"{self.returns}. Should be one of {self._data_types}."
        )

        # Find data keys, append them to the map
        self.data_map = {}
        self.tree_keys = []
        for key, value in kwargs.items():
            if "_event" not in key:
                class_name = self.__class__.__name__
                raise TypeError(f"{class_name} got an unexpected argument: {key}.")

            if value is not None:
                self.data_map[key] = value
                if not isinstance(value, list):
                    if value not in self.tree_keys:
                        self.tree_keys.append(value)

                else:
                    for v in value:
                        if v not in self.tree_keys:
                            self.tree_keys.append(v)

    def get_input_data(self, trees: dict[str, Any]) -> dict[str, Any]:
        """Build the parser-call input dictionary from loaded tree products.

        Parameters
        ----------
        trees : dict
            Mapping from data-product names to loaded event objects.

        Returns
        -------
        dict
            Mapping from parser argument names to the objects that should be
            passed to :meth:`process` or :meth:`__call__`.
        """
        # Build the input to the parser function
        data_dict = {}
        for key, value in self.data_map.items():
            if isinstance(value, str):
                if value not in trees:
                    raise ValueError(f"Must provide {value} for parser `{self.name}`.")
                data_dict[key] = trees[value]

            elif isinstance(value, list):
                for v in value:
                    if v not in trees:
                        raise ValueError(f"Must provide {v} for parser `{self.name}`.")
                data_dict[key] = [trees[v] for v in value]

        return data_dict

    @abstractmethod
    def __call__(self, trees: dict[str, Any]) -> Any:
        """Parse one event entry into a canonical SPINE parser product.

        Parameters
        ----------
        trees : dict
            Mapping from data-product names to loaded event objects.
        """
        raise NotImplementedError("Must define `__call__` method.")
