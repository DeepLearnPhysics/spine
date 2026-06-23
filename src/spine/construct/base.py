"""Base class for all data representation builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from spine.data import ObjectList
from spine.data.out.base import RecoBase, TruthBase
from spine.utils.docstring import merge_ancestor_docstrings

from .utils import BuildMode, RunMode, Units, get_batch_size, is_single_index


class BuilderBase(ABC):
    """Abstract base class for building all data structures

    A Builder class takes input data and full chain result dictionaries
    and processes them into human-readable data structures.
    """

    # Builder name
    name: ClassVar[str | None] = None

    # List of recognized run modes and units
    _run_modes: ClassVar[tuple[RunMode, ...]] = ("reco", "truth", "both", "all")
    _units: ClassVar[tuple[Units, ...]] = ("cm", "px")

    # Types of objects constructed by the builder
    _reco_type: ClassVar[type[RecoBase] | None] = None
    _truth_type: ClassVar[type[TruthBase] | None] = None

    # Necessary/optional data products to build a reconstructed object
    _build_reco_keys = (
        ("points", True),
        ("depositions", True),
        ("sources", False),
        ("orig_index", False),
    )

    # Necessary/optional data products to build a truth object
    _build_truth_keys = (
        ("label_tensor", True),
        ("label_adapt_tensor", False),
        ("label_g4_tensor", False),
        ("points_label", True),
        ("points", False),
        ("points_g4", False),
        ("depositions_label", True),
        ("depositions", False),
        ("depositions_q_label", False),
        ("depositions_g4", False),
        ("sources_label", False),
        ("sources", False),
        ("orig_index_label", False),
    )

    # Necessary/optional data products to load a reconstructed object
    _load_reco_keys = (("points", False), ("depositions", False), ("sources", False))

    # Necessary/optional data products to load a truth object
    _load_truth_keys = (
        ("points_label", False),
        ("points", False),
        ("points_g4", False),
        ("depositions_label", False),
        ("depositions", False),
        ("depositions_q_label", False),
        ("depositions_g4", False),
        ("sources_label", False),
        ("sources", False),
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically merge parent docstrings when BuilderBase is subclassed."""
        super().__init_subclass__(**kwargs)
        merge_ancestor_docstrings(cls)

    def __init__(
        self,
        mode: RunMode,
        units: Units,
    ) -> None:
        """Initializes the builder.

        Parameters
        ----------
        mode : str, default 'both'
            Whether to construct reconstructed objects, true objects or both
            (one of 'reco', 'truth', 'both' or 'all')
        units : str, default 'cm'
            Units in which the position arguments of the constructed objects
            should be expressed (one of 'cm' or 'px')
        """
        if mode not in self._run_modes:
            raise ValueError(
                f"Run mode not recognized: {mode}. Must be one {self._run_modes}"
            )
        if units not in self._units:
            raise ValueError(
                f"Units not recognized: {units}. Must be one {self._units}"
            )

        # Store the mode and units
        self.mode = mode
        self.units = units

    def __call__(self, data: dict[str, Any]) -> None:
        """Build representations for a batch of data.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Dispatch
        dispatch_modes: tuple[tuple[BuildMode, BuildMode], ...] = (
            ("reco", "truth"),
            ("truth", "reco"),
        )
        for mode, avoid in dispatch_modes:
            out_key = f"{mode}_{self.name}s"
            if self.mode != avoid:
                if is_single_index(data["index"]):
                    # Single entry to process
                    data[out_key] = self.process(data, mode)

                else:
                    # Batch of data to process
                    const_list: list[ObjectList] = []
                    for entry in range(get_batch_size(data["index"])):
                        const_list.append(self.process(data, mode, entry))
                    data[out_key] = const_list

    def process(
        self,
        data: dict[str, Any],
        mode: BuildMode,
        entry: int | None = None,
    ) -> ObjectList:
        """Build representations for a single entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        mode : str
            Type of object to reconstruct ('reco' or 'truth')
        entry : int, optional
            Entry to process
        """
        # Dispatch to the appropriate function
        key = f"{mode}_{self.name}s"
        if key in data:
            func = f"load_{mode}"
        else:
            func = f"build_{mode}"

        result = self.construct(func, data, entry)

        # When loading, check that the units are as expected
        if "load" in func:
            self.check_units(data, key, entry)

        return result

    def check_units(
        self, data: dict[str, Any], key: str, entry: int | None = None
    ) -> None:
        """Checks that the objects in the list are expressed in the
        appropriate units. Convert them otherwise.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        key : str
            Dictionary key corresponding to the objects to convert
        entry : int, optional
            Entry to process
        """
        objects = data[key][entry] if entry is not None else data[key]
        for obj in objects:
            if obj.units != self.units:
                if "meta" not in data:
                    raise KeyError("Cannot convert units without metadata information.")
                meta = data["meta"][entry] if entry is not None else data["meta"]
                getattr(obj, f"to_{self.units}")(meta)

    def construct(
        self, func: str, data: dict[str, Any], entry: int | None = None
    ) -> ObjectList:
        """Prepares the input based on the required data and runs constructor.

        Parameters
        ----------
        func : str
            Build function name
        data : dict
            Dictionary of data products
        entry : int, optional
            Entry to process

        Returns
        -------
        List[object]
            List of constructed objects
        """
        # Get the description of the fields needed by this source object
        input_data = {}
        method, dtype = func.split("_")
        keys = getattr(self, f"_{func}_keys")
        for key, req in keys:
            # If the field has no default value, must be provided
            if req and key not in data:
                raise KeyError(
                    f"Must provide `{key}` data product to {method} the "
                    f"{dtype} {self.name}s."
                )

            if key in data:
                if entry is not None:
                    input_data[key] = data[key][entry]
                else:
                    input_data[key] = data[key]

        obj_list = getattr(self, func)(input_data)
        default = getattr(self, f"_{dtype}_type")()

        return ObjectList(obj_list, default)

    @abstractmethod
    def build_reco(self, data: dict[str, Any]) -> Sequence[RecoBase]:
        """Place-holder for a method used to build reconstructed objects.

        Parameters
        ----------
        data : dict
            Dictionary which contains the necessary data products
        """
        raise NotImplementedError

    @abstractmethod
    def build_truth(self, data: dict[str, Any]) -> Sequence[TruthBase]:
        """Place-holder for a method used to build truth objects.

        Parameters
        ----------
        data : dict
            Dictionary which contains the necessary data products
        """
        raise NotImplementedError

    @abstractmethod
    def load_reco(self, data: dict[str, Any]) -> Sequence[RecoBase]:
        """Place-holder for a method used to load reconstructed objects.

        Parameters
        ----------
        data : dict
            Dictionary which contains the necessary data products
        """
        raise NotImplementedError

    @abstractmethod
    def load_truth(self, data: dict[str, Any]) -> Sequence[TruthBase]:
        """Place-holder for a method used to load truth objects.

        Parameters
        ----------
        data : dict
            Dictionary which contains the necessary data products
        """
        raise NotImplementedError
