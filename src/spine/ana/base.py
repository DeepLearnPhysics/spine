"""Base class of all analysis scripts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any
from warnings import warn

from spine.io.write.csv import CSVWriter


class AnaBase(ABC):
    """Parent class for CSV-writing analysis scripts.

    This base class performs the following functions:
    - Ensures that the necessary methods exist
    - Checks that the script is provided the necessary information
    - Writes the output of the analysis to CSV

    Attributes
    ----------
    name : str
        Name of the analysis script (to call it from a configuration file)
    keys : dict
        Data products needed or optionally consumed by the analysis script
    units : str
        Units in which the coordinates are expressed
    """

    # Name of the analysis script (as specified in the configuration)
    name = None

    # Alternative allowed names of the analysis script
    aliases = ()

    # Units in which the analysis script expects objects to be expressed in
    units = "cm"

    # Set of data keys needed for this analysis script to operate
    _keys = ()

    # List of recognized object types
    _obj_types = ("fragment", "particle", "interaction")

    # Valid run modes
    _run_modes = ("reco", "truth", "both", "all")

    # List of known point modes for true particles and their corresponding keys
    _point_modes = (
        ("points", "points_label"),
        ("points_adapt", "points"),
        ("points_g4", "points_g4"),
    )

    # List of known deposition modes for true particles and their corresponding keys
    _dep_modes = (
        ("depositions", "depositions_label"),
        ("depositions_q", "depositions_q_label"),
        ("depositions_adapt", "depositions_label_adapt"),
        ("depositions_adapt_q", "depositions"),
        ("depositions_g4", "depositions_g4"),
    )

    def __init__(
        self,
        obj_type: str | Sequence[str] | None = None,
        run_mode: str | None = None,
        truth_point_mode: str | None = None,
        truth_index_mode: str | None = None,
        truth_dep_mode: str | None = None,
        append: bool = False,
        overwrite: bool = False,
        log_dir: str | None = None,
        prefix: str | None = None,
        buffer_size: int = 1,
    ) -> None:
        """Initialize default analysis script object properties.

        Parameters
        ----------
        obj_type : str or Sequence[str], optional
            Name or list of names of the object types to process
        run_mode : str, optional
            If specified, tells whether the analysis script must run on
            reconstructed ('reco'), true ('true') or both objects
            ('both' or 'all')
        truth_point_mode : str, optional
            If specified, tells which attribute of the :class:`TruthFragment`,
            :class:`TruthParticle` or :class:`TruthInteraction` object to use
            to fetch its point coordinates
        truth_index_mode : str, optional
            If specified, tells which attribute of the :class:`TruthFragment`,
            :class:`TruthParticle` or :class:`TruthInteraction` object to use
            to fetch its index
        truth_dep_mode : str, optional
            If specified, tells which attribute of the :class:`TruthFragment`,
            :class:`TruthParticle` or :class:`TruthInteraction` object to use
            to fetch its energy depositions
        append : bool, default False
            If True, appends existing CSV files instead of creating new ones
        overwrite : bool, default False
            If True and an output CSV file exists, overwrite it
        log_dir : str, optional
            Output CSV file directory (shared with driver log)
        prefix : str, default None
            Name to prefix every output CSV file with
        buffer_size : int, default 1
            CSV file buffer size. 1 is line buffered (safe default),
            -1 uses system default, 0 is unbuffered, >1 is buffer size in bytes
        """
        # Initialize default keys
        self.update_keys(
            {
                "index": True,
                "file_index": True,
                "file_entry_index": False,
                "run_info": False,
            }
        )

        # If run mode is specified, process it
        self.run_mode = run_mode
        if run_mode is not None:
            # Check that the run mode is recognized
            if run_mode not in self._run_modes:
                raise ValueError(
                    f"`run_mode` not recognized: {run_mode}. Must be one of "
                    f"{self._run_modes}."
                )

        self.prefixes: list[str] = []
        if run_mode != "truth":
            self.prefixes.append("reco")
        if run_mode != "reco":
            self.prefixes.append("truth")

        # Check that all the object sources are recognized
        obj_types: list[str] | None = None
        if obj_type is not None:
            if isinstance(obj_type, str):
                obj_types = [obj_type]
            elif isinstance(obj_type, Sequence):
                obj_types = list(obj_type)
            else:
                raise TypeError("`obj_type` must be a string or sequence of strings.")
            for obj in obj_types:
                if obj not in self._obj_types:
                    raise ValueError(
                        f"Object type must be one of {self._obj_types}. Got "
                        f"`{obj}` instead."
                    )
        self.obj_type = obj_types

        # Make a list of object keys to process
        self.fragment_keys: list[str] = []
        self.particle_keys: list[str] = []
        self.interaction_keys: list[str] = []
        for name in self._obj_types:
            # Initialize one list per object type
            setattr(self, f"{name}_keys", [])

            # Skip object types which are not requested
            if obj_types is not None and name in obj_types:
                if run_mode != "truth":
                    getattr(self, f"{name}_keys").append(f"reco_{name}s")
                if run_mode != "reco":
                    getattr(self, f"{name}_keys").append(f"truth_{name}s")

        self.obj_keys: list[str] = (
            self.fragment_keys + self.particle_keys + self.interaction_keys
        )

        # Update underlying keys, if needed
        self.update_keys({k: True for k in self.obj_keys})

        # If a truth point mode is specified, store it
        if truth_point_mode is not None:
            if truth_point_mode not in self.point_modes:
                raise ValueError(
                    "The `truth_point_mode` argument must be one of "
                    f"{self.point_modes.keys()}. Got `{truth_point_mode}` instead."
                )
            self.truth_point_mode = truth_point_mode
            self.truth_point_key = self.point_modes[truth_point_mode]
            self.truth_index_mode = truth_point_mode.replace("points", "index")

        # If a truth index mode is specified, store it
        if truth_index_mode is not None:
            self.truth_index_mode = truth_index_mode

        # If a truth deposition mode is specified, store it
        if truth_dep_mode is not None:
            if truth_dep_mode not in self.dep_modes:
                raise ValueError(
                    "The `truth_dep_mode` argument must be one of "
                    f"{self.dep_modes.keys()}. Got `{truth_dep_mode}` instead."
                )
            if truth_point_mode is not None:
                prefix = truth_point_mode.replace("points", "depositions")
                if not truth_dep_mode.startswith(prefix):
                    raise ValueError(
                        f"Points mode {truth_point_mode} and deposition mode "
                        f"{truth_dep_mode} are incompatible."
                    )
            self.truth_dep_mode = truth_dep_mode
            self.truth_dep_key = self.dep_modes[truth_dep_mode]

        # Store the append flag
        self.append_file = append
        self.overwrite_file = overwrite

        # Initialize a writer dictionary to be filled by the children classes
        self.base_dict: dict[str, Any] = {}
        self.log_dir = log_dir
        self.output_prefix = prefix
        self.buffer_size = buffer_size
        self.writers: dict[str, CSVWriter] = {}

    def __del__(self) -> None:
        """Destructor to ensure CSV files are closed.

        This acts as a safety net in case close_writers() is not called
        explicitly. However, explicit cleanup is preferred.
        """
        self.close_writers()

    def close_writers(self) -> None:
        """Close all CSV writers and flush any remaining data.

        This should be called when the analysis is complete to ensure
        all data is written and files are properly closed.
        """
        for writer in getattr(self, "writers", {}).values():
            writer.close()

    def flush_writers(self) -> None:
        """Flush all CSV writer buffers without closing the files.

        This forces any buffered data to be written to disk. Useful
        for ensuring data persistence at checkpoints.
        """
        for writer in self.writers.values():
            writer.flush()

    def initialize_writer(self, name: str) -> None:
        """Adds a CSV writer to the list of writers for this script.

        Parameters
        ----------
        name : str
            Name of the writer
        """
        # Define the name of the file to write to
        if len(name) == 0:
            raise ValueError("Must provide a non-empty name.")
        file_name = f"{self.name}_{name}.csv"
        if self.output_prefix:
            file_name = f"{self.output_prefix}_{file_name}"
        if self.log_dir:
            file_name = f"{self.log_dir}/{file_name}"

        # Initialize the writer
        self.writers[name] = CSVWriter(
            file_name,
            append=self.append_file,
            overwrite=self.overwrite_file,
            buffer_size=self.buffer_size,
        )

    @property
    def keys(self) -> dict[str, bool]:
        """Dictionary of (key, necessity) pairs which determine which data keys
        are needed or optional for the analysis script to run.

        Returns
        -------
        dict[str, bool]
            Dictionary of (key, necessity) pairs to be used
        """
        return dict(self._keys)

    @keys.setter
    def keys(self, keys: Mapping[str, bool]) -> None:
        """Converts a dictionary of keys to an immutable tuple.

        Parameters
        ----------
        dict[str, bool]
            Dictionary of (key, necessity) pairs to be used
        """
        self._keys = tuple(keys.items())

    @property
    def point_modes(self) -> dict[str, str]:
        """Dictionary which maps the name of a true
        object point attribute with the underlying point tensor it points to.

        Returns
        -------
        dict[str, str]
            Dictionary of (attribute, key) mapping for point coordinates
        """
        return dict(self._point_modes)

    @property
    def dep_modes(self) -> dict[str, str]:
        """Dictionary which maps the name of a true
        object deposition attribute with the underlying deposition array it points to.

        Returns
        -------
        dict[str, str]
            Dictionary of (attribute, key) mapping for point depositions
        """
        return dict(self._dep_modes)

    def update_keys(self, update_dict: Mapping[str, bool]) -> None:
        """Update the underlying set of keys and their necessity in place.

        Parameters
        ----------
        update_dict : Mapping[str, bool]
            Dictionary of (key, necessity) pairs to update the keys with
        """
        if len(update_dict) > 0:
            keys = self.keys
            keys.update(update_dict)
            self._keys = tuple(keys.items())

    def get_base_dict(self, data: Mapping[str, Any]) -> dict[str, Any]:
        """Builds the entry information dictionary.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        dict
            Dictionary of information for this entry
        """
        # Extract basic information to store in every row
        base_dict = {"index": data["index"], "file_index": data["file_index"]}
        if "file_entry_index" in data:
            base_dict["file_entry_index"] = data["file_entry_index"]
        if "run_info" in data:
            base_dict.update(**data["run_info"].scalar_dict())
        else:
            warn("`run_info` is missing; will not be included in CSV file.")

        return base_dict

    def append(self, name: str, **kwargs: Any) -> None:
        """Append a row to a CSV log file.

        Parameters
        ----------
        name : str
            Name of the writer
        **kwargs : dict
            Dictionary of information to save to the writer
        """
        self.writers[name].append({**self.base_dict, **kwargs})

    def __call__(self, data: Mapping[str, Any], entry: int | None = None) -> Any:
        """Runs the analysis script on one entry.

        Parameters
        ----------
        data : dict
            Data dictionary for one entry

        Returns
        -------
        dict
            Update to the input dictionary
        """
        # Fetch the necessary information
        data_filter = {}
        for key, req in self.keys.items():
            # If this key is needed, check that it exists
            if req and key not in data:
                raise KeyError(
                    f"Analysis script `{self.name}` is missing an essential "
                    f"input to be used: `{key}`."
                )

            # Append
            if key in data:
                data_filter[key] = data[key]
                if entry is not None:
                    data_filter[key] = data[key][entry]

        # Fetch the base dictionary
        self.base_dict = self.get_base_dict(data_filter)

        # Run the analysis script
        return self.process(data_filter)

    def get_index(self, obj: Any) -> Any:
        """Get a certain pre-defined index attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects index are obtained using the
        `truth_index_mode` attribute of the class.

        Parameters
        ----------
        obj : FragmentBase or ParticleBase or InteractionBase
            Fragment, Particle or Interaction object

        Results
        -------
        np.ndarray
           (N) Object index
        """
        if not obj.is_truth:
            return obj.index
        else:
            return getattr(obj, self.truth_index_mode)

    def get_points(self, obj: Any) -> Any:
        """Get a certain pre-defined point attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects points are obtained using the
        `truth_point_mode` attribute of the class.

        Parameters
        ----------
        obj : FragmentBase or ParticleBase or InteractionBase
            Fragment, Particle or Interaction object

        Results
        -------
        np.ndarray
           (N, 3) Point coordinates
        """
        if not obj.is_truth:
            return obj.points
        else:
            return getattr(obj, self.truth_point_mode)

    @abstractmethod
    def process(self, data: MutableMapping[str, Any]) -> Any:
        """Place-holder method to be defined in each analysis script.

        Parameters
        ----------
        data : dict
            Filtered data dictionary for one entry
        """
        raise NotImplementedError("Must define the `process` function")
