"""Contains base class of all post-processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any


class PostBase(ABC):
    """Base class of all post-processors.

    This base class performs the following functions:
      - Ensures that the necessary methods exist
      - Checks that the post-processor is provided the necessary information
        to do its job
      - Fetches the appropriate coordinate attributes
      - Ensures that the appropriate units are provided

    Attributes
    ----------
    name : str
        Name of the post-processor as defined in the configuration file
    aliases : tuple[str, ...]
        Alternative acceptable names for a post-processor
    """

    # Name of the post-processor (as specified in the configuration)
    name = None

    # Alternative allowed names of the post-processor
    aliases = ()

    # Units in which the post-processor expects objects to be expressed in
    units = "cm"

    # Whether this post-processor might use paths relative to the parent configuration path
    provide_parent_path = False

    # Set of data keys needed for this post-processor to operate
    _keys = ()

    # Set of post-processors which must be run before this one is
    _upstream = ()

    # List of recognized object types
    _obj_types = ("fragment", "particle", "interaction")

    # List of recognized run modes
    _run_modes = ("reco", "truth", "both", "all")

    # List of known reconstructed particle identification modes
    _pid_modes = ("pid", "chi2_pid")

    # List of known point modes for true particles and their corresponding keys
    _point_modes = (
        ("points", "points_label"),
        ("points_adapt", "points"),
        ("points_g4", "points_g4"),
    )

    # List of known source modes for true particles and their corresponding keys
    _source_modes = (
        ("sources", "sources_label"),
        ("sources_adapt", "sources"),
        ("sources_g4", "sources_g4"),
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
        truth_dep_mode: str | None = None,
        pid_mode: str | None = None,
        parent_path: str | None = None,
    ) -> None:
        """Initialize default post-processor object properties.

        Parameters
        ----------
        obj_type : str or sequence[str], optional
            Name or list of names of the object types to process
        run_mode : str, optional
            If specified, tells whether the post-processor must run on
            reconstructed ('reco'), true ('truth') or both objects
            ('both' or 'all')
        truth_point_mode : str, optional
            If specified, tells which attribute of the :class:`TruthFragment`,
            :class:`TruthParticle` or :class:`TruthInteraction` object to use
            to fetch its point coordinates
        truth_dep_mode : str, optional
            If specified, tells which attribute of the :class:`TruthFragment`,
            :class:`TruthParticle` or :class:`TruthInteraction` object to use
            to fetch its depositions
        parent_path : str, optional
            Path to the parent directory of the main analysis configuration. This
            allows for the use of relative paths in the post-processors.
        """
        # If run mode is specified, process it
        if run_mode is not None:
            # Check that the run mode is recognized
            if run_mode not in self._run_modes:
                raise ValueError(
                    f"`run_mode` not recognized: {run_mode}. Must be one of "
                    f"{self._run_modes}."
                )

        # Check that all the object sources are recognized
        if obj_type is None:
            obj_types: list[str] = []
        elif isinstance(obj_type, str):
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

        # Make a list of object keys to process
        self.obj_type = obj_types
        self.fragment_keys: list[str] = []
        self.particle_keys: list[str] = []
        self.interaction_keys: list[str] = []
        for name in self._obj_types:
            # Initialize one list per object type
            setattr(self, f"{name}_keys", [])

            # Skip object types which are not requested
            if name in obj_types:
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
            self.truth_point_key = self.point_modes[self.truth_point_mode]
            self.truth_source_mode = truth_point_mode.replace("points", "sources")
            self.truth_source_key = self.source_modes[self.truth_source_mode]
            self.truth_index_mode = truth_point_mode.replace("points", "index")

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

        # If a PID mode is specified, store it
        if pid_mode is not None:
            if pid_mode not in self._pid_modes:
                raise ValueError(
                    f"The `pid_mode` argument must be one of {self._pid_modes}. "
                    f"Got {pid_mode} instead."
                )
            self.pid_mode = pid_mode

        # Store the parent path
        self.parent_path = parent_path

    @property
    def keys(self) -> dict[str, bool]:
        """Dictionary of (key, necessity) pairs which determine which data keys
        are needed/optional for the post-processor to run.

        Returns
        -------
        dict[str, bool]
            Dictionary of (key, necessity) pairs to be used
        """
        return dict(self._keys)

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
    def source_modes(self) -> dict[str, str]:
        """Dictionary which maps the name of a true
        object source attribute with the underlying source tensor it points to.

        Returns
        -------
        dict[str, str]
            Dictionary of (attribute, key) mapping for point sources
        """
        return dict(self._source_modes)

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

    def update_upstream(self, key: str) -> None:
        """Update the underlying set of required upstream modules in place.

        Parameters
        ----------
        key : str
            Post-processor module name to add to the list
        """
        self._upstream = (*self._upstream, key)

    def __call__(
        self, data: Mapping[str, Any], entry: int | None = None
    ) -> dict[str, Any] | None:
        """Calls the post processor on one entry.

        Parameters
        ----------
        data : dict
            Dicitionary of data products
        entry : int, optional
            Entry in the batch

        Returns
        -------
        dict
            Update to the input dictionary
        """
        # Fetch the input dictionary
        data_filter = {}
        for key, req in self._keys:
            # If this key is needed, check that it exists
            if req and key not in data:
                raise KeyError(
                    f"Post-processor `{self.name}` is missing an essential "
                    f"input to be used: `{key}`."
                )

            # Append
            if key in data:
                data_filter[key] = data[key]
                if entry is not None:
                    data_filter[key] = data[key][entry]

        # Run the post-processor
        return self.process(data_filter)

    def get_index(self, obj: Any) -> Any:
        """Get a certain pre-defined index attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects index are obtained using the
        `truth_index_mode` attribute of the class.

        Parameters
        ----------
        obj : FragmentBase, ParticleBase or InteractionBase
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
        obj : FragmentBase, ParticleBase or InteractionBase
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

    def get_sources(self, obj: Any) -> Any:
        """Get a certain pre-defined sources attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects sources are obtained using the
        `truth_source_mode` attribute of the class.

        Parameters
        ----------
        obj : FragmentBase, ParticleBase or InteractionBase
            Fragment, Particle or Interaction object

        Results
        -------
        np.ndarray
            (N, 2) Object sources
        """
        if not obj.is_truth:
            return obj.sources
        else:
            return getattr(obj, self.truth_source_mode)

    def get_depositions(self, obj: Any) -> Any:
        """Get a certain pre-defined deposition attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects points are obtained using the
        `truth_dep_mode` attribute of the class.

        Parameters
        ----------
        obj : FragmentBase, ParticleBase or InteractionBase
            Fragment, Particle or Interaction object

        Results
        -------
        np.ndarray
            (N) Depositions
        """
        if not obj.is_truth:
            return obj.depositions
        else:
            return getattr(obj, self.truth_dep_mode)

    def get_pid(self, obj: Any) -> Any:
        """Get a certain pre-defined PID prediction of an object.

        The :class:`TruthParticle` PID predictions are obtained using the
        `pid_mode` attribute of the class.

        Parameters
        ----------
        obj : ParticleBase
            Particle object

        Results
        -------
        int
            Particle identification enumerator
        """
        if not obj.is_truth:
            return getattr(obj, self.pid_mode)
        else:
            return obj.pid

    def check_units(self, obj: Any) -> None:
        """Check that the point coordinates of an object are as expected.

        Parameters
        ----------
        obj : FragmentBase, ParticleBase or InteractionBase
            Particle or interaction object

        Results
        -------
        np.ndarray
            (N, 3) Point coordinates
        """
        if obj.units != self.units:
            raise ValueError(
                f"Coordinates must be expressed in {self.units} but are "
                f"currently in {obj.units} instead."
            )

    @abstractmethod
    def process(self, data: MutableMapping[str, Any]) -> dict[str, Any] | None:
        """Place-holder method to be defined in each post-processor.

        Parameters
        ----------
        data : dict
            Dictionary of processed data products
        """
        raise NotImplementedError("Must define the `process` function.")
