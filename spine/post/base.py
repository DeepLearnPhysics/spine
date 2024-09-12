"""Contains base class of all post-processors."""

from abc import ABC, abstractmethod


class PostBase(ABC):
    """Base class of all post-processors.

    This base class performs the following functions:
      - Ensures that the necessary method exist
      - Checks that the post-processor is provided the necessary information
        to do its job
      - Fetches the appropriate coordinate attributes
      - Ensures that the appropriate units are provided

    Attributes
    ----------
    name : str
        Name of the post-processor as defined in the configuration file
    aliases : []
        Alternative acceptable names for a post-processor
    parent_path : str
        Path to the main configuration file (to access relative configurations)
    keys : Dict[str, bool]
        List of data product keys used to operate the post-processor
    truth_point_mode : str
        Type of `points` attribute to use for the truth particles
    units : str
        Units in which the objects must be expressed (one of 'px' or 'cm')
    """
    name = None
    aliases = ()
    parent_path = ''
    keys = None
    truth_point_mode = 'points'
    units = 'cm'

    # List of recognized object types
    _obj_types = ('fragment', 'particle', 'interaction')

    # List of recognized run modes
    _run_modes = ('reco', 'truth', 'both', 'all')

    # List of known point modes for true particles
    _point_modes = {
            'points': 'points_label',
            'points_adapt': 'points',
            'points_g4': 'points_g4'
    }

    # List of known deposition modes
    _dep_modes = ('depositions', 'depositions_q', 'depositions_adapt',
                  'depositions_adapt_q', 'depositions_g4')

    def __init__(self, obj_type=None, run_mode=None, truth_point_mode=None,
                 truth_dep_mode=None, parent_path=None):
        """Initialize default post-processor object properties.

        Parameters
        ----------
        obj_type : Union[str, List[str]]
            Name or list of names of the object types to process
        run_mode : str, optional
            If specified, tells whether the post-processor must run on
            reconstructed ('reco'), true ('true') or both objects
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
        # Initialize default keys
        if self.keys is None:
            self.keys = {}

        # If run mode is specified, process it
        if run_mode is not None:
            # Check that the run mode is recognized
            assert run_mode in self._run_modes, (
                    f"`run_mode` not recognized: {run_mode}. Must be one of "
                    f"{self._run_modes}.")

        # Check that all the object sources are recognized
        if obj_type is None:
            obj_type = []
        elif isinstance(obj_type, str):
            obj_type = [obj_type]

        for obj in obj_type:
            assert obj in self._obj_types, (
                    f"Object type must be one of {self._obj_types}. Got "
                    f"`{obj}` instead.")

        # Make a list of object keys to process
        for name in self._obj_types:
            # Initialize one list per object type
            setattr(self, f'{name}_keys', [])

            # Skip object types which are not requested
            if name in obj_type:
                if run_mode != 'truth':
                    getattr(self, f'{name}_keys').append(f'reco_{name}s')
                if run_mode != 'reco':
                    getattr(self, f'{name}_keys').append(f'truth_{name}s')

        self.obj_keys = (self.fragment_keys
                         + self.particle_keys
                         + self.interaction_keys)

        self.keys.update({k:True for k in self.obj_keys})

        # If a truth point mode is specified, store it
        if truth_point_mode is not None:
            assert truth_point_mode in self._point_modes, (
                     "The `truth_point_mode` argument must be one of "
                    f"{self._point_modes}. Got `{truth_point_mode}` instead.")
            self.truth_point_mode = truth_point_mode
            self.truth_point_key = self._point_modes[truth_point_mode]
            self.truth_index_mode = truth_point_mode.replace('points', 'index')
            self.truth_source_mode = truth_point_mode.replace('points', 'sources')

        # If a truth deposition mode is specified, store it
        if truth_dep_mode is not None:
            assert truth_dep_mode in self._dep_modes, (
                     "The `truth_dep_mode` argument must be one of "
                    f"{self._dep_modes}. Got `{truth_dep_mode}` instead.")
            self.truth_dep_mode = truth_dep_mode

        # Store the parent path
        self.parent_path = parent_path

    def __call__(self, data, entry=None):
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
        for key, req in self.keys.items():
            # If this key is needed, check that it exists
            assert not req or key in data, (
                    f"Post-processor `{self.name}` is missing an essential "
                    f"input to be used: `{key}`.")

            # Append
            if key in data:
                data_filter[key] = data[key]
                if entry is not None:
                    data_filter[key] = data[key][entry]

        # Run the post-processor
        return self.process(data_filter)

    def get_index(self, obj):
        """Get a certain pre-defined index attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects index are obtained using the
        `truth_index_mode` attribute of the class.

        Parameters
        ----------
        obj : Union[FragmentBase, ParticleBase, InteractionBase]
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

    def get_points(self, obj):
        """Get a certain pre-defined point attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects points are obtained using the
        `truth_point_mode` attribute of the class.

        Parameters
        ----------
        obj : Union[FragmentBase, ParticleBase, InteractionBase]
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

    def get_sources(self, obj):
        """Get a certain pre-defined sources attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects sources are obtained using the
        `truth_source_mode` attribute of the class.

        Parameters
        ----------
        obj : Union[FragmentBase, ParticleBase, InteractionBase]
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

    def get_depositions(self, obj):
        """Get a certain pre-defined deposition attribute of an object.

        The :class:`TruthFragment`, :class:`TruthParticle` and
        :class:`TruthInteraction` objects points are obtained using the
        `truth_dep_mode` attribute of the class.

        Parameters
        ----------
        obj : Union[FragmentBase, ParticleBase, InteractionBase]
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

    def check_units(self, obj):
        """Check that the point coordinates of an object are as expected.

        Parameters
        ----------
        obj : Union[FragmentBase, ParticleBase, InteractionBase]
            Particle or interaction object

        Results
        -------
        np.ndarray
           (N, 3) Point coordinates
        """
        if obj.units != self.units:
            raise ValueError(
                    f"Coordinates must be expressed in "
                    f"{self.units}; currently in {obj.units} instead.")

    @abstractmethod
    def process(self, data):
        """Place-holder method to be defined in each post-processor.

        Parameters
        ----------
        data : dict
            Dictionary of processed data products
        """
        raise NotImplementedError("Must define the `process` function.")
