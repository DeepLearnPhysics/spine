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
    name = ''
    aliases = []
    parent_path = ''
    keys = []
    truth_point_mode = 'points'
    units = 'cm'

    # List of recognized run modes
    _run_modes = ['reco', 'truth', 'both', 'all']
    
    # List of recognized object types
    _obj_types = ['fragment', 'particle', 'interaction']

    def __init__(self, run_mode=None, truth_point_mode=None):
        """Initialize default post-processor object properties.

        Parameters
        ----------
        run_mode : str, optional
           If specified, tells whether the post-processor must run on
           reconstructed ('reco'), true ('true) or both objects ('both', 'all')
        truth_point_mode : str, optional
           If specified, tells which attribute of the `TruthParticle` object
           to use to fetch its point coordinates
        """
        # If run mode is specified, process it
        if run_mode is not None:
            # Check that the run mode is recognized
            assert run_mode in self._run_modes, (
                    f"`run_mode` not recognized: {run_mode}. Must be one of "
                    f"{self._run_modes}.")

        # Make a list of object keys to process
        for name in self._obj_types:
            # Initialize one list per object type
            setattr(self, f'{name}_keys', [])

            # Loop over the requisite keys, store them
            if run_mode != 'truth':
                key = f'reco_{name}s'
                if key in self.keys:
                    getattr(self, f'{name}_keys').append(key)
            if run_mode != 'reco':
                key = f'truth_{name}s'
                if key in self.keys:
                    getattr(self, f'{name}_keys').append(key)

        self.all_keys = (self.fragment_keys 
                         + self.particle_keys 
                         + self.interaction_keys)

        # If a truth point mode is specified, store it
        if truth_point_mode is not None:
            self.truth_point_mode = truth_point_mode

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
        input_data = {}
        for key, req in self.keys.items():
            # If this key is needed, check that it exists
            assert not req or key in data, (
                    f"Post-processor `{self.name}` if missing an essential "
                    f"input to be used: `{key}`.")

            # Append
            if key in data:
                input_data[key] = data[key]
                if entry is not None:
                    input_data[key] = data[key][entry]

        # Run the post-processor
        return self.process(input_data)

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
        """Place-holder function to be defined for each post-processor.

        Parameters
        ----------
        data : dict
            Dictionary of processed data products
        """
        raise NotImplementedError("Must define the `process` function.")
