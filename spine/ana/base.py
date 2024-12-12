"""Base class of all analysis scripts."""

from abc import ABC, abstractmethod
from warnings import warn

from spine.io.write import CSVWriter


class AnaBase(ABC):
    """Parent class of all analysis scripts.

    This base class performs the following functions:
    - Ensures that the necessary methods exist
    - Checks that the script is provided the necessary information
    - Writes the output of the analysis to CSV

    Attributes
    ----------
    name : str
        Name of the analysis script (to call it from a configuration file)
    req_keys : List[str]
        Data products needed to run the analysis script
    opt_keys : List[str]
        Optional data products which will be used only if they are provided
    units : str
        Units in which the coordinates are expressed
    """

    # Name of the analysis script (as specified in the configuration)
    name = None

    # Alternative allowed names of the analysis script
    aliases = ()

    # Units in which the analysis script expects objects to be expressed in
    units = 'cm'

    # Set of data keys needed for this analysis script to operate
    _keys = ()

    # List of recognized object types
    _obj_types = ('fragment', 'particle', 'interaction')

    # Valid run modes
    _run_modes = ('reco', 'truth', 'both', 'all')

    # List of known point modes for true particles and their corresponding keys
    _point_modes = (
            ('points', 'points_label'),
            ('points_adapt', 'points'),
            ('points_g4', 'points_g4')
    )

    def __init__(self, obj_type=None, run_mode=None, truth_point_mode=None,
                 append=False, overwrite=False, log_dir=None, prefix=None):
        """Initialize default anlysis script object properties.

        Parameters
        ----------
        obj_type : Union[str, List[str]]
            Name or list of names of the object types to process
        run_mode : str, optional
            If specified, tells whether the analysis script must run on
            reconstructed ('reco'), true ('true') or both objects
            ('both' or 'all')
        truth_point_mode : str, optional
            If specified, tells which attribute of the :class:`TruthFragment`,
            :class:`TruthParticle` or :class:`TruthInteraction` object to use
            to fetch its point coordinates
        append : bool, default False
            If True, appends existing CSV files instead of creating new ones
        overwrite : bool, default False
            If True and an output CSV file exists, overwrite it
        log_dir : str
            Output CSV file directory (shared with driver log)
        prefix : str, default None
            Name to prefix every output CSV file with
        """
        # Initialize default keys
        self.update_keys({
                'index': True, 'file_index': True,
                'file_entry_index': False, 'run_info': False
        })

        # If run mode is specified, process it
        self.run_mode = run_mode
        if run_mode is not None:
            # Check that the run mode is recognized
            assert run_mode in self._run_modes, (
                    f"`run_mode` not recognized: {run_mode}. Must be one of "
                    f"{self._run_modes}.")

        self.prefixes = []
        if run_mode != 'truth':
            self.prefixes.append('reco')
        if run_mode != 'reco':
            self.prefixes.append('truth')

        # Check that all the object sources are recognized
        self.obj_type = obj_type
        if self.obj_type is not None:
            if isinstance(self.obj_type, str):
                self.obj_type = [self.obj_type]
            for obj in self.obj_type:
                assert obj in self._obj_types, (
                        f"Object type must be one of {self._obj_types}. Got "
                        f"`{obj}` instead.")

        # Make a list of object keys to process
        for name in self._obj_types:
            # Initialize one list per object type
            setattr(self, f'{name}_keys', [])

            # Skip object types which are not requested
            if self.obj_type is not None and name in self.obj_type:
                if run_mode != 'truth':
                    getattr(self, f'{name}_keys').append(f'reco_{name}s')
                if run_mode != 'reco':
                    getattr(self, f'{name}_keys').append(f'truth_{name}s')

        self.obj_keys = (self.fragment_keys 
                         + self.particle_keys 
                         + self.interaction_keys)

        # Update underlying keys, if needed
        self.update_keys({k:True for k in self.obj_keys})

        # If a truth point mode is specified, store it
        if truth_point_mode is not None:
            assert truth_point_mode in self.point_modes, (
                     "The `truth_point_mode` argument must be one of "
                    f"{self.point_modes.keys()}. Got `{truth_point_mode}` instead.")
            self.truth_point_mode = truth_point_mode
            self.truth_index_mode = truth_point_mode.replace('points', 'index')

        # Store the append flag
        self.append_file = append
        self.overwrite_file = overwrite

        # Initialize a writer dictionary to be filled by the children classes
        self.log_dir = log_dir
        self.output_prefix = prefix
        self.writers = {}

    def initialize_writer(self, name):
        """Adds a CSV writer to the list of writers for this script.

        Parameters
        ----------
        name : str
            Name of the writer
        """
        # Define the name of the file to write to
        assert len(name) > 0, "Must provide a non-empty name."
        file_name = f'{self.name}_{name}.csv'
        if self.output_prefix:
            file_name = f'{self.output_prefix}_{file_name}'
        if self.log_dir:
            file_name = f'{self.log_dir}/{file_name}'

        # Initialize the writer
        self.writers[name] = CSVWriter(
                file_name, append=self.append_file,
                overwrite=self.overwrite_file)

    @property
    def keys(self):
        """Dictionary of (key, necessity) pairs which determine which data keys
        are needed/optional for the post-processor to run.

        Returns
        -------
        Dict[str, bool]
            Dictionary of (key, necessity) pairs to be used
        """
        return dict(self._keys)

    @keys.setter
    def keys(self, keys):
        """Converts a dictionary of keys to an immutable tuple.

        Parameters
        ----------
        Dict[str, bool]
            Dictionary of (key, necessity) pairs to be used
        """
        self._keys = tuple(keys.items())

    @property
    def point_modes(self):
        """Dictionary which makes the correspondance between the name of a true
        object point attribute with the underlying point tensor it points to.

        Returns
        -------
        Dict[str, str]
            Dictionary of (attribute, key) mapping for point coordinates
        """
        return dict(self._point_modes)

    def update_keys(self, update_dict):
        """Update the underlying set of keys and their necessity in place.

        Parameters
        ----------
        update_dict : Dict[str, bool]
            Dictionary of (key, necessity) pairs to update the keys with
        """
        if len(update_dict) > 0:
            keys = self.keys
            keys.update(update_dict)
            self._keys = tuple(keys.items())

    def get_base_dict(self, data):
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
        base_dict = {'index': data['index'], 'file_index': data['file_index']}
        if 'file_entry_index' in data:
            base_dict['file_entry_index'] = data['file_entry_index']
        if 'run_info' in data:
            base_dict.update(**data['run_info'].scalar_dict())
        else:
            warn("`run_info` is missing; will not be included in CSV file.")

        return base_dict

    def append(self, name, **kwargs):
        """Apppend a CSV log file with a set of values.

        Parameters
        ----------
        name : str
            Name of the writer
        **kwargs : dict
            Dictionary of information to save to the writer
        """
        self.writers[name].append({**self.base_dict, **kwargs})

    def __call__(self, data, entry=None):
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
            assert not req or key in data, (
                    f"Analysis script `{self.name}` is missing an essential "
                    f"input to be used: `{key}`.")

            # Append
            if key in data:
                data_filter[key] = data[key]
                if entry is not None:
                    data_filter[key] = data[key][entry]

        # Fetch the base dictionary
        self.base_dict = self.get_base_dict(data_filter)

        # Run the analysis script
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

    @abstractmethod
    def process(self, data):
        """Place-holder method to be defined in each analysis script.

        Parameters
        ----------
        data : dict
            Filtered data dictionary for one entry
        """
        raise NotImplementedError('Must define the `process` function')
