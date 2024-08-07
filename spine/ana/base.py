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
    name = ''
    aliases = []
    keys = {'index': True, 'file_index': True,
            'file_entry_index': False, 'run_info': False}
    units = 'cm'

    # List of recognized object types
    _obj_types = ['fragment', 'particle', 'interaction']

    # Valid run modes
    _run_modes = ['reco', 'truth', 'both', 'all']

    def __init__(self, obj_type=None, run_mode=None, append_file=False,
                 overwrite_file=False, output_prefix=None):
        """Initialize default anlysis script object properties.

        Parameters
        ----------
        obj_type : Union[str, List[str]]
            Name or list of names of the object types to process
        run_mode : str, optional
            If specified, tells whether the analysis script must run on
            reconstructed ('reco'), true ('true') or both objects
            ('both' or 'all')
        append_file : bool, default False
            If True, appends existing CSV files instead of creating new ones
        overwrite_file : bool, default False
            If True and the output CSV file exists, overwrite it
        output_prefix : str, default None
            Name to prefix every output CSV file with
        """
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
        if obj_type is not None:
            if isinstance(obj_type, str):
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
            if obj_type is not None and name in obj_type:
                if run_mode != 'truth':
                    getattr(self, f'{name}_keys').append(f'reco_{name}s')
                if run_mode != 'reco':
                    getattr(self, f'{name}_keys').append(f'truth_{name}s')

        self.obj_keys = (self.fragment_keys 
                         + self.particle_keys 
                         + self.interaction_keys)
        self.keys.update({k:True for k in self.obj_keys})

        # Store the append flag
        self.append_file = append_file
        self.overwrite_file = overwrite_file

        # Initialize a writer dictionary to be filled by the children classes
        self.output_prefix = output_prefix
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
        if self.output_prefix is not None:
            file_name = f'{self.output_prefix}_{file_name}.csv'

        # Initialize the writer
        self.writers[name] = CSVWriter(
                file_name, append=self.append_file,
                overwrite=self.overwrite_file)

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

    @abstractmethod
    def process(self, data):
        """Place-holder method to be defined in each analysis script.

        Parameters
        ----------
        data : dict
            Filtered data dictionary for one entry
        """
        raise NotImplementedError('Must define the `process` function')
