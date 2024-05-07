"""Contains base class of all post-processors."""
import time


class PostBase:
    """Base class of all post-processors.
    
    This base class performs the following functions:
      - Ensures that the necessary method exist
      - Checks that the post-processor is provided the necessary information
        to do its job
      - Fetches the appropriate coordinate attributes
      - Ensures that the appropriate units are provided
    """
    name = ''
    parent_path = ''
    data_cap = []
    data_cap_opt = []
    result_cap = []
    result_cap_opt = []
    truth_point_mode = 'points'
    units = 'cm'

    # List of recognized run modes
    _run_modes = ['reco', 'truth', 'both', 'all']
    
    # List of recognized obejct types
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
        req_keys = self.result_cap + self.result_cap_opt
        for name in self._obj_types:
            # Initialize one list per object type
            setattr(self, f'{name}_keys', [])

            # Loop over the requisite keys, store them
            if run_mode in ['reco', 'both', 'all']:
                key = f'reco_{name}'
                if key in req_keys:
                    getattr(self, f'{name}_keys').append(key)
            if run_mode in ['truth', 'both', 'all']:
                key = f'truth_{name}'
                if key in req_keys:
                    getattr(self, f'{name}_keys').append(key)

        self.all_keys = (self.fragment_keys 
                         + self.particle_keys 
                         + self.interaction_keys)

        # If a truth point mode is specified, store it
        if truth_point_mode is not None:
            self.truth_point_mode = truth_point_mode

    def run(self, data_dict, result_dict, image_id):
        """Runs the post processor on one entry.

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        image_id : int
            Entry number in the input/output dictionaries

        Returns
        -------
        data_update : dict
            Update to the input dictionary
        result_update : dict
            Update to the result dictionary
        time : float
            Post-processor execution time
        """
        # Fetch the necessary information
        data_single, result_single = {}, {}
        for data_key in self.data_cap:
            if data_key in data_dict.keys():
                if data_key == 'meta':
                    data_single[data_key] = data_dict[data_key]
                else:
                    data_single[data_key] = data_dict[data_key][image_id]
            else:
                raise KeyError(
                        f"Unable to find {data_key} in data dictionary while "
                        f"running post-processor {self.name}.")

        for result_key in self.result_cap:
            if result_key in result_dict.keys():
                result_single[result_key] = result_dict[result_key][image_id]
            else:
                raise KeyError(
                        f"Unable to find {result_key} in result dictionary "
                        f"while running post-processor {self.name}.")

        # Fetch the optional information, if available
        for data_key in self.data_cap_opt:
            if data_key in data_dict.keys():
                data_single[data_key] = data_dict[data_key][image_id]

        for result_key in self.result_cap_opt:
            if result_key in result_dict.keys():
                result_single[result_key] = result_dict[result_key][image_id]

        # Run the post-processor
        start = time.time()
        data_update, result_update = self.process(data_single, result_single)
        end = time.time()
        process_time = end-start

        return data_update, result_update, process_time

    def process(self, data_dict, result_dict):
        """Function which needs to be defined for each post-processor.

        Parameters
        ----------
        data_dict : dict
            Input data dictionary
        result_dict : dict
            Chain output dictionary
        """
        raise NotImplementedError('Must define the `process` function')

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
                    f"{self.units}, currently in {obj.units} instead.")
