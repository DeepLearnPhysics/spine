"""Base class of all analysis scripts."""

from mlreco.io.write import CSVWriter


class AnaBase:
    """Parent class of all analysis scripts.

    This base class performs the following functions:
      - Ensures that the necessary method exist
      - Checks that the script is provided the necessary information
      - Writes the output of the analysis to CSV
    """
    name = ''
    req_keys = []
    opt_keys = []
    units = 'cm'

    # Valid run modes
    _run_modes = ['reco', 'truth', 'both', 'all']

    def __init__(self, run_mode=None, append_file=False, output_prefix=None):
        """Initialize default anlysis script object properties.

        Parameters
        ----------
        run_mode : str, optional
            If specified, tells whether the analysis script must run on
            reconstructed ('reco'), true ('true) or both objects ('both', 'all')
        append_file : bool, default False
            If True, appends existing CSV files instead of creating new ones
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

        # Store the append flag
        self.append_file = append_file

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
        file_name = f'{self.name}_{name}'
        if self.output_prefix is not None:
            file_name = f'{self.output_prefix}_{file_name}'

        # Initialize the writer
        self.writers[name] = CSVWriter(file_name, append_file)

    def run(self, data, entry):
        """Runs the analysis script on one entry.

        Parameters
        ----------
        data : dict
            Data dictionary for one entry

        Returns
        -------
        float
            Post-processor execution time
        """
        # Fetch the necessary information
        data_filter = {}
        for key in (self.req_keys + self.opt_keys):
            if key in data:
                data_filter[key] = data[key]
            elif key in self.req_keys:
                raise KeyError(
                        f"Unable to find {data_key} in data dictionary while "
                        f"running analysis script {self.name}.")

        # Run the analysis script
        start = time.time()
        data_update, result_update = self.process(data_single, result_single)
        end = time.time()
        process_time = end-start

        return data_update, result_update, process_time

    def process(self, data):
        """Place-holder processing function for analysis scripts.

        Parameters
        ----------
        data : dict
            Filtered data dictionary for one entry
        """
        raise NotImplementedError('Must define the `process` function')
