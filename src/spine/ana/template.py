"""Analysis module template.

Use this template as a basis to build your own analysis script. An analysis
script takes the output of the reconstruction and the post-processors and
performs basic selection cuts and store the output to a CSV file.
"""

# Add the imports specific to this module here
# import ...

# Must import the analysis script base class
from spine.ana.base import AnaBase

# Must list the analysis script(s) here to be found by the factory.
# You must also add it to the list of imported modules in the
# `spine.ana.factories`!
__all__ = ["TemplateAna"]


class TemplateAna(AnaBase):
    """Description of what the analysis script is supposed to be doing."""

    # Name of the analysis script (as specified in the configuration)
    name = "template"

    def __init__(
        self, arg0, arg1, obj_type, run_mode, append_file, overwrite_file, output_prefix
    ):
        """Initialize the analysis script.

        Parameters
        ----------
        arg0 : type
            Description of arg0
        arg1 : type
            Description of arg1
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
        # Initialize the parent class
        super().__init__(obj_type, run_mode, append_file, overwrite_file, output_prefix)

        # Store parameter
        self.arg0 = arg0
        self.arg1 = arg1

        # Initialize the CSV writer(s) you want
        self.initialize_writer("template")

        # Add additional required data products
        self.update_keys({"prod": True})  # Means we must have 'prod' in the dictionary

    def process(self, data):
        """Pass data products corresponding to one entry through the analysis.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Fetch the keys you want
        data = data["prod"]

        # Loop over all requested object types
        for key in self.obj_keys:
            # Loop over all objects of that type
            for obj in data[key]:
                # Do something with the object
                disp = p.end_point - p.start_point

                # Make a dictionary of integer out of it
                out = {"disp_x": disp[0], "disp_y": disp[1], "disp_z": disp[2]}

                # Write the row to file
                self.append("template", **out)
