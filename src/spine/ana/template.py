"""Analysis module template.

Use this template as a basis to build your own analysis script. An analysis
script takes the output of the reconstruction and the post-processors and
performs basic selection cuts and stores the output to a CSV file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Must import the analysis script base class
from spine.ana.base import AnaBase

# Add the imports specific to this module here
# import ...


# Must list the analysis script(s) here to be found by the factory.
# You must also add it to the list of imported modules in the
# `spine.ana.factories`!
__all__ = ["TemplateAna"]


class TemplateAna(AnaBase):
    """Template analysis script showing the expected AnaBase interface.

    An analyzer which can consume projected product chunks may additionally
    override ``process_columnar(data)``. The columnar method receives the same
    declared keys without per-entry slicing; analyzers should only implement
    it when they can preserve the semantics of their event path.
    """

    # Name of the analysis script (as specified in the configuration)
    name = "template"

    def __init__(
        self,
        arg0: Any,
        arg1: Any,
        obj_type: str | Sequence[str] | None = None,
        run_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the analysis script.

        Parameters
        ----------
        arg0 : object
            Example required argument
        arg1 : object
            Example required argument
        obj_type : str or Sequence[str], optional
            Name or list of names of the object types to process
        run_mode : str, optional
            If specified, tells whether the analysis script must run on
            reconstructed ('reco'), true ('true') or both objects
            ('both' or 'all')
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(obj_type=obj_type, run_mode=run_mode, **kwargs)

        # Store parameter
        self.arg0 = arg0
        self.arg1 = arg1

        # Initialize the CSV writer(s) you want
        self.initialize_writer("template")

        # Add additional required data products
        self.update_keys({"prod": True})  # Means we must have 'prod' in the dictionary

    def process(self, data: Mapping[str, Any]) -> None:
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
                disp = obj.end_point - obj.start_point

                # Build a dictionary of scalar values to write
                out = {"disp_x": disp[0], "disp_y": disp[1], "disp_z": disp[2]}

                # Write the row to file
                self.append("template", **out)

    def process_columnar(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        """Process a projected product chunk without rebuilding event objects.

        This example assumes ``prod`` is represented as a mapping containing
        NumPy-like columns and an ``event_offsets`` boundary vector. A real
        analyzer should request only the columns it needs from the columnar
        reader, perform bulk operations where possible, and use the offsets
        when event grouping matters.

        Parameters
        ----------
        data : Mapping[str, Any]
            Filtered bulk products requested by this analyzer.

        Returns
        -------
        Mapping[str, Any]
            Optional bulk columns made available to later analyzers.
        """
        product = data["prod"]
        start_points = product["start_point"]
        end_points = product["end_point"]
        event_offsets = product["event_offsets"]

        # Vectorized over every object in the chunk. No particle objects or
        # per-event loop are constructed merely to subtract these columns.
        displacements = end_points - start_points

        return {
            "displacements": {
                "values": displacements,
                "event_offsets": event_offsets,
            }
        }
