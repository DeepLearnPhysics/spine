"""Module to evaluate diagnostic metrics on showers."""

from spine.ana.base import AnaBase

__all__ = ["ShowerStartDEdxAna"]


class ShowerStartDEdxAna(AnaBase):
    """This analysis script computes the dE/dx value within some distance
    from the start point of an EM shower object.

    This is a useful diagnostic tool to evaluate the calorimetric separability
    of different EM shower types (electron vs photon), which are expected to
    have different dE/dx patterns near their start point.
    """

    # Name of the analysis script (as specified in the configuration)
    name = "shower_start_dedx"

    def __init__(
        self,
        radius,
        obj_type="particle",
        run_mode="both",
        truth_point_mode="points",
        truth_dep_mode="depositions",
        **kwargs,
    ):
        """Initialize the analysis script.

        Parameters
        ----------
        radius : Union[float, List[float]]
            Radius around the start point for which evaluate dE/dx
        **kwargs : dict, optional
            Additional arguments to pass to :class:`AnaBase`
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode, **kwargs)

        # Store parameters
        self.radius = radius

        # Initialize the CSV writer(s) you want
        for obj in self.obj_type:
            self.initialize_writer(obj)

    def process(self, data):
        """Evaluate shower start dE/dx for one entry.

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
