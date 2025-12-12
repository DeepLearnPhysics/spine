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
        super().__init__(obj_type, run_mode, truth_point_mode, truth_dep_mode, **kwargs)

        # Store parameters
        self.radius = radius

    def process(self, data):
        """Evaluate shower start dE/dx for one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # TODO: Implement shower start dE/dx calculation
        raise NotImplementedError("ShowerStartDEdxAna is not yet implemented.")
