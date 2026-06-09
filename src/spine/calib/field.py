"""Applies field non-uniformity corrections."""

from __future__ import annotations

__all__ = ["FieldCalibrator"]


class FieldCalibrator:
    """Applies position corrections to account for field non-uniformities
    (space charge, cathode distrotions, etc.)
    """

    name = "field"

    def __init__(self) -> None:
        """Initialize the field calibrator.

        Notes
        -----
        Placeholder until this module is implemented
        """
        raise NotImplementedError("Field calibrator not yet available.")

    def process(self) -> None:
        """Corrects for field non-uniformities.

        Notes
        -----
        Placeholder until this module is implemented
        """
        raise NotImplementedError("Field calibrator not yet available.")
