"""Manages a singleton instance of a Geometry class."""

import inspect
from typing import Optional

from .base import Geometry
from .factories import geo_factory


class GeoManager:
    """Manages a singleton instance of a Geometry class."""

    _instance: Optional[Geometry] = None

    @classmethod
    def initialize(
        cls, detector: str, tag: Optional[str] = None, version: Optional[str] = None
    ) -> Geometry:
        """Initialize the geometry module for a given detector.

        Parameters
        ----------
        detector : str
            The name of the detector.
        tag : str, optional
            A tag to identify a specific configuration.
        version : str, optional
            A version number for the geometry configuration.

        Returns
        -------
        Geometry
            The initialized geometry instance.
        """
        if cls._instance is not None:
            raise ValueError("Geometry module already initialized.")

        cls._instance = geo_factory(detector, tag, version)

        return cls._instance

    @classmethod
    def initialize_or_get(
        cls, detector: str, tag: Optional[str] = None, version: Optional[str] = None
    ) -> Geometry:
        """Initialize the geometry if needed, or return the existing instance.

        Parameters
        ----------
        detector : str
            The name of the detector.
        tag : str, optional
            A tag to identify a specific configuration.
        version : str, optional
            A version number for the geometry configuration.
        """
        # If the geometry is not initialized, initialize it
        if cls._instance is None:
            cls._instance = geo_factory(detector, tag, version)
            return cls._instance

        # If the geometry is already initialized, check if it matches the
        # requested configuration. If it does not match, initialize it again
        current = cls._instance
        if current.name != detector or current.tag != tag or current.version != version:
            cls._instance = geo_factory(detector, tag, version)

        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the geometry instance is initialized.

        Returns
        -------
        bool
            True if the geometry instance is initialized, False otherwise.
        """
        return cls._instance is not None

    @classmethod
    def get_instance(cls) -> Geometry:
        """Get the current geometry instance.

        Parameters
        ----------
        raise_error : bool, optional
            Whether to raise an error if the instance is not initialized.

        Returns
        -------
        Geometry
            The current geometry instance.
        """
        if cls._instance is None:
            # Raise an error with detailed information about the call site
            frame_info = inspect.stack()[1]
            frame = frame_info.frame
            func = frame.f_code.co_name
            cls_obj = frame.f_locals.get("self")
            class_name = cls_obj.__class__.__name__ if cls_obj else None

            location = f"{class_name}.{func}()" if class_name else f"{func}()"

            raise ValueError(
                "Geometry singleton instance is not initialized.\n"
                f"Attempted access from: {location}\n"
                f"File: {frame_info.filename}:{frame_info.lineno}\n\n"
                "If using the Driver, include a `geo` block in the configuration.\n"
                "If running standalone, initialize geometry with either:\n"
                "    GeoManager.initialize(detector='your_detector_name')\n"
                "    GeoManager.initialize_or_get(detector='your_detector_name')\n"
                "before calling geometry-dependent modules."
            )

        return cls._instance

    @classmethod
    def get_instance_if_initialized(cls) -> Optional[Geometry]:
        """Get the current geometry instance if initialized.

        Returns
        -------
        Optional[Geometry]
            The current geometry instance, or None if not initialized.
        """
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the geometry instance (useful for testing)."""
        cls._instance = None
