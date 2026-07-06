"""Post-processor module template.

Use this template as a basis to build your own post-processor. A post-processor
takes the output of the reconstruction and either
- Sets additional reconstruction attributes (e.g. direction estimates)
- Adds entirely new data products (e.g. trigger time)
"""

# Add the imports specific to this module here
# import ...
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Must import the post-processor base class
from spine.post.base import PostBase

# Must list the post-processor(s) here to be found by the factory.
# You must also add it to the list of imported modules in the
# `spine.post.factories`!
__all__ = ["TemplateProcessor"]


class TemplateProcessor(PostBase):
    """Template post-processor showing the expected PostBase interface."""

    name = "template"  # Name used to call the post-processor in the config

    def __init__(
        self,
        arg0: Any,
        arg1: Any,
        obj_type: str | Sequence[str] | None = None,
        run_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the post-processor.

        Parameters
        ----------
        arg0 : object
            Example required argument
        arg1 : object
            Example required argument
        obj_type : str or sequence[str], optional
            Types of objects needed in this post-processor (fragments,
            particles and/or interactions). This argument is shared between
            all post-processors. If None, does not load these objects.
        run_mode : str
            One of 'reco', 'truth' or 'both'. Determines what kind of object
            the post-processor has to run on.
        **kwargs : dict, optional
            Additional arguments to pass to :class:`PostBase`
        """
        # Initialize the parent class
        super().__init__(obj_type=obj_type, run_mode=run_mode, **kwargs)

        # Store parameter
        self.arg0 = arg0
        self.arg1 = arg1

        # Add additional required data products
        self.update_keys({"prod": True})

    def process(self, data: Mapping[str, Any]) -> dict[str, Any] | None:
        """Pass data products corresponding to one entry through the processor.

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
                # Fetch points attributes
                self.get_points(obj)

                # Get another attribute
                obj.sources

                # Do something...

        # Loop over requested specific types of objects
        for key in self.fragment_keys:
            # Do something...
            pass

        for key in self.particle_keys:
            # Do something...
            pass

        for key in self.interaction_keys:
            # Do something...
            pass

        # Return an update or override to the current data product dictionary
        return {}  # Can have no return as well if objects are edited in place
