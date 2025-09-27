"""Post-processor module template.

Use this template as a basis to build your own post-processor. A post-processor
takes the output of the reconstruction and either
- Sets additional reconstruction attributes (e.g. direction estimates)
- Adds entirely new data products (e.g. trigger time)
"""

# Add the imports specific to this module here
# import ...

# Must import the post-processor base class
from spine.post.base import PostBase

# Must list the post-processor(s) here to be found by the factory.
# You must also add it to the list of imported modules in the
# `spine.post.factories`!
__all__ = ["TemplateProcssor"]


class TemplateProcessor(PostBase):
    """Description of what the post-processor is supposed to be doing."""

    name = "template"  # Name used to call the post-processor in the config

    def __init__(self, arg0, arg1, obj_type, run_mode):
        """Initialize the post-processor.

        Parameters
        ----------
        arg0 : type
            Description of arg0
        arg1 : type
            Description of arg1
        obj_type : Union[str, List[str]]
            Types of objects needed in this post-processor (fragments,
            particles and/or interactions). This argument is shared between
            all post-processors. If None, does not load these objects.
        run_mode : str
            One of 'reco', 'truth' or 'both'. Determines what kind of object
            the post-processor has to run on.
        """
        # Initialize the parent class
        super().__init__(obj_type, run_mode)

        # Store parameter
        self.arg0 = arg0
        self.arg1 = arg1

        # Add additional required data products
        self.keys["prod"] = True  # Means we must have 'prod' in the dictionary

    def process(self, data):
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
                points = self.get_points(obj)

                # Get another attribute
                sources = obj.sources

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
