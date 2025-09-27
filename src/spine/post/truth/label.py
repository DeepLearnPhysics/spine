"""True particle children counting module."""

from collections import Counter, defaultdict

import numpy as np

from spine.post.base import PostBase
from spine.utils.globals import GHOST_SHP, PID_LABELS

__all__ = ["ChildrenProcessor"]


class ChildrenProcessor(PostBase):
    """Count the number of children of a given particle, using the particle
    hierarchy information from :class:`ParticleGraphParser`.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "children_count"

    # Alternative allowed names of the post-processor
    aliases = ("count_children",)

    def __init__(self, mode="shape", obj_type="particle"):
        """Initialize the children counting parameters.

        Parameters
        ----------
        mode : str, default 'shape'
            Attribute name to categorize children. This will count each child
            object for different category separately.
        """
        # Initialize the parent class
        super().__init__(obj_type, "truth")

        # Store the counting mode
        self.mode = mode
        if self.mode == "shape":
            self.num_classes = GHOST_SHP
        elif self.mode == "pid":
            self.num_classes = len(PID_LABELS) - 1
        else:
            raise ValueError(
                f"Child counting mode not recognized: {mode}. Must be "
                "one of 'shape' or 'pid'."
            )

    def process(self, data):
        """Count children of each true particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the requested data products
        for k in self.obj_keys:
            # Build parent-child relationships using simple dictionaries
            nodes = {}  # node_id -> attribute
            children = defaultdict(list)  # parent_id -> [child_ids]

            # First pass: collect all nodes and their attributes
            for obj in data[k]:
                nodes[obj.orig_id] = getattr(obj, self.mode)

            # Second pass: build parent-child relationships
            for obj in data[k]:
                parent_id = obj.parent_id
                if parent_id in nodes and int(parent_id) != int(
                    obj.orig_id
                ):  # Avoid self-loops
                    children[parent_id].append(obj.orig_id)

            # Count children for each object
            for obj in data[k]:
                child_ids = children[obj.orig_id]
                counter = Counter()
                counter.update([nodes[child_id] for child_id in child_ids])
                children_counts = np.zeros(self.num_classes, dtype=np.int64)
                for key, val in counter.items():
                    children_counts[key] = val

                obj.children_counts = children_counts
