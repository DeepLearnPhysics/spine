"""True particle children counting module."""

from collections import Counter

import numpy as np
import networkx as nx

from spine.utils.globals import GHOST_SHP, PID_LABELS

from spine.post.base import PostBase

__all__ = ['ChildrenProcessor']


class ChildrenProcessor(PostBase):
    """Count the number of children of a given particle, using the particle
    hierarchy information from :class:`ParticleGraphParser`.
    """

    # Name of the post-processor (as specified in the configuration)
    name = 'children_count'

    # Alternative allowed names of the post-processor
    aliases = ('count_children',)

    def __init__(self, mode='shape', obj_type='particle'):
        """Initialize the children counting parameters.

        Parameters
        ----------
        mode : str, default 'shape'
            Attribute name to categorize children. This will count each child
            object for different category separately.
        """
        # Initialize the parent class
        super().__init__(obj_type, 'truth')

        # Store the counting mode
        self.mode = mode
        if self.mode == 'shape':
            self.num_classes = GHOST_SHP
        elif self.mode == 'pid':
            self.num_classes = len(PID_LABELS) - 1
        else:
            raise ValueError(
                    f"Child counting mode not recognized: {mode}. Must be "
                     "one of 'shape' or 'pid'.")
            
    def process(self, data):
        """Count children of each true particle in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the requested data products
        for k in self.obj_keys:
            # Build a directed graph on the true objects
            G = nx.DiGraph()
            edges = []
            for obj in data[k]:
                G.add_node(obj.orig_id, attr=getattr(obj, self.mode))
                parent = obj.parent_id
                if parent in G and int(parent) != int(obj.orig_id):
                    edges.append((parent, obj.orig_id))
            G.add_edges_from(edges)
            G.remove_edges_from(nx.selfloop_edges(G))

            # Count children
            for obj in data[k]:
                successors = list(G.successors(obj.orig_id))
                counter = Counter()
                counter.update([G.nodes[succ]['attr'] for succ in successors])
                children_counts = np.zeros(self.num_classes, dtype=np.int64)
                for key, val in counter.items():
                    children_counts[key] = val

                obj.children_counts = children_counts
