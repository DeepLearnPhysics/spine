"""Interaction vertex reconstruction module."""

import numpy as np

from spine.utils.globals import SHOWR_SHP, TRACK_SHP
from spine.utils.vertex import get_vertex

from spine.post.base import PostBase

__all__ = ['VertexProcessor']


class VertexProcessor(PostBase):
    """Reconstruct one vertex for each interaction in the provided list."""

    # Name of the post-processor (as specified in the configuration)
    name = 'vertex'

    # Alternative allowed names of the post-processor
    aliases = ('reconstruct_vertex',)

    def __init__(self, include_shapes=(SHOWR_SHP, TRACK_SHP),
                 use_primaries=True, update_primaries=False,
                 anchor_vertex=True, touching_threshold=2.0,
                 angle_threshold=0.3, run_mode='both',
                 truth_point_mode='points'):
        """Initialize the vertex finder properties.

        Parameters
        ----------
        include_shapes : List[int], default [0, 1]
            List of semantic classes to consider for vertex reconstruction
        use_primaries : bool, default True
            If true, only considers primary particles to reconstruct the vertex
        update_primaries : bool, default False
            Use the reconstructed vertex to update primaries
        anchor_vertex : bool, default True
            If true, anchor the candidate vertex to particle objects,
            with the expection of interactions only composed of showers
        touching_threshold : float, default 2 cm
            Maximum distance for two track points to be considered touching
        angle_threshold : float, default 0.3 radians
            Maximum angle between the vertex-to-start-point vector and a shower
            direction to consider that a shower originated from the vertex
        """
        # Initialize the parent class
        super().__init__('interaction', run_mode, truth_point_mode)

        # Store the relevant parameters
        self.include_shapes = include_shapes
        self.use_primaries = use_primaries
        self.update_primaries = update_primaries
        self.anchor_vertex = anchor_vertex
        self.touching_threshold = touching_threshold
        self.angle_threshold = angle_threshold

    def process(self, data):
        """Reconstruct the vertex position for each interaction in one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over interaction objects
        for k in self.interaction_keys:
            for inter in data[k]:
                self.reconstruct_vertex_single(inter)

    def reconstruct_vertex_single(self, inter):
        """Post-processor which reconstructs one vertex for each interaction
        in the provided list.

        Parameters
        ----------
        inter : List[RecoInteraction, TruthInteraction]
            Reconstructed/truth interaction object
        """
        # Selected the set of particles to use as a basis for vertex prediction
        if self.use_primaries:
            particles = [part for part in inter.particles if (
                    part.is_primary and
                    part.shape in self.include_shapes and
                    part.size > 0)]
        if not self.use_primaries or not len(particles):
            particles = [part for part in inter.particles if (
                    part.shape in self.include_shapes and
                    part.size > 0)]
        if not len(particles):
            particles = [part for part in inter.particles if part.size > 0]

        if len(particles) > 0:
            # Collapse particle objects to start, end points and directions
            start_points = np.vstack([part.start_point for part in particles])
            end_points   = np.vstack([part.end_point for part in particles])
            directions   = np.vstack([part.start_dir for part in particles])
            shapes       = np.array([part.shape for part in particles])

            # Reconstruct the vertex for this interaction
            vtx, _ = get_vertex(
                start_points, end_points, directions, shapes,
                self.anchor_vertex, self.touching_threshold, return_mode=True)

            # Assign it to the appropriate interaction attribute
            if not inter.is_truth:
                inter.vertex = vtx
            else:
                inter.reco_vertex = vtx

            # If requested, update primaries on the basis of the vertex
            if not inter.is_truth and self.update_primaries:
                for part in inter.particles:
                    if part.shape not in [SHOWR_SHP, TRACK_SHP]:
                        part.is_primary = False
                    elif (np.linalg.norm(part.start_point - inter.vertex) 
                          < self.touching_threshold):
                        part.is_primary = True
                    elif (part.shape == SHOWR_SHP and
                          np.dot(part.start_point, inter.vertex) < self.angle_threshold):
                        part.is_primary = True
