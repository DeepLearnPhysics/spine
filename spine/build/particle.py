"""Classes in charge of constructing Particle objects."""

import numpy as np

from scipy.special import softmax

from spine.data.out import RecoParticle, TruthParticle
from spine.utils.globals import COORD_COLS, VALUE_COL, GROUP_COL, TRACK_SHP

from .base import BuilderBase

__all__ = ['ParticleBuilder']


class ParticleBuilder(BuilderBase):
    """Builds reconstructed and truth particles.

    It takes the raw output of the reconstruction chain, extracts the
    necessary information and builds :class:`RecoParticle` and
    :class:`TruthParticle` objects from it.
    """

    # Builder name
    name = 'particle'

    # Types of objects constructed by the builder
    _reco_type = RecoParticle
    _truth_type = TruthParticle

    # Necessary/optional data products to build a reconstructed object
    _build_reco_keys  = (
            ('particle_clusts', True), ('particle_shapes', True),
            ('particle_start_points', True), ('particle_end_points', True),
            ('particle_group_pred', True), ('particle_node_type_pred', True),
            ('particle_node_primary_pred', True),
            ('particle_node_orient_pred', False), ('reco_fragments', False),
            *BuilderBase._build_reco_keys
    )

    # Necessary/optional data products to build a truth object
    _build_truth_keys = (
            ('particles', False),
            *BuilderBase._build_truth_keys
    )

    # Necessary/optional data products to load a reconstructed object
    _load_reco_keys  = (
            ('reco_particles', True),
            *BuilderBase._load_reco_keys
    )

    # Necessary/optional data products to load a truth object
    _load_truth_keys  = (
            ('truth_particles', True),
            *BuilderBase._load_truth_keys
    )

    def build_reco(self, data):
        """Builds :class:`RecoParticle` objects from the full chain output.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        return self._build_reco(**data)

    def _build_reco(self, points, depositions, particle_clusts, particle_shapes,
                    particle_start_points, particle_end_points,
                    particle_group_pred, particle_node_type_pred,
                    particle_node_primary_pred, particle_node_orient_pred=None,
                    reco_fragments=None, sources=None):
        """Builds :class:`RecoParticle` objects from the full chain output.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of deposition coordinates in the image
        depositions : np.ndarray
            (N) Set of deposition values
        particle_clusts : List[np.ndarray]
            (P) List of indexes, each corresponding to a particle instance
        particle_shapes : np.ndarray
            (P) List of particle shapes (shower, track, etc.)
        particle_start_points : np.ndarray
            (P, 3) List of particle start point coordinates
        particle_end_points : np.ndarray
            (P, 3) List of particle end point coordinates
        particle_group_pred : np.ndarray
            (P) Interaction group each particle belongs to
        particle_node_type_pred : np.ndarray
            (P, N_c) Particle identification logits
        particle_node_primary_pred : np.ndarray
            (P, 2) Particle primary classification logits
        particle_node_orient_pred : np.ndarray, optional
            (P, 2) Particle orientation classification logits
        reco_fragments : List[RecoFragment], optional
            (F) List of reconstructed fragments
        sources : np.ndarray, optional
            (N, 2) Tensor which contains the module/tpc information

        Returns
        -------
        List[RecoParticle]
            List of constructed reconstructed particle instances
        """
        # Convert the logits to softmax scores and the scores to a prediction
        pid_scores = softmax(particle_node_type_pred, axis=1)
        primary_scores = softmax(particle_node_primary_pred, axis=1)
        pid_pred = np.argmax(pid_scores, axis=1)
        primary_pred = np.argmax(primary_scores, axis=1)
        if particle_node_orient_pred is not None:
            orient_pred = np.argmax(particle_node_orient_pred, axis=1)

        # Loop over the particle instances
        reco_particles = []
        for i, index in enumerate(particle_clusts):
            # Initialize
            particle = RecoParticle(
                    id=i,
                    interaction_id=particle_group_pred[i],
                    shape=particle_shapes[i],
                    index=index,
                    points=points[index],
                    depositions=depositions[index],
                    pid=pid_pred[i],
                    primary_scores=primary_scores[i],
                    is_primary=bool(primary_pred[i]))

            # Set the PID scores without modifying the default size
            particle.pid_scores[:len(pid_scores[i])] = pid_scores[i]
            particle.pid_scores[len(pid_scores[i]):] = 0.

            # Set the end points
            particle.start_point = particle_start_points[i]
            if particle.shape == TRACK_SHP:
                particle.end_point = particle_end_points[i]

            # If the orientation prediction is provided, use it
            if orient_pred is not None and not orient_pred[i]:
                if particle.shape == TRACK_SHP:
                    particle.start_point, particle.end_point = (
                            particle.end_point, particle.start_point)

            # Add optional arguments
            if sources is not None:
                particle.sources = sources[index]

            # Append
            reco_particles.append(particle)

        return reco_particles

    def build_truth(self, data):
        """Builds :class:`TruthParticle` objects from the full chain output.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        return self._build_truth(**data)

    def _build_truth(self, label_tensor, label_adapt_tensor, particles, points,
                     depositions, points_label, depositions_label,
                     depositions_q_label=None, label_g4_tensor=None,
                     points_g4=None, depositions_g4=None, sources=None,
                     sources_label=None):
        """Builds :class:`TruthParticle` objects from the full chain output.

        Parameters
        ----------
        label_tensor : np.ndarray
            Tensor which contains the cluster labels of each deposition
        label_adapt_tensor : np.ndarray
            Tensor which contains the cluster labels of each deposition,
            adapted to the semantic segmentation prediction.
        particles : List[Particle]
            List of true particles
        points : np.ndarray
            (N, 3) Set of deposition coordinates in the image
        depositions : np.ndarray
            (N) Set of deposition values
        points_label : np.ndarray
            (N', 3) Set of deposition coordinates in the label image (identical
            for pixel TPCs, different if deghosting is involved)
        depositions_label : np.ndarray
            (N') Set of true deposition values in MeV
        depositions_q_label : np.ndarray, optional
            (N') Set of true deposition values in ADC, if relevant
        label_tensor_g4 : np.ndarray, optional
            Tensor which contains the cluster labels of each deposition
            in the Geant4 image (before the detector simulation)
        points_g4 : np.ndarray
            (N'', 3) Set of deposition coordinates in the Geant4 image
        depositions_g4 : np.ndarray
            (N'') Set of deposition values in the Geant4 image
        sources : np.ndarray, optional
            (N, 2) Tensor which contains the module/tpc information
        sources_label : np.ndarray, optional
            (N', 2) Tensor which contains the label module/tpc information

        Returns
        -------
        List[TruthParticle]
            List of restored true particle instances
        """
        # Loop over the true particle instance groups
        truth_particles = []
        unique_group_ids = np.unique(label_tensor[:, GROUP_COL]).astype(int)
        valid_group_ids = unique_group_ids[unique_group_ids > -1]
        for i, group_id in enumerate(valid_group_ids):
            # Load the MC particle information
            assert group_id < len(particles), (
                    "Invalid group ID, cannot build true particle.")
            particle = TruthParticle(**particles[group_id].as_dict())
            assert particle.id == group_id, (
                    "The ordering of the true particle is wrong.")

            # Override the index of the particle but preserve it
            particle.orig_id = group_id
            particle.id = i

            # Update the attributes shared between reconstructed and true
            particle.length = particle.distance_travel
            particle.is_primary = bool(particle.interaction_primary > 0)
            particle.start_point = particle.first_step
            if particle.shape == TRACK_SHP:
                particle.end_point = particle.last_step

            # Update the particle with its long-form attributes
            index = np.where(label_tensor[:, GROUP_COL] == group_id)[0]
            particle.index = index
            particle.points = points_label[index]
            particle.depositions = depositions_label[index]
            if depositions_q_label is not None:
                particle.depositions_q = depositions_q_label[index]
            if sources_label is not None:
                particle.sources = sources_label[index]

            index_adapt = np.where(
                    label_adapt_tensor[:, GROUP_COL] == group_id)[0]
            particle.index_adapt = index_adapt
            particle.points_adapt = points[index_adapt]
            particle.depositions_adapt = depositions[index_adapt]
            if sources is not None:
                particle.sources_adapt = sources[index_adapt]

            if label_g4_tensor is not None:
                index_g4 = np.where(
                        label_g4_tensor[:, GROUP_COL] == group_id)[0]
                particle.index_g4 = index_g4
                particle.points_g4 = points_g4[index_g4]
                particle.depositions_g4 = depositions_g4[index_g4]

            # Append
            truth_particles.append(particle)

        return truth_particles

    def load_reco(self, data):
        """Construct :class:`RecoParticle` objects from their stored versions.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        return self._load_reco(**data)

    def _load_reco(self, reco_particles, points, depositions, sources=None):
        """Construct :class:`RecoParticle` objects from their stored versions.

        Parameters
        ----------
        reco_particles : List[RecoParticle]
            (P) List of partial reconstructed particles
        points : np.ndarray
            (N, 3) Set of deposition coordinates in the image
        depositions : np.ndarray
            (N) Set of deposition values
        sources : np.ndarray, optional
            (N, 2) Tensor which contains the module/tpc information

        Returns
        -------
        List[RecoParticle]
            List of restored reconstructed particle instances
        """
        # Loop over the dictionaries
        for i, particle in enumerate(reco_particles):
            # Check that the particle ID checks out
            assert particle.id == i, (
                    "The ordering of the stored particles is wrong.")

            # Update the particle with its long-form attributes
            particle.points = points[particle.index]
            particle.depositions = depositions[particle.index]
            if sources is not None:
                particle.sources = sources[particle.index]

        return reco_particles

    def load_truth(self, data):
        """Construct :class:`TruthParticle` objects from their stored versions.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        return self._load_truth(**data)

    def _load_truth(self, truth_particles, points, depositions, points_label,
                    depositions_label, depositions_q_label=None, points_g4=None,
                    depositions_g4=None, sources=None, sources_label=None):
        """Construct :class:`TruthParticle` objects from their stored versions.

        Parameters
        ----------
        truth_particles : List[TruthParticle]
            (P) List of partial truth particles
        points : np.ndarray
            (N, 3) Set of deposition coordinates in the image
        depositions : np.ndarray
            (N) Set of deposition values
        points_label : np.ndarray
            (N', 3) Set of deposition coordinates in the label image (identical
            for pixel TPCs, different if deghosting is involved)
        depositions_label : np.ndarray
            (N') Set of true deposition values in MeV
        depositions_q_label : np.ndarray, optional
            (N') Set of true deposition values in ADC, if relevant
        points_g4 : np.ndarray
            (N'', 3) Set of deposition coordinates in the Geant4 image
        depositions_g4 : np.ndarray
            (N'') Set of deposition values in the Geant4 image
        sources : np.ndarray, optional
            (N, 2) Tensor which contains the module/tpc information
        sources_label : np.ndarray, optional
            (N', 2) Tensor which contains the label module/tpc information

        Returns
        -------
        List[TruthParticle]
            List of restored true particle instances
        """
        # Loop over the dictionaries
        for i, particle in enumerate(truth_particles):
            # Check that the particle ID checks out
            assert particle.id == i, (
                    "The ordering of the stored particles is wrong.")

            # Update the particle with its long-form attributes
            particle.points = points_label[particle.index]
            particle.depositions = depositions_label[particle.index]
            if depositions_q_label is not None:
                particle.depositions_q = depositions_q_label[particle.index]
            if sources_label is not None:
                particle.sources = sources_label[particle.index]

            particle.points_adapt = points[particle.index_adapt]
            particle.depositions_adapt = depositions[particle.index_adapt]
            if sources is not None:
                particle.sources_adapt = sources[particle.index_adapt]

            if points_g4 is not None:
                particle.points_g4 = points_g4[particle.index_g4]
                particle.depositions_g4 = depositions_g4[particle.index_g4]

        return truth_particles
