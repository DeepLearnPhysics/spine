"""Class in charge of constructing Interaction objects."""

from collections import defaultdict
from warnings import warn

import numpy as np

from spine.data.out import RecoInteraction, TruthInteraction

from .base import BuilderBase

__all__ = ['InteractionBuilder']


class InteractionBuilder(BuilderBase):
    """Builds reconstructed and truth interactions.

    It takes the raw output of the reconstruction chain, extracts the
    necessary information and builds :class:`RecoInteraction` and
    :class:`TruthInteraction` objects from it.
    """

    # Builder name
    name = 'interaction'

    # Types of objects constructed by the builder
    _reco_type = RecoInteraction
    _truth_type = TruthInteraction

    # Necessary/optional data products to build a reconstructed object
    _build_reco_keys = (
            ('reco_particles', True),
    )

    # Necessary/optional data products to build a truth object
    _build_truth_keys = (
            ('truth_particles', True), ('neutrinos', False)
    )

    # Necessary/optional data products to load a reconstructed object
    _load_reco_keys = (
            ('reco_interactions', True), ('reco_particles', True)
    )

    # Necessary/optional data products to load a truth object
    _load_truth_keys = (
            ('truth_interactions', True), ('truth_particles', True)
    )

    def build_reco(self, data):
        """Builds :class:`RecoInteraction` objects from the full chain output.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        List[RecoInteraction]
            List of constructed reconstructed interaction instances
        """
        return self._build_reco(**data)

    def _build_reco(self, reco_particles):
        """Builds :class:`RecoInteraction` objects from the full chain output.

        This class builds an interaction by assembling particles together.

        Parameters
        ----------
        reco_particles : List[RecoParticle]
            List of reconstructed particle objects

        Returns
        -------
        List[RecoInteraction]
            List of constructed reconstructed interaction instances
        """
        # Loop over unique interaction IDs
        reco_interactions = []
        inter_ids = np.array([p.interaction_id for p in reco_particles])
        for i, inter_id in enumerate(np.unique(inter_ids)):
            # Get the list of particles associates with this interaction
            assert inter_id > -1, (
                    "Invalid reconstructed interaction ID found.")
            particle_ids = np.where(inter_ids == inter_id)[0]
            inter_particles = [reco_particles[j] for j in particle_ids]

            # Build interaction
            interaction = RecoInteraction.from_particles(inter_particles)
            interaction.id = i

            # Match the interaction ID of the constituent particles
            for p in inter_particles:
                p.interaction_id = i

            # Append
            reco_interactions.append(interaction)

        return reco_interactions

    def build_truth(self, data):
        """Builds :class:`TruthInteraction` objects from the full chain output.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        List[TruthInteraction]
            List of constructed truth interaction instances
        """
        return self._build_truth(**data)

    def _build_truth(self, truth_particles, neutrinos=None):
        """Builds :class:`TruthInteraction` objects from the full chain output.

        This class builds an interaction by assembling particles together.

        Parameters
        ----------
        truth_particles : List[TruthParticle]
            List of truth particle objects
        neutrinos : List[Neutrino], optional
            List of true neutrino information from the generator
        """
        # Loop over unique interaction IDs
        truth_interactions = []
        inter_ids = np.array([p.interaction_id for p in truth_particles])
        unique_inter_ids = np.unique(inter_ids)
        valid_inter_ids = unique_inter_ids[unique_inter_ids > -1]
        for i, inter_id in enumerate(valid_inter_ids):
            # Get the list of particles associates with this interaction
            particle_ids = np.where(inter_ids == inter_id)[0]
            inter_particles = [truth_particles[j] for j in particle_ids]

            # Build interaction
            interaction = TruthInteraction.from_particles(inter_particles)
            interaction.id = i
            interaction.orig_id = inter_id

            # Match the interaction ID of the constituent particles
            for p in inter_particles:
                p.orig_interaction_id = inter_id
                p.interaction_id = i

            # Append the neutrino information, if it is provided
            nu_ids = [part.nu_id for part in inter_particles]
            assert len(np.unique(nu_ids)) == 1, (
                    "Interaction made up of particles with different "
                    "neutrino IDs. Must be unique.")
            interaction.nu_id = nu_ids[0]

            if neutrinos is not None and nu_ids[0] > -1:
                interaction.attach_neutrino(neutrinos[nu_ids[0]])

            else:
                anc_pos = [part.ancestor_position for part in inter_particles]
                anc_pos = np.unique(anc_pos, axis=0)
                if len(anc_pos) != 1:
                    warn("Particles making up a true interaction have "
                         "different ancestor positions.")
                    anc_pos = np.max(anc_pos, axis=0)
                interaction.vertex = anc_pos.flatten()

            # Append
            truth_interactions.append(interaction)

        return truth_interactions

    def load_reco(self, data):
        """Load :class:`RecoInteraction` objects from their stored versions.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        List[RecoInteraction]
            List of restored reconstructed interaction instances
        """
        return self._load_reco(**data)

    def _load_reco(self, reco_interactions, reco_particles):
        """Load :class:`RecoInteraction` objects from their stored versions.

        Parameters
        ----------
        reco_interactions : List[RecoInteraction]
            List of partial reconstructed interaction objects
        reco_particles : List[RecoParticle]
            List of reconstructed particle objects

        Returns
        -------
        List[RecoInteraction]
            List of restored reconstructed interaction instances
        """
        # Loop over the dictionaries
        for i, interaction in enumerate(reco_interactions):
            # Check that the interaction ID checks out
            assert interaction.id == i, (
                    "The ordering of the stored ineractions is wrong.")

            # Fetch and assign the list of particles matched to this interaction
            inter_particles = [
                    reco_particles[p] for p in interaction.particle_ids]
            assert len(inter_particles), (
                    "Every interaction should contain >= 1 particle.")
            interaction.particles = inter_particles

            # Update the interaction with its long-form attributes
            for attr in interaction._cat_attrs:
                val_list = [getattr(p, attr) for p in inter_particles]
                setattr(interaction, attr, np.concatenate(val_list))

        return reco_interactions

    def load_truth(self, data):
        """Load :class:`TruthInteraction` objects from their stored versions.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        List[TruthInteraction]
            List of restored truth interaction instances
        """
        return self._load_truth(**data)

    def _load_truth(self, truth_interactions, truth_particles):
        """Load :class:`TruthInteraction` objects from their stored versions.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Parameters
        ----------
        truth_interactions : List[TruthInteraction]
            List of partial truth interaction objects
        truth_particles : List[TruthParticle]
            List of truth particle objects

        Returns
        -------
        List[TruthInteraction]
            List of restored truth interaction instances
        """
        # Loop over the dictionaries
        for i, interaction in enumerate(truth_interactions):
            # Check that the interaction ID checks out
            assert interaction.id == i, (
                    "The ordering of the stored ineractions is wrong.")

            # Fetch and assign the list of particles matched to this interaction
            inter_particles = [
                    truth_particles[p] for p in interaction.particle_ids]
            assert len(inter_particles), (
                    "Every interaction should contain >= 1 particle.")
            interaction.particles = inter_particles

            # Update the interaction with its long-form attributes
            for attr in interaction._cat_attrs:
                val_list = [getattr(p, attr) for p in inter_particles]
                setattr(interaction, attr, np.concatenate(val_list))

        return truth_interactions
