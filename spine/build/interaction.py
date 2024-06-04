"""Class in charge of constructing *Interaction objects."""

from collections import defaultdict
from dataclasses import asdict

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
    name = 'interaction'

    reco_type = RecoInteraction
    truth_type = TruthInteraction

    build_reco_keys = {
            'reco_particles': True
    }

    build_truth_keys = {
            'truth_particles': True, 'neutrinos': False
    }

    load_reco_keys = {
            'reco_interactions': True, 'reco_particles': True
    }

    load_truth_keys = {
            'truth_interactions': True, 'truth_particles': True
    }

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
        for i, inter_id in enumerate(np.unique(inter_ids)):
            # Skip if the interaction ID is invalid
            if inter_id < 0:
                continue

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
            if neutrinos is not None:
                nu_ids = [p.nu_id for p in inter_particles]
                assert len(np.unique(nu_ids)) == 1, (
                        "Interaction made up of particles with different "
                        "neutrino IDs. Must be unique.")
                if nu_ids[0] > -1:
                    interaction.attach_neutrino(neutrinos[nu_ids[0]])

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
        return self._load_reco(data)

    def _load_reco(self, reco_particles, reco_interactions):
        """Load :class:`RecoInteraction` objects from their stored versions.

        Parameters
        ----------
        reco_particles : List[RecoParticle]
            List of reconstructed particle objects
        reco_interactions : List[dict]
            List of dictionary representations of reconstructed interactions

        Returns
        -------
        List[RecoInteraction]
            List of restored reconstructed interaction instances
        """
        # Loop over the dictionaries
        for i, inter_dict in enumerate(reco_interactions):
            # Pass the dictionary to build the interaction
            interaction = RecoInteraction(**inter_dict)
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

            # Append
            reco_interactions[i] = interaction

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
        return self._load_truth(data)

    def load_truth(self, truth_particles, truth_interactions):
        """Load :class:`TruthInteraction` objects from their stored versions.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Parameters
        ----------
        truth_particles : List[TruthParticle]
            List of truth particle objects
        truth_interactions : List[dict]
            List of dictionary representations of truth interactions

        Returns
        -------
        List[TruthInteraction]
            List of restored truth interaction instances
        """
        # Loop over the dictionaries
        for i, inter_dict in enumerate(truth_interactions):
            # Pass the dictionary to build the interaction
            interaction = RecoInteraction(**inter_dict)
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

            # Append
            interactions.append(interaction)

        return interaction
