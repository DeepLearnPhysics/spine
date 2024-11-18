"""Classes in charge of constructing Fragment objects."""

from typing import List
from dataclasses import dataclass
import inspect

import numpy as np
from scipy.special import softmax

from spine.data.out import RecoFragment, TruthFragment
from spine.utils.decorators import inherit_docstring
from spine.utils.globals import CLUST_COL, PART_COL, TRACK_SHP

from .base import BuilderBase

__all__ = ['FragmentBuilder']


@inherit_docstring(BuilderBase)
class FragmentBuilder(BuilderBase):
    """Builds reconstructed and truth fragments.

    It takes the raw output of the reconstruction chain, extracts the
    necessary information and builds :class:`RecoFragment` and
    :class:`TruthFragment` objects from it.
    """

    # Builder name
    name = 'fragment'

    # Types of objects constructed by the builder
    _reco_type = RecoFragment
    _truth_type = TruthFragment

    # Necessary/optional data products to build a reconstructed object
    _build_reco_keys  = (
            ('fragment_clusts', True), ('fragment_shapes', True),
            ('fragment_start_points', False), ('fragment_end_points', False),
            ('fragment_group_pred', False), ('fragment_node_pred', False),
            *BuilderBase._build_reco_keys
    )

    # Necessary/optional data products to build a truth object
    _build_truth_keys = (
            ('particles', False),
            *BuilderBase._build_truth_keys
    )

    # Necessary/optional data products to load a reconstructed object
    _load_reco_keys  = (
            ('reco_fragments', True),
            *BuilderBase._load_reco_keys
    )

    # Necessary/optional data products to load a truth object
    _load_truth_keys  = (
            ('truth_fragments', True),
            *BuilderBase._load_truth_keys
    )

    def build_reco(self, data):
        """Builds :class:`RecoFragment` objects from the full chain output.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        List[RecoFragment]
            List of constructed reconstructed fragment instances
        """
        return self._build_reco(**data)

    def _build_reco(self, points, depositions, fragment_clusts, fragment_shapes,
                    fragment_start_points=None, fragment_end_points=None,
                    fragment_group_pred=None, fragment_node_pred=None,
                    sources=None):
        """Builds :class:`RecoFragment` objects from the full chain output.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) Set of deposition coordinates in the image
        depositions : np.ndarray
            (N) Set of deposition values
        fragment_clusts : List[np.ndarray]
            (P) List of indexes, each corresponding to a fragment instance
        fragment_shapes : np.ndarray
            (P) List of fragment shapes (shower, track, etc.)
        fragment_start_points : np.ndarray, optional
            (P, 3) List of fragment start point coordinates
        fragment_end_points : np.ndarray, optional
            (P, 3) List of fragment end point coordinates
        fragment_group_pred : np.ndarray, optional
            (P) Interaction group each fragment belongs to
        sources : np.ndarray, optional
            (N, 2) Tensor which contains the module/tpc information

        Returns
        -------
        List[RecoFragment]
            List of constructed reconstructed fragment instances
        """
        # Convert the logits to softmax scores and the scores to a prediction
        if fragment_node_pred is not None:
            primary_scores = softmax(fragment_node_pred, axis=1)
            primary_pred = np.argmax(primary_scores, axis=1)

        # Loop over the fragment instances
        reco_fragments = []
        for i, index in enumerate(fragment_clusts):
            # Initialize
            fragment = RecoFragment(
                    id=i,
                    shape=fragment_shapes[i],
                    index=index,
                    points=points[index],
                    depositions=depositions[index])

            # Add optional arguments
            if sources is not None:
                fragment.sources = sources[index]
            if fragment_start_points is not None:
                fragment.start_point = fragment_start_points[i]
            if fragment_end_points is not None and fragment.shape == TRACK_SHP:
                fragment.end_point = fragment_end_points[i]
            if fragment_group_pred is not None:
                fragment.particle_id = fragment_group_pred[i]
            if fragment_node_pred is not None:
                fragment.primary_scores = primary_scores[i]
                fragment.is_primary = bool(primary_pred[i])

            # Append
            reco_fragments.append(fragment)

        return reco_fragments

    def build_truth(self, data):
        """Builds :class:`TruthFragment` objects from the full chain output.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        List[TruthFragment]
            List of constructed true fragment instances
        """
        return self._build_truth(**data)

    def _build_truth(self, label_tensor, label_adapt_tensor, points,
                     depositions, points_label, depositions_label,
                     depositions_q_label=None, label_g4_tensor=None,
                     points_g4=None, depositions_g4=None, sources=None,
                     sources_label=None, particles=None):
        """Builds :class:`TruthFragment` objects from the full chain output.

        Parameters
        ----------
        label_tensor : np.ndarray
            Tensor which contains the cluster labels of each deposition
        label_adapt_tensor : np.ndarray
            Tensor which contains the cluster labels of each deposition,
            adapted to the semantic segmentation prediction.
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
        particles : List[Particle], optional
            List of true particles

        Returns
        -------
        List[TruthFragment]
            List of constructed true fragment instances
        """
        # Check once if the fragment labels are untouched
        broken = (label_tensor[:, CLUST_COL] != label_tensor[:, PART_COL]).any()

        # Loop over the true fragment instances in the *adapted* label tensor.
        # The label tensor does not necessarily contain the correct fragments.
        truth_fragments = []
        unique_fragment_ids = np.unique(label_adapt_tensor[:, CLUST_COL])
        valid_fragment_ids = unique_fragment_ids[unique_fragment_ids > -1]
        for i, frag_id in enumerate(valid_fragment_ids):
            # Initialize fragment
            fragment = TruthFragment(id=i)

            # Find the particle which matches this fragment best
            index_adapt = np.where(
                    label_adapt_tensor[:, CLUST_COL] == frag_id)[0]
            if particles is not None:
                part_ids, counts = np.unique(
                        label_adapt_tensor[index_adapt, PART_COL],
                        return_counts=True)
                part_id = int(part_ids[np.argmax(counts)])
                if part_id > -1:
                    assert part_id < len(particles), (
                            "Invalid particle ID found in fragment labels.")
                    fragment = TruthFragment(**particles[part_id].as_dict())
                    fragment.id = i

            # Always fill adapted long-form attributes
            fragment.index_adapt = index_adapt
            fragment.points_adapt = points[index_adapt]
            fragment.depositions_adapt = depositions[index_adapt]
            if sources is not None:
                fragment.sources_adapt = sources[index_adapt]

            # If the input cluster label is not adapted, fill other long-form
            if id(label_tensor) == id(label_adapt_tensor):
                # Update the fragment with its true long-form attributes
                index = np.where(label_tensor[:, CLUST_COL] == frag_id)[0]
                fragment.index = index
                fragment.points = points_label[index]
                fragment.depositions = depositions_label[index]
                if depositions_q_label is not None:
                    fragment.depositions_q = depositions_q_label[index]
                if sources_label is not None:
                    fragment.sources = sources_label[index]

                # If the fragments are not broken, can match to G4 info
                if not broken:
                    # If available, append the Geant4 information
                    if label_g4_tensor is not None:
                        index_g4 = np.where(
                                label_g4_tensor[:, CLUST_COL] == frag_id)[0]
                        fragment.index_g4 = index_g4
                        fragment.points_g4 = poins_g4[index_g4]
                        fragment.depositions_g4 = depositions_g4[index_g4]

            # Append
            truth_fragments.append(fragment)

        return truth_fragments

    def load_reco(self, data):
        """Load :class:`RecoFragment` objects from their stored versions.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        List[RecoFragment]
            List of restored reconstructed fragment instances
        """
        return self._load_reco(**data)

    def _load_reco(self, reco_fragments, points, depositions, sources=None):
        """Load :class:`RecoFragment` objects from their stored versions.

        Parameters
        ----------
        reco_fragments : List[RecoFragment]
            (F) List of partial reconstructed fragments
        points : np.ndarray
            (N, 3) Set of deposition coordinates in the image
        depositions : np.ndarray
            (N) Set of deposition values
        sources : np.ndarray, optional
            (N, 2) Tensor which contains the module/tpc information

        Returns
        -------
        List[RecoFragment]
            List of restored reconstructed fragment instances
        """
        # Loop over the dictionaries
        for i, fragment in enumerate(reco_fragments):
            # Check that the fragment ID checks out
            assert fragment.id == i, (
                    "The ordering of the stored fragments is wrong.")

            # Update the fragment with its long-form attributes
            fragment.points = points[fragment.index]
            fragment.depositions = depositions[fragment.index]
            if sources is not None:
                fragment.sources = sources[fragment.index]

        return reco_fragments

    def load_truth(self, data):
        """Load :class:`TruthFragment` objects from their stored versions.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Returns
        -------
        List[TruthFragment]
            List of restored true fragment instances
        """
        return self._load_truth(**data)

    def _load_truth(self, truth_fragments, points, depositions, points_label,
                    depositions_label, depositions_q_label=None, points_g4=None,
                    depositons_g4=None, sources=None, sources_label=None):
        """Load :class:`TruthFragment` objects from their stored versions.

        Parameters
        ----------
        reco_fragments : List[TruthFragment]
            (F) List of partial truth fragments
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
        List[TruthFragment]
            List of restored true fragment instances
        """
        # Loop over the dictionaries
        for i, fragment in enumerate(truth_fragments):
            # Check that the fragment ID checks out
            assert fragment.id == i, (
                    "The ordering of the stored fragments is wrong.")

            # Update the fragment with its long-form attributes
            fragment.points = points_label[fragment.index]
            fragment.depositions = depositions_label[fragment.index]
            if depositions_q_label is not None:
                fragment.depositions_q = depositions_q_label[fragment.index]
            if sources_label is not None:
                fragment.sources = sources_label[fragment.index]

            fragment.points_adapt = points[fragment.index_adapt]
            fragment.depositions_adapt = depositions[fragment.index_adapt]
            if sources is not None:
                fragment.sources_adapt = sources[fragment.index_adapt]

            if points_g4 is not None:
                fragment.points_g4 = points_g4[fragment.index_g4]
                fragment.depositions_g4 = depositions_g4[fragment.index_g4]

        return truth_fragments
