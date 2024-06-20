"""Match objects and their label counterparts and vice versa."""

from dataclasses import dataclass

import numpy as np
import numba as nb

import spine.utils.match

from spine.post.base import PostBase

__all__ = ['MatchProcessor']


class MatchProcessor(PostBase):
    """Does the matching between reconstructed and true objects."""
    name = 'match'
    
    def __init__(self, fragment=None, particle=None, interaction=None):
        """Initializes the matching post-processor.
        
        Parameters
        ----------
        fragment: dict, optional
            Matching configuration for fragments
        particle: dict, optional
            Matching configuration for particles
        interaction: dict, optional
            Matching configuration for interactions
        """
        # Initialize the necessary matchers
        self.matchers = {}
        if fragment is not None:
            self.matchers['fragment'] = self.Matcher(**fragment)
        if particle is not None:
            self.matchers['particle'] = self.Matcher(**particle)
        if interaction is not None:
            self.matchers['interaction'] = self.Matcher(**interaction)

        assert len(self.matchers), (
                "Must specify one of 'fragment', 'particle' or 'interaction'.")

        # Initialize the parent class
        super().__init__(list(self.matchers.keys()), 'both')

    @dataclass
    class Matcher:
        """Simple data class to store matching methods per object.

        Attributes
        ----------
        fn : function
            Function which computes overlaps between pairs of objects
        match_mode : str, defualt 'both'
            Matching mode. One of 'reco_to_truth', 'truth_to_reco' or 'both'
        overlap_mode : str, default 'iou'
            Overlap estimatation method. One of 'count', 'iou', 'dice', 'chamfer'
        min_overlap : float, default 0.
            Overlap value above which a pair is considered a match
        weight_overlap : bool, default False
            Whether to weight the overlap metric
        """
        fn: object = None
        match_mode: str = 'both'
        overlap_mode: str = 'iou'
        min_overlap: float = 0.
        weight_overlap: bool = False

        # Valid match modes
        _match_modes = ['reco_to_truth', 'truth_to_reco', 'both', 'all']

        # Valid overlap modes
        _overlap_modes = ['count', 'iou', 'dice', 'chamfer']

        def __post_init__(self):
            """Check that the values provided are valid."""
            # Check match mode
            assert self.match_mode in self._match_modes, (
                f"Invalid matching mode: {self.match_mode}. Must be one "
                f"of {self._match_modes}.")

            # Check the overlap mode
            assert self.overlap_mode in self._overlap_modes, (
                f"Invalid overlap computation mode: {self.overlap_mode}. "
                f"Must be one of {self._overlap_modes}.")

            # Check that the overlap mode and weighting are compatible
            assert not self.weight_overlap or overlap in ['iou', 'dice'], (
                    "Only IoU and Dice-based overlap functions can be weighted.")

            # Initialize the match overlap function
            prefix = 'overlap' if not self.weight_overlap else 'overlap_weighted'
            self.fn = getattr(
                    spine.utils.match, f'{prefix}_{self.overlap_mode}')
        
    def process(self, data):
        """Match all the requested objects in one entry.

        Parameters
        ----------
        data: dict
            Dictionary of data products
        """
        # Loop over the matchers
        result = {}
        for name, matcher in self.matchers.items():
            # Fetch the required data products
            reco_objs = data[f'reco_{name}s']
            truth_objs = data[f'truth_{name}s']

            # Pass it to the individual processor
            res_one = self.process_single(reco_objs, truth_objs, matcher, name)
            result.update(**res_one)

        return result

    def process_single(self, reco_objs, truth_objs, matcher, name):
        """Match all the requested objects in a single category.

        Parameters
        ----------
        reco_objs : List[object]
            List of reconstructed objects
        truth_objs : List[object]
            List of truth objects
        matcher : MatchProcessor.Matcher
            Matching method and function
        name : str
            Object type name
        """
        # Convert the object list into an index/coordinate list
        # TODO: more flexibility with indexes needed! Should be able
        # to convert coordinates to index (with meta) and use that to
        # match across points from different input/label tensors
        if matcher.overlap_mode != 'chamfer':
            reco_input = nb.typed.List([p.index for p in reco_objs])
            truth_input = nb.typed.List([p.index for p in truth_objs])
        else:
            reco_input = nb.typed.List([p.points for p in reco_objs])
            truth_input = nb.typed.List([p.points for p in truth_objs])

        # Pass lists to the matching function to compute overlaps
        # TODO: the validity check makes no sense for Chamfer distance
        ovl_matrix = matcher.fn(reco_input, truth_input)
        ovl_valid = ovl_matrix > matcher.min_overlap

        # Produce matches
        result = {}
        if matcher.match_mode != 'truth_to_reco':
            pairs, overlaps = self.generate_matches(
                    reco_objs, truth_objs, ovl_matrix, ovl_valid)
            result[f'{name[:-1]}_matches_r2t'] = pairs
            result[f'{name[:-1]}_matches_r2t_overlap'] = overlaps

        if matcher.match_mode != 'reco_to_truth':
            pairs, overlaps = self.generate_matches(
                    truth_objs, reco_objs, ovl_matrix.T, ovl_valid.T)
            result[f'{name[:-1]}_matches_t2r'] = pairs
            result[f'{name[:-1]}_matches_t2r_overlap'] = overlaps

        return result

    @staticmethod
    def generate_matches(source_objs, target_objs, ovl_matrix, ovl_valid):
        """Generate pairs for a srt of sources and targets.

        Parameters
        ----------
        source_objs : List[object]
            (N) List of source objects
        target_objs : List[object]
            (M) List of truth objects
        ovl_matrix : np.ndarray
            (N, M) Matrix of overlap values
        ovl_valid : np.ndarray
            (N, M) Matrix of overlap validity

        Returns
        -------
        pairs : List[tuple]
            (N) List of (source, target) matched pairs (best match only)
        overlaps : List[float]
            (N) List of overlap between each source and the best matched target
        """
        # Build the matches based on the threshold
        pairs, pair_overlaps = [], []
        for i, s in enumerate(source_objs):
            # Get the list of valid matches
            match_idxs = np.where(ovl_valid[i])[0]
            if not len(match_idxs):
                # If there are no matches, fill dummy values
                s.is_matched = False
                s.match_ids = np.empty(0, dtype=np.int64)
                s.match_overlaps = np.empty(0, dtype=np.float32)

                pairs.append((s, None))
                pair_overlaps.append(-1.)

            else:
                # If there are matches, order them by decreasing overlap
                overlaps = ovl_matrix[i, match_idxs]
                perm = np.argsort(overlaps)[::-1]
                s.is_matched = True
                s.match_ids = match_idxs[perm]
                s.match_overlaps = overlaps[perm]

                best_idx = s.match_ids[0]
                pairs.append((s, target_objs[best_idx]))
                pair_overlaps.append(s.match_overlaps[0])

        # Fill the match lists
        return pairs, pair_overlaps
