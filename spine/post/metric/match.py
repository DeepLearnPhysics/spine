"""Match objects and their label counterparts and vice versa."""

from dataclasses import dataclass

import numpy as np
import numba as nb

import spine.utils.match

from spine.post.base import PostBase

__all__ = ['MatchProcessor']


class MatchProcessor(PostBase):
    """Does the matching between reconstructed and true objects."""

    # Name of the post-processor (as specified in the configuration)
    name = 'match'

    def __init__(self, fragment=None, particle=None, interaction=None,
                 truth_point_mode='points', **kwargs):
        """Initializes the matching post-processor.

        Parameters
        ----------
        fragment: Union[bool, dict], optional
            Matching flag or configuration for fragments
        particle: Union[bool, dict], optional
            Matching flag or configuration for particles
        interaction: Union[bool, dict], optional
            Matching flag or configuration for interactions
        truth_point_mode : str, default 'points'
            Type of truth points to use to do the matching
        **kwargs : dict, optional
            Matching parameters shared between all matching processes
        """
        # Initialize the necessary matchers
        configs = {'fragment': fragment, 'particle': particle,
                   'interaction': interaction}
        keys = {}
        self.matchers = {}
        for key, cfg in configs.items():
            if cfg is not None and cfg != False:
                # Initialize the matcher
                if isinstance(cfg, bool):
                    cfg = {}
                self.matchers[key] = self.Matcher(**cfg, **kwargs)

                # If any matcher includes ghost points, must load meta
                if self.matchers[key].ghost:
                    keys['meta'] = True

        assert len(self.matchers), (
                "Must specify one of 'fragment', 'particle' or 'interaction'.")

        # Update the set of keys necessary for this post-processor
        self.update_keys(keys)

        # Initialize the parent class
        super().__init__(list(self.matchers.keys()), 'both', truth_point_mode)

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
        ghost : bool, default False
            Whether a deghosting process was applied (in which case the indexes
            of the reco and the truth particles do not align)
        """
        fn: object = None
        match_mode: str = 'both'
        overlap_mode: str = 'iou'
        min_overlap: float = 0.
        weight_overlap: bool = False
        ghost: bool = False

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

            # Fetch the metadata, if needed
            meta = None
            if matcher.ghost:
                meta = data['meta']

            # Pass it to the individual processor
            res_one = self.process_single(
                    reco_objs, truth_objs, matcher, name, meta)
            result.update(**res_one)

        return result

    def process_single(self, reco_objs, truth_objs, matcher, name, meta=None):
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
        meta : Meta, optional
            Metadata information to convert position to index
        """
        # Convert the object list into an index/coordinate list
        if matcher.overlap_mode != 'chamfer':
            # For overlap matches, use pixel indexes (faster)
            if not matcher.ghost or self.truth_point_mode == 'points_adapt':
                # The indexes of reco and truth point to the same point set
                reco_input = nb.typed.List([self.get_index(p) for p in reco_objs])
                truth_input = nb.typed.List([self.get_index(p) for p in truth_objs])

            else:
                # The indexes of reco and truth point to different point sets.
                # In this case, convert the positions to indexes
                reco_input = []
                for p in reco_objs:
                    coords = self.get_points(p)
                    if p.units != 'px':
                        coords = meta.to_px(coords, floor=True)
                    reco_input.append(meta.index(coords))

                truth_input = []
                for p in truth_objs:
                    coords = self.get_points(p)
                    if p.units != 'px':
                        coords = meta.to_px(coords, floor=True)
                    truth_input.append(meta.index(coords))

        else:
            # For the chamfer distance, simply use the point positions
            reco_input = nb.typed.List([self.get_points(p) for p in reco_objs])
            truth_input = nb.typed.List([self.get_points(p) for p in truth_objs])

        # Pass lists to the matching function to compute overlaps
        if len(reco_input) and len(truth_input):
            ovl_matrix = matcher.fn(reco_input, truth_input)
        else:
            ovl_matrix = np.empty((len(reco_input), len(truth_input)))

        # Make the overlap selection cut, if requested
        if matcher.overlap_mode != 'chamfer':
            ovl_valid = ovl_matrix > matcher.min_overlap
        else:
            ovl_valid = ovl_matrix < match.min_overlap

        # Produce matches
        result = {}
        if matcher.match_mode != 'truth_to_reco':
            pairs, overlaps = self.generate_matches(
                    reco_objs, truth_objs, ovl_matrix, ovl_valid)
            result[f'{name}_matches_r2t'] = pairs
            result[f'{name}_matches_r2t_overlap'] = overlaps

        if matcher.match_mode != 'reco_to_truth':
            pairs, overlaps = self.generate_matches(
                    truth_objs, reco_objs, ovl_matrix.T, ovl_valid.T)
            result[f'{name}_matches_t2r'] = pairs
            result[f'{name}_matches_t2r_overlap'] = overlaps

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
