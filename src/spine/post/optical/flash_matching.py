"""Post-processor in charge of finding matches between charge and light."""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from spine.data import FlashHypothesis
from spine.data.out.base import OutBase
from spine.geo import GeoManager
from spine.post.base import PostBase
from spine.utils.optical import FlashMerger

from .barycenter import BarycenterFlashMatcher
from .likelihood import LikelihoodFlashMatcher

__all__ = ["FlashMatchProcessor"]


class FlashMatchProcessor(PostBase):
    """Associates TPC interactions with optical flashes."""

    # Name of the post-processor (as specified in the configuration)
    name = "flash_match"

    # Alternative allowed names of the post-processor
    aliases = ("run_flash_matching",)

    # Whether this post-processor may use paths relative to the parent configuration path
    provide_parent_path = True

    # Whether the post-processor needs geometry information
    need_geometry = True

    def __init__(
        self,
        flash_key: str,
        volume: str,
        ref_volume_id: int | None = None,
        method: str = "likelihood",
        time_contained: bool = False,
        max_cathode_offset: float | None = None,
        run_mode: str = "reco",
        truth_point_mode: str = "points",
        truth_dep_mode: str = "depositions",
        parent_path: str | None = None,
        merge: Mapping[str, Any] | None = None,
        update_flashes: bool = False,
        store_hypotheses: bool = False,
        store_all_hypotheses: bool = False,
        hypothesis_key: str = "flash_hypotheses",
        **kwargs: Any,
    ) -> None:
        """Initialize the flash matching algorithm.

        Parameters
        ----------
        flash_key : str
            Flash data product name. In most cases, this is unambiguous, unless
            there are multiple types of segregated optical detectors
        volume : str
            Physical volume corresponding to each flash ('module' or 'tpc')
        ref_volume_id : str, optional
            If specified, the flash matching expects all interactions/flashes
            to live into a specific optical volume. Must shift everything.
        method : str, default 'likelihood'
            Flash matching method (one of 'likelihood' or 'barycenter')
        time_contained : bool, default False
            If `True`, only match interactions which are time contained
        max_cathode_offset : float, optional
            If specified, only match cathode-crossing interactions which are
            offset from the cathode less than this threshold
        parent_path : str, optional
            Path to the parent directory of the main analysis configuration.
            This allows for the use of relative paths in the post-processors.
        merge : dict, optional
            Flash merging configuration
        update_flashes : bool, default False
            If `True` and merging flashes, replaces the original list of
            flashes in place with the list of merged flashes
        store_hypotheses : bool, default False
            If ``True``, store the predicted per-channel optical response for
            every eligible interaction and optical volume
        store_all_hypotheses : bool, default False
            If ``True``, store the match-specific prediction and score for
            every positive-scoring interaction/flash candidate. This implies
            ``store_hypotheses`` and requires ``StoreFullResult: true`` in the
            OpT0Finder ``FlashMatchManager`` configuration
        hypothesis_key : str, default 'flash_hypotheses'
            Event-data key under which optical hypotheses are stored
        **kwargs : dict
            Keyword arguments to pass to specific flash matching algorithms
        """
        # Initialize the parent class
        super().__init__(
            "interaction",
            run_mode,
            truth_point_mode,
            truth_dep_mode,
            parent_path=parent_path,
        )

        # Make sure the flash data product is available, store
        self.flash_key: str = flash_key
        self.update_keys({flash_key: True})

        # Get the volume within which each flash is confined
        if volume not in ("tpc", "module"):
            raise ValueError("The `volume` must be one of 'tpc' or 'module'.")
        self.volume = volume
        self.ref_volume_id = ref_volume_id

        # Store the timing checks to be performed
        self.time_contained = time_contained
        self.max_cathode_offset = max_cathode_offset
        if self.time_contained:
            self.update_upstream("time_containment")
        if self.max_cathode_offset is not None:
            self.update_upstream("cathode_crosser")

        # Fetch the detector geometry
        self.geo = GeoManager.get_instance()

        # Initialize the flash matching algorithm
        self.method = method
        self.store_all_hypotheses = store_all_hypotheses
        self.store_hypotheses = store_hypotheses or store_all_hypotheses
        self.hypothesis_key = hypothesis_key
        if self.store_hypotheses and method != "likelihood":
            raise ValueError(
                "Optical hypotheses can only be stored with likelihood matching."
            )

        if method == "barycenter":
            self.matcher = BarycenterFlashMatcher(**kwargs)

        elif method == "likelihood":
            self.matcher = LikelihoodFlashMatcher(
                detector=self.geo.name.lower(), parent_path=self.parent_path, **kwargs
            )

        else:
            raise ValueError(f"Flash matching method not recognized: {method}")

        # Initialize the flash merging class, if needed
        self.merger = None
        if merge is not None:
            self.merger = FlashMerger(**merge)
        self.update_flashes = update_flashes

    def process(self, data: Mapping[str, Any]) -> dict[str, Any] | None:
        """Find [interaction, flash] pairs.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Notes
        -----
        This post-processor modifies the list of `interaction` objects
        in-place by filling the following attributes:
        - interaction.is_flash_matched: (bool)
               Indicator for whether the given interaction has a flash match
        - interaction.flash_ids: np.ndarray
               The flash IDs in the flash list
        - interaction.flash_volume_ids: np.ndarray
               The flash optical volume IDs in the flash list
        - interaction.flash_times: np.ndarray
               The flash time(s) in microseconds
        - interaction.flash_scores: np.ndarray
               The flash scores(s) (larger is better)
        - interaction.flash_total_pe: float
               Total number of PEs associated with the matched flash(es)
        - interaction.flash_hypo_pe: float, optional
               Total number of PEss associated with the hypothesis flash
        - interaction.flash_hypothesis_ids: np.ndarray, optional
               IDs of stored per-channel optical hypotheses
        """
        # Fetch the optical flashes
        flashes = data[self.flash_key]

        # Merge flashes based on timing, if requested
        orig_ids = np.arange(len(flashes))
        if self.merger is not None:
            flashes, orig_ids = self.merger(flashes)

        # Loop over the optical volumes, run flash matching
        volume_ids = np.asarray([f.volume_id for f in flashes], dtype=np.int32)
        if self.store_hypotheses:
            if self.geo.optical is None:
                raise RuntimeError(
                    "Cannot produce optical hypotheses without optical geometry."
                )
            volume_ids = np.union1d(
                volume_ids, np.arange(self.geo.optical.num_volumes, dtype=np.int32)
            )

        hypotheses: list[FlashHypothesis] = []
        for k in self.interaction_keys:
            # Fetch interactions, nothing to do if there are not any
            interactions = data[k]
            if len(interactions) == 0:
                continue

            # Make sure the interaction coordinates are expressed in cm
            self.check_units(interactions[0])

            # Clear previous flash matching information
            for inter in interactions:
                inter.reset_flash_match()

            # Loop over the optical volumes
            flash_ids = defaultdict(list)
            flash_volume_ids = defaultdict(list)
            flash_times = defaultdict(list)
            flash_scores = defaultdict(list)
            hypothesis_ids = defaultdict(list)
            for volume_id in np.unique(volume_ids):
                # Get the list of flashes associated with this optical volume
                flashes_v = []
                for flash in flashes:
                    if flash.volume_id == volume_id:
                        flashes_v.append(flash)

                # Crop interactions to only include depositions in the optical volume
                interactions_v = []
                for inter in interactions:
                    # If requested, skip interactions which are not time contained
                    if self.time_contained and not inter.is_time_contained:
                        continue

                    # If requested, skip out-of-time cathode crossers
                    if (
                        self.max_cathode_offset is not None
                        and inter.is_cathode_crosser
                        and abs(inter.cathode_offset) > self.max_cathode_offset
                    ):
                        continue

                    # Fetch the points in the current optical volume
                    sources = self.get_sources(inter)
                    if self.volume == "module":
                        index = self.geo.get_volume_index(sources, volume_id)

                    elif self.volume == "tpc":
                        num_cpm = self.geo.tpc.num_chambers_per_module
                        module_id, tpc_id = volume_id // num_cpm, volume_id % num_cpm
                        index = self.geo.get_volume_index(sources, module_id, tpc_id)
                    else:
                        raise ValueError(f"Volume not recognized: {self.volume}")

                    # If there are no points in this volume, proceed
                    if len(index) == 0:
                        continue

                    # Fetch points and depositions
                    points = self.get_points(inter)[index]
                    depositions = self.get_depositions(inter)[index]
                    if self.ref_volume_id is not None:
                        # If the reference volume is specified, shift positions
                        points = self.geo.translate(
                            points, volume_id, self.ref_volume_id
                        )

                    # Create an interaction which holds positions/depositions
                    inter_v = OutBase(
                        id=inter.id, points=points, depositions=depositions
                    )
                    interactions_v.append(inter_v)

                # Run flash matching
                matches = self.matcher.get_matches(interactions_v, flashes_v)

                # Produce one standalone hypothesis per eligible interaction.
                # Match-specific predictions replace these below when available.
                volume_hypotheses = {}
                if self.store_hypotheses:
                    matcher = cast(LikelihoodFlashMatcher, self.matcher)
                    predictions = matcher.get_hypotheses(interactions_v)
                    for inter_v, pe_per_ch in predictions:
                        inter = interactions[inter_v.id]
                        hypothesis = FlashHypothesis(
                            id=len(hypotheses),
                            interaction_id=inter.id,
                            volume_id=int(volume_id),
                            is_truth=bool(inter.is_truth),
                            pe_per_ch=pe_per_ch,
                        )
                        hypotheses.append(hypothesis)
                        volume_hypotheses[inter.id] = hypothesis
                        hypothesis_ids[inter.id].append(hypothesis.id)

                    # By default, retain only accepted match predictions. If
                    # requested, replace these with every scored candidate.
                    hypothesis_matches = matches
                    if self.store_all_hypotheses and interactions_v and flashes_v:
                        hypothesis_matches = matcher.get_match_candidates()

                    num_hypothesis_matches = defaultdict(int)
                    for inter_v, flash, match in hypothesis_matches:
                        match_obj: Any = match
                        if not hasattr(match_obj, "hypothesis"):
                            continue

                        inter = interactions[inter_v.id]
                        match_pe = np.asarray(
                            list(match_obj.hypothesis), dtype=np.float32
                        )
                        score = (
                            float(match_obj.score)
                            if hasattr(match_obj, "score")
                            else -1.0
                        )
                        if self.merger is not None and not self.update_flashes:
                            matched_flash_ids = np.asarray(
                                [
                                    data[self.flash_key][i].id
                                    for i in orig_ids[flash.id]
                                ],
                                dtype=np.int32,
                            )
                        else:
                            matched_flash_ids = np.asarray([flash.id], dtype=np.int32)

                        if (
                            num_hypothesis_matches[inter.id] == 0
                            and inter.id in volume_hypotheses
                        ):
                            hypothesis = volume_hypotheses[inter.id]
                            hypothesis.pe_per_ch = match_pe
                            hypothesis.flash_ids = matched_flash_ids
                            hypothesis.score = score
                        else:
                            hypothesis = FlashHypothesis(
                                id=len(hypotheses),
                                interaction_id=inter.id,
                                volume_id=int(volume_id),
                                is_truth=bool(inter.is_truth),
                                pe_per_ch=match_pe,
                                flash_ids=matched_flash_ids,
                                score=score,
                            )
                            hypotheses.append(hypothesis)
                            hypothesis_ids[inter.id].append(hypothesis.id)
                        num_hypothesis_matches[inter.id] += 1

                # Store flash information
                for inter_v, flash, match in matches:
                    # Get the interaction that matches the cropped version
                    inter = interactions[inter_v.id]

                    # Get the flash hypothesis (if the matcher produces one)
                    hypo_pe, score = -1.0, -1.0
                    if np.isscalar(match):
                        score = float(np.asarray(match, dtype=np.float64).item())
                    else:
                        match_obj: Any = match
                        if hasattr(match_obj, "hypothesis"):
                            match_pe = np.asarray(
                                list(match_obj.hypothesis), dtype=np.float32
                            )
                            hypo_pe = float(np.sum(match_pe))
                        if hasattr(match_obj, "score"):
                            score = float(match_obj.score)

                    # Update
                    if not inter.is_flash_matched:
                        inter.is_flash_matched = True
                        inter.flash_total_pe = float(flash.total_pe)
                        inter.flash_hypo_pe = hypo_pe
                    else:
                        inter.flash_total_pe += float(flash.total_pe)
                        inter.flash_hypo_pe += hypo_pe

                    if self.merger is not None and not self.update_flashes:
                        orig_flashes = [
                            data[self.flash_key][i] for i in orig_ids[flash.id]
                        ]
                        flash_ids[inter.id].extend([f.id for f in orig_flashes])
                        flash_volume_ids[inter.id].extend(
                            [f.volume_id for f in orig_flashes]
                        )
                        flash_times[inter.id].extend([f.time for f in orig_flashes])
                        flash_scores[inter.id].extend([score for _ in orig_flashes])
                    else:
                        flash_ids[inter.id].append(int(flash.id))
                        flash_volume_ids[inter.id].append(int(flash.volume_id))
                        flash_times[inter.id].append(float(flash.time))
                        flash_scores[inter.id].append(score)

            # Cast list attributes to numpy arrays
            for inter_id in flash_ids:
                inter = interactions[inter_id]
                inter.flash_ids = np.asarray(flash_ids[inter_id], dtype=np.int32)
                inter.flash_volume_ids = np.asarray(
                    flash_volume_ids[inter_id], dtype=np.int32
                )
                inter.flash_times = np.asarray(flash_times[inter_id], dtype=np.float32)
                inter.flash_scores = np.asarray(
                    flash_scores[inter_id], dtype=np.float32
                )

            for inter_id in hypothesis_ids:
                interactions[inter_id].flash_hypothesis_ids = np.asarray(
                    hypothesis_ids[inter_id], dtype=np.int32
                )

        # Return generated products, if requested
        result = {}
        if self.update_flashes:
            result[self.flash_key] = flashes
        if self.store_hypotheses:
            result[self.hypothesis_key] = hypotheses
        return result or None
