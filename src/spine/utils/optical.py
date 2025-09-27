"""Defines objects and methods related to optical information."""

from copy import deepcopy

import numpy as np

from .geo import Geometry

__all__ = ["FlashMerger"]


class FlashMerger:
    """Class which takes care of merging flashes together."""

    def __init__(self, threshold=1.0, window=None, combine_volumes=True):
        """Initialize the flash merging class.

        Parameters
        ----------
        threshold : float, default 1.0
            Maximum time difference (in us) between two successive flashes for
            them to be merged into one combined flash
        window : List[float], optional
            Time window (in us) within which to merge flashes. If flash times
            are outside of this window they are not considered for merging.
        combine_volumes : bool, default True
            If `True`, merge flashes from different optical volumes
        """
        # Store merging parameters
        self.threshold = threshold
        self.window = window
        self.combine_volumes = combine_volumes

        # Check on the merging time window formatting
        assert window is None or (
            hasattr(window, "__len__") and len(window) == 2
        ), "The `window` parameter should be a list/tuple of two numbers."

    def __call__(self, flashes):
        """Combine flashes if they are compatible in time.

        Parameters
        ----------
        flashes : List[Flash]
            List of flash objects

        Returns
        -------
        List[Flash]
            (M) List of merged flashes
        List[List[int]]
            (M) List of original flash indexes which make up the merged flashes
        """
        # If there is less than two flashes, nothing to do
        if len(flashes) < 2:
            return flashes, np.arange(len(flashes))

        # Dispatch
        if not self.combine_volumes:
            # Only merge flashes when they belong to the same volume
            volume_ids = np.array([f.volume_id for f in flashes])
            new_flashes, orig_ids = [], []
            for volume_id in np.unique(volume_ids):
                index = np.where(volume_ids == volume_id)[0]
                flashes_i, orig_ids_i = self.merge([flashes[i] for i in index])
                new_flashes.extend(flashes_i)
                for ids in orig_ids_i:
                    orig_ids.append(index[ids])

            return new_flashes, orig_ids

        else:
            # Merge flashes regardless of their optical volume
            return self.merge(flashes)

    def merge(self, flashes):
        """Merge flashes if they are compatible in time.

        Parameters
        ----------
        flashes : List[Flash]
            List of flash objects

        Returns
        -------
        List[Flash]
            (M) List of merged flashes
        List[List[int]]
            (M) List of original flash indexes which make up the merged flashes
        """
        # Order the flashes in time, merge them if they are compatible
        times = [f.time for f in flashes]
        perm = np.argsort(times)
        new_flashes = [deepcopy(flashes[perm[0]])]
        new_flashes[-1].id = 0
        orig_ids = [[perm[0]]]
        in_window = True
        for i in range(1, len(perm)):
            # Check the both flashes to be merged are in the merging window
            prev, curr = flashes[perm[i - 1]], flashes[perm[i]]
            if self.window is not None:
                in_window = prev.time > self.window[0] and curr.time < self.window[1]

            # Check that the two consecutive flashes are compatible in time
            if in_window and (curr.time - prev.time) < self.threshold:
                # Merge the successive flashes if they are comptible in time
                new_flashes[-1].merge(curr)
                orig_ids[-1].append(perm[i])

            else:
                # If the two flashes are not compatible, add a new one
                new_flashes.append(deepcopy(curr))
                orig_ids.append([perm[i]])

                # Reset the flash index to match the new list
                new_flashes[-1].id = len(new_flashes) - 1

        # Reset the volume IDs, if necessary
        if self.combine_volumes:
            for flash in new_flashes:
                flash.volume_id = 0

        return new_flashes, orig_ids
