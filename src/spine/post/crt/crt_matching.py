"""Post-processor in charge of finding matches between charge and CRT."""

import numpy as np

from spine.post.base import PostBase

from .match import CRTMatcher

__all__ = ["CRTMatchProcessor"]


class CRTMatchProcessor(PostBase):
    """Associates TPC particles with CRT hits."""

    # Name of the post-processor (as specified in the configuration)
    name = "crt_match"

    # Alternative allowed names of the post-processor
    aliases = ("run_crt_matching",)

    # Set of post-processors which must be run before this one is
    _upstream = ("direction",)

    def __init__(self, crt_key, run_mode="reco", **kwargs):
        """Initialize the CRT/TPC matching post-processor.

        Parameters
        ----------
        crt_key : str
            Data product key which provides the CRT information
        **kwargs : dict
            Keyword arguments to pass to the CRT-TPC matching algorithm
        """
        # Initialize the parent class
        super().__init__("particle", run_mode)

        # Make sure the CRT hit data product is available, store
        self.crt_key = crt_key
        self.update_keys({crt_key: True})

        # Initialzie the matcher
        self.matcher = CRTMatcher(**kwargs)

    def process(self, data):
        """Find [particle, crthit] pairs.

        Parameters
        ----------
        data : dict
            Dictionary of data products

        Notes
        -----
        This post-processor modifies the list of `particle` objects
        in-place by filling the following attributes:
        - particle.is_crt_matched: (bool)
               Indicator for whether the given particle has a CRT hit match
        - particle.crt_ids: np.ndarray
               The CRT hit IDs in the CRT hit list
        - particle.crt_times: np.ndarray
               The CRT hit time(s) in microseconds
        """
        # Fetch the CRT hits, nothing to do here if there are none
        crthits = data[self.crt_key]
        if not len(crthits):
            return

        # Loop over the object types
        for k in self.particle_keys:
            # Fetch particle list, nothing to do here if there are none
            particles = data[k]
            if not len(particles):
                continue

            # Make sure the particle coordinates are expressed in cm
            self.check_units(particles[0])

            # Clear previous crt matching information
            for part in particles:
                part.reset_crt_match(typed=False)

            # Run CRT matching
            matches = self.matcher.get_matches(particles, crthits)

            # Store the CRT information
            for part, crt, match in matches:
                part.is_crt_matched = True
                part.crt_ids.append(crt.id)
                part.crt_times.append(crt.time)
                part.crt_scores.append(match)

            # Cast list attributes to numpy arrays
            for part in particles:
                part.crt_ids = np.asarray(part.crt_ids, dtype=np.int32)
                part.crt_times = np.asarray(part.crt_times, dtype=np.float32)
                part.crt_scores = np.asarray(part.crt_scores, dtype=np.float32)
