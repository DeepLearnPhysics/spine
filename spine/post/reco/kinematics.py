import numpy as np

from spine.utils.globals import (
        TRACK_SHP, PID_MASSES, SHP_TO_PID, SHP_TO_PRIMARY)

from spine.post.base import PostBase

__all__ = ['ParticleShapeLogicProcessor', 'ParticleThresholdProcessor',
           'InteractionTopologyProcessor']


class ParticleShapeLogicProcessor(PostBase):
    """Enforce logical connections between semantic predictions and
    particle-level predictions (PID and primary).

    In particular:
    - If a particle has shower shape, it can only have a shower PID
    - If a particle has track shape, it can only have a track PID
    - If a particle has delta/michel shape, it can only be a secondary electron
    """
    name = 'shape_logic'
    aliases = ['enforce_particle_semantics']

    def __init__(self, enforce_pid=True, enforce_primary=True):
        """Store information about which particle properties should
        or should not be updated.

        Parameters
        ----------
        enforce_pid : bool, default True
            Enforce the PID prediction based on the semantic type
        enforce_primary : bool, default True
            Enforce the primary prediction based on the semantic type
        """
        # Intialize the parent class
        super().__init__('particle', 'reco')

        # Store parameters
        self.enforce_pid = enforce_pid
        self.enforce_primary = enforce_primary

    def process(self, data):
        """Update PID and primary predictions of each particle in one entry

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the particle objects
        for part in data['reco_particles']:
            # Reset the PID scores
            if self.enforce_pid:
                pid_range = SHP_TO_PID[part.shape]
                pid_range = pid_range[pid_range < len(part.pid_scores)]

                pid_scores = np.zeros(
                        len(part.pid_scores), dtype=part.pid_scores.dtype)
                pid_scores[pid_range] = part.pid_scores[pid_range]
                pid_scores /= np.sum(pid_scores)
                part.pid_scores = pid_scores
                part.pid = np.argmax(pid_scores)

            # Reset the primary scores
            if self.enforce_primary:
                primary_range = SHP_TO_PRIMARY[part.shape]

                primary_scores = np.zeros(
                        len(part.primary_scores),
                        dtype=part.primary_scores.dtype)
                primary_scores[primary_range] = part.primary_scores[primary_range]
                primary_scores /= np.sum(primary_scores)
                part.primary_scores = primary_scores
                part.is_primary = np.argmax(primary_scores)


class ParticleThresholdProcessor(PostBase):
    """Adjust the particle PID and primary properties according to customizable
    thresholds and priority orderings.
    """
    name = 'particle_threshold'
    aliases = ['adjust_particle_properties']

    def __init__(self, shower_pid_thresholds=None, track_pid_thresholds=None,
                 primary_threshold=None):
        """Store the new thresholds to be used to update the PID and primary
        information of particles.

        Parameters
        ----------
        shower_pid_thresholds : dict, optional
            Dictionary which maps an EM PID output to a threshold value,
            in order
        track_pid_thresholds : dict, optional
            Dictionary which maps a track PID output to a threshold value,
            in order
        primary_treshold : float, optional
            Primary score above which a particle is considered a primary
        """
        # Intialize the parent class
        super().__init__('particle', 'reco')

        # Check that there is something to do, throw otherwise
        if (shower_pid_thresholds is not None and
            track_pid_thresholds is not None and primary_threshold is None):
            raise ValueError(
                    "Specify one of `shower_pid_thresholds`, `track_pid_thresholds` "
                    "or `primary_threshold` for this function to do anything.")

        # Store the thresholds
        self.shower_pid_thresholds = shower_pid_thresholds
        self.track_pid_thresholds = track_pid_thresholds
        self.primary_threshold = primary_threshold

    def process(self, data):
        """Update PID predictions of each particle one entry.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the particle objects
        for part in data['reco_particles']:
            # Fetch the appropriate thresholds
            if part.shape == TRACK_SHP:
                pid_thresholds = self.track_pid_thresholds
            else:
                pid_thresholds = self.shower_pid_thresholds

            # Adjust the particle ID
            if pid_thresholds is not None:
                assigned = False
                scores = np.copy(part.pid_scores)
                for k, v in pid_thresholds.items():
                    if scores[k] >= v:
                        # Assign a PID
                        part.pid = k
                        assigned = True
                        break
                    else:
                        # Re-normalize softmax probabilities
                        scores *= 1./(1 - scores[k])

                assert assigned, (
                        "Must specify a PID threshold for all or no particle "
                        "type.")

            # Adjust the primary ID
            if self.primary_threshold is not None:
                part.is_primary = part.primary_scores[1] >= self.primary_threshold


class InteractionTopologyProcessor(PostBase):
    """Adjust the topology of interactions by applying thresholds on the
    minimum kinetic energy of particles.
    """
    name = 'topology_threshold'
    aliases = ['adjust_interaction_topology']

    def __init__(self, ke_thresholds, reco_ke_mode='ke',
                 truth_ke_mode='energy_deposit', run_mode='both'):
        """Store the new thresholds to be used to update interaction topologies.

        Parameters
        ----------
        ke_thresholds : Union[float, dict]
            If a scalr, it specifies a blanket KE cut to apply to all
            particles. If it is a dictionary, it maps an PID to a KE threshold.
            If a 'default' key is provided, it is used for all particles,
            unless a number is provided for a specific PID.
        reco_ke_mode : str, default 'ke'
            Which `Particle` attribute to use to apply the KE thresholds
        truth_ke_mode : str, default 'energy_deposit'
            Which `TruthParticle` attribute to use to apply the KE thresholds
        """
        # Initialize the run mode
        super().__init__('interaction', run_mode)

        # Store the attributes that should be used to evaluate the KE
        self.reco_ke_mode = reco_ke_mode
        self.truth_ke_mode = truth_ke_mode

        # Store the thresholds in a dictionary
        if np.isscalar(ke_thresholds):
            ke_thresholds = {'default': float(ke_thresholds)}

        self.ke_thresholds = {}
        for pid in PID_MASSES.keys():
            if pid in ke_thresholds:
                self.ke_thresholds[pid] = ke_thresholds[pid]
            elif 'default' in ke_thresholds:
                self.ke_thresholds[pid] = ke_thresholds['default']
            else:
                self.ke_thresholds[pid] = 0.

    def process(self, data):
        """Update each interaction topology in one interaction.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over the interaction types
        for k in self.interaction_keys:
            # Check which attribute should be used for KE
            if 'reco' in k:
                ke_attr = self.reco_ke_mode
            else:
                ke_attr = self.truth_ke_mode

            # Loop over interactions
            for inter in data[k]:
                # Loop over particles, select the ones that pass a threshold
                for part in inter.particles:
                    ke = getattr(part, ke_attr)
                    if ke_attr == 'energy_init' and part.pid > -1:
                        ke -= PID_MASSES[part.pid]
                    if part.pid > -1 and ke < self.ke_thresholds[part.pid]:
                        part.is_valid = False
                    else:
                        part.is_valid = True
