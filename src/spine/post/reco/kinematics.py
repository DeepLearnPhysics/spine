"""Kinematics update module."""

import numpy as np

from spine.post.base import PostBase
from spine.utils.globals import (
    MICHL_SHP,
    MUON_PID,
    PID_MASSES,
    PION_PID,
    SHOWR_SHP,
    SHP_TO_PID,
    SHP_TO_PRIMARY,
    TRACK_SHP,
)

__all__ = [
    "ParticleShapeLogicProcessor",
    "ParticleThresholdProcessor",
    "ParticleNeutrinoLogicProcessor",
    "InteractionTopologyProcessor",
]


class ParticleShapeLogicProcessor(PostBase):
    """Enforce logical connections between semantic predictions and
    particle-level predictions (PID and primary).

    In particular:
    - If a particle has shower shape, it can only have a shower PID
    - If a particle has track shape, it can only have a track PID
    - If a particle has delta/michel shape, it can only be a secondary electron

    Optionally:
    - If it has a calorimetric KE above some threshold, it cannot be a Michel
    """

    # Name of the post-processor (as specified in the configuration)
    name = "shape_logic"

    # Alternative allowed names of the post-processor
    aliases = ("enforce_particle_semantics",)

    def __init__(self, enforce_pid=True, enforce_primary=True, maximum_michel_ke=None):
        """Store information about which particle properties should
        or should not be updated.

        Parameters
        ----------
        enforce_pid : bool, default True
            Enforce the PID prediction based on the semantic type
        enforce_primary : bool, default True
            Enforce the primary prediction based on the semantic type
        maximum_michel_ke : float, optional
            If provided, the processor will not enforce secondary status
            for reconstructed Michel electrons above a certain kinetic energy
        """
        # Intialize the parent class
        super().__init__("particle", "reco")

        # Store parameters
        self.enforce_pid = enforce_pid
        self.enforce_primary = enforce_primary
        self.maximum_michel_ke = maximum_michel_ke

        # If the Michel KE is to be checked, must run the calorimetric KE PP
        if self.maximum_michel_ke is not None:
            self.update_upstream("calo_ke")

    def process(self, data):
        """Update PID and primary predictions of each particle in one entry

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over the particle objects
        for part in data["reco_particles"]:
            # If the particle is a Michel with too high a KE, override to shower
            if (
                self.maximum_michel_ke is not None
                and part.shape == MICHL_SHP
                and part.ke > self.maximum_michel_ke
            ):
                part.shape = SHOWR_SHP

            # Reset the PID scores based on shape
            if self.enforce_pid:
                pid_range = SHP_TO_PID[part.shape]
                pid_range = pid_range[pid_range < len(part.pid_scores)]

                pid_scores = np.zeros(len(part.pid_scores), dtype=part.pid_scores.dtype)
                pid_scores[pid_range] = part.pid_scores[pid_range]
                pid_scores /= np.sum(pid_scores)
                part.pid_scores = pid_scores
                part.pid = np.argmax(pid_scores)

            # Reset the primary scores based on shape
            if self.enforce_primary:
                primary_range = SHP_TO_PRIMARY[part.shape]

                primary_scores = np.zeros(
                    len(part.primary_scores), dtype=part.primary_scores.dtype
                )
                primary_scores[primary_range] = part.primary_scores[primary_range]
                primary_scores /= np.sum(primary_scores)
                part.primary_scores = primary_scores
                part.is_primary = bool(np.argmax(primary_scores))


class ParticleThresholdProcessor(PostBase):
    """Adjust the particle PID and primary properties according to customizable
    thresholds and priority orderings.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "particle_threshold"

    # Alternative allowed names of the post-processor
    aliases = ("adjust_particle_properties",)

    def __init__(
        self,
        shower_pid_thresholds=None,
        track_pid_thresholds=None,
        primary_threshold=None,
    ):
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
        super().__init__("particle", "reco")

        # Check that there is something to do, throw otherwise
        if (
            (shower_pid_thresholds is None)
            and (track_pid_thresholds is None)
            and (primary_threshold is None)
        ):
            raise ValueError(
                "Specify one of `shower_pid_thresholds`, `track_pid_thresholds` "
                "or `primary_threshold` for this class to do anything."
            )

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
        for part in data["reco_particles"]:
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
                        scores *= 1.0 / (1 - scores[k])

                assert assigned, (
                    "Must specify a PID threshold for all or no particle " "type."
                )

            # Adjust the primary ID
            if self.primary_threshold is not None:
                part.is_primary = bool(part.primary_scores[1] >= self.primary_threshold)


class ParticleNeutrinoLogicProcessor(PostBase):
    """Enforce that there is at most 1 primary lepton per interaction.

    In particular:
    - If there is no muon and the interactions with a MIP are required to have
      one, turn one of the MIPs into a muon (and neutralize the pion score)
    - If there are more than 1 muon per interaction, pick one muon and switch
      other muons to pions (and neutralize the muon score)
    """

    # Name of the post-processor (as specified in the configuration)
    name = "neutrino_logic"

    # Alternative allowed names of the post-processor
    aliases = ("enforce_neutrino_topology",)

    # Lepton selection method
    _methods = ("size", "score")

    def __init__(self, method="size", cc_only=True):
        """Store information about how to enforce neutrino logic.

        Parameters
        ----------
        method : str, default 'size'
            Method used to select the lepton: select the largest MIP
            ('size') or the MIP with the highest lepton score ('score')
        cc_only : bool, default `True`
            If there are no leptons but MIPs are present, ensure that one
            of the MIPs is labeled as a lepton (CC-like)
        """
        # Intialize the parent class
        super().__init__("particle", "reco")

        # Store parameters
        assert method in self._methods, (
            f"Lepton selection method not recognized ({method}). Must "
            f"be one of {self._methods}."
        )
        self.method = method
        self.cc_only = cc_only

    def process(self, data):
        """Update PID and primary predictions of each particle in one entry

        Parameters
        ----------
        data : dict
            Dictionaries of data products
        """
        # Loop over unique interaction groups
        particles = data["reco_particles"]
        inter_ids = np.array([part.interaction_id for part in particles])
        pids = np.array([part.pid for part in particles])
        pid_scores = np.vstack([part.pid_scores for part in particles])
        for inter_id in np.unique(inter_ids):
            # Build a mask for this interaction
            inter_index = np.where(inter_ids == inter_id)[0]

            # Count the number of MIPs in the event
            muon_index = inter_index[pids[inter_index] == MUON_PID]
            pion_index = inter_index[pids[inter_index] == PION_PID]
            mip_index = np.concatenate((muon_index, pion_index))

            # If this is a CC interaction but there are MIPs with no muons, correct
            if self.cc_only and len(muon_index) < 1 and len(pion_index) > 0:
                if self.method == "size":
                    amax = np.argmax([particles[i].size for i in mip_index])
                else:
                    amax = np.argmax([pid_scores[i][MUON_PID] for i in mip_index])

                best_id = mip_index[amax]
                particles[best_id].pid = MUON_PID
                particles[best_id].pid_scores[PION_PID] = -1.0

            # If there are more then 1 muon, down-select to 1
            if len(muon_index) > 1:
                if self.method == "size":
                    amax = np.argmax([particles[i].size for i in muon_index])
                else:
                    amax = np.argmax([pid_scores[i][MUON_PID] for i in muon_index])

                for i in muon_index:
                    if i != muon_index[amax]:
                        particles[i].pid = PION_PID
                        particles[i].pid_scores[MUON_PID] = -1.0


class InteractionTopologyProcessor(PostBase):
    """Adjust the topology of interactions by applying thresholds on the
    minimum kinetic energy of particles.
    """

    # Name of the post-processor (as specified in the configuration)
    name = "topology_threshold"

    # Alternative allowed names of the post-processor
    aliases = ("adjust_interaction_topology",)

    # Set of post-processors which must be run before this one is
    _upstream = ("calo_ke", "csda_ke", "mcs_ke")

    def __init__(
        self,
        ke_thresholds=None,
        reco_ke_mode="ke",
        truth_ke_mode="energy_deposit",
        run_mode="both",
    ):
        """Store the new thresholds to be used to update interaction topologies.

        Parameters
        ----------
        ke_thresholds : Union[float, dict]
            If a scalar, it specifies a blanket KE cut to apply to all
            particles. If it is a dictionary, it maps an PID to a KE threshold.
            If a 'default' key is provided, it is used for all particles,
            unless a number is provided for a specific PID.
        reco_ke_mode : str, default 'ke'
            Which `Particle` attribute to use to apply the KE thresholds
        truth_ke_mode : str, default 'energy_deposit'
            Which `TruthParticle` attribute to use to apply the KE thresholds
        """
        # Initialize the run mode
        super().__init__("interaction", run_mode)

        # Store the attributes that should be used to evaluate the KE
        self.reco_ke_mode = reco_ke_mode
        self.truth_ke_mode = truth_ke_mode

        # Store the thresholds in a dictionary
        if np.isscalar(ke_thresholds):
            ke_thresholds = {"default": float(ke_thresholds)}

        self.ke_thresholds = {}
        for pid in PID_MASSES.keys():
            if pid in ke_thresholds:
                self.ke_thresholds[pid] = ke_thresholds[pid]
            elif "default" in ke_thresholds:
                self.ke_thresholds[pid] = ke_thresholds["default"]
            else:
                self.ke_thresholds[pid] = 0.0

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
            if "reco" in k:
                ke_attr = self.reco_ke_mode
            else:
                ke_attr = self.truth_ke_mode

            # Loop over interactions
            for inter in data[k]:
                # Loop over particles, select the ones that pass a threshold
                for part in inter.particles:
                    ke = getattr(part, ke_attr)
                    if ke_attr == "energy_init" and part.pid > -1:
                        ke -= PID_MASSES[part.pid]
                    if part.pid > -1 and ke < self.ke_thresholds[part.pid]:
                        part.is_valid = False
                    else:
                        part.is_valid = True
