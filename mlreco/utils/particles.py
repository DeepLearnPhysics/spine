"""Particle truth information post-processors.

They add/correct information stored in LArCV particles.
"""

import larcv
import numpy as np
from typing import List
from warnings import warn

from .globals import (SHOWR_SHP, TRACK_SHP, MICHL_SHP, DELTA_SHP,
                      INVAL_ID, INVAL_TID, PDG_TO_PID)


def process_particles(particles, particle_event, particle_mpv_event=None,
                      neutrino_event=None):
    """Process Particle object list to add/correct attributes in place.

    Does the following:
    - Adds interaction ID information if it is not provided
    - Adds the true neutrino ID this particle came from
    - Adds a simplified enumerated particle species ID
    - Adds a flag as to whether a particle is a primary within its interaction
    - Adds a flag as to whether a particle is a primary within its group

    Parameters
    ----------
    particles : List[Particle]
        (P) List of true particle instances
    particle_event : larcv.EventParticle
        (P) List of true particle instances
    particle_mpv_event : larcv.EventParticle, optional
        (M) List of true MPV particle instances
    neutrino_event : larcv.EventNeutrino, optional
        (N) List of true neutrino instances
    """
    # If the list is empty, there is nothing to do
    if not len(particles):
        return

    # Get the additional attributes
    (interaction_ids, nu_ids, group_primary_ids,
     inter_primary_ids, pids) = process_particle_event(
             particle_event, particle_mpv_event, neutrino_event)

    # Update the particles objects in place
    for i, p in enumerate(particles):
        p.interaction_id = interaction_ids[i]
        p.nu_id = nu_ids[i]
        p.group_primary = group_primary_ids[i]
        p.interaction_primary = inter_primary_ids[i]
        p.pid = pids[i]


def process_particle_event(particle_event, particle_mpv_event=None,
                           neutrino_event=None):
    """Corrects/fetches attributes for a larcv.EventParticle object.

    Does the following:
    - Builds the interaction ID information if it is not provided
    - Gets the true neutrino ID this particle came from
    - Gets a simplified enumerated particle species ID
    - Gets a flag as to whether a particle is a primary within its interaction
    - Gets a flag as to whether a particle is a primary within its group

    Parameters
    ----------
    particle_event : larcv.EventParticle
        (P) List of true particle instances
    particle_mpv_event : larcv.EventParticle, optional
        (M) List of true MPV particle instances
    neutrino_event : larcv.EventNeutrino, optional
        (N) List of true neutrino instances

    Returns
    -------
    interaction_ids : np.ndarray
        (P) List of interaction IDs, one per true particle instance
    nu_ids : np.ndarray
        (P) List of neutrino IDs, one per true particle instance
    group_primary_ids : np.ndarray
        (P) List of particle group primary IDs, one per true particle instance
    inter_primary_ids : np.ndarray
        (P) List of particle primary IDs, one per true particle instance
    pids : np.ndarray
        (P) List of particle IDs, one per true particle instance
    """
    # Converts the input to simple python lists of objects
    particles = list(particle_event.as_vector())
    particles_mpv, neutrinos = None, None
    if particle_mpv_event is not None:
        particles_mpv = list(particle_mpv_event.as_vector())
    if neutrino_event is not None:
        neutrinos = list(neutrino_event.as_vector())

    # Get the mask of valid particle labels
    valid_mask = get_valid_mask(particles)

    # Get the interaction ID of each particle
    interaction_ids = get_interaction_ids(particles, valid_mask)

    # Get the neutrino ID of each particle
    nu_ids = get_nu_ids(particles, interaction_ids, particles_mpv, neutrinos)

    # Get the group primary status of each particle
    group_primary_ids = get_group_primary_ids(particles, valid_mask)

    # Get the interaction primary status of each particle
    inter_primary_ids = get_inter_primary_ids(particles, valid_mask)

    # Get the particle species (PID) of each particle
    pids = get_particle_ids(particles, valid_mask)

    # Return
    return interaction_ids, nu_ids, group_primary_ids, inter_primary_ids, pids


def get_valid_mask(particles):
    """Gets a mask corresponding to particles with valid labels.

    This function checks that the particle labels have been filled properly at
    the Supera level. It checks that the ancestor track ID of each particle is
    not an invalid number and that the ancestor creation process is filled.

    Parameters
    ----------
    particles : List[larcv.Particle]
        (P) List of true particle instances

    Results
    -------
    np.ndarray
        (P) Boolean list of validity, one per true particle instance
    """
    # If there are no particles, nothing to do here
    if not len(particles):
        return np.empty(0, dtype=bool)

    # If the interaction IDs are set in the particle tree, simply use that
    inter_ids = np.array([p.interaction_id() for p in particles], dtype=int)
    if np.any(inter_ids != INVAL_ID):
        return inter_ids != INVAL_ID

    # Otherwise, check that the ancestor track ID and creation process are valid
    mask  = np.array([p.ancestor_track_id() != INVAL_TID for p in particles])
    mask &= np.array([bool(p.ancestor_creation_process()) for p in particles])

    return mask


def get_interaction_ids(particles, valid_mask=None):
    """Gets the interaction ID of each particle.

    If the `interaction_id` attribute of the Particle class is filled,
    it simply uses that quantity.

    Otherwise, it leverages shared ancestor position as a basis for
    interaction building and sets the interaction ID to -1 for particles with
    invalid ancestor track IDs.

    Parameters
    ----------
    particles : List[larcv.Particle]
        (P) List of true particle instances
    valid_mask : np.ndarray, optional
        (P) Particle label validity mask

    Results
    -------
    np.ndarray
        (P) List of interaction IDs, one per true particle instance
    """
    # If there are no particles, nothing to do here
    if not len(particles):
        return np.empty(0, dtype=int)

    # Compute the validity mask if it is not provided
    if valid_mask is None:
        valid_mask = get_valid_mask(particles)

    # If the interaction IDs are set in the particle tree, simply use that
    inter_ids = np.array([p.interaction_id() for p in particles], dtype=int)
    if np.any(inter_ids != INVAL_ID):
        inter_ids[~valid_mask] = -1
        return inter_ids

    # Otherwise, define interaction IDs on the basis of sharing
    # an ancestor vertex position
    anc_pos = np.vstack([get_coords(p.ancestor_position()) for p in particles])
    inter_ids = np.unique(anc_pos, axis=0, return_inverse=True)[-1]

    # Now set the interaction ID of particles with an undefined ancestor to -1
    inter_ids[~valid_mask] = -1

    return inter_ids


def get_nu_ids(particles, inter_ids, particles_mpv=None, neutrinos=None):
    """Gets the neutrino-like ID of each partcile

    Convention: -1 for non-neutrinos, neutrino index for others

    If a list of multi-particle vertex (MPV) particles or neutrinos is
    provided, that information is leveraged to identify which interactions
    are neutrino-like and which are not.

    If `particles_mpv` and `neutrinos` are not specified, it assumes that
    only neutrino-like interactions have more than one true primary
    particle in a single interaction.

    Parameters
    ----------
    particles : List[larcv.Particle]
        (P) List of true particle instances
    inter_ids : np.ndarray
        (P) Array of interaction ID values, one per true particle instance
    particles_mpv : List[larcv.Particle], optional
        (M) List of true MPV particle instances
    neutrinos : list(larcv.Neutrino), optional
        (N) List of true neutrino instances

    Results
    -------
    np.ndarray
        (P) List of neutrino IDs, one per true particle instance
    """
    # If there are no particles, nothing to do here
    if not len(particles):
        return np.empty(0, dtype=int)

    # Make sure there is only either MPV particles or neutrinos specified, not both
    assert particles_mpv is None or neutrinos is None, (
            "Do not specify both particle_mpv_event and neutrino_event "
            "in `get_neutrino_ids`. Can only use one of them.")

    # Initialize neutrino IDs
    nu_ids = -np.ones(len(particles), dtype=int)
    if particles_mpv is None and neutrinos is None:
        # Warn that this is ad-hoc
        warn("Neutrino IDs are being produced on the basis of interaction "
             "multiplicity (i.e. neutrino if >= 2 primaries). This is "
             "not an exact method and might lead to unexpected results.")
             
        # Loop over the interactions
        primary_ids = get_inter_primary_ids(particles, inter_ids > -1)
        nu_id = 0
        for i in np.unique(inter_ids):
            # If the interaction ID is invalid, skip
            if i < 0: continue

            # If there are at least two primaries, the interaction is nu-like
            inter_index = np.where(inter_ids == i)[0]
            if np.sum(primary_ids[inter_index] == 1) > 1:
                nu_ids[inter_index] = nu_id
                nu_id += 1
    else:
        # Find the reference positions to gauge if a particle comes from a
        # nu-like interaction
        ref_pos = None
        if particles_mpv and len(particles_mpv):
            ref_pos = np.vstack(
                    [get_coords(p.position()) for p in particles_mpv])
            ref_pos = np.unique(ref_pos, axis=0)
        elif neutrinos and len(neutrinos):
            ref_pos = np.vstack([get_coords(n.position()) for n in neutrinos])

        # If any particle in an interaction shares its ancestor position with
        # an MPV particle or a neutrino, the whole interaction is a
        # nu-like interaction.
        if ref_pos is not None:
            anc_pos = np.vstack(
                    [get_coords(p.ancestor_position()) for p in particles])
            for i in np.unique(inter_ids):
                if i < 0: continue
                inter_index = np.where(inter_ids == i)[0]
                for ref_id, pos in enumerate(ref_pos):
                    if np.any((anc_pos[inter_index] == pos).all(axis=1)):
                        nu_ids[inter_index] = ref_id
                        break

    return nu_ids


def get_group_primary_ids(particles, valid_mask):
    """Gets the group primary status of particle fragments.

    This could be handled somewhere else (e.g. Supera).

    Parameters
    ----------
    particles : List[larcv.Particle]
        (P) List of true particle instances
    valid_mask : np.ndarray, optional
        (P) Particle label validity mask

    Results
    -------
    np.ndarray
        (P) List of particle group primary IDs, one per true particle instance
    """
    # Compute the validity mask if it is not provided
    if valid_mask is None:
        valid_mask = get_valid_mask(particles)

    # Loop over the list of particle groups
    primary_ids = np.zeros(len(particles), dtype=int)
    group_ids   = np.array([p.group_id() for p in particles], dtype=int)
    for g in np.unique(group_ids):
        # If the particle group has invalid labeling, the concept of group
        # primary is ill-defined
        p = particles[g]
        group_index = np.where(group_ids == g)[0]
        if g == INVAL_ID or not valid_mask[g]:
            primary_ids[group_index] = -1
            continue

        # If a group originates from a Delta or a Michel, it has a primary
        if p.shape() == MICHL_SHP or p.shape() == DELTA_SHP:
            primary_ids[g] = 1
            continue

        # If a particle group's parent fragment is the first in time,
        # it is a valid primary. TODO: use first step time.
        clust_times = np.array([particles[i].t() for i in group_index])
        min_id = np.argmin(clust_times)
        if group_index[min_id] == g:
            primary_ids[g] = 1

    return primary_ids


def get_inter_primary_ids(particles, valid_mask):
    """Gets the interaction primary ID for each particle.

    Parameters
    ----------
    particles : List[larcv.Particle]
        (P) List of true particle instances
    valid_mask : np.ndarray, optional
        (P) Particle label validity mask

    Results
    -------
    np.ndarray
        (P) List of particle primary IDs, one per true particle instance
    """
    # Compute the validity mask if it is not provided
    if valid_mask is None:
        valid_mask = get_valid_mask(particles)

    # Loop over the list of particles
    primary_ids = -np.ones(len(particles), dtype=int)
    for i, p in enumerate(particles):
        # If the particle has invalid labeling, it has invalid primary status
        if p.group_id() == INVAL_ID or not valid_mask[i]:
            continue

        # If the particle originates from a primary pi0, label as primary
        # Small issue with photo-nuclear activity here, but very rare
        group_p = particles[p.group_id()]
        if group_p.ancestor_pdg_code() == 111:
            primary_ids[i] = 1
            continue

        # If the origin of a particle agrees with the origin of its ancestor,
        # label as primary
        group_position = get_coords(group_p.position())
        ancestor_position = get_coords(p.ancestor_position())
        primary_ids[i] = (group_position == ancestor_position).all()

    return primary_ids


def get_particle_ids(particles, valid_mask):
    """Gets a particle species ID (PID) for each particle.
    
    This function ensures:
    - All shower daughters are labeled the same as their primary. This
      makes sense as otherwise an electron primary gets overruled by
      its many photon daughters (voxel-wise majority vote). This can
      lead to problems as, if an electron daughter is not clustered with
      the primary, it is labeled electron, which is counter-intuitive.
      This is handled downstream with the high_purity flag.
    - Particles that are not in the list target are labeled -1

    Parameters
    ----------
    particles : List[Particle]
        (P) List of true particle instances
    valid_mask : np.ndarray, optional
        (P) Particle label validity mask

    Returns
    -------
    np.ndarray
        (P) List of particle IDs, one per true particle instance
    """
    # Compute the validity mask if it is not provided
    if valid_mask is None:
        valid_mask = get_valid_mask(particles)

    # Loop over the list of particles
    particle_ids = -np.ones(len(particles), dtype=int)
    for i, p in enumerate(particles):
        # If the primary ID is invalid, skip
        if not valid_mask[i]: continue

        # If the particle type exists in the predefined list, assign
        group_id = p.group_id()
        t = particles[group_id].pdg_code()
        if t in PDG_TO_PID.keys():
            particle_ids[i] = PDG_TO_PID[t]

    return particle_ids


def get_coords(position):
    """Gets the coordinates of a larcv.Vertex object.

    Parameters
    ----------
    position : larcv.Vertex
        Encodes the position of a point with attributes x, y, z and t

    Returns
    -------
    List[float]
        Coordinates of the point (x, y, z)
    """
    return np.array([getattr(position, a)() for a in ['x', 'y', 'z']])
