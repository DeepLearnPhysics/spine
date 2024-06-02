"""Tools to draw true particles information."""

import numpy as np

from spine.utils.globals import PART_COL

from .point import scatter_points
from .layout import HIGH_CONTRAST_COLORS


def scatter_particles(cluster_label, particles, part_col=PART_COL,
                      markersize=1, **kwargs):
    """Builds a graph of true particles in the image.

    Function which returns a graph object per true particle in the
    particle list, provided that the particle deposited energy in the
    detector which appears in the cluster_label tensor.

    Parameters
    ----------
    cluster_label : np.ndarray
        (N, M) Tensor of pixel coordinates and their associated cluster ID
    particles : List[Particle]
        (P) List of true particle objects
    part_col : int
        Index of the column in the label tensor that contains the particle ID
    **kwargs : dict, optional
        List of additional arguments to pass to plotly.graph_objs.Scatter3D that
        make up the output list

    Returns
    -------
    List[plotly.graph_objs.Scatter3D]
        List of particle traces
    """
    # Initialize one graph per particle
    traces = []
    colors = HIGH_CONTRAST_COLORS
    for i, p in enumerate(particles):
        # Get a mask that corresponds to the particle entry, skip if empty
        index = np.where(cluster_label[:, part_col] == i)[0]
        if not index.shape[0]:
            continue

        # Initialize the information string
        label = f'Particle {p.id}'
        hovertext_dict = {'Particle ID': p.id,
                          'Group ID': p.group_id,
                          'Parent ID': p.parent_id,
                          'Inter. ID': p.interaction_id,
                          'Neutrino ID': p.nu_id,
                          'Type ID': p.pid,
                          'Group primary': p.group_primary,
                          'Inter. primary': p.interaction_primary,
                          'Shape ID': p.shape,
                          'PDG code': p.pdg_code,
                          'Parent PDG code': p.parent_pdg_code,
                          'Anc. PDG code': p.ancestor_pdg_code,
                          'Process': p.creation_process,
                          'Parent process': p.parent_creation_process,
                          'Anc. process': p.ancestor_creation_process,
                          'Initial E': f'{p.energy_init:0.1f} MeV',
                          'Deposited E': f'{p.energy_deposit:0.1f} MeV',
                          'Time': f'{p.t:0.1f} ns',
                          'First step': p.first_step,
                          'Last step': p.last_step,
                          'Creation point': p.position,
                          'Anc. creation point': p.ancestor_position}

        hovertext = ''.join(
                [f'{l}:   {v}<br>' for l, v in hovertext_dict.items()])

        # Append a scatter plot trace
        trace = scatter_points(
                cluster_label[index],
                color=str(colors[i%len(colors)]), hovertext=hovertext,
                markersize=markersize, name=label, **kwargs)

        traces += trace

    return traces
