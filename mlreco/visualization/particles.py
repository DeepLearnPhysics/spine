import numpy as np

from .points import scatter_points
from .plotly_layouts import HIGH_CONTRAST_COLORS

from mlreco.utils.globals import COORD_COLS, PART_COL


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
    for i in range(len(particles)):
        # Get a mask that corresponds to the particle entry, skip if empty
        mask = cluster_label[:, part_col] == i
        if not np.sum(mask):
            continue
            
        # Initialize the information string
        p = particles[i]
        pos_str = ', '.join([f'{p.position[i]:0.3e}' for i in range(3)])
        start_str = ', '.join([f'{p.first_step[i]:0.3e}' for i in range(3)])
        anc_pos_str = ', '.join(
                [f'{p.ancestor_position[i]:0.3e}' for i in range(3)])
        
        label = f'Particle {p.id}'
        hovertext_dict = {'Particle ID': p.id,
                          'Group ID': p.group_id,
                          'Parent ID': p.parent_id,
                          'Inter. ID': p.interaction_id,
                          'Neutrino ID': p.nu_id,
                          'Type ID': p.pid,
                          'Shower primary': p.shower_primary,
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
                          'Time': f'{p.t} ns',
                          'Position': pos_str,
                          'Start point': start_str,
                          'Anc. start point': anc_pos_str}

        hovertext = ''.join(
                [f'{l}:   {v}<br>' for l, v in hovertext_dict.items()])
        
        # Append a scatter plot trace
        trace = scatter_points(
                cluster_label[mask][:, COORD_COLS],
                color=str(colors[i%len(colors)]), hovertext=hovertext,
                markersize=markersize, **kwargs)
        trace[0]['name'] = label
        
        traces += trace
        
    return traces
