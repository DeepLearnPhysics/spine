"""Point-coordinate and label helpers for LArCV parsers."""

from warnings import warn

import numpy as np
from numpy.typing import DTypeLike

from spine.constants import LOWES_SHP, TRACK_SHP, UNKWN_SHP

__all__ = [
    "get_ppn_labels",
    "get_vertex_labels",
    "image_contains",
    "image_coordinates",
    "image_coordinates_batch",
]


# The PPN label contract exposes independent filtering and output controls.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def get_ppn_labels(
    particle_v,
    meta,
    dtype,
    dim=3,
    min_voxel_count=1,
    min_energy_deposit=0,
    include_point_tagging=True,
):
    """Gets particle point coordinates and informations for running PPN.

    We skip some particles under specific conditions (e.g. low energy deposit,
    low voxel count, nucleus track, etc.)

    Parameters
    ----------
    particle_v : List[larcv.Particle]
        List of LArCV particle objects in the image
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    dtype : str
        Typing of the output PPN labels
    dim : int, default 3
        Number of dimensions of the image
    min_voxel_count : int, default 5
        Minimum number of voxels associated with a particle to be included
    min_energy_deposit : float, default 0
        Minimum energy deposition associated with a particle to be included
    include_point_tagging : bool, default True
        If True, include an a label of 0 for start points and 1 for end points

    Returns
    -------
    np.array
        Array of points of shape (N, 5/6) where 5/6 = x,y,z + point type
        + particle index [+ start (0) or end (1) point tagging]
    """
    # Check on dimension
    if dim not in [2, 3]:
        raise ValueError(
            "The image dimension must be either 2 or 3, " f"got {dim} instead."
        )

    # Loop over true particles
    part_info = []
    for part_index, particle in enumerate(particle_v):
        # Check that the particle has the expected index
        if part_index != particle.id():
            warn("Particle list index does not match its `id` attribute.")

        # If the particle does not meet minimum energy/size requirements, skip
        if (
            particle.energy_deposit() < min_energy_deposit
            or particle.num_voxels() < min_voxel_count
        ):
            continue

        # Nuclear fragments do not define useful PPN targets.
        pdg_code = abs(particle.pdg_code())
        if pdg_code > 1000000000:
            continue

        # Shower starts outside the image cannot be represented as targets.
        if pdg_code in (11, 22):
            if not image_contains(meta, particle.first_step(), dim):
                continue

        # Skip low energy scatters and unknown shapes
        shape = particle.shape()
        if particle.shape() in [LOWES_SHP, UNKWN_SHP]:
            continue

        # Append the start point with the rest of the particle information
        first_step = image_coordinates(meta, particle.first_step(), dim)
        part_extra = (
            [shape, part_index, 0] if include_point_tagging else [shape, part_index]
        )
        part_info.append(first_step + part_extra)

        # Append the end point as well, for tracks only
        if shape == TRACK_SHP:
            last_step = image_coordinates(meta, particle.last_step(), dim)
            part_extra = (
                [shape, part_index, 1] if include_point_tagging else [shape, part_index]
            )
            part_info.append(last_step + part_extra)

    if len(part_info) == 0:
        return np.empty((0, 5 + include_point_tagging), dtype=dtype)

    return np.array(part_info, dtype=dtype)


def get_vertex_labels(particle_v, neutrino_v, meta, dtype):
    """Gets particle vertex coordinates.

    It provides the coordinates of points where multiple particles originate:

    - If `neutrino_v` is provided, it uses the neutrino interaction points.
    - If `particle_v` is provided instead, it looks for ancestor positions
      shared by at least two primary particles.

    Parameters
    ----------
    particle_v : List[larcv.Particle]
        List of LArCV particle objects in the image
    neutrino_v : List[larcv.Neutrino]
        List of LArCV neutrino objects in the image
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    dtype : str
        Typing of the output PPN labels

    Returns
    -------
    np.array
        Array of points of shape (N, 4) where 4 = x, y, z, vertex_id
    """
    # If the particles are provided, find unique ancestors
    vertexes = []
    if particle_v is not None:
        # Fetch all ancestor positions of primary particles
        anc_positions = []
        for i, p in enumerate(particle_v):
            if p.parent_id() == p.id() or p.ancestor_pdg_code() == 111:
                if image_contains(meta, p.ancestor_position()):
                    anc_pos = image_coordinates(meta, p.ancestor_position())
                    anc_positions.append(anc_pos)

        # If there is no primary, nothing to do
        if len(anc_positions) == 0:
            return np.empty((0, 4), dtype=dtype)

        # Find those that appear > once
        anc_positions = np.vstack(anc_positions)
        unique_positions, counts = np.unique(anc_positions, return_counts=True, axis=0)
        for i, idx in enumerate(np.where(counts > 1)[0]):
            vertexes.append([*unique_positions[idx], i])

    # If the neutrinos are provided, straightforward
    if neutrino_v is not None:
        for i, n in enumerate(neutrino_v):
            if image_contains(meta, n.position()):
                nu_pos = image_coordinates(meta, n.position())
                vertexes.append([*nu_pos, i])

    # If there are no vertex, nothing to do
    if len(vertexes) == 0:
        return np.empty((0, 4), dtype=dtype)

    return np.vstack(vertexes).astype(dtype)


def image_contains(meta, point, dim=3):
    """Checks whether a point is contained in the image box defined by meta.

    Parameters
    ----------
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    point : larcv::Point3D or larcv::Point2D
        Point to check on
    dim: int, default 3
         Number of dimensions of the image

    Returns
    -------
    bool
        True if the point is contained in the image box
    """
    if dim == 3:
        return (
            point.x() >= meta.min_x()
            and point.y() >= meta.min_y()
            and point.z() >= meta.min_z()
            and point.x() <= meta.max_x()
            and point.y() <= meta.max_y()
            and point.z() <= meta.max_z()
        )
    return (
        point.x() >= meta.min_x()
        and point.x() <= meta.max_x()
        and point.y() >= meta.min_y()
        and point.y() <= meta.max_y()
    )


def image_coordinates(meta, point, dim=3):
    """Returns the coordinates of a point in units of pixels with an image.

    Parameters
    ----------
    meta : larcv::Voxel3DMeta or larcv::ImageMeta
        Metadata information
    point : larcv::Point3D or larcv::Point2D
        Point to convert the units of
    dim: int, default 3
         Number of dimensions of the image

    Returns
    -------
    bool
        True if the point is contained in the image box
    """
    x, y = point.x(), point.y()
    if dim == 3:
        z = point.z()
        x = (x - meta.min_x()) / meta.size_voxel_x()
        y = (y - meta.min_y()) / meta.size_voxel_y()
        z = (z - meta.min_z()) / meta.size_voxel_z()
        return [x, y, z]
    x = (x - meta.min_x()) / meta.size_voxel_x()
    y = (y - meta.min_y()) / meta.size_voxel_y()
    return [x, y]


# Coordinate conversion deliberately caches each C++ metadata accessor.
# pylint: disable-next=too-many-locals
def image_coordinates_batch(
    meta,
    objects,
    dim=3,
    dtype: DTypeLike = np.float32,
    position_attr=None,
):
    """Convert a sequence of physical positions to image coordinates.

    Unlike :func:`image_coordinates`, this function fetches the image origin
    and voxel sizes only once. This matters for LArCV objects because each
    metadata or point accessor crosses the Python/C++ boundary.

    Parameters
    ----------
    meta : larcv.Voxel3DMeta or larcv.ImageMeta
        Image metadata used for the coordinate conversion.
    objects : iterable
        Sequence of LArCV point objects, or objects which provide the position
        accessor specified by ``position_attr``.
    dim : int, default 3
        Number of spatial dimensions.
    dtype : numpy dtype, default numpy.float32
        Output coordinate dtype.
    position_attr : str, optional
        Name of the position getter to call on each input object, e.g.
        ``"ancestor_position"`` for a sequence of LArCV particles. Leaving
        this unset treats each input object as a position directly.

    Returns
    -------
    numpy.ndarray
        ``(N, dim)`` array of coordinates in voxel units.
    """
    if not hasattr(objects, "__len__"):
        objects = list(objects)
    coords = np.empty((len(objects), dim), dtype=dtype)

    min_x, min_y = meta.min_x(), meta.min_y()
    size_x, size_y = meta.size_voxel_x(), meta.size_voxel_y()
    min_z, size_z = 0.0, 1.0
    if dim == 3:
        min_z, size_z = meta.min_z(), meta.size_voxel_z()

    if position_attr is not None and len(objects):
        # Resolve bound C++ method dispatch once instead of using getattr for
        # every particle. PyROOT exposes its methods on the proxy type.
        position_getter = getattr(type(objects[0]), position_attr)
        for i, obj in enumerate(objects):
            point = position_getter(obj)
            coords[i, 0] = (point.x() - min_x) / size_x
            coords[i, 1] = (point.y() - min_y) / size_y
            if dim == 3:
                coords[i, 2] = (point.z() - min_z) / size_z
    else:
        for i, point in enumerate(objects):
            coords[i, 0] = (point.x() - min_x) / size_x
            coords[i, 1] = (point.y() - min_y) / size_y
            if dim == 3:
                coords[i, 2] = (point.z() - min_z) / size_z

    return coords
