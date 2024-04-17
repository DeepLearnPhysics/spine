"""Fixtures used to text the parsers.

These fixtures generate dummy LArCV data, which is what the parsers expect
to receive as an input.
"""

import pytest

import numpy as np
from larcv import larcv


@pytest.fixture(name='sparse2d_event')
def fixture_sparse2d_event(request, N):
    """Generates one larcv.EventSparseTensor2D.

    Returns
    -------
    larcv.EventSparseTensor2D
        Single dummy 2D sparse tensor
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Intitialize one set of metadata per projection
    meta = []
    for p in range(3):
        meta.append(generate_meta2d(N, p))

    # Initialize a single larcv sparse tensor
    return generate_sparse2d_event(meta)


@pytest.fixture(name='sparse2d_event_list')
def fixture_sparse2d_event_list(request, N):
    """Generates one larcv.EventSparseTensor2D.

    Returns
    -------
    List[larcv.EventSparseTensor2D]
        List of dummy 2D sparse tensor
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Intitialize one set of metadata per projection
    meta = []
    for p in range(3):
        meta.append(generate_meta2d(N, p))

    # Initialize a single larcv sparse tensor
    event_list = []
    for _ in range(request.param):
        event_list.append(generate_sparse2d_event(meta))

    return event_list


@pytest.fixture(name='sparse3d_event')
def fixture_sparse3d_event(request, N):
    """Generates one larcv.EventSparseTensor3D.

    Returns
    -------
    larcv.EventSparseTensor3D
        Single dummy 3D sparse tensor
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Intitialize the metadata
    meta = generate_meta3d(N)

    # Initialize a single larcv sparse tensor
    return generate_sparse3d_event(meta)


@pytest.fixture(name='sparse3d_event_list')
def fixture_sparse3d_event_list(request, N):
    """Generates a list of larcv.EventSparseTensor3D.

    Returns
    -------
    List[larcv.EventSparseTensor3D]
        List of dummy 3D sparse tensor
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Intitialize the metadata
    meta = generate_meta3d(N)

    # Initialize a list of larcv sparse tensors
    event_list = []
    for _ in range(request.param):
        event_list.append(generate_sparse3d_event(meta))

    return event_list


@pytest.fixture(name='sparse3d_seg_event')
def fixture_sparse3d_seg_event(request, N):
    """Generates a single segmentation label larcv.EventSparseTensor3D.

    Returns
    -------
    larcv.EventSparseTensor3D
        Single dummy segmentation label 3D sparse tensor
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Intitialize the metadata
    meta = generate_meta3d(N)

    # Initialize a single larcv sparse tensor
    return generate_sparse3d_event(meta, segmentation=True)


@pytest.fixture(name='cluster2d_event')
def fixture_cluster2d_event(request, N):
    """Generates one larcv.EventClusterPixel2D.

    Returns
    -------
    larcv.EventClusterPixel2D
        Single dummy cluster of 2D sparse tensor
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Intitialize one set of metadata per projection
    meta = []
    for p in range(3):
        meta.append(generate_meta2d(N, p))

    # Initialize the 3 clusters to be filled (one per projection)
    num_clusters = request.param
    clusters = []
    for p in range(3):
        clusters.append(larcv.ClusterPixel2D())
        clusters[p].resize(num_clusters)
        clusters[p].meta(meta[p])

    # Loop over the cluster IDs
    for i in range(num_clusters):
        # Build a 2D sparse tensors corresponding to this cluster
        num_pixels = np.random.randint(low=0, high=20)
        pixel_set = generate_sparse2d_event(meta, num_pixels)

        for p, meta_p in enumerate(meta):
            # Write cluster i to projection p
            v = clusters[p].writeable_voxel_set(i)
            for pix in pixel_set.sparse_tensor_2d(p).as_vector():
                v.insert(pix)

    # Emplace the 3 cluster objects into an event
    event = larcv.EventClusterPixel2D()
    for p in range(3):
        event.set(clusters[p], meta[p])

    return event


@pytest.fixture(name='cluster3d_event')
def fixture_cluster3d_event(request, N):
    """Generates one larcv.EventClusterVoxel3D.

    Returns
    -------
    larcv.EventClusterVoxel3D
        Single dummy cluster of 3D sparse tensor
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Intitialize the metadata
    meta = generate_meta3d(N)

    # Initalize a cluster of sparse 3D tensors
    num_clusters = request.param
    event = larcv.EventClusterVoxel3D()
    event.resize(num_clusters)
    for i in range(num_clusters):
        # Build a 3D sparse tensor corresponding to this cluster
        num_voxels = np.random.randint(low=0, high=20)
        voxel_set = generate_sparse3d_event(meta, num_voxels)

        # Write cluster i
        v = event.writeable_voxel_set(i)
        for vox in voxel_set.as_vector():
            if meta.invalid_voxel_id() != vox.id():
                v.insert(vox)

    # Set the metadata
    event.meta(meta)

    return event


@pytest.fixture(name='particle_event')
def fixture_particle_event(request):
    """Generates one larcv.EventParticle.

    Fills some of the attributes of the dummy particles to be considered valid
    in the parsers used to process them.

    Returns
    -------
    larcv.EventParticle
        List of larcv.Particle objects
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Loop over the requested number of particles
    num_particles = request.param
    particles = larcv.EventParticle()
    for idx in range(num_particles):
        particles.append(generate_particle(idx))

    return particles


@pytest.fixture(name='neutrino_event')
def fixture_neutrino_event(request):
    """Generates one larcv.EventNeutrino.

    Fills some of the attributes of the dummy neutrinos to be considered valid
    in the parsers used to process them.

    Returns
    -------
    larcv.EventNeutrino
        List of larcv.Neutrino objects
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Loop over the requested number of neutrinos
    num_neutrinos = request.param
    neutrinos = larcv.EventNeutrino()
    for idx in range(num_neutrinos):
        neutrinos.append(generate_neutrino(idx))

    return neutrinos


@pytest.fixture(name='flash_event')
def fixture_flash_event(request):
    """Generates one larcv.EventFlash.

    Returns
    -------
    larcv.EventFlash
        Dummy list of larcv flashes
    """
    # Initialize the number of requested flashes in an event
    num_flashes = request.param
    event = larcv.EventFlash()
    for _ in range(num_flashes):
        event.append(larcv.Flash())

    return event


@pytest.fixture(name='flash_event_list')
def fixture_flash_event_list(request):
    """Generates a list of larcv.EventFlash.

    Returns
    -------
    larcv.EventFlash
        Dummy list of larcv flashes
    """
    # Set the random seed so that there are no surprises
    np.random.seed(seed=0)

    # Initialize the number of requested flashes in an event
    num_events = request.param
    events = []
    for e in range(num_events):
        num_flashes = np.random.randint(0, 10)
        events.append(larcv.EventFlash())
        for _ in range(num_flashes):
            events[e].append(larcv.Flash())

    return events


@pytest.fixture(name='crthit_event')
def fixture_crthit_event(request):
    """Generates one larcv.EventCRTHit.

    Returns
    -------
    larcv.EventCRTHit
        Dummy list of larcv CRT hits
    """
    # Initialize the number of requested CRT hits in an event
    num_crthits = request.param
    event = larcv.EventCRTHit()
    for _ in range(num_crthits):
        event.append(larcv.CRTHit())

    return event


@pytest.fixture(name='trigger_event')
def fixture_trigger_event(request):
    """Generates one larcv.EventTrigger.

    Returns
    -------
    larcv.EventTrigger
        Dummy trigger event
    """
    # Initialize a single trigger information object
    try:
        return larcv.EventTrigger()
    except AttributeError:
        return None


def generate_meta2d(shape, projection):
    """Generates random 3D metadata information.

    Parameters
    ----------
    shape : int
        Number of voxels in each dimension
    projection : int
        Projection ID

    Returns
    -------
    larcv.ImageMeta
        2D sparse tensor metadata information
    """
    xmin, ymin = np.random.uniform(-500, 500, size=2)
    meta = larcv.ImageMeta(
            xmin, ymin,
            xmin + shape*0.3, ymin + shape*0.3,
            shape, shape, projection)

    return meta


def generate_meta3d(shape):
    """Generates random 3D metadata information.

    Parameters
    ----------
    shape : int
        Number of voxels in each dimension

    Returns
    -------
    larcv.Voxel3DMeta
        3D sparse tensor metadata information
    """
    meta = larcv.Voxel3DMeta()
    xmin, ymin, zmin = np.random.uniform(-500, 500, size=3)
    meta.set(xmin, ymin, zmin,
             xmin + shape*0.3, ymin + shape*0.3, zmin + shape*0.3,
             shape, shape, shape)

    return meta


def generate_sparse2d_event(meta, num_voxels=10, segmentation=False):
    """Generates a dummy 2D larcv tensor.

    Parameters
    ----------
    meta : larcv.ImageMeta
        2D sparse tensor metadata information
    num_voxels : int, default 10
        Number of voxels in the image

    Returns
    -------
    larcv.EventSparseTensor2D
        Event containing one 2D larcv sparse tensor
    """
    # Loop over the three projections
    event = larcv.EventSparseTensor2D()
    for p, meta_p in enumerate(meta):
        # Fetch the boundaries of the box to generate points in
        range_x = [meta_p.min_x(), meta_p.max_x()]
        range_y = [meta_p.min_y(), meta_p.max_y()]
        boundaries = np.vstack((range_x, range_y))
        ranges = boundaries[:, 1] - boundaries[:, 0]

        # Generate tensor
        # TODO: Must convert point coordinates to pixel indexes manually
        # TODO: because of a LArCV2 bug. Fix this in the future.
        pixels = boundaries[:, 0] + np.random.random((num_voxels, 2))*ranges
        for i, pix in enumerate(pixels): # TMP
            pixels[i] = [meta_p.col(pix[0]), meta_p.row(pix[1])] # TMP
        pixels = pixels.astype(np.int32) # TMP
        features = 10*np.random.rand(num_voxels, 1)
        features = features.flatten().astype(np.float32) # TMP
        #data = np.hstack((pixels, features)).astype(np.float32)

        # Build a SparseTensor2D, append it
        #pixel_set = larcv.as_tensor2d(data, meta_p, -0.01)
        pixel_set = larcv.as_tensor2d(pixels, features, meta_p, -0.01) # TMP
        event.set(pixel_set, meta_p)

    return event


def generate_sparse3d_event(meta, num_voxels=10, segmentation=False,
                            ghost=True):
    """Generates a dummy 3D larcv tensor.

    Parameters
    ----------
    meta : larcv.Voxel3DMeta
        3D sparse tensor metadata information
    num_voxels : int, default 10
        Number of voxels in the image
    segmentation : bool, default False
        If `True`, generate features that are segmentation-like
    ghost : bool, default True
        If `True`, include ghosts in the segmentation labels

    Returns
    -------
    larcv.EventSparseTensor3D
        Event containing one 3D larcv sparse tensor
    """
    # Fetch the boundaries of the box to generate points in
    range_x = [meta.min_x(), meta.max_x()]
    range_y = [meta.min_y(), meta.max_y()]
    range_z = [meta.min_z(), meta.max_z()]
    boundaries = np.vstack((range_x, range_y, range_z))
    ranges = boundaries[:, 1] - boundaries[:, 0]

    # Generate tensor
    voxels = boundaries[:, 0] + np.random.random((num_voxels, 3))*ranges
    if not segmentation:
        features = 10*np.random.rand(num_voxels, 1)
    else:
        features = np.random.randint(0, 5 + int(ghost), size=(num_voxels, 1))
    data = np.hstack((voxels, features)).astype(np.float32)

    # Build a SparseTensor3D, set it
    voxel_set = larcv.as_tensor3d(data, meta, -0.01)
    event = larcv.EventSparseTensor3D()
    event.set(voxel_set, meta)

    return event


def generate_particle(idx):
    """Generates a dummy larcv particle.

    Parameters
    ----------
    idx : int
        Index of the particle in the list

    Returns
    -------
    larcv.Particle
        Single particle truth object
    """
    # Initialize the object
    p = larcv.Particle()

    # Set IDs
    p.id(idx)
    p.track_id(idx+1)
    p.parent_id(max(0, idx-1))
    p.group_id(idx)
    p.interaction_id(0)
    
    # Set energy deposition
    p.num_voxels(10)
    p.energy_deposit(np.random.random()*100)

    # Set shape and particle type attributes
    shape = np.random.randint(low=0, high=5)
    p.shape(shape)
    if shape == 0:
        p.pdg_code(int(np.random.choice([22, 11])))
        p.creation_process(np.random.choice(
            ["primary", "nCapture", "conv"]))

    elif shape == 2:
        p.pdg_code(int(np.random.choice([2112, 211, -211, 13, -13])))

    elif shape == 3:
        p.pdg_code(11)
        p.parent_pdg_code(13)
        p.creation_process(np.random.choice(
            ["muMinusCaptureAtRest", "muPlusCaptureAtRest", "Decay"]))

    elif shape == 4:
        p.pdg_code(11)
        p.parent_pdg_code(13)
        p.creation_process(np.random.choice(
            ["muIoni", "hIoni"]))

    return p


def generate_neutrino(idx):
    """Generates a dummy larcv neutrino.

    Parameters
    ----------
    idx : int
        Index of the neutrino in the list

    Returns
    -------
    larcv.Neutrino
        Single neutrino truth object
    """
    # Initialize the object
    n = larcv.Neutrino()

    # Set IDs
    n.id(idx)

    # Set type characteristics
    n.pdg_code(int(np.random.choice([12, -12, 14, -14, 16, -16])))

    return n
