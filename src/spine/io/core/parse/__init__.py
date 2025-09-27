"""I/O parsers are used to read data products from a LArCV ROOT file.

Parsers are listed under :mod:`io.dataset.schema` in the configuration.
`schema` is a list of named values. Each name is arbitrary and will be
used as a key to access the output of the parser in a dictionary.

List of existing parsers
------------------------

.. csv-table:: Sparse parsers
    :header: Parser name, Description

    ``sparse2d``, Retrieve sparse tensor input from
    larcv::EventSparseTensor2D object
    ``sparse3d``, Retrieve sparse tensor input from
    larcv::EventSparseTensor3D object
    ``sparse3d_ghost``, Takes semantics tensor and turns its labels into
    ghost labels

.. csv-table:: Cluster parsers
    :header: Parser name, Description

    ``cluster2d``, Retrieve list of sparse tensor input from
    larcv::EventClusterPixel2D
    ``cluster3d``, Retrieve list of sparse tensor input from
    larcv::EventClusterVoxel3D

.. csv-table:: Particle parsers
    :header: Parser name, Description

    ``particle``, Retrieve array of larcv::Particle
    ``neutrino``, Retrieve array of larcv::Neutrino
    ``particle_points``, Retrieve array of larcv::Particle ground truth
    points tensor
    ``particle_coords``, Retrieve array of larcv::Particle coordinates
    (start and end) and start time
    ``particle_graph``, Construct edges between particles (i.e. clusters)
    from larcv::EventParticle
    ``single_particle_pid``, Get a single larcv::Particle PDG code
    ``single_particle_energy``, Get a single larcv::Particle initial
    energy

.. csv-table:: Miscellaneous parsers
    :header: Parser name, Description

    ``meta``, Get the meta information to translate into real world coordinates
    ``run_info``, Parse run info (run, subrun, event number)
    ``flash``, Parse optical flashes
    ``crthit``, Parse cosmic ray tagger hits
    ``trigger``, Parse trigger information


What does a typical parser configuration look like?
---------------------------------------------------
If the configuration looks like this, for example:

..  code-block:: yaml

    schema:
      input_data:
        parser: sparse3d
        sevent_list:
          - sparse3d_reco
          - sparse3d_reco_chi2

Where `input_data` is an arbitrary name chosen by the user, which will be the
key to access the output of the parser `sparse3d`. The parser arguments
can be ROOT TTree names that will be fed to the parser or parser arguments. The
arguments must be passed as a dictionary of (argument name, value) pairs. In
this example, the parser will be called with a list of 2 objects: A
:class:`larcv::EventSparseTensor3D` coming from the ROOT TTree `sparse3d_reco`,
and another one coming from the TTree `sparse3d_reco_chi2`.

How do I know what a parser requires?
-------------------------------------
To be completed.

How do I know what my ROOT file contains?
-----------------------------------------
To be completed.
"""

from .cluster import *
from .misc import *
from .particle import *
from .sparse import *
