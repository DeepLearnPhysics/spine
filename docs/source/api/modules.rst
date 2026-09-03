Complete Module Index
=====================

This index covers every importable SPINE implementation module not already
listed on a focused API page. Each generated module page links its public
classes and functions to full signature and docstring references. Modules
that require an unavailable optional runtime are documented on explicit
configuration pages instead.

Core modules
------------

.. autosummary::
   :toctree: generated/modules

   spine.banner
   spine.driver
   spine.version

Ana modules
-----------

.. autosummary::
   :toctree: generated/modules

   spine.ana.calib.mcs
   spine.ana.diag.graph
   spine.ana.diag.point
   spine.ana.diag.shower
   spine.ana.diag.track
   spine.ana.metric.cluster
   spine.ana.metric.optical
   spine.ana.metric.point
   spine.ana.metric.segment
   spine.ana.script.save

Bin modules
-----------

.. autosummary::
   :toctree: generated/modules

   spine.bin.cli
   spine.bin.config
   spine.bin.dataset
   spine.bin.info
   spine.bin.report
   spine.bin.source
   spine.bin.weight

Calib modules
-------------

.. autosummary::
   :toctree: generated/modules

   spine.calib.factories
   spine.calib.manager

Cluster modules
---------------

.. autosummary::
   :toctree: generated/modules

   spine.cluster.quality

Config modules
--------------

.. autosummary::
   :toctree: generated/modules

   spine.config.api
   spine.config.download
   spine.config.errors
   spine.config.factory
   spine.config.inference
   spine.config.load
   spine.config.loader
   spine.config.meta
   spine.config.normalize
   spine.config.operations

Constants modules
-----------------

.. autosummary::
   :toctree: generated/modules

   spine.constants.columns
   spine.constants.enums
   spine.constants.factory
   spine.constants.labels
   spine.constants.physics
   spine.constants.sentinels

Construct modules
-----------------

.. autosummary::
   :toctree: generated/modules

   spine.construct.base
   spine.construct.fragment
   spine.construct.interaction
   spine.construct.manager
   spine.construct.particle
   spine.construct.utils

Data modules
------------

.. autosummary::
   :toctree: generated/modules

   spine.data.base
   spine.data.decorator
   spine.data.field
   spine.data.larcv.crt
   spine.data.larcv.meta
   spine.data.larcv.neutrino
   spine.data.larcv.optical
   spine.data.larcv.particle
   spine.data.larcv.run_info
   spine.data.larcv.trigger
   spine.data.out.base
   spine.data.out.fragment
   spine.data.out.interaction
   spine.data.out.optical
   spine.data.out.particle
   spine.data.product.base
   spine.data.product.batch.base
   spine.data.product.batch.cluster
   spine.data.product.batch.edge_index
   spine.data.product.batch.index
   spine.data.product.batch.object
   spine.data.product.batch.tensor
   spine.data.product.cluster
   spine.data.product.edge_index
   spine.data.product.index
   spine.data.product.object
   spine.data.product.tensor

Geo modules
-----------

.. autosummary::
   :toctree: generated/modules

   spine.geo.base
   spine.geo.detector.base
   spine.geo.detector.crt
   spine.geo.detector.optical
   spine.geo.detector.tpc
   spine.geo.factories
   spine.geo.manager
   spine.geo.utils

Io modules
----------

.. autosummary::
   :toctree: generated/modules

   spine.io.augment.base
   spine.io.augment.crop
   spine.io.augment.flip
   spine.io.augment.jitter
   spine.io.augment.manager
   spine.io.augment.mask
   spine.io.augment.rotate
   spine.io.augment.translate
   spine.io.dataset.base
   spine.io.dataset.hdf5
   spine.io.dataset.joint
   spine.io.dataset.larcv
   spine.io.dataset.mixed
   spine.io.manager
   spine.io.parse.hdf5.utils
   spine.io.parse.larcv.utils.particle
   spine.io.parse.larcv.utils.point
   spine.io.read.base
   spine.io.read.hdf5.common
   spine.io.read.hdf5.product
   spine.io.read.hdf5.reader
   spine.io.read.hdf5.region
   spine.io.read.larcv
   spine.io.read.stage_hdf5
   spine.io.transform.hdf5
   spine.io.write.hdf5.common
   spine.io.write.hdf5.product
   spine.io.write.hdf5.region
   spine.io.write.hdf5.schema
   spine.io.write.hdf5.writer
   spine.io.write.stage_hdf5
   spine.io.write.stage_hdf5.file
   spine.io.write.stage_hdf5.sidecar
   spine.io.write.stage_hdf5.state
   spine.io.write.stage_hdf5.writer

Logging modules
---------------

.. autosummary::
   :toctree: generated/modules

   spine.logging.csv
   spine.logging.manager

Math modules
------------

.. autosummary::
   :toctree: generated/modules

   spine.math.metrics.base
   spine.math.metrics.cluster

Model modules
-------------

.. autosummary::
   :toctree: generated/modules

   spine.model.cnn.act_norm
   spine.model.cnn.blocks
   spine.model.cnn.configuration
   spine.model.cnn.encoder
   spine.model.cnn.factories
   spine.model.cnn.fpn
   spine.model.cnn.mcdropout
   spine.model.cnn.mobilenet
   spine.model.cnn.nonlinearities
   spine.model.cnn.normalizations
   spine.model.cnn.senet
   spine.model.cnn.uresnet_layers
   spine.model.cnn.uresnext
   spine.model.common.act_norm
   spine.model.common.dbscan
   spine.model.common.evidential
   spine.model.common.factories
   spine.model.common.final
   spine.model.common.losses
   spine.model.common.metric
   spine.model.common.mlp
   spine.model.common.point_break
   spine.model.common.quality
   spine.model.common.weighting
   spine.model.export
   spine.model.full_chain.config
   spine.model.full_chain.label
   spine.model.full_chain.model
   spine.model.full_chain.ops
   spine.model.full_chain.point
   spine.model.full_chain.providers.aggregation
   spine.model.full_chain.providers.calibration
   spine.model.full_chain.providers.deghost
   spine.model.full_chain.providers.fragmentation
   spine.model.full_chain.providers.image
   spine.model.full_chain.providers.transform.track_breaking
   spine.model.full_chain.providers.vertexing
   spine.model.full_chain.registry
   spine.model.full_chain.stage
   spine.model.full_chain.state
   spine.model.graph_spice.connected
   spine.model.graph_spice.constructor
   spine.model.graph_spice.embedder
   spine.model.graph_spice.factories
   spine.model.graph_spice.kernel
   spine.model.graph_spice.loss
   spine.model.graph_spice.model
   spine.model.graph_spice.orphan
   spine.model.grappa.encode.cnn
   spine.model.grappa.encode.empty
   spine.model.grappa.encode.geometric
   spine.model.grappa.encode.mixed
   spine.model.grappa.encode.voxel
   spine.model.grappa.evaluation
   spine.model.grappa.factories
   spine.model.grappa.graph.base
   spine.model.grappa.graph.bipartite
   spine.model.grappa.graph.complete
   spine.model.grappa.graph.delaunay
   spine.model.grappa.graph.knn
   spine.model.grappa.graph.loop
   spine.model.grappa.graph.mst
   spine.model.grappa.loss.edge_channel
   spine.model.grappa.loss.node_class
   spine.model.grappa.loss.node_orient
   spine.model.grappa.loss.node_reg
   spine.model.grappa.loss.node_shower_primary
   spine.model.grappa.loss.node_vertex
   spine.model.grappa.loss.target
   spine.model.grappa.message_passing.factories
   spine.model.grappa.message_passing.layers.agnnconv
   spine.model.grappa.message_passing.layers.econv
   spine.model.grappa.message_passing.layers.gatconv
   spine.model.grappa.message_passing.layers.mlp
   spine.model.grappa.message_passing.layers.nnconv
   spine.model.grappa.message_passing.meta
   spine.model.grappa.model
   spine.model.grappa.vertex
   spine.model.image.encoder
   spine.model.image.loss
   spine.model.image.model
   spine.model.image.object
   spine.model.optim.factory
   spine.model.pointcloud.pointnet
   spine.model.registry
   spine.model.sparse.backend
   spine.model.sparse.backends.minkowski
   spine.model.sparse.functional
   spine.model.sparse.modules
   spine.model.sparse.tensor
   spine.model.spice.cluster
   spine.model.spice.embedder
   spine.model.spice.loss
   spine.model.spice.model
   spine.model.uresnet.model

Post modules
------------

.. autosummary::
   :toctree: generated/modules

   spine.post.crt.crt_matching
   spine.post.crt.match
   spine.post.optical.barycenter
   spine.post.optical.flash_matching
   spine.post.optical.likelihood
   spine.post.optical.opt0finder
   spine.post.reco.calo
   spine.post.reco.calorimetric_direction
   spine.post.reco.cathode_cross
   spine.post.reco.cluster
   spine.post.reco.direction
   spine.post.reco.geometry
   spine.post.reco.kinematics
   spine.post.reco.mcs
   spine.post.reco.pid
   spine.post.reco.points
   spine.post.reco.ppn
   spine.post.reco.shower
   spine.post.reco.source
   spine.post.reco.topology
   spine.post.reco.tracking
   spine.post.reco.vertex
   spine.post.trigger.trigger
   spine.post.truth.label
   spine.post.truth.match

Utils modules
-------------

.. autosummary::
   :toctree: generated/modules

   spine.utils.torch.devices
   spine.utils.torch.runtime

Vis modules
-----------

.. autosummary::
   :toctree: generated/modules

   spine.vis.drawer.geo
   spine.vis.drawer.lite
   spine.vis.drawer.network
   spine.vis.drawer.out.colors
   spine.vis.drawer.out.drawer
   spine.vis.drawer.out.formatting
   spine.vis.drawer.out.layers
   spine.vis.drawer.out.scene
   spine.vis.drawer.out.traces
   spine.vis.drawer.particle
   spine.vis.drawer.train.drawer
   spine.vis.drawer.train.io
   spine.vis.drawer.train.style
   spine.vis.layout.colors
   spine.vis.layout.matplotlib
   spine.vis.layout.plotly
   spine.vis.metric.confmat
   spine.vis.metric.distribution
   spine.vis.metric.heatmap
   spine.vis.metric.plot
   spine.vis.metric.report
   spine.vis.metric.report.base
   spine.vis.metric.report.classification
   spine.vis.metric.report.cluster
   spine.vis.metric.report.manager
   spine.vis.metric.report.node
   spine.vis.metric.report.point
   spine.vis.metric.report.segment
   spine.vis.scene.adapter
   spine.vis.scene.backend
   spine.vis.scene.model
   spine.vis.scene.plotly
   spine.vis.trace.arrow
   spine.vis.trace.box
   spine.vis.trace.cluster
   spine.vis.trace.cone
   spine.vis.trace.cylinder
   spine.vis.trace.ellipsoid
   spine.vis.trace.hull
   spine.vis.trace.point
   spine.vis.trace.utils


.. generated-count: 266
