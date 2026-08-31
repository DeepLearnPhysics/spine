Production Workflows
====================

This guide covers the supported ways to take a maintained SPINE configuration
from input selection to a reproducible output. Run model workflows in the
release-tagged container described in :doc:`installation`; the commands below
assume the repository is mounted at ``/workspace`` and is the working
directory.

Choose A Maintained Configuration
---------------------------------

Start from a maintained file under ``config/`` instead of assembling a model
configuration from scratch. The ``*_train.yaml`` files contain optimization
settings; the ``*_test.yaml`` files are deterministic, short inference
contracts used by CI.

.. list-table:: Maintained model families
   :header-rows: 1
   :widths: 26 38 36

   * - Family
     - Configuration
     - Purpose
   * - Semantic segmentation
     - ``config/uresnet/uresnet_{train,test}.yaml``
     - UResNet semantic classification
   * - Bayesian segmentation
     - ``config/uresnet/bayes/uresnet_bayes_{train,test}.yaml``
     - UResNet with uncertainty estimates
   * - Point and vertex proposals
     - ``config/uresnet/ppn/uresnet_*_{train,test}.yaml``
     - UResNet with PPN and optional vertex heads
   * - Graph-SPICE
     - ``config/graph_spice/graph_spice_{train,test}.yaml``
     - Supervised voxel clustering
   * - GrapPA aggregation
     - ``config/grappa_{shower,track,inter}/grappa_*_{train,test}.yaml``
     - Fragment-to-particle and particle-to-interaction aggregation
   * - Image-level tasks
     - ``config/image/{pid,energy}/image_*_{train,test}.yaml``
     - Particle PID and energy tasks, including ancestor variants
   * - End-to-end reconstruction
     - ``config/full_chain/full_chain_{train,test,regression}.yaml``
     - Full model chain; the regression variant also builds, post-processes,
       and writes reconstructed objects

The full-chain provider ordering and product contracts are explained in
``config/full_chain/README.md`` in the source repository. CI loads every
maintained model configuration in the complete runtime image.

Inspect The Resolved Configuration
----------------------------------

Includes, removals, overrides, environment variables, and ``!download`` tags
can make the effective configuration different from one source file. Inspect
it before a production launch without downloading remote artifacts:

.. code-block:: bash

   spine-config dump config/full_chain/full_chain_test.yaml \
     --output resolved.yaml

Compare two resolved configurations when reviewing a change:

.. code-block:: bash

   spine-config diff candidate.yaml approved.yaml

Keep ``resolved.yaml`` with the run record. At startup, SPINE also writes the
effective configuration and runtime context to the primary process log.

Run Inference
-------------

Use the matching ``*_test.yaml`` file when one exists. Supply input and
checkpoint paths on the command line so the maintained configuration remains
unchanged:

.. code-block:: bash

   spine -c config/uresnet/uresnet_test.yaml \
     --source /data/events.root \
     --weight-path /weights/uresnet.ckpt \
     --world-size 1 \
     --minibatch-size 16 \
     --num-workers 4 \
     --iterations 100

To use a training configuration as the source of truth, ``--inference``
removes training and validation, selects deterministic sequential loading, and
preserves the model and downstream reconstruction stages:

.. code-block:: bash

   spine -c config/full_chain/full_chain_train.yaml \
     --inference \
     --source /data/events.root \
     --weight-path /weights/full-chain.ckpt \
     --world-size 1 \
     --minibatch-size 8

``--batch-size`` is the global batch size, whereas ``--minibatch-size`` is the
per-process size. They are mutually exclusive. ``--iterations`` and
``--epochs`` are likewise mutually exclusive.

Run Full Reconstruction And Write HDF5
--------------------------------------

The checkpoint-pinned regression configuration includes object building,
post-processing, and an HDF5 writer. Override both its source and output for a
real run:

.. code-block:: bash

   spine -c config/full_chain/full_chain_regression.yaml \
     --source /data/events.root \
     --output /results/spine-output.h5 \
     --iterations 100

Output flags only affect a configured ``io.writer``. If a configuration has no
writer, SPINE emits a warning and does not create an output file. The writer's
``keys`` list is the output contract; review it before launching a large job.
See :doc:`data_model` for object semantics and :ref:`inspect-existing-output`
for reading the resulting file.

Post-process And Analyze Existing Output
----------------------------------------

A model is not required when the necessary products already exist in a SPINE
HDF5 file. The following configuration rebuilds reconstructed particle and
interaction views, reruns geometric vertex reconstruction, and writes their
attributes to analysis CSV files:

.. code-block:: yaml

   base:
     iterations: -1
     log_dir: analysis-logs
     overwrite_log: true

   io:
     reader:
       name: hdf5
       file_keys: /data/spine-output.h5
       keep_open: false

   build:
     mode: reco
     units: cm
     fragments: false
     particles: true
     interactions: true

   post:
     vertex:
       run_mode: reco
       use_primaries: true
       touching_threshold: 2.0

   ana:
     overwrite: true
     save:
       obj_type: [particle, interaction]
       run_mode: reco
       match_mode: null

Save it as ``postprocess.yaml`` and run:

.. code-block:: bash

   spine -c postprocess.yaml --iterations 100

The HDF5 reader restores serialized records, ``build`` reconnects their
long-form products, ``post`` mutates or derives event products in memory, and
``ana`` writes study outputs under ``base.log_dir``. Add an ``io.writer`` only
when the post-processed products must also be persisted to a new HDF5 file.
The selected processor and analyzer constructor references remain the
authoritative configuration contracts; see :doc:`api/post` and :doc:`api/ana`.

Train With Validation
---------------------

Training is selected by the presence of the top-level ``train`` block. Use
independent source lists for training and validation and keep resource choices
as launch-time overrides:

.. code-block:: bash

   spine -c config/uresnet/uresnet_train.yaml \
     --source-list train-files.txt \
     --val-source-list validation-files.txt \
     --world-size 4 \
     --minibatch-size 64 \
     --num-workers 8 \
     --epochs 25 \
     --weight-prefix /results/uresnet/snapshot \
     --log-dir /results/uresnet/logs \
     --tensorboard

On-the-fly validation requires a ``validation`` block and a checkpoint cadence
through ``train.save_step`` or ``train.save_epoch``. It inherits the training
schema and batching but replaces data sources and disables random augmentation
and sampling. See :class:`spine.model.validation.ValidationManager` for metric,
early-stopping, and best-checkpoint settings.

Resume Or Initialize From A Checkpoint
--------------------------------------

Use ``--resume`` to require complete training-state restoration. It restores
model, optimizer, scheduler, progress, and available stochastic runtime state;
an incompatible or incomplete checkpoint is rejected. Use ``--no-resume`` to
load parameters while starting a new optimization run:

.. code-block:: bash

   spine -c config/uresnet/uresnet_train.yaml \
     --source-list train-files.txt \
     --weight-path /results/uresnet/snapshot-1000.ckpt \
     --resume

Each new-format checkpoint has an adjacent ``.sha256`` file. Verify provenance
without constructing a model:

.. code-block:: python

   from spine.model import inspect_checkpoint, verify_checkpoint

   checkpoint = "/results/uresnet/snapshot-1000.ckpt"
   if not verify_checkpoint(checkpoint):
       raise RuntimeError("Checkpoint checksum verification failed")

   info = inspect_checkpoint(checkpoint, verify=True)
   print(info["manifest"])
   print(info["config"])
   print(info["datasets"])

Exact stochastic continuation requires the same distributed world size.
Worker prefetch queues and external-library random state are not fully
serializable, so a resumed run can preserve sample order without being
bit-for-bit identical in every environment.

Run From Python
---------------

Use :class:`spine.driver.Driver` when another application owns the event loop.
Reader and inference configurations support random-access processing:

.. code-block:: python

   from spine.config import load_config_file
   from spine.driver import Driver

   cfg = load_config_file("inspect.yaml")
   driver = Driver(cfg)
   event = driver.process(entry=0)
   print(sorted(event))

Use :meth:`spine.driver.Driver.run` for the complete configured loop. Training
configurations should use ``run()`` or the ``spine`` CLI rather than repeated
random-access ``process(entry=...)`` calls.

Operational Handoff
-------------------

Before promoting a run, work through :doc:`operations`. For failures, begin
with :doc:`troubleshooting`; it maps common symptoms to the owning
configuration block and API reference.
