Troubleshooting
===============

Start every investigation by recording the SPINE version, container tag,
``spine --info`` output, full command, resolved configuration, and complete
traceback. The startup log prints the effective runtime and configuration; use
that version rather than assuming the source YAML was applied unchanged.

Installation And Imports
------------------------

``ImportError`` for PyTorch, MinkowskiEngine, torch-geometric, or LArCV
   Full model execution needs a mutually compatible compiled stack. Run the
   release-tagged SPINE container. A core ``pip install spine`` intentionally
   does not install the complete ML runtime.

The container cannot see input, checkpoint, or output paths
   Container paths are not host paths. Bind the containing directory with
   Docker ``-v`` or the corresponding Apptainer bind option and use the path as
   seen inside the container.

Apple Silicon container or notebook problems
   Run the published image as ``linux/amd64``. For Jupyter, avoid Docker
   Desktop's Apple Virtualization Framework with Rosetta enabled; see
   :doc:`quickstart` for the verified combinations.

Configuration
-------------

``Configuration file must contain an io block``
   Every driver run needs ``io`` with a reader and/or loader. A model requires
   ``io.loader``; random-access inspection of existing HDF5 output normally
   uses ``io.reader``.

``Unrecognized keys in base configuration``
   A key is in the wrong block or is obsolete. Compare it with
   :meth:`spine.driver.Driver.initialize_base` and inspect the resolved YAML.
   Training settings belong in top-level ``train``, not ``base``.

An include or relative file cannot be found
   Includes are resolved relative to the including file, then through
   ``SPINE_CONFIG_PATH``. The CLI also sets ``base.parent_path`` to the selected
   configuration directory for downstream relative paths. Use
   ``spine-config dump`` from the production environment to reproduce lookup.

A command-line override has no effect
   Confirm the dot path against the resolved configuration. ``--output`` only
   modifies an existing ``io.writer``; it does not create one. Source overrides
   target the configured reader or loader dataset. Composite datasets may
   require ``TARGET=PATH`` syntax.

Reader, Loader, And Products
----------------------------

``The model can only be used in conjunction with a loader``
   Models consume collated batches. Configure ``io.loader`` rather than
   ``io.reader`` for model training or inference.

A post-processor reports a missing key
   Post-processors declare required products and execute after model forwarding,
   unwrapping, and object building. Confirm the selected model/writer keys,
   ``base.unwrap``, and ``build`` stages produce the requested object family.
   Follow the processor class from :doc:`api/post` for its inputs.

Objects exist but have no points or depositions after HDF5 reading
   Serialized objects store compact references. Enable the corresponding
   ``build`` family so the builder reconnects long-form point, deposition, and
   index products. See :ref:`inspect-existing-output`.

No output file is created
   Ensure ``io.writer`` exists, its destination is writable inside the
   container, and its ``keys`` list is nonempty. Output CLI flags are ignored
   with a warning when no writer is configured.

Training And Checkpoints
------------------------

Validation initialization fails
   Validation is training-only, requires ``io.loader``, and runs at checkpoint
   boundaries. Configure ``train.save_step`` or ``train.save_epoch``. A
   metric-aware checkpoint scheduler also requires the named validation metric.

Strict resume rejects a checkpoint
   ``--resume`` requires compatible optimizer and progress state. Verify the
   adjacent checksum and inspect the checkpoint manifest. Use ``--no-resume``
   only when intentionally starting a new optimization history from weights.

Resume is not bit-for-bit identical
   Exact continuation requires the same distributed world size. Worker queues,
   stochastic worker transforms, Numba state, and external-library RNG state
   cannot always be restored exactly. See :doc:`workflows` and
   :doc:`operations` for the reproducibility boundary.

CUDA out of memory
   Reduce ``--minibatch-size`` first. Remember that it is per process/GPU;
   ``--batch-size`` is global. Also reduce worker prefetching if host memory is
   exhausted. Record changed resource settings with the run artifacts.

Distributed startup hangs or ranks disagree
   Check that every process sees consistent ``RANK``, ``WORLD_SIZE``, master
   address/port, input files, and checkpoint paths. An explicit
   ``--world-size`` must match the launcher environment. Test one node and a
   short input subset before scaling out.

Getting Help
------------

If the issue remains, open a GitHub issue with a minimal configuration and
input reproducer when possible. Include the diagnostic context listed at the
top of this page; remove credentials, private storage paths, and event data
that cannot be shared.
