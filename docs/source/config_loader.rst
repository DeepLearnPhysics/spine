Advanced YAML Config Loader
============================

The enhanced ``spine.config`` module provides four powerful features for managing YAML configuration files using standard YAML syntax.

Features
--------

1. Top-Level File Includes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include entire configuration files using the ``include:`` key (similar to GitLab CI):

**base_config.yaml:**

.. code-block:: yaml

   base:
     world_size: 1
     iterations: -1
     seed: 0

   geo:
     detector: icarus
     tag: icarus_v4

**my_config.yaml:**

.. code-block:: yaml

   include: base_config.yaml

   # You can still add or override settings
   model:
     provider: uresnet

**Multiple includes are supported:**

.. code-block:: yaml

   include:
     - base_config.yaml
     - network_defaults.yaml
     - io_settings.yaml

   # Your custom settings here

Or as a list (equivalent):

.. code-block:: yaml

   include: [base_config.yaml, network_defaults.yaml, io_settings.yaml]

   # Your custom settings here

2. Inline File Includes
~~~~~~~~~~~~~~~~~~~~~~~~

Include files within specific configuration blocks using ``!include``:

**network_config.yaml:**

.. code-block:: yaml

   depth: 5
   filters: 32
   num_classes: 5
   activation:
     provider: lrelu
     negative_slope: 0.1

**main_config.yaml:**

.. code-block:: yaml

   model:
     provider: full_chain
     modules:
       uresnet: !include network_config.yaml
       ppn: !include ppn_config.yaml

3. Dot-Notation Override
~~~~~~~~~~~~~~~~~~~~~~~~

Override specific nested parameters without duplicating entire blocks using the ``override:`` block with dot-separated keys:

.. code-block:: yaml

   include: icarus_base.yaml

   # Override specific parameters using dot notation
   override:
     io.loader.batch_size: 8
     io.loader.dataset.file_keys: [data, seg_label, clust_label]
     base.iterations: 1000
     model.modules.uresnet.depth: 6

This is equivalent to:

.. code-block:: yaml

   include: icarus_base.yaml

   io:
     loader:
       batch_size: 8
       dataset:
         file_keys: [data, seg_label, clust_label]

   base:
     iterations: 1000

   model:
     modules:
       uresnet:
         depth: 6

4. Removing Keys
~~~~~~~~~~~~~~~~

Remove keys from included files using either the ``remove:`` directive or by setting values to ``null`` in ``override:``:

**Using the remove directive:**

.. code-block:: yaml

   include: base_config.yaml

   # Remove specific keys
   remove: io.loader.shuffle

   # Or remove multiple keys
   remove:
     - io.loader.shuffle
     - model.dropout_rate
     - base.debug_mode

**Assigning a null value:**

.. code-block:: yaml

   include: base_config.yaml

   # Keep the key with a null value
   override:
     io.loader.shuffle: null
     model.dropout_rate: null

Only ``remove:`` deletes keys. An explicit null remains in the resolved
configuration and is passed to the consuming component.

Complete Example
----------------

**icarus_base.yaml:**

.. code-block:: yaml

   base:
     world_size: 1
     iterations: -1
     seed: 0
     dtype: float32

   geo:
     detector: icarus
     tag: icarus_v4

   io:
     loader:
       batch_size: 4
       shuffle: false
       num_workers: 8
       dataset:
         provider: larcv
         file_keys: null

**uresnet_config.yaml:**

.. code-block:: yaml

   num_input: 2
   num_classes: 5
   filters: 32
   depth: 5
   activation:
     provider: lrelu
     negative_slope: 0.1

**icarus_full_chain.yaml:**

.. code-block:: yaml

   include: icarus_base.yaml

   # Include network configuration inline
   model:
     provider: full_chain
     modules:
       uresnet: !include uresnet_config.yaml

   # Override specific parameters
   override:
     io.loader.batch_size: 8
     io.loader.dataset.file_keys: [data, seg_label, clust_label]
     base.iterations: 1000

Usage in Python
---------------

.. code-block:: python

   from spine.config import load_config_file

   # Load your config file
   cfg = load_config_file("icarus_full_chain.yaml")

   # Access configuration values
   print(cfg['base']['iterations'])  # 1000
   print(cfg['io']['loader']['batch_size'])  # 8
   print(cfg['model']['modules']['uresnet']['depth'])  # 5

Command-Line Overrides
----------------------

When using the SPINE CLI, you can override any configuration parameter using the ``--set`` flag with dot notation:

.. code-block:: bash

   # Override a single parameter
   spine -c config.yaml --set io.loader.batch_size=8

   # Override multiple parameters
   spine -c config.yaml \
     --set base.iterations=1000 \
     --set io.loader.batch_size=16 \
     --set io.loader.dataset.file_keys=[file1.root,file2.root]

   # Mix with other CLI options
   spine -c config.yaml \
     --source /data/input.root \
     --output /data/output.h5 \
     --set model.weight_path=/weights/model.ckpt

The ``--set`` flag accepts any valid YAML value:

- **Strings**: ``--set model.name=my_model``
- **Numbers**: ``--set base.iterations=1000`` or ``--set base.learning_rate=0.001``
- **Booleans**: ``--set io.loader.shuffle=true``
- **Lists**: ``--set io.loader.dataset.file_keys=[file1.root,file2.root]``
- **Nested paths**: ``--set io.loader.dataset.schema.data.num_features=8``

This is particularly useful for:

- **Hyperparameter sweeps**: Quickly test different values without editing config files
- **Production runs**: Override paths and settings for different environments
- **Debugging**: Enable/disable features or adjust batch sizes on the fly

Benefits
--------

1. **DRY Principle**: Define common settings once, reuse everywhere
2. **Easy Experimentation**: Create new configs by including base configs and overriding only what you need
3. **Modular Configuration**: Split large configs into logical, reusable components
4. **Quick Overrides**: Test different parameters without editing base files
5. **Nested Includes**: Included files can themselves include other files
6. **Key Removal**: Delete unwanted keys from included files without editing the original

Notes
-----

- All file paths in ``include`` statements are relative to the directory containing the config file
- Later includes and override values take precedence over earlier ones
- Dot-notation override and removals happen after all includes are processed
- The ``!include`` directive can be used at any level of nesting
- Both ``.yaml`` and ``.yml`` extensions are supported
- The ``include:`` key uses standard YAML syntax (similar to GitLab CI, Docker Compose)
- You can use either ``include: file.yaml`` or ``include: [file1.yaml, file2.yaml]`` syntax
- Keys set to ``null`` retain a null value; use ``remove:`` to delete them
- The ``remove:`` directive accepts single keys or lists of keys to delete

Component configuration conventions
-----------------------------------

Use ``provider`` to select a component implementation, including readers,
writers, datasets, samplers, parsers, model utilities, and the top-level model.
Image encoders, image task losses, and GraphSPICE backbones use the same
selector and parameter rules. Image task ``weight`` is orchestrator metadata:
it stays alongside ``provider``, outside a nested ``config``.
For example:

.. code-block:: yaml

   io:
     reader:
       provider: hdf5
       file_keys: input.h5
     writer:
       provider: hdf5
       config:
         file_name: output.h5

Factory constructor parameters may be inline or nested under ``config``, but the two forms cannot
be mixed. The old ``args`` and ``kwargs`` wrappers are no longer accepted:
convert positional arguments to named parameters and move keyword arguments
into ``config``. Configured parameters cannot collide with runtime-injected
arguments.

Legacy ``name`` implementation selectors and context-specific ``parser`` and
``collate_fn`` selectors remain supported with ``DeprecationWarning`` warnings.
Replace them with ``provider``. Multiple implementation selectors are rejected,
even when their values agree. Generic factories also retain scalar provider
shorthand.

Managers accepting ordered modules use a ``stages`` list:

.. code-block:: yaml

   stages:
     - name: first
       provider: implementation
       config:
         option: value
     - name: implementation

Here ``name`` is a required, unique instance identity. It is not deprecated.
An omitted ``provider`` defaults to ``name``. List order is execution order;
stage-level ``priority`` is rejected. Legacy module mappings retain their
existing priority behavior and default the provider to the mapping key.
A manager cannot mix ``stages`` with legacy module entries.

I/O schemas remain mappings from output-product names to parser descriptors.
Those keys do not imply a provider:

.. code-block:: yaml

   schema:
     data:
       provider: sparse3d
       config:
         sparse_event: sparse3d_data

Metadata names and geometry detector names are not implementation selectors.
These conventions concern component construction, not arbitrary fields called
``name``.


Override and removal ordering
-----------------------------

All loading entry points, including ordinary ``include:`` and inline
``!include``, apply ``override:`` entries in declaration order followed by
``remove:`` paths. The placement of these directive blocks in YAML does not
change their execution order. Includes are processed in listed order, followed
by the including file's own content and directives. A later include may restore
a key removed by an earlier modifier.

This is a behavior change for included files that previously removed keys
before applying overrides. Delete redundant removals of keys being replaced,
or of children already omitted by a replacement mapping. Keep removals of
unrelated paths. Missing-target and deferred-override policies are described below.


Deferred operations and optional targets
----------------------------------------

Assignments and collection operations in reusable fragments may wait for an
enclosing configuration to supply their targets. Pending operations retain
their order, repeated paths, source files, strictness, and append settings.
They are retried after subsequent include content is merged. A later operation
on the same path or an overlapping parent/child path cannot overtake them.
Named-list edits remain strict: missing names, missing lists, and edits blocked
by unresolved overlapping operations are errors.

Declare deliberately optional operations in the file that contains them:

.. code-block:: yaml

   __meta__:
     kind: fragment
     optional_paths:
       - post.time_containment.run_mode

   override:
     post.time_containment.run_mode: reco

Each optional path must exactly match a local override or removal target,
without an operator suffix. Optionality is not inherited, does not hide type
errors, and cannot apply to named-list edits. Optional deferred assignments
still apply when an enclosing configuration supplies their parent.

At final resolution, an unresolved ordinary assignment currently skips with a
``FutureWarning`` for compatibility. Silent skipping is deprecated; a future
release will apply the declaring file's ``strict`` setting. Missing collection
targets and explicit removals already honor the declaring file's strictness:
``error`` raises and ``warn`` warns and skips. An enclosing file cannot weaken
that policy. Incomplete modifiers should therefore be loaded with their bases.

Removing an absent value or dictionary member from an existing collection is
idempotent. Missing containers follow the strict/optional policy instead.
Ordinary assignments may create a new leaf key when its parent exists.

When migrating, mark intentional optionality, correct accidental missing paths,
and check configurations that depended on deferred operations running out of
order or repeated appends being collapsed.
