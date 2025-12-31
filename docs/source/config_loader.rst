Advanced YAML Config Loader
============================

The enhanced ``spine.utils.config`` module provides three powerful features for managing YAML configuration files using standard YAML syntax.

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
     name: uresnet

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
     name: lrelu
     negative_slope: 0.1

**main_config.yaml:**

.. code-block:: yaml

   model:
     name: full_chain
     modules:
       uresnet: !include network_config.yaml
       ppn: !include ppn_config.yaml

3. Dot-Notation Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~

Override specific nested parameters without duplicating entire blocks using dot-separated keys:

.. code-block:: yaml

   include: icarus_base.yaml

   # Override specific parameters using dot notation
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
         name: larcv
         file_keys: null

**uresnet_config.yaml:**

.. code-block:: yaml

   num_input: 2
   num_classes: 5
   filters: 32
   depth: 5
   activation:
     name: lrelu
     negative_slope: 0.1

**icarus_full_chain.yaml:**

.. code-block:: yaml

   include: icarus_base.yaml

   # Include network configuration inline
   model:
     name: full_chain
     modules:
       uresnet: !include uresnet_config.yaml

   # Override specific parameters
   io.loader.batch_size: 8
   io.loader.dataset.file_keys: [data, seg_label, clust_label]
   base.iterations: 1000

Usage in Python
---------------

.. code-block:: python

   from spine.utils.config import load_config

   # Load your config file
   cfg = load_config('icarus_full_chain.yaml')

   # Access configuration values
   print(cfg['base']['iterations'])  # 1000
   print(cfg['io']['loader']['batch_size'])  # 8
   print(cfg['model']['modules']['uresnet']['depth'])  # 5

Benefits
--------

1. **DRY Principle**: Define common settings once, reuse everywhere
2. **Easy Experimentation**: Create new configs by including base configs and overriding only what you need
3. **Modular Configuration**: Split large configs into logical, reusable components
4. **Quick Overrides**: Test different parameters without editing base files
5. **Nested Includes**: Included files can themselves include other files

Notes
-----

- All file paths in ``include`` statements are relative to the directory containing the config file
- Later includes and overrides take precedence over earlier ones
- Dot-notation overrides happen after all includes are processed
- The ``!include`` directive can be used at any level of nesting
- Both ``.yaml`` and ``.yml`` extensions are supported
- The ``include:`` key uses standard YAML syntax (similar to GitLab CI, Docker Compose)
- You can use either ``include: file.yaml`` or ``include: [file1.yaml, file2.yaml]`` syntax
