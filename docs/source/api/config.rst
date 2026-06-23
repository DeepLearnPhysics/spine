Configuration Module
====================

The ``spine.config`` package provides SPINE's configuration loading system. It is the canonical entry point for reading YAML configuration files with SPINE-specific features such as hierarchical includes, overrides, collection operations, and optional downloads.

.. automodule:: spine.config
   :no-members:

Core Interfaces
---------------

Most users interact with the configuration package through the file-based loader and the exception types it raises when configuration resolution fails.

.. autosummary::
   :toctree: generated

   load.load_config
   load.load_config_file
   download.download_from_url
   download.get_cache_dir
   errors.ConfigError
   errors.ConfigIncludeError
   errors.ConfigCycleError
   errors.ConfigPathError
   errors.ConfigTypeError
   errors.ConfigOperationError
   errors.ConfigValidationError
