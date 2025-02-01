"""Module in charge of loading SPINE configuration files."""

import os

import yaml


class ConfigLoader(yaml.SafeLoader):
    """Configuration loader class.

    This class implements a more complex YAML loader than the standard loader in
    order to support more advanced functions such as:
    - Include YAML configuration files into another YAML configuration file;
    - Edit an included YAML dictionary with one liners (to modify single
      configuration parameters without replicating a configuration block).
    """

    def __init__(self, stream):
        """Initialize the laoder.

        Parameters
        ----------
        stream : _io.TextIOWrapper
            Output of python's `open` function on a yaml file
        """
        # Fetch the parent directory where the configuration file lives
        self._root = os.path.split(stream.name)[0]

        # Initialize the base loader
        super().__init__(stream)

    def include(self, node):
        """Load and include a YAML file that is requested in the base config.

        Parameters
        ----------
        node : str
            Name of the YAML block to load
        """
        # Look for the file in the same directory as the main config file
        filename = os.path.join(self._root, self.construct_scalar(node))

        # Load the file within the base configruation
        with open(filename, 'r') as f:
            return yaml.load(f, Loader=ConfigLoader)


# Add the include constructor
ConfigLoader.add_constructor('!include', ConfigLoader.include)


def load_config(cfg_path):
    """Load a configuration file to a dictionary.

    Parameters
    ----------
    cfg_path : str
        Path to the configuration file
    """
    with open(cfg_path, 'r', encoding='utf-8') as cfg_stream:
        cfg = yaml.load(cfg_stream, Loader=ConfigLoader)

    return cfg
