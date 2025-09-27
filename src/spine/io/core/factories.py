"""Functions that instantiate PyTorch-independent IO tools from configuration blocks."""

from warnings import warn

from spine.utils.factory import instantiate, module_dict

from . import read, write

READER_DICT = module_dict(read)
WRITER_DICT = module_dict(write)

__all__ = ["reader_factory", "writer_factory"]


def reader_factory(reader_cfg):
    """Instantiates reader based on type specified in configuration under
    `io.reader.name`. The name must match the name of a class under
    `spine.io.readers`.

    Parameters
    ----------
    reader_cfg : dict
        Writer configuration dictionary

    Returns
    -------
    object
        Writer object

    Note
    ----
    Currently the choice is limited to `HDF5Writer` only.
    """
    # Initialize reader
    return instantiate(READER_DICT, reader_cfg)


def writer_factory(writer_cfg, prefix=None, split=False):
    """Instantiates writer based on type specified in configuration under
    `io.writer.name`. The name must match the name of a class under
    `spine.io.writers`.

    Parameters
    ----------
    writer_cfg : dict
        Writer configuration dictionary
    prefix : str, optional
        Input file prefix to use as an output name
    split : bool, default False
        Split the output into one file per input file

    Returns
    -------
    object
        Writer object

    Note
    ----
    Currently the choice is limited to `HDF5Writer` only.
    """
    # Initialize writer
    return instantiate(WRITER_DICT, writer_cfg, prefix=prefix, split=split)
