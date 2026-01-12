"""Contains a reader class dedicated to loading data from LArCV files."""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from spine.utils.conditional import ROOT
from spine.utils.docstring import inherit_docstring
from spine.utils.logger import logger

from .base import ReaderBase

__all__ = ["LArCVReader"]


@inherit_docstring(ReaderBase)
class LArCVReader(ReaderBase):
    """Class which reads information stored in LArCV files.

    This class inherits from the :class:`ReaderBase` class. It provides
    methods to load LArCV2 files and extract their data products:
      - EventSparseTensor: voxel IDs and their values
      - EventClusterSparseTensor: list of sparse tensors
      - EventParticle: list of Geant4 particle information
      - EventNeutrino: list of generstor neutrino information
      - EventFlash: list of optical flashes information
      - EventCRTHit: list of cosmic-ray tagger hits
      - EventTrigger: trigger information

    It builds a TChain from the list of files provided with the appropriate
    trees corresponding to each of the requested data products.
    """

    name = "larcv"

    def __init__(
        self,
        file_keys: Optional[Union[str, List[str]]] = None,
        file_list: Optional[str] = None,
        tree_keys: Optional[List[str]] = None,
        limit_num_files: Optional[int] = None,
        max_print_files: int = 10,
        n_entry: Optional[int] = None,
        n_skip: Optional[int] = None,
        entry_list: Optional[List[int]] = None,
        skip_entry_list: Optional[List[int]] = None,
        run_event_list: Optional[List[List[int]]] = None,
        skip_run_event_list: Optional[List[List[int]]] = None,
        create_run_map: bool = False,
        run_info_key: Optional[str] = None,
        allow_missing: bool = False,
    ):
        """Initialize the LArCV file reader.

        Parameters
        ----------
        file_keys : Union[str, List[str]], optional
            Path or list of paths to the LArCV files to be read
        file_list : str, optional
            Path to a text file containing a list of file paths to be read
        tree_keys : List[str]
            List of data keys to load from the LArCV files
        limit_num_files : Optional[int], optional
            Integer limiting number of files to be taken per data directory
        max_print_files : int, default 10
            Maximum number of loaded file names to be printed
        n_entry : Optional[int], optional
            Maximum number of entries to load
        n_skip : Optional[int], optional
            Number of entries to skip at the beginning
        entry_list : Optional[List[int]]
            List of integer entry IDs to add to the index
        skip_entry_list : Optional[List[int]]
            List of integer entry IDs to skip from the index
        run_event_list: Optional[List[List[int]]], optional
            List of (run, subrun, event) triplets to add to the index
        skip_run_event_list: Optional[List[List[int]]], optional
            List of (run, subrun, event) triplets to skip from the index
        create_run_map : bool, default False
            Initialize a map between (run, subrun, event) triplets and entries.
            For large files, this can be quite expensive (must load every entry).
        run_info_key : Optional[str], optional
            Key of the tree in the file to get the run information from
        allow_missing : bool, default False
            If `True`, allows missing entries in the entry or event list
        """
        # Process the file_paths
        self.process_file_paths(file_keys, file_list, limit_num_files, max_print_files)
        assert (
            tree_keys is not None and len(tree_keys) > 0
        ), "No input `tree_keys` provided, abort."

        # If an entry list is requested based on run/subrun/event ID, create map
        if run_event_list is not None or skip_run_event_list is not None:
            create_run_map = True

        # Prepare TTrees and load files
        self.num_entries = -1
        self.trees = {}
        self.trees_ready = False
        self.file_offsets = np.empty(len(self.file_paths), dtype=np.int64)
        file_counts = []
        for key in tree_keys:
            # Check data TTree exists, and entries are identical across all
            # trees. Do not register these TTrees in yet in order to support
            # > 1 workers by the DataLoader object downstrean.
            logger.info("Loading tree %s", key)
            chain = ROOT.TChain(f"{key}_tree")  # pylint: disable=E1101
            for i, f in enumerate(self.file_paths):
                self.file_offsets[i] = chain.GetEntries()
                chain.AddFile(f)
                if key == tree_keys[0]:
                    count = chain.GetEntries() - self.file_offsets[i]
                    file_counts.append(count)

            if self.num_entries >= 0:
                assert self.num_entries == chain.GetEntries(), (
                    f"Mismatch between the number of entries for {key} "
                    f"({chain.GetEntries()}) and the number of entries "
                    f"in other data products ({self.num_entries})."
                )
            else:
                self.num_entries = chain.GetEntries()

            self.trees[key] = None
        logger.info("")

        # Dump the number of entries to load
        logger.info("Total number of entries in the file(s): %d\n", self.num_entries)

        # Build a file index
        self.file_index = np.repeat(np.arange(len(self.file_paths)), file_counts)

        # If requested, must extract the run information for each entry
        if create_run_map:
            # Initialize the TChain object
            assert run_info_key is not None and run_info_key in tree_keys, (
                "Must provide the `run_info_key` if a run map is needed. "
                "The key must appear in the list of `tree_keys`"
            )
            chain = ROOT.TChain(f"{run_info_key}_tree")  # pylint: disable=E1101
            for f in self.file_paths:
                chain.AddFile(f)

            # Loop over entries
            self.run_info = []
            for i in range(self.num_entries):
                chain.GetEntry(i)
                info = getattr(chain, f"{run_info_key}_branch")
                self.run_info.append((info.run(), info.subrun(), info.event()))

        # Process the run information
        self.process_run_info()

        # Process the entry list
        self.process_entry_list(
            n_entry,
            n_skip,
            entry_list,
            skip_entry_list,
            run_event_list,
            skip_run_event_list,
            allow_missing,
        )

    def get(self, idx: int) -> Dict[str, Any]:
        """Returns a specific entry in the file.

        Parameters
        ----------
        idx : int
            Integer entry ID to access

        Returns
        -------
        dict
            Dictionary which maps each data product key to an entry in the tree
        """
        # Get the appropriate entry index
        assert idx < len(self.entry_index)
        entry_idx = self.entry_index[idx]

        # If this is the first data loading, instantiate chains
        if not self.trees_ready:
            for key in self.trees:
                chain = ROOT.TChain(f"{key}_tree")  # pylint: disable=E1101
                for f in self.file_paths:
                    chain.AddFile(f)
                self.trees[key] = chain
            self.trees_ready = True

        # Move the entry pointer
        for tree in self.trees.values():
            tree.GetEntry(entry_idx)

        # Load the relevant data products
        data = {}
        for key, tree in self.trees.items():
            data[key] = getattr(tree, f"{key}_branch")

        return data

    @staticmethod
    def list_data(file_path: str) -> Dict[str, List[str]]:
        """Dumps top-level information about the contents of a LArCV root file.

        Parameters
        ----------
        file_path : str
            Path to the file to scan

        Returns
        -------
        dict
            Dictionary which maps data types onto a list of keys
        """
        # Load up the file
        f = ROOT.TFile.Open(file_path, "r")  # pylint: disable=E1101

        # Loop over the list of keys
        data = {"sparse3d": [], "cluster3d": [], "particle": []}
        for k in f.GetListOfKeys():
            # The the key name
            name = k.GetName()

            # Only look at tree names
            if not name.endswith("_tree"):
                continue
            if len(name.split("_")) < 3:
                continue

            # Get the data type name, skip if not recognized
            key = name.split("_")[0]
            if key not in data:
                continue

            # Append this specific tree name
            data[key].append(name[: name.rfind("_")])

        return data
