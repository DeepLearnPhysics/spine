"""LArCV ROOT-file inspector for generic entry filtering."""

from __future__ import annotations

from typing import Any, Mapping

from spine.utils.conditional import LARCV_AVAILABLE, ROOT, ROOT_AVAILABLE, larcv

from .base import EntryInspector

__all__ = ["LArCVEntryInspector"]


class LArCVEntryInspector(EntryInspector):
    """Measure top-level LArCV product sizes without invoking SPINE parsers.

    ``product_size`` denotes the length of the event product's ``as_vector``
    representation.  Requested trees are opened independently and must expose
    identical entry counts.
    """

    name = "larcv"
    version = 1

    def inspect(
        self, source: str, measurements: Mapping[str, Mapping[str, Any]]
    ) -> tuple[int, dict[str, list[int]]]:
        """Inspect all configured LArCV products in a single source file.

        Parameters
        ----------
        source : str
            LArCV ROOT file to inspect.
        measurements : mapping
            Measurement names mapped to specifications.  The initial backend
            accepts only ``kind: product_size``.

        Returns
        -------
        num_entries : int
            Entry count shared by all requested product trees.
        values : dict[str, list[int]]
            Top-level product sizes for each file-local entry.
        """
        if not ROOT_AVAILABLE:
            raise ImportError("ROOT is required to inspect LArCV files.")
        if not LARCV_AVAILABLE:
            raise ImportError("larcv is required to inspect LArCV files.")

        # Register LArCV dictionaries before PyROOT materializes branch data.
        _ = larcv.__name__
        root_file = ROOT.TFile.Open(source, "READ")
        if not root_file or root_file.IsZombie():
            raise OSError(f"Could not open LArCV source: {source}")

        try:
            num_entries: int | None = None
            values: dict[str, list[int]] = {}
            for name, request in measurements.items():
                if request.get("kind") != "product_size":
                    raise ValueError(
                        f"LArCV measurement `{name}` has unsupported kind "
                        f"`{request.get('kind')}`."
                    )

                tree_name = f"{name}_tree"
                branch_name = f"{name}_branch"
                tree = root_file.Get(tree_name)
                if not tree:
                    raise KeyError(f"Missing requested LArCV tree `{tree_name}`.")

                count = int(tree.GetEntries())
                if num_entries is None:
                    num_entries = count
                elif count != num_entries:
                    raise ValueError(
                        f"LArCV tree `{tree_name}` has {count} entries; expected "
                        f"{num_entries}."
                    )

                # Disable unrelated branches when the tree contains more than
                # the canonical event-product branch.
                if hasattr(tree, "SetBranchStatus"):
                    tree.SetBranchStatus("*", 0)
                    tree.SetBranchStatus(branch_name, 1)

                measured = []
                for entry in range(count):
                    tree.GetEntry(entry)
                    product = getattr(tree, branch_name)
                    if not hasattr(product, "as_vector"):
                        raise TypeError(
                            f"LArCV branch `{branch_name}` does not expose "
                            "`as_vector()`."
                        )
                    measured.append(int(product.as_vector().size()))
                values[name] = measured

            return int(num_entries or 0), values
        finally:
            root_file.Close()
