"""Shared helpers for LArCV utility scripts."""

from typing import Any


def list_tree_keys(root_file: Any) -> list[str]:
    """Return top-level key names following the LArCV ``*_tree`` convention.

    Parameters
    ----------
    root_file : ROOT.TFile
        Open ROOT file containing LArCV TTrees.

    Returns
    -------
    list[str]
        Names of top-level objects ending in ``"_tree"``.
    """
    return [
        key.GetName()
        for key in root_file.GetListOfKeys()
        if key.GetName().endswith("_tree")
    ]


def get_tree_key(root_file: Any, tree_name: str | None = None) -> str:
    """Return a LArCV tree key from an optional short tree name.

    Parameters
    ----------
    root_file : ROOT.TFile
        Open ROOT file containing LArCV TTrees. Used only when ``tree_name`` is
        not provided.
    tree_name : str, optional
        Short tree name without the ``"_tree"`` suffix.

    Returns
    -------
    str
        Full ROOT key for the selected LArCV tree.
    """
    if tree_name is not None:
        return f"{tree_name}_tree"

    return list_tree_keys(root_file)[0]


def get_branch_key(tree_key: str) -> str:
    """Return the LArCV branch key corresponding to a tree key.

    Parameters
    ----------
    tree_key : str
        Full ROOT tree key ending in ``"_tree"``.

    Returns
    -------
    str
        Matching branch key ending in ``"_branch"``.
    """
    return tree_key.replace("_tree", "_branch")


def get_tree(root_file: Any, tree_key: str) -> Any:
    """Return a typed PyROOT TTree proxy.

    PyROOT can return a generic ``TObject`` proxy through ``TFile.Get`` in some
    environments. Attribute lookup gives the typed tree proxy needed by callers
    to access ``GetEntries``, ``GetEntry`` and LArCV branches.

    Parameters
    ----------
    root_file : ROOT.TFile
        Open ROOT file containing the requested tree.
    tree_key : str
        Name of the tree object to retrieve.

    Returns
    -------
    Any
        PyROOT TTree proxy. The return type is intentionally dynamic because
        PyROOT exposes it at runtime.
    """
    return getattr(root_file, tree_key)
