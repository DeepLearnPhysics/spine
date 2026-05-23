"""Log-loading helpers for training and validation metric histories."""

from __future__ import annotations

import glob
import os
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["find_key", "get_training_df", "get_validation_df"]


def find_key(
    df: pd.DataFrame | dict[str, Any], key_list: str, separator: str
) -> tuple[str, str]:
    """Resolve the first available metric name from a separator-delimited list.

    Parameters
    ----------
    df : Union[pd.DataFrame, Dict[str, Any]]
        Data container whose keys are searched.
    key_list : str
        Separator-delimited list of acceptable metric names.
    separator : str
        Character used to split ``key_list`` into candidates.

    Returns
    -------
    Tuple[str, str]
        Matched key and canonical display key name.
    """
    key_candidates = key_list.split(separator)
    key_name = key_candidates[0]
    key_found = np.array([key in df.keys() for key in key_candidates])
    if not np.any(key_found):
        raise KeyError("Could not find any of the keys provided:", key_candidates)
    key = key_candidates[np.where(key_found)[0][0]]

    return key, key_name


def get_training_df(
    log_dir: str, keys: list[str], train_prefix: str, separator: str
) -> pd.DataFrame:
    """Load and concatenate segmented training logs from one model directory.

    Parameters
    ----------
    log_dir : str
        Directory containing one model's training logs.
    keys : List[str]
        Metrics to extract from the logs.
    train_prefix : str
        Filename prefix identifying training logs.
    separator : str
        Character used to separate acceptable metric aliases.

    Returns
    -------
    pd.DataFrame
        Concatenated training logs with canonicalized metric columns.
    """
    log_files = glob.glob(f"{log_dir}/{train_prefix}*")
    if not log_files:
        raise FileNotFoundError(
            f"Found no train log with prefix '{train_prefix}' under {log_dir}."
        )

    # Each segmented log resumes from a known starting iteration. Sort by that
    # boundary so overlapping segments can be trimmed consistently.
    start_iter = np.zeros(len(log_files), dtype=np.int64)
    for i, log_file in enumerate(log_files):
        start_iter[i] = int(log_file.split("-")[-1].split(".csv")[0])
    order = np.argsort(start_iter)
    end_points = np.append(start_iter[order], 1e12)

    log_dfs = []
    for i, log_file in enumerate(np.array(log_files)[order]):
        df = pd.read_csv(log_file, nrows=int(end_points[i + 1] - end_points[i]))
        if len(df) == 0:
            continue
        # Copy each requested metric to a canonical output column so downstream
        # plotting code can use one stable key regardless of source aliasing.
        for key_list in keys:
            key, key_name = find_key(df, key_list, separator)
            df[key_name] = df[key]
        log_dfs.append(df)

    return pd.concat(log_dfs, sort=True)


def get_validation_df(
    log_dir: str, keys: list[str], val_prefix: str, separator: str
) -> pd.DataFrame:
    """Summarize validation logs into means and standard errors per iteration.

    Parameters
    ----------
    log_dir : str
        Directory containing one model's validation logs.
    keys : List[str]
        Metrics to extract from the logs.
    val_prefix : str
        Filename prefix identifying validation logs.
    separator : str
        Character used to separate acceptable metric aliases.

    Returns
    -------
    pd.DataFrame
        Validation summary with one row per checkpoint iteration.
    """
    val_data: dict[str, list[Any]] = {"iter": []}
    for key in keys:
        key_name = key.split(separator)[0]
        val_data[f"{key_name}_mean"] = []
        val_data[f"{key_name}_err"] = []

    log_files = np.array(glob.glob(f"{log_dir}/{val_prefix}*"))
    for log_file in log_files:
        df = pd.read_csv(log_file)
        iteration = int(os.path.basename(log_file).split("-")[-1].split(".")[0])
        val_data["iter"].append(iteration - 1)
        # Validation files contain one distribution per checkpoint; summarize
        # each requested metric by mean and standard error.
        for key_list in keys:
            key, key_name = find_key(df, key_list, separator)
            mean = df[key].mean()
            err = df[key].std() / np.sqrt(len(df[key]))
            val_data[f"{key_name}_mean"].append(mean)
            val_data[f"{key_name}_err"].append(err)

    # Keep the final frame ordered by iteration regardless of the glob order.
    order = np.argsort(val_data["iter"])
    for key, value in val_data.items():
        val_data[key] = [value[i] for i in order]

    return pd.DataFrame(val_data)
