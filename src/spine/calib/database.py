"""SQLite calibration database parsing."""

from __future__ import annotations

import sqlite3 as sql
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.integer]


class CalibrationDatabase:
    """Wraps basic SQLite loading/querying functions to provide a more
    user-friendly API to the calibration classes.

    Notes
    -----
    This class assumes that the structure of the SQLite libraries used
    is that of ICARUS calibration databases, for now.
    """

    _db_types = ("value", "map")

    def __init__(
        self,
        db_path: str,
        num_tpcs: int,
        db_type: str = "value",
        value_key: str = "scale",
        value_keys: list[str] | tuple[str, ...] | None = None,
        value_scale: float = 1.0,
    ) -> None:
        """Given a path to a calibration data base, load the
        information into a dictionary.

        Parameters
        ----------
        db_path : str
            Path to a SQLite database
        num_tpcs : int
            Expected number of TPCs
        db_type : str, default 'value'
            Type of database (One 'value' or one 'map' per TPC)
        value_key : str, default 'scale'
            Name of the quantity to load for each bin when using 'map' db_type
        value_keys : sequence[str], optional
            Ordered columns containing one value per TPC in each row when using
            ``'value'`` db_type. If omitted, values are loaded from one row per
            channel using the quantity inferred from the database filename.
        value_scale : float, default 1.0
            Multiplicative scale applied to values loaded from a ``'value'``
            database, for example 1000 to convert milliseconds to microseconds.

        Returns
        -------
        dict
            Dictionary which maps a run onto a set of values (one per TPC)

        Notes
        -----
        This makes assumptions about how the database is structured for
        ICARUS calibration for now as of the time of implementation.
        """
        # Make sure the type of database is recognized
        if db_type not in self._db_types:
            raise ValueError(
                f"Type of database not recognized: {db_type}. "
                f"Must be one of {self._db_types}."
            )
        if value_keys is not None and len(value_keys) != num_tpcs:
            raise ValueError(
                "Must provide exactly one database value key per TPC " f"({num_tpcs})."
            )

        # Load the database into a pandas dataframe
        stem = Path(db_path).stem
        quantity = "_".join(stem.split("_")[1:-1])

        db = sql.connect(db_path)
        df = pd.read_sql_query(f"SELECT * from {stem}_data", db)
        run_df = pd.read_sql_query(f"SELECT * from {stem}_iovs", db)
        db.close()

        df = df.merge(run_df, left_on="__iov_id", right_on="iov_id")
        df = df[df.active == 1]

        # Loop over unique runs, store the values per TPCs for each run
        self.num_tpcs = num_tpcs
        self.dict: dict[int, FloatArray | list[CalibrationLUT]] = {}
        for run in np.unique(df.begin_time):
            df_run = cast(pd.DataFrame, df[df.begin_time == run])
            run_id = run - int(1e9)
            if db_type == "value":
                self.dict[run_id] = self.load_values(
                    df_run, quantity, value_keys, value_scale
                )
            else:
                self.dict[run_id] = self.load_tables(df_run, value_key)

        # Create a list of boundary runs
        self.runs = np.sort(list(self.dict.keys()))

    def load_values(
        self,
        df_run: pd.DataFrame,
        quantity: str,
        value_keys: list[str] | tuple[str, ...] | None = None,
        value_scale: float = 1.0,
    ) -> FloatArray:
        """Loads one value per TPC.

        Parameters
        ----------
        df_run : pd.DataFrame
            Dataframe which corresponds to the run being loaded
        quantity : str
            Name of the quantity to load
        value_keys : sequence[str], optional
            Ordered columns containing one value per TPC in a single row.
        value_scale : float, default 1.0
            Multiplicative scale applied to the loaded values.

        Returns
        -------
        np.ndarray
            (N_tpc) Array of calibration values
        """
        # Load one TPC value from each requested column of a single payload row
        if value_keys is not None:
            if len(df_run) != 1:
                raise ValueError(
                    "Column-mapped value databases must provide one row per IOV."
                )

            return (
                np.asarray(df_run.iloc[0][list(value_keys)], dtype=float) * value_scale
            )

        # Check that there is exactly one value per tpc
        if len(df_run) != self.num_tpcs:
            raise ValueError("There should be one quantity specified per TPC")

        # Store the values into an array
        array = np.empty(self.num_tpcs)
        for i in range(len(df_run)):
            channel = int(df_run.iloc[i].channel)
            value = df_run.iloc[i][quantity]
            array[channel] = value * value_scale

        return array

    def load_tables(
        self, df_run: pd.DataFrame, quantity: str
    ) -> list["CalibrationLUT"]:
        """Loads one look-up table per TPC.

        Parameters
        ----------
        df_run : pd.DataFrame
            Dataframe which corresponds to the run being loaded
        quantity : str
            Name of the quantity to load for each bin

        Returns
        -------
        np.ndarray
            (N_tpc) Array of calibration look-up tables
        """
        tpc_luts = []
        tpc_keys = ["EE", "EW", "WE", "WW"]
        for tpc_key in tpc_keys:
            df_tpc = cast(pd.DataFrame, df_run[df_run.tpc == tpc_key])
            tpc_luts.append(CalibrationLUT.from_dataframe(df_tpc, quantity))

        return tpc_luts

    def __getitem__(self, run_id: int) -> FloatArray | list["CalibrationLUT"]:
        """Mirrors the `query` function.

        Parameters
        ----------
        run_id : int
            ID of the run to get the values for

        Returns
        -------
        np.ndarray
            List of values per channel
        """
        return self.query(run_id)

    def query(self, run_id: int) -> FloatArray | list[CalibrationLUT]:
        """Gets the database information for a given run. If the run does not
        exist in the list, pick the one closest but earlier than it.

        Parameters
        ----------
        run_id : int
            ID of the run to get the values for

        Returns
        -------
        np.ndarray
            List of values per channel
        """
        # Identify the closest run that is before the queried run
        if run_id < self.runs[0]:
            raise IndexError(
                "No calibration information for run " f"{run_id} < {self.runs[0]}"
            )

        closest_run = int(self.runs[np.where(self.runs <= run_id)[0][-1]])

        return self.dict[closest_run]


class CalibrationLUT:
    """Look-up table for calibration values. Given a set of coordinates,
    returns a calibration value.
    """

    def __init__(
        self,
        dims: list[int],
        bins: list[int],
        ranges: list[list[float]],
        values: FloatArray,
        dummy: float | None = -999.0,
    ) -> None:
        """Initialize the calibration map.

        Parameters
        ----------
        dims : List[int]
            List of dimensions (0: x, 1: y, 2: z)
        bins : List[int]
            Number of bins in each dimension
        ranges : List[List[float]]
            Axis range in each dimension
        values : np.ndarray
            Values in each bin
        dummy : float
            Dummy values which should be overwritten with 1. (no information)
        """
        # Store metadata information
        if len(ranges) != len(dims) or len(bins) != len(dims):
            raise ValueError("Must provide a bin count and range per dimension.")
        self.dims = dims
        self.range = np.array(ranges)
        self.bins = np.array(bins)
        self.bin_sizes = (self.range[:, 1] - self.range[:, 0]) / self.bins

        # Store the values in each bin. Should be a dense matrix
        if not np.all(values.shape == self.bins):
            raise ValueError("Must provide one calibration value per bin.")
        self.values = values

        # Overwrite dummy values to 1.
        if dummy is not None:
            self.values[self.values == dummy] = 1.0

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        value_key: str,
        dims: tuple[int, int] = (1, 2),
        bin_keys: tuple[str, str] = ("ybin", "zbin"),
        low_keys: tuple[str, str] = ("ylow", "zlow"),
        high_keys: tuple[str, str] = ("yhigh", "zhigh"),
        dummy: float | None = -999.0,
    ) -> "CalibrationLUT":
        """Build a two-dimensional LUT from a table of binned values.

        The default column names describe the ICARUS transparency database,
        while the explicit column arguments keep the conversion independent
        of database and IOV handling.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Table containing one row per bin, including bin indexes, lower and
            upper bin edges, and the calibration value.
        value_key : str
            Column containing the calibration value for each bin.
        dims : tuple[int, int], default (1, 2)
            Coordinate dimensions represented by the two table axes, using
            0=x, 1=y and 2=z.
        bin_keys : tuple[str, str], default ('ybin', 'zbin')
            Columns containing the zero-based bin indexes for each axis.
        low_keys : tuple[str, str], default ('ylow', 'zlow')
            Columns containing the lower bin edge for each axis.
        high_keys : tuple[str, str], default ('yhigh', 'zhigh')
            Columns containing the upper bin edge for each axis.
        dummy : float, optional
            Value interpreted as missing calibration information and replaced
            with 1. Set to ``None`` to preserve every value.

        Returns
        -------
        CalibrationLUT
            Dense two-dimensional calibration look-up table.

        Raises
        ------
        ValueError
            If the table is empty, contains negative bin indexes, or does not
            provide exactly one value for every bin in the dense grid.
        """
        # Make sure the input table contains calibration information
        if dataframe.empty:
            raise ValueError("Cannot build a calibration LUT from an empty table.")

        # Extract and validate the bin indexes
        bin_ids = np.asarray(dataframe.loc[:, list(bin_keys)], dtype=int)
        if np.any(bin_ids < 0):
            raise ValueError("Calibration LUT bin indexes must be non-negative.")

        # Infer the dense map shape and check that every bin appears exactly once
        bins = (bin_ids.max(axis=0) + 1).tolist()
        unique_bins = np.unique(bin_ids, axis=0)
        if len(dataframe) != int(np.prod(bins)) or len(unique_bins) != len(dataframe):
            raise ValueError("Must provide exactly one calibration value per bin.")

        # Copy the tabular values into their dense bin locations
        values = np.empty(bins, dtype=float)
        values[tuple(bin_ids.T)] = np.asarray(dataframe[value_key], dtype=float)

        # Get the full range covered by each table axis
        ranges = [
            [float(dataframe[low].min()), float(dataframe[high].max())]
            for low, high in zip(low_keys, high_keys)
        ]

        # Initialize the corresponding look-up table
        return cls(list(dims), bins, ranges, values, dummy=dummy)

    @classmethod
    def from_root_histogram(
        cls,
        histogram: Any,
        axis_dims: tuple[int, int] = (2, 1),
        reciprocal: bool = False,
    ) -> "CalibrationLUT":
        """Build a two-dimensional LUT from a ROOT ``TH2`` histogram.

        Parameters
        ----------
        histogram : ROOT.TH2
            Histogram to copy. Underflow and overflow bins are not included.
        axis_dims : tuple[int, int], default (2, 1)
            Coordinate dimensions represented by the ROOT X and Y axes. The
            default corresponds to an X=z, Y=y histogram.
        reciprocal : bool, default False
            Store the reciprocal of each valid histogram value. Zero and
            non-finite values are treated as missing information and set to 1.

        Returns
        -------
        CalibrationLUT
            Dense two-dimensional calibration look-up table, reordered by
            coordinate dimension when the ROOT axes are not in that order.

        Raises
        ------
        ValueError
            If ``axis_dims`` does not identify two distinct dimensions.
        """
        # Make sure each histogram axis maps onto a unique coordinate
        if len(axis_dims) != 2 or len(set(axis_dims)) != 2:
            raise ValueError("Must provide two distinct ROOT histogram dimensions.")

        # Extract the histogram bin counts and ranges
        axes = (histogram.GetXaxis(), histogram.GetYaxis())
        bins = (int(histogram.GetNbinsX()), int(histogram.GetNbinsY()))
        ranges = [[float(axis.GetXmin()), float(axis.GetXmax())] for axis in axes]

        # Copy the regular bin contents, excluding underflow and overflow bins
        values = np.empty(bins, dtype=float)
        for ix in range(bins[0]):
            for iy in range(bins[1]):
                values[ix, iy] = histogram.GetBinContent(ix + 1, iy + 1)

        # Keep LUT dimensions in coordinate order. This transposes the SBND
        # ROOT maps from their native (z, y) storage to (y, z).
        order = np.argsort(axis_dims)
        dims = [int(axis_dims[i]) for i in order]
        ordered_bins = [bins[i] for i in order]
        ordered_ranges = [ranges[i] for i in order]
        values = np.transpose(values, axes=order)

        # Replace missing values with the neutral factor and optionally invert
        valid = np.isfinite(values) & (values != 0.0)
        if reciprocal:
            converted = np.ones_like(values)
            converted[valid] = 1.0 / values[valid]
            values = converted
        else:
            values = values.copy()
            values[~valid] = 1.0

        # Initialize the corresponding look-up table
        return cls(dims, ordered_bins, ordered_ranges, values, dummy=None)

    @property
    def edges(self) -> list[FloatArray]:
        """Returns the bin edges in each axis.

        Returns
        -------
        List[np.ndarray]
            (D) List of (N_i + 1) edges per dimension, with N_i the number
            of bins in the the ith dimension
        """
        edges: list[FloatArray] = []
        for i, ran in enumerate(self.range):
            edges.append(np.arange(ran[0], ran[1] + 1e-9, self.bin_sizes[i]))

        return edges

    def query(self, points: FloatArray) -> FloatArray:
        """Queries the LUT to get the calibration values for a set of points.

        Parameters
        ----------
        points: np.ndarry
            (N, 3) Coordinates of the points to query a calibration for

        Returns
        -------
        np.ndarray
            Calibration constants
        """
        # Get the bin the position belongs to:
        offsets = points[:, self.dims] - self.range[:, 0]
        bin_ids = (offsets / self.bin_sizes).astype(int)

        # Collapse to the closest bin if it is outisde of range
        bad_mask = np.where(bin_ids < 0)
        bin_ids[bad_mask] = 0
        bad_mask = np.where(bin_ids >= self.bins)
        bin_ids[bad_mask] = self.bins[bad_mask[-1]] - 1

        # Get the corrections
        return self.values[tuple(bin_ids.T)]
