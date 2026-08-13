"""Module to write log files to CSV."""

import os
from collections.abc import Iterable, Mapping
from types import TracebackType
from typing import Any

__all__ = ["CSVLogger"]


class CSVLogger:
    """Writes data to a CSV file with optimized performance.

    Builds a CSV file to store the output of the analysis tools. It can only be
    used to store relatively basic quantities (scalars, strings, etc.).

    **Performance Optimization**: This writer keeps the file handle open during
    its lifetime, eliminating the overhead of opening/closing the file on every
    write operation. This provides significant speedup when writing many rows.
    By default, uses line buffering (buffer_size=1) to ensure each row is safely
    written while maintaining excellent performance.

    **Usage**: The writer should be properly closed when done:

    1. Using context manager (recommended):

       .. code-block:: python

           with CSVLogger('output.csv') as writer:
               writer.append({'col1': 1, 'col2': 2})
               writer.append({'col1': 3, 'col2': 4})
           # File automatically closed and flushed

    2. Manual management (used by AnaBase):

       .. code-block:: python

           writer = CSVLogger('output.csv')
           writer.append({'col1': 1, 'col2': 2})
           writer.close()  # Must call explicitly!

    **Configuration**: Buffer size can be configured:

    - In analysis scripts (YAML config):

      .. code-block:: yaml

          ana:
            buffer_size: 1  # Line buffered (default, safe and fast)
            my_analysis:
              ...

    - In driver logging (YAML config):

      .. code-block:: yaml

          base:
            csv_buffer_size: 1  # For driver log file
    """

    name = "csv"

    def __init__(
        self,
        file_name: str = "output.csv",
        directory: str | None = None,
        overwrite: bool = False,
        append: bool = False,
        accept_missing: bool = False,
        buffer_size: int = 1,
    ) -> None:
        """Initialize the basics of the output file.

        Parameters
        ----------
        file_name : str, default 'output.csv'
            Name of the output CSV file
        directory : str, optional
            Output directory. When provided, the CSV file is written under
            this directory using the basename of ``file_name``.
        overwrite : bool, default False
            If True, overwrite the output file if it already exists
        append : bool, default False
            If True, add more rows to an existing CSV file
        accept_missing : bool, default False
            Tolerate missing keys
        buffer_size : int, default 1
            Buffer size for file writing. 1 is line buffered (default, safe),
            -1 uses system default buffering,
            >1 is buffer size in bytes
        """
        if directory is not None:
            file_name = os.path.join(directory, os.path.basename(file_name))

        # Check that output file does not already exist, if requested
        if not overwrite and not append and os.path.isfile(file_name):
            raise FileExistsError(f"File with name {file_name} already exists.")

        # Store persistent attributes
        self.file_name = file_name
        self.append_file = append
        self.accept_missing = accept_missing
        self.buffer_size = buffer_size
        self.keys = None
        self.file_handle = None

        # If appending, check that the file exists and read the header
        if self.append_file:
            if not os.path.isfile(file_name):
                raise FileNotFoundError(
                    f"File not found at path: {file_name}. When using "
                    "`append=True` in `CSVLogger`, the file must exist at "
                    "the prescribed path before data is written to it."
                )

            with open(self.file_name, "r", encoding="utf-8") as out_file:
                self.keys = out_file.readline().strip().split(",")

    def __enter__(self) -> "CSVLogger":
        """Context manager entry. Opens the file handle.

        Returns
        -------
        CSVLogger
            Self reference for context manager
        """
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Context manager exit. Closes the file handle.

        Parameters
        ----------
        exc_type : type
            Exception type if an exception occurred
        exc_val : Exception
            Exception value if an exception occurred
        exc_tb : TracebackType
            Exception traceback if an exception occurred

        Returns
        -------
        bool
            False to propagate exceptions
        """
        self.close()
        return False

    def open(self) -> None:
        """Open the file handle for writing.

        If the file handle is already open, this does nothing.
        The file is opened in append mode if append_file is True
        and the file exists, otherwise in write mode.
        """
        if self.file_handle is None:
            mode = (
                "a"
                if (self.append_file or self.keys is not None)
                and os.path.isfile(self.file_name)
                else "w"
            )
            if mode == "w":
                dir_name = os.path.dirname(self.file_name)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
            self.file_handle = open(
                self.file_name, mode, encoding="utf-8", buffering=self.buffer_size
            )

    def close(self) -> None:
        """Close the file handle and ensure all data is written.

        This flushes any buffered data before closing. After calling
        this, the writer cannot be used unless open() is called again.
        """
        if self.file_handle is not None:
            self.file_handle.flush()
            self.file_handle.close()
            self.file_handle = None

    def flush(self) -> None:
        """Explicitly flush the file buffer to disk.

        This forces any buffered data to be written to disk without
        closing the file. Useful for ensuring data persistence at
        specific checkpoints.
        """
        if self.file_handle is not None:
            self.file_handle.flush()

    def create(self, data: dict[str, Any]) -> None:
        """Initialize the header of the CSV file, record the keys to be stored.

        Parameters
        ----------
        data : dict
            Dictionary containing the output of the reconstruction chain
        """
        # Open before recording the initialized keys. This distinguishes a
        # first write, which may overwrite, from reopening an existing writer,
        # which must append without replacing its earlier rows.
        self.open()

        # File handle is guaranteed to be open here
        assert self.file_handle is not None

        # Save the list of keys to store
        self.keys = list(data.keys())

        # Create a header and write it to file
        header_str = ",".join(self.keys)
        self.file_handle.write(header_str + "\n")

    def append(self, data: dict[str, Any]) -> None:
        """Append the CSV file with the output.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        """
        # Fetch the values to store
        if self.keys is None:
            # If this function has never been called, initialiaze the CSV file
            self.create(data)

        else:
            # If it has, check that the list of keys is identical
            if list(data.keys()) != self.keys:
                # If it is not identical, check the discrepancies
                missing = self.array_diff(self.keys, data.keys())
                excess = self.array_diff(data.keys(), self.keys)
                if len(excess):
                    raise AssertionError(
                        "There are keys in this entry which were not "
                        "present when the CSV file was initialized. "
                        f"New keys: {list(excess)}"
                    )

                if len(missing) and not self.accept_missing:
                    raise AssertionError(
                        "There are keys missing in this entry which were "
                        "present when the CSV file was initialized. "
                        f"Missing keys: {list(missing)}"
                    )

                new_data = {k: -1 for k in self.keys}
                for k, v in data.items():
                    new_data[k] = v
                data = new_data

        # Ensure file is open
        if self.file_handle is None:
            self.open()

        # File handle is guaranteed to be open here
        assert self.file_handle is not None
        assert self.keys is not None

        # Append to file (no open/close overhead!)
        result_str = ",".join([str(data[k]) for k in self.keys])
        self.file_handle.write(result_str + "\n")

    def append_columns(self, data: Mapping[str, Any]) -> None:
        """Append equal-length columns without constructing row dictionaries.

        Parameters
        ----------
        data : Mapping[str, Sequence]
            Ordered output columns. Every column must have the same length.
        """
        keys = list(data)
        if self.keys is None:
            self.create({key: None for key in keys})
        elif keys != self.keys:
            raise AssertionError(
                "Columnar CSV keys must match the initialized header. "
                f"Expected {self.keys}, got {keys}."
            )

        lengths = {len(data[key]) for key in keys}
        if len(lengths) > 1:
            raise ValueError("All columnar CSV fields must have the same length.")
        count = lengths.pop() if lengths else 0
        if self.file_handle is None:
            self.open()
        assert self.file_handle is not None
        self.file_handle.writelines(
            ",".join(str(data[key][i]) for key in keys) + "\n" for i in range(count)
        )

    @staticmethod
    def array_diff(array_x: Iterable[str], array_y: Iterable[str]) -> set[str]:
        """Compare the content of two arrays.

        This functions returns the elemnts of the first array that
        do not appear in the second array.

        Parameters
        ----------
        array_x : List[str]
            First array of strings
        array_y : List[str]
            Second array of strings

        Returns
        -------
        Set[str]
            Set of keys that appear in `array_x` but not in `array_y`.
        """
        return set(array_x).difference(set(array_y))
