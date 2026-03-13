"""Module to write log files to CSV."""

import os

__all__ = ["CSVWriter"]


class CSVWriter:
    """Writes data to a CSV file with optimized performance.

    Builds a CSV file to store the output of the analysis tools. It can only be
    used to store relatively basic quantities (scalars, strings, etc.).

    **Performance Optimization**: This writer keeps the file handle open during
    its lifetime, eliminating the overhead of opening/closing the file on every
    write operation. This provides significant speedup when writing many rows.

    **Usage**: The writer should be properly closed when done:

    1. Using context manager (recommended):

       .. code-block:: python

           with CSVWriter('output.csv') as writer:
               writer.append({'col1': 1, 'col2': 2})
               writer.append({'col1': 3, 'col2': 4})
           # File automatically closed and flushed

    2. Manual management (used by AnaBase):

       .. code-block:: python

           writer = CSVWriter('output.csv')
           writer.append({'col1': 1, 'col2': 2})
           writer.close()  # Must call explicitly!
    """

    name = "csv"

    def __init__(
        self,
        file_name="output.csv",
        overwrite=False,
        append=False,
        accept_missing=False,
        buffer_size=-1,
    ):
        """Initialize the basics of the output file.

        Parameters
        ----------
        file_name : str, default 'output.csv'
            Name of the output CSV file
        overwrite : bool, default False
            If True, overwrite the output file if it already exists
        append : bool, default False
            If True, add more rows to an existing CSV file
        accept_missing : bool, default True
            Tolerate missing keys
        buffer_size : int, default -1
            Buffer size for file writing. -1 uses system default,
            0 is unbuffered, 1 is line buffered, >1 is buffer size in bytes
        """
        # Check that output file does not already exist, if requested
        if not overwrite and not append and os.path.isfile(file_name):
            raise FileExistsError(f"File with name {file_name} already exists.")

        # Store persistent attributes
        self.file_name = file_name
        self.append_file = append
        self.accept_missing = accept_missing
        self.buffer_size = buffer_size
        self.result_keys = None
        self.file_handle = None

        # If appending, check that the file exists and read the header
        if self.append_file:
            if not os.path.isfile(file_name):
                raise FileNotFoundError(
                    f"File not found at path: {file_name}. When using "
                    "`append=True` in CSVWriter, the file must exist at "
                    "the prescribed path before data is written to it."
                )

            with open(self.file_name, "r", encoding="utf-8") as out_file:
                self.result_keys = out_file.readline().strip().split(",")

    def __enter__(self):
        """Context manager entry. Opens the file handle.

        Returns
        -------
        CSVWriter
            Self reference for context manager
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit. Closes the file handle.

        Parameters
        ----------
        exc_type : type
            Exception type if an exception occurred
        exc_val : Exception
            Exception value if an exception occurred
        exc_tb : traceback
            Exception traceback if an exception occurred

        Returns
        -------
        bool
            False to propagate exceptions
        """
        self.close()
        return False

    def open(self):
        """Open the file handle for writing.

        If the file handle is already open, this does nothing.
        The file is opened in append mode if append_file is True
        and the file exists, otherwise in write mode.
        """
        if self.file_handle is None:
            mode = "a" if self.append_file and os.path.isfile(self.file_name) else "w"
            self.file_handle = open(
                self.file_name, mode, encoding="utf-8", buffering=self.buffer_size
            )

    def close(self):
        """Close the file handle and ensure all data is written.

        This flushes any buffered data before closing. After calling
        this, the writer cannot be used unless open() is called again.
        """
        if self.file_handle is not None:
            self.file_handle.flush()
            self.file_handle.close()
            self.file_handle = None

    def flush(self):
        """Explicitly flush the file buffer to disk.

        This forces any buffered data to be written to disk without
        closing the file. Useful for ensuring data persistence at
        specific checkpoints.
        """
        if self.file_handle is not None:
            self.file_handle.flush()

    def create(self, result_blob):
        """Initialize the header of the CSV file, record the keys to be stored.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        """
        # Save the list of keys to store
        self.result_keys = list(result_blob.keys())

        # Open the file handle if not already open
        self.open()

        # File handle is guaranteed to be open here
        assert self.file_handle is not None

        # Create a header and write it to file
        header_str = ",".join(self.result_keys)
        self.file_handle.write(header_str + "\n")

    def append(self, result_blob):
        """Append the CSV file with the output.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        """
        # Fetch the values to store
        if self.result_keys is None:
            # If this function has never been called, initialiaze the CSV file
            self.create(result_blob)

        else:
            # If it has, check that the list of keys is identical
            if list(result_blob.keys()) != self.result_keys:
                # If it is not identical, check the discrepancies
                missing = self.array_diff(self.result_keys, result_blob.keys())
                excess = self.array_diff(result_blob.keys(), self.result_keys)
                if len(excess):
                    raise AssertionError(
                        "There are keys in this entry which were not "
                        "present when the CSV file was initialized. "
                        f"New keys: {list(excess)}"
                    )

                if not self.accept_missing:
                    raise AssertionError(
                        "There are keys missing in this entry which were "
                        "present when the CSV file was initialized. "
                        f"Missing keys: {list(missing)}"
                    )

                new_result_blob = {k: -1 for k in self.result_keys}
                for k, v in result_blob.items():
                    new_result_blob[k] = v
                result_blob = new_result_blob

        # Ensure file is open
        if self.file_handle is None:
            self.open()

        # File handle is guaranteed to be open here
        assert self.file_handle is not None
        assert self.result_keys is not None

        # Append to file (no open/close overhead!)
        result_str = ",".join([str(result_blob[k]) for k in self.result_keys])
        self.file_handle.write(result_str + "\n")

    @staticmethod
    def array_diff(array_x, array_y):
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
