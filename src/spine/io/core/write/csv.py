"""Module to write log files to CSV."""

import os

__all__ = ["CSVWriter"]


class CSVWriter:
    """Writes data to a CSV file.

    Builds a CSV file to store the output of the analysis tools. It can only be
    used to store relatively basic quantities (scalars, strings, etc.).

    Typical configuration should look like:

    .. code-block:: yaml

        io:
          ...
          writer:
            name: csv
            file_name: output.csv
    """

    name = "csv"

    def __init__(
        self,
        file_name="output.csv",
        overwrite=False,
        append=False,
        accept_missing=False,
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
        """
        # Check that output file does not already exist, if requestes
        if not overwrite and os.path.isfile(file_name):
            raise FileExistsError(f"File with name {file_name} already exists.")

        # Store persistent attributes
        self.file_name = file_name
        self.append_file = append
        self.accept_missing = accept_missing
        self.result_keys = None
        if self.append_file:
            if not os.path.isfile(file_name):
                raise FileNotFoundError(
                    f"File not found at path: {file_name}. When using "
                    "`append=True` in CSVWriter, the file must exist at "
                    "the prescribed path before data is written to it."
                )

            with open(self.file_name, "r", encoding="utf-8") as out_file:
                self.result_keys = out_file.readline().split(",")

    def create(self, result_blob):
        """Initialize the header of the CSV file, record the keys to be stored.

        Parameters
        ----------
        result_blob : dict
            Dictionary containing the output of the reconstruction chain
        """
        # Save the list of keys to store
        self.result_keys = list(result_blob.keys())

        # Create a header and write it to file
        with open(self.file_name, "w", encoding="utf-8") as out_file:
            header_str = ",".join(self.result_keys)
            out_file.write(header_str + "\n")

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

        # Append file
        with open(self.file_name, "a", encoding="utf-8") as out_file:
            result_str = ",".join([str(result_blob[k]) for k in self.result_keys])
            out_file.write(result_str + "\n")

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
