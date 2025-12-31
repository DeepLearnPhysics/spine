import numpy as np


def write_lut(data_manager, path, lut, name=None):
    lut_meta, lut_arr = lut.to_array()
    path = path if name is None else f"{path}/{name}"
    data_manager.create_dset(path, dtype=lut_arr.dtype)
    data_manager.reserve_data(path, slice(0, len(lut_arr)))
    data_manager.write_data(path, slice(0, len(lut_arr)), lut_arr)
    data_manager.set_attrs(path, meta=lut_meta)


def read_lut(data_manager, path, name=None):
    path = path if name is None else f"{path}/{name}"
    lut_arr = data_manager.get_dset(path)
    lut_meta = data_manager.get_attrs(path)["meta"]
    return LUT.from_array(lut_meta, lut_arr)


class LUT(object):
    """
    Creates a lookup table that can be used to quickly access data based
    on tuples of integers. Works best if keys are contiguous within
    each position of the tuple. E.g.::

        key0 = [0,1,2]
        key1 = [30,31,32]

    is 10x more memory efficient than::

        key0 = [10,20,30]
        key1 = [300,310,320]

    Initialize with tuples of min and max values for each of the used key
    values::

        key0 = [0,1,2,3]
        key1 = [5,6,7,8]
        shape = (2,)
        dtype = 'f8'
        lut = LUT(dtype, (min(key0), max(key0)), (min(key1), max(key1)), shape=shape)

    Data can then be stored in the table using a tuple of key arrays::

        lut[(key0,key1)] = np.array([[0,0],[1,1],[2,2],[3,3]])

    and accessed::

        lut[(key0,key1)] # np.array([[0,0],[1,1],[2,2],[3,3]])

    A default value should be set for keys that are not found in the table::

        lut.default = np.array([-1,-1])
        lut[([0],[0])] # np.array([-1,-1])

    """

    def __init__(self, dtype, *min_max_keys, default=None, shape=None):
        self.dtype = dtype
        self.min_max_keys = np.array(min_max_keys, dtype="i8")
        self.lengths = np.array([max_ - min_ + 1 for min_, max_ in self.min_max_keys])
        self.max_hash = int(self._hash(*[max_ for min_, max_ in min_max_keys]))
        shape = (self.max_hash + 1,) + shape if shape else (self.max_hash + 1,)
        self._data = np.zeros(shape, dtype=self.dtype)
        self._filled = np.zeros(shape[0], dtype=bool)
        if default is not None:
            self.default = default

    def __repr__(self):
        str_ = "LUT("
        str_ += repr(self.dtype)
        for min_max in self.min_max_keys:
            str_ += ", " + repr(min_max)
        str_ += f", default={repr(self.default)}"
        str_ += f", shape={repr(self._data.shape[1:])}"
        str_ += ")"
        return str_

    def __eq__(self, other):
        self_arr = self.to_array()
        other_arr = other.to_array()
        return all(self_arr[0].ravel() == other_arr[0].ravel()) and all(
            self_arr[1].ravel() == other_arr[1].ravel()
        )

    @property
    def nbytes(self):
        """
        :returns: number of bytes used by underlying arrays
        """
        return (
            self._data.nbytes
            + self._filled.nbytes
            + self.min_max_keys.nbytes
            + self.lengths.nbytes
        )

    def compress(self, sel=tuple()):
        """
        :param sel: optional, for multi-dimensional LUT data apply this selection to the returned data

        :returns: compressed array of entry data that has been filled
        """
        if len(sel):
            sel = (...,) + sel
            return np.compress(self._filled, self._data[sel], axis=0)
        else:
            return np.compress(self._filled, self._data, axis=0)

    def min(self, sel=tuple()):
        """
        :param sel: optional, for multi-dimensional LUT data apply this selection to the returned data

        :returns: minimum value of compressed LUT data
        """
        return self.compress(sel).min()

    def max(self, sel=tuple()):
        """
        :param sel: optional, for multi-dimensional LUT data apply this selection to the returned data

        :returns: maximum value of compressed LUT data
        """
        return self.compress(sel).max()

    @staticmethod
    def from_array(meta_arr, data_arr):
        """
        Convert an array-based representation of a lookup table (as returned
        by ``LUT.to_array()``) to a ``LUT`` object.

        :param meta_arr: array containing meta data

        :param data_arr: array containing lookup table data

        :returns: ``LUT`` object
        """
        min_max_keys = meta_arr[0]["min_max_keys"]
        default = meta_arr[0]["default"]
        data = data_arr["data"]
        filled = data_arr["filled"]

        lut = LUT(data.dtype, *min_max_keys, shape=data.shape[1:])
        # initialization order is important here!
        lut._data = data
        lut._filled = filled
        lut.default = default

        return lut

    def to_array(self):
        """
        Generate an array-based representation of a ``LUT`` object. Returns
        two arrays. The first has a datatype::

            dtype([('min_max_keys', 'i8', ({nkeys}, 2)), ('default', {data_dtype}, {data_shape})])

        and ``shape: (1,)``. This contains meta-data needed to reconstruct
        the LUT hashing function and the default value. The second has a
        datatype::

            dtype([('data', {dtype}, {data_shape}), ('filled', bool, {data_shape})])

        and ``shape: (N,)``. This represents the data stored in the lookup
        table.

        :returns: ``tuple`` of meta-array and data-array as described above
        """
        dtype_meta = np.dtype(
            [
                ("min_max_keys", "i8", (len(self.min_max_keys), 2)),
                ("default", self.dtype, self._data.shape[1:]),
            ]
        )
        dtype_data = np.dtype(
            [("data", self.dtype, self._data.shape[1:]), ("filled", self._filled.dtype)]
        )
        meta_arr = np.zeros((1,), dtype=dtype_meta)
        meta_arr["min_max_keys"] = self.min_max_keys
        meta_arr["default"] = self.default

        data_arr = np.zeros(self._data.shape[0], dtype=dtype_data)
        data_arr["data"] = self._data
        data_arr["filled"] = self._filled

        return meta_arr, data_arr

    def _hash(self, *keys):
        val = 1 + np.array(keys[0]).astype("i8") - self.min_max_keys[0][0]
        for i, key in enumerate(keys[1:]):
            val += (np.array(key).astype("i8") - self.min_max_keys[i + 1][0]) * np.prod(
                self.lengths[: i + 1]
            )

        return val.astype(int).ravel()

    def hash(self, *keys):
        """
        Generate a hash index from key value arrays

        :param *keys: arrays of each key value, ``shape: (N,)``

        :returns: array of hash index, ``shape: (N,)``
        """
        val = self._hash(*keys)
        val[val < 0] = 0
        val[val > self.max_hash] = 0
        return val

    def _reverse_hash(self, hashed_id):
        val = np.array(hashed_id).astype("i8").ravel()
        val -= 1
        keys = []

        for i in reversed(range(len(self.lengths))):
            stride = np.prod(self.lengths[:i]) if i > 0 else 1
            key_offset = val // stride
            keys.insert(0, (key_offset + self.min_max_keys[i][0]))
            val = val % stride

        return tuple(np.array(k, dtype="i8") for k in keys)

    def reverse_hash(self, hashed_id):
        """
        Reverses a hashed index back into the original key tuple.

        :param hashed_id: hashed index to reverse, ``int``

        :returns: tuple of keys, ``shape: (N,)``

        """
        hashed_id = int(hashed_id)

        return self._reverse_hash(hashed_id)

    @property
    def default(self):
        """
        Default value to return if key not found in table. Datatype is
        same as lookup table
        """
        return self._data[0]  # position 0 is reserved for the default value

    @default.setter
    def default(self, val):
        if isinstance(val, np.number):
            new_default = val
        else:
            new_default = np.broadcast_to(
                np.expand_dims(val, axis=0), self._data.shape
            )[~self._filled]

        self._data[~self._filled] = new_default

    def clear(self, *keys):
        """
        Remove stored value for specified keys

        :param *keys: arrays of key values, ``shape: (N,)``
        """
        idx = self.hash(*keys)
        self._data[idx] = self.default
        self._filled[idx] = False

    def keys(self):
        """
        Return existing keys

        :returns: tuple of arrays, each ``shape: (N,)``
        """
        shapes = [
            [1] * i + [n] + [1] * (len(self.lengths) - i - 1)
            for i, n in enumerate(self.lengths)
        ]
        keys = [
            np.arange(min_, min_ + n).reshape(shape)
            for min_, n, shape in zip(self.min_max_keys[:, 0], self.lengths, shapes)
        ]
        keys = [k.ravel() for k in np.broadcast_arrays(*keys)]
        idx = self.hash(*keys)
        filled = self._filled[idx]

        rv = tuple(
            [
                key[
                    (
                        filled
                        if len(filled.shape) == 1
                        else np.all(filled.reshape(len(key), -1), axis=-1)
                    )
                ]
                for key in keys
            ]
        )
        return rv

    def __getitem__(self, keys):
        return self._data[self.hash(*keys)]

    def __setitem__(self, keys, val):
        idx = self.hash(*keys)

        self._data[idx] = val
        self._filled[idx] = True

        if self._filled[0]:
            i = np.where(idx == 0)[0]
            raise RuntimeError(
                f"invalid key tried to overwrite default: {[np.array(key)[i] for key in keys]}, value={np.array(val)[i]}"
            )

    def get_keys_from_val(self, val):
        """
        Retrieves the first key tuples that map to the given value.

        :param val: tuple of value, ``shape: (N,)``

        :returns: tuple of key corresponding to the parameter value, ``shape: (N,)``
        """
        if self.default[0] in val:
            raise KeyError(
                f"A default value was given (given: {val}, default: {self.default[0]}), no matching key exist."
            )

        val = np.asarray(val)
        matching_keys = []

        if val.shape != self._data[0].shape:
            raise ValueError(
                f"Invalid value format. Expected shape: {self._data[0].shape}, but got: {val.shape}."
            )

        for idx, stored_val in enumerate(self._data):
            if np.array_equal(stored_val, val):
                return np.ravel(self.reverse_hash(idx))

        raise KeyError(f"No matching key found for the given value: {val}")
