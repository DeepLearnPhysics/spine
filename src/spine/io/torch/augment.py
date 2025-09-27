"""Module with methods to augment the input data."""

import numpy as np

from spine.data import Meta

__all__ = ["Augmenter"]


class Augmenter:
    """Generic class to handle data augmentation."""

    def __init__(self, translate=None):
        """Initialize the augmenter.

        Parameters
        ----------
        translate : dict, optional
            Translation configuration (move input image around)
        """
        # Make sure at least one augmentation scheme is requested
        assert (
            translate is not None
        ), "Must provide `translate` block minimally to do any augmentation."

        # Parse the translation configuration
        self.translater = None
        if translate is not None:
            self.translater = Translater(**translate)

    def __call__(self, data):
        """Augment the data products in one event.

        Parameters
        ----------
        data : dict
           Data product dictionary
        """
        # Get the list of keys to augment and the shared metadata
        augment_keys = []
        meta = None
        for key, value in data.items():
            if (
                isinstance(value, tuple)
                and len(value) == 3
                and isinstance(value[2], Meta)
            ):
                augment_keys.append(key)
                if meta is None:
                    meta = value[2]
                else:
                    assert (
                        meta == value[2]
                    ), "Metadata should be shared by all data products."
            elif isinstance(value, Meta):
                augment_keys.append(key)
                meta = value

        # If there are no sparse tensors in the input data, nothing to do
        if meta is None:
            return data

        # Translate
        if self.translater is not None:
            data = self.translater(data, meta, augment_keys)

        return data


class Translater:
    """Generic class to handle moving images around."""

    def __init__(self, lower, upper):
        """Initialize the translater..

        This defines a way to move the image around within a volume greater
        than that define by the image metadata. The box must be larger than
        the image itself.

        Parameters
        ----------
        lower : np.ndarray
            Lower bounds of the box in which to move the image around
        upper : np.ndarray
            Upper bounds of the box in which to move the image around
        """
        # Sanity check
        assert (
            len(lower) == len(upper) == 3
        ), "Must provide boundaries for each dimension."

        # Define a new image metadata corresponding to the full range
        self.meta = Meta(lower=np.asarray(lower), upper=np.asarray(upper))

    def __call__(self, data, meta, keys):
        """Move an image around within the the pre-defined volume.

        Parameters
        ----------
        data : dict
            Dictionary of data products to offset
        meta : Meta
            Shared image metadata
        keys : List[str]
            List of keys with coordinates to offset

        Returns
        -------
        np.ndarray
            (N, 3) Translated points
        """
        # Set the target volume pixel pitch to match that of the original image
        if np.all(self.meta.size < 0.0):
            self.meta.size = meta.size
            self.meta.count = (self.meta.upper - self.meta.lower) // meta.size
            self.meta.count = self.meta.count.astype(int)

        # Generate an offset
        offset = self.generate_offset(meta)

        # Offset all coordinates
        for key in keys:
            # If the key is the metadata, modify and continue
            if isinstance(data[key], Meta):
                data[key] = self.meta
                continue

            # Fetch attributes to modify
            voxels, features, _ = data[key]

            # Translate
            width = voxels.shape[1]
            voxels = (voxels.reshape(-1, 3) + offset).reshape(-1, width)

            # Update
            data[key] = (voxels, features, self.meta)

        return data

    def generate_offset(self, meta):
        """Generate an offset to apply to all the voxel index sets.

        This offset is such that the the voxels will be randomly shifted
        within the target bounding box.

        Parameters
        ----------
        meta : Meta
            Metadata of the original image

        Returns
        -------
        np.ndarray
            Value by which to shift the pixels by
        """
        # Check that the original metadata is compatible with the target volume
        assert np.all(
            meta.count <= self.meta.count
        ), "The input image is larger than the target translation volume."

        # Generate an offset with respect to the voxel indices
        offset = np.random.randint((self.meta.count - meta.count) + 1)

        return offset
