
from spine.data import Flash, ObjectList
from spine.utils.conditional import larcv
from spine.io.parse.base import ParserBase
from spine.utils.optical import merge_flashes
import numpy as np

__all__ = ['FlashParser']

class FlashParser(ParserBase):
    """Copy construct Flash and return an array of `Flash`.

    This parser also takes care of flashes that have been split between their
    respective optical volumes, provided a `flash_event_list`. This parser
    assumes that the trees are provided in order of the volume ID they
    correspond to.

    .. code-block. yaml
        schema:
          flashes:
            parser: flash
            flash_event_list:
              - flash_cryoE
              - flash_cryoW
    """

    # Name of the parser (as specified in the configuration)
    name = 'flash'

    # Alternative allowed names of the parser
    aliases = ('opflash',)

    def __init__(self, merge=False, merge_threshold=1.0, time_method='min', **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        merge : bool, default False
            If True, merge the flashes within the merge_threshold
        merge_threshold : float, default 1.0 [us]
            Threshold for merging flashes
        time_method : str, default 'min'
            Method to use to merge the flash times. Options are 'min' or 'mean'
        """
        super().__init__(**kwargs)
        self.merge = merge
        self.merge_threshold = merge_threshold
        self.time_method = time_method

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def process(self, flash_event=None, flash_event_list=None):
        """Fetches the list of optical flashes.

        Parameters
        -------------
        flash_event : larcv.EventFlash, optional
            Optical flash event which contains a list of flash objects
        flash_event_list : List[larcv.EventFlash], optional
            List of optical flash events, each a list of flash objects

        Returns
        -------
        List[Flash]
            List of optical flash objects
        """
        print('*'*50)
        print(f'Processing flash event list: {flash_event_list}')
        # Check on the input
        assert ((flash_event is not None) ^
                (flash_event_list is not None)), (
                "Must specify either `flash_event` or `flash_event_list`.")

        # Parse flash objects
        if flash_event is not None:
            # If there is a single flash event, parse it as is
            flash_list = flash_event.as_vector()
            flashes = [Flash.from_larcv(larcv.Flash(f)) for f in flash_list]

        else:
            # Otherwise, set the volume ID of the flash to the source index
            # and count the flash index from 0 to the largest number
            flashes = []
            idx = 0
            for volume_id, flash_event in enumerate(flash_event_list):
                for f in flash_event.as_vector():
                    # Cast and update attributes
                    flash = Flash.from_larcv(f)
                    flash.id = idx
                    flash.volume_id = volume_id

                    # Append, increment counter
                    flashes.append(flash)
                    idx += 1
            # If we are merging, find flashes within merge_threshold that are in different volumes
            if self.merge:
                flashes,_ = merge_flashes(flashes, self.merge_threshold, self.time_method)
        return ObjectList(flashes, Flash())