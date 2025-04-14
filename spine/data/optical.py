"""Module with a data class object which represents optical information.

This copies the internal structure of :class:`larcv.Flash`.
"""

from dataclasses import dataclass

import numpy as np

from .base import PosDataBase

__all__ = ['Flash']


@dataclass(eq=False)
class Flash(PosDataBase):
    """Optical flash information.

    Attributes
    ----------
    id : int
        Index of the flash in the list
    volume_id : int
        Index of the optical volume in which the flahs was recorded
    time : float
        Time with respect to the trigger in microseconds
    time_width : float
        Width of the flash in microseconds
    time_abs : float
        Time in units of PMT readout clock
    frame : int
        Frame number
    in_beam_frame : bool
        Whether the flash is in the beam frame
    on_beam_time : bool
        Whether the flash time is consistent with the beam window
    total_pe : float
        Total number of PE in the flash
    fast_to_total : float
        Fraction of the total PE contributed by the fast component
    pe_per_optdet : np.ndarray
        (N) Fixed-length array of the number of PE per optical detector
    center : np.ndarray
        Barycenter of the flash in detector coordinates
    width : np.ndarray
        Spatial width of the flash in detector coordinates
    units : str
        Units in which the position coordinates are expressed
    """
    id: int = -1
    volume_id: int = -1
    frame: int = -1
    in_beam_frame: bool = False
    on_beam_time: bool = False
    time: float = -1.0
    time_width: float = -1.0
    time_abs: float = -1.0
    total_pe: float = -1.0
    fast_to_total: float = -1.0
    pe_per_ch: np.ndarray = None
    center: np.ndarray = None
    width: np.ndarray = None
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = (('center', 3), ('width', 3))

    # Variable-length attributes
    _var_length_attrs = (('pe_per_ch', np.float32),)

    # Attributes specifying coordinates
    _pos_attrs = ('center',)

    # Attributes specifying vector components
    _vec_attrs = ('width',)

    @classmethod
    def from_larcv(cls, flash):
        """Builds and returns a Flash object from a LArCV Flash object.

        Parameters
        ----------
        flash : larcv.Flash
            LArCV-format optical flash

        Returns
        -------
        Flash
            Flash object
        """
        # Get the physical center and width of the flash
        axes = ('x', 'y', 'z')
        center = np.array([getattr(flash, f'{a}Center')() for a in axes])
        width = np.array([getattr(flash, f'{a}Width')() for a in axes])

        # Get the number of PEs per optical channel
        pe_per_ch = np.array(list(flash.PEPerOpDet()), dtype=np.float32)

        # Get the volume ID, if it is filled (TODO: simplify with update)
        volume_id = -1
        for attr in ('tpc', 'volume_id'):
            if hasattr(flash, attr):
                volume_id = getattr(flash, attr)()

        return cls(id=flash.id(), volume_id=volume_id, frame=flash.frame(),
                   in_beam_frame=flash.inBeamFrame(),
                   on_beam_time=flash.onBeamTime(), time=flash.time(),
                   time_abs=flash.absTime(), time_width=flash.timeWidth(),
                   total_pe=flash.TotalPE(), pe_per_ch=pe_per_ch,
                   center=center, width=width)
    
    def merge(self, other, time_method='min'):
        """Merge two flashes.
        
        Parameters
        ----------
        other : Flash
            Flash to merge with
        time_method : str, default 'min'
            Method to use to merge the times. Options are 'min' or 'mean'.
        """
        assert self.units == other.units, f'Flash units are not the same: {self.units} != {other.units}'


        #Merge the times
        if time_method == 'min':
            new_time = min(self.time, other.time)
            # For time_width, take the span covering both flashes
            end_time_self = self.time + self.time_width
            end_time_other = other.time + other.time_width
            new_time_width = max(end_time_self, end_time_other) - new_time
            new_time_abs = min(self.time_abs, other.time_abs)

            # Keep timing information of the earliest flash
            new_in_beam_frame = bool(self.in_beam_frame if new_time == self.time else other.in_beam_frame)
            new_on_beam_time = bool(self.on_beam_time if new_time == self.time else other.on_beam_time)
            new_frame = int(self.frame if new_time == self.time else other.frame)
        elif time_method == 'mean':
            # Weighted mean by total_pe might be more appropriate if available and > 0
            pe_self = self.total_pe if self.total_pe > 0 else 1.0
            pe_other = other.total_pe if other.total_pe > 0 else 1.0
            total_pe_sum = pe_self + pe_other
            if total_pe_sum > 0:
                new_time = (self.time * pe_self + other.time * pe_other) / total_pe_sum
                new_time_abs = (self.time_abs * pe_self + other.time_abs * pe_other) / total_pe_sum
                # Time width average might not be physically meaningful, consider 'min' logic?
                new_time_width = (self.time_width * pe_self + other.time_width * pe_other) / total_pe_sum
            else: # Avoid division by zero if both PEs are zero/negative
                new_time = (self.time + other.time) / 2.0
                new_time_abs = (self.time_abs + other.time_abs) / 2.0
                new_time_width = (self.time_width + other.time_width) / 2.0

            # TODO: Handle bools/frame in 'mean' mode
            # How to merge bools/frame in 'mean' mode? Taking 'self' as default for now.
            # Or perhaps logical OR for bools? E.g., new_in_beam_frame = bool(self.in_beam_frame or other.in_beam_frame)
            new_in_beam_frame = bool(self.in_beam_frame)
            new_on_beam_time = bool(self.on_beam_time)
            new_frame = int(self.frame) # Or perhaps max(self.frame, other.frame)? Needs definition.
        else:
            raise ValueError(f'Invalid time method: {time_method}')

        # Assign merged time values, casting explicitly
        self.time = float(new_time)
        self.time_width = float(new_time_width)
        self.time_abs = float(new_time_abs)
        self.frame = int(new_frame)
        self.in_beam_frame = bool(new_in_beam_frame)
        self.on_beam_time = bool(new_on_beam_time)

        # --- Handle Spatial and PE Merging ---
        #Find the PE weighted center using the arithmetic mean of the centers
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean

        # Calculate weights safely, avoiding division by zero or using negative widths
        # Use total_pe as weights if widths are unreliable? For now, stick to width.
        weight_self = np.zeros_like(self.width, dtype=float)
        weight_other = np.zeros_like(other.width, dtype=float)

        valid_width_self = self.width > 1e-6 # Check for small positive width
        valid_width_other = other.width > 1e-6

        weight_self[valid_width_self] = 1.0 / (self.width[valid_width_self]**2)
        weight_other[valid_width_other] = 1.0 / (other.width[valid_width_other]**2)

        total_weight = weight_self + weight_other

        # Merge center (Weighted Average)
        new_center = self.center.copy() # Start with self's center
        valid_weights = total_weight > 1e-9 # Where total weight is non-zero
        if np.any(valid_weights):
             new_center[valid_weights] = (self.center[valid_weights] * weight_self[valid_weights] + \
                                          other.center[valid_weights] * weight_other[valid_weights]) / total_weight[valid_weights]
        # Handle cases where total_weight is zero (e.g., both widths were zero)
        # We might take the average, or keep self's value (as done by initializing with self.center.copy())
        # Or if only one had valid width, that one's center should dominate (handled implicitly above)
        self.center = new_center.astype(np.float64) # Ensure final type

        # Merge width (Combine variances)
        new_width = self.width.copy() # Start with self's width
        if np.any(valid_weights):
            new_width[valid_weights] = np.sqrt(1.0 / total_weight[valid_weights])
        # Keep original width where weights were invalid
        self.width = new_width.astype(np.float64) # Ensure final type

        # Merge fast_to_total (Weighted Average)
        # Need a scalar weight sum for this. Let's average the component weights? Or use total_pe?
        # Using average spatial weights for now:
        scalar_weight_self = np.mean(weight_self[valid_width_self]) if np.any(valid_width_self) else 0.0
        scalar_weight_other = np.mean(weight_other[valid_width_other]) if np.any(valid_width_other) else 0.0
        scalar_total_weight = scalar_weight_self + scalar_weight_other

        if scalar_total_weight > 1e-9 and isinstance(self.fast_to_total, (int, float)) and isinstance(other.fast_to_total, (int, float)):
             new_fast_to_total = (self.fast_to_total * scalar_weight_self + other.fast_to_total * scalar_weight_other) / scalar_total_weight
             self.fast_to_total = float(new_fast_to_total)
        elif isinstance(self.fast_to_total, (int, float)): # If other is invalid/non-numeric, keep self
             self.fast_to_total = float(self.fast_to_total)
        elif isinstance(other.fast_to_total, (int, float)): # If self is invalid/non-numeric, take other
             self.fast_to_total = float(other.fast_to_total)
        else: # Both invalid, keep default? Or set to a specific value like -1?
             self.fast_to_total = -1.0 # Assign a default float value

        # Merge total_pe (Sum)
        self.total_pe = float(self.total_pe + other.total_pe)

        # --- Handle PE per Channel Merging ---
        pe_self = self.pe_per_ch
        pe_other = other.pe_per_ch

        # Handle cases where one or both might be None
        if pe_self is None and pe_other is None:
            self.pe_per_ch = None # Or np.array([], dtype=np.float32) ? Depends on desired behavior.
            return # Nothing more to do for pe_per_ch
        elif pe_self is None:
            self.pe_per_ch = pe_other.astype(np.float32) # Take other's, ensure dtype
            return
        elif pe_other is None:
            self.pe_per_ch = pe_self.astype(np.float32) # Keep self's, ensure dtype
            return

        # Both are arrays, proceed with padding and adding
        len_self = len(pe_self)
        len_other = len(pe_other)

        if len_self != len_other:
            if len_self < len_other:
                num_to_pad = len_other - len_self
                pe_self_padded = np.pad(pe_self, (0, num_to_pad), mode='constant', constant_values=0)
                pe_other_final = pe_other
            else: # len_other < len_self
                num_to_pad = len_self - len_other
                pe_other_padded = np.pad(pe_other, (0, num_to_pad), mode='constant', constant_values=0)
                pe_self_padded = pe_self
                pe_other_final = pe_other_padded # Use the padded version
        else:
             # Lengths are equal
             pe_self_padded = pe_self
             pe_other_final = pe_other


        # Perform addition and ensure final dtype is float32
        self.pe_per_ch = (pe_self_padded + pe_other_final).astype(np.float32)
