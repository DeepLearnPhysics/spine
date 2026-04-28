"""Module with a data class object which represents true particle information.

This copies the internal structure of :class:`larcv.Particle`.
"""

from dataclasses import dataclass, field
from warnings import warn

import numpy as np

from spine.constants import PID_LABELS, SHAPE_LABELS, ParticlePID, ParticleShape
from spine.data.base import PosDataBase
from spine.data.decorator import stored_property
from spine.data.field import FieldMetadata

__all__ = ["Particle"]


@dataclass(eq=False, repr=False)
class Particle(PosDataBase):
    """Particle truth information.

    Attributes
    ----------
    id : int
        Index of the particle in the list
    mct_index : int
        Index in the original MCTruth array from whence it came
    mcst_index : int
        Index in the original MCTrack/MCShower array from whence it came
    group_id : int
        Index of the group the particle belongs to
    interaction_id : int
        Index of the interaction the partile belongs to
    nu_id : int
        Index of the neutrino this particle belongs to
    interaction_primary : int
        Whether the particle is primary in its interaction or not
    group_primary : int
        Whether this particle is primary in its group or not
    parent_id : int
        Index of the parent particle
    children_id : np.ndarray
        List of indexes of the children particles
    track_id : int
        Geant4 track ID
    parent_track_id : int
        Geant4 track ID of the parent particle
    ancestor_track_id : int
        Geant4 track ID of the ancestor particle
    shape : int
        Enumerated semantic type of the particle
    num_voxels : int
        Number of voxels matched to this particle instance
    energy_init : float
        True initial energy in MeV
    energy_deposit : float
        Amount of energy matched to this particle instance in MeV
    distance_travel : float
        True amount of distance traveled by the particle in the active volume
    creation_process : str
        Creation process
    parent_creation_process : str
        Creation process of the parent particle
    ancestor_creation_process : str
        Creation process of the ancestor particle
    pid : int
        Enumerated particle species type of the particle
    pdg_code : int
        Particle PDG code
    parent_pdg_code : int
        Particle PDG code of the parent particle
    ancestor_pdg_code : int
        Particle PDG code of the ancestor particle
    t : float
        Particle creation time (ns)
    end_t : float
        Particle death time (ns)
    parent_t : float
        Particle creation time of the parent particle (ns)
    ancestor_t : float
        Particle creation time of the ancestor particle (ns)
    position : np.ndarray
        Location of the creation point of the particle
    end_position : np.ndarray
        Location where the particle stopped
    parent_position : np.ndarry
        Location of the creation point of the parent particle
    ancestor_position : np.ndarray
        Location of the creation point of the ancestor particle
    first_step : np.ndarray
        Location of the first energy deposition of the particle
    last_step : np.ndarray
        Location of the last energy deposition of the particle
    momentum : np.ndarray
        3-momentum of the particle at the production point
    end_momentum : np.ndarray
        3-momentum of the particle at where it stops or exits the detector
    p : float
        Momentum magnitude of the particle at the production point
    end_p : float
        Momentum magnitude of the particle where it stops or exits the detector
    mass : float
        Rest mass of the particle in MeV/c^2
    units : str
        Units in which the position attributes are expressed
    """

    # Index attributes
    id: int = field(default=-1, metadata=FieldMetadata(index=True))
    parent_id: int = field(default=-1, metadata=FieldMetadata(index=True))
    group_id: int = field(default=-1, metadata=FieldMetadata(index=True))
    interaction_id: int = field(default=-1, metadata=FieldMetadata(index=True))
    nu_id: int = field(default=-1, metadata=FieldMetadata(index=True))
    children_id: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32),
        metadata=FieldMetadata(dtype=np.int32, index=True),
    )

    # Scalar attributes
    mct_index: int = -1
    mcst_index: int = -1
    group_primary: int = -1
    interaction_primary: int = -1
    track_id: int = -1
    parent_track_id: int = -1
    ancestor_track_id: int = -1
    pdg_code: int = -1
    parent_pdg_code: int = -1
    ancestor_pdg_code: int = -1
    num_voxels: int = -1

    t: float = field(default=np.nan, metadata=FieldMetadata(units="ns"))
    end_t: float = field(default=np.nan, metadata=FieldMetadata(units="ns"))
    parent_t: float = field(default=np.nan, metadata=FieldMetadata(units="ns"))
    ancestor_t: float = field(default=np.nan, metadata=FieldMetadata(units="ns"))
    energy_init: float = field(default=np.nan, metadata=FieldMetadata(units="MeV"))
    energy_deposit: float = field(default=np.nan, metadata=FieldMetadata(units="MeV"))
    distance_travel: float = field(default=np.nan, metadata=FieldMetadata(units="cm"))

    creation_process: str = ""
    parent_creation_process: str = ""
    ancestor_creation_process: str = ""
    units: str = "cm"

    # Enumerated attributes
    shape: int = field(default=-1, metadata=FieldMetadata(enum=ParticleShape))
    pid: int = field(default=-1, metadata=FieldMetadata(enum=ParticlePID))

    # Vector attributes
    position: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, position=True, units="instance"
        ),
    )
    end_position: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, position=True, units="instance"
        ),
    )
    parent_position: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, position=True, units="instance"
        ),
    )
    ancestor_position: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, position=True, units="instance"
        ),
    )
    first_step: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, position=True, units="instance"
        ),
    )
    last_step: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3, dtype=np.float32, position=True, units="instance"
        ),
    )
    momentum: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True, units="MeV/c"),
    )
    end_momentum: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True, units="MeV/c"),
    )

    @property
    @stored_property(units="MeV/c")
    def p(self) -> float:
        """Computes the magnitude of the initial momentum.

        Returns
        -------
        float
            Norm of the initial momentum vector
        """
        return float(np.linalg.norm(self.momentum))

    @property
    @stored_property(units="MeV/c")
    def end_p(self) -> float:
        """Computes the magnitude of the final momentum.

        Returns
        -------
        float
            Norm of the final momentum vector
        """
        return float(np.linalg.norm(self.end_momentum))

    @property
    @stored_property(units="MeV/c^2")
    def mass(self) -> float:
        """Computes the rest mass of the particle from its energy/momentum.

        Returns
        -------
        float
            Rest mass of the particle in MeV/c^2
        """
        if np.isnan(self.energy_init) or np.isnan(self.momentum).any():
            return np.nan

        return np.sqrt(max(0.0, self.energy_init**2 - np.sum(self.momentum**2)))

    @classmethod
    def from_larcv(cls, particle) -> "Particle":
        """Builds and returns a Particle object from a LArCV Particle object.

        Parameters
        ----------
        particle : larcv.Particle
            LArCV-format particle object

        Returns
        -------
        Particle
            Particle object
        """
        # Initialize the dictionary to initialize the object with
        obj_dict = {}

        # Load the scalar attributes
        for prefix in ("", "parent_", "ancestor_"):
            for key in ("track_id", "pdg_code", "creation_process", "t"):
                obj_dict[prefix + key] = getattr(particle, prefix + key)()
        for key in (
            "id",
            "group_id",
            "interaction_id",
            "parent_id",
            "mct_index",
            "mcst_index",
            "num_voxels",
            "shape",
            "energy_init",
            "energy_deposit",
            "distance_travel",
        ):
            if not hasattr(particle, key):
                warn(
                    f"The LArCV Particle object is missing the {key} "
                    "attribute. It will miss from the Particle object."
                )
                continue
            obj_dict[key] = getattr(particle, key)()

        obj_dict["end_t"] = particle.end_position().t()

        # Load the positional attribute
        axes = ("x", "y", "z")
        for key in (
            "position",
            "end_position",
            "parent_position",
            "ancestor_position",
            "first_step",
            "last_step",
        ):
            vector = getattr(particle, key)()
            obj_dict[key] = np.asarray(
                [getattr(vector, a)() for a in axes], dtype=np.float32
            )

        # Load the other array attributes (special care needed). Note for future
        # self: need the list comprehension. Direct casting is INSANELY slow...
        obj_dict["children_id"] = np.asarray(
            [i for i in particle.children_id()], dtype=np.int32
        )

        mom_attrs = ("px", "py", "pz")
        for prefix in ("", "end_"):
            key = prefix + "momentum"
            if not hasattr(particle, key):
                warn(
                    f"The LArCV Particle object is missing the {key} "
                    "attribute. It will miss from the Particle object."
                )
                continue
            obj_dict[key] = np.asarray(
                [getattr(particle, prefix + a)() for a in mom_attrs], dtype=np.float32
            )

        return cls(**obj_dict)
