"""Module with a data class object which represents true neutrino information.

This copies the internal structure of :class:`larcv.Neutrino`.
"""

from dataclasses import dataclass, field
from typing import Any
from warnings import warn

import numpy as np

from spine.constants import (
    GenieNuInteractionType,
    LArSoftNuInteractionType,
    NuCurrentType,
    NuInteractionScheme,
)
from spine.data.base import PosDataBase
from spine.data.field import FieldMetadata

__all__ = ["Neutrino"]


@dataclass(eq=False, repr=False)
class Neutrino(PosDataBase):
    """Neutrino truth information.

    Attributes
    ----------
    id : int
        Index of the neutrino in the list
    interaction_id : int
        Index of the neutrino at the generator stage (e.g. Genie)
    mct_index : int
        Index in the original MCTruth array from whence it came
    track_id : int
        Geant4 track ID of the neutrino
    lepton_track_id : int
        Geant4 track ID of the lepton (if CC)
    pdg_code : int
        PDG code of the neutrino
    lepton_pdg_code : int
        PDG code of the outgoing lepton
    current_type : int
        Enumerated current type of the neutrino interaction
    interaction_scheme : int
        Enumerated scheme used to interpret the interaction mode/type codes
    interaction_mode : int
        Source-native neutrino interaction mode code
    interaction_type : int
        Source-native neutrino interaction type code
    target : int
        PDG code of the target object
    nucleon : int
        PDG code of the target nucleon (if QE)
    quark : int
        PDG code of the target quark (if DIS)
    energy_init : float
        Energy of the neutrino at its interaction point in GeV
    hadronic_invariant_mass : float
        Hadronic invariant mass (W) in GeV/c^2
    bjorken_x : float
        Bjorken scaling factor (x)
    inelasticity : float
        Inelasticity (y)
    momentum_transfer : float
        Squared momentum transfer (Q^2) in (GeV/c)^2
    momentum_transfer_mag : float
        Magnitude of the momentum transfer (Q3) in GeV/c
    energy_transfer : float
        Energy transfer (Q0) in GeV
    lepton_p : float
        Absolute momentum of the lepton
    distance_travel : float
        True amount of distance traveled by the neutrino before interacting
    theta : float
        Angle between incoming and outgoing leptons in radians
    t : float
        Interaction time (ns)
    creation_process : str
        Creation process of the neutrino
    position : np.ndarray
        Location of the neutrino interaction
    momentum : np.ndarray
        3-momentum of the neutrino at its interaction point
    units : str
        Units in which the position coordinates are expressed
    """

    # Index attributes
    id: int = field(default=-1, metadata=FieldMetadata(index=True))
    interaction_id: int = field(default=-1, metadata=FieldMetadata(index=True))

    # Enumerated attributes
    current_type: int = field(default=-1, metadata=FieldMetadata(enum=NuCurrentType))
    interaction_scheme: int = field(
        default=-1, metadata=FieldMetadata(enum=NuInteractionScheme)
    )
    interaction_mode: int = -1
    interaction_type: int = -1

    # Scalar attributes
    mct_index: int = -1
    track_id: int = -1
    lepton_track_id: int = -1
    pdg_code: int = -1
    lepton_pdg_code: int = -1
    target: int = -1
    nucleon: int = -1
    quark: int = -1

    energy_init: float = field(default=np.nan, metadata=FieldMetadata(units="GeV"))
    hadronic_invariant_mass: float = field(
        default=np.nan, metadata=FieldMetadata(units="GeV/c^2")
    )
    momentum_transfer: float = field(
        default=np.nan, metadata=FieldMetadata(units="(GeV/c)^2")
    )
    momentum_transfer_mag: float = field(
        default=np.nan, metadata=FieldMetadata(units="GeV/c")
    )
    energy_transfer: float = field(default=np.nan, metadata=FieldMetadata(units="GeV"))
    lepton_p: float = field(default=np.nan, metadata=FieldMetadata(units="GeV/c"))
    distance_travel: float = field(
        default=np.nan, metadata=FieldMetadata(units="instance")
    )
    t: float = field(default=np.nan, metadata=FieldMetadata(units="ns"))
    theta: float = field(default=np.nan, metadata=FieldMetadata(units="rad"))
    bjorken_x: float = np.nan
    inelasticity: float = np.nan

    creation_process: str = ""
    units: str = "cm"

    # Vector attributes
    position: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(
            length=3,
            dtype=np.float32,
            position=True,
            units="instance",
        ),
    )

    momentum: np.ndarray = field(
        default_factory=lambda: np.full(3, np.nan, dtype=np.float32),
        metadata=FieldMetadata(length=3, dtype=np.float32, vector=True, units="MeV/c"),
    )

    @property
    def interaction_mode_enum(
        self,
    ) -> LArSoftNuInteractionType | GenieNuInteractionType | None:
        """Interpret the interaction mode under the stored interaction scheme."""
        return self._resolve_interaction_enum(self.interaction_mode)

    @property
    def interaction_type_enum(
        self,
    ) -> LArSoftNuInteractionType | GenieNuInteractionType | None:
        """Interpret the interaction type under the stored interaction scheme."""
        return self._resolve_interaction_enum(self.interaction_type)

    def _resolve_interaction_enum(
        self, value: int
    ) -> LArSoftNuInteractionType | GenieNuInteractionType | None:
        """Resolve a raw interaction code to the appropriate source enum."""
        scheme_to_enum = {
            int(NuInteractionScheme.LARSOFT): LArSoftNuInteractionType,
            int(NuInteractionScheme.GENIE): GenieNuInteractionType,
        }
        enum_type = scheme_to_enum.get(self.interaction_scheme)
        if enum_type is None:
            return None

        try:
            return enum_type(value)
        except ValueError:
            return None

    @classmethod
    def from_larcv(
        cls,
        neutrino,
        interaction_scheme: int | NuInteractionScheme = NuInteractionScheme.LARSOFT,
    ) -> "Neutrino":
        """Builds and returns a Neutrino object from a LArCV Neutrino object.

        Parameters
        ----------
        neutrino : larcv.Neutrino
            LArCV-format neutrino object
        interaction_scheme : int or NuInteractionScheme, default LARSOFT
            Convention used to interpret the interaction mode/type codes.

        Returns
        -------
        Neutrino
            Neutrino object
        """
        # Initialize the dictionary to initialize the object with
        obj_dict: dict[str, Any] = {"interaction_scheme": int(interaction_scheme)}

        # Load the scalar attributes
        for key in (
            "id",
            "interaction_id",
            "mct_index",
            "nu_track_id",
            "lepton_track_id",
            "pdg_code",
            "lepton_pdg_code",
            "current_type",
            "interaction_mode",
            "interaction_type",
            "target",
            "nucleon",
            "quark",
            "energy_init",
            "hadronic_invariant_mass",
            "bjorken_x",
            "inelasticity",
            "momentum_transfer",
            "momentum_transfer_mag",
            "energy_transfer",
            "lepton_p",
            "distance_travel",
            "theta",
            "creation_process",
        ):
            # Backwards compatibility: some older LArCV versions may be missing
            # some of these attributes, warn if that's the case and skip them
            if not hasattr(neutrino, key):
                warn(
                    f"The LArCV Neutrino object is missing the {key} "
                    "attribute. It will miss from the Neutrino object."
                )
                continue

            # Backwards compatibility: renamed "nu_track_id" to "track_id"
            if key != "nu_track_id":
                obj_dict[key] = getattr(neutrino, key)()
            else:
                obj_dict["track_id"] = getattr(neutrino, key)()

        obj_dict["t"] = neutrino.position().t()

        # Load the positional attribute
        obj_dict["position"] = np.asarray(
            [neutrino.position().x(), neutrino.position().y(), neutrino.position().z()],
            dtype=np.float32,
        )

        # Load the momentum attribute (special care needed)
        if not hasattr(neutrino, "momentum"):
            warn(
                "The LArCV Neutrino object is missing the momentum "
                "attribute. It will miss from the Neutrino object."
            )
        else:
            obj_dict["momentum"] = np.asarray(
                [neutrino.px(), neutrino.py(), neutrino.pz()], dtype=np.float32
            )

        return cls(**obj_dict)
