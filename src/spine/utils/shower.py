"""Shower physics utilities."""

import math
from typing import Tuple, Union

import numba as nb
import numpy as np
from scipy.special import gammaincinv  # pylint: disable=E0611

from .globals import LAR_E_CRIT, LAR_X0, LAR_Z


@nb.njit(cache=True)
def shower_long_profile(
    depth: Union[float, np.ndarray], energy: float, use_gp: bool = False
) -> Union[float, np.ndarray]:
    """Compute the longitudinal shower profile using the Gamma distribution.

    Parameters
    ----------
    depth : Union[float, np.ndarray]
        Depths at which to compute the profile in radiation lengths.
    energy : float
        Energy of the shower in MeV.
    use_gp : bool, default False
        Whether to use the Grindhammer and Peters (2000) parametrization.

    Returns
    -------
    Union[float, np.ndarray]
        The longitudinal shower profile at the specified depths.
    """
    # Parametrize the longitudinal profile in units of radiation length
    t = depth / LAR_X0

    # Compute the shape (a) and scale (b) parameters of the Gamma distribution
    if use_gp:
        a, b = shower_long_params_gp(energy)
    else:
        a, b = shower_long_params_lar(energy)

    # Compute the Gamma distribution profile in the depth(s) requested
    coeff = energy * b / math.gamma(a)
    return coeff * (b * t) ** (a - 1) * np.exp(-b * t)


def shower_long_quantile(energy: float, quantile: float, use_gp: bool = False) -> float:
    """Compute the depth corresponding to a given quantile of the longitudinal shower profile.

    Parameters
    ----------
    energy : float
        Energy of the shower in MeV.
    quantile : float
        Quantile to compute (between 0 and 1).
    use_gp : bool, default False
        Whether to use the Grindhammer and Peters (2000) parametrization.

    Returns
    -------
    float
        Depth corresponding to the specified quantile in cm.
    """
    # Compute the shape (a) and scale (b) parameters of the Gamma distribution
    if use_gp:
        a, b = shower_long_params_gp(energy)
    else:
        a, b = shower_long_params_lar(energy)

    # Use the inverse of the regularized incomplete gamma function to find t
    t = gammaincinv(a, quantile) / b  # in units of radiation length

    return t * LAR_X0  # convert to cm


@nb.njit(cache=True)
def shower_long_mode_gp(energy: float) -> float:
    """Compute the depth of the shower maximum, according to Grindhammer and Peters (2000).

    Parameters
    ----------
    energy : float
        Energy of the shower in MeV.

    Returns
    -------
    float
        Depth of the shower maximum in cm.

    Notes
    -----
    Based on the parametrization from Grindhammer and Peters (2000).
    Source: https://arxiv.org/pdf/hep-ex/0001020
    """
    y = energy / LAR_E_CRIT
    t_max = np.log(y) - 0.858  # in units of radiation length

    return t_max * LAR_X0  # convert to cm


@nb.njit(cache=True)
def shower_long_maximum_lar(energy: float) -> float:
    """Compute the depth of the shower maximum.

    Parameters
    ----------
    energy : float
        Energy of the shower in MeV.

    Returns
    -------
    float
        Depth of the shower maximum in cm.

    Notes
    -----
    Based on a fit done to an edep-sim simulation of single electrons in LAr.
    """
    y = energy / LAR_E_CRIT
    t_max = 0.689 * np.log(1 + 0.305 * y**1.68)

    return t_max * LAR_X0  # convert to cm


@nb.njit(cache=True)
def shower_long_params_gp(energy: float) -> Tuple[float, float]:
    """Compute the longitudinal shower profile parameters a and b according
    to Grindhammer and Peters (2000).

    Parameters
    ----------
    energy : float
        Energy of the shower in MeV.

    Returns
    -------
    Tuple[float, float]
        Tuple containing shape parameter a and scale parameter b.

    Notes
    -----
    Based on the parametrization from Grindhammer and Peters (2000).
    Source: https://arxiv.org/pdf/hep-ex/0001020
    """
    y = energy / LAR_E_CRIT
    t_max = np.log(y) - 0.858
    a = 0.21 + (0.492 + 2.38 / LAR_Z) * np.log(y)
    b = (a - 1) / t_max

    return a, b


@nb.njit(cache=True)
def shower_long_params_lar(energy: float) -> Tuple[float, float]:
    """Compute the longitudinal shower profile parameters a and b according
    to a general log-based parametrization of the form:

    t_max = B_t * log(1 + B_t*(E / E_crit)**gamma_t) for t
    a = 1 + B_a * log(1 + B_a(E / E_crit)**gamma_a) for a

    Parameters
    ----------
    energy : float
        Energy of the shower in MeV.

    Returns
    -------
    Tuple[float, float]
        Tuple containing shape parameter a and scale parameter b.

    Notes
    -----
    Based on a fit done to an edep-sim simulation of single electrons in LAr.
    """
    y = energy / LAR_E_CRIT
    t_max = 0.689 * np.log(1 + 0.305 * y**1.68)
    a = 1.4 + 0.083 * np.log(1 + 0.21 * y**7.83)
    b = (a - 1) / t_max

    return a, b


@nb.njit(cache=True)
def shower_angle_profile(
    angle: Union[float, np.ndarray], energy: float
) -> Union[float, np.ndarray]:
    """Compute the angular shower profile using a Stacy parametrization.

    Parameters
    ----------
    angle : Union[float, np.ndarray]
        Angles at which to compute the profile in radians.
    energy : float
        Energy of the shower in MeV.

    Returns
    -------
    Union[float, np.ndarray]
        The angular shower profile at the specified angles.
    """
    # Compute the Stacy distribution parameters a, b and c
    a, b, c = shower_angle_params(energy)

    # Compute the Stacy distribution profile at the angle(s) requested
    coeff = c / (b**a * math.gamma(a / c))
    return coeff * angle ** (a - 1) * np.exp(-((angle / b) ** c))


def shower_angle_mode(energy: float) -> float:
    """Compute the mode of the angular shower profile.

    Parameters
    ----------
    energy : float
        Energy of the shower in MeV.

    Returns
    -------
    float
        Mode of the angular shower profile in radians.
    """
    # Compute the Stacy distribution parameters a, b and c
    a, b, c = shower_angle_params(energy)

    # Compute the mode of the distribution
    mode = b * ((a - 1) / c) ** (1 / c)

    return mode  # in radians


def shower_angle_quantile(energy: float, quantile: float) -> float:
    """Compute the angle corresponding to a given quantile of the angular shower profile.

    Parameters
    ----------
    energy : float
        Energy of the shower in MeV.
    quantile : float
        Quantile to compute (between 0 and 1).

    Returns
    -------
    float
        Angle corresponding to the specified quantile in radians.
    """
    # Compute the Stacy distribution parameters a, b and c
    a, b, c = shower_angle_params(energy)

    # Use the inverse of the regularized incomplete gamma function to find angle
    angle = b * (gammaincinv(a / c, quantile)) ** (1 / c)

    return angle  # in radians


@nb.njit(cache=True)
def shower_angle_params(energy: float) -> Tuple[float, float, float]:
    """Compute the shower angle parameters a, b and c according to a Stacy
    parametrization of the angular distribution.

    Source: https://arxiv.org/pdf/2303.10226

    Parameters
    ----------
    energy : float
        Energy of the shower in MeV.

    Returns
    -------
    float
        Stacy distribution parameter a.
    float
        Stacy distribution parameter b.
    float
        Stacy distribution parameter c.

    Notes
    -----
    Based on a simple parametrization.
    """
    # Simple parametrization of shower angle distribution parameters
    # t params [ 0.00282092  0.19161785  2.13132419 -0.7097491 ]
    # a params [1.48734863 0.09577318 1.53568148 2.62746278]
    # c params [ 1.21861046 -0.01702748  8.02936571  8.91356063]
    y = energy / LAR_E_CRIT
    t_max = 0.192 * np.log(1 + 2.131 * y**-0.71)
    a = 1.49 + 0.0958 * np.log(1 + 1.53 * y**2.63)
    c = 1.22 - 0.017 * np.log(1 + 8.05 * y**8.94)
    b = t_max * ((a - 1) / c) ** (-1 / c)

    return a, b, c
