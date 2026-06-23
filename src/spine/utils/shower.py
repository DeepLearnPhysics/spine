"""Shower physics utilities."""

import math
from typing import List, Optional, Tuple, Union

import numba as nb
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaincinv  # pylint: disable=E0611

from spine.constants import LAR_E_CRIT, LAR_RM, LAR_X0, LAR_Z


class ShowerEnergyFitter:
    """Fit shower energy from box-wise deposited energy observations.

    This fitter performs forward regression of the shower energy by:

    1. Sampling 3D points from a parametric shower model at a trial energy
    2. Counting the sampled points falling inside each box
    3. Converting those counts into predicted box energies
    4. Comparing predicted and reconstructed box energies with a chi-square

    The uncertainty model is based on Poisson statistics of the sampled point
    counts in each box, with an additional additive uncertainty floor.

    Parameters
    ----------
    boundaries : List[NDArray[np.float64]], shape (n_boxes, 3, 2)
        Axis-aligned box boundaries. For each box:
            boundaries[i, 0] = [x_min, x_max]
            boundaries[i, 1] = [y_min, y_max]
            boundaries[i, 2] = [z_min, z_max]
    n_points : int, default=100000
        Number of shower points to sample for each trial energy.
    sigma_floor : float, default=1.0
        Additive per-box uncertainty floor in MeV, added in quadrature to
        the Monte Carlo sampling uncertainty.
    seed : int, default=12345
        Seed used to initialize the random number generator. Keeping this
        fixed across evaluations makes the fit objective smoother.

    Notes
    -----
    For a trial energy ``E`` and sampled counts ``n_i`` in each box out of
    ``N`` total points, the predicted box energy is estimated as

        E_i = E * n_i / N

    and its Monte Carlo variance is approximated as

        var(E_i) ~= (E / N)^2 * n_i.

    This class stores only the static fitter configuration. Per-shower inputs
    such as the shower start position and direction are provided when calling
    the prediction or fit methods.
    """

    def __init__(
        self,
        boundaries: List[np.ndarray],
        n_points: int = 100000,
        sigma_floor: float = 1.0,
        seed: int = 12345,
        energy_bounds: Tuple[float, float] = (1.0, 10000.0),
        xatol: Optional[float] = 1.0,
        use_gp: bool = False,
    ) -> None:
        """Initialize the shower energy fitter.

        Parameters
        ----------
        boundaries : List[NDArray[np.float64]], shape (n_boxes, 3, 2)
            Axis-aligned box boundaries. For each box:
                boundaries[i, 0] = [x_min, x_max]
                boundaries[i, 1] = [y_min, y_max]
                boundaries[i, 2] = [z_min, z_max]
        n_points : int, default=100000
            Number of shower points to sample for each trial energy.
        sigma_floor : float, default=1.0
            Additive per-box uncertainty floor in MeV, added in quadrature to
            the Monte Carlo sampling uncertainty.
        seed : int, default=12345
            Seed used to initialize the random number generator. Keeping this
            fixed across evaluations makes the fit objective smoother.
        energy_bounds : Tuple[float, float], default=(1.0, 10000.0)
            Lower and upper bounds on the fitted energy, in MeV. These are used
            as the bounds for the scalar minimization in the `fit` method.
        xatol : Optional[float], default=1.0
            Absolute tolerance on the fitted energy, in MeV, passed to
            `scipy.optimize.minimize_scalar`. If not provided, SciPy uses
            its default tolerance.
        use_gp : bool, default False
            Whether to use the Grindhammer and Peters (2000) parametrization for the
            longitudinal profile. If False, a custom log-based parametrization is
            used instead.
        """
        self.boundaries = np.asarray(boundaries, dtype=np.float64)
        self.n_points = int(n_points)
        self.sigma_floor = float(sigma_floor)
        self.seed = int(seed)
        self.energy_bounds = energy_bounds
        self.xatol = xatol
        self.use_gp = bool(use_gp)

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate fitter inputs."""
        # Boundaries should have shape (n_boxes, 3, 2)
        if self.boundaries.ndim != 3 or self.boundaries.shape[1:] != (3, 2):
            raise ValueError("`boundaries` must have shape (n_boxes, 3, 2).")

        # The number of sampled points must be positive
        if self.n_points <= 0:
            raise ValueError("`n_points` must be strictly positive.")

        # The uncertainty floor must be non-negative
        if self.sigma_floor < 0.0:
            raise ValueError("`sigma_floor` must be non-negative.")

        # Box boundaries must satisfy min < max in each dimension
        widths = self.boundaries[:, :, 1] - self.boundaries[:, :, 0]
        if np.any(widths <= 0.0):
            raise ValueError("All box widths must be strictly positive.")

        # Energy bounds must satisfy 0 < min < max
        e_min, e_max = self.energy_bounds
        if e_min <= 0.0 or e_max <= 0.0 or e_max <= e_min:
            raise ValueError("`energy_bounds` must satisfy 0 < bounds[0] < bounds[1].")

        # When using the GP parametrization, the longitudinal profile breaks down at
        # low energies, so we require the lower energy bound to be above that scale
        if self.use_gp and e_min < 1000.0:
            raise ValueError(
                "When `use_gp` is True, the parametrization breaks down below ~1000 "
                "MeV. Please set `energy_bounds[0]` to a value >= 1000 MeV."
            )

    @property
    def n_boxes(self) -> int:
        """Number of boxes used by the fitter.

        Returns
        -------
        int
            Number of boxes.
        """
        return len(self.boundaries)

    @staticmethod
    def _validate_shower_inputs(
        shower_start: np.ndarray,
        direction: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate per-shower inputs.

        Parameters
        ----------
        shower_start : NDArray[np.float64], shape (3,)
            Shower start position in cm.
        direction : NDArray[np.float64], shape (3,)
            Shower direction vector. It does not need to be normalized.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            Tuple containing validated and converted shower start and direction.
        """
        shower_start = np.asarray(shower_start, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)

        if shower_start.shape != (3,):
            raise ValueError("`shower_start` must have shape (3,).")

        if direction.shape != (3,):
            raise ValueError("`direction` must have shape (3,).")

        if np.linalg.norm(direction) <= 0.0:
            raise ValueError("`direction` must have non-zero norm.")

        return shower_start, direction

    def sample_points(
        self,
        energy: float,
        shower_start: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        """Sample 3D points from the shower model at a given trial energy.

        Parameters
        ----------
        energy : float
            Trial shower energy in MeV.
        shower_start : NDArray[np.float64], shape (3,)
            Shower start position in cm.
        direction : NDArray[np.float64], shape (3,)
            Shower direction vector. It does not need to be normalized.

        Returns
        -------
        NDArray[np.float64], shape (n_points, 3)
            Sampled 3D points in cm.
        """
        shower_start, direction = self._validate_shower_inputs(shower_start, direction)

        rng = np.random.default_rng(self.seed)

        return sample_shower_points(
            n=self.n_points,
            energy=float(energy),
            shower_start=shower_start,
            direction=direction,
            rng=rng,
            use_gp=self.use_gp,
        )

    def count_points_in_boxes(self, points: np.ndarray) -> np.ndarray:
        """Count sampled points falling inside each box.

        Parameters
        ----------
        points : NDArray[np.float64], shape (n_points, 3)
            Sampled 3D points.

        Returns
        -------
        NDArray[np.int64], shape (n_boxes,)
            Number of sampled points inside each box.
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        counts = np.empty(self.n_boxes, dtype=np.int64)

        for i, bounds in enumerate(self.boundaries):
            counts[i] = np.count_nonzero(
                (x >= bounds[0, 0])
                & (x < bounds[0, 1])
                & (y >= bounds[1, 0])
                & (y < bounds[1, 1])
                & (z >= bounds[2, 0])
                & (z < bounds[2, 1])
            )

        return counts

    def predict_box_energy(
        self,
        energy: float,
        shower_start: np.ndarray,
        direction: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict energy and uncertainty in each box at a trial energy.

        Parameters
        ----------
        energy : float
            Trial shower energy in MeV.
        shower_start : NDArray[np.float64], shape (3,)
            Shower start position in cm.
        direction : NDArray[np.float64], shape (3,)
            Shower direction vector. It does not need to be normalized.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]
            Tuple containing:
            - predicted box energies in MeV
            - predicted per-box uncertainties in MeV
            - raw sampled point counts in each box

        Notes
        -----
        If ``n_i`` is the sampled count in box ``i``, the predicted box energy is

            E_i = E * n_i / N

        where ``E`` is the trial shower energy and ``N`` is the total number
        of sampled points.

        The per-box uncertainty is estimated as

            sigma_i^2 = sigma_floor^2 + (E / N)^2 * n_i.
        """
        energy = float(energy)
        points = self.sample_points(energy, shower_start, direction)
        counts = self.count_points_in_boxes(points)

        pred = energy * counts.astype(np.float64) / float(self.n_points)

        sigma2_mc = (energy / float(self.n_points)) ** 2 * counts.astype(np.float64)
        sigma2 = self.sigma_floor**2 + sigma2_mc
        sigma = np.sqrt(sigma2)

        return pred, sigma, counts

    def chi2(
        self,
        energy: float,
        reco_box_energy: np.ndarray,
        shower_start: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        """Compute the chi-square at a trial shower energy.

        Parameters
        ----------
        energy : float
            Trial shower energy in MeV.
        reco_box_energy : NDArray[np.float64], shape (n_boxes,)
            Reconstructed energy in each box, in MeV.
        shower_start : NDArray[np.float64], shape (3,)
            Shower start position in cm.
        direction : NDArray[np.float64], shape (3,)
            Shower direction vector. It does not need to be normalized.

        Returns
        -------
        float
            Chi-square value comparing reconstructed and predicted box energies.
        """
        reco_box_energy = np.asarray(reco_box_energy, dtype=np.float64)

        if reco_box_energy.shape != (self.n_boxes,):
            raise ValueError(f"`reco_box_energy` must have shape ({self.n_boxes},).")

        pred, sigma, _ = self.predict_box_energy(energy, shower_start, direction)
        resid = reco_box_energy - pred

        return float(np.sum((resid / sigma) ** 2))

    def fit(
        self,
        reco_box_energy: np.ndarray,
        shower_start: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        """Fit the shower energy by minimizing the box-based chi-square.

        Parameters
        ----------
        reco_box_energy : NDArray[np.float64], shape (n_boxes,)
            Reconstructed energy in each box, in MeV.
        shower_start : NDArray[np.float64], shape (3,)
            Shower start position in cm.
        direction : NDArray[np.float64], shape (3,)
            Shower direction vector. It does not need to be normalized.
        xatol : float, optional
            Absolute tolerance on the fitted energy passed to
            `scipy.optimize.minimize_scalar`. If not provided, SciPy uses
            its default tolerance.

        Returns
        -------
        float
            Fitted shower energy in MeV.

        Notes
        -----
        This performs a one-dimensional bounded minimization of `chi2(E)`.
        """
        reco_box_energy = np.asarray(reco_box_energy, dtype=np.float64)

        if reco_box_energy.shape != (self.n_boxes,):
            raise ValueError(f"`reco_box_energy` must have shape ({self.n_boxes},).")

        shower_start, direction = self._validate_shower_inputs(shower_start, direction)

        options = {}
        if self.xatol is not None:
            options["xatol"] = float(self.xatol)

        return minimize_scalar(
            self.chi2,
            bounds=self.energy_bounds,
            method="bounded",
            args=(reco_box_energy, shower_start, direction),
            options=options,
        ).x


def sample_shower_points(
    n: int,
    energy: float,
    shower_start: np.ndarray,
    direction: np.ndarray,
    rng: np.random.Generator,
    use_gp: bool = False,
) -> np.ndarray:
    """Sample 3D points from a factorized shower model.

    Samples points distributed according to:
        rho(l, r) = g(l | E) * f(r | l, E)

    where:
        - g(l | E) is the longitudinal profile (Gamma distribution)
        - f(r | l, E) is the GP transverse radial profile

    The sampling proceeds by:
        1. Sampling longitudinal depth l from the Gamma distribution
        2. Sampling transverse radius r from the GP radial distribution
        3. Sampling azimuthal angle phi uniformly
        4. Converting (l, r, phi) into Cartesian coordinates

    Parameters
    ----------
    n : int
        Number of points to sample.
    energy : float
        Shower energy in MeV.
    shower_start : np.ndarray, shape (3,)
        Shower start position in cm.
    direction : np.ndarray, shape (3,)
        Shower direction vector. Does not need to be normalized.
    rng : np.random.Generator
        NumPy random number generator.
    use_gp : bool, default False
        Whether to use the Grindhammer and Peters (2000) parametrization for the
        longitudinal profile. If False, a custom log-based parametrization is
        used instead.

    Returns
    -------
    np.ndarray
        Sampled 3D points in cm, distributed according to the shower model.

    Notes
    -----
    - The longitudinal sampling uses the Gamma distribution parametrization.
    - The transverse sampling uses the GP (Grindhammer–Peters) radial model.
    - The returned points follow the full 3D shower density up to normalization.

    This can be used to estimate energy deposition in volumes via:
        E_box ≈ E_total * (N_in_box / N_total)
    """
    # Normalize direction
    norm = np.linalg.norm(direction)
    if norm <= 0.0:
        raise ValueError("Direction vector must have non-zero norm.")

    u = direction / norm

    # Build transverse orthonormal basis (v, w)
    if abs(u[2]) < 0.9:
        v = np.cross(u, np.array([0.0, 0.0, 1.0]))
    else:
        v = np.cross(u, np.array([0.0, 1.0, 0.0]))
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    # Longitudinal parameters
    if use_gp:
        a, b = shower_long_params_gp(energy)
    else:
        a, b = shower_long_params_lar(energy)

    # Sample depth (Gamma in radiation lengths → convert to cm)
    t = rng.gamma(shape=a, scale=1.0 / b, size=n)
    l = t * LAR_X0

    # Transverse parameters (vectorized)
    R_core, R_tail, p_core = shower_transverse_params_gp(l, energy)

    # Choose component (core vs tail)
    u_choice = rng.uniform(size=n)
    choose_core = u_choice < p_core
    R = np.where(choose_core, R_core, R_tail)

    # Sample radius from GP CDF inversion
    u_radial = rng.uniform(size=n)
    r = R * np.sqrt(u_radial / (1.0 - u_radial))

    # Sample azimuth
    phi = 2.0 * np.pi * rng.uniform(size=n)

    # Convert to Cartesian coordinates
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    points = (
        shower_start[None, :]
        + l[:, None] * u[None, :]
        + r[:, None] * (cos_phi[:, None] * v[None, :] + sin_phi[:, None] * w[None, :])
    )

    return points


@nb.njit(cache=True)
def shower_energy_density_3d(
    points, shower_start, direction, energy, eps=1e-6, use_gp=False
):
    """Compute the shower volumetric energy density at 3D points.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        3D coordinates of the query points, in cm.
    shower_start : np.ndarray, shape (3,)
        Shower start position, in cm. Defines the origin of the shower frame.
    direction : np.ndarray, shape (3,)
        Shower direction vector. Does not need to be normalized.
    energy : float
        Shower energy, in MeV.
    eps : float, default=1e-6
        Small regularization scale used to avoid division by zero at r = 0.
    use_gp : bool, default=False
        Whether to use the Grindhammer and Peters (2000) parametrization.

    Returns
    -------
    np.ndarray, shape (N,)
        Volumetric shower energy density at each point, in MeV/cm^3.

    Notes
    -----
    The returned density is based on:
        u(r, l) = g(l, E) * f(r | l, E) / (2 pi r)

    where:
        - g(l, E) is the longitudinal energy density in MeV/cm
        - f(r | l, E) is the radial marginal transverse PDF in 1/cm

    so that u(r, l) is a volumetric density in MeV/cm^3.

    The longitudinal coordinate l is defined by projection of the point
    relative to `shower_start` onto the shower `direction`. The transverse
    radius r is the distance to the shower axis.
    """
    n = points.shape[0]
    out = np.empty(n, dtype=np.float64)

    # Normalize shower direction
    dir_norm = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
    if dir_norm <= 0.0:
        raise ValueError("Shower direction must have non-zero norm.")

    dir_unit = direction / dir_norm

    for i in range(n):
        dx = points[i, 0] - shower_start[0]
        dy = points[i, 1] - shower_start[1]
        dz = points[i, 2] - shower_start[2]

        # Longitudinal coordinate
        l = dx * dir_unit[0] + dy * dir_unit[1] + dz * dir_unit[2]

        # Transverse distance to shower axis
        d2 = dx * dx + dy * dy + dz * dz
        r2 = d2 - l * l
        if r2 < 0.0:
            r2 = 0.0
        r = np.sqrt(r2)

        g = shower_long_profile(l, energy, use_gp=use_gp)
        f = shower_trans_profile(r, l, energy)

        out[i] = g * f / (2.0 * np.pi * max(r, eps))

    return out


@nb.njit(cache=True)
def shower_energy_density(
    depth: Union[float, np.ndarray],
    radius: Union[float, np.ndarray],
    energy: float,
    use_gp: bool = False,
) -> Union[float, np.ndarray]:
    """Compute the shower energy density in (depth, radius) space.

    Parameters
    ----------
    depth : Union[float, np.ndarray]
        Longitudinal depth coordinate(s), in cm.
    radius : Union[float, np.ndarray]
        Radial coordinate(s), in cm.
    energy : float
        Shower energy, in MeV.
    use_gp : bool, default False
        Whether to use the Grindhammer and Peters (2000) parametrization.

    Returns
    -------
    Union[float, np.ndarray]
        Shower energy density in (depth, radius) space, in MeV/cm^2.

    Notes
    -----
    This combines:
        - a longitudinal profile dE/dl(depth, energy), in MeV/cm
        - a transverse radial profile f(r | depth, energy), in 1/cm

    such that:
        rho(depth, radius, energy)
            = dE/dl(depth, energy) * f(radius | depth, energy)

    The transverse profile is assumed to be normalized as:
        integral_0^inf f(r | depth, energy) dr = 1

    so that:
        integral_0^inf rho(depth, r, energy) dr = dE/dl(depth, energy).
    """
    longitudinal = shower_long_profile(depth, energy, use_gp=use_gp)
    transverse = shower_trans_profile(radius, depth, energy)

    return longitudinal * transverse


@nb.njit(cache=True)
def shower_long_profile(
    depth: Union[float, np.ndarray], energy: float, use_gp: bool = False
) -> Union[float, np.ndarray]:
    """Compute the longitudinal shower profile using the Gamma distribution.

    Parameters
    ----------
    depth : Union[float, np.ndarray]
        Depths at which to compute the profile in radiation lengths, in units of cm.
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
    t = np.maximum(t, 0.0)
    coeff = energy * b / math.gamma(a) / LAR_X0
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
def shower_trans_profile(
    radius: Union[float, np.ndarray], depth: Union[float, np.ndarray], energy: float
) -> Union[float, np.ndarray]:
    """Compute the transverse shower profile according to Grindhammer and
    Peters (2000).

    Parameters
    ----------
    radius : Union[float, np.ndarray]
        Radial coordinate(s) at which to evaluate the profile, in cm.
    depth : Union[float, np.ndarray]
        Shower depth(s) at which to evaluate the profile, in cm.
    energy : float
        Shower energy in MeV.

    Returns
    -------
    Union[float, np.ndarray]
        Transverse radial profile dE(t, r) / (dE(t) dr) evaluated at the
        specified radius/radii and depth(s), in cm^-1.

    Notes
    -----
    This is the GP/GFLASH homogeneous-medium transverse profile:
        f(r) = p f_C(r) + (1 - p) f_T(r)
    with
        f_{C,T}(r) = 2 r R_{C,T}^2 / (r^2 + R_{C,T}^2)^2

    The GP parameterization defines R_C and R_T in units of the Moliere
    radius. Here they are converted to cm using a liquid-argon Moliere
    radius estimate.
    """
    r_core, r_tail, p_core = shower_transverse_params_gp(depth, energy)

    core = (2.0 * radius * r_core**2) / (radius**2 + r_core**2) ** 2
    tail = (2.0 * radius * r_tail**2) / (radius**2 + r_tail**2) ** 2

    return p_core * core + (1.0 - p_core) * tail


@nb.njit(cache=True)
def shower_transverse_params_gp(
    depth: Union[float, np.ndarray], energy: float
) -> Tuple[
    Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]
]:
    """Compute the GP transverse shower profile parameters.

    Parameters
    ----------
    depth : Union[float, np.ndarray]
        Shower depth(s) at which to evaluate the parameters, in cm.
    energy : float
        Shower energy in MeV.

    Returns
    -------
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray],
          Union[float, np.ndarray]]
        Tuple containing:
        - core radius R_C in cm
        - tail radius R_T in cm
        - core weight p

    Notes
    -----
    Based on the homogeneous-medium parameterization from Grindhammer and
    Peters (2000). The depth dependence is expressed in terms of the
    normalized depth tau = t / T, where t is the depth in radiation lengths
    and T is the depth of shower maximum in radiation lengths.

    The GP appendix parameterizes z1, ..., p3 using ln(E), with E given in
    GeV. Since this utility takes energy in MeV, the conversion is applied
    internally.
    """
    # Convert depth to radiation lengths
    t = depth / LAR_X0

    # Shower maximum depth in radiation lengths
    t_max = shower_long_mode_gp(energy) / LAR_X0

    # Avoid division by zero / negative values for pathological inputs
    t_max = max(t_max, 1e-6)
    tau = t / t_max

    # GP appendix uses ln(E[GeV])
    energy_gev = max(energy / 1e3, 1e-6)
    loge = np.log(energy_gev)

    # GP homogeneous-medium coefficients
    z1 = 0.0251 + 0.00319 * loge
    z2 = 0.1162 - 0.000381 * LAR_Z

    k1 = 0.659 - 0.00309 * LAR_Z
    k2 = 0.645
    k3 = -2.59
    k4 = 0.3585 + 0.0421 * loge

    p1 = 2.632 - 0.00094 * LAR_Z
    p2 = 0.401 + 0.00187 * LAR_Z
    p3 = 1.313 - 0.0686 * loge

    # Radii in units of Moliere radius
    r_core_rm = z1 + z2 * tau
    r_tail_rm = k1 * (np.exp(k3 * (tau - k2)) + np.exp(k4 * (tau - k2)))

    x = (p2 - tau) / p3
    p_core = p1 * np.exp(x - np.exp(x))

    # Convert radii to cm
    r_core = r_core_rm * LAR_RM
    r_tail = r_tail_rm * LAR_RM

    # Keep parameters in a sensible physical range
    r_core = np.maximum(r_core, 1e-6)
    r_tail = np.maximum(r_tail, 1e-6)
    p_core = np.minimum(np.maximum(p_core, 0.0), 1.0)

    return r_core, r_tail, p_core


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
