#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

# GOLD SOLUTION

import numpy as np
def compute_bond_kinetic_prefactors(
    n_bonds: int,
    atom_mass_kg: float,
    bond_length_m: float,
    temperature_k: float,
    tethered: bool,
) -> "np.ndarray":
    """Reference implementation from the effective bond-length masses."""
    import numpy as np

    if isinstance(n_bonds, bool) or not isinstance(n_bonds, (int, np.integer)) or n_bonds < 1:
        raise ValueError("n_bonds must be an integer of at least 1")
    for value in (atom_mass_kg, bond_length_m, temperature_k):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("physical parameters must be real numbers")
        if not np.isfinite(value) or value <= 0:
            raise ValueError("physical parameters must be finite and positive")
    if not isinstance(tethered, (bool, np.bool_)):
        raise ValueError("tethered must be a bool")
    kbt = 1.380649e-23 * float(temperature_k)
    # A bond between two mobile atoms is their relative coordinate, with half the atomic
    # mass; a bond attached to the fixed atom moves with that of one atom.
    effective_mass = np.full(int(n_bonds), 0.5 * float(atom_mass_kg))
    if tethered:
        effective_mass[0] = float(atom_mass_kg)
    mean_positive_velocity = np.sqrt(kbt / (2.0 * np.pi * effective_mass))
    return mean_positive_velocity / float(bond_length_m)

# GOLD SOLUTION

import numpy as np
def build_bending_kernel(
    n_theta: int,
    n_omega: int,
    stiffness_reduced: "np.ndarray",
    equilibrium_angle_deg: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation using log-add-exp azimuthal accumulation."""
    import numpy as np

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    if not (_is_int(n_theta) and n_theta >= 2):
        raise ValueError("n_theta must be an integer of at least 2")
    if not (_is_int(n_omega) and n_omega >= 4):
        raise ValueError("n_omega must be an integer of at least 4")
    stiffness = np.asarray(stiffness_reduced, dtype=float)
    angle_deg = np.asarray(equilibrium_angle_deg, dtype=float)
    if stiffness.ndim != 1 or stiffness.size == 0 or not np.all(np.isfinite(stiffness)) or np.any(stiffness < 0):
        raise ValueError("stiffness_reduced must be a non-empty finite nonnegative 1D array")
    if angle_deg.shape != stiffness.shape or not np.all(np.isfinite(angle_deg)) or np.any((angle_deg < 0) | (angle_deg > 180)):
        raise ValueError("equilibrium_angle_deg must match stiffness_reduced and lie in [0, 180]")
    nodes, _ = np.polynomial.legendre.leggauss(int(n_theta))
    theta = 0.5 * np.pi * (nodes + 1.0)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    log_kernel = np.full((stiffness.size, int(n_theta), int(n_theta)), -np.inf)
    for k in range(int(n_omega)):
        omega = 2.0 * np.pi * k / int(n_omega)
        cos_phi = np.outer(sin_t, sin_t) * np.cos(omega) + np.outer(cos_t, cos_t)
        phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))
        terms = -0.5 * (stiffness / np.pi**2)[:, None, None] * (
            phi[None, :, :] - np.deg2rad(angle_deg)[:, None, None]
        ) ** 2
        log_kernel = np.logaddexp(log_kernel, terms)
    return log_kernel + np.log(2.0 * np.pi / int(n_omega))

# GOLD SOLUTION

import numpy as np
def compute_intact_bond_weights(
    thresholds: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
    n_theta: int,
    n_length: int,
) -> "np.ndarray":
    """Reference implementation with log-sum-exp length quadrature."""
    import numpy as np
    from scipy.special import logsumexp

    def _is_int(value):
        return not isinstance(value, bool) and isinstance(value, (int, np.integer))

    upper = np.asarray(thresholds, dtype=float)
    if upper.ndim != 1 or upper.size == 0 or not np.all(np.isfinite(upper)) or np.any(upper <= 0.5):
        raise ValueError("thresholds must be a non-empty 1D array of finite values above 0.5")
    if not np.isfinite(force_reduced) or force_reduced < 0:
        raise ValueError("force_reduced must be finite and nonnegative")
    for value in (beta_de, a_le):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("beta_de and a_le must be finite and positive")
    if not (_is_int(n_theta) and n_theta >= 2 and _is_int(n_length) and n_length >= 2):
        raise ValueError("n_theta and n_length must be integers of at least 2")
    angle_nodes, _ = np.polynomial.legendre.leggauss(int(n_theta))
    theta = 0.5 * np.pi * (angle_nodes + 1.0)
    length_nodes, length_weights = np.polynomial.legendre.leggauss(int(n_length))
    coupling = float(beta_de) * float(force_reduced)
    log_weights = np.empty((upper.size, int(n_theta)))
    for j, x_max in enumerate(upper):
        half = 0.5 * (x_max - 0.5)
        x = half * length_nodes + 0.5 + half
        stretch = float(beta_de) * (1.0 - np.exp(-float(a_le) * (x - 1.0))) ** 2
        radial = np.log(half * length_weights) + 2.0 * np.log(x) - stretch
        exponent = radial[None, :] + coupling * np.outer(np.cos(theta), x)
        log_weights[j] = np.log(np.sin(theta)) + logsumexp(exponent, axis=1)
    return log_weights

# GOLD SOLUTION

import numpy as np
def compute_angular_weights(log_intact: "np.ndarray", log_kernels: "np.ndarray") -> "np.ndarray":
    """Reference implementation with log-domain directional messages."""
    import numpy as np
    from scipy.special import logsumexp

    local = np.asarray(log_intact, dtype=float)
    coupling = np.asarray(log_kernels, dtype=float)
    if local.ndim != 2 or local.shape[0] < 1 or local.shape[1] < 2 or not np.all(np.isfinite(local)):
        raise ValueError("log_intact must be a finite 2D array with at least one row and two columns")
    n_bonds, n_theta = local.shape
    if coupling.shape != (n_bonds - 1, n_theta, n_theta) or not np.all(np.isfinite(coupling)):
        raise ValueError("log_kernels must contain one finite square log kernel per chain link")
    _, nodes_weights = np.polynomial.legendre.leggauss(n_theta)
    quad = 0.5 * np.pi * nodes_weights
    log_quad = np.log(quad)
    forward = np.empty((n_bonds, n_theta), dtype=float)
    backward = np.empty((n_bonds, n_theta), dtype=float)
    forward[0] = -np.log(np.pi)
    backward[-1] = -np.log(np.pi)
    for i in range(n_bonds - 1):
        source = log_quad + forward[i] + local[i]
        message = logsumexp(coupling[i] + source[None, :], axis=1)
        forward[i + 1] = message - logsumexp(log_quad + message)
    for i in range(n_bonds - 1, 0, -1):
        source = log_quad + backward[i] + local[i]
        message = logsumexp(coupling[i - 1] + source[:, None], axis=0)
        backward[i - 1] = message - logsumexp(log_quad + message)
    return np.exp(forward + backward)

# GOLD SOLUTION

import numpy as np
def evaluate_bond_pmf(
    x_values: "np.ndarray",
    angular_weight: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
) -> "np.ndarray":
    """Reference PMF implementation with analytic angular cumulants."""
    import numpy as np
    from scipy.special import logsumexp

    x = np.asarray(x_values, dtype=float)
    weight = np.asarray(angular_weight, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("x_values must be a non-empty 1D array of finite positive values")
    if (weight.ndim != 1 or weight.size < 2 or not np.all(np.isfinite(weight))
            or np.any(weight < 0) or not np.sum(weight) > 0):
        raise ValueError("angular_weight must be a finite nonnegative 1D array with a positive sum")
    if not np.isfinite(force_reduced) or force_reduced < 0:
        raise ValueError("force_reduced must be finite and nonnegative")
    for value in (beta_de, a_le):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("beta_de and a_le must be finite and positive")
    nodes, node_weights = np.polynomial.legendre.leggauss(weight.size)
    theta = 0.5 * np.pi * (nodes + 1.0)
    with np.errstate(divide="ignore"):
        log_angular = np.log(0.5 * np.pi * node_weights * weight * np.sin(theta))
    coupling = float(beta_de) * float(force_reduced)
    stretch = float(beta_de) * (1.0 - np.exp(-float(a_le) * (x - 1.0))) ** 2
    # The x**2 Jacobian and force-biased angular partition enter as entropy.
    cos_t = np.cos(theta)
    exponent = log_angular[None, :] + coupling * np.outer(x, cos_t)
    log_partition = logsumexp(exponent, axis=1)
    angular_share = np.exp(exponent - log_partition[:, None])
    mean_cos = angular_share @ cos_t
    variance_cos = angular_share @ (cos_t ** 2) - mean_cos ** 2
    pmf = stretch - 2.0 * np.log(x) - log_partition
    decay = np.exp(-float(a_le) * (x - 1.0))
    slope = (
        2.0 * float(beta_de) * float(a_le) * (1.0 - decay) * decay
        - 2.0 / x
        - coupling * mean_cos
    )
    curvature = (
        2.0 * float(beta_de) * float(a_le) ** 2 * (2.0 * decay ** 2 - decay)
        + 2.0 / x ** 2
        - coupling ** 2 * variance_cos
    )
    return np.column_stack((pmf - pmf[0], slope, curvature))

# GOLD SOLUTION

import numpy as np
def locate_pmf_stationary_points(
    angular_weights: "np.ndarray",
    force_reduced: float,
    beta_de: float,
    a_le: float,
) -> "np.ndarray":
    """Reference implementation bracketing sign changes of the analytic derivative."""
    import numpy as np
    from scipy.optimize import brentq
    from scipy.special import logsumexp

    weights = np.asarray(angular_weights, dtype=float)
    if (weights.ndim != 2 or weights.shape[0] < 1 or weights.shape[1] < 2
            or not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.all(weights.sum(axis=1) > 0)):
        raise ValueError("angular_weights must be a finite nonnegative 2D array with positive rows")
    if not np.isfinite(force_reduced) or force_reduced < 0:
        raise ValueError("force_reduced must be finite and nonnegative")
    for value in (beta_de, a_le):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("beta_de and a_le must be finite and positive")
    nodes, node_weights = np.polynomial.legendre.leggauss(weights.shape[1])
    theta = 0.5 * np.pi * (nodes + 1.0)
    cos_t = np.cos(theta)
    coupling = float(beta_de) * float(force_reduced)
    grid = 0.9 + 0.002 * np.arange(2051)

    def _derivative(x, log_angular):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        decay = np.exp(-float(a_le) * (x - 1.0))
        exponent = log_angular[None, :] + coupling * np.outer(x, cos_t)
        share = np.exp(exponent - logsumexp(exponent, axis=1, keepdims=True))
        return 2.0 * float(beta_de) * float(a_le) * (1.0 - decay) * decay - 2.0 / x - coupling * share @ cos_t

    points = np.empty((weights.shape[0], 2))
    for i, row in enumerate(weights):
        with np.errstate(divide="ignore"):
            log_angular = np.log(0.5 * np.pi * node_weights * row * np.sin(theta))
        profile = evaluate_bond_pmf(
            grid, row, force_reduced, beta_de, a_le
        )
        slope = profile[:, 1]
        rise = np.flatnonzero((slope[:-1] < 0) & (slope[1:] >= 0))
        fall = np.flatnonzero((slope[:-1] > 0) & (slope[1:] <= 0))
        fall = fall[fall > rise[0]] if rise.size else fall[:0]
        if fall.size == 0:
            raise ValueError("a bond has no bonded minimum followed by a barrier top")
        scalar = lambda x, la=log_angular: float(_derivative(x, la)[0])
        roots = np.array([
            brentq(scalar, grid[k], grid[k + 1], xtol=1e-14, maxiter=500)
            for k in (rise[0], fall[0])
        ])
        root_profile = evaluate_bond_pmf(
            roots, row, force_reduced, beta_de, a_le
        )
        if not (root_profile[0, 2] > 0.0 and root_profile[1, 2] < 0.0):
            raise ValueError("stationary points do not form a bonded minimum and barrier top")
        points[i] = roots
    return points

# GOLD SOLUTION

import numpy as np
def compute_bond_scission_rates(
    pmf_table: "np.ndarray",
    barrier_tops: "np.ndarray",
    prefactors: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the full (non-harmonic) transition-state rate."""
    import numpy as np
    from scipy.special import logsumexp

    table = np.asarray(pmf_table, dtype=float)
    tops = np.asarray(barrier_tops, dtype=float)
    nu = np.asarray(prefactors, dtype=float)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 3 or not np.all(np.isfinite(table)):
        raise ValueError("pmf_table must be a finite 2D array with at least one row and three columns")
    n_bonds, n_length = table.shape[0], table.shape[1] - 1
    if tops.shape != (n_bonds,) or not np.all(np.isfinite(tops)) or np.any(tops <= 0.5):
        raise ValueError("barrier_tops must match pmf_table and lie above 0.5")
    if nu.shape != (n_bonds,) or not np.all(np.isfinite(nu)) or np.any(nu <= 0):
        raise ValueError("prefactors must match pmf_table and be finite and positive")
    _, node_weights = np.polynomial.legendre.leggauss(n_length)
    rates = np.empty(n_bonds)
    for i in range(n_bonds):
        half = 0.5 * (tops[i] - 0.5)
        # Normalized Boltzmann density of the PMF at the threshold, times the mean positive velocity.
        log_population = logsumexp(np.log(half * node_weights) - (table[i, 1:] - table[i, 0]))
        rates[i] = nu[i] * np.exp(-log_population)
    return rates

# GOLD SOLUTION

import numpy as np
def predict_tethered_scission_lifetime(
    measured_lifetime_ns: float = 18.921926,
    measured_terminal_share: float = 0.99662638,
    measured_force_reduced: float = 0.94,
    predicted_force_reduced: float = 1.04,
    n_bonds: int = 10,
    beta_de: float = 279.0,
    a_le: float = 2.15,
    atom_mass_kg: float = 1.99e-26,
    bond_length_m: float = 1.525e-10,
    temperature_k: float = 293.15,
    n_theta: int = 160,
    n_omega: int = 256,
    n_length: int = 600,
    stiffness_low: float = 1000.0,
    stiffness_high: float = 8000.0,
    angle_low_deg: float = 50.0,
    angle_high_deg: float = 100.0,
) -> float:
    """Reference two-observable inverse fit using only the reference chain."""
    import numpy as np
    from scipy.optimize import least_squares

    scalars = (
        measured_lifetime_ns, measured_terminal_share, stiffness_low,
        stiffness_high, angle_low_deg, angle_high_deg,
    )
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("measurements and fit bounds must be finite")
    if measured_lifetime_ns <= 0.0 or not 0.0 < measured_terminal_share < 1.0:
        raise ValueError("need a positive lifetime and a terminal share strictly between zero and one")
    if isinstance(n_bonds, bool) or not isinstance(n_bonds, (int, np.integer)) or n_bonds < 3:
        raise ValueError("n_bonds must be an integer of at least 3")
    if not 0.0 < stiffness_low < stiffness_high:
        raise ValueError("need 0 < stiffness_low < stiffness_high")
    if not 0.0 <= angle_low_deg < angle_high_deg <= 180.0:
        raise ValueError("angle bounds must be ordered inside [0, 180]")

    def _rates(stiffness, angle_deg, force, tethered):
        prefactors = compute_bond_kinetic_prefactors(
            n_bonds, atom_mass_kg, bond_length_m, temperature_k, tethered
        )
        link_stiffness = np.full(int(n_bonds) - 1, float(stiffness))
        link_angles = np.full(int(n_bonds) - 1, float(angle_deg))
        log_kernels = build_bending_kernel(
            n_theta, n_omega, link_stiffness, link_angles
        )
        thresholds = np.full(int(n_bonds), 2.0)
        for _ in range(60):
            log_intact = compute_intact_bond_weights(
                thresholds, force, beta_de, a_le, n_theta, n_length
            )
            weights = compute_angular_weights(log_intact, log_kernels)
            points = locate_pmf_stationary_points(weights, force, beta_de, a_le)
            change = float(np.max(np.abs(points[:, 1] - thresholds)))
            thresholds = points[:, 1].copy()
            if change < 1e-11:
                break
        else:
            raise ValueError("rupture thresholds did not converge within 60 sweeps")

        nodes, _ = np.polynomial.legendre.leggauss(int(n_length))
        table = np.empty((thresholds.size, nodes.size + 1))
        for i, top in enumerate(thresholds):
            half = 0.5 * (top - 0.5)
            x = np.concatenate(([top], half * nodes + 0.5 + half))
            table[i] = evaluate_bond_pmf(
                x, weights[i], force, beta_de, a_le
            )[:, 0]
        return compute_bond_scission_rates(table, thresholds, prefactors)

    target_log_lifetime = np.log(float(measured_lifetime_ns) * 1.0e-9)
    target_log_odds = np.log(float(measured_terminal_share)) - np.log1p(-float(measured_terminal_share))

    def _residual(parameters):
        stiffness = np.exp(parameters[0])
        rates = _rates(stiffness, parameters[1], measured_force_reduced, False)
        terminal_rate = float(rates[0] + rates[-1])
        interior_rate = float(np.sum(rates[1:-1]))
        log_lifetime = -np.log(float(np.sum(rates)))
        log_odds = np.log(terminal_rate) - np.log(interior_rate)
        return np.array([log_lifetime - target_log_lifetime, log_odds - target_log_odds])

    lower = np.array([np.log(float(stiffness_low)), float(angle_low_deg)])
    upper = np.array([np.log(float(stiffness_high)), float(angle_high_deg)])
    start = 0.5 * (lower + upper)
    fit = least_squares(
        _residual, start, bounds=(lower, upper),
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=100,
    )
    residual = _residual(fit.x)
    if not fit.success or not np.all(np.isfinite(residual)) or np.max(np.abs(residual)) > 2e-6:
        raise ValueError("the two observations cannot be matched inside the parameter bounds")

    predicted_rates = _rates(
        np.exp(fit.x[0]), fit.x[1], predicted_force_reduced, True
    )
    return float(1.0e12 / np.sum(predicted_rates))
SCICODE_GOLD_EOF
