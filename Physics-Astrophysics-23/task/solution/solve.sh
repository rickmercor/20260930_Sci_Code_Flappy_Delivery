#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def _acoustic_phase(nu: float, delta_nu: float, nu_p: float) -> float:
    """theta_p = pi (nu - nu_p) / delta_nu with frequencies in microhertz."""
    return np.pi * (float(nu) - float(nu_p)) / float(delta_nu)


def infer_coupling_factor(nu_a: float, nu_b: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    """Reference implementation of the phase-free quadratic for q."""
    nu_a, nu_b, delta_nu, nu_p, delta_pi = (float(v) for v in (nu_a, nu_b, delta_nu, nu_p, delta_pi))
    if delta_nu <= 0.0 or delta_pi <= 0.0:
        raise ValueError("delta_nu and delta_pi must be strictly positive")
    if nu_a == nu_b:
        raise ValueError("the two mixed modes must have different frequencies")
    tan_a = np.tan(_acoustic_phase(nu_a, delta_nu, nu_p))
    tan_b = np.tan(_acoustic_phase(nu_b, delta_nu, nu_p))
    # Frequencies in hertz for the gravity-phase difference.
    kappa = np.tan(np.pi / delta_pi * (1.0 / (nu_b * 1e-6) - 1.0 / (nu_a * 1e-6)))
    # Quadratic A u^2 - B u + kappa = 0 in u = 1 / q.
    coef_a = kappa * tan_a * tan_b
    coef_b = tan_b - tan_a
    if coef_a == 0.0:
        # A mode on the comb (tan = 0) or kappa = 0 makes the equation linear in u.
        roots_u = [kappa / coef_b] if coef_b != 0.0 else []
    else:
        disc = coef_b * coef_b - 4.0 * coef_a * kappa
        if disc < 0.0:
            raise ValueError("the two frequencies do not determine a real coupling factor")
        sqrt_disc = np.sqrt(disc)
        roots_u = [(coef_b + sqrt_disc) / (2.0 * coef_a), (coef_b - sqrt_disc) / (2.0 * coef_a)]
    candidates = []
    for root_u in roots_u:
        if np.isfinite(root_u) and root_u != 0.0:
            q_candidate = 1.0 / root_u
            if 0.0 < q_candidate < 1.0:
                candidates.append(float(q_candidate))
    if len(candidates) != 1:
        raise ValueError("exactly one admissible coupling factor in (0, 1) is required")
    return candidates[0]

import numpy as np


def infer_gravity_phase(nu_a: float, q: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    """Reference implementation: invert the coupling relation for the gravity phase."""
    nu_a, q, delta_nu, nu_p, delta_pi = (float(v) for v in (nu_a, q, delta_nu, nu_p, delta_pi))
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in the open interval (0, 1)")
    if nu_a <= 0.0 or delta_nu <= 0.0 or delta_pi <= 0.0:
        raise ValueError("nu_a, delta_nu and delta_pi must be strictly positive")
    theta_p = _acoustic_phase(nu_a, delta_nu, nu_p)
    # theta_g modulo pi from tan(theta_g) = tan(theta_p) / q.
    theta_g = np.arctan(np.tan(theta_p) / q)
    eps_g = 1.0 / (delta_pi * nu_a * 1e-6) - theta_g / np.pi
    return float(eps_g % 1.0)

import numpy as np
from scipy.optimize import brentq


def _coupling_residual(nu: "np.ndarray", q: float, eps_g: float, delta_nu: float, nu_p: float, delta_pi: float) -> "np.ndarray":
    """Pole-free residual sin(theta_p) cos(theta_g) - q cos(theta_p) sin(theta_g), nu in microhertz."""
    nu = np.asarray(nu, dtype=float)
    theta_p = np.pi * (nu - nu_p) / delta_nu
    theta_g = np.pi / (delta_pi * nu * 1e-6) - np.pi * eps_g
    return np.sin(theta_p) * np.cos(theta_g) - q * np.cos(theta_p) * np.sin(theta_g)


def find_mixed_modes(nu_lo: float, nu_hi: float, q: float, eps_g: float, delta_nu: float, nu_p: float, delta_pi: float) -> "np.ndarray":
    """Reference implementation: one Brent solve per interval between consecutive poles."""
    nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi = (float(v) for v in (nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi))
    if nu_lo <= 0.0 or nu_hi <= nu_lo:
        raise ValueError("the window must satisfy 0 < nu_lo < nu_hi")
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in the open interval (0, 1)")
    if delta_nu <= 0.0 or delta_pi <= 0.0:
        raise ValueError("delta_nu and delta_pi must be strictly positive")
    # tan(theta_p) - q tan(theta_g) increases monotonically from -inf to +inf between
    # consecutive poles (nu_p + (k + 1/2) delta_nu for theta_p, the pure gravity modes
    # 1e6 / (delta_pi (n + 1/2 + eps_g)) for theta_g), so every such interval holds
    # exactly one mixed mode; bracketing them removes any dependence on a scan step.
    k_lo = np.floor((nu_lo - nu_p) / delta_nu - 0.5) - 1
    k_hi = np.ceil((nu_hi - nu_p) / delta_nu - 0.5) + 1
    p_poles = nu_p + (np.arange(k_lo, k_hi + 1) + 0.5) * delta_nu
    n_hi = np.ceil(1e6 / (delta_pi * nu_lo) - 0.5 - eps_g) + 1
    n_lo = np.floor(1e6 / (delta_pi * nu_hi) - 0.5 - eps_g) - 1
    n_vals = np.arange(n_lo, n_hi + 1)
    n_vals = n_vals[n_vals + 0.5 + eps_g > 0.0]
    g_poles = 1e6 / (delta_pi * (n_vals + 0.5 + eps_g))
    poles = np.sort(np.concatenate([p_poles, g_poles]))
    roots = []
    for left, right in zip(poles[:-1], poles[1:]):
        if right <= left or right < nu_lo or left > nu_hi:
            continue
        margin = 1e-6 * (right - left)
        lo, hi = left + margin, right - margin
        if hi <= lo:
            continue
        f_lo = _coupling_residual(lo, q, eps_g, delta_nu, nu_p, delta_pi)
        f_hi = _coupling_residual(hi, q, eps_g, delta_nu, nu_p, delta_pi)
        if f_lo == 0.0:
            root = lo
        elif f_hi == 0.0:
            root = hi
        elif f_lo * f_hi < 0.0:
            root = brentq(_coupling_residual, lo, hi, args=(q, eps_g, delta_nu, nu_p, delta_pi),
                          xtol=1e-13, rtol=4 * np.finfo(float).eps, maxiter=500)
        else:
            continue
        if nu_lo <= root <= nu_hi:
            roots.append(float(root))
    return np.array(sorted(set(roots)), dtype=float)

import numpy as np


def compute_trapping_fraction(nu: float, q: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    """Reference implementation of the asymptotic zeta function."""
    nu, q, delta_nu, nu_p, delta_pi = (float(v) for v in (nu, q, delta_nu, nu_p, delta_pi))
    if nu <= 0.0 or delta_nu <= 0.0 or delta_pi <= 0.0:
        raise ValueError("nu, delta_nu and delta_pi must be strictly positive")
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in the open interval (0, 1)")
    theta_p = _acoustic_phase(nu, delta_nu, nu_p)
    nu_hz, delta_nu_hz = nu * 1e-6, delta_nu * 1e-6
    ratio = nu_hz ** 2 * delta_pi / delta_nu_hz
    denominator = q * np.cos(theta_p) ** 2 + np.sin(theta_p) ** 2 / q
    return float(1.0 / (1.0 + ratio / denominator))

import numpy as np


def _degree_norm() -> float:
    """L = sqrt(l (l + 1)) for the quadrupolar modes (l = 2)."""
    return np.sqrt(6.0)


def compute_core_coupling(nu_i: float, nu_j: float, zeta_i: float, zeta_j: float, delta_pi: float) -> float:
    """Reference implementation of the asymptotic gamma_c."""
    nu_i, nu_j, zeta_i, zeta_j, delta_pi = (float(v) for v in (nu_i, nu_j, zeta_i, zeta_j, delta_pi))
    if nu_i <= 0.0 or nu_j <= 0.0:
        raise ValueError("mode frequencies must be strictly positive")
    if nu_i == nu_j:
        raise ValueError("the coupling coefficient is defined for two distinct modes")
    if not (0.0 < zeta_i <= 1.0 and 0.0 < zeta_j <= 1.0):
        raise ValueError("trapping fractions must lie in (0, 1]")
    if delta_pi <= 0.0:
        raise ValueError("delta_pi must be strictly positive")
    big_l = _degree_norm()
    reduced_pi = big_l * delta_pi
    f_i, f_j = nu_i * 1e-6, nu_j * 1e-6  # hertz
    prefactor = reduced_pi * (big_l ** 2 - 1.0) / (np.pi * big_l ** 3) * np.sqrt(zeta_i * zeta_j)
    diff_term = f_i * f_j / (f_j - f_i) * np.sin(np.pi * big_l / reduced_pi * (f_j - f_i) / (f_i * f_j))
    sum_term = f_i * f_j / (f_j + f_i) * (np.cos(np.pi * big_l / reduced_pi * (f_j + f_i) / (f_i * f_j)) - 1.0)
    return float(prefactor * (diff_term + sum_term))

import numpy as np


def assemble_rotation_matrix(nu_modes: "np.ndarray", q: float, delta_nu: float, nu_p: float, delta_pi: float, nu_core: float, nu_env: float, m: int) -> "np.ndarray":
    """Reference implementation of the two-zone rotational coupling matrix."""
    nu_modes = np.atleast_1d(np.asarray(nu_modes, dtype=float))
    if nu_modes.ndim != 1 or nu_modes.size < 1 or np.any(nu_modes <= 0.0):
        raise ValueError("nu_modes must be a one-dimensional array of strictly positive frequencies")
    if np.unique(nu_modes).size != nu_modes.size:
        raise ValueError("nu_modes must be distinct")
    if isinstance(m, bool) or int(m) != m or abs(int(m)) > 2:
        raise ValueError("m must be an integer with |m| <= 2")
    m = int(m)
    nu_core_uhz, nu_env_uhz = float(nu_core) * 1e-3, float(nu_env) * 1e-3
    big_l2 = _degree_norm() ** 2
    n_modes = nu_modes.size
    zeta = np.array([compute_trapping_fraction(nu, q, delta_nu, nu_p, delta_pi) for nu in nu_modes])
    rot_matrix = np.zeros((n_modes, n_modes), dtype=float)
    for i in range(n_modes):
        rot_matrix[i, i] = 2.0 * m * (zeta[i] * (1.0 - 1.0 / big_l2) * nu_core_uhz + (1.0 - zeta[i]) * nu_env_uhz)
        for j in range(i + 1, n_modes):
            gamma_c = compute_core_coupling(nu_modes[i], nu_modes[j], zeta[i], zeta[j], delta_pi)
            gamma_e = big_l2 / (1.0 - big_l2) * gamma_c
            rot_matrix[i, j] = rot_matrix[j, i] = 2.0 * m * (gamma_c * nu_core_uhz + gamma_e * nu_env_uhz)
    return rot_matrix

import numpy as np


def solve_rotational_multiplets(nu_modes: "np.ndarray", rot_matrix: "np.ndarray") -> "np.ndarray":
    """Reference implementation: companion linearization of the quadratic eigenvalue problem."""
    nu_modes = np.atleast_1d(np.asarray(nu_modes, dtype=float))
    rot_matrix = np.asarray(rot_matrix, dtype=float)
    n_modes = nu_modes.size
    if nu_modes.ndim != 1 or n_modes < 1 or np.any(nu_modes <= 0.0):
        raise ValueError("nu_modes must be a one-dimensional array of strictly positive frequencies")
    if np.unique(nu_modes).size != n_modes:
        raise ValueError("nu_modes must be distinct")
    if rot_matrix.shape != (n_modes, n_modes) or not np.allclose(rot_matrix, rot_matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("rot_matrix must be a real symmetric (N, N) array")
    # Linearization: with b = nu a, [R  D; I  0] [b; a] = nu [b; a], D = diag(nu_0^2).
    companion = np.zeros((2 * n_modes, 2 * n_modes), dtype=float)
    companion[:n_modes, :n_modes] = rot_matrix
    companion[:n_modes, n_modes:] = np.diag(nu_modes ** 2)
    companion[n_modes:, :n_modes] = np.eye(n_modes)
    eigenvalues, eigenvectors = np.linalg.eig(companion)
    eigenvalues = eigenvalues.real
    positive = np.where(eigenvalues > 0.0)[0]
    if positive.size != n_modes:
        raise ValueError("the quadratic eigenvalue problem must have exactly N positive eigenvalues")
    weights = np.abs(eigenvectors[n_modes:, positive])
    weights /= np.linalg.norm(weights, axis=0, keepdims=True)
    dominant = np.argmax(weights, axis=0)  # unperturbed mode dominating each positive eigenvector
    if np.unique(dominant).size != n_modes:
        raise ValueError("the dominant-component rule does not label the eigenfrequencies one to one")
    nu_perturbed = np.zeros(n_modes, dtype=float)
    nu_perturbed[dominant] = eigenvalues[positive]
    return nu_perturbed

import numpy as np


def compute_multiplet_asymmetry(nu_a: float, nu_b: float, delta_nu: float, nu_p: float, delta_pi: float, nu_core: float, nu_env: float, nu_lo: float, nu_hi: float, nu_target: float, m: int = 2) -> float:
    """Reference end-to-end pipeline."""
    if isinstance(m, bool) or int(m) != m or int(m) not in (1, 2):
        raise ValueError("m must be 1 or 2")
    m = int(m)
    q = infer_coupling_factor(nu_a, nu_b, delta_nu, nu_p, delta_pi)
    eps_g = infer_gravity_phase(nu_a, q, delta_nu, nu_p, delta_pi)
    nu_modes = find_mixed_modes(nu_lo, nu_hi, q, eps_g, delta_nu, nu_p, delta_pi)
    if nu_modes.size == 0:
        raise ValueError("the window contains no mixed mode")
    index = int(np.argmin(np.abs(nu_modes - float(nu_target))))
    nu_plus = solve_rotational_multiplets(
        nu_modes, assemble_rotation_matrix(nu_modes, q, delta_nu, nu_p, delta_pi, nu_core, nu_env, m))
    nu_minus = solve_rotational_multiplets(
        nu_modes, assemble_rotation_matrix(nu_modes, q, delta_nu, nu_p, delta_pi, nu_core, nu_env, -m))
    return float((nu_plus[index] + nu_minus[index] - 2.0 * nu_modes[index]) * 1e3)
SCICODE_GOLD_EOF
