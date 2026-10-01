"""
Determine the rupture thresholds of all bonds of an n_bonds chain self-consistently: starting from a common initial threshold l_init, form the log intact weights of all bonds at the current thresholds, locate the barrier top of every bond's constrained potential of mean force, set each threshold to its own barrier top, and repeat until the largest threshold change is below tol (source, Sec. S2.2, the self-consistent scheme; at most max_iter sweeps). Call the earlier step functions for the intact weights and the stationary points. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

The dividing surface of the multichannel rupture problem is placed variationally at the barrier tops of the bond-resolved potentials of mean force. Because each profile is computed with the other bonds constrained below their own thresholds, the thresholds are coupled and must be iterated to a fixed point.

Returns
-------
ndarray of float64, shape (n_bonds,): converged rupture thresholds in units of l_e, bond 1 first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def self_consistent_thresholds(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", f_red: float,
                                       beta_de: float, a_le: float, n_bonds: int, l_init: float,
                                       tol: float = 1e-10, max_iter: int = 50, n_l: int = 4001) -> "np.ndarray":
    """Determine the rupture thresholds of all bonds of an n_bonds chain self-consistently: starting from a common initial threshold l_init, form the log intact weights of all bonds at the current thresholds, locate the barrier top of every bond's constrained potential of mean force, set each threshold to its own barrier top, and repeat until the largest threshold change is below tol (source, Sec. S2.2, the self-consistent scheme; at most max_iter sweeps). Call the earlier step functions for the intact weights and the stationary points. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid strictly inside (0, pi), increasing, shape (n_theta,).
    w_theta : np.ndarray
        Positive quadrature weights of the theta grid, shape (n_theta,).
    kernel : np.ndarray
        Bending kernel of shape (n_theta, n_theta).
    f_red : float
        Reduced force f l_e / D_e (positive).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    n_bonds : int
        Number of bonds (positive).
    l_init : float
        Common initial threshold in units of l_e (positive).
    tol : float
        Convergence tolerance on the largest threshold change (default 1e-10).
    max_iter : int
        Maximum number of sweeps (default 50).
    n_l : int
        Trapezoid points of the intact-weight integrals (default 4001).

    Returns
    -------
    thresholds : np.ndarray
        Converged thresholds of shape (n_bonds,).

    Raises
    ------
    ValueError
        If the grid is invalid, n_bonds or max_iter is not a positive integer, l_init or tol is not finite positive, any earlier step raises for the supplied data, or the iteration does not converge within max_iter sweeps.
    """
    return thresholds

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _check_grid(theta, w_theta):
    theta = np.asarray(theta, dtype=float)
    w_theta = np.asarray(w_theta, dtype=float)
    if theta.ndim != 1 or theta.size < 2 or w_theta.shape != theta.shape:
        raise ValueError("theta and w_theta must be one-dimensional arrays of equal length >= 2")
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(w_theta))):
        raise ValueError("theta and w_theta must be finite")
    if np.any(theta <= 0.0) or np.any(theta >= np.pi) or np.any(np.diff(theta) <= 0.0) or np.any(w_theta <= 0.0):
        raise ValueError("theta must be strictly increasing inside (0, pi) with positive weights")
    return theta, w_theta


def _oracle_self_consistent_thresholds(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", f_red: float,
                                       beta_de: float, a_le: float, n_bonds: int, l_init: float,
                                       tol: float = 1e-10, max_iter: int = 50, n_l: int = 4001) -> "np.ndarray":
    theta, w_theta = _check_grid(theta, w_theta)
    if not isinstance(n_bonds, (int, np.integer)) or n_bonds < 1:
        raise ValueError("n_bonds must be a positive integer")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    for name, value in (("l_init", l_init), ("tol", tol)):
        _check_positive(name, value)
    thresholds = np.full(int(n_bonds), float(l_init))
    for _ in range(int(max_iter)):
        log_i = np.array([_oracle_log_intact_weight(theta, thresholds[i], f_red, beta_de, a_le, n_l)
                          for i in range(int(n_bonds))])
        new = np.array([_oracle_pmf_stationary_points(theta, w_theta, kernel, log_i, i, f_red, beta_de, a_le)[1]
                        for i in range(int(n_bonds))])
        change = float(np.max(np.abs(new - thresholds)))
        thresholds = new
        if change < tol:
            return thresholds
    raise ValueError("self-consistent thresholds did not converge within max_iter iterations")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_init = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nn_bonds = 6\n",
            "call": "self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le, n_bonds, l_init)",
            "gold_call": "_oracle_self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le, n_bonds, l_init)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nf_red = 0.5\nl_init = 3.0\nn_bonds = 3\n",
            "call": "self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le, n_bonds, l_init)",
            "gold_call": "_oracle_self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le, n_bonds, l_init)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, 0.0, phi_e)\nl_init = 1.7\nn_bonds = 4\n",
            "call": "self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le, n_bonds, l_init)",
            "gold_call": "_oracle_self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le, n_bonds, l_init)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_init = 1.7\ndef run_model():\n    try:\n        self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le, 0, l_init)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_self_consistent_thresholds(theta, w_theta, kernel, f_red, beta_de, a_le, 0, l_init)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
