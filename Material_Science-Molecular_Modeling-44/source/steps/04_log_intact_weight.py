"""
Compute, on the polar-angle grid, the natural logarithm of the local intact weight of one bond whose rupture threshold is l_thr (source, Sec. S2.1, the local intact weight): I(theta) = sin(theta) * integral from 0 to l_thr of exp(-beta_de * [v_str(l) - f_red * l * cos(theta)]) * l^2 dl, with v_str the Morse stretching energy, i.e. the bond's Boltzmann factor integrated over its allowed length interval with the full three-dimensional bond-vector measure l^2 sin(theta) dl dtheta (the azimuth already integrated out). The sin(theta) factor of the measure is included in this weight, so that the polar-angle integrals of the later steps use plain dtheta quadrature weights. Use the trapezoid rule with n_l equally spaced points on [0, l_thr] and evaluate in the log domain so that the result is finite for beta_de of several hundred. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

The intact basin of a breakable chain constrains every bond length below its rupture threshold. The weight a bond contributes at a given orientation is the integral of its Boltzmann factor over that interval; because the bond potential is deep on the thermal scale, the integrand spans hundreds of orders of magnitude and must be handled logarithmically.

Returns
-------
ndarray of float64, shape (n_theta,): natural logarithm of the intact weight at each theta.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def log_intact_weight(theta: "np.ndarray", l_thr: float, f_red: float, beta_de: float, a_le: float,
                              n_l: int = 4001) -> "np.ndarray":
    """Compute, on the polar-angle grid, the natural logarithm of the local intact weight of one bond whose rupture threshold is l_thr (source, Sec. S2.1, the local intact weight): I(theta) = sin(theta) * integral from 0 to l_thr of exp(-beta_de * [v_str(l) - f_red * l * cos(theta)]) * l^2 dl, with v_str the Morse stretching energy, i.e. the bond's Boltzmann factor integrated over its allowed length interval with the full three-dimensional bond-vector measure l^2 sin(theta) dl dtheta (the azimuth already integrated out). The sin(theta) factor of the measure is included in this weight, so that the polar-angle integrals of the later steps use plain dtheta quadrature weights. Use the trapezoid rule with n_l equally spaced points on [0, l_thr] and evaluate in the log domain so that the result is finite for beta_de of several hundred. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid strictly inside (0, pi), shape (n_theta,).
    l_thr : float
        Rupture threshold of the bond in units of l_e (positive).
    f_red : float
        Reduced force f l_e / D_e (positive).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    n_l : int
        Number of trapezoid points on [0, l_thr] (default 4001).

    Returns
    -------
    log_i : np.ndarray
        Log intact weight on the theta grid.

    Raises
    ------
    ValueError
        If theta is not a finite one-dimensional array strictly inside (0, pi), l_thr, f_red, beta_de or a_le is not finite positive, or n_l is not an integer >= 3.
    """
    return log_i

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _morse(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return (1.0 - e) ** 2


def _log_trapezoid(log_f, x):
    """log of the trapezoidal integral of exp(log_f) over x (log_f may contain -inf)."""
    dx = np.diff(x)
    log_w = np.log(np.concatenate(([dx[0] / 2.0], (dx[:-1] + dx[1:]) / 2.0, [dx[-1] / 2.0])))
    e = log_f + log_w
    m = e.max()
    return float(m + np.log(np.exp(e - m).sum()))


def _oracle_log_intact_weight(theta: "np.ndarray", l_thr: float, f_red: float, beta_de: float, a_le: float,
                              n_l: int = 4001) -> "np.ndarray":
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.size < 1 or not np.all(np.isfinite(theta)):
        raise ValueError("theta must be a one-dimensional finite array")
    if np.any(theta <= 0.0) or np.any(theta >= np.pi):
        raise ValueError("theta must lie strictly inside (0, pi)")
    for name, value in (("l_thr", l_thr), ("f_red", f_red), ("beta_de", beta_de), ("a_le", a_le)):
        _check_positive(name, value)
    if not isinstance(n_l, (int, np.integer)) or n_l < 3:
        raise ValueError("n_l must be an integer >= 3")
    l = np.linspace(0.0, l_thr, int(n_l))
    with np.errstate(divide="ignore"):
        log_l2 = 2.0 * np.log(l)                                  # -inf at l = 0: zero measure there
    expo = (-beta_de * _morse(l, a_le) + log_l2)[None, :] + beta_de * f_red * l[None, :] * np.cos(theta)[:, None]
    out = np.empty(theta.size)
    for k in range(theta.size):
        out[k] = _log_trapezoid(expo[k], l)
    return np.log(np.sin(theta)) + out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nl_thr = 1.766\n",
            "call": "log_intact_weight(theta, l_thr, f_red, beta_de, a_le)",
            "gold_call": "_oracle_log_intact_weight(theta, l_thr, f_red, beta_de, a_le)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nf_red, l_thr = 0.02, 3.0\n",
            "call": "log_intact_weight(theta, l_thr, f_red, beta_de, a_le)",
            "gold_call": "_oracle_log_intact_weight(theta, l_thr, f_red, beta_de, a_le)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le = 279.0, 2.15\nbeta_kphi, phi_e = 1.82e3 / np.pi ** 2, np.deg2rad(69.0)\npref1, pref2 = 1.18e12, 1.67e12\nf_red, l_thr = 0.2, 1.9\n",
            "call": "log_intact_weight(theta, l_thr, f_red, beta_de, a_le)",
            "gold_call": "_oracle_log_intact_weight(theta, l_thr, f_red, beta_de, a_le)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\ndef run_model():\n    try:\n        log_intact_weight(theta, 0.0, f_red, beta_de, a_le)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_log_intact_weight(theta, 0.0, f_red, beta_de, a_le)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
