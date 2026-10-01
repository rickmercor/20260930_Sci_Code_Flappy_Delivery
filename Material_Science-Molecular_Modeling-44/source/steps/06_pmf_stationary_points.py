"""
Locate the bonded minimum and the barrier top of the constrained potential of mean force of bond `bond_index` (the profile of the previous step) and return them together with the activation barrier W_i(l_bar) - W_i(l_min) in D_e. Bracket sign changes of the derivative on n_scan equally spaced lengths in [l_lo, l_hi], take the first minimum and the first maximum beyond it, and refine each root to 1e-13 in l. Call the previous step's function for the profile and its derivative. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

The rupture threshold of each bond is placed variationally at the top of its potential of mean force, and the activation barrier of the bond is the height of that top above the bonded well. At finite force the profile of every bond has a metastable well and a barrier; they merge at the bond's critical force, beyond which no barrier exists.

Returns
-------
tuple of three floats (l_min, l_bar, barrier): stationary points in l_e and activation barrier in D_e.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pmf_stationary_points(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", log_i: "np.ndarray",
                                  bond_index: int, f_red: float, beta_de: float, a_le: float,
                                  l_lo: float = 0.8, l_hi: float = 6.0, n_scan: int = 2601) -> tuple:
    """Locate the bonded minimum and the barrier top of the constrained potential of mean force of bond `bond_index` (the profile of the previous step) and return them together with the activation barrier W_i(l_bar) - W_i(l_min) in D_e. Bracket sign changes of the derivative on n_scan equally spaced lengths in [l_lo, l_hi], take the first minimum and the first maximum beyond it, and refine each root to 1e-13 in l. Call the previous step's function for the profile and its derivative. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid strictly inside (0, pi), increasing, shape (n_theta,).
    w_theta : np.ndarray
        Positive quadrature weights of the theta grid, shape (n_theta,).
    kernel : np.ndarray
        Bending kernel of shape (n_theta, n_theta).
    log_i : np.ndarray
        Log intact weights of all bonds at their thresholds, shape (n_bonds, n_theta).
    bond_index : int
        Index of the selected bond, 0 .. n_bonds - 1.
    f_red : float
        Reduced force f l_e / D_e (positive).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    l_lo : float
        Lower end of the scan interval in l_e (default 0.8).
    l_hi : float
        Upper end of the scan interval in l_e (default 6.0).
    n_scan : int
        Number of equally spaced scan lengths (default 2601).

    Returns
    -------
    result : tuple
        (l_min, l_bar, barrier) as native floats.

    Raises
    ------
    ValueError
        If the profile step raises for the supplied data, l_lo or l_hi is not finite positive with l_hi > l_lo, n_scan is not an integer >= 3, or the profile has no bonded minimum or no barrier top on the scan interval (force at or above the critical force).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _oracle_pmf_stationary_points(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", log_i: "np.ndarray",
                                  bond_index: int, f_red: float, beta_de: float, a_le: float,
                                  l_lo: float = 0.8, l_hi: float = 6.0, n_scan: int = 2601) -> tuple:
    for name, value in (("l_lo", l_lo), ("l_hi", l_hi)):
        _check_positive(name, value)
    if l_hi <= l_lo:
        raise ValueError("l_hi must exceed l_lo")
    if not isinstance(n_scan, (int, np.integer)) or n_scan < 3:
        raise ValueError("n_scan must be an integer >= 3")
    grid = np.linspace(l_lo, l_hi, int(n_scan))
    profile = _oracle_constrained_pmf(theta, w_theta, kernel, log_i, bond_index, grid, f_red, beta_de, a_le)
    d = profile[1]
    idx = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    mins = [i for i in idx if d[i] < 0.0]
    if not mins:
        raise ValueError("no bonded minimum of the potential of mean force on the scan interval")
    i_min = mins[0]
    maxs = [i for i in idx if d[i] > 0.0 and i > i_min]
    if not maxs:
        raise ValueError("no barrier top of the potential of mean force: force at or above the critical force")

    slope = lambda l: float(_oracle_constrained_pmf(theta, w_theta, kernel, log_i, bond_index, np.array([l]),
                                                    f_red, beta_de, a_le)[1, 0])
    l_min = brentq(slope, grid[i_min], grid[i_min + 1], xtol=1e-13, rtol=4.0 * np.finfo(float).eps)
    l_bar = brentq(slope, grid[maxs[0]], grid[maxs[0] + 1], xtol=1e-13, rtol=4.0 * np.finfo(float).eps)
    ends = _oracle_constrained_pmf(theta, w_theta, kernel, log_i, bond_index, np.array([l_min, l_bar]),
                                   f_red, beta_de, a_le)[0]
    return float(l_min), float(l_bar), float(ends[1] - ends[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le) for i in range(5)])\nbond_index = 0\n",
            "call": "pmf_stationary_points(theta, w_theta, kernel, log_i, bond_index, f_red, beta_de, a_le)",
            "gold_call": "_oracle_pmf_stationary_points(theta, w_theta, kernel, log_i, bond_index, f_red, beta_de, a_le)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le)])\nbond_index = 0\n",
            "call": "pmf_stationary_points(theta, w_theta, kernel, log_i, bond_index, f_red, beta_de, a_le)",
            "gold_call": "_oracle_pmf_stationary_points(theta, w_theta, kernel, log_i, bond_index, f_red, beta_de, a_le)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le = 279.0, 2.15\nbeta_kphi, phi_e = 1.82e3 / np.pi ** 2, np.deg2rad(69.0)\npref1, pref2 = 1.18e12, 1.67e12\nf_red = 0.2\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le)])\nbond_index = 0\n",
            "call": "pmf_stationary_points(theta, w_theta, kernel, log_i, bond_index, f_red, beta_de, a_le)",
            "gold_call": "_oracle_pmf_stationary_points(theta, w_theta, kernel, log_i, bond_index, f_red, beta_de, a_le)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le)])\ndef run_model():\n    try:\n        pmf_stationary_points(theta, w_theta, kernel, log_i, 0, 1.2, beta_de, a_le)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_pmf_stationary_points(theta, w_theta, kernel, log_i, 0, 1.2, beta_de, a_le)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
