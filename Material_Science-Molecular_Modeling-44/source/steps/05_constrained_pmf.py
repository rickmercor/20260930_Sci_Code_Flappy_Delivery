"""
Evaluate, at the supplied lengths, the constrained potential of mean force of one bond of the chain and its length derivative. For bond `bond_index` (0-based; bond 0 is attached to the anchored atom), W_i(l) = -(1/beta) ln G_i(l), where G_i(l) is the configurational weight of the chain with bond i held at length l and every other bond j integrated over its allowed interval [0, l_j_thr] with the full three-dimensional bond-vector measure and the bending coupling between successive bonds (source, Secs. S1.4 and S2). Return a (2, n_l) array: row 0 is W_i(l) - W_i(l_e), i.e. the profile with its additive constant fixed by its value at l = l_e (= 1 in reduced units), row 1 is dW_i/dl. The polar-angle grid with its quadrature weights, the bending kernel of the kernel step and the log intact weights of all bonds (one row per bond, thresholds already applied) are supplied; the nearest-neighbour orientational correlations of the whole chain must be resolved exactly on this grid, without sampling. At l = 0 the profile is +inf and the derivative -inf. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

Holding one bond at a given length while the rest of the chain fluctuates inside the intact basin defines a free-energy profile along that bond, the potential of mean force. Because bending couples neighbouring orientations, the profile of a bond depends on where it sits in the chain: the surrounding subchains constrain which absolute orientations the bond can adopt, and the applied force acts on the bond through its polar angle.

Returns
-------
ndarray of float64, shape (2, n_l): row 0 = W_i(l) - W_i(l_e) in D_e, row 1 = dW_i/dl in D_e per l_e, at the supplied lengths.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def constrained_pmf(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", log_i: "np.ndarray",
                            bond_index: int, l_values: "np.ndarray", f_red: float, beta_de: float,
                            a_le: float) -> "np.ndarray":
    """Evaluate, at the supplied lengths, the constrained potential of mean force of one bond of the chain and its length derivative. For bond `bond_index` (0-based; bond 0 is attached to the anchored atom), W_i(l) = -(1/beta) ln G_i(l), where G_i(l) is the configurational weight of the chain with bond i held at length l and every other bond j integrated over its allowed interval [0, l_j_thr] with the full three-dimensional bond-vector measure and the bending coupling between successive bonds (source, Secs. S1.4 and S2). Return a (2, n_l) array: row 0 is W_i(l) - W_i(l_e), i.e. the profile with its additive constant fixed by its value at l = l_e (= 1 in reduced units), row 1 is dW_i/dl. The polar-angle grid with its quadrature weights, the bending kernel of the kernel step and the log intact weights of all bonds (one row per bond, thresholds already applied) are supplied; the nearest-neighbour orientational correlations of the whole chain must be resolved exactly on this grid, without sampling. At l = 0 the profile is +inf and the derivative -inf. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    theta : np.ndarray
        Polar-angle grid strictly inside (0, pi), increasing, shape (n_theta,).
    w_theta : np.ndarray
        Positive quadrature weights of the theta grid, shape (n_theta,).
    kernel : np.ndarray
        Bending kernel of shape (n_theta, n_theta) from the kernel step.
    log_i : np.ndarray
        Log intact weights of all bonds at their thresholds, shape (n_bonds, n_theta).
    bond_index : int
        Index of the selected bond, 0 .. n_bonds - 1 (0 = bond attached to the anchor).
    l_values : np.ndarray
        Lengths at which to evaluate the profile, in units of l_e, non-negative, shape (n_l,).
    f_red : float
        Reduced force f l_e / D_e (positive).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.

    Returns
    -------
    profile : np.ndarray
        Array of shape (2, n_l): relative profile and its derivative.

    Raises
    ------
    ValueError
        If the grid is invalid, the kernel is not a finite non-negative (n_theta, n_theta) array, log_i is not an (n_bonds, n_theta) array without NaN or +inf, bond_index is outside [0, n_bonds), l_values is not a one-dimensional finite non-negative array, f_red, beta_de or a_le is not finite positive, or the orientational weight underflows to zero.
    """
    return profile

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


def _morse(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return (1.0 - e) ** 2


def _morse_prime(l, a_le):
    e = np.exp(-a_le * (l - 1.0))
    return 2.0 * a_le * e * (1.0 - e)


def _angular_terms(theta, w_theta, log_w, f_red, beta_de, l):
    """For each l: log of  int w(theta) exp(beta f l cos theta) sin theta dtheta  and the cos-average under it."""
    l = np.atleast_1d(np.asarray(l, dtype=float))
    c = np.cos(theta)
    base = log_w + np.log(np.sin(theta)) + np.log(w_theta)
    e = base[None, :] + beta_de * f_red * l[:, None] * c[None, :]
    m = e.max(axis=1)
    z = np.exp(e - m[:, None])
    s = z.sum(axis=1)
    return m + np.log(s), (z * c[None, :]).sum(axis=1) / s


def _pmf_value(l, theta, w_theta, log_w, f_red, beta_de, a_le):
    """W(l) in D_e for one bond's log angular weight; accepts a scalar or an array of lengths (in l_e)."""
    la, _ = _angular_terms(theta, w_theta, log_w, f_red, beta_de, l)
    out = _morse(np.atleast_1d(l), a_le) - (2.0 * np.log(np.atleast_1d(l)) + la) / beta_de
    return float(out[0]) if np.ndim(l) == 0 else out


def _pmf_slope(l, theta, w_theta, log_w, f_red, beta_de, a_le):
    """W'(l) in D_e / l_e; accepts a scalar or an array of lengths."""
    _, cm = _angular_terms(theta, w_theta, log_w, f_red, beta_de, l)
    out = _morse_prime(np.atleast_1d(l), a_le) - (2.0 / np.atleast_1d(l)) / beta_de - f_red * cm
    return float(out[0]) if np.ndim(l) == 0 else out


def _messages_at(theta, w_theta, kernel, log_i, bond_index):
    """Normalized forward and backward message shapes arriving at bond `bond_index` (S72-S76)."""
    n_bonds = log_i.shape[0]
    left = np.full(theta.size, 1.0 / np.pi)
    for i in range(bond_index):                                    # bonds 1 .. bond_index (0-based i)
        shift = log_i[i].max()
        prop = kernel @ (left * np.exp(log_i[i] - shift) * w_theta)
        left = prop / (prop * w_theta).sum()
    right = np.full(theta.size, 1.0 / np.pi)
    for i in range(n_bonds - 1, bond_index, -1):                   # bonds N .. bond_index + 2
        shift = log_i[i].max()
        prop = kernel.T @ (right * np.exp(log_i[i] - shift) * w_theta)
        right = prop / (prop * w_theta).sum()
    return left, right


def _oracle_constrained_pmf(theta: "np.ndarray", w_theta: "np.ndarray", kernel: "np.ndarray", log_i: "np.ndarray",
                            bond_index: int, l_values: "np.ndarray", f_red: float, beta_de: float,
                            a_le: float) -> "np.ndarray":
    theta, w_theta = _check_grid(theta, w_theta)
    kernel = np.asarray(kernel, dtype=float)
    log_i = np.asarray(log_i, dtype=float)
    n = theta.size
    if kernel.shape != (n, n) or not np.all(np.isfinite(kernel)) or np.any(kernel < 0.0):
        raise ValueError("kernel must be a finite non-negative (n_theta, n_theta) array")
    if log_i.ndim != 2 or log_i.shape[1] != n or log_i.shape[0] < 1 or np.any(np.isnan(log_i)) or np.any(log_i == np.inf):
        raise ValueError("log_i must be an (n_bonds, n_theta) array without NaN or +inf")
    if not isinstance(bond_index, (int, np.integer)) or bond_index < 0 or bond_index >= log_i.shape[0]:
        raise ValueError("bond_index must be an integer in [0, n_bonds)")
    l_values = np.atleast_1d(np.asarray(l_values, dtype=float))
    if l_values.ndim != 1 or l_values.size < 1 or not np.all(np.isfinite(l_values)) or np.any(l_values < 0.0):
        raise ValueError("l_values must be a one-dimensional array of finite non-negative lengths")
    for name, value in (("f_red", f_red), ("beta_de", beta_de), ("a_le", a_le)):
        _check_positive(name, value)
    left, right = _messages_at(theta, w_theta, kernel, log_i, int(bond_index))
    weight = left * right
    norm = (weight * w_theta).sum()
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("angular weight lost normalization: messages underflowed")
    with np.errstate(divide="ignore"):
        log_w = np.log(np.pi * weight / norm)                      # mean-normalized angular weight (S107)
    args = (theta, w_theta, log_w, f_red, beta_de, a_le)
    with np.errstate(divide="ignore"):
        pmf = _pmf_value(l_values, *args) - _pmf_value(np.array([1.0]), *args)[0]
        slope = _pmf_slope(l_values, *args)
    return np.vstack([pmf, slope])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le) for i in range(5)])\nl_values = np.linspace(0.9, 2.2, 14)\nbond_index = 0\n",
            "call": "constrained_pmf(theta, w_theta, kernel, log_i, bond_index, l_values, f_red, beta_de, a_le)",
            "gold_call": "_oracle_constrained_pmf(theta, w_theta, kernel, log_i, bond_index, l_values, f_red, beta_de, a_le)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le)])\nl_values = np.linspace(0.95, 1.9, 8)\nbond_index = 0\n",
            "call": "constrained_pmf(theta, w_theta, kernel, log_i, bond_index, l_values, f_red, beta_de, a_le)",
            "gold_call": "_oracle_constrained_pmf(theta, w_theta, kernel, log_i, bond_index, l_values, f_red, beta_de, a_le)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, 0.0, phi_e)\nlog_i = np.array([log_intact_weight(theta, 1.75, f_red, beta_de, a_le) for i in range(4)])\nl_values = np.array([1.0, 1.1, 1.5, 1.75])\nbond_index = 2\n",
            "call": "constrained_pmf(theta, w_theta, kernel, log_i, bond_index, l_values, f_red, beta_de, a_le)",
            "gold_call": "_oracle_constrained_pmf(theta, w_theta, kernel, log_i, bond_index, l_values, f_red, beta_de, a_le)",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le) for i in range(5)])\nl_values = np.linspace(0.9, 2.2, 14)\ndef run_model():\n    try:\n        constrained_pmf(theta, w_theta, kernel, log_i, 5, l_values, f_red, beta_de, a_le)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_constrained_pmf(theta, w_theta, kernel, log_i, 5, l_values, f_red, beta_de, a_le)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
