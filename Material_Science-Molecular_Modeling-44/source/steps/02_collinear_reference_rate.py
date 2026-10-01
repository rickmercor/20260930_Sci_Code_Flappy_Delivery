"""
Compute the collinear (one-dimensional) reference of the chain-scission problem: every bond is aligned with the force, so the bond-length potential of mean force is the Morse stretching energy tilted by the force (source, Sec. S3.2). Return the bonded minimum and the barrier top of that tilted potential (closed form), the activation barrier between them, and the bond-resolved transition-state-theory rate of one such bond evaluated from the full flux-over-population expression of the source (its Eq. S96) with the supplied kinetic prefactor, integrating the Boltzmann factor of the potential over the bonded interval [0, l_bar] with n_l trapezoid points. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

Single-bond mechanochemistry is usually described by transition-state theory on a bond potential tilted by the applied force: the force lowers the barrier between the bonded well and the dissociated state, and the rate follows the Arrhenius trend up to the critical force at which the well and the barrier merge. A chain of collinear, identically loaded bonds is the standard bead-string idealization against which three-dimensional conformational effects are measured.

Returns
-------
tuple of four floats (l_min, l_bar, barrier, rate): stationary points in l_e, barrier in D_e, rate in 1/s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def collinear_reference_rate(f_red: float, beta_de: float, a_le: float, prefactor: float,
                                     n_l: int = 20001) -> tuple:
    """Compute the collinear (one-dimensional) reference of the chain-scission problem: every bond is aligned with the force, so the bond-length potential of mean force is the Morse stretching energy tilted by the force (source, Sec. S3.2). Return the bonded minimum and the barrier top of that tilted potential (closed form), the activation barrier between them, and the bond-resolved transition-state-theory rate of one such bond evaluated from the full flux-over-population expression of the source (its Eq. S96) with the supplied kinetic prefactor, integrating the Boltzmann factor of the potential over the bonded interval [0, l_bar] with n_l trapezoid points. Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    f_red : float
        Reduced force f l_e / D_e, positive and below a_le / 2.
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    a_le : float
        Reduced Morse range parameter a l_e.
    prefactor : float
        Kinetic prefactor of the bond in 1/s (one-sided mean bond-length velocity divided by l_e).
    n_l : int
        Number of trapezoid points on [0, l_bar] (default 20001).

    Returns
    -------
    result : tuple
        (l_min, l_bar, barrier, rate) as native floats.

    Raises
    ------
    ValueError
        If f_red, beta_de, a_le or prefactor is not finite positive, n_l is not an integer >= 3, or f_red is at or above the collinear critical force a_le / 2.
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


def _oracle_collinear_reference_rate(f_red: float, beta_de: float, a_le: float, prefactor: float,
                                     n_l: int = 20001) -> tuple:
    for name, value in (("f_red", f_red), ("beta_de", beta_de), ("a_le", a_le), ("prefactor", prefactor)):
        _check_positive(name, value)
    if not isinstance(n_l, (int, np.integer)) or n_l < 3:
        raise ValueError("n_l must be an integer >= 3")
    if f_red >= 0.5 * a_le:
        raise ValueError("f_red must be below the collinear critical force a_le / 2")
    s = np.sqrt(1.0 - 2.0 * f_red / a_le)
    l_min = 1.0 - np.log((1.0 + s) / 2.0) / a_le          # tilted-Morse stationary points (S95)
    l_bar = 1.0 - np.log((1.0 - s) / 2.0) / a_le

    w_min = _morse(l_min, a_le) - f_red * l_min
    barrier = float(_morse(l_bar, a_le) - f_red * l_bar - w_min)
    l = np.linspace(0.0, l_bar, int(n_l))
    log_int = _log_trapezoid(-beta_de * (_morse(l, a_le) - f_red * l - w_min), l)
    rate = float(prefactor * np.exp(-beta_de * barrier - log_int))
    return float(l_min), float(l_bar), barrier, rate

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\ndef pack(t):\n    return (float(t[0]), float(t[1]), float(t[2]), float(np.log(t[3])))\n",
            "call": "pack(collinear_reference_rate(f_red, beta_de, a_le, pref2))",
            "gold_call": "pack(_oracle_collinear_reference_rate(f_red, beta_de, a_le, pref2))",
        },
        {
            "setup": "import numpy as np\nbeta_de, a_le = 279.0, 2.15\nbeta_kphi, phi_e = 1.82e3 / np.pi ** 2, np.deg2rad(69.0)\npref1, pref2 = 1.18e12, 1.67e12\nf_red = 0.6\ndef pack(t):\n    return (float(t[0]), float(t[1]), float(t[2]), float(np.log(t[3])))\n",
            "call": "pack(collinear_reference_rate(f_red, beta_de, a_le, pref2))",
            "gold_call": "pack(_oracle_collinear_reference_rate(f_red, beta_de, a_le, pref2))",
        },
        {
            "setup": "import numpy as np\nbeta_de, a_le = 279.0, 2.15\nbeta_kphi, phi_e = 1.82e3 / np.pi ** 2, np.deg2rad(69.0)\npref1, pref2 = 1.18e12, 1.67e12\nf_red = 0.05\ndef pack(t):\n    return (float(t[0]), float(t[1]), float(t[2]), float(np.log(t[3])))\n",
            "call": "pack(collinear_reference_rate(f_red, beta_de, a_le, pref2))",
            "gold_call": "pack(_oracle_collinear_reference_rate(f_red, beta_de, a_le, pref2))",
        },
        {
            "setup": "import numpy as np\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\ndef run_model():\n    try:\n        collinear_reference_rate(1.2, beta_de, a_le, pref2)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_collinear_reference_rate(1.2, beta_de, a_le, pref2)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
