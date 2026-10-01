"""
Evaluate the bond-resolved transition-state-theory rate of one bond in 1/s from its tabulated constrained potential of mean force on a length grid that runs from 0 to the bond's barrier top (the last grid point) and from the bond's kinetic prefactor: the source's full flux-over-population expression (its Eq. S50), the prefactor times the Boltzmann factor of the profile at the barrier top divided by the integral of the Boltzmann factor of the profile over the bonded interval [0, l_bar], using the trapezoid rule on the supplied grid and working in the log domain. The additive constant of the profile is irrelevant; the value at l = 0 may be +inf (zero weight). Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

Transition-state theory gives the rate as the equilibrium one-way flux through the dividing surface divided by the population of the reactant basin. Along a single reaction coordinate with a potential of mean force this is a kinetic prefactor, the one-sided thermal average of the coordinate velocity, times the ratio of the Boltzmann weight at the barrier to the integrated weight of the well; the harmonic-well limit recovers the Arrhenius form.

Returns
-------
float, the bond-resolved rate in 1/s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bond_tst_rate(l_values: "np.ndarray", pmf_values: "np.ndarray", beta_de: float, prefactor: float) -> float:
    """Evaluate the bond-resolved transition-state-theory rate of one bond in 1/s from its tabulated constrained potential of mean force on a length grid that runs from 0 to the bond's barrier top (the last grid point) and from the bond's kinetic prefactor: the source's full flux-over-population expression (its Eq. S50), the prefactor times the Boltzmann factor of the profile at the barrier top divided by the integral of the Boltzmann factor of the profile over the bonded interval [0, l_bar], using the trapezoid rule on the supplied grid and working in the log domain. The additive constant of the profile is irrelevant; the value at l = 0 may be +inf (zero weight). Reduced units: lengths in units of the equilibrium bond length l_e, energies in units of the dissociation energy D_e, so beta_de = D_e/(k_B T), a_le = a l_e and f_red = f l_e / D_e; angles in radians.

    Parameters
    ----------
    l_values : np.ndarray
        Strictly increasing lengths in units of l_e from 0 to the barrier top, shape (n_l,), n_l >= 3.
    pmf_values : np.ndarray
        Constrained potential of mean force at l_values in D_e (any additive constant; +inf allowed at l = 0).
    beta_de : float
        Reduced dissociation energy D_e / (k_B T).
    prefactor : float
        Kinetic prefactor of the bond in 1/s.

    Returns
    -------
    rate : float
        Bond-resolved TST rate in 1/s.

    Raises
    ------
    ValueError
        If l_values is not a strictly increasing one-dimensional array starting at 0 with at least 3 points, pmf_values does not match it or contains NaN, -inf or a non-finite value beyond the first point, or beta_de or prefactor is not finite positive.
    """
    return rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize


def _check_positive(name, value):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _log_trapezoid(log_f, x):
    """log of the trapezoidal integral of exp(log_f) over x (log_f may contain -inf)."""
    dx = np.diff(x)
    log_w = np.log(np.concatenate(([dx[0] / 2.0], (dx[:-1] + dx[1:]) / 2.0, [dx[-1] / 2.0])))
    e = log_f + log_w
    m = e.max()
    return float(m + np.log(np.exp(e - m).sum()))


def _oracle_bond_tst_rate(l_values: "np.ndarray", pmf_values: "np.ndarray", beta_de: float, prefactor: float) -> float:
    l_values = np.asarray(l_values, dtype=float)
    pmf_values = np.asarray(pmf_values, dtype=float)
    if l_values.ndim != 1 or l_values.size < 3 or pmf_values.shape != l_values.shape:
        raise ValueError("l_values and pmf_values must be one-dimensional arrays of equal length >= 3")
    if not np.all(np.isfinite(l_values)) or l_values[0] != 0.0 or np.any(np.diff(l_values) <= 0.0):
        raise ValueError("l_values must increase strictly from 0 to the barrier top")
    if np.any(np.isnan(pmf_values)) or np.any(pmf_values == -np.inf) or not np.all(np.isfinite(pmf_values[1:])):
        raise ValueError("pmf_values must be finite (the value at l = 0 may be +inf)")
    for name, value in (("beta_de", beta_de), ("prefactor", prefactor)):
        _check_positive(name, value)
    w_ref = float(np.min(pmf_values[1:]))
    log_f = np.where(np.isfinite(pmf_values), -beta_de * (pmf_values - w_ref), -np.inf)
    log_int = _log_trapezoid(log_f, l_values)
    return float(prefactor * np.exp(-beta_de * (pmf_values[-1] - w_ref) - log_int))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le)])\nl_min, l_bar, _ = pmf_stationary_points(theta, w_theta, kernel, log_i, 0, f_red, beta_de, a_le)\nwell = np.linspace(0.0, l_bar, 20001)\nprof = constrained_pmf(theta, w_theta, kernel, log_i, 0, well, f_red, beta_de, a_le)[0]\n",
            "call": "np.log(bond_tst_rate(well, prof, beta_de, pref2))",
            "gold_call": "np.log(_oracle_bond_tst_rate(well, prof, beta_de, pref2))",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le) for i in range(5)])\nl_min, l_bar, _ = pmf_stationary_points(theta, w_theta, kernel, log_i, 0, f_red, beta_de, a_le)\nwell = np.linspace(0.0, l_bar, 20001)\nprof = constrained_pmf(theta, w_theta, kernel, log_i, 0, well, f_red, beta_de, a_le)[0]\npref2 = pref1\n",
            "call": "np.log(bond_tst_rate(well, prof, beta_de, pref2))",
            "gold_call": "np.log(_oracle_bond_tst_rate(well, prof, beta_de, pref2))",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le = 279.0, 2.15\nbeta_kphi, phi_e = 1.82e3 / np.pi ** 2, np.deg2rad(69.0)\npref1, pref2 = 1.18e12, 1.67e12\nf_red = 0.6\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le)])\nl_min, l_bar, _ = pmf_stationary_points(theta, w_theta, kernel, log_i, 0, f_red, beta_de, a_le)\nwell = np.linspace(0.0, l_bar, 20001)\nprof = constrained_pmf(theta, w_theta, kernel, log_i, 0, well, f_red, beta_de, a_le)[0]\n",
            "call": "np.log(bond_tst_rate(well, prof, beta_de, pref2))",
            "gold_call": "np.log(_oracle_bond_tst_rate(well, prof, beta_de, pref2))",
        },
        {
            "setup": "import numpy as np\nx, w = np.polynomial.legendre.leggauss(200)\ntheta, w_theta = 0.5 * np.pi * (x + 1.0), 0.5 * np.pi * w\nbeta_de, a_le, f_red = 279.19347375490736, 2.1487249999999998, 0.6988553811079767\nbeta_kphi, phi_e = 123.53693528978204, np.deg2rad(70.5)\npref1, pref2 = 1179777948830.073, 1668457975824.2007\nkernel = bending_kernel(theta, beta_kphi, phi_e)\nl_thr = collinear_reference_rate(f_red, beta_de, a_le, pref2)[1]\nlog_i = np.array([log_intact_weight(theta, l_thr, f_red, beta_de, a_le)])\nl_min, l_bar, _ = pmf_stationary_points(theta, w_theta, kernel, log_i, 0, f_red, beta_de, a_le)\nwell = np.linspace(0.0, l_bar, 20001)\nprof = constrained_pmf(theta, w_theta, kernel, log_i, 0, well, f_red, beta_de, a_le)[0]\ndef run_model():\n    try:\n        bond_tst_rate(well[1:], prof[1:], beta_de, pref2)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_bond_tst_rate(well[1:], prof[1:], beta_de, pref2)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
