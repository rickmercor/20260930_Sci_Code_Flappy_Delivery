"""
Compute the low-frequency capacitance of the injected-carrier model at the voltage V. Return C / C_geo, the relative excess capacitance Delta C / C_geo = C / C_geo - 1, its inverse (Delta C / C_geo)^-1, the total contact factor eta(V) = eta_an + eta_cat, and, for comparison, the relative excess capacitance Delta C_weak / C_geo of the weak-injection limit, the limit in which both Debye lengths are much larger than d. Raise ValueError if par does not have 14 entries, if V is not below V_bi,0, or if the closed form gives a non-positive capacitance.

The capacitance of an m-i-m diode exceeds the geometric value by an amount governed by the energy-level bending at the contacts, and the regime of two weakly injecting contacts behaves differently from the ohmic regime.

Returns
-------
A (5,) float64 array [C / C_geo, Delta C / C_geo, (Delta C / C_geo)^-1, eta(V), Delta C_weak / C_geo].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def analytic_capacitance(voltage: float, par: "np.ndarray") -> "np.ndarray":
    """Derive the low-frequency capacitance of the injected-carrier model at the voltage V, in
    closed form from the effective built-in potential and the contact factors of the previous
    step. A (5,) float64 array [C / C_geo, Delta C / C_geo, (Delta C / C_geo)^-1, eta(V),
    Delta C_weak / C_geo].
    Parameters
    ----------
    voltage : float
        Applied bias V in V, below the nominal built-in potential V_bi,0.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If par does not have 14 entries, if V is not below V_bi,0, or if the
        closed form gives a non-positive capacitance.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


_Q = 1.602176634e-19

_KB = 1.380649e-23

_E0 = 8.8541878128e-12

def _oracle_analytic_capacitance(voltage: float, par: "np.ndarray") -> "np.ndarray":
    par = np.asarray(par, dtype=float).ravel()
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    V = float(voltage); VT = par[0]; Vbi0 = par[1]; d = par[10] * 1e-9
    Vb, eta_an, eta_cat, eb, wa, wc, xs = _oracle_effective_built_in_potential(V, par)
    eta = eta_an + eta_cat
    c_rel = 1.0 / (1.0 - 2.0 * VT / (Vb - V) * eta)
    if not (c_rel > 0.0):
        raise ValueError("the analytic capacitance is not defined this close to the built-in potential")
    dc_rel = c_rel - 1.0
    inv = ((Vb - V) - 2.0 * eta * VT) / (2.0 * eta * VT)               # (dC/Cgeo)^-1 = q/(2 eta kT) [Vbi(V) - V - 2 eta kT/q]
    p_an = np.exp(par[5]); n_cat = np.exp(par[6]); ee = par[11] * _E0
    dc_weak = 2.0 * (_KB * (VT * _Q / _KB)) ** 2 * d / (_Q * (Vbi0 - V) ** 3) * (p_an + n_cat) / (ee / d)   # Eq. 13, relative to Cgeo
    return np.array([c_rel, dc_rel, inv, eta, dc_weak], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nvoltage = 0.0\n',
         'call': 'analytic_capacitance(voltage, par)',
         'gold_call': '_oracle_analytic_capacitance(voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nvoltage = 0.8\n',
         'call': 'analytic_capacitance(voltage, par)',
         'gold_call': '_oracle_analytic_capacitance(voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.25, 0.25, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.25, 0.25, 1.0e-8, 5.0e-9, 0.1)\nvoltage = 0.5\n',
         'call': 'analytic_capacitance(voltage, par)',
         'gold_call': '_oracle_analytic_capacitance(voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\nvoltage = -0.3\n',
         'call': 'analytic_capacitance(voltage, par)',
         'gold_call': '_oracle_analytic_capacitance(voltage, par_gold)'},
    ]
