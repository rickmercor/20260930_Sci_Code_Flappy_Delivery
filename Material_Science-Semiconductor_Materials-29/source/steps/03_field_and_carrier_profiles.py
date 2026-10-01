"""
Evaluate the injected-carrier model's field and carrier profiles at the positions x (nm, 0 <= x <= d) for an applied voltage V. Use the hyperbolic-cotangent field branch of the anode region for x up to the crossover x* of the previous step and the branch of the cathode region beyond it, both with the bulk field and the effective built-in potential of that step. Return, per position, the field in MV/m, the hole density relative to its contact value p / p_an in the hole-dominated region (zero elsewhere) and the electron density relative to its contact value n / n_cat in the electron-dominated region (zero elsewhere), the densities following from the Boltzmann relation with the electrostatic potential obtained by integrating the field from the respective contact. Raise ValueError if par does not have 14 entries or if a position lies outside the layer.

The field is strongly non-uniform within a few Debye lengths of an injecting contact and saturates to the bulk value inside the layer; the accumulation regions are highly conductive and act as virtual extensions of the contacts, effectively thinning the active layer, which is the origin of the voltage-dependent excess capacitance.

Returns
-------
An (m, 3) float64 array with columns [E in MV/m, p / p_an or 0, n / n_cat or 0] for the m positions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def field_and_carrier_profiles(positions: "np.ndarray", voltage: float, par: "np.ndarray") -> "np.ndarray":
    """Evaluate the injected-carrier model's field and carrier profiles at the positions x (nm,
    0 <= x <= d) for an applied voltage V. Use the hyperbolic-cotangent field branch of the
    anode region for x up to the crossover x* of the previous step and the branch of the
    cathode region beyond it, both with the bulk field and the effective built-in potential
    of that step. An (m, 3) float64 array with columns [E in MV/m, p / p_an or 0, n / n_cat
    or 0] for the m positions.
    Parameters
    ----------
    positions : np.ndarray
        Positions x in nm at which to evaluate the profiles, 0 <= x <= d.
    voltage : float
        Applied bias V in V.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If par does not have 14 entries, or if any position lies outside the
        active layer 0 <= x <= d.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_field_and_carrier_profiles(positions: "np.ndarray", voltage: float, par: "np.ndarray") -> "np.ndarray":
    par = np.asarray(par, dtype=float).ravel(); x = np.atleast_1d(np.asarray(positions, dtype=float)) * 1e-9
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    d = par[10] * 1e-9
    if x.ndim != 1 or x.size == 0 or np.any(x < 0.0) or np.any(x > d):
        raise ValueError("positions must lie inside the active layer 0 <= x <= d")
    VT = par[0]; lam_an = par[3] * 1e-9; lam_cat = par[4] * 1e-9
    Vb, eta_an, eta_cat, eb, wa, wc, xs_nm = _oracle_effective_built_in_potential(voltage, par)
    E = eb * 1e6; Ea = abs(E); xs = xs_nm * 1e-9
    k = Ea / (2.0 * VT)
    aa = np.arcsinh(Ea * lam_an / (2.0 * VT)); ac = np.arcsinh(Ea * lam_cat / (2.0 * VT))
    left = x <= xs
    Ex = np.where(left, E / np.tanh(k * x + aa), E / np.tanh(k * (d - x) + ac))
    phi_left = -(E / k) * (np.log(np.sinh(k * x + aa)) - np.log(np.sinh(aa)))           # phi(x) = -int_0^x E
    phi_right = (E / k) * (np.log(np.sinh(k * (d - x) + ac)) - np.log(np.sinh(ac)))      # phi(x) - phi(d)
    p_rel = np.where(left, np.exp(-phi_left / VT), 0.0)                                   # p / p_an in the hole region
    n_rel = np.where(left, 0.0, np.exp(phi_right / VT))                                   # n / n_cat in the electron region
    return np.stack([Ex * 1e-6, p_rel, n_rel], axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npositions = np.array([0.0, 1.0, 5.0, 50.0, 95.0, 100.0])\nvoltage = 0.3\n',
         'call': 'field_and_carrier_profiles(positions, voltage, par)',
         'gold_call': '_oracle_field_and_carrier_profiles(positions, voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\npositions = np.array([2.0, 40.0, 71.0, 72.0, 99.0])\nvoltage = 0.2\n',
         'call': 'field_and_carrier_profiles(positions, voltage, par)',
         'gold_call': '_oracle_field_and_carrier_profiles(positions, voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(200.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(200.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npositions = np.array([0.5, 100.0, 199.5])\nvoltage = -1.0\n',
         'call': 'field_and_carrier_profiles(positions, voltage, par)',
         'gold_call': '_oracle_field_and_carrier_profiles(positions, voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\npar_gold = _oracle_diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\npositions = np.linspace(0.0, 80.0, 17)\nvoltage = 0.5\n',
         'call': 'field_and_carrier_profiles(positions, voltage, par)',
         'gold_call': '_oracle_field_and_carrier_profiles(positions, voltage, par_gold)'},
    ]
