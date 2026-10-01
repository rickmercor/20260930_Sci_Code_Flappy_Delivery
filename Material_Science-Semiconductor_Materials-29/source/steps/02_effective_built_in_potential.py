"""
Compute the effective built-in potential of the injected-carrier model at an applied voltage V below V_bi,0, together with the quantities derived from it. Return V_bi(V) in V, the contact factors eta_an and eta_cat, the bulk field E_bulk in MV/m (negative for V < V_bi), the effective widths of the accumulation regions at the anode and at the cathode in nm, and the crossover position x* in nm at which the field branches of the two contacts take the same value. The relation that defines V_bi(V) is iterated to machine precision, stopping when the change falls below 1e-15 V. Raise ValueError if par does not have 14 entries, if V is not below V_bi,0, or if the iteration leaves the physical domain V_bi(V) > V.

The energy-level bending induced by the accumulation of injected carriers near the contacts lowers the built-in potential experienced by carriers inside the layer relative to the nominal work-function difference; the source shows that this bending, and hence V_bi(V), depends on the applied voltage, and that for ohmic contacts (Debye length much smaller than the layer) the contact factor tends to one at each contact, while for a non-injecting contact it vanishes.

Returns
-------
A (7,) float64 array [V_bi(V) in V, eta_an, eta_cat, E_bulk in MV/m, Delta_w_an in nm, Delta_w_cat in nm, x* in nm].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_built_in_potential(voltage: float, par: "np.ndarray") -> "np.ndarray":
    """Derive the effective built-in potential of the injected-carrier model at an applied
    voltage V below V_bi,0. A (7,) float64 array [V_bi(V) in V, eta_an, eta_cat, E_bulk in
    MV/m, Delta_w_an in nm, Delta_w_cat in nm, x* in nm].
    Parameters
    ----------
    voltage : float
        Applied bias V in V, below the nominal built-in potential V_bi,0.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If par does not have 14 entries; if V is not below V_bi,0; or if the
        iteration leaves the physical domain V_bi(V) > V.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_effective_built_in_potential(voltage: float, par: "np.ndarray") -> "np.ndarray":
    par = np.asarray(par, dtype=float).ravel()
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    V = float(voltage); VT, Vbi0 = par[0], par[1]; d = par[10] * 1e-9; lam_an = par[3] * 1e-9; lam_cat = par[4] * 1e-9
    if not (V < Vbi0):
        raise ValueError("the injected-carrier model requires V below the nominal built-in potential")
    Vb = Vbi0
    for it in range(10000):
        ga = np.sqrt(1.0 + (2.0 * VT * d / ((Vb - V) * lam_an)) ** 2)
        gc = np.sqrt(1.0 + (2.0 * VT * d / ((Vb - V) * lam_cat)) ** 2)
        Vn = Vbi0 - 2.0 * VT * np.log(0.5 * (ga + 1.0)) - 2.0 * VT * np.log(0.5 * (gc + 1.0))
        if not (Vn > V):
            raise ValueError("no physical solution of the effective built-in potential at this bias")
        if abs(Vn - Vb) < 1e-15:
            Vb = Vn; break
        Vb = Vn
    ga = np.sqrt(1.0 + (2.0 * VT * d / ((Vb - V) * lam_an)) ** 2); gc = np.sqrt(1.0 + (2.0 * VT * d / ((Vb - V) * lam_cat)) ** 2)
    eta_an = 1.0 - 1.0 / ga; eta_cat = 1.0 - 1.0 / gc
    ebulk = (V - Vb) / d
    w_an = 2.0 * VT * d / (Vb - V) * eta_an; w_cat = 2.0 * VT * d / (Vb - V) * eta_cat
    k = abs(ebulk) / (2.0 * VT)                                                     # q|E_bulk|/(2kT), 1/m
    aa = np.arcsinh(abs(ebulk) * lam_an / (2.0 * VT)); ac = np.arcsinh(abs(ebulk) * lam_cat / (2.0 * VT))
    xs = 0.5 * d + (ac - aa) / (2.0 * k)                                            # crossover of the two field branches
    return np.array([Vb, eta_an, eta_cat, ebulk * 1e-6, w_an * 1e9, w_cat * 1e9, xs * 1e9], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nvoltage = 0.0\n',
         'call': 'effective_built_in_potential(voltage, par)',
         'gold_call': '_oracle_effective_built_in_potential(voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)\nvoltage = 0.7\n',
         'call': 'effective_built_in_potential(voltage, par)',
         'gold_call': '_oracle_effective_built_in_potential(voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\npar_gold = _oracle_diode_parameters(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1)\nvoltage = -0.5\n',
         'call': 'effective_built_in_potential(voltage, par)',
         'gold_call': '_oracle_effective_built_in_potential(voltage, par_gold)'},
        {'setup': 'import numpy as np\npar = diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\npar_gold = _oracle_diode_parameters(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)\nvoltage = 0.3\n',
         'call': 'effective_built_in_potential(voltage, par)',
         'gold_call': '_oracle_effective_built_in_potential(voltage, par_gold)'},
    ]
