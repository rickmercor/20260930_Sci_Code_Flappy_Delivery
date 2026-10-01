"""
Return the single-base transition rates of the force-dependent nucleation-zipper model in units of the attempt rate k_a, as the array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)]: k_ot is the terminal opening rate of the last base pair, k_a exp(dG_non / RT), set by the non-stacking free energy dG_non = dH_non - T dS_non (initiation plus terminal penalties); k_o(n) is the opening rate of the n-th base pair (n -> n - 1 closed pairs) and k_c(n) the closing rate of the (n - 1) -> n transition. At zero force every closing rate equals k_a and every internal opening rate equals k_a exp(dG_s / RT) with the stacking free energy dG_s = dH_s - T dS_s. Under force the ratio k_c(n) / k_o(n) must equal the force-dependent equilibrium constant exp(-dG_s / RT) exp(W_n(F) / k_B T), with W_n the mechanical work of the previous step; how the factor exp(W_n / k_B T) is apportioned between k_c(n) and k_o(n) is fixed by the source's placement of the transition state along the extension coordinate, and the source's placement must be used. Enthalpies in kJ/mol, entropies in kJ/(mol K), R = k_B N_A with N_A = 6.02214076e23.

Each base-pair transition has an attempt rate, an equilibrium constant fixed by the stacking thermodynamics and the mechanical work, and a transition-state position that decides how the work splits between the forward and backward rates. The terminal pair has no stacking partner and opens at a rate governed by the non-stacking initiation and end-penalty free energies instead.

Returns
-------
ndarray of float64, shape (2 n_bp - 1,), the rates [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)] in units of k_a.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transition_rates(force: float, n_bp: int, temperature: float, dh_stack: float, ds_stack: float,
                             dh_non: float, ds_non: float, persistence_length: float,
                             interphosphate_distance: float) -> "np.ndarray":
    """Return the single-base transition rates of the force-dependent nucleation-zipper model in units of the attempt rate k_a, as the array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)]: k_ot is the terminal opening rate of the last base pair, k_a exp(dG_non / RT), set by the non-stacking free energy dG_non = dH_non - T dS_non (initiation plus terminal penalties); k_o(n) is the opening rate of the n-th base pair (n -> n - 1 closed pairs) and k_c(n) the closing rate of the (n - 1) -> n transition. At zero force every closing rate equals k_a and every internal opening rate equals k_a exp(dG_s / RT) with the stacking free energy dG_s = dH_s - T dS_s. Under force the ratio k_c(n) / k_o(n) must equal the force-dependent equilibrium constant exp(-dG_s / RT) exp(W_n(F) / k_B T), with W_n the mechanical work of the previous step; how the factor exp(W_n / k_B T) is apportioned between k_c(n) and k_o(n) is fixed by the source's placement of the transition state along the extension coordinate, and the source's placement must be used. Enthalpies in kJ/mol, entropies in kJ/(mol K), R = k_B N_A with N_A = 6.02214076e23.

    Parameters
    ----------
    force : float
        Shear force in pN (> 0).
    n_bp : int
        Number of base pairs (>= 2).
    temperature : float
        Temperature in K (> 0).
    dh_stack : float
        Stacking enthalpy per base-pair step, kJ/mol.
    ds_stack : float
        Salt-corrected stacking entropy per step, kJ/(mol K).
    dh_non : float
        Non-stacking enthalpy (initiation plus terminal penalties), kJ/mol.
    ds_non : float
        Salt-corrected non-stacking entropy, kJ/(mol K).
    persistence_length : float
        lambda_ss in nm.
    interphosphate_distance : float
        l_ss in nm.

    Returns
    -------
    rates : np.ndarray
        Array [k_ot, k_o(2), ..., k_o(n_bp), k_c(2), ..., k_c(n_bp)].

    Raises
    ------
    ValueError
        If force, temperature or the polymer parameters are not finite and positive, n_bp < 2, or a thermodynamic parameter is not finite.
    """
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def _check_int(n, name, lo):
    if isinstance(n, bool) or int(n) != n or int(n) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(n)


def _thermal_energy_pn_nm(temperature):
    """k_B T in pN nm."""
    return 1.380649e-23 * temperature * 1e21


def _thermal_energy_kj_mol(temperature):
    """R T in kJ/mol."""
    return 1.380649e-23 * 6.02214076e23 * temperature / 1000.0


def _oracle_transition_rates(force: float, n_bp: int, temperature: float, dh_stack: float, ds_stack: float,
                             dh_non: float, ds_non: float, persistence_length: float,
                             interphosphate_distance: float) -> "np.ndarray":
    """[k_ot, k_o(2), ..., k_o(N), k_c(2), ..., k_c(N)] in units of k_a, eqs (6)-(8) with alpha = 1.

    Source convention (alpha = 1, transition state at the closed configuration): the whole mechanical work
    W_n(F) sits on the closing transition, k_c^{n-1->n} = exp(W_n / k_B T), and every opening rate stays
    force independent, k_o(n) = exp(dG_s / RT); k_ot = exp(dG_non / RT); dG = dH - T dS.
    """
    F = _check_pos(force, "force")
    N = _check_int(n_bp, "n_bp", 2)
    T = _check_pos(temperature, "temperature")
    for v, nm in ((dh_stack, "dh_stack"), (ds_stack, "ds_stack"), (dh_non, "dh_non"), (ds_non, "ds_non")):
        if isinstance(v, bool) or not np.isfinite(v):
            raise ValueError(f"{nm} must be finite")
    lam = _check_pos(persistence_length, "persistence_length")
    l_ss = _check_pos(interphosphate_distance, "interphosphate_distance")
    rt = _thermal_energy_kj_mol(T)
    kt = _thermal_energy_pn_nm(T)
    dg_s = float(dh_stack) - T * float(ds_stack)
    dg_non = float(dh_non) - T * float(ds_non)
    out = np.zeros(2 * N - 1, dtype=np.float64)
    out[0] = np.exp(dg_non / rt)
    for n in range(2, N + 1):
        w_n = _oracle_mechanical_work(n, F, N, T, lam, l_ss)
        out[n - 1] = np.exp(dg_s / rt)                 # opening n -> n-1: alpha = 1, no force factor
        out[N + n - 2] = np.exp(w_n / kt)              # closing n-1 -> n carries exp(W_n / k_B T)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\nforce, n_bp, temperature = 3.0, 9, 303.15\n",
            "call": "np.asarray(transition_rates(force, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss))",
            "gold_call": "np.asarray(_oracle_transition_rates(force, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss))",
        },
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\nforce, n_bp, temperature = 9.0, 9, 303.15\n",
            "call": "np.asarray(transition_rates(force, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss))",
            "gold_call": "np.asarray(_oracle_transition_rates(force, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss))",
        },
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\nforce, n_bp, temperature = 10.0, 2, 298.15\n",
            "call": "np.asarray(transition_rates(force, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss))",
            "gold_call": "np.asarray(_oracle_transition_rates(force, n_bp, temperature, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss))",
        },
        {
            "setup": "import numpy as np\ndh_stack, ds_stack, dh_non, ds_non = -7.6 * 4.184, -0.0920, 4.6 * 4.184, 0.0502\nlam_ss, l_ss = 0.77, 0.7\ndef run_model():\n    try:\n        transition_rates(3.0, 1, 303.15, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_transition_rates(3.0, 1, 303.15, dh_stack, ds_stack, dh_non, ds_non, lam_ss, l_ss)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
