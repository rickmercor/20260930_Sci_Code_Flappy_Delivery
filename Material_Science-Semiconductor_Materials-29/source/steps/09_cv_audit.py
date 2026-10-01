"""
Run the complete chain for every diode design in design_table, a list of rows (d in nm, eps, T in K, N_c, N_v in 1/m^3, E_g in eV, phi_an, phi_cat in eV, mu_n, mu_p in m^2/(V s), zeta), on the grid of N intervals, and return one table. For each design convert the parameters, evaluate the effective built-in potential and the closed-form capacitance at zero bias and at the probe bias V_probe = V_bi(0) / 2, solve the exact drift-diffusion states at both biases and evaluate the exact capacitances, evaluate the field profile of the injected-carrier model at mid-layer x = d / 2 at the probe bias, the weak-injection excess capacitance at the probe bias, the built-in potential extraction over the fit voltages and the validity threshold at the given level with the given march step. Each design row holds the 22 columns [V_bi,0, V_bi(0), eta(0), C_analytic(0) / C_geo, C_exact(0) / C_geo, delta(0) in percent, V_probe, C_analytic(V_probe) / C_geo, C_exact(V_probe) / C_geo, delta(V_probe) in percent, Q / (eps eps0 E_mid) at the probe, E at mid-layer at the probe in MV/m, x* at zero bias in nm, Delta C_weak / C_geo at the probe, Delta C / Delta C_weak at the probe, S, eta_ext, V_bi,ext, V_bi,ext - V_bi(0), V_thr, V_bi(V_thr) - V_thr, z at the threshold]. Row 0 is the head row [V_thr of the first design, number of designs, N, level, zeros]. Raise ValueError if design_table is empty or a row does not have 11 entries, or if N is not an even integer of at least 4.

The head scalar, the forward bias up to which the source's analytical capacitance of the design-point diode is accurate to one percent, is what a practitioner needs before applying the built-in potential extraction to a measured C-V curve: beyond it the closed form leaves the drift-diffusion reference, and a few tenths of a volt further it ceases to have a solution at all.

Returns
-------
A (1 + n_designs, 22) float64 array; row 0 is the head row and row i the row of design i, with the columns listed in the description.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cv_audit(design_table: list, intervals: int, level: float, fit_voltages: "np.ndarray", march_step: float) -> "np.ndarray":
    """Run the complete chain for every diode design in design_table, a list of rows (d in nm,
    eps, T in K, N_c, N_v in 1/m^3, E_g in eV, phi_an, phi_cat in eV, mu_n, mu_p in m^2/(V
    s), zeta), on the grid of N intervals, and return one table. A (1 + n_designs, 22)
    float64 array; row 0 is the head row and row i the row of design i, with the columns
    listed in the description.
    Parameters
    ----------
    design_table : list
        Rows of 11 diode parameters, each row in the argument order of the
        parameter step.
    intervals : int
        Number N of grid intervals, an even integer of at least 4.
    level : float
        Deviation level in percent that defines the validity threshold.
    fit_voltages : np.ndarray
        Bias values in V at which the extraction protocol fits.
    march_step : float
        Bias step in V used to march up from zero bias.

    Raises
    ------
    ValueError
        If a design_table row does not hold the 11 diode parameters, or if N
        is not an even integer of at least 4.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cv_audit(design_table: list, intervals: int, level: float, fit_voltages: "np.ndarray", march_step: float) -> "np.ndarray":
    rows = [tuple(float(v) for v in r) for r in design_table]
    if not rows or any(len(r) != 11 for r in rows):
        raise ValueError("design_table rows must hold the 11 diode parameters")
    N = int(intervals)
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    fv = np.atleast_1d(np.asarray(fit_voltages, dtype=float))
    NC = 22
    out = np.zeros((1 + len(rows), NC))
    for i, r in enumerate(rows, 1):
        par = _oracle_diode_parameters(*r)
        eff0 = _oracle_effective_built_in_potential(0.0, par)
        ca0 = _oracle_analytic_capacitance(0.0, par)
        st0 = _oracle_drift_diffusion_state(0.0, N, par)
        ex0 = _oracle_exact_capacitance(st0, 0.0, par)
        Vp = 0.5 * eff0[0]
        prof = _oracle_field_and_carrier_profiles([0.5 * par[10]], Vp, par)
        cap = _oracle_analytic_capacitance(Vp, par)
        stp = _oracle_drift_diffusion_state(Vp, N, par)
        exp_ = _oracle_exact_capacitance(stp, Vp, par)
        ext = _oracle_built_in_extraction(fv, N, par)
        thr = _oracle_validity_threshold(level, N, par, march_step)
        out[i] = [par[1], eff0[0], eff0[1] + eff0[2], ca0[0], ex0[1], 100.0 * (ca0[0] / ex0[1] - 1.0),
                  Vp, cap[0], exp_[1], 100.0 * (cap[0] / exp_[1] - 1.0), exp_[2], prof[0, 0], eff0[6],
                  cap[4], cap[1] / cap[4], ext[0], ext[2], ext[3], ext[4], thr[0], thr[1], thr[2]]
    out[0, 0] = out[1, 19]
    out[0, 1] = len(rows); out[0, 2] = N; out[0, 3] = float(level)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndesign_table = [(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.02, 1.0e-8, 5.0e-9, 0.1)]\nintervals = 400\nlevel = 1.0\nfit_voltages = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])\nmarch_step = 0.05\n',
         'call': 'cv_audit(design_table, intervals, level, fit_voltages, march_step)',
         'gold_call': '_oracle_cv_audit(design_table, intervals, level, fit_voltages, march_step)'},
        {'setup': 'import numpy as np\ndesign_table = [(100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.02, 0.50, 1.0e-8, 5.0e-9, 0.1), (100.0, 3.5, 300.0, 1.0e26, 1.0e26, 1.5, 0.25, 0.25, 1.0e-8, 5.0e-9, 0.1)]\nintervals = 200\nlevel = 1.0\nfit_voltages = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])\nmarch_step = 0.05\n',
         'call': 'cv_audit(design_table, intervals, level, fit_voltages, march_step)',
         'gold_call': '_oracle_cv_audit(design_table, intervals, level, fit_voltages, march_step)'},
        {'setup': 'import numpy as np\ndesign_table = [(80.0, 3.0, 320.0, 5.0e25, 2.0e26, 1.7, 0.10, 0.15, 2.0e-8, 1.0e-8, 0.05)]\nintervals = 200\nlevel = 2.0\nfit_voltages = np.array([-0.1, 0.0, 0.1])\nmarch_step = 0.1\n',
         'call': 'cv_audit(design_table, intervals, level, fit_voltages, march_step)',
         'gold_call': '_oracle_cv_audit(design_table, intervals, level, fit_voltages, march_step)'},
    ]
