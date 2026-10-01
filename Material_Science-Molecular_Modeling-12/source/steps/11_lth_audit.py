"""
Run the complete structure-aware audit of the LTh parametrisation for one macroscopic reference state, state = (temperature in K, mass density in kg/m^3, pressure in Pa, specific isochoric heat capacity in J/(kg K), isothermal compressibility in 1/Pa, thermal expansion coefficient in 1/K), and a list of (R_cut*, f_cut) pairs, chaining the previous steps: reduce the state, determine the first-estimate LTh parameters, and for each cutoff solve the self-consistent HNC structure on the grid (n_grid, dr), recover its mean field, the corrected-density derivative zeta and the interaction coefficients [W_n] and [W_nn] at that mean field, verify that the HNC solution for the pair potential built from that [W_n] reproduces the structure (report the largest absolute difference), evaluate the equations of state and the density fluctuations, the response implied at the first-estimate parameters, the calibration of pi00 to the reduced reference pressure and the response at the calibrated pi00. Return a table with one head row followed by one row per cutoff. The head row holds [sum over the cutoffs of (calibrated pi00 minus first-estimate pi00), sum over the cutoffs of the calibrated pi00, T*, P*, C_V*, alpha*, theta0, n00, pi00, kappa_T, alpha, C_V, zeros]. Each cutoff row holds [R_cut*, f_cut, nb, n, zeta, W_n, W_nn, P, mean-field pair term, fluctuation pair term, fluctuation triplet term, U/N, variance of the primitive density, relative fluctuation, second-order energy correction, g at the first grid node, largest g, position of the largest g, largest absolute difference between the re-solved and the self-consistent g, implied kappa at the first-estimate parameters, implied alpha at the first-estimate parameters, calibrated pi00, calibrated minus first-estimate pi00, n of the calibrated model, U/N of the calibrated model, implied kappa at the calibrated pi00, implied alpha at the calibrated pi00]. Raise ValueError if state does not have six entries or if no cutoff is given.

For liquid argon (125.7 K, 1419.7 kg/m^3, 85.31 MPa, c_V = 520 J/(kg K), kappa_T = 1.49e-9 1/Pa, alpha = 2.64e-3 1/K, M_w = 0.040 kg/mol, phi = 5) the source uses the cutoffs R_cut* = 1.3365, 1.6839 and 2.1564 with f_cut = 1.41, 1.35 and 1.33 and reports for its HNC route densities n* = 1.034, 1.019, 1.010 and pressures P* = 0.169, 0.148, 0.136 against the nominal P* = 0.127, and states that at the largest cutoff the reference particle pressure would have to be lowered by 0.0088 to compensate; the supercritical state (418.8 K, 695.99 kg/m^3, 85.31 MPa, c_V = 356 J/(kg K), kappa_T = 6.83e-9 1/Pa, alpha = 1.97e-3 1/K) is treated with the same cutoffs. The audit quantifies the exact calibration and the response coefficients that the source does not compute.

Returns
-------
A (1 + n_cutoffs, 27) float64 array; row 0 is the head row and row c the row of cutoff c, with the columns listed in the description.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas):
    """Run the complete structure-aware audit of the LTh parametrisation for one macroscopic
    reference state, state = (temperature in K, mass density in kg/m^3, pressure in Pa,
    specific isochoric heat capacity in J/(kg K), isothermal compressibility in 1/Pa,
    thermal expansion coefficient in 1/K), and a list of (R_cut*, f_cut) pairs, chaining the
    previous steps: reduce the state, determine the first-estimate LTh parameters, and for
    each cutoff solve the self-consistent HNC structure on the grid (n_grid, dr), recover
    its mean field, the corrected-density derivative zeta and the interaction coefficients
    [W_n] and [W_nn] at that mean field, verify that the HNC solution for the pair potential
    built from that [W_n] reproduces the structure (report the largest absolute difference),
    evaluate the equations of state and the density fluctuations, the response implied at
    the first-estimate parameters, the calibration of pi00 to the reduced reference pressure
    and the response at the calibrated pi00. A (1 + n_cutoffs, 27) float64 array; row 0 is
    the head row and row c the row of cutoff c, with the columns listed in the description."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _kernel(r, rcut):
    w = np.where(r < rcut, 15.0/(2.0*np.pi*rcut**3)*(1.0 - r/rcut)**2, 0.0)
    wp = np.where(r < rcut, -15.0/(np.pi*rcut**4)*(1.0 - r/rcut), 0.0)
    return w, wp

def _grid(n_grid, dr):
    r = dr*np.arange(1, n_grid + 1)
    k = np.pi*np.arange(1, n_grid + 1)/((n_grid + 1)*dr)
    return r, k

def _oracle_lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas):
    if len(state) != 6:
        raise ValueError("state must hold temperature, mass density, pressure, specific heat, compressibility, expansion")
    if len(cutoffs) == 0:
        raise ValueError("at least one cutoff is required")
    red = _oracle_reduced_state(*state, molar_mass, cg_degree)
    t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar = [float(x) for x in red[:6]]
    params = _oracle_lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar)
    r, k = _grid(n_grid, dr)
    vol = 4.0*np.pi*dr*r*r
    rows = []
    for rcut, fcut in cutoffs:
        w, wp = _kernel(r, rcut)
        g = _oracle_self_consistent_structure(t_star, cbar, rcut, fcut, params, n_grid, dr)
        nb = cbar*float(np.sum(vol*w*g))
        pv = _oracle_particle_volume(nb, rcut, fcut)
        coef = _oracle_interaction_coefficients(nb, t_star, rcut, fcut, params)
        g_check = _oracle_hnc_structure(2.0*float(coef[3])*w/t_star, cbar, dr)
        eos = _oracle_eos_state(g, t_star, cbar, rcut, fcut, params, dr)
        fl = _oracle_density_fluctuations(g, t_star, cbar, rcut, fcut, params, dr)
        rc0 = _oracle_response_coefficients(t_star, cbar, rcut, fcut, params, n_grid, dr, deltas)
        cal = _oracle_calibrate_reference_pressure(t_star, cbar, rcut, fcut, params, p_star, n_grid, dr)
        q = [float(x) for x in params]; q[2] = float(cal[0])
        rc1 = _oracle_response_coefficients(t_star, cbar, rcut, fcut, q, n_grid, dr, deltas)
        i = int(np.argmax(g))
        rows.append([rcut, fcut, float(eos[0]), float(eos[1]), float(pv[1]), float(coef[3]), float(coef[4]),
                     float(eos[2]), float(eos[3]), float(eos[4]), float(eos[5]), float(eos[6]),
                     float(fl[0]), float(fl[1]), float(fl[3]),
                     float(g[0]), float(g[i]), float(r[i]), float(np.max(np.abs(g_check - g))),
                     float(rc0[2]), float(rc0[3]),
                     float(cal[0]), float(cal[0]) - float(params[2]), float(cal[2]), float(cal[3]), float(rc1[2]), float(rc1[3])])
    body = np.array(rows)
    head = np.zeros(body.shape[1])
    head[0] = float(np.sum(body[:, 22]))          # total calibration shift of the reference particle pressure
    head[1] = float(np.sum(body[:, 21]))
    head[2] = t_star
    head[3] = p_star
    head[4] = cv_bar
    head[5] = alpha_bar
    head[6:12] = params
    return np.vstack([head, body])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\n# normal: benchmark liquid state with one production cutoff\nstate = (125.7, 1419.7, 85.31e6, 520.0, 1.49e-9, 2.64e-3)\ncutoffs = [(2.1564, 1.33)]\nmolar_mass = 0.040\ncg_degree = 5\nn_grid = 100\ndr = 0.025\ndeltas = (1.0e-2, 1.0e-2)\n',
         'call': 'lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas)',
         'gold_call': '_oracle_lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas)'},
        {'setup': 'import numpy as np\n# boundary: one cutoff, minimum cg_degree = 1 and minimum HNC grid length n_grid = 8\nstate = (418.8, 695.99, 85.31e6, 356.0, 6.83e-9, 1.97e-3)\ncutoffs = [(1.0, 1.0)]\nmolar_mass = 0.040\ncg_degree = 1\nn_grid = 8\ndr = 0.15\ndeltas = (1.0e-2, 1.0e-2)\n',
         'call': 'lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas)',
         'gold_call': '_oracle_lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas)'},
        {'setup': 'import numpy as np\n# edge: two different cutoffs in one audit, exercising aggregation over multiple rows\nstate = (418.8, 695.99, 85.31e6, 356.0, 6.83e-9, 1.97e-3)\ncutoffs = [(1.0, 1.5), (1.5, 1.5)]\nmolar_mass = 0.040\ncg_degree = 5\nn_grid = 32\ndr = 0.05\ndeltas = (1.0e-2, 1.0e-2)\n',
         'call': 'lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas)',
         'gold_call': '_oracle_lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas)'},
    ]
