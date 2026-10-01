"""
Run the whole analysis on the configuration and report it. Build the temperature grid, and for each lattice relaxation in turn reduce the level to its configuration coordinate diagram, build the phonon matrix elements, propagate the rigorous cross section and the older model's cross section from the same reference value, convert both to emission rates, and then for each fit window apply the constant-cross-section analysis to both. Accumulate, over every relaxation and every window, the base-ten logarithm of the ratio between the cross section the constant-cross-section analysis extracts from the rigorous rates and the true rigorous cross section at the reference temperature; that running total is the reported value. Record one row per relaxation and window, with the run-level quantities in a leading row.

The source compares twenty-one defects in twelve semiconductors and finds that the older model misassigns the temperature dependence of the capture cross section by up to six orders of magnitude at room temperature. This audit reproduces that comparison on a configuration where the two levels share a phonon energy, a transition energy and a thermodynamic level and differ only in lattice relaxation, which is the case the older model cannot tell apart at all.

Returns
-------
A (n_relaxation * n_window + 1, 9) float64 array. Row zero holds [reported total, oscillator length, ratio of the transition energy to the phonon energy, averaged initial-state phonon energy of the first relaxation at the reference temperature, quantum-statistical average oscillator energy at the reference temperature, lineshape function of the first relaxation at the reference temperature, squared Franck-Condon factor out of the initial ground state into the energy-conserving final state of the first relaxation, quadrature spacing of the reduced coordinate grid, 0]. Each later row holds [lattice relaxation, lower edge of the window in inverse kilokelvin, upper edge, apparent thermodynamic level in eV, decades of error in the extracted cross section, apparent level minus the true level in eV, base-ten logarithm of the apparent cross section in square centimetre, decades of error the older model would give, Huang-Rhys factor].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref, n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma, t_lo, t_hi, n_t, t_ref, windows, n_fit):
    """Run the whole analysis on the configuration and report it. A (n_relaxation * n_window +
    1, 9) float64 array. Row zero holds [reported total, oscillator length, ratio of the
    transition energy to the phonon energy, averaged initial-state phonon energy of the
    first relaxation at the reference temperature, quantum-statistical average oscillator
    energy at the reference temperature, lineshape function of the first relaxation at the
    reference temperature, squared Franck-Condon factor out of the initial ground state into
    the energy-conserving final state of the first relaxation, quadrature spacing of the
    reduced coordinate grid, 0]. Each later row holds [lattice relaxation, lower edge of the
    window in inverse kilokelvin, upper edge, apparent thermodynamic level in eV, decades of
    error in the extracted cross section, apparent level minus the true level in eV, base-
    ten logarithm of the apparent cross section in square centimetre, decades of error the
    older model would give, Huang-Rhys factor]."""
    return np.zeros((len(dq_values)*len(windows) + 1, 9))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


K_B     = 8.617333262e-5

ANG2_PER_CM2 = 1.0e-16

def _oracle_dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref,
                       n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma,
                       t_lo, t_hi, n_t, t_ref, windows, n_fit):
    dq_values = [float(x) for x in dq_values]
    if len(dq_values) == 0:
        raise ValueError("at least one lattice relaxation is required")
    t_grid = np.linspace(float(t_lo), float(t_hi), int(n_t))
    rows = []
    total = 0.0
    for dq in dq_values:
        cc = _oracle_configuration_coordinate(dq, hw, delta_e)
        qm = _oracle_coordinate_matrix_elements(dq, hw, n_phonon, u_lo, u_hi, n_quad)
        sigma_a2 = float(sigma_ref)/ANG2_PER_CM2
        sig_nmp = _oracle_nmp_cross_section(qm, hw, delta_e, t_grid, broaden, n_sigma,
                                            t_ref, sigma_a2)
        sig_hl = _oracle_henry_lang_cross_section(float(cc[3]), t_grid, t_ref, sigma_a2)
        e_nmp = _oracle_emission_rate(sig_nmp, t_grid, level, mstar, degeneracy)
        e_hl = _oracle_emission_rate(sig_hl, t_grid, level, mstar, degeneracy)
        sig_true = float(np.interp(t_ref, t_grid, sig_nmp))
        for w in windows:
            fit_nmp = _oracle_arrhenius_signature(e_nmp, t_grid, w, n_fit, mstar, degeneracy)
            fit_hl = _oracle_arrhenius_signature(e_hl, t_grid, w, n_fit, mstar, degeneracy)
            decades = float(np.log10(fit_nmp[1]/sig_true))
            total += decades
            rows.append([dq, float(w[0]), float(w[1]), float(fit_nmp[0]), decades,
                         float(fit_nmp[0]) - level,
                         float(np.log10(fit_nmp[1]*ANG2_PER_CM2)),
                         float(np.log10(fit_hl[1]/sig_true)), float(cc[1])])
    fc = _oracle_franck_condon_overlaps(dq_values[0], hw, n_phonon, u_lo, u_hi, n_quad)
    basis = _oracle_oscillator_basis(n_phonon, u_lo, u_hi, n_quad)
    audit = np.array(rows, dtype=float)
    head = np.zeros((1, audit.shape[1]))
    head[0, 0] = total
    head[0, 1] = float(_oracle_configuration_coordinate(dq_values[0], hw, delta_e)[0])
    head[0, 2] = delta_e/hw
    head[0, 3] = float(_oracle_lineshape_function(
        _oracle_coordinate_matrix_elements(dq_values[0], hw, n_phonon, u_lo, u_hi, n_quad),
        hw, delta_e, np.array([t_ref]), broaden, n_sigma)[0, 1])
    head[0, 4] = 0.5*hw/np.tanh(0.5*hw/(K_B*t_ref))
    head[0, 6] = float(fc[0, int(round(delta_e/hw))]**2)
    head[0, 7] = float(basis[0, 1]-basis[0, 0])
    head[0, 5] = float(_oracle_lineshape_function(
        _oracle_coordinate_matrix_elements(dq_values[0], hw, n_phonon, u_lo, u_hi, n_quad),
        hw, delta_e, np.array([t_ref]), broaden, n_sigma)[0, 0])
    return np.vstack([head, audit])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nhw = 0.038\ndelta_e = 1.02\nlevel = 0.62\ndq_values = [4.90, 1.20]\nmstar = 0.29\ndegeneracy = 2.0\nsigma_ref = 1.0e-15\nn_phonon = 180\nu_lo = -32.0\nu_hi = 55.0\nn_quad = 40001\nbroaden = 0.012\nn_sigma = 5.0\nt_lo = 60.0\nt_hi = 620.0\nn_t = 1121\nt_ref = 300.0\nwindows = [(2.0, 3.5), (6.0, 10.0)]\nn_fit = 29\n',
         'call': 'dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref, n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma, t_lo, t_hi, n_t, t_ref, windows, n_fit)',
         'gold_call': '_oracle_dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref, n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma, t_lo, t_hi, n_t, t_ref, windows, n_fit)'},
        {'setup': 'import numpy as np\nhw = 0.045\ndelta_e = 0.90\nlevel = 0.41\ndq_values = [2.50]\nmstar = 0.15\ndegeneracy = 1.0\nsigma_ref = 4.0e-16\nn_phonon = 140\nu_lo = -26.0\nu_hi = 40.0\nn_quad = 24001\nbroaden = 0.020\nn_sigma = 4.0\nt_lo = 100.0\nt_hi = 500.0\nn_t = 801\nt_ref = 250.0\nwindows = [(2.5, 4.0)]\nn_fit = 21\n',
         'call': 'dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref, n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma, t_lo, t_hi, n_t, t_ref, windows, n_fit)',
         'gold_call': '_oracle_dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref, n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma, t_lo, t_hi, n_t, t_ref, windows, n_fit)'},
        {'setup': 'import numpy as np\nhw = 0.038\ndelta_e = 1.02\nlevel = 0.62\ndq_values = [4.90, 3.00, 1.20]\nmstar = 0.29\ndegeneracy = 2.0\nsigma_ref = 1.0e-15\nn_phonon = 200\nu_lo = -34.0\nu_hi = 58.0\nn_quad = 44001\nbroaden = 0.012\nn_sigma = 5.0\nt_lo = 80.0\nt_hi = 600.0\nn_t = 1041\nt_ref = 300.0\nwindows = [(3.0, 5.0)]\nn_fit = 25\n',
         'call': 'dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref, n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma, t_lo, t_hi, n_t, t_ref, windows, n_fit)',
         'gold_call': '_oracle_dlts_audit(hw, delta_e, level, dq_values, mstar, degeneracy, sigma_ref, n_phonon, u_lo, u_hi, n_quad, broaden, n_sigma, t_lo, t_hi, n_t, t_ref, windows, n_fit)'},
    ]
