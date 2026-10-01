"""
Convert a capture cross section into the thermal emission rate that a transient spectroscopy experiment measures, using detailed balance: the rate is the cross section times the carrier thermal velocity times the effective density of states of the band, divided by the ratio of the degeneracies of the two charge states, times the Boltzmann factor of the thermodynamic level. Take the effective density of states to be two times the usual (2 pi m* k_B T / h^2) raised to the three halves. The thermal velocity is the one the source writes, including its numerical factor.

The emission rate is the only quantity the experiment actually records; the cross section and the level are inferred from its temperature dependence. Because the thermal velocity and the density of states together carry two powers of temperature, the standard analysis plots the logarithm of the rate divided by the square of the temperature, which is where the square in that plot comes from.

Returns
-------
A (n_temperature,) float64 array of emission rates in inverse seconds, for a cross section supplied in square Angstrom.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def emission_rate(sigma, t_grid, level, mstar, degeneracy):
    """Convert a capture cross section into the thermal emission rate that a transient
    spectroscopy experiment measures, using detailed balance: the rate is the cross section
    times the carrier thermal velocity times the effective density of states of the band,
    divided by the ratio of the degeneracies of the two charge states, times the Boltzmann
    factor of the thermodynamic level. A (n_temperature,) float64 array of emission rates in
    inverse seconds, for a cross section supplied in square Angstrom."""
    return np.zeros(len(t_grid))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


K_B     = 8.617333262e-5

H_PLANCK = 4.135667696e-15

M_E     = 5.6095886e-32

def _oracle_emission_rate(sigma, t_grid, level, mstar, degeneracy):
    sigma = np.asarray(sigma, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    level = float(level); mstar = float(mstar); degeneracy = float(degeneracy)
    if sigma.size != t_grid.size:
        raise ValueError("sigma and t_grid must have the same length")
    if mstar <= 0.0 or degeneracy <= 0.0:
        raise ValueError("mstar and degeneracy must be positive")
    m_eff = mstar*M_E
    v_th = np.sqrt(3.0*K_B*t_grid/m_eff)
    n_c = 2.0*(2.0*np.pi*m_eff*K_B*t_grid/(H_PLANCK*H_PLANCK))**1.5
    return (sigma*v_th*n_c/degeneracy)*np.exp(-level/(K_B*t_grid))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nt_grid = np.linspace(60.0, 620.0, 1121)\nsigma = np.full(t_grid.size, 10.0)\nlevel = 0.62\nmstar = 0.29\ndegeneracy = 2.0\n',
         'call': 'emission_rate(sigma, t_grid, level, mstar, degeneracy)',
         'gold_call': '_oracle_emission_rate(sigma, t_grid, level, mstar, degeneracy)'},
        {'setup': 'import numpy as np\ndef _fx_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):\n    _hbar2 = 4.18005956e-3\n    ell = np.sqrt(_hbar2 / hw)\n    def _fx_basis(lo, hi):\n        u = np.linspace(lo, hi, n_quad)\n        out = np.empty((n_phonon + 2, n_quad))\n        out[0] = u\n        out[1] = np.pi ** -0.25 * np.exp(-0.5 * u * u)\n        if n_phonon >= 1:\n            out[2] = np.sqrt(2.0) * u * out[1]\n        for k in range(2, n_phonon + 1):\n            out[k + 1] = np.sqrt(2.0 / k) * u * out[k] - np.sqrt((k - 1.0) / k) * out[k - 1]\n        return out\n    tab = _fx_basis(u_lo, u_hi)\n    u_grid, phi = tab[0], tab[1:]\n    du = u_grid[1] - u_grid[0]\n    shift = delta_q / ell\n    phi_f = _fx_basis(u_lo - shift, u_hi - shift)[1:]\n    return ((phi * u_grid * du) @ phi_f.T) * ell\nq = _fx_coordinate_matrix_elements(4.90, 0.038, 220, -35.0, 60.0, 47501)\nt_grid = np.linspace(60.0, 620.0, 1121)\nsigma = _oracle_nmp_cross_section(q, 0.038, 1.02, t_grid, 0.012, 5.0, 300.0, 10.0)\nlevel = 0.62\nmstar = 0.29\ndegeneracy = 2.0\n',
         'call': 'emission_rate(sigma, t_grid, level, mstar, degeneracy)',
         'gold_call': '_oracle_emission_rate(sigma, t_grid, level, mstar, degeneracy)'},
        {'setup': 'import numpy as np\nt_grid = np.linspace(120.0, 480.0, 361)\nsigma = _oracle_henry_lang_cross_section(0.25, t_grid, 250.0, 2.0)\nlevel = 0.41\nmstar = 0.15\ndegeneracy = 1.0\n',
         'call': 'emission_rate(sigma, t_grid, level, mstar, degeneracy)',
         'gold_call': '_oracle_emission_rate(sigma, t_grid, level, mstar, degeneracy)'},
    ]
