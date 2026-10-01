"""
Propagate the capture cross section across the temperature grid from its value at the reference temperature, using the closed-form result the source obtains by integrating its own expression for the temperature derivative. The integral runs from the reference temperature to each grid temperature and its integrand is built from the two energies of the previous step; carry it out with a cumulative Simpson rule on the given grid. The source's result also carries an algebraic prefactor in temperature that does not come from the integral; include it. Require the reference temperature to be a node of the grid and raise if it is not, so that the integral is anchored without interpolation.

The source does not evaluate the absolute rate; it derives the logarithmic temperature derivative and integrates it, so that one measured cross section at one temperature fixes the whole curve. That is what makes the model usable on experimental data, and it is also why the absolute electronic coupling never has to be known.

Returns
-------
A (n_temperature,) float64 array of capture cross sections in the same unit as the reference value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma, t_ref, sigma_ref):
    """Propagate the capture cross section across the temperature grid from its value at the
    reference temperature, using the closed-form result the source obtains by integrating
    its own expression for the temperature derivative. A (n_temperature,) float64 array of
    capture cross sections in the same unit as the reference value."""
    return np.zeros(len(t_grid))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import cumulative_simpson


K_B     = 8.617333262e-5

def _oracle_nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma,
                              t_ref, sigma_ref):
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    t_ref = float(t_ref); sigma_ref = float(sigma_ref)
    if t_grid.size < 3 or np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing with at least three nodes")
    if sigma_ref <= 0.0 or t_ref <= 0.0:
        raise ValueError("t_ref and sigma_ref must be positive")
    node = int(np.argmin(np.abs(t_grid-t_ref)))
    if abs(t_grid[node]-t_ref) > 1.0e-9*max(1.0, t_ref):
        raise ValueError("t_ref must coincide with a node of t_grid")
    tab = _oracle_lineshape_function(q_elements, hw, delta_e, t_grid, broaden, n_sigma)
    integrand = (tab[:, 1]-tab[:, 2])/(K_B*t_grid**2)
    cum = cumulative_simpson(integrand, x=t_grid, initial=0.0)
    cum = cum - cum[node]
    return sigma_ref*np.sqrt(t_ref/t_grid)*np.exp(cum)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndef _fx_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):\n    _hbar2 = 4.18005956e-3\n    ell = np.sqrt(_hbar2 / hw)\n    def _fx_basis(lo, hi):\n        u = np.linspace(lo, hi, n_quad)\n        out = np.empty((n_phonon + 2, n_quad))\n        out[0] = u\n        out[1] = np.pi ** -0.25 * np.exp(-0.5 * u * u)\n        if n_phonon >= 1:\n            out[2] = np.sqrt(2.0) * u * out[1]\n        for k in range(2, n_phonon + 1):\n            out[k + 1] = np.sqrt(2.0 / k) * u * out[k] - np.sqrt((k - 1.0) / k) * out[k - 1]\n        return out\n    tab = _fx_basis(u_lo, u_hi)\n    u_grid, phi = tab[0], tab[1:]\n    du = u_grid[1] - u_grid[0]\n    shift = delta_q / ell\n    phi_f = _fx_basis(u_lo - shift, u_hi - shift)[1:]\n    return ((phi * u_grid * du) @ phi_f.T) * ell\nq_elements = _fx_coordinate_matrix_elements(1.20, 0.038, 220, -35.0, 60.0, 47501)\nhw = 0.038\ndelta_e = 1.02\nt_grid = np.linspace(60.0, 620.0, 1121)\nbroaden = 0.012\nn_sigma = 5.0\nt_ref = 300.0\nsigma_ref = 10.0\n',
         'call': 'nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma, t_ref, sigma_ref)',
         'gold_call': '_oracle_nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma, t_ref, sigma_ref)'},
        {'setup': 'import numpy as np\ndef _fx_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):\n    _hbar2 = 4.18005956e-3\n    ell = np.sqrt(_hbar2 / hw)\n    def _fx_basis(lo, hi):\n        u = np.linspace(lo, hi, n_quad)\n        out = np.empty((n_phonon + 2, n_quad))\n        out[0] = u\n        out[1] = np.pi ** -0.25 * np.exp(-0.5 * u * u)\n        if n_phonon >= 1:\n            out[2] = np.sqrt(2.0) * u * out[1]\n        for k in range(2, n_phonon + 1):\n            out[k + 1] = np.sqrt(2.0 / k) * u * out[k] - np.sqrt((k - 1.0) / k) * out[k - 1]\n        return out\n    tab = _fx_basis(u_lo, u_hi)\n    u_grid, phi = tab[0], tab[1:]\n    du = u_grid[1] - u_grid[0]\n    shift = delta_q / ell\n    phi_f = _fx_basis(u_lo - shift, u_hi - shift)[1:]\n    return ((phi * u_grid * du) @ phi_f.T) * ell\nq_elements = _fx_coordinate_matrix_elements(4.90, 0.038, 220, -35.0, 60.0, 47501)\nhw = 0.038\ndelta_e = 1.02\nt_grid = np.linspace(60.0, 620.0, 1121)\nbroaden = 0.012\nn_sigma = 5.0\nt_ref = 300.0\nsigma_ref = 10.0\n',
         'call': 'nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma, t_ref, sigma_ref)',
         'gold_call': '_oracle_nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma, t_ref, sigma_ref)'},
        {'setup': 'import numpy as np\ndef _fx_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):\n    _hbar2 = 4.18005956e-3\n    ell = np.sqrt(_hbar2 / hw)\n    def _fx_basis(lo, hi):\n        u = np.linspace(lo, hi, n_quad)\n        out = np.empty((n_phonon + 2, n_quad))\n        out[0] = u\n        out[1] = np.pi ** -0.25 * np.exp(-0.5 * u * u)\n        if n_phonon >= 1:\n            out[2] = np.sqrt(2.0) * u * out[1]\n        for k in range(2, n_phonon + 1):\n            out[k + 1] = np.sqrt(2.0 / k) * u * out[k] - np.sqrt((k - 1.0) / k) * out[k - 1]\n        return out\n    tab = _fx_basis(u_lo, u_hi)\n    u_grid, phi = tab[0], tab[1:]\n    du = u_grid[1] - u_grid[0]\n    shift = delta_q / ell\n    phi_f = _fx_basis(u_lo - shift, u_hi - shift)[1:]\n    return ((phi * u_grid * du) @ phi_f.T) * ell\nq_elements = _fx_coordinate_matrix_elements(2.50, 0.045, 140, -26.0, 40.0, 24001)\nhw = 0.045\ndelta_e = 0.90\nt_grid = np.linspace(100.0, 500.0, 801)\nbroaden = 0.020\nn_sigma = 4.0\nt_ref = 250.0\nsigma_ref = 3.5\n',
         'call': 'nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma, t_ref, sigma_ref)',
         'gold_call': '_oracle_nmp_cross_section(q_elements, hw, delta_e, t_grid, broaden, n_sigma, t_ref, sigma_ref)'},
    ]
