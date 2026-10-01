"""
Evaluate, at each requested temperature, the lineshape function of the transition rate and the two energies whose difference governs its temperature derivative. The lineshape function is the thermally weighted sum over initial phonon states of the squared matrix element of the previous step, with the final phonon state fixed by energy conservation; replace the energy-conserving delta by a normalised Gaussian of the given width in energy, summed over final states out to the given multiple of that width, and normalise the Gaussian weights over the retained final states. A final state is retained when its energy lies within that multiple of the width of the energy-conserving target, so the retained states are those whose quantum number differs from the target by at most the given multiple of the width divided by the phonon energy. Then return the averaged initial-state phonon energy that the source defines, whose weight is the product of the thermal occupation and the squared matrix element, and the ordinary quantum-statistical average energy of one oscillator at that temperature. Take the thermal occupation to be the normalised Boltzmann weight of the initial-state oscillator.

This is the heart of the source. The temperature dependence of the capture cross section is not governed by a fixed barrier; it is governed by the difference between an averaged phonon energy that the transition itself selects and the average energy the oscillator would have anyway. The selection is what the squared matrix element does to the weight: states whose phonon wavefunctions overlap the final state well are promoted, and for a strongly relaxing defect those are high-lying states far above the thermal average.

Returns
-------
An (n_temperature, 3) float64 array whose columns are [lineshape function, averaged initial-state phonon energy in eV, quantum-statistical average oscillator energy in eV].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma):
    """Evaluate, at each requested temperature, the lineshape function of the transition rate
    and the two energies whose difference governs its temperature derivative. An
    (n_temperature, 3) float64 array whose columns are [lineshape function, averaged
    initial-state phonon energy in eV, quantum-statistical average oscillator energy in eV]."""
    return np.zeros((len(np.atleast_1d(temperature)), 3))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


K_B     = 8.617333262e-5

def _oracle_lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma):
    q_elements = np.asarray(q_elements, dtype=float)
    hw = float(hw); delta_e = float(delta_e); broaden = float(broaden); n_sigma = float(n_sigma)
    temperature = np.atleast_1d(np.asarray(temperature, dtype=float))
    if q_elements.ndim != 2 or q_elements.shape[0] != q_elements.shape[1]:
        raise ValueError("q_elements must be a square matrix")
    if broaden <= 0.0 or np.any(temperature <= 0.0):
        raise ValueError("broaden and every temperature must be positive")
    n_max = q_elements.shape[0]-1
    m = np.arange(n_max+1, dtype=float)
    target = m + delta_e/hw
    reach = n_sigma*broaden/hw
    overlap = np.zeros(n_max+1)
    for i in range(n_max+1):
        lo = max(0, int(np.ceil(target[i] - reach)))
        hi = min(n_max, int(np.floor(target[i] + reach)))
        if hi < lo:
            continue
        n = np.arange(lo, hi+1)
        w = np.exp(-0.5*(((n-target[i])*hw/broaden)**2))
        tot = w.sum()
        if tot <= 0.0:
            continue
        overlap[i] = float(np.sum(w*q_elements[i, n]**2)/tot)
    out = np.empty((temperature.size, 3))
    for j, t in enumerate(temperature):
        x = np.exp(-hw/(K_B*t))
        rho = (1.0-x)*x**m
        weight = rho*overlap
        tot = weight.sum()
        out[j, 0] = tot
        out[j, 1] = float(np.sum((m+0.5)*hw*weight)/tot) if tot > 0.0 else np.nan
        out[j, 2] = 0.5*hw/np.tanh(0.5*hw/(K_B*t))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndef _fx_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):\n    _hbar2 = 4.18005956e-3\n    ell = np.sqrt(_hbar2 / hw)\n    def _fx_basis(lo, hi):\n        u = np.linspace(lo, hi, n_quad)\n        out = np.empty((n_phonon + 2, n_quad))\n        out[0] = u\n        out[1] = np.pi ** -0.25 * np.exp(-0.5 * u * u)\n        if n_phonon >= 1:\n            out[2] = np.sqrt(2.0) * u * out[1]\n        for k in range(2, n_phonon + 1):\n            out[k + 1] = np.sqrt(2.0 / k) * u * out[k] - np.sqrt((k - 1.0) / k) * out[k - 1]\n        return out\n    tab = _fx_basis(u_lo, u_hi)\n    u_grid, phi = tab[0], tab[1:]\n    du = u_grid[1] - u_grid[0]\n    shift = delta_q / ell\n    phi_f = _fx_basis(u_lo - shift, u_hi - shift)[1:]\n    return ((phi * u_grid * du) @ phi_f.T) * ell\nq_elements = _fx_coordinate_matrix_elements(1.20, 0.038, 220, -35.0, 60.0, 47501)\nhw = 0.038\ndelta_e = 1.02\ntemperature = np.array([100.0, 300.0, 500.0])\nbroaden = 0.012\nn_sigma = 5.0\n',
         'call': 'lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma)',
         'gold_call': '_oracle_lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma)'},
        {'setup': 'import numpy as np\ndef _fx_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):\n    _hbar2 = 4.18005956e-3\n    ell = np.sqrt(_hbar2 / hw)\n    def _fx_basis(lo, hi):\n        u = np.linspace(lo, hi, n_quad)\n        out = np.empty((n_phonon + 2, n_quad))\n        out[0] = u\n        out[1] = np.pi ** -0.25 * np.exp(-0.5 * u * u)\n        if n_phonon >= 1:\n            out[2] = np.sqrt(2.0) * u * out[1]\n        for k in range(2, n_phonon + 1):\n            out[k + 1] = np.sqrt(2.0 / k) * u * out[k] - np.sqrt((k - 1.0) / k) * out[k - 1]\n        return out\n    tab = _fx_basis(u_lo, u_hi)\n    u_grid, phi = tab[0], tab[1:]\n    du = u_grid[1] - u_grid[0]\n    shift = delta_q / ell\n    phi_f = _fx_basis(u_lo - shift, u_hi - shift)[1:]\n    return ((phi * u_grid * du) @ phi_f.T) * ell\nq_elements = _fx_coordinate_matrix_elements(4.90, 0.038, 220, -35.0, 60.0, 47501)\nhw = 0.038\ndelta_e = 1.02\ntemperature = np.linspace(80.0, 600.0, 14)\nbroaden = 0.012\nn_sigma = 5.0\n',
         'call': 'lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma)',
         'gold_call': '_oracle_lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma)'},
        {'setup': 'import numpy as np\ndef _fx_coordinate_matrix_elements(delta_q, hw, n_phonon, u_lo, u_hi, n_quad):\n    _hbar2 = 4.18005956e-3\n    ell = np.sqrt(_hbar2 / hw)\n    def _fx_basis(lo, hi):\n        u = np.linspace(lo, hi, n_quad)\n        out = np.empty((n_phonon + 2, n_quad))\n        out[0] = u\n        out[1] = np.pi ** -0.25 * np.exp(-0.5 * u * u)\n        if n_phonon >= 1:\n            out[2] = np.sqrt(2.0) * u * out[1]\n        for k in range(2, n_phonon + 1):\n            out[k + 1] = np.sqrt(2.0 / k) * u * out[k] - np.sqrt((k - 1.0) / k) * out[k - 1]\n        return out\n    tab = _fx_basis(u_lo, u_hi)\n    u_grid, phi = tab[0], tab[1:]\n    du = u_grid[1] - u_grid[0]\n    shift = delta_q / ell\n    phi_f = _fx_basis(u_lo - shift, u_hi - shift)[1:]\n    return ((phi * u_grid * du) @ phi_f.T) * ell\nq_elements = _fx_coordinate_matrix_elements(2.50, 0.045, 140, -26.0, 40.0, 24001)\nhw = 0.045\ndelta_e = 0.90\ntemperature = np.array([150.0, 450.0])\nbroaden = 0.020\nn_sigma = 4.0\n',
         'call': 'lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma)',
         'gold_call': '_oracle_lineshape_function(q_elements, hw, delta_e, temperature, broaden, n_sigma)'},
    ]
