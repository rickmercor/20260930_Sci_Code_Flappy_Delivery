"""
Evaluate the older model's capture cross section on the same grid, anchored to the same value at the same reference temperature. That model puts the classical barrier of the configuration coordinate diagram into an activated exponential and carries one algebraic power of temperature in front of it; the source writes both and states which power it is. Return the curve normalised so that it agrees with the reference value at the reference temperature.

This is the model that most published analyses use, and reproducing it is what lets the error be quantified rather than asserted. Its defining property is that it depends on the defect only through the classical barrier, so two defects with very different lattice relaxation but the same barrier are assigned the same temperature dependence.

Returns
-------
A (n_temperature,) float64 array of capture cross sections in the same unit as the reference value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref):
    """Evaluate the older model's capture cross section on the same grid, anchored to the same
    value at the same reference temperature. A (n_temperature,) float64 array of capture
    cross sections in the same unit as the reference value."""
    return np.zeros(len(t_grid))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


K_B     = 8.617333262e-5

def _oracle_henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref):
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    barrier = float(barrier); t_ref = float(t_ref); sigma_ref = float(sigma_ref)
    if sigma_ref <= 0.0 or t_ref <= 0.0 or np.any(t_grid <= 0.0):
        raise ValueError("t_ref, sigma_ref and every temperature must be positive")
    return sigma_ref*(t_ref/t_grid)*np.exp(-barrier/K_B*(1.0/t_grid - 1.0/t_ref))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nbarrier = 0.5894988642\nt_grid = np.linspace(60.0, 620.0, 1121)\nt_ref = 300.0\nsigma_ref = 10.0\n',
         'call': 'henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref)',
         'gold_call': '_oracle_henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref)'},
        {'setup': 'import numpy as np\nbarrier = 0.5979195167\nt_grid = np.linspace(60.0, 620.0, 1121)\nt_ref = 300.0\nsigma_ref = 10.0\n',
         'call': 'henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref)',
         'gold_call': '_oracle_henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref)'},
        {'setup': 'import numpy as np\nbarrier = 0.25\nt_grid = np.linspace(120.0, 480.0, 361)\nt_ref = 250.0\nsigma_ref = 2.0\n',
         'call': 'henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref)',
         'gold_call': '_oracle_henry_lang_cross_section(barrier, t_grid, t_ref, sigma_ref)'},
    ]
