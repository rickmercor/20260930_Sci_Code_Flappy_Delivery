"""
Evaluate the source's Morse bond-stretching potential at the given bond length(s), for dissociation energy De, inverse range parameter a and equilibrium length le (the paper's Eq. S8).

Bond rupture is driven by stretching along a Morse well whose depth sets the dissociation energy.

Returns
-------
return array like l (float64): the bond stretching energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def morse_potential(l, De, a, le):
    """l: bond length(s); De, a, le: Morse parameters. Returns float64 array
    shaped like l with the source's Morse stretching energy (paper Eq. S8)."""
    return np.zeros_like(np.asarray(l, dtype=np.float64))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 1: Morse bond-stretching potential (Eq. S8)."""

import numpy as np


def _oracle_morse_potential(l, De, a, le):
    l = np.asarray(l, dtype=np.float64)
    return De * (1.0 - np.exp(-a * (l - le))) ** 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nl=_n.array([0.8,1.0,1.3,2.0,3.0]);De=1.0;a=2.15;le=1.0', "call": 'morse_potential(l, De, a, le)', "gold_call": '_oracle_morse_potential(l, De, a, le)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nl=_n.linspace(0.7,2.5,7);De=2.0;a=1.8;le=1.1', "call": 'morse_potential(l, De, a, le)', "gold_call": '_oracle_morse_potential(l, De, a, le)', "tol": 1e-10},
        {"setup": 'import numpy as _n\nl=_n.array([1.0,1.5,2.2]);De=0.5;a=2.6;le=0.95', "call": 'morse_potential(l, De, a, le)', "gold_call": '_oracle_morse_potential(l, De, a, le)', "tol": 1e-10},
    ]
