"""
Given the grid values v of a real periodic potential on the uniform grid of the cell [0, a), return its norm in the source's potential space for periodic systems, evaluated with Fourier-space (spectral) arithmetic. Raise ValueError if v is not a one-dimensional array with at least 8 points or a is not positive.

Potentials in the source's periodic setting live in the dual of its density space, with a norm the source defines in Fourier terms. Errors of inverted potentials are reported by the source in exactly this norm.

Returns
-------
float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def potential_norm(v: "np.ndarray", a: float) -> float:
    """float."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _wavenumbers(a, M):
    return 2.0 * np.pi * np.fft.fftfreq(int(M), d=1.0 / int(M)) / float(a)
 
def _coef(f, a):
    """Orthonormal-plane-wave Fourier coefficients of grid values f."""
    M = f.size
    return np.fft.fft(f) * (float(a) / M) / np.sqrt(float(a))
 
 
def _oracle_potential_norm(v: "np.ndarray", a: float) -> float:
    """Eq 9: the norm of the potential space X* = H^1_{per,hom},
    ||v||_{X*} = sqrt(Σ_G |G|^2 |v̂_G|^2) = sqrt(∫ |v'(x)|^2 dx). A constant shift of v has
    norm zero, which is the gauge freedom of potentials."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or v.size < 8 or a <= 0.0:
        raise ValueError("v must be a 1-D grid array with at least 8 points and a > 0")
    G = _wavenumbers(a, v.size)
    c = _coef(v, a)
    return float(np.sqrt(np.sum(G ** 2 * np.abs(c) ** 2)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "a = 10.0\nx = np.arange(64) * a / 64\nv = np.cos(2 * np.pi * x / a) + 2.0",
            "call": "potential_norm(v, a)",
            "gold_call": "_oracle_potential_norm(v, a)",
        },
        {
            "setup": "a = 10.0\nv = _oracle_external_potential(a, 128)",
            "call": "potential_norm(v, a)",
            "gold_call": "_oracle_potential_norm(v, a)",
        },
        {
            "setup": "a = 7.0\nx = np.arange(96) * a / 96\nv = np.sin(4 * np.pi * x / a) ** 2",
            "call": "potential_norm(v, a)",
            "gold_call": "_oracle_potential_norm(v, a)",
        },
    ]
