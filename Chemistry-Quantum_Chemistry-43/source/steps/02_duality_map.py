"""
Given the grid values f of a real periodic function on the uniform grid of the cell [0, a), return the grid values of its image under the duality mapping of the source's density space, i.e. the map that sends an element of the density space to the element of the potential space that represents it (the source's Definition 1 applied to its periodic Sobolev setting), using Fourier-space (spectral) arithmetic. Raise ValueError if f is not a one-dimensional array with at least 8 points or a is not positive.

Densities and potentials live in dual Sobolev spaces in the source's periodic setting. The duality mapping identifies the two and is what turns a density difference into a potential; its exact form, including how the constant Fourier mode is treated, is fixed by the source's choice of spaces.

Returns
-------
ndarray of float64 with shape (M,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def duality_map(f: "np.ndarray", a: float) -> "np.ndarray":
    """ndarray of float64 with shape (M,)."""
    return result

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
 
def _from_coef(c, a):
    M = c.size
    return np.real(np.fft.ifft(c) * np.sqrt(float(a)) * M / float(a))
 
 
def _oracle_duality_map(f: "np.ndarray", a: float) -> "np.ndarray":
    """Eq 12: the duality (Riesz) map J : X -> X* of the periodic homogeneous Sobolev pair,
    J(f) = Σ_{G≠0} (f̂_G / |G|^2) e_G, i.e. the zero-mean solution of −J(f)'' = f − mean(f).
    The G = 0 coefficient of f carries no norm and is dropped."""
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 1 or f.size < 8 or a <= 0.0:
        raise ValueError("f must be a 1-D grid array with at least 8 points and a > 0")
    G = _wavenumbers(a, f.size)
    c = _coef(f, a)
    out = np.zeros_like(c)
    nz = G != 0.0
    out[nz] = c[nz] / G[nz] ** 2
    return _from_coef(out, a)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "a = 10.0\nx = np.arange(64) * a / 64\nf = np.cos(2 * np.pi * x / a) + 0.5 * np.sin(6 * np.pi * x / a)",
            "call": "duality_map(f, a)",
            "gold_call": "_oracle_duality_map(f, a)",
        },
        {
            "setup": "a = 10.0\nf = _oracle_external_potential(a, 128)",
            "call": "duality_map(f, a)",
            "gold_call": "_oracle_duality_map(f, a)",
        },
        {
            "setup": "a = 7.0\nx = np.arange(96) * a / 96\nf = np.exp(-((x - a / 2) / 1.5) ** 2) + 3.0",
            "call": "duality_map(f, a)",
            "gold_call": "_oracle_duality_map(f, a)",
        },
    ]
