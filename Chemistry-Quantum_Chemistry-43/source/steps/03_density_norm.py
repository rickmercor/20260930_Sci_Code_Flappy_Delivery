"""
Given the grid values f of a real periodic function on the uniform grid of the cell [0, a), return its norm in the source's density space for periodic systems, evaluated with Fourier-space (spectral) arithmetic. Raise ValueError if f is not a one-dimensional array with at least 8 points or a is not positive.

The source's density space is chosen to penalise rapid oscillations, which is what an inversion that targets a potential needs. Its norm is what the Moreau-Yosida term measures and what the source's Figure 1 uses to report the distance between the proximal and the reference density.

Returns
-------
float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def density_norm(f: "np.ndarray", a: float) -> float:
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
 
 
def _oracle_density_norm(f: "np.ndarray", a: float) -> float:
    """Eq 11: the norm of the density space X = H^{-1}_{per,hom},
    ||f||_X = sqrt(Σ_{G≠0} |f̂_G|^2 / |G|^2) = sqrt(<f, J f>). The G = 0 coefficient (the mass)
    carries no norm, which is why X is used for density DIFFERENCES."""
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 1 or f.size < 8 or a <= 0.0:
        raise ValueError("f must be a 1-D grid array with at least 8 points and a > 0")
    G = _wavenumbers(a, f.size)
    c = _coef(f, a)
    nz = G != 0.0
    return float(np.sqrt(np.sum(np.abs(c[nz]) ** 2 / G[nz] ** 2)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "a = 10.0\nx = np.arange(64) * a / 64\nf = np.cos(2 * np.pi * x / a) + 0.5 * np.sin(6 * np.pi * x / a)",
            "call": "density_norm(f, a)",
            "gold_call": "_oracle_density_norm(f, a)",
        },
        {
            "setup": "a = 10.0\nf = _oracle_external_potential(a, 128) + 0.5",
            "call": "density_norm(f, a)",
            "gold_call": "_oracle_density_norm(f, a)",
        },
        {
            "setup": "a = 7.0\nx = np.arange(96) * a / 96\nf = np.exp(-((x - a / 2) / 1.5) ** 2)",
            "call": "density_norm(f, a)",
            "gold_call": "_oracle_density_norm(f, a)",
        },
    ]
