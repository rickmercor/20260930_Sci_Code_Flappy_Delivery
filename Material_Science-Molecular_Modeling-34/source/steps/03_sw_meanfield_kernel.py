"""
Build the planar-projected mean-field kernel of a square-well attraction.

In the mean-field (random-phase) treatment the attractive contribution to the free energy is a convolution of the density with the pair attraction. For a square well of uniform depth acting out to a finite range, projecting the three-dimensional attraction onto the interface normal gives a one-dimensional kernel whose integral equals the full three-dimensional integrated strength of the well.

Returns
-------
numpy float64 array of shape (M2,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def sw_meanfield_kernel(
    beps: float, rng: float, dz: float
) -> np.ndarray:
    r"""Build the planar-projected mean-field square-well kernel.

    beps: finite positive reduced well depth, beta * epsilon.
    rng: finite positive outer interaction range, in sigma1 units.
    dz: finite positive spacing, with dz <= rng.

    Numerical representation:
        Use float64 offsets t = np.arange(-rng, rng + dz / 2, dz).
        Let M2 = t.size; M2 is not required to be odd, and t is not
        required to contain zero. Evaluate the transverse-integrated
        square-well polynomial at every offset selected by this array.
        Do not recenter, clip, or renormalize the sampled values.

        The continuum kernel integral equals the three-dimensional
        well strength. The discrete rectangular sum approximates that
        integral; it is not constrained to equal it exactly at finite dz.
        Recover the physical separation-range convention from the paper.

    Returns:
        A fresh numpy float64 array of shape (M2,).

    Raises:
        ValueError: if any argument is not finite and positive,
            or if dz > rng.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve


def _f64(a):
    return np.asarray(a, dtype=np.float64)

def _scalar(x, name, positive=False):
    v = float(x)
    if not np.isfinite(v):
        raise ValueError("non-finite " + name)
    if positive and v <= 0.0:
        raise ValueError("non-positive " + name)
    return v

def _grid(x, name):
    a = _f64(x)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        raise ValueError("bad 1D array " + name)
    return a

def _profile(P):
    P = _f64(P)
    if P.ndim != 2 or P.shape[0] != 2 or P.shape[1] < 8 or not np.all(np.isfinite(P)):
        raise ValueError("bad (2,N) profile array")
    return P

def _weights(W):
    W = _f64(W)
    if W.ndim != 2 or W.shape[0] != 6 or W.shape[1] < 1 or not np.all(np.isfinite(W)):
        raise ValueError("bad (6,M) weight array")
    return W

def _pad_conv(f, w, dz):
    hw = len(w) // 2
    fp = np.pad(f, hw, mode="edge")
    return (fftconvolve(fp, w, mode="same") * dz)[hw:hw + len(f)]

def _muhs_ex(r):
    e = np.pi * r / 6.0
    return (8 * e - 9 * e ** 2 + 3 * e ** 3) / (1 - e) ** 3

def _wb_derivs(n0, n1, n2, n3, nv1, nv2):
    n3 = np.clip(n3, 1e-10, 1 - 1e-9)
    s = 1 - n3
    Lg = np.log(s)
    g3 = (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2)
    dh = ((n3 - 2 * s * Lg) * n3 * s - (n3 + s ** 2 * Lg) * 2 * (1 - 2 * n3)) / (n3 ** 3 * s ** 3)
    g3p = dh / (36 * np.pi)
    dp0 = -Lg
    dp1 = n2 / s
    dp2 = n1 / s + (3 * n2 ** 2 - 3 * nv2 ** 2) * g3
    dp3 = n0 / s + (n1 * n2 - nv1 * nv2) / s ** 2 + (n2 ** 3 - 3 * n2 * nv2 ** 2) * g3p
    dpv1 = -nv2 / s
    dpv2 = -nv1 / s - 6 * n2 * nv2 * g3
    return dp0, dp1, dp2, dp3, dpv1, dpv2

def _oracle_sw_meanfield_kernel(beps, rng, dz):
    """Transverse integral of -beps over the disk of radius sqrt(rng^2 - t^2)."""
    be = _scalar(beps, "beps", positive=True)
    rg = _scalar(rng, "rng", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    if dd > rg:
        raise ValueError("dz must not exceed rng")
    t = np.arange(-rg, rg + dd / 2, dd)
    return (-np.pi * be * (rg ** 2 - t ** 2)).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'beps=1.0\n'
                'rng=1.5\n'
                'dz=0.02\n'
            ),
            "call": 'sw_meanfield_kernel(beps, rng, dz)',
            "gold_call": '_oracle_sw_meanfield_kernel(beps, rng, dz)',
        },
        {
            "setup": (
                'beps=1.0954451\n'
                'rng=2.0\n'
                'dz=0.02\n'
            ),
            "call": 'sw_meanfield_kernel(beps, rng, dz)',
            "gold_call": '_oracle_sw_meanfield_kernel(beps, rng, dz)',
        },
        {
            "setup": (
                'beps=0.6\n'
                'rng=1.75\n'
                'dz=0.025\n'
            ),
            "call": 'sw_meanfield_kernel(beps, rng, dz)',
            "gold_call": '_oracle_sw_meanfield_kernel(beps, rng, dz)',
        },
        {
            "setup": (
                'beps=1.2\n'
                'rng=1.5\n'
                'dz=0.05\n'
            ),
            "call": 'sw_meanfield_kernel(beps, rng, dz)',
            "gold_call": '_oracle_sw_meanfield_kernel(beps, rng, dz)',
        },
        {
            "setup": (
                'beps=1.0\n'
                'rng=1.5\n'
                'dz=0.04\n'
            ),
            "call": 'sw_meanfield_kernel(beps, rng, dz)',
            "gold_call": '_oracle_sw_meanfield_kernel(beps, rng, dz)',
        },
    ]
