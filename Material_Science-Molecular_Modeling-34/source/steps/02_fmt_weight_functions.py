"""
Build the planar fundamental-measure (FMT) weight functions of a hard sphere.

Fundamental measure theory represents the hard-sphere excess free energy through weighted densities $n_\alpha(z) = \int dz'\, \rho(z') w_\alpha(z - z')$. For a planar geometry the three-dimensional weight functions integrate over the transverse plane to a set of one-dimensional profiles supported on $|t| \le R$: two scalar surface/volume weights and a vector weight, together with the four remaining scalar/vector weights obtained by the standard radial rescalings. The scalar weights must reproduce the exact three-dimensional geometric moments (the sphere surface area and volume).

Returns
-------
numpy float64 array of shape (6, M): rows [w0,w1,w2,w3,wv1,wv2]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fmt_weight_functions(R: float, dz: float) -> np.ndarray:
    r"""Build the six planar hard-sphere FMT weight arrays.

    R: finite positive hard-sphere radius.
    dz: finite positive grid spacing, with dz <= R.

    Numerical representation:
        Use float64 offsets t = np.arange(-R, R + dz / 2, dz).
        Let M = t.size; M is not required to be odd, and t is not
        required to contain zero. Return rows in the order
        [w0, w1, w2, w3, wv1, wv2], with shape (6, M).

        Evaluate the planar sphere polynomials on this offset array.
        Use equal rectangular quadrature weights, not endpoint weights.
        Uniformly rescale w2 and w3 separately so that
        dz * sum(w2) = 4 * pi * R**2 and
        dz * sum(w3) = 4 * pi * R**3 / 3.
        Obtain w1 and w0 from the corrected w2 by the standard radial
        rescalings. Use wv2 = 2 * pi * t and wv1 = wv2 / (4 * pi * R).
        Do not symmetrize, recenter, clip, or add grid points.
        This fixed sampling convention also applies when R / dz is
        nonintegral; the last offset is the one selected by np.arange.

    Returns:
        A fresh numpy float64 array of shape (6, M).

    Raises:
        ValueError: if R or dz is not finite and positive, or dz > R.
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

def _oracle_fmt_weight_functions(R, dz):
    """Planar FMT weights on |t|<=R; scalar weights rescaled to exact 3D moments."""
    Rr = _scalar(R, "R", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    if dd > Rr:
        raise ValueError("dz must not exceed R")
    t = np.arange(-Rr, Rr + dd / 2, dd)
    w3 = np.pi * (Rr ** 2 - t ** 2)
    w2 = np.full_like(t, 2 * np.pi * Rr)
    wv2 = 2 * np.pi * t
    w3 *= (4 * np.pi * Rr ** 3 / 3) / (w3.sum() * dd)
    w2 *= (4 * np.pi * Rr ** 2) / (w2.sum() * dd)
    w1 = w2 / (4 * np.pi * Rr)
    w0 = w2 / (4 * np.pi * Rr ** 2)
    wv1 = wv2 / (4 * np.pi * Rr)
    return np.vstack([w0, w1, w2, w3, wv1, wv2]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'R=0.5\n'
                'dz=0.02\n'
            ),
            "call": 'fmt_weight_functions(R, dz)',
            "gold_call": '_oracle_fmt_weight_functions(R, dz)',
        },
        {
            "setup": (
                'R=1.0\n'
                'dz=0.02\n'
            ),
            "call": 'fmt_weight_functions(R, dz)',
            "gold_call": '_oracle_fmt_weight_functions(R, dz)',
        },
        {
            "setup": (
                'R=0.75\n'
                'dz=0.025\n'
            ),
            "call": 'fmt_weight_functions(R, dz)',
            "gold_call": '_oracle_fmt_weight_functions(R, dz)',
        },
        {
            "setup": (
                'R=0.5\n'
                'dz=0.05\n'
            ),
            "call": 'fmt_weight_functions(R, dz)',
            "gold_call": '_oracle_fmt_weight_functions(R, dz)',
        },
        {
            "setup": (
                'R=0.5\n'
                'dz=0.04\n'
            ),
            "call": 'fmt_weight_functions(R, dz)',
            "gold_call": '_oracle_fmt_weight_functions(R, dz)',
        },
    ]
