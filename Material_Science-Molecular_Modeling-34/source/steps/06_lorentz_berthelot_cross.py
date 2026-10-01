"""
Apply the model's mixing rules to obtain the cross square-well interaction parameters.

An additive binary square-well mixture fixes the inter-component interaction from the two pure-component parameters. The cross diameter is additive, the cross depth is the geometric mean of the two depths, and the cross range is the size-weighted combination of the two ranges prescribed by the model. The cross-well outer range that enters the interaction kernel is the product of the cross range and the cross diameter.

Returns
-------
numpy float64 array of shape (2,): [rng12, eps12]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lorentz_berthelot_cross(sig1: float, lam1: float, beps1: float, sig2: float, lam2: float, beps2: float) -> np.ndarray:
    r"""sig1, lam1, beps1: solvent (component 1) diameter, square-well range, depth.
    sig2, lam2, beps2: solute (component 2) diameter, square-well range, depth.

    Returns a numpy float64 array of shape $(2,)$: $[\,r_{12},\ \beta\varepsilon_{12}\,]$, the
    outer range $r_{12}$ (a length) and depth $\beta\varepsilon_{12}$ of the cross square-well
    interaction between the two components under the mixing rules of the model.

    Raises:
        ValueError: if any argument is not a finite positive scalar.
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

def _oracle_lorentz_berthelot_cross(sig1, lam1, beps1, sig2, lam2, beps2):
    """eps12 = sqrt(eps1 eps2); sigma12 = (sig1+sig2)/2; lam12 = (lam1 sig1 + lam2 sig2)/(sig1+sig2);
    outer range r12 = lam12 * sigma12 = (lam1 sig1 + lam2 sig2)/2."""
    if min(float(sig1), float(lam1), float(beps1), float(sig2), float(lam2), float(beps2)) <= 0.0:
        raise ValueError("all mixing parameters must be positive")
    s1 = _scalar(sig1, "sig1", positive=True); l1 = _scalar(lam1, "lam1", positive=True)
    e1 = _scalar(beps1, "beps1", positive=True); s2 = _scalar(sig2, "sig2", positive=True)
    l2 = _scalar(lam2, "lam2", positive=True); e2 = _scalar(beps2, "beps2", positive=True)
    eps12 = np.sqrt(e1 * e2)
    rng12 = (l1 * s1 + l2 * s2) / 2.0
    return np.array([rng12, eps12], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'sig1=1.0;lam1=1.5;beps1=1.0\nsig2=2.0;lam2=1.25;beps2=1.20\n',
            "call": 'lorentz_berthelot_cross(sig1,lam1,beps1,sig2,lam2,beps2)',
            "gold_call": '_oracle_lorentz_berthelot_cross(sig1,lam1,beps1,sig2,lam2,beps2)',
        },
        {
            "setup": 'sig1=1.0;lam1=1.5;beps1=1.0\nsig2=1.0;lam2=1.5;beps2=0.4235\n',
            "call": 'lorentz_berthelot_cross(sig1,lam1,beps1,sig2,lam2,beps2)',
            "gold_call": '_oracle_lorentz_berthelot_cross(sig1,lam1,beps1,sig2,lam2,beps2)',
        },
        {
            "setup": 'sig1=1.0;lam1=1.5;beps1=1.0\nsig2=1.5;lam2=1.3333333;beps2=0.8559\n',
            "call": 'lorentz_berthelot_cross(sig1,lam1,beps1,sig2,lam2,beps2)',
            "gold_call": '_oracle_lorentz_berthelot_cross(sig1,lam1,beps1,sig2,lam2,beps2)',
        },
        {
            "setup": 'sig1=1.0;lam1=1.5;beps1=1.0\nsig2=3.0;lam2=1.1666667;beps2=2.6047\n',
            "call": 'lorentz_berthelot_cross(sig1,lam1,beps1,sig2,lam2,beps2)',
            "gold_call": '_oracle_lorentz_berthelot_cross(sig1,lam1,beps1,sig2,lam2,beps2)',
        },
    ]
