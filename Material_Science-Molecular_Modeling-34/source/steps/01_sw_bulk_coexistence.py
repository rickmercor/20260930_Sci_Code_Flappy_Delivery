"""
Locate the bulk liquid-vapor coexistence of a one-component square-well fluid.

A square-well fluid is modelled as a hard-sphere reference fluid plus a mean-field (random-phase) attraction. The Helmholtz free energy density is the Carnahan-Starling hard-sphere term plus the mean-field square-well term; the pressure and chemical potential follow by the usual thermodynamic derivatives. Two phases coexist when their pressures and chemical potentials are equal. All lengths are in units of the hard-sphere diameter $\sigma = 1$ and energies in units of $kT$.

Returns
-------
numpy float64 array of shape (2,): [rho_l, rho_v]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sw_bulk_coexistence(beps: float, lam: float) -> np.ndarray:
    r"""beps: float reduced square-well depth $\beta\varepsilon$ of the one-component fluid, > 0.
    lam: float square-well range $\lambda$ in units of the hard-sphere diameter, > 1.

    Returns a numpy float64 array of shape $(2,)$: the coexisting liquid and vapor number
    densities $[\rho_l, \rho_v]$ (in units of $\sigma^{-3}$, $\sigma = 1$) of the bulk
    square-well fluid at the given temperature, with $\rho_l > \rho_v$.

    Raises:
        ValueError: if beps is not a finite positive scalar, or lam is not a finite scalar > 1.
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

def _oracle_sw_bulk_coexistence(beps, lam):
    """CS hard-sphere + RPA square-well mean field; solve equal mu and equal P."""
    be = _scalar(beps, "beps", positive=True)
    lm = _scalar(lam, "lam")
    if lm <= 1.0:
        raise ValueError("lam must exceed 1")
    def bmu(r):
        return np.log(r) + _muhs_ex(r) - (4 * np.pi / 3) * lm ** 3 * be * r
    def bP(r):
        e = np.pi * r / 6.0
        return r * (1 + e + e ** 2 - e ** 3) / (1 - e) ** 3 - (2 * np.pi / 3) * r ** 2 * lm ** 3 * be
    def F(y):
        rl, rv = np.exp(y[0]), np.exp(y[1])
        return [bmu(rl) - bmu(rv), bP(rl) - bP(rv)]
    for gl, gv in [(0.60, 0.04), (0.67, 0.02), (0.72, 0.01), (0.55, 0.08),
                   (0.78, 0.004), (0.50, 0.10), (0.83, 0.002)]:
        y = fsolve(F, [np.log(gl), np.log(gv)])
        rl, rv = float(np.exp(y[0])), float(np.exp(y[1]))
        if np.max(np.abs(F(y))) < 1e-9 and rl > 1.01 * rv and rl < 0.95:
            return np.array([rl, rv], dtype=np.float64)
    raise ValueError("no coexistence found for these parameters")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'beps=1.0\nlam=1.5\n',
            "call": 'sw_bulk_coexistence(beps, lam)',
            "gold_call": '_oracle_sw_bulk_coexistence(beps, lam)',
        },
        {
            "setup": 'beps=1.1\nlam=1.5\n',
            "call": 'sw_bulk_coexistence(beps, lam)',
            "gold_call": '_oracle_sw_bulk_coexistence(beps, lam)',
        },
        {
            "setup": 'beps=1.0\nlam=1.75\n',
            "call": 'sw_bulk_coexistence(beps, lam)',
            "gold_call": '_oracle_sw_bulk_coexistence(beps, lam)',
        },
        {
            "setup": 'beps=1.2\nlam=1.5\n',
            "call": 'sw_bulk_coexistence(beps, lam)',
            "gold_call": '_oracle_sw_bulk_coexistence(beps, lam)',
        },
    ]
