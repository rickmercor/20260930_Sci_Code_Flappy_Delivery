"""
Compute the reduced surface tension of the solvent liquid-vapor interface.

The surface tension is the excess grand potential per unit area of the interface: the integral over $z$ of the local grand-potential density (Helmholtz free-energy density of the ideal, hard-sphere-FMT and mean-field square-well contributions, minus $\mu\rho$) relative to the bulk value $-P$. The bulk pressure is fixed by the liquid reservoir.

Returns
-------
Python float: beta*gamma*sigma1^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def surface_tension(profile: np.ndarray, weights: np.ndarray, kernel: np.ndarray, beps: float, lam: float, dz: float) -> float:
    r"""profile: $(2, N)$ array [$z$, $\rho_1(z)$] from the solvent interface solve.
    weights: $(6, M)$ solvent FMT weights; kernel: $(M_2,)$ solvent mean-field kernel.
    beps: float solvent depth $\beta\varepsilon_1$; lam: float range $\lambda_1$; dz: float spacing.

    Returns a Python float: the reduced interfacial surface tension
    $\beta\gamma\sigma_1^2 = \int dz\,[\beta\omega(z) + \beta P]$, where $\omega$ is the grand
    potential density of the profile and $P$ the coexistence pressure.

    Raises:
        ValueError: on malformed profile/weights/kernel or non-finite scalar inputs.
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

def _oracle_surface_tension(profile, weights, kernel, beps, lam, dz):
    """Excess grand potential per area of the solvent profile."""
    if np.asarray(profile, dtype=float).ndim != 2 or float(dz) <= 0.0:
        raise ValueError("profile must be a (2, N) array and dz must be positive")
    P = _profile(profile)
    W = _weights(weights)
    K = _grid(kernel, "kernel")
    be = _scalar(beps, "beps", positive=True)
    lm = _scalar(lam, "lam", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    w0, w1, w2, w3, wv1, wv2 = W
    rho1 = P[1]
    N = rho1.size
    def bPhi(rho):
        n0 = _pad_conv(rho, w0, dd); n1 = _pad_conv(rho, w1, dd)
        n2 = _pad_conv(rho, w2, dd); n3 = _pad_conv(rho, w3, dd)
        nv1 = _pad_conv(rho, wv1, dd); nv2 = _pad_conv(rho, wv2, dd)
        n3 = np.clip(n3, 1e-10, 1 - 1e-9); s = 1 - n3; Lg = np.log(s)
        return (-n0 * Lg + (n1 * n2 - nv1 * nv2) / s
                + (n2 ** 3 - 3 * n2 * nv2 ** 2) * (n3 + s ** 2 * Lg) / (36 * np.pi * n3 ** 2 * s ** 2))
    def bf(rho):
        return rho * (np.log(np.clip(rho, 1e-30, None)) - 1) + bPhi(rho) + 0.5 * rho * _pad_conv(rho, K, dd)
    rho_l = rho1[-1]
    bmu = np.log(rho_l) + _muhs_ex(rho_l) - (4 * np.pi / 3) * lm ** 3 * be * rho_l
    bP = bmu * rho_l - bf(np.full(N, rho_l))[N // 2]
    return float(np.sum(bf(rho1) - bmu * rho1 + bP) * dd)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\nR=0.5;dz=0.04;L=12.0\nW=_oracle_fmt_weight_functions(R,dz)\nK=_oracle_sw_meanfield_kernel(1.0,1.5,dz)\nprof=_oracle_solvent_density_profile(0.6016,0.0370,W,K,1.0,1.5,dz,L)\n',
            "call": 'surface_tension(prof, W, K, 1.0, 1.5, dz)',
            "gold_call": '_oracle_surface_tension(prof, W, K, 1.0, 1.5, dz)',
        },
        {
            "setup": 'import numpy as np\nR=0.5;dz=0.05;L=14.0\nW=_oracle_fmt_weight_functions(R,dz)\nK=_oracle_sw_meanfield_kernel(1.05,1.5,dz)\ncx=_oracle_sw_bulk_coexistence(1.05,1.5)\nprof=_oracle_solvent_density_profile(cx[0],cx[1],W,K,1.05,1.5,dz,L)\n',
            "call": 'surface_tension(prof, W, K, 1.05, 1.5, dz)',
            "gold_call": '_oracle_surface_tension(prof, W, K, 1.05, 1.5, dz)',
        },
        {
            "setup": 'import numpy as np\nR=0.5;dz=0.04;L=12.0\nW=_oracle_fmt_weight_functions(R,dz)\nK=_oracle_sw_meanfield_kernel(1.1,1.5,dz)\ncx=_oracle_sw_bulk_coexistence(1.1,1.5)\nprof=_oracle_solvent_density_profile(cx[0],cx[1],W,K,1.1,1.5,dz,L)\n',
            "call": 'surface_tension(prof, W, K, 1.1, 1.5, dz)',
            "gold_call": '_oracle_surface_tension(prof, W, K, 1.1, 1.5, dz)',
        },
    ]
