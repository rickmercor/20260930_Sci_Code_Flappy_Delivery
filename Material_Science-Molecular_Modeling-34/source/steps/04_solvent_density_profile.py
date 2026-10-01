"""
Solve the planar solvent density profile at liquid-vapor coexistence by Picard iteration.

The equilibrium density profile minimises the grand potential and satisfies the Euler-Lagrange equation $\rho(z) = \exp(\beta\mu + c^{(1)}(z))$, where the one-body direct correlation $c^{(1)}$ is minus the functional derivative of the excess free energy (hard-sphere FMT plus mean-field square well) evaluated on the current profile. This fixed-point equation is solved by damped iteration from a smooth vapor-to-liquid guess; every weighted density and mean-field convolution must use the correct bulk behaviour far from the interface, and the two bulk phases are held at their coexistence densities.

Returns
-------
numpy float64 array of shape (2, N): [z, rho1]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solvent_density_profile(
    rho_l: float,
    rho_v: float,
    weights: np.ndarray,
    kernel: np.ndarray,
    beps: float,
    lam: float,
    dz: float,
    L: float,
) -> np.ndarray:
    r"""Solve the finite-box solvent interface by damped Picard iteration.

    rho_l, rho_v: finite positive coexistence densities, rho_l > rho_v.
    weights: solvent FMT weights, shape (6, M), for radius R = 0.5.
    kernel: one-dimensional solvent mean-field kernel.
    beps, lam: finite positive solvent reduced depth and range.
    dz, L: finite positive spacing and box length.

    Numerical convention:
        Use z = np.arange(-L / 2, L / 2 + dz / 2, dz).
        Initialize rho = rho_v + (rho_l - rho_v) *
                         (1 + np.tanh(z)) / 2.
        Keep the array alignment supplied by Steps 02 and 03.

        For a kernel w, set h = len(w) // 2, extend the input by
        h endpoint-value cells on each side, take the same-mode
        discrete convolution, multiply by dz, and retain indices
        h through h + len(input) - 1.

        Use the liquid-reservoir bulk chemical potential. For each
        unmixed Euler-Lagrange update, pin int(lam / dz) + 10 cells
        on the left to rho_v and on the right to rho_l. Start with the
        initialization above; do not translate the returned profile.
        The reference mixes 15 percent of the updated density with
        85 percent of the previous density and stops after an unmixed
        maximum absolute density residual below 1e-6, or 25000 updates.
        More tightly converged profiles are acceptable within the
        declared density comparison tolerance.

    Returns:
        A fresh numpy float64 array of shape (2, N), where N = z.size.
        Row 0 is z and row 1 is rho. Vapor is on the left, liquid on
        the right. Density tests use rtol = atol = 5e-4, while grid
        coordinates must agree within an absolute tolerance of 1e-10.

    Raises:
        ValueError: on malformed weights or kernel, non-finite or
            non-positive scalar inputs, or rho_l <= rho_v.
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

def _oracle_solvent_density_profile(rho_l, rho_v, weights, kernel, beps, lam, dz, L):
    """Damped Picard on ln rho with edge (bulk) padding and pinned bulk tails."""
    W = _weights(weights)
    K = _grid(kernel, "kernel")
    rl = _scalar(rho_l, "rho_l", positive=True)
    rv = _scalar(rho_v, "rho_v", positive=True)
    be = _scalar(beps, "beps", positive=True)
    lm = _scalar(lam, "lam", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    LL = _scalar(L, "L", positive=True)
    if rl <= rv:
        raise ValueError("need rho_l > rho_v")
    w0, w1, w2, w3, wv1, wv2 = W
    z = np.arange(-LL / 2, LL / 2 + dd / 2, dd)
    def c1_fmt(rho):
        n0 = _pad_conv(rho, w0, dd); n1 = _pad_conv(rho, w1, dd)
        n2 = _pad_conv(rho, w2, dd); n3 = _pad_conv(rho, w3, dd)
        nv1 = _pad_conv(rho, wv1, dd); nv2 = _pad_conv(rho, wv2, dd)
        d0, d1, d2, d3, dv1, dv2 = _wb_derivs(n0, n1, n2, n3, nv1, nv2)
        return -(_pad_conv(d0, w0, dd) + _pad_conv(d1, w1, dd) + _pad_conv(d2, w2, dd)
                 + _pad_conv(d3, w3, dd) + _pad_conv(dv1, -wv1, dd) + _pad_conv(dv2, -wv2, dd))
    bmu = np.log(rl) + _muhs_ex(rl) - (4 * np.pi / 3) * lm ** 3 * be * rl
    rho = rv + (rl - rv) * 0.5 * (1 + np.tanh(z / 1.0))
    npin = int(lm / dd) + 10
    for _ in range(25000):
        rt = np.exp(np.clip(bmu + c1_fmt(rho) - _pad_conv(rho, K, dd), -40, 3))
        rt[:npin] = rv
        rt[-npin:] = rl
        d = np.max(np.abs(rt - rho))
        rho = 0.15 * rt + 0.85 * rho
        if d < 1e-6:
            break
    return np.vstack([z, rho]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'import numpy as np\n'
                'rho_l,rho_v=0.6016,0.0370\n'
                'R=0.5;dz=0.04;L=12.0\n'
                'weights=_oracle_fmt_weight_functions(R,dz)\n'
                'kernel=_oracle_sw_meanfield_kernel(1.0,1.5,dz)\n'
            ),
            "call": (
                '(lambda p: (bool(p.shape == (2, np.arange(-L/2, L/2+dz/2, dz).size)), bool(n'
                'p.allclose(p[0], np.arange(-L/2, L/2+dz/2, dz), rtol=0.0, atol=1e-10)), p[1]'
                '))(solvent_density_profile(rho_l, rho_v, weights, kernel, 1.0, 1.5, dz, L))'
            ),
            "gold_call": (
                '(lambda p: (True, True, p[1]))(_oracle_solvent_density_profile(rho_l, rho_v,'
                ' weights, kernel, 1.0, 1.5, dz, L))'
            ),
            "tol": 0.0005,
        },
        {
            "setup": (
                'import numpy as np\n'
                'cx=_oracle_sw_bulk_coexistence(1.05,1.5)\n'
                'R=0.5;dz=0.05;L=14.0\n'
                'weights=_oracle_fmt_weight_functions(R,dz)\n'
                'kernel=_oracle_sw_meanfield_kernel(1.05,1.5,dz)\n'
            ),
            "call": (
                '(lambda p: (bool(p.shape == (2, np.arange(-L/2, L/2+dz/2, dz).size)), bool(n'
                'p.allclose(p[0], np.arange(-L/2, L/2+dz/2, dz), rtol=0.0, atol=1e-10)), p[1]'
                '))(solvent_density_profile(cx[0], cx[1], weights, kernel, 1.05, 1.5, dz, L))'
            ),
            "gold_call": (
                '(lambda p: (True, True, p[1]))(_oracle_solvent_density_profile(cx[0], cx[1],'
                ' weights, kernel, 1.05, 1.5, dz, L))'
            ),
            "tol": 0.0005,
        },
        {
            "setup": (
                'import numpy as np\n'
                'cx=_oracle_sw_bulk_coexistence(1.1,1.5)\n'
                'R=0.5;dz=0.04;L=12.0\n'
                'weights=_oracle_fmt_weight_functions(R,dz)\n'
                'kernel=_oracle_sw_meanfield_kernel(1.1,1.5,dz)\n'
            ),
            "call": (
                '(lambda p: (bool(p.shape == (2, np.arange(-L/2, L/2+dz/2, dz).size)), bool(n'
                'p.allclose(p[0], np.arange(-L/2, L/2+dz/2, dz), rtol=0.0, atol=1e-10)), p[1]'
                '))(solvent_density_profile(cx[0], cx[1], weights, kernel, 1.1, 1.5, dz, L))'
            ),
            "gold_call": (
                '(lambda p: (True, True, p[1]))(_oracle_solvent_density_profile(cx[0], cx[1],'
                ' weights, kernel, 1.1, 1.5, dz, L))'
            ),
            "tol": 0.0005,
        },
    ]
