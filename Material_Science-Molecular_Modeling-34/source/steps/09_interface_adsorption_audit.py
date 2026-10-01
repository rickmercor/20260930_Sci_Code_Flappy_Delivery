"""
Assemble the full pipeline and report the peak interfacial adsorption of the dilute solute.

The complete calculation chains the bulk coexistence, the FMT weight functions and mean-field kernel, the solvent interface profile and its surface tension, the cross-interaction mixing rules, and the solute one-body correlation into the final interfacial enhancement of the nano-particle solute. The solute diameter is $\sigma_2 = \text{ratio}$ and its radius $R_2 = \sigma_2/2$.

Returns
-------
Python float: peak interfacial enhancement P
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interface_adsorption_audit(beps1: float, lam1: float, ratio: float, beps2: float, dz: float, L: float) -> float:
    r"""beps1, lam1: solvent square-well depth $\beta\varepsilon_1$ and range $\lambda_1$.
    ratio: float solute/solvent size ratio $\sigma_2/\sigma_1$ (with $\sigma_1 = 1$).
    beps2: float solute square-well depth $\beta\varepsilon_2$.
    dz: float grid spacing; L: float box length.

    Returns a Python float: the peak interfacial enhancement $P = \max_z \rho_2(z)/\rho_{2,0}$ of
    the dilute solute at the solvent liquid-vapor interface. The solute range obeys the
    equal-well-width rule $\lambda_2 = 1 + (\lambda_1 - 1)/\text{ratio}$.

    Numerical accuracy:
        End-to-end peak comparisons use rtol = atol = 1e-2.
        This accommodates the finite-box profile convergence tolerance
        declared in Step 04; intermediate analytic checks remain separate.

    Raises:
        ValueError: whenever any of the eight earlier functions it calls would raise.
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

def _oracle_interface_adsorption_audit(beps1, lam1, ratio, beps2, dz, L):
    """Orchestrator: reaches every earlier step through its _oracle_ twin."""
    be1 = _scalar(beps1, "beps1", positive=True)
    l1 = _scalar(lam1, "lam1", positive=True)
    rt = _scalar(ratio, "ratio", positive=True)
    be2 = _scalar(beps2, "beps2", positive=True)
    dd = _scalar(dz, "dz", positive=True)
    LL = _scalar(L, "L", positive=True)
    cx = _oracle_sw_bulk_coexistence(be1, l1)                       # step 1
    rho_l, rho_v = float(cx[0]), float(cx[1])
    R1 = 0.5
    sig2 = rt                                                        # sigma1 = 1
    R2 = sig2 / 2.0
    lam2 = 1.0 + (l1 - 1.0) / rt                                    # equal-width rule
    W1 = _oracle_fmt_weight_functions(R1, dd)                       # step 2
    W2 = _oracle_fmt_weight_functions(R2, dd)                       # step 2 (solute)
    K1 = _oracle_sw_meanfield_kernel(be1, l1, dd)                   # step 3
    prof = _oracle_solvent_density_profile(rho_l, rho_v, W1, K1, be1, l1, dd, LL)   # step 4
    gamma = _oracle_surface_tension(prof, W1, K1, be1, l1, dd)      # step 5 (audit checkpoint)
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("solvent interface failed to converge")
    cr = _oracle_lorentz_berthelot_cross(1.0, l1, be1, sig2, lam2, be2)   # step 6
    rng12, eps12 = float(cr[0]), float(cr[1])
    Kc = _oracle_sw_meanfield_kernel(eps12, rng12, dd)             # step 3 (cross)
    c12 = _oracle_solute_one_body_correlation(prof, W1, W2, Kc, dd)  # step 7
    return _oracle_solute_profile_peak(c12)                          # step 8

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'beps1=1.0;lam1=1.5;ratio=2.0;beps2=1.20;dz=0.04;L=12.0\n'
            ),
            "call": 'interface_adsorption_audit(beps1, lam1, ratio, beps2, dz, L)',
            "gold_call": '_oracle_interface_adsorption_audit(beps1, lam1, ratio, beps2, dz, L)',
            "tol": 0.01,
        },
        {
            "setup": (
                'beps1=1.0;lam1=1.5;ratio=1.5;beps2=0.8559;dz=0.04;L=12.0\n'
            ),
            "call": 'interface_adsorption_audit(beps1, lam1, ratio, beps2, dz, L)',
            "gold_call": '_oracle_interface_adsorption_audit(beps1, lam1, ratio, beps2, dz, L)',
            "tol": 0.01,
        },
        {
            "setup": (
                'beps1=1.0;lam1=1.5;ratio=1.0;beps2=0.4235;dz=0.04;L=12.0\n'
            ),
            "call": 'interface_adsorption_audit(beps1, lam1, ratio, beps2, dz, L)',
            "gold_call": '_oracle_interface_adsorption_audit(beps1, lam1, ratio, beps2, dz, L)',
            "tol": 0.01,
        },
    ]
