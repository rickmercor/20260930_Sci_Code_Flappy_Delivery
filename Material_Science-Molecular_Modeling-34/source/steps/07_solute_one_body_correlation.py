"""
Compute the inhomogeneous one-body direct correlation of a dilute solute in the solvent field.

Inserting one solute particle into the inhomogeneous solvent, the solute one-body direct correlation is minus the functional derivative of the excess free energy with respect to the solute density, taken in the dilute limit. The hard-sphere part convolves the solvent's fundamental-measure derivative fields with the solute's own weight functions (which carry the solute size), and the square-well part convolves the solvent density with the cross mean-field kernel. Their sum is the position-dependent insertion correlation that governs the solute's interfacial adsorption.

Returns
-------
numpy float64 array of shape (N,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solute_one_body_correlation(profile: np.ndarray, weights1: np.ndarray, weights2: np.ndarray, cross_kernel: np.ndarray, dz: float) -> np.ndarray:
    r"""profile: $(2, N)$ array [$z$, $\rho_1(z)$] of the solvent interface.
    weights1: $(6, M_1)$ FMT weights of the solvent hard sphere ($R_1$).
    weights2: $(6, M_2)$ FMT weights of the solute hard sphere ($R_2$).
    cross_kernel: $(M_c,)$ mean-field kernel of the cross square-well interaction.
    dz: float grid spacing.

    Returns a numpy float64 array of shape $(N,)$: the inhomogeneous one-body direct correlation
    $c_2^{(1)}(z)$ of a single dilute solute particle in the solvent field $\rho_1(z)$, i.e. the
    sum of its hard-sphere (FMT) and cross square-well contributions.

    Raises:
        ValueError: on malformed profile/weights/kernel or non-finite dz.
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

def _oracle_solute_one_body_correlation(profile, weights1, weights2, cross_kernel, dz):
    """c1_2^hs (solvent FMT derivative fields convolved with SOLUTE weights) + c1_2^sw."""
    if np.asarray(profile, dtype=float).ndim != 2 or float(dz) <= 0.0:
        raise ValueError("profile must be a (2, N) array and dz must be positive")
    P = _profile(profile)
    W1 = _weights(weights1); W2 = _weights(weights2)
    Kc = _grid(cross_kernel, "cross_kernel")
    dd = _scalar(dz, "dz", positive=True)
    rho1 = P[1]
    a0, a1, a2, a3, av1, av2 = W1
    b0, b1, b2, b3, bv1, bv2 = W2
    n0 = _pad_conv(rho1, a0, dd); n1 = _pad_conv(rho1, a1, dd)
    n2 = _pad_conv(rho1, a2, dd); n3 = _pad_conv(rho1, a3, dd)
    nv1 = _pad_conv(rho1, av1, dd); nv2 = _pad_conv(rho1, av2, dd)
    D0, D1, D2, D3, Dv1, Dv2 = _wb_derivs(n0, n1, n2, n3, nv1, nv2)
    c1hs = -(_pad_conv(D0, b0, dd) + _pad_conv(D1, b1, dd) + _pad_conv(D2, b2, dd)
             + _pad_conv(D3, b3, dd) + _pad_conv(Dv1, -bv1, dd) + _pad_conv(Dv2, -bv2, dd))
    c1sw = -_pad_conv(rho1, Kc, dd)
    return (c1hs + c1sw).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndz=0.04;L=12.0\nW1=_oracle_fmt_weight_functions(0.5,dz)\nW2=_oracle_fmt_weight_functions(1.0,dz)\nK=_oracle_sw_meanfield_kernel(1.0,1.5,dz)\nprof=_oracle_solvent_density_profile(0.6016,0.0370,W1,K,1.0,1.5,dz,L)\ncr=_oracle_lorentz_berthelot_cross(1.0,1.5,1.0,2.0,1.25,1.20)\nKc=_oracle_sw_meanfield_kernel(cr[1],cr[0],dz)\n',
            "call": 'solute_one_body_correlation(prof, W1, W2, Kc, dz)',
            "gold_call": '_oracle_solute_one_body_correlation(prof, W1, W2, Kc, dz)',
        },
        {
            "setup": 'import numpy as np\ndz=0.04;L=12.0\nW1=_oracle_fmt_weight_functions(0.5,dz)\nW2=_oracle_fmt_weight_functions(0.75,dz)\nK=_oracle_sw_meanfield_kernel(1.0,1.5,dz)\nprof=_oracle_solvent_density_profile(0.6016,0.0370,W1,K,1.0,1.5,dz,L)\ncr=_oracle_lorentz_berthelot_cross(1.0,1.5,1.0,1.5,1.3333333,0.8559)\nKc=_oracle_sw_meanfield_kernel(cr[1],cr[0],dz)\n',
            "call": 'solute_one_body_correlation(prof, W1, W2, Kc, dz)',
            "gold_call": '_oracle_solute_one_body_correlation(prof, W1, W2, Kc, dz)',
        },
        {
            "setup": 'import numpy as np\ndz=0.04;L=12.0\nW1=_oracle_fmt_weight_functions(0.5,dz)\nW2=_oracle_fmt_weight_functions(1.5,dz)\nK=_oracle_sw_meanfield_kernel(1.0,1.5,dz)\nprof=_oracle_solvent_density_profile(0.6016,0.0370,W1,K,1.0,1.5,dz,L)\ncr=_oracle_lorentz_berthelot_cross(1.0,1.5,1.0,3.0,1.1666667,2.6047)\nKc=_oracle_sw_meanfield_kernel(cr[1],cr[0],dz)\n',
            "call": 'solute_one_body_correlation(prof, W1, W2, Kc, dz)',
            "gold_call": '_oracle_solute_one_body_correlation(prof, W1, W2, Kc, dz)',
        },
    ]
