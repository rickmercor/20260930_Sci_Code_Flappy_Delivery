"""
Convert the solute one-body correlation into the normalised adsorption profile and its peak.

In the dilute limit the solute density profile follows from its one-body direct correlation, $\rho_2(z) \propto \exp(c_2^{(1)}(z))$. Normalising to the liquid reservoir gives the enhancement factor $\rho_2(z)/\rho_{2,0}$; its maximum measures how strongly the solute is adsorbed at the liquid-vapor interface.

Returns
-------
Python float: peak enhancement max_z rho2(z)/rho2,0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solute_profile_peak(c1_2: np.ndarray) -> float:
    r"""c1_2: $(N,)$ inhomogeneous one-body direct correlation of the solute across the interface,
    with the liquid reservoir at the last grid point ($z \to +\infty$).

    Returns a Python float: the peak interfacial enhancement
    $P = \max_z\, \rho_2(z)/\rho_{2,0}$ of the dilute solute, where
    $\rho_2(z)/\rho_{2,0} = \exp\!\big(c_2^{(1)}(z) - c_2^{(1)}(+\infty)\big)$ is normalised so
    that the solute density approaches its liquid-reservoir value $\rho_{2,0}$ as $z \to +\infty$.

    Raises:
        ValueError: if c1_2 is not a finite 1D array of length >= 3.
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

def _oracle_solute_profile_peak(c1_2):
    """rho2(z)/rho2,0 = exp(c1_2 - c1_2[-1]); return the maximum."""
    if np.asarray(c1_2, dtype=float).ndim != 1 or np.asarray(c1_2).size < 3:
        raise ValueError("c1_2 must be a 1D array of length >= 3")
    c = _grid(c1_2, "c1_2")
    prof = np.exp(c - c[-1])
    return float(prof.max())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np\ndz=0.04;L=12.0\nW1=_oracle_fmt_weight_functions(0.5,dz)\nW2=_oracle_fmt_weight_functions(1.0,dz)\nK=_oracle_sw_meanfield_kernel(1.0,1.5,dz)\nprof=_oracle_solvent_density_profile(0.6016,0.0370,W1,K,1.0,1.5,dz,L)\ncr=_oracle_lorentz_berthelot_cross(1.0,1.5,1.0,2.0,1.25,1.20)\nKc=_oracle_sw_meanfield_kernel(cr[1],cr[0],dz)\nc12=_oracle_solute_one_body_correlation(prof,W1,W2,Kc,dz)\n',
            "call": 'solute_profile_peak(c12)',
            "gold_call": '_oracle_solute_profile_peak(c12)',
        },
        {
            "setup": 'import numpy as np\ndz=0.04;L=12.0\nW1=_oracle_fmt_weight_functions(0.5,dz)\nW2=_oracle_fmt_weight_functions(0.75,dz)\nK=_oracle_sw_meanfield_kernel(1.0,1.5,dz)\nprof=_oracle_solvent_density_profile(0.6016,0.0370,W1,K,1.0,1.5,dz,L)\ncr=_oracle_lorentz_berthelot_cross(1.0,1.5,1.0,1.5,1.3333333,1.20)\nKc=_oracle_sw_meanfield_kernel(cr[1],cr[0],dz)\nc12=_oracle_solute_one_body_correlation(prof,W1,W2,Kc,dz)\n',
            "call": 'solute_profile_peak(c12)',
            "gold_call": '_oracle_solute_profile_peak(c12)',
        },
        {
            "setup": 'import numpy as np\nc12=np.zeros(401)\n',
            "call": 'solute_profile_peak(c12)',
            "gold_call": '_oracle_solute_profile_peak(c12)',
        },
    ]
