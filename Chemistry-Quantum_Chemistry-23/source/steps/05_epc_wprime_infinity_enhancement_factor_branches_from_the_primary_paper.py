"""
For supplied reduced-gradient values s and relative spin polarization zeta, return the two enhancement-factor branches used by the primary ePC meta-GGA model for W'_inf.

The primary paper restores the point-charge second-order gradient expansion in the slowly-varying branch of W'_inf and fits a separate iso-orbital branch to few-electron data. The branch definitions, fitted constants and spin factor are primary-paper method content and are not supplied in the task text.

Returns
-------
a numpy array of shape (2, len(s)) containing the published ePC W'_inf branches F0' and F1'. Raise ``ValueError`` for invalid s or zeta.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def epc_wprime_branches(s: "np.ndarray", zeta: float = 0.0) -> "np.ndarray":
    """Return the published ePC W'_inf branches F0'(s) and F1'(s,zeta).

    ``s`` must be finite and nonnegative and ``zeta`` must lie in [-1,1].
    Recover the branch definitions, the restored gradient coefficient, fitted
    constants and spin factor from the primary ePC paper. Return shape
    (2,len(s)), first row F0' and second row F1'.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_epc_wprime_branches(s: np.ndarray, zeta: float = 0.0) -> np.ndarray:
    s=np.asarray(s,dtype=float); zeta=float(zeta)
    if np.any(~np.isfinite(s)) or np.any(s<0.0) or not np.isfinite(zeta) or abs(zeta)>1.0:
        raise ValueError("Invalid s or zeta.")
    mu_p=0.491
    b1,b2,b3=0.04865,4.3217,16.581
    s2=s**2
    f0=(1.0+(mu_p+1.0)*s2)/(1.0+s2)
    f1=(b1+(b1+b2*s2)*np.exp(-b3*s2**3))*(1.0-zeta**10)
    return np.array([f0,f1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    return [
        {"setup":"import numpy as np","call":"epc_wprime_branches(np.array([0.0,0.1,0.5,1.0,3.0]),0.0)","gold_call":"_oracle_epc_wprime_branches(np.array([0.0,0.1,0.5,1.0,3.0]),0.0)"},
        {"setup":"import numpy as np","call":"epc_wprime_branches(np.array([0.25,0.75,1.5,4.0]),0.4)","gold_call":"_oracle_epc_wprime_branches(np.array([0.25,0.75,1.5,4.0]),0.4)"},
        {"setup":"import numpy as np","call":"epc_wprime_branches(np.array([1e-8,10.0,100.0]),1.0)","gold_call":"_oracle_epc_wprime_branches(np.array([1e-8,10.0,100.0]),1.0)"},
    ]
