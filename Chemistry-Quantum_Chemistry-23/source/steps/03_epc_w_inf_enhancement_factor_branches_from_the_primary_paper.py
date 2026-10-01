"""
For supplied reduced-gradient values s, return the two enhancement-factor branches used by the primary ePC meta-GGA model for W_inf: the slowly-varying z=0 branch and the iso-orbital z=1 branch.

The primary ePC construction repairs the strong-interaction point-charge model by combining a slowly-varying branch with an iso-orbital branch. This step requires the W_inf branch definitions and fitted constants of the primary paper; they are not supplied in the task text.

Returns
-------
a numpy array of shape (2, len(s)) containing the published ePC W_inf branches F0 and F1. Raise ``ValueError`` for negative or nonfinite s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def epc_w_inf_branches(s: "np.ndarray") -> "np.ndarray":
    """Return the published ePC W_inf branches F0(s) and F1(s).

    ``s`` is a finite nonnegative array. Recover the two branch definitions,
    constants and limiting behavior from the primary ePC paper. Return a
    float array of shape (2,len(s)) whose first row is the z=0 branch F0 and
    second row is the z=1 branch F1. Raise ValueError for negative or
    nonfinite s.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_epc_w_inf_branches(s: np.ndarray) -> np.ndarray:
    s=np.asarray(s,dtype=float)
    if np.any(~np.isfinite(s)) or np.any(s<0.0):
        raise ValueError("s must be finite and nonnegative.")
    A=-1.451
    a_x=-0.75*(3.0/np.pi)**(1.0/3.0)
    kappa=1.0-a_x/A
    mu=0.14
    a1,a2,a3=0.1,0.9342,0.22447
    s2=s**2
    x=mu*s2/kappa
    f0=1.0-kappa+kappa/(1.0+x+x*x)
    f1=a1+a2/(1.0+a3*s2**4)
    return np.array([f0,f1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    return [
        {"setup":"import numpy as np","call":"epc_w_inf_branches(np.array([0.0,0.1,0.5,1.0,3.0]))","gold_call":"_oracle_epc_w_inf_branches(np.array([0.0,0.1,0.5,1.0,3.0]))"},
        {"setup":"import numpy as np","call":"epc_w_inf_branches(np.array([0.25,0.75,1.5,4.0]))","gold_call":"_oracle_epc_w_inf_branches(np.array([0.25,0.75,1.5,4.0]))"},
        {"setup":"import numpy as np","call":"epc_w_inf_branches(np.array([1e-8,10.0,100.0]))","gold_call":"_oracle_epc_w_inf_branches(np.array([1e-8,10.0,100.0]))"},
    ]
