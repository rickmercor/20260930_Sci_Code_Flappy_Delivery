"""
Calculate metabolite flux-sums from an irreversible flux vector.

Each flux-sum is the stoichiometrically weighted sum of fluxes that produce the metabolite. Only positive stoichiometric coefficients contribute.

Returns
-------
one float64 flux-sum per metabolite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def flux_sums(stoichiometry: "np.ndarray", flux: "np.ndarray") -> "np.ndarray":
    """Calculate metabolite flux-sums.

    Parameters
    ----------
    stoichiometry
        A finite metabolite-by-reaction matrix using products minus reactants.
    flux
        A finite nonnegative reaction vector aligned with the matrix columns.

    Returns
    -------
    np.ndarray
        One float64 flux-sum per metabolite.

    Raises
    ------
    ValueError
        If the matrix and vector are misaligned, empty or non-finite, or if a
        flux is negative.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_flux_sums(stoichiometry: "np.ndarray", flux: "np.ndarray") -> "np.ndarray":
    stoichiometry = np.asarray(stoichiometry, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if (
        stoichiometry.ndim != 2
        or stoichiometry.shape[0] == 0
        or stoichiometry.shape[1] == 0
        or flux.shape != (stoichiometry.shape[1],)
        or not np.isfinite(stoichiometry).all()
        or not np.isfinite(flux).all()
        or np.any(flux < 0.0)
    ):
        raise ValueError("stoichiometry and irreversible flux must be finite and aligned")
    return (np.maximum(stoichiometry, 0.0) @ flux).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
s=np.array([[1,-1,0],[0,1,-1]],float); v=np.array([2.,1.5,.5])
cs,cv=s.copy(),v.copy(); gs,gv=s.copy(),v.copy()""",
            "call": "flux_sums(cs,cv)",
            "gold_call": "_oracle_flux_sums(gs,gv)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
s=np.array([[2,0,-1],[-1,3,0]],float); v=np.array([0.,4.,5.])
cs,cv=s.copy(),v.copy(); gs,gv=s.copy(),v.copy()""",
            "call": "flux_sums(cs,cv)",
            "gold_call": "_oracle_flux_sums(gs,gv)",
            "tol": 0.0,
        },
        {
            "setup": """import numpy as np
s=np.array([[-1.,1.]]); v=np.array([0.,0.])
cs,cv=s.copy(),v.copy(); gs,gv=s.copy(),v.copy()""",
            "call": "flux_sums(cs,cv)",
            "gold_call": "_oracle_flux_sums(gs,gv)",
            "tol": 0.0,
        },
        {
            "setup": """import numpy as np
s=np.eye(2); v=np.array([1.,-1.])
cs,cv=s.copy(),v.copy(); gs,gv=s.copy(),v.copy()
def case_raises(fn,*args):
    try: fn(*args)
    except ValueError: return 1.0
    return 0.0""",
            "call": "case_raises(flux_sums,cs,cv)",
            "gold_call": "case_raises(_oracle_flux_sums,gs,gv)",
            "tol": 0.0,
        },
    ]
