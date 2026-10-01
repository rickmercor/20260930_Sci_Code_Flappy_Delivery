"""
Construct condition-specific enzyme abundances for one design.

Normalize the design-modified high-growth allocation once, calculate the kinetic factors and double-graded responses, and multiply the normalized allocation by those responses.

Returns
-------
a float64 condition-by-reaction enzyme-abundance matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enzyme_profiles(
    base_phi: "np.ndarray",
    design_multiplier: "np.ndarray",
    affinity: "np.ndarray",
    turnover: "np.ndarray",
    pathway_weight: "np.ndarray",
    reaction_sector: "np.ndarray",
    metabolite_sector: "np.ndarray",
    restriction: "np.ndarray",
    growth_ratios: "np.ndarray",
    metabolome_fraction: float,
) -> "np.ndarray":
    """Construct condition-specific enzyme abundances.

    Parameters
    ----------
    base_phi, design_multiplier, affinity, turnover, pathway_weight
        Aligned finite positive reaction vectors.
    reaction_sector, metabolite_sector, restriction, growth_ratios,
    metabolome_fraction
        Inputs following the growth-response contract.

    Returns
    -------
    np.ndarray
        A float64 condition-by-reaction enzyme-abundance matrix.

    Raises
    ------
    ValueError
        If the reaction vectors are misaligned or nonpositive, or if any
        downstream kinetic-factor or growth-response requirement is violated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_enzyme_profiles(
    base_phi: "np.ndarray",
    design_multiplier: "np.ndarray",
    affinity: "np.ndarray",
    turnover: "np.ndarray",
    pathway_weight: "np.ndarray",
    reaction_sector: "np.ndarray",
    metabolite_sector: "np.ndarray",
    restriction: "np.ndarray",
    growth_ratios: "np.ndarray",
    metabolome_fraction: float,
) -> "np.ndarray":
    base_phi = np.asarray(base_phi, dtype=np.float64)
    design_multiplier = np.asarray(design_multiplier, dtype=np.float64)
    if (
        base_phi.ndim != 1
        or base_phi.size == 0
        or design_multiplier.shape != base_phi.shape
        or not np.isfinite(base_phi).all()
        or not np.isfinite(design_multiplier).all()
        or np.any(base_phi <= 0.0)
        or np.any(design_multiplier <= 0.0)
    ):
        raise ValueError("base allocation and design multiplier must align and be positive")
    modified = base_phi * design_multiplier
    phi_high = modified / modified.sum()
    kinetic = _oracle_kinetic_factors(affinity, turnover, pathway_weight)
    packed = _oracle_growth_response(
        phi_high,
        kinetic,
        reaction_sector,
        metabolite_sector,
        restriction,
        growth_ratios,
        metabolome_fraction,
    )
    n_conditions = np.asarray(growth_ratios).size
    n_reactions = base_phi.size
    q = packed[1 : 1 + n_conditions * n_reactions].reshape(n_conditions, n_reactions)
    return (q * phi_high[None, :]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
args=(np.array([.2,.3,.25,.25]),np.array([1.1,.9,1.2,.8]),np.array([.1,.2,.15,.08]),np.array([20.,25.,18.,30.]),np.array([1.,1.,.8,.8]),np.array([0,0,1,1]),np.array([0,1]),np.array([1.,.05]),np.array([1.,.6,.3]),.12)
cargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
gargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)""",
            "call": "enzyme_profiles(*cargs)",
            "gold_call": "_oracle_enzyme_profiles(*gargs)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
args=(np.ones(3),np.ones(3),np.array([.1,.1,.1]),np.array([10.,10.,10.]),np.ones(3),np.zeros(3,dtype=int),np.array([0]),np.array([1.]),np.array([1.,0.]),.1)
cargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
gargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)""",
            "call": "enzyme_profiles(*cargs)",
            "gold_call": "_oracle_enzyme_profiles(*gargs)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
args=(np.array([.7,.3]),np.array([1.,2.]),np.array([.05,.2]),np.array([40.,10.]),np.ones(2),np.array([0,1]),np.array([1]),np.array([.9,.1]),np.array([.5]),.2)
cargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
gargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)""",
            "call": "enzyme_profiles(*cargs)",
            "gold_call": "_oracle_enzyme_profiles(*gargs)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
args=(np.array([.5,.5]),np.array([1.,0.]),np.array([.1,.1]),np.array([10.,10.]),np.ones(2),np.array([0,0]),np.array([0]),np.array([1.]),np.array([.5]),.1)
cargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
gargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
def case_raises(fn,*args):
    try: fn(*args)
    except ValueError: return 1.0
    return 0.0""",
            "call": "case_raises(enzyme_profiles,*cargs)",
            "gold_call": "case_raises(_oracle_enzyme_profiles,*gargs)",
            "tol": 0.0,
        },
    ]
