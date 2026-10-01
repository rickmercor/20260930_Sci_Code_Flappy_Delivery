"""
Calculate double-graded enzyme and metabolite responses.

The enzyme response combines the local kinetic factor with the restriction factor of its sector. The metabolite response uses the source's nonlinear restriction law. The network average is mass weighted.

Returns
-------
c-bar, then row-major enzyme responses, then row-major metabolite responses in one float64 vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def growth_response(
    phi_high: "np.ndarray",
    kinetic_factor: "np.ndarray",
    reaction_sector: "np.ndarray",
    metabolite_sector: "np.ndarray",
    restriction: "np.ndarray",
    growth_ratios: "np.ndarray",
    metabolome_fraction: float,
) -> "np.ndarray":
    """Calculate double-graded growth responses.

    Parameters
    ----------
    phi_high, kinetic_factor
        Aligned finite positive reaction vectors. ``phi_high`` must sum to one.
    reaction_sector, metabolite_sector
        Zero-based sector labels for reactions and metabolites.
    restriction
        Finite nonnegative restriction factor for each sector.
    growth_ratios
        Finite one-dimensional values in the closed interval [0, 1].
    metabolome_fraction
        Positive finite high-growth metabolome-to-proteome mass fraction.

    Returns
    -------
    np.ndarray
        c-bar, row-major enzyme responses, and row-major metabolite responses.

    Raises
    ------
    ValueError
        If shapes, labels, normalization, ranges, or finite-value requirements
        are violated, or if the mass-weighted kinetic average is not positive.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_growth_response(
    phi_high: "np.ndarray",
    kinetic_factor: "np.ndarray",
    reaction_sector: "np.ndarray",
    metabolite_sector: "np.ndarray",
    restriction: "np.ndarray",
    growth_ratios: "np.ndarray",
    metabolome_fraction: float,
) -> "np.ndarray":
    phi_high = np.asarray(phi_high, dtype=np.float64)
    kinetic_factor = np.asarray(kinetic_factor, dtype=np.float64)
    reaction_sector = np.asarray(reaction_sector)
    metabolite_sector = np.asarray(metabolite_sector)
    restriction = np.asarray(restriction, dtype=np.float64)
    growth_ratios = np.asarray(growth_ratios, dtype=np.float64)
    if (
        phi_high.ndim != 1
        or phi_high.size == 0
        or kinetic_factor.shape != phi_high.shape
        or reaction_sector.shape != phi_high.shape
        or metabolite_sector.ndim != 1
        or metabolite_sector.size == 0
        or restriction.ndim != 1
        or restriction.size == 0
        or growth_ratios.ndim != 1
        or growth_ratios.size == 0
        or not np.isfinite(phi_high).all()
        or not np.isfinite(kinetic_factor).all()
        or not np.isfinite(restriction).all()
        or not np.isfinite(growth_ratios).all()
        or np.any(phi_high <= 0.0)
        or np.any(kinetic_factor <= 0.0)
        or np.any(restriction < 0.0)
        or np.any(growth_ratios < 0.0)
        or np.any(growth_ratios > 1.0)
        or not np.isfinite(metabolome_fraction)
        or metabolome_fraction <= 0.0
        or not np.isclose(phi_high.sum(), 1.0, rtol=0.0, atol=1e-12)
        or not np.issubdtype(reaction_sector.dtype, np.integer)
        or not np.issubdtype(metabolite_sector.dtype, np.integer)
        or np.any(reaction_sector < 0)
        or np.any(metabolite_sector < 0)
        or np.any(reaction_sector >= restriction.size)
        or np.any(metabolite_sector >= restriction.size)
    ):
        raise ValueError("growth-response inputs violate the stated contract")
    xi_r = restriction[reaction_sector]
    cbar = float(np.sum(phi_high * kinetic_factor * xi_r))
    if not np.isfinite(cbar) or cbar <= 0.0:
        raise ValueError("the mass-weighted kinetic average must be positive")
    r = growth_ratios[:, None]
    q = r + (kinetic_factor[None, :] / cbar) * xi_r[None, :] * (1.0 - r)
    xi_m = restriction[metabolite_sector][None, :]
    numerator = metabolome_fraction * r
    denominator = numerator + xi_m * (1.0 - r)
    qrho = np.divide(
        numerator,
        denominator,
        out=np.ones((growth_ratios.size, metabolite_sector.size), dtype=np.float64),
        where=denominator > 0.0,
    )
    return np.concatenate(([cbar], q.ravel(), qrho.ravel())).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
args=(np.array([0.4,0.35,0.25]),np.array([1.2,0.8,1.5]),np.array([0,1,0]),np.array([0,1]),np.array([1.0,0.1]),np.array([1.0,0.6,0.2]),0.12)
cargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
gargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)""",
            "call": "growth_response(*cargs)",
            "gold_call": "_oracle_growth_response(*gargs)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
args=(np.array([0.5,0.5]),np.array([2.,2.]),np.array([0,0]),np.array([0]),np.array([1.]),np.array([1.,0.]),0.1)
cargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
gargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)""",
            "call": "growth_response(*cargs)",
            "gold_call": "_oracle_growth_response(*gargs)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
args=(np.array([0.25,0.25,0.25,0.25]),np.ones(4),np.array([0,1,1,0]),np.array([1,0,1]),np.array([0.8,0.0]),np.array([0.75]),0.2)
cargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
gargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)""",
            "call": "growth_response(*cargs)",
            "gold_call": "_oracle_growth_response(*gargs)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
args=(np.array([0.6,0.5]),np.ones(2),np.array([0,0]),np.array([0]),np.array([1.]),np.array([0.5]),0.1)
cargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
gargs=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in args)
def case_raises(fn,*args):
    try: fn(*args)
    except ValueError: return 1.0
    return 0.0""",
            "call": "case_raises(growth_response,*cargs)",
            "gold_call": "case_raises(_oracle_growth_response,*gargs)",
            "tol": 0.0,
        },
    ]
