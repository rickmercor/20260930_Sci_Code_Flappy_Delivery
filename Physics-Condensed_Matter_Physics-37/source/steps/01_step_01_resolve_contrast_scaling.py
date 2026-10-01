"""
Report the resonator parameters and both contrasts explicitly. Which of delta and eps appears where is the single most consequential piece of bookkeeping in this problem, and working out which of them sets the resonant frequencies is part of the task rather than something this stage settles.

Returns
-------
dict, holding lam_in, mu_in and rho_in for the resonator, tau for the wave speed contrast, and c_s and c_p for the background.

A phononic crystal of this kind is built from two elastic solids whose Lame parameters and whose densities are both far apart, and the analysis that follows is organised entirely around how far apart each pair is. Write the background, the soft matrix filling the cell outside the resonator, with Lame parameters (lam, mu) and density rho. The resonator carries Lame parameters (lam, mu) divided by delta and density rho divided by eps, so that delta and eps are the reciprocals of the stiffness contrast and of the density contrast. Both are small and positive.

Two derived quantities fix the regime. The wave speed contrast

$$tau = c_{s} / c_{s,in} = c_{p} / c_{p,in} = (delta / eps)^{1/2}$$

compares shear and compressional speeds inside and outside the resonator, the subscript in marking the resonator; the single ratio serves for both because the same delta scales both Lame parameters. The analysis assumes tau stays of order one while delta and eps go to zero, which is a statement that the resonator is heavy and stiff in the same proportion, not that it is simply rigid. The background speeds themselves are

$$c_{s} = (mu / rho)^{1/2}, c_{p} = [(lam + 2 mu) / rho]^{1/2}.$$

The background must satisfy the strong convexity conditions that make the static operator coercive, namely mu above zero and three lam plus two mu above zero in three dimensions. Reject parameters that fail them, because everything downstream rests on a coercive form.

Returns
-------
dict, holding lam_in, mu_in and rho_in for the resonator, tau for the wave speed contrast, and c_s and c_p for the background.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def resolve_contrast_scaling(
    lam: float,
    mu: float,
    rho: float,
    delta: float,
    eps: float,
) -> dict:
    """Turn the background parameters and the two contrasts into the resonator parameters and the regime numbers.

    Parameters
    ----------
    lam : float
        First Lame parameter of the soft background in pascal.
    mu : float
        Shear modulus of the soft background in pascal, above zero.
    rho : float
        Density of the soft background in kilogram per cubic metre, above zero.
    delta : float
        Reciprocal stiffness contrast, above zero and below one.
    eps : float
        Reciprocal density contrast, above zero and below one.

    Returns
    -------
    dict
        Under the keys lam_in, mu_in, rho_in, tau, c_s and c_p.

    Raises
    ------
    ValueError
        When any argument fails to be finite, when mu or rho fails to be above zero, when three lam plus two mu fails to be above zero, or when delta or eps falls outside the open interval from zero to one.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


import math


def _finite(value, label):
    """Return an argument as a float once it is known to be finite."""
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("%s must be finite" % label)
    return out


def _contrast(value, label):
    """Return a contrast as a float once it is known to lie strictly between zero and one."""
    out = _finite(value, label)
    if not 0.0 < out < 1.0:
        raise ValueError("%s must lie strictly between zero and one" % label)
    return out


def _oracle_resolve_contrast_scaling(
    lam: float,
    mu: float,
    rho: float,
    delta: float,
    eps: float,
) -> dict:
    """Reference implementation."""
    lam = _finite(lam, "lam")
    mu = _finite(mu, "mu")
    rho = _finite(rho, "rho")
    delta = _contrast(delta, "delta")
    eps = _contrast(eps, "eps")
    if mu <= 0.0:
        raise ValueError("mu must be above zero")
    if rho <= 0.0:
        raise ValueError("rho must be above zero")
    if 3.0 * lam + 2.0 * mu <= 0.0:
        raise ValueError("three lam plus two mu must be above zero")
    return {
        "lam_in": lam / delta,
        "mu_in": mu / delta,
        "rho_in": rho / eps,
        "tau": math.sqrt(delta / eps),
        "c_s": math.sqrt(mu / rho),
        "c_p": math.sqrt((lam + 2.0 * mu) / rho),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
def digest(out):
    return (round(out["lam_in"], 6), round(out["mu_in"], 6), round(out["rho_in"], 6),
            round(out["tau"], 12), round(out["c_s"], 10), round(out["c_p"], 10))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat(digest(resolve_contrast_scaling(1.5e6, 5.0e5, 1200.0, 1.0e-2, 5.0e-3)))",
            "gold_call": "flat(digest(_oracle_resolve_contrast_scaling(1.5e6, 5.0e5, 1200.0, 1.0e-2, 5.0e-3)))",
        },
        {
            "setup": """
def digest(out):
    return (round(out["tau"], 12), round(out["rho_in"], 6), round(out["mu_in"], 6))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat(digest(resolve_contrast_scaling(2.0e6, 8.0e5, 1000.0, 4.0e-3, 4.0e-3)))",
            "gold_call": "flat(digest(_oracle_resolve_contrast_scaling(2.0e6, 8.0e5, 1000.0, 4.0e-3, 4.0e-3)))",
        },
        {
            "setup": """
def verdict(fn, lam=1.5e6, mu=5.0e5, rho=1200.0, delta=1.0e-2, eps=5.0e-3):
    try:
        fn(lam, mu, rho, delta, eps)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
def verdicts(fn):
    return flat((verdict(fn, mu=-1.0), verdict(fn, rho=0.0), verdict(fn, lam=-1.0e6), verdict(fn, delta=1.0), verdict(fn, eps=0.0), verdict(fn, mu=float('nan')), verdict(fn)))
""",
            'call': 'verdicts(resolve_contrast_scaling)',
            'gold_call': 'verdicts(_oracle_resolve_contrast_scaling)',
        },
    ]
