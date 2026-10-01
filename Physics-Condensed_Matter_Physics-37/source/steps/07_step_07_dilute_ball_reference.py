"""
Take r as the radius of the ball of the same volume as the resonator actually used, so that the comparison holds the amount of heavy material fixed and varies only the shape. Treat the interval as a reference comparison and not as a bound. The bracketing statement belongs to a dilute arrangement of balls, and neither changing the shape at fixed volume nor leaving the dilute regime preserves it without a further argument that is not supplied here, so a computed frequency outside the interval would not by itself prove an error and one inside proves nothing.

Returns
-------
dict, holding radius, beta_ball, omega_min, omega_max, hertz_min, hertz_max and width_ratio.

The computed edge is worth nothing without something independent to hold it against, and the analysis supplies exactly one closed form. In the dilute regime, where resonators are far apart compared with their own size so that they stop interacting, the capacity matrix of a single resonator replaces the quasi-periodic one, and for a ball it can be evaluated in closed form. The elastic single-layer potential on a sphere of radius r acts on a constant field as multiplication by minus the combination (5 mu + 2 lam) divided by three mu times the quantity two mu plus lam, times r, and inverting it gives a capacity matrix that is a multiple of the identity,

$$beta_ball = 12 * mu * pi * r * (2 * mu + lam) / (5 * mu + 2 * lam),$$

the same value three times over, because a ball has no preferred direction. Feeding that through the frequency relation with the volume of the ball gives the lower endpoint of the interval that the dilute analysis brackets the edge within,

$$omega_min = sqrt(9 * mu * (2 * mu + lam) / ((5 * mu + 2 * lam) * rho * r^2)) * eps^(1/2).$$

The upper endpoint of that interval is a different combination altogether, belonging to the largest rather than the smallest resonance of one hard obstacle in an unbounded soft medium, and it is not obtainable from beta_ball:

$$omega_max = sqrt(15 * mu / (rho * r^2)) * eps^(1/2).$$

The two endpoints carry different dependence on lam, which is the point of reporting both: the lower one is sensitive to the compressional stiffness of the background and the upper one is not, so the interval is not symmetric about anything and its width is a statement about how much the shape of the resonator can matter.

Returns
-------
dict, holding radius, beta_ball, omega_min, omega_max, hertz_min, hertz_max and width_ratio.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dilute_ball_reference(
    lam: float,
    mu: float,
    rho: float,
    eps: float,
    volume_D: float,
) -> dict:
    """Evaluate the dilute-limit interval for a ball of the same volume as the resonator.

    Parameters
    ----------
    lam : float
        First Lame parameter of the background in pascal.
    mu : float
        Shear modulus of the background in pascal, above zero.
    rho : float
        Background density in kilogram per cubic metre, above zero.
    eps : float
        Reciprocal density contrast, above zero and below one.
    volume_D : float
        Resonator volume in cubic metre, above zero.

    Returns
    -------
    dict
        Under the keys radius, beta_ball, omega_min, omega_max, hertz_min, hertz_max and width_ratio.
        width_ratio is the dimensionless endpoint ratio omega_max / omega_min,
        equivalently hertz_max / hertz_min.

    Raises
    ------
    ValueError
        When any argument fails to be finite, when mu, rho or volume_D fails to be above zero, when eps falls outside the open interval from zero to one, or when five mu plus two lam or two mu plus lam fails to be above zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


import math


def _positive(value, label):
    """Return an argument as a float once it is known to be finite and above zero."""
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError("%s must be finite and above zero" % label)
    return out


def _oracle_dilute_ball_reference(
    lam: float,
    mu: float,
    rho: float,
    eps: float,
    volume_D: float,
) -> dict:
    """Reference implementation."""
    lam = float(lam)
    if not math.isfinite(lam):
        raise ValueError("lam must be finite")
    mu = _positive(mu, "mu")
    rho = _positive(rho, "rho")
    vol = _positive(volume_D, "volume_D")
    eps = float(eps)
    if not math.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("eps must lie strictly between zero and one")
    denom = 5.0 * mu + 2.0 * lam
    if denom <= 0.0:
        raise ValueError("five mu plus two lam must be above zero")
    if 2.0 * mu + lam <= 0.0:
        raise ValueError("two mu plus lam must be above zero")

    radius = (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0)
    beta = 12.0 * mu * math.pi * radius * (2.0 * mu + lam) / denom
    omega_min = math.sqrt(9.0 * mu * (2.0 * mu + lam) / (denom * rho * radius ** 2)) * math.sqrt(eps)
    omega_max = math.sqrt(15.0 * mu / (rho * radius ** 2)) * math.sqrt(eps)
    return {
        "radius": radius,
        "beta_ball": beta,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "hertz_min": omega_min / (2.0 * math.pi),
        "hertz_max": omega_max / (2.0 * math.pi),
        "width_ratio": omega_max / omega_min,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
def digest(out):
    return (round(float(out["radius"]), 12), round(float(out["beta_ball"]), 8),
            round(float(out["omega_min"]), 8), round(float(out["omega_max"]), 8),
            round(float(out["hertz_min"]), 8), round(float(out["hertz_max"]), 8),
            round(float(out["width_ratio"]), 10))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat(digest(dilute_ball_reference(1.5e6, 5.0e5, 1200.0, 5.0e-3, 3.75e-07)))",
            "gold_call": "flat(digest(_oracle_dilute_ball_reference(1.5e6, 5.0e5, 1200.0, 5.0e-3, 3.75e-07)))",
        },
        {
            "setup": """import math
def consistency(out, rho, eps, vol):
    # the lower endpoint must be exactly what the ball capacity gives through the frequency relation
    implied = math.sqrt(out["beta_ball"] / ((rho / eps) * vol))
    return (round(abs(implied - out["omega_min"]), 10), round(out["width_ratio"], 10))
def flat(x):
    # flatten to a tuple of plain numeric terminals, which is what AutoQC compares
    if isinstance(x, (tuple, list)):
        return tuple(v for e in x for v in flat(e))
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    return (int(x),) if isinstance(x, bool) else (x,)
""",
            "call": "flat((consistency(dilute_ball_reference(1.5e6, 5.0e5, 1200.0, 5.0e-3, 3.75e-07), 1200.0, 5.0e-3, 3.75e-07), consistency(dilute_ball_reference(0.0, 1.0e6, 900.0, 1.0e-2, 1.0e-06), 900.0, 1.0e-2, 1.0e-06)))",
            "gold_call": "flat((consistency(_oracle_dilute_ball_reference(1.5e6, 5.0e5, 1200.0, 5.0e-3, 3.75e-07), 1200.0, 5.0e-3, 3.75e-07), consistency(_oracle_dilute_ball_reference(0.0, 1.0e6, 900.0, 1.0e-2, 1.0e-06), 900.0, 1.0e-2, 1.0e-06)))",
        },
        {
            "setup": """
def verdict(fn, lam=1.5e6, mu=5.0e5, rho=1200.0, eps=5e-3, vol=3.75e-07):
    try:
        fn(lam, mu, rho, eps, vol)
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
    return flat((verdict(fn, mu=0.0), verdict(fn, rho=-1.0), verdict(fn, vol=0.0), verdict(fn, eps=1.0), verdict(fn, lam=float('nan')), verdict(fn, lam=-2.0e6, mu=5.0e5), verdict(fn, lam=-2.1, mu=1.0, rho=1000.0, vol=1e-6), verdict(fn)))
""",
            'call': 'verdicts(dilute_ball_reference)',
            'gold_call': 'verdicts(_oracle_dilute_ball_reference)',
        },
    ]
