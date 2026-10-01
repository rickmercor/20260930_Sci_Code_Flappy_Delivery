"""
Given a set of points on the compactified radial coordinate together with the spacetime dimension, the multipole number and the spin label, return the compactification weight, its first derivative and the reduced potential at those points, in units where the horizon radius is one, along with the values of the reduced potential at the two ends of the interval.

A Tangherlini black hole is the d-dimensional generalisation of the Schwarzschild solution, with metric function f(r) = 1 - (r_h / r)^(d - 3) and event horizon at r = r_h. Linear perturbations of this background reduce to a single master wave equation in the tortoise coordinate x, defined by dx = dr / f, with an effective potential V that depends on the multipole number ell and on a spin label s. The label s = 0 covers massless scalar and gravitational tensor perturbations, s = 2 covers gravitational vector perturbations, and the potential collects the three contributions into one bracket: V = [f / (4 r^2)] times the sum of 4 ell (ell + d - 3), of (d - 2)(d - 4), and of (1 - s^2)(d - 2)^2 (r_h / r)^(d - 3).

The total transmission mode problem is posed on a compact domain by the inverse radial coordinate sigma = r_h / r, which maps the exterior onto the unit interval with sigma = 0 at infinity and sigma = 1 at the horizon. Two coefficient functions govern everything that follows. The first is the weight p(sigma) = -r_h dsigma / dx, which is the Jacobian of the compactification measured against the tortoise coordinate and equals sigma^2 f. It vanishes quadratically at infinity and linearly at the horizon, and that double degeneracy is what later removes the need to impose boundary conditions by hand. Its derivative is wanted as well, because the second-order operator built from it in step 3 is expanded rather than left in divergence form. The second is the reduced potential q(sigma) = r_h^2 V / p. Forming this ratio is worthwhile because the factors of f and of r^(-2) cancel exactly against the weight, leaving a polynomial in sigma of degree d - 3 with no singularity anywhere on the closed interval, which a spectral method represents exactly rather than approximately.

The sign of the reduced potential is not uniform. For s = 0 every term in the bracket is non-negative, but for s = 2 the coefficient (1 - s^2) equals -3, so the last term subtracts and the reduced potential becomes negative near the horizon once the dimension is large enough. At d = 14 and ell = 2 it falls from 56 at infinity to -52 at the horizon. This is the sector that carries the anomalously well conditioned mode, so the negativity has to be carried through the construction rather than assumed away, and it is the reason the positive definiteness of the energy inner product built later is a statement that needs checking rather than an immediate consequence of a positive potential.

Returns
-------
dict holding the array weight, p(sigma); the array weight_derivative, the derivative of p with respect to sigma; the array reduced_potential, q(sigma); the float potential_at_horizon, q evaluated at sigma equal to one; and the float potential_at_infinity, q evaluated at sigma equal to zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compactified_coefficients(
    sigma: np.ndarray,
    d: int,
    ell: int,
    s: float,
) -> dict:
    """Evaluate the compactification weight and the reduced potential of a Tangherlini background.

    Parameters
    ----------
    sigma : np.ndarray
        Points on the compactified radial coordinate, in the closed unit interval.
    d : int
        Spacetime dimension.
    ell : int
        Multipole number.
    s : float
        Spin label; zero for scalar and tensor perturbations, two for gravitational vector perturbations.

    Returns
    -------
    dict
        Under the keys weight, weight_derivative, reduced_potential, potential_at_horizon and
        potential_at_infinity.

    Raises
    ------
    ValueError
        When sigma is not a finite one-dimensional array of at least two points inside the closed unit
        interval, when d is not an integer of at least four, when ell is not a non-negative integer, or
        when s is not finite and non-negative.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validate_background(sigma, d, ell, s):
    grid = np.asarray(sigma, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("sigma must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(grid)):
        raise ValueError("sigma must be finite")
    if np.any(grid < 0.0) or np.any(grid > 1.0):
        raise ValueError("sigma must lie in the closed interval from zero to one")
    if not isinstance(d, (int, np.integer)) or isinstance(d, bool) or int(d) < 4:
        raise ValueError("d must be an integer of at least four")
    if not isinstance(ell, (int, np.integer)) or isinstance(ell, bool) or int(ell) < 0:
        raise ValueError("ell must be a non-negative integer")
    spin = float(s)
    if not np.isfinite(spin) or spin < 0.0:
        raise ValueError("s must be finite and non-negative")
    return grid, int(d), int(ell), spin


def _reduced_potential_parts(d, ell, s):
    """Constant and sigma^(d - 3) coefficients of the reduced potential q."""
    constant = float(ell) * (ell + d - 3) + (d - 2) * (d - 4) / 4.0
    varying = (1.0 - s ** 2) * (d - 2) ** 2 / 4.0
    return constant, varying


def _oracle_compactified_coefficients(
    sigma: np.ndarray,
    d: int,
    ell: int,
    s: float,
) -> dict:
    """Reference implementation."""
    grid, d, ell, spin = _validate_background(sigma, d, ell, s)
    m = d - 3
    weight = grid ** 2 - grid ** (m + 2)
    weight_derivative = 2.0 * grid - (m + 2) * grid ** (m + 1)
    constant, varying = _reduced_potential_parts(d, ell, spin)
    reduced = constant + varying * grid ** m
    return {
        "weight": weight,
        "weight_derivative": weight_derivative,
        "reduced_potential": reduced,
        "potential_at_horizon": float(constant + varying),
        "potential_at_infinity": float(constant),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
def flat(x):
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

    SETUP = """
import numpy as np
GRID = np.linspace(0.0, 1.0, 11)
def digest(out):
    return (np.round(out["weight"], 12), np.round(out["weight_derivative"], 12),
            np.round(out["reduced_potential"], 12),
            round(out["potential_at_horizon"], 12), round(out["potential_at_infinity"], 12))
"""
    return [
        {
            # the graded sector: gravitational vector perturbations of a fourteen-dimensional background,
            # where the reduced potential changes sign between infinity and the horizon
            "setup": SETUP + FLAT,
            "call": "flat(digest(compactified_coefficients(GRID, 14, 2, 2.0)))",
            "gold_call": "flat(digest(_oracle_compactified_coefficients(GRID, 14, 2, 2.0)))",
        },
        {
            # boundary: the four-dimensional Schwarzschild limit, where the weight is sigma^2 (1 - sigma)
            # and the reduced potential of the vector sector is 6 - 3 sigma for ell = 2, staying positive
            "setup": SETUP + """
def schwarzschild(fn):
    out = fn(GRID, 4, 2, 2.0)
    exact_p = GRID ** 2 * (1.0 - GRID)
    exact_dp = 2.0 * GRID - 3.0 * GRID ** 2
    exact_q = 6.0 - 3.0 * GRID
    return (int(np.max(np.abs(out["weight"] - exact_p)) < 1e-14),
            int(np.max(np.abs(out["weight_derivative"] - exact_dp)) < 1e-14),
            int(np.max(np.abs(out["reduced_potential"] - exact_q)) < 1e-14),
            round(out["potential_at_horizon"], 12), round(out["potential_at_infinity"], 12))
""" + FLAT,
            "call": "flat(schwarzschild(compactified_coefficients))",
            "gold_call": "flat(schwarzschild(_oracle_compactified_coefficients))",
        },
        {
            # edge: the endpoints alone, where the weight vanishes at both ends for every dimension,
            # together with the scalar sector, whose reduced potential is positive throughout
            "setup": SETUP + FLAT,
            "call": "flat(digest(compactified_coefficients(np.array([0.0, 0.5, 1.0]), 9, 2, 0.0)))",
            "gold_call": "flat(digest(_oracle_compactified_coefficients(np.array([0.0, 0.5, 1.0]), 9, 2, 0.0)))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(sigma=GRID, d=14, ell=2, s=2.0)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(compactified_coefficients, sigma=np.array([0.0, 1.5])), "
                    "verdict(compactified_coefficients, d=3), "
                    "verdict(compactified_coefficients, ell=-1), "
                    "verdict(compactified_coefficients, s=-1.0), "
                    "verdict(compactified_coefficients, sigma=np.array([0.5]))))",
            "gold_call": "flat((verdict(_oracle_compactified_coefficients, sigma=np.array([0.0, 1.5])), "
                         "verdict(_oracle_compactified_coefficients, d=3), "
                         "verdict(_oracle_compactified_coefficients, ell=-1), "
                         "verdict(_oracle_compactified_coefficients, s=-1.0), "
                         "verdict(_oracle_compactified_coefficients, sigma=np.array([0.5]))))",
        },
    ]
