"""
Invert a model exchange hole pointwise to find the normalisation it would need in order to reproduce a given exact exchange energy density. Define first the model potential P(rho, Q, N) at the reference point of a hole of normalisation N sitting in a density rho with curvature Q: solve x*exp(-2*x/3)/(x-2) = (2/3)*(pi*rho/N)**(2/3)*rho/Q for x, taking the unique root larger than two when the right-hand side is positive and the unique root between zero and two when it is negative; then set alpha = (8*pi*rho*exp(x)/N)**(1/3) and b = x/alpha, and take P = -N*(1 - exp(-x) - x*exp(-x)/2)/b. The target at each point is twice the exact exchange energy density divided by the density. Solve P(rho, Q, N) equals that target for N, searching only normalisations between zero and one, and return one wherever no normalisation at or below one reaches the target. P is monotone in N over that interval, so the solution is unique wherever one exists, and the returned values therefore never exceed one.

The model replaces the true hole by a normalised exponential displaced from the reference point, whose two parameters are fixed by matching the density and the curvature there. Its defining equation has exactly one root on each side of two, and which side is correct is decided by the sign of the curvature, so a solver that brackets blindly converges to the wrong branch wherever the curvature turns negative. For a channel holding one electron in one exponential orbital the model is exact and the inversion returns one everywhere; scaling that orbital's occupation to a fraction returns exactly that fraction, which is what makes the quantity a local measure of how much of an exchange electron a region actually holds. Values the inversion would place above one are artefacts of a model hole that is slightly too shallow there, not regions holding more than one electron.

Returns
-------
ndarray of shape (len(density),): the effective hole normalisation at each point, dimensionless and never above one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_hole_normalisation(density: "np.ndarray", curvature: "np.ndarray", exact_exchange_energy_density: "np.ndarray") -> "np.ndarray":
    '''Invert a model exchange hole pointwise to find the normalisation it would need in order to reproduce a given exact exchange energy density. Define first the model potential P(rho, Q, N) at the reference point of a hole of normalisation N sitting in a density rho with curvature Q: solve x*exp(-2*x/3)/(x-2) = (2/3)*(pi*rho/N)**(2/3)*rho/Q for x, taking the unique root larger than two when the right-hand side is positive and the unique root between zero and two when it is negative; then set alpha = (8*pi*rho*exp(x)/N)**(1/3) and b = x/alpha, and take P = -N*(1 - exp(-x) - x*exp(-x)/2)/b. The target at each point is twice the exact exchange energy density divided by the density. Solve P(rho, Q, N) equals that target for N, searching only normalisations between zero and one, and return one wherever no normalisation at or below one reaches the target. P is monotone in N over that interval, so the solution is unique wherever one exists, and the returned values therefore never exceed one.

    Parameters
    ----------
    density : np.ndarray
        Strictly positive spin density in inverse bohr cubed.
    curvature : np.ndarray
        Model-hole curvature of the same channel, same length; zero is a valid value.
    exact_exchange_energy_density : np.ndarray
        Strictly negative exact exchange energy density of the same channel, same length.

    Returns
    -------
    normalisation : np.ndarray
        ndarray of shape (len(density),): the effective hole normalisation at each point, dimensionless and never above one.

    Raises
    ------
    ValueError
        if density, curvature and exact_exchange_energy_density do not all have the same length, if any density value is not positive, or if any exact exchange energy density is not negative. A curvature of exactly zero is valid and is handled as the limiting case, not rejected.
    '''
    return normalisation  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _brx(x, rhs):
    """Residual and derivative of the Becke-Roussel defining equation."""
    e = np.exp(-2.0 * x / 3.0)
    return x * e / (x - 2.0) - rhs, 2.0 / 3.0 * (2.0 * x - x * x - 3.0) / (x - 2.0) ** 2 * e


def _bisect(fun, lo, hi, iters=200):
    """Vectorised bisection on a monotone residual, bracketed by lo and hi."""
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    flo = fun(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = fun(mid)
        same = np.sign(fm) == np.sign(flo)
        lo = np.where(same, mid, lo)
        flo = np.where(same, fm, flo)
        hi = np.where(same, hi, mid)
    return 0.5 * (lo + hi)


def _br_potential(density, curvature, hole_normalisation):
    """Becke-Roussel model exchange potential at the reference point."""
    rho = np.asarray(density, dtype=float).ravel()
    q = np.asarray(curvature, dtype=float).ravel()
    n = np.asarray(hole_normalisation, dtype=float).ravel()
    if rho.size != q.size:
        raise ValueError("density and curvature must have the same length")
    if n.size == 1:
        n = np.full(rho.size, float(n[0]))
    if n.size != rho.size:
        raise ValueError("hole_normalisation must be a scalar or match the density length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    if np.any(n <= 0.0):
        raise ValueError("every hole normalisation must be positive")
    flat = q == 0.0
    safe = np.where(flat, 1.0, q)
    rhs = (2.0 / 3.0) * (math.pi * rho / n) ** (2.0 / 3.0) * rho / safe
    lo = np.where(rhs < 0.0, 1e-12, 2.0 + 1e-13)
    hi = np.where(rhs < 0.0, 2.0 - 1e-13, 400.0)
    x = _bisect(lambda v: _brx(v, rhs)[0], lo, hi)
    for _ in range(80):
        f, df = _brx(x, rhs)
        x = x - f / df
    # A vanishing curvature sends the right-hand side to infinity from either side, and
    # the left-hand side diverges only at x = 2, so x = 2 IS the limit -- not an error.
    x = np.where(flat, 2.0, x)
    e = np.exp(-x)
    alpha = (8.0 * math.pi * rho / e / n) ** (1.0 / 3.0)
    b = x / alpha
    return -n * (1.0 - e - 0.5 * x * e) / b


def _oracle_effective_hole_normalisation(density: "np.ndarray", curvature: "np.ndarray", exact_exchange_energy_density: "np.ndarray") -> "np.ndarray":
    """Effective exchange-hole normalisation from the inverse Becke-Roussel procedure."""
    rho = np.asarray(density, dtype=float).ravel()
    q = np.asarray(curvature, dtype=float).ravel()
    eps = np.asarray(exact_exchange_energy_density, dtype=float).ravel()
    if not (rho.size == q.size == eps.size):
        raise ValueError("density, curvature and exact_exchange_energy_density must have the same length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    if np.any(eps >= 0.0):
        raise ValueError("every exact exchange energy density must be negative")
    target = 2.0 * eps / rho
    root = _bisect(lambda n: _br_potential(rho, q, n) - target,
                   np.full(rho.size, 1e-8), np.ones(rho.size))
    return np.minimum(root, 1.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)",
         "call": "effective_hole_normalisation(FA[0], FA[4], XA)",
         "gold_call": "_oracle_effective_hole_normalisation(FA[0], FA[4], XA)"},   # normal
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)",
         "call": "effective_hole_normalisation(FB[0], FB[4], XB)",
         "gold_call": "_oracle_effective_hole_normalisation(FB[0], FB[4], XB)"},   # boundary
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)",
         "call": "effective_hole_normalisation(FA[0], np.where(np.arange(FA[0].size) % 2 == 0, 0.0, FA[4]), XA)",
         "gold_call": "_oracle_effective_hole_normalisation(FA[0], np.where(np.arange(FA[0].size) % 2 == 0, 0.0, FA[4]), XA)"},   # boundary, vanishing curvature
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)",
         "call": "effective_hole_normalisation(FA[0], FA[4], 0.5*XA)",
         "gold_call": "_oracle_effective_hole_normalisation(FA[0], FA[4], 0.5*XA)"},   # edge
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)",
         "call": "effective_hole_normalisation(FA[0], FA[4], 3.0*XA)",
         "gold_call": "_oracle_effective_hole_normalisation(FA[0], FA[4], 3.0*XA)"},   # edge, target unreachable below one
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\ndef _exc(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_exc(lambda: effective_hole_normalisation(FA[0], FA[4], -XA))",
         "gold_call": "_exc(lambda: _oracle_effective_hole_normalisation(FA[0], FA[4], -XA))"},   # invalid input
    ]
