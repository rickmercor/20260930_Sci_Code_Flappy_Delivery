"""
Implement kinetic_bounds to return the maximal admissible Boltzmann-prefactor product, the global minimum nonnegative gap for α=1, and the global maximum α at a supplied gap. Apply the result to the retrieved ROKS gap, identify the nonradiative-loss boundary, and assess the conditional validity of α=1.

The ROKS gap is a source quantity. A failure of the combined energy-and-rate-ratio assumption does not alone falsify the calculated energy; physical admissibility depends on nonnegative nonradiative decay.

Returns
-------
np.ndarray, shape (3,), float64: [z_max, minimum_gap_at_alpha_one (eV), maximum_alpha_at_given_gap]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def kinetic_bounds(H: float, decay: float, kappa: float,
                   theta: float, gap: float) -> np.ndarray:
    """Compute global kinetic bounds from d=decay-H*z and d>=kappa.

    H>0, decay>kappa>0 are s^-1; theta>0 and gap>=0 are eV.
    The gap argument is the retrieved ROKS T2-T1 gap when applying this
    step to the paper. z=alpha*exp(-gap/theta). Return its maximal value,
    the smallest nonnegative gap compatible with alpha=1, and the
    maximal alpha at the supplied gap. Bounds are attained at d=kappa,
    except that the minimum gap is zero if alpha=1 already fits there.
    Do not infer an unconditional measured energy gap from this result.

    Returns
    -------
    np.ndarray
        Shape (3,), [z_max, minimum_gap, maximum_alpha].

    Raises
    ------
    ValueError
        If inputs are not finite real scalars, H,theta,kappa are
        nonpositive, decay<=kappa, gap<0, or bounds are nonfinite or
        z_max is nonpositive in floating-point arithmetic.
    """
    return np.empty(3, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_kinetic_bounds(H, decay, kappa, theta, gap):
    import numpy as np
    try:
        raw = [H,decay,kappa,theta,gap]
        if np.iscomplexobj(raw):
            raise ValueError("real inputs required")
        v = np.asarray(raw,dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalars required") from exc
    if v.shape != (5,) or not np.all(np.isfinite(v)):
        raise ValueError("finite scalars required")
    H,decay,kappa,theta,gap = v
    if min(H,kappa,theta) <= 0 or decay <= kappa or gap < 0:
        raise ValueError("invalid bound domain")
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        zmax = (decay-kappa)/H
        logz = np.log(zmax)
        minimum = max(0.0,float(-theta*logz))
        maximum = np.exp(logz+gap/theta)
    result = np.array([zmax,minimum,maximum],dtype=float)
    if zmax <= 0 or not np.all(np.isfinite(result)):
        raise ValueError("nonfinite or unrepresentable bounds")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Valid cases compare outputs; invalid cases explicitly require ValueError."""
    return [
        {
            "setup": "",
            "call": "kinetic_bounds(750019.1987918274, 1/.12, 1., .0256, .041)",
            "gold_call": "_oracle_kinetic_bounds(750019.1987918274, 1/.12, 1., .0256, .041)"
        },
        {
            "setup": "",
            "call": "kinetic_bounds(2., 3., 1., .0256, 0.)",
            "gold_call": "_oracle_kinetic_bounds(2., 3., 1., .0256, 0.)"
        },
        {
            "setup": "",
            "call": "kinetic_bounds(1., 3., 1., .05, .02)",
            "gold_call": "_oracle_kinetic_bounds(1., 3., 1., .05, .02)"
        },
        {
            "setup": "def _expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError(\"Expected ValueError for invalid input\")",
            "call": "_expect_value_error(kinetic_bounds, 10., 1., 1., .0256, .041)",
            "gold_call": "_expect_value_error(_oracle_kinetic_bounds, 10., 1., 1., .0256, .041)"
        }
    ]
