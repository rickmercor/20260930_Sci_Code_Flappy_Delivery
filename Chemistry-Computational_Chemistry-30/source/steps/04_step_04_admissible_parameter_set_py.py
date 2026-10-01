"""
Determine the entire set of nonnegative Δ, α and nonnegative triplet nonradiative loss consistent with the lifetime. Give both the necessary inequality and a constructive expression for d.

The independent radiative rate imposes d≥κ, not merely d≥0. The requested set may be parameterized in any equivalent way; numerical examples or a scan are not sufficient.

Returns
-------
np.ndarray, shape (4,), float64: [z, required_d (s^-1), z_max, admissible_flag (0.0 or 1.0)]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def admissible_point(H: float, decay: float, kappa: float, theta: float,
                     gap: float, alpha: float) -> np.ndarray:
    """Evaluate a member of the exact lifetime-matching parameter family.

    H>0 is from tail_coefficients; decay>=kappa>0 are s^-1;
    theta>0 and gap>=0 are eV; alpha>=0 is dimensionless.
    Calculate z=alpha*exp(-gap/theta), d=decay-H*z and the maximum
    allowed z from d>=kappa. A negative required d is diagnostic output,
    not an exception. The flag is 1 only for an admissible d.
    To handle arithmetic at the boundary, snap d to kappa when their
    difference is at most 64*machine_epsilon*max(1,decay,kappa).
    This numerical guard is not an experimental uncertainty allowance.

    Returns
    -------
    np.ndarray
        Shape (4,), [z,d,z_max,flag], with flag equal to 0.0 or 1.0.
        Together with gap>=0, alpha>=0 and d=decay-H*z, z<=z_max
        characterizes the whole physically admissible family.

    Raises
    ------
    ValueError
        If inputs are not finite real scalars, H,theta,kappa are
        nonpositive, decay<kappa, gap or alpha is negative, or
        calculated outputs are nonfinite.
    """
    return np.empty(4, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_admissible_point(H, decay, kappa, theta, gap, alpha):
    import numpy as np
    try:
        raw = [H, decay, kappa, theta, gap, alpha]
        if np.iscomplexobj(raw):
            raise ValueError("real inputs required")
        v = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalar inputs required") from exc
    if v.shape != (6,) or not np.all(np.isfinite(v)):
        raise ValueError("finite scalars required")
    H, decay, kappa, theta, gap, alpha = v
    if min(H,kappa,theta) <= 0 or decay < kappa or min(gap,alpha) < 0:
        raise ValueError("invalid parameter domain")
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        z = alpha*np.exp(-gap/theta)
        d = decay-H*z
        zmax = (decay-kappa)/H
    if not np.all(np.isfinite([z,d,zmax])):
        raise ValueError("nonfinite family evaluation")
    tol = 64*np.finfo(float).eps*max(1.0,decay,kappa)
    if abs(d-kappa) <= tol:
        d = kappa
    return np.array([z,d,zmax,float(d >= kappa)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Valid cases compare outputs; invalid cases explicitly require ValueError."""
    return [
        {
            "setup": "",
            "call": "admissible_point(750019.1987918274, 1.0/.12, 1., .0256, .041, 2e-5)",
            "gold_call": "_oracle_admissible_point(750019.1987918274, 1.0/.12, 1., .0256, .041, 2e-5)"
        },
        {
            "setup": "",
            "call": "admissible_point(10., 3., 1., .0256, 0., .2)",
            "gold_call": "_oracle_admissible_point(10., 3., 1., .0256, 0., .2)"
        },
        {
            "setup": "",
            "call": "admissible_point(10., 1., 1., .0256, .04, 0.)",
            "gold_call": "_oracle_admissible_point(10., 1., 1., .0256, .04, 0.)"
        },
        {
            "setup": "",
            "call": "admissible_point(750019.1987918274, 1.0/.12, 1., .0256, .041, 1.)",
            "gold_call": "_oracle_admissible_point(750019.1987918274, 1.0/.12, 1., .0256, .041, 1.)"
        },
        {
            "setup": "def _expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    raise AssertionError(\"Expected ValueError for invalid input\")",
            "call": "_expect_value_error(admissible_point, 10., 3., 1., .0256, -.01, .2)",
            "gold_call": "_expect_value_error(_oracle_admissible_point, 10., 3., 1., .0256, -.01, .2)"
        }
    ]
