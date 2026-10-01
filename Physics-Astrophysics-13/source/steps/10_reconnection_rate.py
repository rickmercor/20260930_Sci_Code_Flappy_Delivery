"""
Step 10: the normalized reconnection rate (final orchestrator step).

Contract



Chain the earlier steps end to end on the benchmark configuration and return

the single dimensionless number the task asks for:



  (1) take the step-05 distinguished point of the step-04 candidate curve over

     the closed height interval [z_lo, z_hi];

  (2) take the step-06 curve through that point, sampled at n_points heights

     equally spaced over the same closed interval;

  (3) form the step-08 curve average of the step-07 field-aligned electric

     field over those samples, giving the numerator R_0 in statvolt/cm;

  (4) form the step-09 inflow quantities at displacement delta;

  (5) return R_0 divided by B_in V_A / c, with c the speed of light in

     Gaussian units.



The default arguments are the benchmark configuration and reproduce the

answer quoted in the golden solution.



Inputs



z_lo, z_hi : float

    Height interval, default -0.8 and 0.8.

n_points : int

    Number of equally spaced sample heights, default 81, at least 2.

delta : float

    Inflow displacement, default 0.05, strictly positive and finite.



Returns



float

    The dimensionless normalized reconnection rate.

Returns
-------
A single float, the dimensionless normalized reconnection rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reconnection_rate(z_lo=-0.8, z_hi=0.8, n_points=81, delta=0.05):
    """Return the dimensionless normalized reconnection rate.

    Parameters
    ----------
    z_lo, z_hi : float
        Height interval; z_lo < z_hi.
    n_points : int
        Number of equally spaced sample heights, at least 2.
    delta : float
        Strictly positive finite inflow displacement.

    Returns
    -------
    float
        The dimensionless rate.

    Raises
    ------
    ValueError
        If the interval is not valid, if n_points is not an integer of at
        least 2, or if delta is not a positive finite real.
    """
    return rate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

C_LIGHT = 2.99792458e10


def _oracle_reconnection_rate(z_lo=-0.8, z_hi=0.8, n_points=81, delta=0.05):
    """Reference implementation of reconnection_rate (deterministic).

    Composes the step-05, step-06, step-07, step-08 and step-09 oracle
    functions directly, so this reference value never depends on a submitted
    implementation.
    """
    try:
        lo = float(z_lo)
        hi = float(z_hi)
    except (TypeError, ValueError):
        raise ValueError("z_lo and z_hi must be real numbers")
    if not (np.isfinite(lo) and np.isfinite(hi)) or not (lo < hi):
        raise ValueError("require finite z_lo < z_hi")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("n_points must be an integer")
    if int(n_points) < 2:
        raise ValueError("n_points must be at least 2")
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        raise ValueError("delta must be a real number")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be a positive finite real")

    seed = _oracle_seed_point(lo, hi)
    zs = np.linspace(lo, hi, int(n_points))
    pts = _oracle_quasi_x_line(zs, seed)

    epar = _oracle_parallel_electric_field(pts)
    r0, total_length = _oracle_line_average(pts, epar)
    b_in, rho_s, v_a = _oracle_inflow_quantities(pts, delta)

    # Certify the delivered configuration before reporting. The candidate curve
    # must actually solve the alignment condition; the curve that is sampled
    # must carry a nowhere-vanishing field and a finite gradient tensor, and it
    # must stay on the positive-discriminant branch along its whole length,
    # which is the filter the extraction ends with.
    candidate = _oracle_candidate_line(zs)
    residual = np.asarray(_oracle_alignment_residual(candidate), dtype=np.float64)
    if not np.all(np.abs(residual[:, 0:3]) < 1.0e-9):
        raise ValueError("candidate curve does not solve the alignment condition")

    field_on_line = np.asarray(_oracle_field_samples(pts, "B"), dtype=np.float64)
    tensor_on_line = np.asarray(_oracle_gradient_tensor(pts), dtype=np.float64)
    if np.any(np.linalg.norm(field_on_line, axis=-1) <= 0.0):
        raise ValueError("magnetic field vanishes on the sampled curve")
    if not np.all(np.isfinite(tensor_on_line)):
        raise ValueError("non-finite gradient tensor on the sampled curve")

    diagnostics = np.asarray(_oracle_alignment_residual(pts), dtype=np.float64)
    if not np.all(diagnostics[:, 3] > 0.0):
        raise ValueError("sampled curve leaves the positive-discriminant branch")

    if not (total_length > 0.0 and b_in > 0.0 and rho_s > 0.0 and v_a > 0.0):
        raise ValueError("degenerate curve or inflow normalisation")
    if not np.isfinite(r0):
        raise ValueError("non-finite arclength-averaged parallel electric field")

    return float(r0) / (b_in * v_a / C_LIGHT)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for reconnection_rate."""
    return [
        {'setup': 'import numpy as np\n', 'call': 'reconnection_rate()', 'gold_call': '_oracle_reconnection_rate()'},
        {'setup': 'import numpy as np\n', 'call': 'reconnection_rate(-0.8, 0.8, 161, 0.05)', 'gold_call': '_oracle_reconnection_rate(-0.8, 0.8, 161, 0.05)'},
        {'setup': 'import numpy as np\n', 'call': 'reconnection_rate(-0.8, 0.8, 2, 0.05)', 'gold_call': '_oracle_reconnection_rate(-0.8, 0.8, 2, 0.05)'},
        {'setup': 'import numpy as np\n', 'call': 'reconnection_rate(0.1, 0.7, 41, 0.2)', 'gold_call': '_oracle_reconnection_rate(0.1, 0.7, 41, 0.2)'},
        {'setup': 'def _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(reconnection_rate, -0.8, 0.8, 1, 0.05)', 'gold_call': '_status(_oracle_reconnection_rate, -0.8, 0.8, 1, 0.05)'},
        {'setup': 'def _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(reconnection_rate, -0.8, 0.8, 81, -0.05)', 'gold_call': '_status(_oracle_reconnection_rate, -0.8, 0.8, 81, -0.05)'},
    ]
