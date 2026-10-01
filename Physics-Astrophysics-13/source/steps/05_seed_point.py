"""
Step 05: the distinguished point of the step-04 candidate curve.

Contract



Consider the step-04 curve restricted to the closed height interval

[z_lo, z_hi]. Return the single point of that restricted curve at which the

step-03 discriminant is largest, as a float64 array of shape (3,) ordered

(x, y, z).



The maximiser is unique on any interval on which the curve exists. It may fall

in the interior or at either end of the interval; both cases must be handled.

Locate it to a relative accuracy of 1e-12 or better - a grid search coarser

than that does not meet the contract.



Inputs



z_lo, z_hi : float

    Finite reals with z_lo < z_hi, the whole closed interval admissible for

    the step-04 curve.



Returns



numpy.ndarray

    Shape (3,), dtype float64.

Returns
-------
A numpy.ndarray of shape (3,) and dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def seed_point(z_lo, z_hi):
    """Return the point of the step-04 curve of largest step-03 discriminant.

    Parameters
    ----------
    z_lo, z_hi : float
        Finite reals with z_lo < z_hi, the closed interval to search.

    Returns
    -------
    numpy.ndarray
        Shape (3,), dtype float64, ordered (x, y, z).

    Raises
    ------
    ValueError
        If z_lo or z_hi is not a finite real, if z_lo >= z_hi, or if the
        step-04 curve does not exist somewhere on the closed interval.
    """
    return point  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

AL = 0.4
CC = 0.2


def _oracle_seed_point(z_lo, z_hi):
    """Reference implementation of seed_point (deterministic).

    Composes the step-03 and step-04 oracle functions directly, so this
    reference value never depends on a submitted implementation.
    """
    try:
        lo = float(z_lo)
        hi = float(z_hi)
    except (TypeError, ValueError):
        raise ValueError("z_lo and z_hi must be real numbers")
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("z_lo and z_hi must be finite")
    if not (lo < hi):
        raise ValueError("z_lo must be strictly less than z_hi")

    # The discriminant along the step-04 curve is the square root of a downward
    # parabola in the height, so it is strictly concave and its maximiser over
    # a closed interval is either the stationary point or one of the two ends.
    # All three are evaluated with the step-03 discriminant itself and the
    # largest is selected, so the selection rests on the measured quantity
    # rather than on the algebra used to shortlist the candidates. Passing all
    # three heights to step 04 at once also enforces existence over the whole
    # closed interval, not merely at the maximiser.
    z_star = -AL * CC / 2.0
    heights = np.array([min(max(z_star, lo), hi), lo, hi], dtype=np.float64)
    curve = _oracle_candidate_line(heights)
    phi = np.asarray(_oracle_alignment_residual(curve), dtype=np.float64)[:, 3]
    return np.asarray(curve[int(np.argmax(phi))], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for seed_point."""
    return [
        {'setup': 'import numpy as np\n', 'call': 'seed_point(-0.8, 0.8)', 'gold_call': '_oracle_seed_point(-0.8, 0.8)'},
        {'setup': 'import numpy as np\n', 'call': 'seed_point(0.2, 0.8)', 'gold_call': '_oracle_seed_point(0.2, 0.8)'},
        {'setup': 'import numpy as np\n', 'call': 'seed_point(-1.0, -0.5)', 'gold_call': '_oracle_seed_point(-1.0, -0.5)'},
        {'setup': 'import numpy as np\n', 'call': 'seed_point(-0.05, -0.03)', 'gold_call': '_oracle_seed_point(-0.05, -0.03)'},
        {'setup': 'def _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(seed_point, 0.5, -0.5)', 'gold_call': '_status(_oracle_seed_point, 0.5, -0.5)'},
        {'setup': 'def _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(seed_point, -0.5, 1.4)', 'gold_call': '_status(_oracle_seed_point, -0.5, 1.4)'},
    ]
