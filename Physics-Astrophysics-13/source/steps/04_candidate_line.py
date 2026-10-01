"""
Step 04: the candidate curve on which the step-03 residual vanishes.

Contract



Return a float64 array of shape (N, 3) whose n-th row is the sought point at

z = z_values[n], the row ordered (x, y, z) with the third entry equal to the

supplied z.



At each admissible height the step-03 residual vanishes on exactly two points.

Return the one whose step-03 discriminant is strictly positive; that selection

is the whole content of this step. At heights where no such point exists the

step raises rather than returning a complex or clipped value.



This step returns a CANDIDATE curve only. Nothing here asserts that the curve

it returns is the curve the benchmark finally samples.



Input validation happens on the array as supplied. A scalar, a zero-dimensional

array and any array of two or more dimensions are all rejected; the argument is

not flattened first.



Inputs



z_values : array_like

    One-dimensional, finite, length >= 1, every entry admissible in the sense

    above.



Returns



numpy.ndarray

    Shape (len(z_values), 3), dtype float64.

Returns
-------
A numpy.ndarray of shape (N, 3) and dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def candidate_line(z_values):
    """Return the (N, 3) sample points of the candidate curve.

    Of the two points at which the step-03 vector residual vanishes at a given
    height, this returns the one carrying a strictly positive step-03
    discriminant.

    Parameters
    ----------
    z_values : array_like
        One-dimensional finite heights, length >= 1, each admissible in the
        sense of the module docstring.

    Returns
    -------
    numpy.ndarray
        Shape (N, 3), dtype float64.

    Raises
    ------
    ValueError
        If z_values is not a one-dimensional finite array of length >= 1, or if
        the sought point does not exist at some supplied height.
    """
    return points  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

AL = 0.4
B0 = 0.5
CC = 0.2
MU = 1.0


def _oracle_candidate_line(z_values):
    """Reference implementation of candidate_line (deterministic)."""
    z = np.asarray(z_values, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("z_values must be one-dimensional with length >= 1")
    if not np.all(np.isfinite(z)):
        raise ValueError("z_values must be finite")

    disc = AL ** 4 + 8.0 * AL ** 2 + 4.0 * MU + 4.0 * CC * B0 \
        - 4.0 * z ** 2 - 4.0 * AL * CC * z
    if np.any(disc <= 0.0):
        raise ValueError(
            "every z must satisfy 4*z**2 + 4*AL*CC*z < "
            "AL**4 + 8*AL**2 + 4*MU + 4*CC*B0")

    d = np.sqrt(disc)
    y = 2.0 + 0.5 * AL ** 2 - 0.5 * d
    by = B0 * (2.0 * z + AL * CC) / d
    x = -by - AL * y + CC * z
    return np.stack([x, y, z], axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for candidate_line."""
    return [
        {'setup': 'import numpy as np\nzs = np.linspace(-0.8, 0.8, 81)\n', 'call': 'candidate_line(zs)', 'gold_call': '_oracle_candidate_line(zs)'},
        {'setup': 'import numpy as np\nzs = np.array([-0.04])\n', 'call': 'candidate_line(zs)', 'gold_call': '_oracle_candidate_line(zs)'},
        {'setup': 'import numpy as np\nzs = np.array([-1.15, -0.5, 0.0, 0.5, 1.05])\n', 'call': 'candidate_line(zs)', 'gold_call': '_oracle_candidate_line(zs)'},
        {'setup': 'import numpy as np\nzs = np.array([0.0, 1.3])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(candidate_line, zs)', 'gold_call': '_status(_oracle_candidate_line, zs)'},
        {'setup': 'import numpy as np\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(candidate_line, 0.3)', 'gold_call': '_status(_oracle_candidate_line, 0.3)'},
        {'setup': 'import numpy as np\nzs = np.array([[-0.2, 0.1], [0.3, 0.4]])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(candidate_line, zs)', 'gold_call': '_status(_oracle_candidate_line, zs)'},
        {'setup': 'import numpy as np\nzs = np.array([])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(candidate_line, zs)', 'gold_call': '_status(_oracle_candidate_line, zs)'},
    ]
