"""
Step 06: the magnetic field line through a supplied point.

Contract



Return a float64 array of shape (N, 3) whose n-th row is the point of the

integral curve of the magnetic field through seed at height

z = z_values[n], the row ordered (x, y, z) with the third entry equal to the

supplied z.



Convention (pinned)



The third component of the magnetic field is a positive constant, so the curve

is parameterised by the third coordinate throughout: the first two coordinates

obey the ordinary differential equations



    dx/dz = B_x / B_z ,      dy/dz = B_y / B_z ,



with the magnetic field taken from the step-01 public function. Integration

starts at seed, which lies on the curve by definition and is reproduced

exactly when its own height is requested, and proceeds outward in both

directions. Requested heights need not be sorted.



Accuracy: the returned coordinates must be correct to a relative accuracy of

1e-12 or better. Classical fourth-order Runge-Kutta with a uniform step of at

most 2^-12 in the third coordinate meets this, as does an adaptive

Runge-Kutta method of order five or higher run at a relative tolerance of

1e-12 or tighter.



Inputs



z_values : array_like

    One-dimensional, finite, length >= 1. Validated as supplied; a scalar or

    an array of two or more dimensions is rejected rather than flattened.

seed : array_like

    Shape (3,), finite.



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


def quasi_x_line(z_values, seed):
    """Return the field line through seed, sampled at the requested heights.

    Parameters
    ----------
    z_values : array_like
        One-dimensional finite heights, length >= 1.
    seed : array_like
        Shape (3,), finite; a point of the curve.

    Returns
    -------
    numpy.ndarray
        Shape (N, 3), dtype float64.

    Raises
    ------
    ValueError
        If z_values is not a one-dimensional finite array of length >= 1, or
        if seed is not a finite array of shape (3,).
    """
    return points  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

_RK_STEP = 2.0 ** -12


def _oracle_quasi_x_line(z_values, seed):
    """Reference implementation of quasi_x_line (deterministic).

    Composes the step-01 oracle function directly, so this reference value
    never depends on a submitted implementation.
    """
    z = np.asarray(z_values, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("z_values must be one-dimensional with length >= 1")
    if not np.all(np.isfinite(z)):
        raise ValueError("z_values must be finite")

    s = np.asarray(seed, dtype=np.float64)
    if s.ndim != 1 or s.size != 3:
        raise ValueError("seed must have shape (3,)")
    if not np.all(np.isfinite(s)):
        raise ValueError("seed must be finite")

    z0 = float(s[2])

    def _slope(zz, u):
        pt = np.array([[u[0], u[1], zz]], dtype=np.float64)
        b = np.asarray(_oracle_field_samples(pt, "B"), dtype=np.float64)[0]
        return np.array([b[0] / b[2], b[1] / b[2]], dtype=np.float64)

    out = np.empty((z.size, 3), dtype=np.float64)
    order = np.argsort(z)
    upper = [int(i) for i in order if z[i] > z0]
    lower = [int(i) for i in order[::-1] if z[i] < z0]

    for run in (upper, lower):
        u = np.array([s[0], s[1]], dtype=np.float64)
        cur = z0
        for idx in run:
            target = float(z[idx])
            nsub = max(1, int(math.ceil(abs(target - cur) / _RK_STEP)))
            h = (target - cur) / nsub
            for _ in range(nsub):
                k1 = _slope(cur, u)
                k2 = _slope(cur + 0.5 * h, u + 0.5 * h * k1)
                k3 = _slope(cur + 0.5 * h, u + 0.5 * h * k2)
                k4 = _slope(cur + h, u + h * k3)
                u = u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                cur = cur + h
            out[idx] = [u[0], u[1], target]

    for i in range(z.size):
        if z[i] == z0:
            out[i] = [s[0], s[1], z0]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for quasi_x_line."""
    return [
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\nzs = np.linspace(-0.8, -0.04, 39)\n', 'call': 'quasi_x_line(zs, sd)', 'gold_call': '_oracle_quasi_x_line(zs, sd)'},
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\nzs = np.array([0.2, -0.6, -0.04, -0.3, 0.05, -0.45])\n', 'call': 'quasi_x_line(zs, sd)', 'gold_call': '_oracle_quasi_x_line(zs, sd)'},
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\nzs = np.array([-0.04])\n', 'call': 'quasi_x_line(zs, sd)', 'gold_call': '_oracle_quasi_x_line(zs, sd)'},
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-1.0, -0.5)\nzs = np.array([-0.8, -0.65, -0.5, -0.35])\n', 'call': 'quasi_x_line(zs, sd)', 'gold_call': '_oracle_quasi_x_line(zs, sd)'},
        {'setup': 'import numpy as np\nzs = np.array([-0.2, 0.0])\nbad = np.array([0.1, 0.2])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(quasi_x_line, zs, bad)', 'gold_call': '_status(_oracle_quasi_x_line, zs, bad)'},
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\nzs = np.array([[-0.2, 0.0], [0.1, 0.2]])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(quasi_x_line, zs, sd)', 'gold_call': '_status(_oracle_quasi_x_line, zs, sd)'},
    ]
