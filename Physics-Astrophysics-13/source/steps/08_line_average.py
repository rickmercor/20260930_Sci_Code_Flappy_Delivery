"""
Step 08: the curve average of a sampled quantity.

Convention (pinned)



Given an ordered polyline of sample points and one value per sample, return



  average  the trapezoidal integral of the values in arclength, taken with the

           straight-line distances between consecutive sample points, divided

           by the total sampled arclength;

  total    that total sampled arclength.



Values may be scalar, shape (N,), or vector, shape (N, 3); in the vector case

the same rule is applied component by component and the average comes back with

shape (3,). The value array is validated as supplied - an array whose shape is

neither (N,) nor (N, 3) is rejected rather than flattened to fit.



Inputs



points : array_like

    Shape (N, 3), N >= 2, all entries finite.

values : array_like

    Shape (N,) or (N, 3), all entries finite.



Returns



tuple

    (average, total). average is a float for scalar values and a float64

    array of shape (3,) for vector values; total is a float.

Returns
-------
A tuple (average, total): average is a float for scalar values and a numpy.ndarray of shape (3,) for vector values; total is a float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def line_average(points, values):
    """Return the arclength-weighted average and the total sampled arclength.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 2, all finite.
    values : array_like
        Shape (N,) or (N, 3), all finite.

    Returns
    -------
    average : float or numpy.ndarray
        Float for scalar values; shape (3,) for vector values.
    total : float
        The total sampled arclength.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 2, if values is not a
        finite array of shape (N,) or (N, 3), or if the total sampled
        arclength is zero.
    """
    return average, total  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_line_average(points, values):
    """Reference implementation of line_average (deterministic)."""
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("points must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")
    if v.ndim == 1:
        if v.shape[0] != p.shape[0]:
            raise ValueError("values must have length N")
    elif v.ndim == 2:
        if v.shape != (p.shape[0], 3):
            raise ValueError("vector values must have shape (N, 3)")
    else:
        raise ValueError("values must have shape (N,) or (N, 3)")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")

    ds = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    total = float(np.sum(ds))
    if total == 0.0:
        raise ValueError("total sampled arclength is zero")

    w = ds if v.ndim == 1 else ds[:, None]
    integral = np.sum(0.5 * (v[1:] + v[:-1]) * w, axis=0)
    average = integral / total
    if v.ndim == 1:
        return float(average), total
    return np.asarray(average, dtype=np.float64), total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for line_average."""
    return [
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\nzs = np.linspace(-0.8, 0.8, 81)\npts = _oracle_quasi_x_line(zs, sd)\nvals = _oracle_parallel_electric_field(pts) * 1.0e6\n', 'call': 'line_average(pts, vals)', 'gold_call': '_oracle_line_average(pts, vals)'},
        {'setup': "import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\nzs = np.linspace(-0.8, 0.8, 81)\npts = _oracle_quasi_x_line(zs, sd)\nvals = _oracle_field_samples(pts, 'B')\ndef pack(res):\n    avg, tot = res\n    return np.concatenate([np.atleast_1d(np.asarray(avg, dtype=float)),\n                           np.array([float(tot)])])\n", 'call': 'pack(line_average(pts, vals))', 'gold_call': 'pack(_oracle_line_average(pts, vals))'},
        {'setup': 'import numpy as np\npts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])\nvals = np.array([2.0, 8.0])\n', 'call': 'line_average(pts, vals)', 'gold_call': '_oracle_line_average(pts, vals)'},
        {'setup': 'import numpy as np\npts = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0],\n                [0.02, 0.0, 0.0], [5.0, 0.0, 0.0]])\nvals = np.array([1.0, 1.0, 1.0, 9.0])\n', 'call': 'line_average(pts, vals)', 'gold_call': '_oracle_line_average(pts, vals)'},
        {'setup': 'import numpy as np\npts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],\n                [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])\nbad = np.array([[1.0, 2.0], [3.0, 4.0]])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(line_average, pts, bad)', 'gold_call': '_status(_oracle_line_average, pts, bad)'},
        {'setup': 'import numpy as np\npts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],\n                [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])\nbad = np.arange(4.0).reshape(4, 1)\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(line_average, pts, bad)', 'gold_call': '_status(_oracle_line_average, pts, bad)'},
        {'setup': 'import numpy as np\npts = np.array([[0.0, 0.0, 0.0]])\nvals = np.array([1.0])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(line_average, pts, vals)', 'gold_call': '_status(_oracle_line_average, pts, vals)'},
    ]
