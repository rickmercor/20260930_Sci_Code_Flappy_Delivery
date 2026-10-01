"""
Step 07: the field-aligned electric field at supplied positions.

Convention (pinned)



For each position return the scalar product of the electric field with the

magnetic field, divided by the magnitude of the magnetic field - not by its

square, and not projected onto a unit vector formed from anything else. Both

fields come from the step-01 public function. The result is in statvolt/cm.



Inputs



points : array_like

    Shape (N, 3), N >= 1, all entries finite.



Returns



numpy.ndarray

    Shape (N,), dtype float64.

Returns
-------
A numpy.ndarray of shape (N,) and dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def parallel_electric_field(points):
    """Return the field-aligned electric field at every supplied position.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 1, all finite.

    Returns
    -------
    numpy.ndarray
        Shape (N,), dtype float64, in statvolt/cm.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 1, or if the magnetic
        field magnitude vanishes at some supplied position.
    """
    return values  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_parallel_electric_field(points):
    """Reference implementation of parallel_electric_field (deterministic).

    Composes the step-01 oracle function directly, so this reference value
    never depends on a submitted implementation.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")

    B = np.asarray(_oracle_field_samples(p, "B"), dtype=np.float64)
    E = np.asarray(_oracle_field_samples(p, "E"), dtype=np.float64)
    mag = np.linalg.norm(B, axis=-1)
    if np.any(mag <= 0.0):
        raise ValueError("magnetic field magnitude must be positive")
    return np.sum(E * B, axis=-1) / mag

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for parallel_electric_field."""
    return [
        {'setup': 'import numpy as np\nsd = _oracle_seed_point(-0.8, 0.8)\nzs = np.linspace(-0.8, 0.8, 81)\npts = _oracle_quasi_x_line(zs, sd)\n', 'call': 'parallel_electric_field(pts)', 'gold_call': '_oracle_parallel_electric_field(pts)'},
        {'setup': 'import numpy as np\npts = np.array([[0.30, 0.55, -0.40], [-0.90, 1.60, 0.25],\n                [0.75, -0.30, 0.60], [-0.20, 2.40, -0.70]])\n', 'call': 'parallel_electric_field(pts)', 'gold_call': '_oracle_parallel_electric_field(pts)'},
        {'setup': 'import numpy as np\npts = np.array([[-0.4, 1.5, 0.3]])\n', 'call': 'parallel_electric_field(pts)', 'gold_call': '_oracle_parallel_electric_field(pts)'},
        {'setup': 'import numpy as np\npts = np.array([[8.0, -5.0, 6.0], [-12.0, 19.0, -7.0]])\n', 'call': 'parallel_electric_field(pts)', 'gold_call': '_oracle_parallel_electric_field(pts)'},
        {'setup': 'import numpy as np\nbad = np.array([0.0, 1.0, 0.0])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(parallel_electric_field, bad)', 'gold_call': '_status(_oracle_parallel_electric_field, bad)'},
        {'setup': "import numpy as np\nbad = np.array([[0.0, 1.0, float('inf')]])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n", 'call': '_status(parallel_electric_field, bad)', 'gold_call': '_status(_oracle_parallel_electric_field, bad)'},
    ]
