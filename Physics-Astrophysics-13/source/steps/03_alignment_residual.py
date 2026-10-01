"""
Step 03: alignment residual and in-plane discriminant.

Convention (pinned)



For each supplied position return four numbers, in this order:



  columns 0:3  the cross product of the magnetic field with the product of the

               step-02 gradient tensor and that same magnetic field, i.e. the

               components of cross(B, G @ B);

  column  3    the negated product of the largest and the smallest of the three

               real parts of the eigenvalues of G: take the real parts, sort

               them by DESCENDING value, then negate the product of the first

               and the last. The recipe governs; do not substitute a sign rule

               for it, and do not discard rows whose eigenvalues are complex.



Both the magnetic field and the gradient tensor come from the step-01 and

step-02 public functions.



Inputs



points : array_like

    Shape (N, 3), N >= 1, all entries finite.



Returns



numpy.ndarray

    Shape (N, 4), dtype float64.

Returns
-------
A numpy.ndarray of shape (N, 4) and dtype float64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def alignment_residual(points):
    """Return the (N, 4) residual-and-discriminant array.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 1, all finite.

    Returns
    -------
    numpy.ndarray
        Shape (N, 4), dtype float64: three residual components followed by the
        discriminant.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 1.
    """
    return out  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_alignment_residual(points):
    """Reference implementation of alignment_residual (deterministic).

    Composes the step-01 and step-02 oracle functions directly, so this
    reference value never depends on a submitted implementation.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")

    B = np.asarray(_oracle_field_samples(p, "B"), dtype=np.float64)
    G = np.asarray(_oracle_gradient_tensor(p), dtype=np.float64)

    GB = np.einsum("nij,nj->ni", G, B)
    residual = np.cross(B, GB)

    out = np.empty((p.shape[0], 4), dtype=np.float64)
    out[:, 0:3] = residual
    for n in range(p.shape[0]):
        ev = np.real(np.linalg.eigvals(G[n]))
        ev = np.sort(ev)[::-1]
        out[n, 3] = -(ev[0] * ev[-1])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for alignment_residual."""
    return [
        {'setup': 'import numpy as np\npts = np.array([[0.30, 0.55, -0.40], [-0.90, 1.60, 0.25],\n                [0.75, -0.30, 0.60], [-0.20, 2.40, -0.70]])\n', 'call': 'alignment_residual(pts)', 'gold_call': '_oracle_alignment_residual(pts)'},
        {'setup': 'import numpy as np\npts = np.array([[0.10, 1.50, 0.10], [0.10, 2.50, 0.10],\n                [-0.60, 3.10, -0.30], [1.20, 0.40, 0.80]])\n', 'call': 'alignment_residual(pts)', 'gold_call': '_oracle_alignment_residual(pts)'},
        {'setup': 'import numpy as np\npts = np.array([[0.4, 2.08, -0.15]])\n', 'call': 'alignment_residual(pts)', 'gold_call': '_oracle_alignment_residual(pts)'},
        {'setup': 'import numpy as np\npts = np.array([[9.0, -6.0, 7.0], [-14.0, 22.0, -8.0]])\n', 'call': 'alignment_residual(pts)', 'gold_call': '_oracle_alignment_residual(pts)'},
        {'setup': 'import numpy as np\nbad = np.zeros((0, 3))\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(alignment_residual, bad)', 'gold_call': '_status(_oracle_alignment_residual, bad)'},
        {'setup': 'import numpy as np\nbad = np.zeros((3, 4))\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_status(alignment_residual, bad)', 'gold_call': '_status(_oracle_alignment_residual, bad)'},
    ]
