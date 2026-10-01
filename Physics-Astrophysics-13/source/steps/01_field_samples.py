"""
Step 01: evaluate the three prescribed fields at supplied positions.

Convention (pinned)



Positions are rows (x, y, z) of an (N, 3) array. The selector which picks the

returned quantity and fixes its shape and units:



  "B"   -> (N, 3), Gauss.

  "E"   -> (N, 3), statvolt/cm.

  "rho" -> (N,), the mass density DIVIDED BY the reference density

           1.0e-14 g/cm^3, so the returned numbers are of order unity. Every

           later step that needs a physical density multiplies by 1.0e-14

           itself.



The three closed forms are those of the problem setup and are reproduced here

verbatim so that this step fixes them once for the whole chain:



  B   = ( (y-2)^2 - 1.0 + z^2 + 0.4 x ,  -x - 0.4 y + 0.2 z ,  0.5 )

  E   = 7.5e-6 ( 0 , 2 z , 3 - 2 y )

  rho / 1.0e-14 = 1.0 + 0.6 ((y-2)^2 + z^2)



Inputs



points : array_like

    Shape (N, 3), N >= 1, all entries finite.

which : str

    One of "B", "E", "rho". Case sensitive.



Returns



numpy.ndarray

    (N, 3) float64 for "B" and "E"; (N,) float64 for "rho".

Returns
-------
A numpy.ndarray of shape (N, 3) and dtype float64 for 'B' and for 'E'; a numpy.ndarray of shape (N,) and dtype float64 for 'rho', in units of 1.0e-14 g/cm^3.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def field_samples(points, which):
    """Return the requested field evaluated at every supplied position.

    Parameters
    ----------
    points : array_like
        Shape (N, 3), N >= 1, all finite.
    which : str
        "B", "E" or "rho".

    Returns
    -------
    numpy.ndarray
        (N, 3) for "B" and "E"; (N,) for "rho", in units of 1.0e-14 g/cm^3.

    Raises
    ------
    ValueError
        If points is not a finite (N, 3) array with N >= 1, or if which is not
        one of the three accepted selectors.
    """
    return values  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

MU = 1.0
B0 = 0.5
AL = 0.4
CC = 0.2
E0 = 7.5e-6
AA = 0.6
RHO_SCALE = 1.0e-14


def _oracle_field_samples(points, which):
    """Reference implementation of field_samples (deterministic)."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")
    if which not in ("B", "E", "rho"):
        raise ValueError("which must be one of 'B', 'E', 'rho'")

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    if which == "B":
        return np.stack([(y - 2.0) ** 2 - MU + z ** 2 + AL * x,
                         -x - AL * y + CC * z,
                         np.full_like(x, B0)], axis=-1)
    if which == "E":
        return E0 * np.stack([np.zeros_like(x), 2.0 * z, 3.0 - 2.0 * y],
                             axis=-1)
    return 1.0 + AA * ((y - 2.0) ** 2 + z ** 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test cases for field_samples."""
    return [
        {'setup': 'import numpy as np\npts = np.array([[-0.35, 0.9, -0.04], [0.12, 1.4, 0.5],\n                [-0.8, 0.7, -0.8], [0.45, 0.68, 0.8]])\n', 'call': "field_samples(pts, 'B')", 'gold_call': "_oracle_field_samples(pts, 'B')"},
        {'setup': 'import numpy as np\npts = np.array([[-0.35, 0.9, -0.04], [0.12, 1.4, 0.5],\n                [-0.8, 0.7, -0.8], [0.45, 0.68, 0.8]])\n', 'call': "field_samples(pts, 'E')", 'gold_call': "_oracle_field_samples(pts, 'E')"},
        {'setup': 'import numpy as np\npts = np.array([[-0.35, 0.9, -0.04], [0.12, 1.4, 0.5],\n                [-0.8, 0.7, -0.8], [0.45, 0.68, 0.8]])\n', 'call': "field_samples(pts, 'rho')", 'gold_call': "_oracle_field_samples(pts, 'rho')"},
        {'setup': 'import numpy as np\npts = np.array([[0.0, 2.0, 0.0]])\n', 'call': "field_samples(pts, 'rho')", 'gold_call': "_oracle_field_samples(pts, 'rho')"},
        {'setup': 'import numpy as np\npts = np.array([[12.0, -7.5, 9.0], [-30.0, 40.0, -25.0]])\n', 'call': "field_samples(pts, 'B')", 'gold_call': "_oracle_field_samples(pts, 'B')"},
        {'setup': 'import numpy as np\npts = np.array([[0.0, 1.0, 0.0]])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': "_status(field_samples, pts, 'density')", 'gold_call': "_status(_oracle_field_samples, pts, 'density')"},
        {'setup': 'import numpy as np\nbad = np.array([0.0, 1.0, 0.0])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': "_status(field_samples, bad, 'B')", 'gold_call': "_status(_oracle_field_samples, bad, 'B')"},
    ]
