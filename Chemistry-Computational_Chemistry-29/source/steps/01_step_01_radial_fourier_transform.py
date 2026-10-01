"""
Discrete three-dimensional Fourier transform of radially symmetric functions on the uniform grid that every later step uses, in the forward and the inverse direction.

For a function of the distance r alone, the three-dimensional Fourier transform reduces to a sine transform, F(q) = (4 pi / q) times the integral over r of r f(r) sin(q r), and the inverse is f(r) = (1 / (2 pi^2 r)) times the integral over q of q F(q) sin(q r). Every radial function in this task is sampled on r_i = i dr for i = 1, ..., N - 1, and its transform on q_j = j dq with dq = pi / (N dr), where N - 1 is the number of samples.

Both integrals are replaced by rectangle-rule sums over these points: F(q_j) = (4 pi dr / q_j) sum over i of r_i f(r_i) sin(q_j r_i), and f(r_i) = (dq / (2 pi^2 r_i)) sum over j of q_j F(q_j) sin(q_j r_i). With this pairing the two sums are exact inverses, so a forward transform followed by the inverse gives back the input to rounding error.

The transform acts along the last axis of the input, so a stack of radial functions, for example one per species pair of a mixture, is transformed in a single call.

Returns
-------
numpy.ndarray with the shape of f: the forward (inverse=False) or inverse (inverse=True) rectangle-rule radial Fourier transform along the last axis
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def radial_fourier_transform(f: np.ndarray, dr: float, inverse: bool) -> np.ndarray:
    '''Rectangle-rule radial Fourier transform along the last axis.

    Parameters
    ----------
    f : numpy.ndarray
        Samples along the last axis, at r_i = i * dr (forward) or at
        q_j = j * pi / (N * dr) (inverse), where N - 1 is the length of that
        axis. Any leading axes are carried through unchanged.
    dr : float
        Real-space grid spacing, the same in both directions.
    inverse : bool
        False for the forward transform r -> q, True for the inverse q -> r.

    Returns
    -------
    transformed : numpy.ndarray
        Array of the same shape as f holding the transform on the conjugate
        grid.

    Raises
    ------
    ValueError
        If f is not finite or has fewer than two samples on its last axis, if
        dr is not positive and finite, or if inverse is not a bool.
    '''
    return transformed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _sine_transform_pair(f, dr, inverse):
    """Rectangle-rule radial Fourier pair along the last axis (grid r_i = i dr, q_j = j pi/(N dr))."""
    import numpy as np
    from scipy.fft import dst
    f = np.asarray(f, dtype=float)
    n = f.shape[-1] + 1
    k = np.arange(1, n)
    r = dr * k
    dq = np.pi / (n * dr)
    q = dq * k
    if not inverse:
        return (2.0 * np.pi * dr / q) * dst(r * f, type=1, axis=-1)
    return (dq / (4.0 * np.pi ** 2 * r)) * dst(q * f, type=1, axis=-1)


def _oracle_radial_fourier_transform(f: np.ndarray, dr: float, inverse: bool) -> np.ndarray:
    import numpy as np
    f = np.asarray(f, dtype=float)
    if f.ndim < 1 or f.shape[-1] < 2 or not np.all(np.isfinite(f)):
        raise ValueError("f must be a finite array with at least two radial samples on its last axis")
    dr = float(dr)
    if not np.isfinite(dr) or dr <= 0.0:
        raise ValueError("dr must be a positive finite number")
    if not isinstance(inverse, (bool, np.bool_)):
        raise ValueError("inverse must be a bool")
    return _sine_transform_pair(f, dr, bool(inverse))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: forward transform of a Gaussian on a 512-interval grid ---
        {
            "setup": """import numpy as np
n = 512
dr = 0.02
r = dr * np.arange(1, n)
f = np.exp(-r ** 2)
""",
            "call": "radial_fourier_transform(f.copy(), dr, False)",
            "gold_call": "_oracle_radial_fourier_transform(f.copy(), dr, False)",
            "tol": 1e-10,
        },
        # --- Boundary: inverse transform of a stack of two functions sampled on the conjugate grid ---
        {
            "setup": """import numpy as np
n = 256
dr = 0.05
q = np.pi * np.arange(1, n) / (n * dr)
F = np.stack([np.exp(-q ** 2 / 4.0), 1.0 / (1.0 + q ** 2)])
""",
            "call": "radial_fourier_transform(F.copy(), dr, True)",
            "gold_call": "_oracle_radial_fourier_transform(F.copy(), dr, True)",
            "tol": 1e-10,
        },
        # --- Edge: the shortest admissible input, two samples ---
        {
            "setup": """import numpy as np
f = np.array([1.0, -0.5])
dr = 0.3
""",
            "call": "radial_fourier_transform(f.copy(), dr, False)",
            "gold_call": "_oracle_radial_fourier_transform(f.copy(), dr, False)",
            "tol": 1e-10,
        },
        # --- Normal: forward transform of a (2, 2) stack of soft-core profiles ---
        {
            "setup": """import numpy as np
n = 128
dr = 0.05
r = dr * np.arange(1, n)
w = np.where(r < 1.0, (1.0 - r) ** 2, 0.0)
f = np.array([[w, 0.5 * w], [0.5 * w, np.exp(-r) * w]])
""",
            "call": "radial_fourier_transform(f.copy(), dr, False)",
            "gold_call": "_oracle_radial_fourier_transform(f.copy(), dr, False)",
            "tol": 1e-10,
        },
        # --- Invalid: non-positive grid spacing ---
        {
            "setup": """import numpy as np
f = np.array([1.0, 2.0, 3.0])
dr = 0.0
def run_model():
    try:
        radial_fourier_transform(f.copy(), dr, False)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_radial_fourier_transform(f.copy(), dr, False)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
