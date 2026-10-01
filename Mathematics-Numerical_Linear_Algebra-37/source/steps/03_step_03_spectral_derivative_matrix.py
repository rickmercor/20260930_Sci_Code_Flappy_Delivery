"""
Build the differentiation matrix that acts on values sampled at n_points equispaced nodes covering one period, with the left endpoint included and the right endpoint excluded, so that the matrix returned differentiates the periodic interpolant of a real-valued function exactly. Raise ValueError if n_points is not an integer, if it is smaller than two, or if the period is not a finite positive real number.

Both terms of the transport operator require a derivative, one in position and one in wave vector, and the accuracy of those derivatives sets the spatial accuracy of the whole scheme. On a periodic domain the natural choice is to differentiate the trigonometric interpolant rather than a local polynomial one, which converges faster than any fixed algebraic order for smooth data and, applied to a function band-limited by the grid, is exact rather than merely accurate. The construction is most cleanly expressed in the transform domain, where differentiation becomes multiplication by the frequency variable, so the operator is obtained by transforming, scaling mode by mode, and transforming back. Because the low-rank factors are stored as columns sampled on the grid, it is convenient to have the operator available as an explicit matrix acting on those columns.



Two structural properties follow and are worth checking. The operator is skew-symmetric, which is what makes the resulting factor system hyperbolic and keeps the discrete dynamics from spuriously growing or damping. Its diagonal vanishes, reflecting that the interpolant's derivative at a node takes no contribution from the value at that node. An even node count needs care: the highest representable frequency is shared between positive and negative wavenumbers, and its contribution to the derivative of a real function cancels rather than being assigned to either. Neglecting that cancellation leaves a spurious imaginary part or, equivalently, an operator that is no longer skew-symmetric.

Returns
-------
np.ndarray of shape (n_points, n_points), the periodic spectral differentiation matrix, dtype float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_derivative_matrix(n_points: int, period: float) -> "np.ndarray":
    '''Differentiation matrix for a periodic equispaced grid.

    Parameters
    ----------
    n_points : int
        Number of equispaced nodes over one period, at least 2.
    period : float
        Length of the periodic interval, finite and positive.

    Returns
    -------
    D : np.ndarray
        (n_points, n_points) differentiation matrix, float64.
    '''
    return D  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_spectral_derivative_matrix(n_points: int, period: float) -> "np.ndarray":
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("n_points must be an integer")
    n = int(n_points)
    if n < 2:
        raise ValueError("n_points must be at least 2")
    if not isinstance(period, (int, float, np.floating, np.integer)):
        raise ValueError("period must be a real number")
    p = float(period)
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError("period must be finite and positive")
    w = 2.0 * np.pi * np.fft.fftfreq(n, d=p / n)
    eye = np.eye(n, dtype=float)
    D = np.real(np.fft.ifft(1j * w[:, None] * np.fft.fft(eye, axis=0), axis=0))
    return D.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the production grid ---
        {
            "setup": """import numpy as np
n_points = 128
period = 20.0
""",
            "call": "spectral_derivative_matrix(n_points, period)",
            "gold_call": "_oracle_spectral_derivative_matrix(n_points, period)",
        },
        # --- edge: odd node count, no shared highest frequency ---
        {
            "setup": """import numpy as np
n_points = 7
period = 2.0*np.pi
""",
            "call": "spectral_derivative_matrix(n_points, period)",
            "gold_call": "_oracle_spectral_derivative_matrix(n_points, period)",
        },
        # --- edge: short period, entries two orders larger than production ---
        {
            "setup": """import numpy as np
n_points = 64
period = 1.0
""",
            "call": "spectral_derivative_matrix(n_points, period)",
            "gold_call": "_oracle_spectral_derivative_matrix(n_points, period)",
        },
        # --- boundary: minimal even grid, resolving only the shared
        #     highest frequency, whose contribution must cancel entirely ---
        {
            "setup": """import numpy as np
n_points = 2
period = 20.0
""",
            "call": "spectral_derivative_matrix(n_points, period)",
            "gold_call": "_oracle_spectral_derivative_matrix(n_points, period)",
        },
        # --- boundary: minimal odd grid ---
        {
            "setup": """import numpy as np
n_points = 3
period = 0.5
""",
            "call": "spectral_derivative_matrix(n_points, period)",
            "gold_call": "_oracle_spectral_derivative_matrix(n_points, period)",
        },
        # --- edge: large grid, accumulated transform roundoff ---
        {
            "setup": """import numpy as np
n_points = 512
period = 3.0
""",
            "call": "spectral_derivative_matrix(n_points, period)",
            "gold_call": "_oracle_spectral_derivative_matrix(n_points, period)",
        },
        # --- structural probe: skew-symmetry, vanishing diagonal, and the
        #     count of modes differentiated exactly, as integers ---
        {
            "setup": """import numpy as np
def probe(fn):
    out = []
    for n, p in [(128, 20.0), (7, 2.0*np.pi), (2, 20.0), (64, 1.0)]:
        D = np.asarray(fn(n, p), dtype=float)
        sc = max(np.abs(D).max(), 2.0*np.pi/p)
        out.append(int(np.abs(D + D.T).max()/sc < 1e-12))
        out.append(int(np.abs(np.diag(D)).max()/sc < 1e-12))
        g = -p/2.0 + p/n*np.arange(n)
        cnt = 0
        for m in range(1, max(n//2, 1)):
            f = np.sin(2.0*np.pi*m*g/p)
            ex = (2.0*np.pi*m/p)*np.cos(2.0*np.pi*m*g/p)
            if np.abs(D @ f - ex).max() < 1e-10*max(1.0, abs(2.0*np.pi*m/p)):
                cnt += 1
        out.append(cnt)
    return out
def run_model():
    return probe(spectral_derivative_matrix)
def run_gold():
    return probe(_oracle_spectral_derivative_matrix)
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: node count below the minimum ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        spectral_derivative_matrix(1, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_spectral_derivative_matrix(1, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-integer node count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        spectral_derivative_matrix(8.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_spectral_derivative_matrix(8.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive period ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        spectral_derivative_matrix(8, -3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_spectral_derivative_matrix(8, -3.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
