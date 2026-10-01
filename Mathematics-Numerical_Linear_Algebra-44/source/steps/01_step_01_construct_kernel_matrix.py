"""
The tall Gaussian RBF data matrix A.

Build A on [0, 1] with the instance grid convention (endpoint-inclusive

nodes, not a midpoint layout), default bandwidth equal to the centre

spacing when h is omitted (unit bandwidth when n equals one), and the

problem's overall scale. Other node layouts are not accepted.

Returns
-------
ndarray of shape (m, n), float64: the scaled Gaussian radial-basis matrix A.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 1: """

import numpy as np

def construct_kernel_matrix(m, n, h=None):
    """m, n: integers with m >= n >= 1, the row and column counts. h: positive
    bandwidth, or None to use the default centre spacing: 1/(n-1) when n > 1,
    and 1.0 when n == 1 (the n-1 denominator is undefined). Returns (m, n)
    float64: the scaled Gaussian radial-basis matrix for this configuration.
    Raises ValueError for m < n, non-positive sizes, or a non-positive
    bandwidth."""
    return np.zeros((m, n))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_construct_kernel_matrix(m, n, h=None):
    if isinstance(m, bool) or isinstance(n, bool):
        raise ValueError("m and n must be integers")
    if not isinstance(m, (int, np.integer)) or not isinstance(n, (int, np.integer)):
        raise ValueError("m and n must be integers")
    if m < 1 or n < 1:
        raise ValueError("m and n must be positive")
    if m < n:
        raise ValueError("require m >= n (tall matrix)")
    if h is None:
        h = 1.0 if n == 1 else 1.0 / (n - 1)
    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be a positive finite float")
    t = np.linspace(0.0, 1.0, int(m))
    c = np.linspace(0.0, 1.0, int(n))
    A = np.exp(-((t[:, None] - c[None, :]) ** 2) / (2.0 * h * h)) / np.sqrt(float(m))
    return np.asarray(A, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    return [
        {"setup": 'm, n = 128, 8', "call": 'construct_kernel_matrix(m, n)', "gold_call": '_oracle_construct_kernel_matrix(m, n)', "tol": 1e-12},
        {"setup": 'm, n, h = 4, 4, 0.5', "call": 'construct_kernel_matrix(m, n, h)', "gold_call": '_oracle_construct_kernel_matrix(m, n, h)', "tol": 1e-12},
        {"setup": 'm, n = 5, 1', "call": 'construct_kernel_matrix(m, n)', "gold_call": '_oracle_construct_kernel_matrix(m, n)', "tol": 1e-12},
        # Endpoint-inclusive fingerprint (midpoint grid gives ~0.08126)
        {"setup": 'A = construct_kernel_matrix(128, 8)', "call": 'float(A[0, 0])', "gold_call": '0.088388347648', "tol": 1e-12},
        {"setup": 'A = construct_kernel_matrix(128, 8)', "call": 'float(np.linalg.norm(A))', "gold_call": '1.314792801251', "tol": 1e-12},
        # Midpoint grid must disagree
        {"setup": 'import numpy as np\nA = construct_kernel_matrix(128, 8)\nt = (np.arange(128) + 0.5) / 128.0\nc = (np.arange(8) + 0.5) / 8.0\nh = 1.0 / 7.0\nA_mid = np.exp(-((t[:, None] - c[None, :]) ** 2) / (2.0 * h * h)) / np.sqrt(128.0)', "call": 'float(np.linalg.norm(A - A_mid) > 0.05)', "gold_call": '1.0', "tol": 0.0},
        {"setup": 'm, n, h = 64, 6, 0.3', "call": 'construct_kernel_matrix(m, n, h)', "gold_call": '_oracle_construct_kernel_matrix(m, n, h)', "tol": 1e-12},
        {"setup": 'm, n = 96, 5', "call": 'construct_kernel_matrix(m, n)', "gold_call": '_oracle_construct_kernel_matrix(m, n)', "tol": 1e-12},
    ]
