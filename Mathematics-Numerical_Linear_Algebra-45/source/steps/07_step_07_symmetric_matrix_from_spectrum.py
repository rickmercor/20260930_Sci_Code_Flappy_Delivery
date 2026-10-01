"""
Turn a list of eigenvalues into a dense symmetric matrix by conjugating it with a fixed orthogonal transform.

Any dense symmetric matrix with a prescribed spectrum is obtained by conjugating the diagonal of that spectrum with an orthogonal matrix, and a reproducible choice is the sine transform with entries sqrt(2/(n+1))*sin(pi*j*k/(n+1)) for j, k running from 1 to n. That transform is its own inverse, so the conjugation costs nothing beyond two products and returns a matrix whose eigenvalues are exactly the supplied list.

Returns
-------
np.ndarray, float, shape (n, n): the symmetric matrix with the prescribed spectrum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def symmetric_matrix_from_spectrum(eigenvalues: np.ndarray) -> np.ndarray:
    """Build a dense symmetric matrix with the prescribed spectrum.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Shape (n,) array of real eigenvalues, with n >= 1.

    Returns
    -------
    matrix : np.ndarray
        Shape (n, n) symmetric float array whose eigenvalues are the entries
        of ``eigenvalues``.

    Raises
    ------
    ValueError
        If ``eigenvalues`` is not a non-empty one-dimensional array, or if
        any of its entries is not finite. The function must raise rather
        than return a placeholder or flatten a two-dimensional input.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros((len(eigenvalues), len(eigenvalues)), dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_symmetric_matrix_from_spectrum(eigenvalues: np.ndarray) -> np.ndarray:
    import numpy as np

    ev = np.asarray(eigenvalues, dtype=float)
    if ev.ndim != 1 or ev.shape[0] < 1:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(ev)):
        raise ValueError("eigenvalues must be finite")

    n = ev.shape[0]
    idx = np.arange(1, n + 1, dtype=float)
    q = np.sqrt(2.0 / (n + 1.0)) * np.sin(np.pi * np.outer(idx, idx) / (n + 1.0))
    matrix = q @ (ev[:, None] * q)
    return 0.5 * (matrix + matrix.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values obtained independently of the oracle: conjugation by
        # an orthogonal matrix preserves the trace and the sum of squares, so
        # both must equal the corresponding sums over the eigenvalues, while the
        # symmetry of the result is checked at the same time.
        {
            "setup": """import numpy as np
ev = np.array([0.03, 0.17, 0.41, 0.62, 0.88, 0.95, 1.0])
EXPECTED = float(np.sum(ev) + 100.0 * np.sum(ev ** 2))
""",
            "call": ("float(np.trace(symmetric_matrix_from_spectrum(ev))"
                     " + 100.0 * np.sum(symmetric_matrix_from_spectrum(ev) ** 2)"
                     " + 1e6 * np.max(np.abs(symmetric_matrix_from_spectrum(ev)"
                     " - symmetric_matrix_from_spectrum(ev).T)))"),
            "gold_call": "EXPECTED",
        },
        # --- Pinned: the recovered eigenvalues must reproduce the input list,
        # which no oracle is needed to state.
        {
            "setup": """import numpy as np
ev = np.array([0.05, 0.62, 0.18, 0.95, 0.37, 0.71, 0.09, 0.44, 0.88, 0.26,
               0.53, 0.80])
w = np.array([(-1.0) ** j * (j + 1) ** 2 for j in range(12)])
EXPECTED = float(np.dot(w, np.sort(ev)))
""",
            "call": ("float(np.dot(w, np.sort(np.linalg.eigvalsh("
                     "symmetric_matrix_from_spectrum(ev)))))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a spectrum split into two blocks around a narrow gap
        # (normal scenario) ---
        {
            "setup": """import numpy as np
ev = np.concatenate([np.linspace(0.0, 0.3498, 18), np.linspace(0.3502, 1.0, 32)])
w = np.cos(0.23 * np.arange(50))
""",
            "call": "float(w @ symmetric_matrix_from_spectrum(ev) @ w)",
            "gold_call": "float(w @ _oracle_symmetric_matrix_from_spectrum(ev) @ w)",
        },
        # --- Valid: an odd size with mixed signs ---
        {
            "setup": """import numpy as np
ev = np.array([-0.9, -0.35, 0.0, 0.22, 0.77, 1.4, 2.1, -1.05, 0.5])
w = np.arange(1.0, 10.0)
""",
            "call": "float(w @ symmetric_matrix_from_spectrum(ev) @ w)",
            "gold_call": "float(w @ _oracle_symmetric_matrix_from_spectrum(ev) @ w)",
        },
        # --- Boundary: a single eigenvalue, where the transform is the scalar
        # one and the matrix is that eigenvalue itself ---
        {
            "setup": """import numpy as np
ev = np.array([0.625])
""",
            "call": "float(symmetric_matrix_from_spectrum(ev)[0, 0])",
            "gold_call": "float(_oracle_symmetric_matrix_from_spectrum(ev)[0, 0])",
        },
        # --- Edge: a repeated eigenvalue list makes the matrix a multiple of the
        # identity, so every off-diagonal entry has to cancel ---
        {
            "setup": """import numpy as np
ev = np.full(11, 0.37)
""",
            "call": ("float(np.trace(symmetric_matrix_from_spectrum(ev))"
                     " + 1e6 * np.sum(np.abs(symmetric_matrix_from_spectrum(ev)"
                     " - 0.37 * np.eye(11))))"),
            "gold_call": ("float(np.trace(_oracle_symmetric_matrix_from_spectrum(ev))"
                          " + 1e6 * np.sum(np.abs(_oracle_symmetric_matrix_from_spectrum(ev)"
                          " - 0.37 * np.eye(11))))"),
        },
        # --- Invalid: a two-dimensional eigenvalue argument ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        symmetric_matrix_from_spectrum(np.ones((3, 3)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_symmetric_matrix_from_spectrum(np.ones((3, 3)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-finite eigenvalue ---
        {
            "setup": """import numpy as np
bad = np.array([0.1, np.nan, 0.9])
def run_model():
    try:
        symmetric_matrix_from_spectrum(bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_symmetric_matrix_from_spectrum(bad)
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
