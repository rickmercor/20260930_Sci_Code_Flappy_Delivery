"""
Step 7: the growth rate and the stable age distribution.

A projection matrix updates the age distribution by one year. For a non-negative primitive matrix, repeated projection converges to a fixed composition regardless of the starting vector.

The dominant eigenvalue is the asymptotic growth rate lambda. Its right eigenvector, normalized to sum to one, is the stable age distribution. This is the long-run proportion of females in each class at the census.

The dominant eigenvalue follows from the characteristic equation and is computed here by eigendecomposition. The stable distribution is the normalized eigenvector, taken positive as Perron-Frobenius guarantees for this class of matrices.

Returns
-------
np.concatenate([[growth], shares]) : np.ndarray, shape (n+1,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dominant_eigenpair(
    matrix: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Dominant eigenvalue and stable age distribution of a projection matrix.

    Parameters
    ----------
    matrix : array-like of shape (n, n)
        A projection matrix over the age classes (see step 06) with
        non-negative entries.

    Returns
    -------
    numpy.ndarray
        One array of length ``n + 1``: the dominant eigenvalue first, then the
        corresponding right eigenvector normalized to sum to one, holding the stable
        proportion of each class.

    Raises
    ------
    ValueError
        If the input is not a finite, non-negative square matrix with at least two
        classes, if the dominant eigenvalue is not real and positive, or if the
        stable vector cannot be normalized to a non-negative distribution.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_dominant_eigenpair(
    matrix: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reference implementation for dominant_eigenpair."""
    import numpy as np

    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    if m.shape[0] < 2:
        raise ValueError("matrix must hold at least two classes")
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix entries must be finite")
    if np.any(m < 0.0):
        raise ValueError("projection matrix entries must be non-negative")

    values, vectors = np.linalg.eig(m)
    k = int(np.argmax(values.real))
    growth = values[k]
    if abs(growth.imag) > 1e-8 * max(1.0, abs(growth.real)) or growth.real <= 0.0:
        raise ValueError("the dominant eigenvalue must be real and positive")
    growth = float(growth.real)

    shares = np.asarray(vectors[:, k].real, dtype=float)
    total = shares.sum()
    if abs(total) < 1e-12:
        raise ValueError("the stable vector cannot be normalized")
    shares = shares / total
    if shares.sum() < 0:
        shares = -shares
    if np.any(shares < -1e-9):
        raise ValueError("the stable vector must be non-negative")
    shares = np.clip(shares, 0.0, None)
    shares = shares / shares.sum()
    return np.concatenate([[growth], shares])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        dominant_eigenpair(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_dominant_eigenpair(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: the paper's three-class matrix.
        {"setup": ("import numpy as np\n"
                   "L = np.array([[0.75, 1.25, 1.25],\n"
                   "              [0.60, 0.00, 0.00],\n"
                   "              [0.00, 0.70, 0.00]])\n"),
         "call": "dominant_eigenpair(L)",
         "gold_call": "_oracle_dominant_eigenpair(L)"},
        # Normal: a four-class matrix with a pooled terminal self-loop.
        {"setup": ("import numpy as np\n"
                   "L = np.array([[0.00, 0.65, 0.85, 0.81],\n"
                   "              [0.70, 0.00, 0.00, 0.00],\n"
                   "              [0.00, 0.77, 0.00, 0.00],\n"
                   "              [0.00, 0.00, 0.95, 0.63]])\n"),
         "call": "dominant_eigenpair(L)",
         "gold_call": "_oracle_dominant_eigenpair(L)"},
        # Boundary: a primitive Leslie matrix with a unit growth rate
        # (0.5 + 0.5 = 1), whose stable distribution is uniform.
        {"setup": ("import numpy as np\n"
                   "L = np.array([[0.0, 0.5, 0.5],\n"
                   "              [1.0, 0.0, 0.0],\n"
                   "              [0.0, 1.0, 0.0]])\n"),
         "call": "dominant_eigenpair(L)",
         "gold_call": "_oracle_dominant_eigenpair(L)"},
        # Invalid: a negative entry.
        {"setup": ("import numpy as np\n"
                   "L = np.array([[0.5, -1.0], [0.5, 0.0]])\n" + invalid),
         "call": "run_model(matrix=L)",
         "gold_call": "run_gold(matrix=L)"},
        # Invalid: a non-finite entry.
        {"setup": ("import numpy as np\n"
                   "L = np.array([[0.5, float('nan')], [0.5, 0.0]])\n" + invalid),
         "call": "run_model(matrix=L)",
         "gold_call": "run_gold(matrix=L)"},
        # Invalid: a non-square input.
        {"setup": ("import numpy as np\n"
                   "L = np.zeros((2, 3))\n" + invalid),
         "call": "run_model(matrix=L)",
         "gold_call": "run_gold(matrix=L)"},
    ]
