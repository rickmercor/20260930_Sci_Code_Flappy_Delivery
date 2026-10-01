"""
Step 8: reproductive values: the left eigenvector of the projection matrix.

The stable age distribution says how the population is composed. The reproductive values say what each class is worth to the population's future. The reproductive value vector is the dominant left eigenvector of the projection matrix. It assigns class i its expected relative contribution to all future offspring.

This step takes the growth rate and the stable distribution from step 07 and finds the left eigenvector that belongs to that growth rate. It is the counterpart of the stable distribution, which is the dominant right eigenvector. Different census timings share their dynamics but not their reproductive values.

Like any eigenvector, it is defined only up to scale, so a normalization must be fixed. The convention used here anchors the vector to the youngest class. Every class value is expressed relative to that class, which is set to 1.

Returns
-------
rv : np.ndarray, shape (n,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reproductive_values(
    matrix: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reproductive values of a projection matrix, anchored to the youngest class.

    Parameters
    ----------
    matrix : array-like of shape (n, n)
        A projection matrix over the age classes (see step 06), with non-negative
        entries.

    Returns
    -------
    numpy.ndarray
        The dominant left eigenvector, normalized so that its first entry (the
        youngest class) is 1: ``rv[0] = 1`` and ``rv[i]`` is the reproductive value
        of class ``i`` relative to the youngest class.

    Raises
    ------
    ValueError
        If the input is not a finite, non-negative square matrix with at least two
        classes, if the dominant eigenvalue is not real and positive, or if the
        vector cannot be anchored because its first entry vanishes.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reproductive_values(
    matrix: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reference implementation for reproductive_values."""
    import numpy as np

    m = np.asarray(matrix, dtype=float)
    packed = _oracle_dominant_eigenpair(m)
    growth, stable = float(packed[0]), packed[1:]

    values_t, vectors_t = np.linalg.eig(m.T)
    k = int(np.argmin(np.abs(values_t - growth)))
    left = np.asarray(vectors_t[:, k].real, dtype=float)
    left = left / (left @ stable)
    if abs(left[0]) < 1e-12:
        raise ValueError("the reproductive value vector cannot be anchored")
    rv = left / left[0]
    if np.any(rv < -1e-9):
        raise ValueError("the reproductive values must be non-negative")
    rv = np.clip(rv, 0.0, None)
    return rv

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        reproductive_values(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_reproductive_values(**kw)\n"
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
         "call": "reproductive_values(L)",
         "gold_call": "_oracle_reproductive_values(L)"},
        # Normal: a four-class matrix with a pooled terminal self-loop.
        {"setup": ("import numpy as np\n"
                   "L = np.array([[0.00, 0.65, 0.85, 0.81],\n"
                   "              [0.70, 0.00, 0.00, 0.00],\n"
                   "              [0.00, 0.77, 0.00, 0.00],\n"
                   "              [0.00, 0.00, 0.95, 0.63]])\n"),
         "call": "reproductive_values(L)",
         "gold_call": "_oracle_reproductive_values(L)"},
        # Boundary: the smallest reducible matrix.
        {"setup": ("import numpy as np\n"
                   "L = np.array([[0.20, 0.80], [0.50, 0.10]])\n"),
         "call": "reproductive_values(L)",
         "gold_call": "_oracle_reproductive_values(L)"},
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
        # Invalid: a single class cannot be anchored.
        {"setup": ("import numpy as np\n"
                   "L = np.array([[0.5]])\n" + invalid),
         "call": "run_model(matrix=L)",
         "gold_call": "run_gold(matrix=L)"},
    ]
