"""
Step 2: build the transition matrix.

The study keeps survival in its own matrix, separate from reproduction. In this model the only transition from one year to the next is ageing by one class. So the transition matrix is zero everywhere except on the first sub-diagonal. There, the entry moving a female from age a to age a+1 is the annual survival probability s_a. The oldest class is pooled, meaning all animals of that age and above. Its members can survive and remain in the same class, so the last diagonal entry is the terminal survival probability s_(n-1). Nothing in this matrix depends on when the population is counted.

Returns
-------
u : np.ndarray, shape (n, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transition_matrix(
    survival: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Build the age-structured transition matrix.

    Parameters
    ----------
    survival : sequence of float
        Annual survival probabilities in age order, one per class, as estimated by
        step 01. The last entry is the pooled terminal class's survival.

    Returns
    -------
    numpy.ndarray
        The ``(n, n)`` transition matrix: entry ``[a+1, a] = survival[a]`` for every
        ``a < n-1``, and ``[n-1, n-1] = survival[n-1]`` for the pooled terminal class.
        All other entries are zero.

    Raises
    ------
    ValueError
        If the input is not a finite 1-D sequence of at least two probabilities in
        [0, 1].
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_transition_matrix(
    survival: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reference implementation for transition_matrix."""
    import numpy as np

    s = np.asarray(survival, dtype=float)
    if s.ndim != 1 or s.size < 2:
        raise ValueError("survival must be a 1-D sequence of at least two classes")
    if not np.all(np.isfinite(s)):
        raise ValueError("survival probabilities must be finite")
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("survival probabilities must lie in [0, 1]")

    n = s.size
    u = np.zeros((n, n), dtype=float)
    for a in range(n - 1):
        u[a + 1, a] = s[a]
    u[n - 1, n - 1] = s[n - 1]
    return u

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        transition_matrix(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_transition_matrix(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: a four-class vector in the paper's simple-table style.
        {"setup": "import numpy as np\ns = np.array([0.5, 0.6, 0.7, 0.0])\n",
         "call": "transition_matrix(s)",
         "gold_call": "_oracle_transition_matrix(s)"},
        # Normal: a finite terminal survival exercises the pooled self-loop.
        {"setup": "import numpy as np\ns = np.array([0.699, 0.774, 0.947, 0.947, 0.634])\n",
         "call": "transition_matrix(s)",
         "gold_call": "_oracle_transition_matrix(s)"},
        # Boundary: the smallest possible age structure.
        {"setup": "import numpy as np\ns = np.array([1.0, 0.5])\n",
         "call": "transition_matrix(s)",
         "gold_call": "_oracle_transition_matrix(s)"},
        # Invalid: a probability above one.
        {"setup": "import numpy as np\ns = np.array([0.5, 1.4])\n" + invalid,
         "call": "run_model(survival=s)",
         "gold_call": "run_gold(survival=s)"},
        # Invalid: a negative probability.
        {"setup": "import numpy as np\ns = np.array([0.5, -0.1])\n" + invalid,
         "call": "run_model(survival=s)",
         "gold_call": "run_gold(survival=s)"},
        # Invalid: a two-dimensional input.
        {"setup": "import numpy as np\ns = np.array([[0.5, 0.6], [0.7, 0.8]])\n" + invalid,
         "call": "run_model(survival=s)",
         "gold_call": "run_gold(survival=s)"},
        # Invalid: a single class gives no age structure.
        {"setup": "import numpy as np\ns = np.array([0.5])\n" + invalid,
         "call": "run_model(survival=s)",
         "gold_call": "run_gold(survival=s)"},
    ]
