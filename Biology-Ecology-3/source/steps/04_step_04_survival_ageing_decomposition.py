"""
Step 4: decompose the annual transition into survival and ageing.

The study separates the annual transition of this model into two operators. 

The first is a diagonal matrix S holding the annual survival probability of each class. It scales every class in place. The second is the ageing operator T, which moves each class to the next one over the year. In an age-classified model T is the identity shifted one place down, a sub-diagonal of ones.

The oldest class is pooled, so its members can remain in it. The last diagonal entry of T is 1. That is what lets the product ``T S`` reproduce the usual transition matrix, whose terminal entry carries the pooled annual survival. The decomposition matters because it lets survival be split around the birth pulse, which is how the study handles censuses taken at other times of the year. This step takes the transition matrix built in step 02 and returns both operators.

Returns
-------
survival_operator, ageing : tuple of np.ndarray, each shape (n, n)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def survival_ageing_decomposition(
    transition: "numpy.typing.ArrayLike",
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Split the annual transition matrix into survival and ageing operators.

    Parameters
    ----------
    transition : array-like of shape (n, n)
        The age-structured transition matrix of step 02: the annual survival
        probability of class a at ``transition[a+1, a]`` for every ``a < n-1``, the
        pooled terminal class's survival at ``transition[n-1, n-1]``, and zeros
        everywhere else.

    Returns
    -------
    tuple of numpy.ndarray
        ``(S, T)``: the diagonal survival operator, with ``S[a, a] = transition[a+1, a]``
        for every ``a < n-1`` and ``S[n-1, n-1] = transition[n-1, n-1]``; and the ageing
        operator ``T``, with ``T[a+1, a] = 1`` for every ``a < n-1`` and
        ``T[n-1, n-1] = 1`` so that the pooled terminal class keeps its position. The
        product ``T S`` equals the input matrix.

    Raises
    ------
    ValueError
        If the input is not a square matrix with at least two classes, if an entry is
        not finite or lies outside [0, 1], or if any entry other than the first
        sub-diagonal and the last diagonal entry is non-zero.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_survival_ageing_decomposition(
    transition: "numpy.typing.ArrayLike",
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Reference implementation for survival_ageing_decomposition."""
    import numpy as np

    u = np.asarray(transition, dtype=float)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("the transition matrix must be square")
    if u.shape[0] < 2:
        raise ValueError("the transition matrix must hold at least two classes")
    if not np.all(np.isfinite(u)):
        raise ValueError("transition entries must be finite")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("transition entries must lie in [0, 1]")

    n = u.shape[0]
    ageing = np.zeros((n, n), dtype=float)
    for a in range(n - 1):
        ageing[a + 1, a] = 1.0
    ageing[n - 1, n - 1] = 1.0
    if np.any(u[ageing == 0.0] != 0.0):
        raise ValueError("only the first sub-diagonal and the last diagonal entry may "
                         "be non-zero")

    survival = np.array([u[a + 1, a] for a in range(n - 1)] + [u[n - 1, n - 1]])
    survival_operator = np.diag(survival)
    return survival_operator, ageing

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    invalid = (
        "import numpy as np\n"
        "def run_model(**kw):\n"
        "    try:\n"
        "        survival_ageing_decomposition(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "def run_gold(**kw):\n"
        "    try:\n"
        "        _oracle_survival_ageing_decomposition(**kw)\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # Normal: the transition matrix of the paper's four-class simple table.
        {"setup": ("import numpy as np\n"
                   "U = np.array([[0.0, 0.0, 0.0, 0.0],\n"
                   "              [0.5, 0.0, 0.0, 0.0],\n"
                   "              [0.0, 0.6, 0.0, 0.0],\n"
                   "              [0.0, 0.0, 0.7, 0.0]])\n"),
         "call": "survival_ageing_decomposition(U)",
         "gold_call": "_oracle_survival_ageing_decomposition(U)"},
        # Normal: a pooled terminal class with finite survival.
        {"setup": ("import numpy as np\n"
                   "U = np.array([[0.000, 0.000, 0.000, 0.000, 0.000],\n"
                   "              [0.699, 0.000, 0.000, 0.000, 0.000],\n"
                   "              [0.000, 0.774, 0.000, 0.000, 0.000],\n"
                   "              [0.000, 0.000, 0.947, 0.000, 0.000],\n"
                   "              [0.000, 0.000, 0.000, 0.947, 0.634]])\n"),
         "call": "survival_ageing_decomposition(U)",
         "gold_call": "_oracle_survival_ageing_decomposition(U)"},
        # Boundary: the smallest age structure.
        {"setup": "import numpy as np\nU = np.array([[0.0, 0.0], [1.0, 0.5]])\n",
         "call": "survival_ageing_decomposition(U)",
         "gold_call": "_oracle_survival_ageing_decomposition(U)"},
        # Invalid: a probability above one.
        {"setup": "import numpy as np\nU = np.array([[0.0, 0.0], [0.5, 1.4]])\n" + invalid,
         "call": "run_model(transition=U)",
         "gold_call": "run_gold(transition=U)"},
        # Invalid: a negative probability.
        {"setup": "import numpy as np\nU = np.array([[0.0, 0.0], [0.5, -0.1]])\n" + invalid,
         "call": "run_model(transition=U)",
         "gold_call": "run_gold(transition=U)"},
        # Invalid: a non-zero entry outside the sub-diagonal and the last diagonal entry.
        {"setup": "import numpy as np\nU = np.array([[0.2, 0.0], [0.5, 0.6]])\n" + invalid,
         "call": "run_model(transition=U)",
         "gold_call": "run_gold(transition=U)"},
        # Invalid: a single class.
        {"setup": "import numpy as np\nU = np.array([[0.5]])\n" + invalid,
         "call": "run_model(transition=U)",
         "gold_call": "run_gold(transition=U)"},
    ]
