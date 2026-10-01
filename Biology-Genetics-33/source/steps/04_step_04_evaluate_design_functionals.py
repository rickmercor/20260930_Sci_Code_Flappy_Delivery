"""
Reduce a level-increment matrix and the two design matrices of the nested model to the normalised traces of every ordered product of the associated design operators, up to a given word length.

The design enters the asymptotic moments only through normalised traces of products of a family of operators built from it. Tabulating those traces once removes the design from the rest of the calculation.

Returns
-------
np.ndarray of shape (2 ** (order + 1) - 2,), float: the normalised trace of every design-operator word.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_design_functionals(projector: np.ndarray,
                                family_sizes: np.ndarray,
                                n_traits: int,
                                order: int) -> np.ndarray:
    """Tabulate the normalised traces of products of the design operators.

    The design operator of level ``r`` is the level-increment matrix multiplied
    by the outer product of the level-``r`` design matrix with itself; level
    zero is the family level and level one the individual level, whose design
    matrix is the identity.

    Words over the two levels are indexed first by length and then
    lexicographically, so the word ``(w_1, ..., w_s)`` sits at position
    ``2**s - 2 + sum_t w_t * 2**(s - t)``.

    Parameters
    ----------
    projector : np.ndarray
        Square array of shape ``(n, n)`` holding one level-increment matrix of
        the nested design.
    family_sizes : np.ndarray
        One-dimensional array of the number of individuals in each family,
        summing to ``n`` and ordered as the rows of ``projector``.
    n_traits : int
        Number of measured traits, the normalisation of every trace; a positive
        integer.
    order : int
        Longest word length to tabulate; a positive integer.

    Returns
    -------
    functionals : np.ndarray
        Array of shape ``(2 ** (order + 1) - 2,)`` holding the normalised trace
        of the ordered product of design operators for every word of length one
        to ``order``, in the index order described above.

    Raises
    ------
    ValueError
        If ``projector`` is not a square two-dimensional array of finite
        entries, if ``family_sizes`` is not a one-dimensional array of integer
        values of at least one summing to the size of ``projector``, or if
        ``n_traits`` or ``order`` is not an integer value of at least one.
    """
    return functionals  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_design_functionals(projector: np.ndarray,
                                        family_sizes: np.ndarray,
                                        n_traits: int,
                                        order: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    matrix = np.asarray(projector, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("projector must be a square two-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("projector must contain only finite entries")

    sizes = np.asarray(family_sizes, dtype=float)
    if sizes.ndim != 1 or sizes.size < 1 or not np.all(np.isfinite(sizes)):
        raise ValueError("family_sizes must be a non-empty one-dimensional finite array")
    if not np.allclose(sizes, np.round(sizes), rtol=0.0, atol=1e-12):
        raise ValueError("family_sizes entries must be integer valued")
    sizes = np.asarray(np.round(sizes), dtype=np.int64)
    if np.any(sizes < 1):
        raise ValueError("every family must contain at least one individual")
    if int(sizes.sum()) != matrix.shape[0]:
        raise ValueError("family_sizes must sum to the dimension of projector")

    for name, value in (("n_traits", n_traits), ("order", order)):
        if not _is_number(value) or float(value) != float(int(value)) or int(value) < 1:
            raise ValueError(f"{name} must be an integer value of at least one")
    n_traits = int(n_traits)
    order = int(order)

    n_individuals = matrix.shape[0]
    membership = np.zeros((n_individuals, int(sizes.size)), dtype=float)
    start = 0
    for family, size in enumerate(sizes):
        membership[start:start + int(size), family] = 1.0
        start += int(size)

    # Design operators: the family level carries the family membership outer
    # product, the individual level the identity, so it reduces to the projector.
    operators = [matrix @ (membership @ membership.T), matrix.copy()]

    functionals = np.zeros((1 << (order + 1)) - 2, dtype=float)
    for length in range(1, order + 1):
        base = (1 << length) - 2
        for offset in range(1 << length):
            # Words of the given length are enumerated by reading the offset in
            # binary, which is the index order the functional table uses.
            word = [(offset >> (length - 1 - place)) & 1 for place in range(length)]
            if length == 1:
                value = float(np.trace(operators[word[0]]))
            else:
                accumulated = operators[word[0]]
                for letter in word[1:-1]:
                    accumulated = accumulated @ operators[letter]
                value = float(np.einsum("ij,ji->", accumulated, operators[word[-1]]))
            functionals[base + offset] = value / float(n_traits)

    return functionals

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the between-family increment of a small unbalanced design at
        #     the order the pipeline uses (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
n = int(family_sizes.sum())
U1 = np.zeros((n, family_sizes.size))
o = 0
for i, j in enumerate(family_sizes):
    U1[o:o + j, i] = 1.0
    o += j
P1 = U1 @ (U1.T / family_sizes[:, None].astype(float))
projector = P1 - np.full((n, n), 1.0 / n)
""",
            "call": "sig(evaluate_design_functionals(projector, family_sizes, 24, 2), 1.0)",
            "gold_call": "sig(_oracle_evaluate_design_functionals(projector, family_sizes, 24, 2), 1.0)",
        },
        # --- Valid: the within-family increment, for which the family-level
        #     operator vanishes identically ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)
n = int(family_sizes.sum())
U1 = np.zeros((n, family_sizes.size))
o = 0
for i, j in enumerate(family_sizes):
    U1[o:o + j, i] = 1.0
    o += j
projector = np.eye(n) - U1 @ (U1.T / family_sizes[:, None].astype(float))
""",
            "call": "sig(evaluate_design_functionals(projector, family_sizes, 24, 2), 1.0)",
            "gold_call": "sig(_oracle_evaluate_design_functionals(projector, family_sizes, 24, 2), 1.0)",
        },
        # --- Valid: third-order words on a strongly unbalanced design ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([5, 1, 2, 1], dtype=np.int64)
n = int(family_sizes.sum())
U1 = np.zeros((n, family_sizes.size))
o = 0
for i, j in enumerate(family_sizes):
    U1[o:o + j, i] = 1.0
    o += j
P1 = U1 @ (U1.T / family_sizes[:, None].astype(float))
projector = P1 - np.full((n, n), 1.0 / n)
""",
            "call": "sig(evaluate_design_functionals(projector, family_sizes, 12, 3), 1.0e2)",
            "gold_call": "sig(_oracle_evaluate_design_functionals(projector, family_sizes, 12, 3), 1.0e2)",
        },
        # --- Boundary: order one, where only the two single-letter traces exist ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([2, 2, 2, 2], dtype=np.int64)
n = int(family_sizes.sum())
U1 = np.zeros((n, family_sizes.size))
o = 0
for i, j in enumerate(family_sizes):
    U1[o:o + j, i] = 1.0
    o += j
projector = np.eye(n) - U1 @ (U1.T / family_sizes[:, None].astype(float))
""",
            "call": "sig(evaluate_design_functionals(projector, family_sizes, 30, 1), 1.0)",
            "gold_call": "sig(_oracle_evaluate_design_functionals(projector, family_sizes, 30, 1), 1.0)",
        },
        # --- Edge: a general symmetric matrix that is not a projection, which
        #     separates the two directions of a two-letter word ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
family_sizes = np.array([2, 1, 3], dtype=np.int64)
n = int(family_sizes.sum())
grid = np.arange(n, dtype=float)
projector = np.cos(0.4 * (grid[:, None] - grid[None, :])) / (1.0 + grid[:, None] + grid[None, :])
projector = 0.5 * (projector + projector.T)
""",
            "call": "sig(evaluate_design_functionals(projector, family_sizes, 9, 3), 1.0)",
            "gold_call": "sig(_oracle_evaluate_design_functionals(projector, family_sizes, 9, 3), 1.0)",
        },
        # --- Invalid: family sizes that do not sum to the dimension ---
        {
            "setup": """import numpy as np
family_sizes = np.array([2, 2], dtype=np.int64)
projector = np.eye(5)
def run_model():
    try:
        evaluate_design_functionals(projector, family_sizes, 10, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_design_functionals(projector, family_sizes, 10, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive number of traits ---
        {
            "setup": """import numpy as np
family_sizes = np.array([2, 2], dtype=np.int64)
projector = np.eye(4)
def run_model():
    try:
        evaluate_design_functionals(projector, family_sizes, 0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_design_functionals(projector, family_sizes, 0, 2)
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
