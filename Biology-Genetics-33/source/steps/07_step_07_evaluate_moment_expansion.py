"""
Combine the tabulated design and covariance functionals over every non-crossing pairing and every assignment of levels to produce the almost-sure limit of one trace moment of a sum-of-squares matrix.

In the high-dimensional limit the empirical trace moments of a sum-of-squares matrix stop fluctuating and converge to a deterministic polynomial in the design and covariance functionals, indexed by the pairings of the moment's points.

Returns
-------
float: the almost-sure limit of the l-th normalised trace moment, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_moment_expansion(design_functionals: np.ndarray,
                              spectral_functionals: np.ndarray,
                              block_table: np.ndarray) -> float:
    """Evaluate the almost-sure limit of one trace moment of a sum-of-squares matrix.

    The moment order ``l`` is half the number of columns of ``block_table``.
    Both functional tables index words over the two levels first by length and
    then lexicographically, so the word ``(w_1, ..., w_s)`` sits at position
    ``2**s - 2 + sum_t w_t * 2**(s - t)``.

    Parameters
    ----------
    design_functionals : np.ndarray
        One-dimensional table of normalised traces of design-operator words, of
        length at least ``2 ** (l + 1) - 2``.
    spectral_functionals : np.ndarray
        One-dimensional table of normalised traces of covariance-component
        words, of length at least ``2 ** (l + 1) - 2``.
    block_table : np.ndarray
        Integer array of shape ``(M, 2 * l)`` holding, for each of the ``M``
        non-crossing pairings of ``2 * l`` points labelled ``0, ..., 2 * l - 1``,
        the index of the Kreweras block containing each point.

    Returns
    -------
    moment : float
        The limiting value of the ``l``-th normalised trace moment, as a native
        Python float.

    Raises
    ------
    ValueError
        If either functional table is not a one-dimensional array of finite
        entries long enough for the moment order, if ``block_table`` is not a
        two-dimensional, non-empty array of non-negative integer values, or if
        its second dimension is not a positive even number.
    """
    return moment  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_moment_expansion(design_functionals: np.ndarray,
                                      spectral_functionals: np.ndarray,
                                      block_table: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    table = np.asarray(block_table)
    if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("block_table must be a non-empty two-dimensional array")
    as_float = np.asarray(table, dtype=float)
    if not np.all(np.isfinite(as_float)):
        raise ValueError("block_table must contain only finite entries")
    if not np.allclose(as_float, np.round(as_float), rtol=0.0, atol=1e-12):
        raise ValueError("block_table entries must be integer valued")
    table = np.asarray(np.round(as_float), dtype=np.int64)
    if np.any(table < 0):
        raise ValueError("block_table entries must be non-negative")
    if table.shape[1] % 2 != 0:
        raise ValueError("block_table must have an even number of columns")

    order = table.shape[1] // 2
    required = (1 << (order + 1)) - 2
    design = np.asarray(design_functionals, dtype=float).ravel()
    spectral = np.asarray(spectral_functionals, dtype=float).ravel()
    for name, values in (("design_functionals", design), ("spectral_functionals", spectral)):
        if values.size < required:
            raise ValueError(f"{name} must be tabulated to at least the moment order")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite entries")

    def _index(word):
        offset = 0
        for letter in word:
            offset = 2 * offset + letter
        return (1 << len(word)) - 2 + offset

    moment = 0.0
    for row in range(table.shape[0]):
        labels = table[row]
        blocks = {}
        for position, label in enumerate(labels):
            blocks.setdefault(int(label), []).append(position)
        for code in range(1 << order):
            # Each assignment of a level to the order factors of the moment is
            # one binary word of length equal to the moment order.
            levels = [(code >> (order - 1 - place)) & 1 for place in range(order)]
            term = 1.0
            for positions in blocks.values():
                positions = sorted(positions)
                if positions[0] % 2 == 0:
                    # Even-labelled points carry the design operators; the point
                    # at position v belongs to the v/2-th factor of the moment.
                    word = tuple(levels[v // 2] for v in positions)
                    term *= design[_index(word)]
                else:
                    # Odd-labelled points carry the covariance components, with
                    # the last point wrapping back onto the first factor.
                    word = tuple(levels[((v + 1) // 2) % order] for v in positions)
                    term *= spectral[_index(word)]
            moment += term

    return float(moment)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the second moment on the two pairings of four points, from
        #     genuine matrix functionals (normal scenario) ---
        {
            "setup": """import numpy as np
def words(A, B, order):
    from itertools import product as cp
    out = np.zeros((1 << (order + 1)) - 2)
    M = [A, B]
    for L in range(1, order + 1):
        for w in cp(range(2), repeat=L):
            o = 0
            for x in w:
                o = 2 * o + x
            P = M[w[0]]
            for x in w[1:]:
                P = P @ M[x]
            out[(1 << L) - 2 + o] = np.trace(P) / A.shape[0]
    return out
g = np.arange(7, dtype=float)
A = np.cos(0.3 * (g[:, None] + 2.0 * g[None, :]))
B = np.sin(0.5 + 0.2 * g[:, None] * g[None, :])
A, B = 0.5 * (A + A.T), 0.5 * (B + B.T)
C = np.exp(-0.1 * (g[:, None] - g[None, :]) ** 2)
D = 1.0 / (1.0 + g[:, None] + g[None, :])
C, D = 0.5 * (C + C.T), 0.5 * (D + D.T)
design = words(A, B, 2)
spectral = words(C, D, 2)
block_table = np.array([[0, 1, 2, 1], [0, 1, 0, 2]], dtype=np.int64)
""",
            "call": "round(float(evaluate_moment_expansion(design, spectral, block_table)), 9)",
            "gold_call": "round(float(_oracle_evaluate_moment_expansion(design, spectral, block_table)), 9)",
        },
        # --- Valid: the third moment on the five non-crossing pairings of six
        #     points, where three-point blocks and the wrap-around first appear ---
        {
            "setup": """import numpy as np
def words(A, B, order):
    from itertools import product as cp
    out = np.zeros((1 << (order + 1)) - 2)
    M = [A, B]
    for L in range(1, order + 1):
        for w in cp(range(2), repeat=L):
            o = 0
            for x in w:
                o = 2 * o + x
            P = M[w[0]]
            for x in w[1:]:
                P = P @ M[x]
            out[(1 << L) - 2 + o] = np.trace(P) / A.shape[0]
    return out
g = np.arange(6, dtype=float)
A = np.cos(0.4 * (g[:, None] + g[None, :]))
B = np.sin(0.9 + 0.15 * g[:, None] * g[None, :])
A, B = 0.5 * (A + A.T), 0.5 * (B + B.T)
C = np.exp(-0.2 * np.abs(g[:, None] - g[None, :]))
D = np.cos(0.7 * g[:, None] * g[None, :] / 5.0)
C, D = 0.5 * (C + C.T), 0.5 * (D + D.T)
design = words(A, B, 3)
spectral = words(C, D, 3)
block_table = np.array([[0, 1, 2, 1, 3, 1],
                        [0, 1, 2, 3, 2, 1],
                        [0, 1, 0, 2, 3, 2],
                        [0, 1, 0, 2, 0, 3],
                        [0, 1, 2, 1, 0, 3]], dtype=np.int64)
""",
            "call": "round(float(evaluate_moment_expansion(design, spectral, block_table)), 9)",
            "gold_call": "round(float(_oracle_evaluate_moment_expansion(design, spectral, block_table)), 9)",
        },
        # --- Valid: the fourth moment on all fourteen non-crossing pairings of
        #     eight points, where four-point blocks and several distinct
        #     wrap-around patterns occur together ---
        {
            "setup": """import numpy as np
def words(A, B, order):
    out = np.zeros((1 << (order + 1)) - 2)
    M = [A, B]
    for L in range(1, order + 1):
        for o in range(1 << L):
            w = [(o >> (L - 1 - t)) & 1 for t in range(L)]
            P = M[w[0]]
            for x in w[1:]:
                P = P @ M[x]
            out[(1 << L) - 2 + o] = np.trace(P) / A.shape[0]
    return out
g = np.arange(5, dtype=float)
A = np.cos(0.25 * (g[:, None] + 3.0 * g[None, :]))
B = np.sin(1.1 + 0.3 * g[:, None] * g[None, :])
A, B = 0.5 * (A + A.T), 0.5 * (B + B.T)
C = np.exp(-0.35 * np.abs(g[:, None] - g[None, :]))
D = 1.0 / (2.0 + g[:, None] + 2.0 * g[None, :])
C, D = 0.5 * (C + C.T), 0.5 * (D + D.T)
design = words(A, B, 4)
spectral = words(C, D, 4)
block_table = np.array([[0, 1, 2, 1, 3, 1, 4, 1],
                        [0, 1, 2, 1, 3, 4, 3, 1],
                        [0, 1, 2, 3, 2, 1, 4, 1],
                        [0, 1, 2, 3, 2, 4, 2, 1],
                        [0, 1, 2, 3, 4, 3, 2, 1],
                        [0, 1, 0, 2, 3, 2, 4, 2],
                        [0, 1, 0, 2, 3, 4, 3, 2],
                        [0, 1, 0, 2, 0, 3, 4, 3],
                        [0, 1, 2, 1, 0, 3, 4, 3],
                        [0, 1, 0, 2, 0, 3, 0, 4],
                        [0, 1, 0, 2, 3, 2, 0, 4],
                        [0, 1, 2, 1, 0, 3, 0, 4],
                        [0, 1, 2, 1, 3, 1, 0, 4],
                        [0, 1, 2, 3, 2, 1, 0, 4]], dtype=np.int64)
""",
            "call": "round(float(evaluate_moment_expansion(design, spectral, block_table)), 9)",
            "gold_call": "round(float(_oracle_evaluate_moment_expansion(design, spectral, block_table)), 9)",
        },
        # --- Valid: the third moment again with the two levels carrying very
        #     different scales, so a term attached to the wrong factor of the
        #     moment cannot cancel against another ---
        {
            "setup": """import numpy as np
design = np.array([3.0, 0.05,
                   9.0, 0.15, 0.15, 0.0025,
                   27.0, 0.45, 0.45, 0.0075, 0.45, 0.0075, 0.0075, 0.000125])
spectral = np.array([0.49, 0.011,
                     0.686, 0.0054, 0.0054, 0.00012,
                     0.9604, 0.00756, 0.00756, 0.0000594,
                     0.00756, 0.0000594, 0.0000594, 0.0000013])
block_table = np.array([[0, 1, 2, 1, 3, 1],
                        [0, 1, 2, 3, 2, 1],
                        [0, 1, 0, 2, 3, 2],
                        [0, 1, 0, 2, 0, 3],
                        [0, 1, 2, 1, 0, 3]], dtype=np.int64)
""",
            "call": "round(float(evaluate_moment_expansion(design, spectral, block_table)), 9)",
            "gold_call": "round(float(_oracle_evaluate_moment_expansion(design, spectral, block_table)), 9)",
        },
        # --- Boundary: the first moment, whose single pairing has two singleton
        #     blocks ---
        {
            "setup": """import numpy as np
design = np.array([0.75, 1.25, 0.4, 0.1, 0.1, 0.9])
spectral = np.array([0.49, 0.48, 0.686, 0.2359, 0.2359, 0.288])
block_table = np.array([[0, 1]], dtype=np.int64)
""",
            "call": "round(float(evaluate_moment_expansion(design, spectral, block_table)), 9)",
            "gold_call": "round(float(_oracle_evaluate_moment_expansion(design, spectral, block_table)), 9)",
        },
        # --- Edge: functionals that vanish on one level, which must kill every
        #     assignment touching it ---
        {
            "setup": """import numpy as np
design = np.array([0.0, 1.25, 0.0, 0.0, 0.0, 1.25])
spectral = np.array([0.49, 0.48, 0.686, 0.2359, 0.2359, 0.288])
block_table = np.array([[0, 1, 2, 1], [0, 1, 0, 2]], dtype=np.int64)
""",
            "call": "round(float(evaluate_moment_expansion(design, spectral, block_table)), 9)",
            "gold_call": "round(float(_oracle_evaluate_moment_expansion(design, spectral, block_table)), 9)",
        },
        # --- Edge: functional tables longer than the moment order requires ---
        {
            "setup": """import numpy as np
design = np.array([0.75, 1.25, 0.4, 0.1, 0.1, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
spectral = np.array([0.49, 0.48, 0.686, 0.2359, 0.2359, 0.288,
                     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
block_table = np.array([[0, 1, 2, 1], [0, 1, 0, 2]], dtype=np.int64)
""",
            "call": "round(float(evaluate_moment_expansion(design, spectral, block_table)), 9)",
            "gold_call": "round(float(_oracle_evaluate_moment_expansion(design, spectral, block_table)), 9)",
        },
        # --- Invalid: functional tables too short for the moment order ---
        {
            "setup": """import numpy as np
design = np.array([0.75, 1.25])
spectral = np.array([0.49, 0.48])
block_table = np.array([[0, 1, 2, 1]], dtype=np.int64)
def run_model():
    try:
        evaluate_moment_expansion(design, spectral, block_table)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_moment_expansion(design, spectral, block_table)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an odd number of points ---
        {
            "setup": """import numpy as np
design = np.array([0.75, 1.25, 0.4, 0.1, 0.1, 0.9])
spectral = np.array([0.49, 0.48, 0.686, 0.2359, 0.2359, 0.288])
block_table = np.array([[0, 1, 2]], dtype=np.int64)
def run_model():
    try:
        evaluate_moment_expansion(design, spectral, block_table)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_moment_expansion(design, spectral, block_table)
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
