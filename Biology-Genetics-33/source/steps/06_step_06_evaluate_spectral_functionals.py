"""
Reduce the two covariance components to the normalised traces of every ordered product of them, up to a given word length.

The covariance components enter the asymptotic moments only through normalised traces of ordered products of themselves. Tabulating those traces once removes the components from the rest of the calculation.

Returns
-------
np.ndarray of shape (2 ** (order + 1) - 2,), float: the normalised trace of every covariance-component word.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_spectral_functionals(components: np.ndarray, order: int) -> np.ndarray:
    """Tabulate the normalised traces of ordered products of the components.

    Words over the two components are indexed first by length and then
    lexicographically, so the word ``(w_1, ..., w_s)`` sits at position
    ``2**s - 2 + sum_t w_t * 2**(s - t)``; letter zero is the family-level
    component and letter one the individual-level component.

    Parameters
    ----------
    components : np.ndarray
        Array of shape ``(2, p, p)`` holding the two trait-space covariance
        matrices.
    order : int
        Longest word length to tabulate; a positive integer.

    Returns
    -------
    functionals : np.ndarray
        Array of shape ``(2 ** (order + 1) - 2,)`` holding the trace of the
        ordered product of components for every word of length one to ``order``,
        each divided by the number of traits, in the index order described
        above.

    Raises
    ------
    ValueError
        If ``components`` is not a three-dimensional array of finite entries
        with leading dimension two and square trailing dimensions, or if
        ``order`` is not an integer value of at least one.
    """
    return functionals  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_spectral_functionals(components: np.ndarray, order: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    block = np.asarray(components, dtype=float)
    if block.ndim != 3 or block.shape[0] != 2:
        raise ValueError("components must have shape (2, p, p)")
    if block.shape[1] != block.shape[2] or block.shape[1] < 1:
        raise ValueError("components must hold square matrices of positive size")
    if not np.all(np.isfinite(block)):
        raise ValueError("components must contain only finite entries")

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    if not _is_number(order) or float(order) != float(int(order)) or int(order) < 1:
        raise ValueError("order must be an integer value of at least one")
    order = int(order)

    n_traits = block.shape[1]
    functionals = np.zeros((1 << (order + 1)) - 2, dtype=float)
    for length in range(1, order + 1):
        base = (1 << length) - 2
        for offset in range(1 << length):
            # Words of the given length are enumerated by reading the offset in
            # binary, which is the index order the functional table uses.
            word = [(offset >> (length - 1 - place)) & 1 for place in range(length)]
            if length == 1:
                value = float(np.trace(block[word[0]]))
            else:
                accumulated = block[word[0]]
                for letter in word[1:-1]:
                    accumulated = accumulated @ block[letter]
                value = float(np.einsum("ij,ji->", accumulated, block[word[-1]]))
            functionals[base + offset] = value / float(n_traits)

    return functionals

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: rotated step spectra at the order the pipeline uses (normal
        #     scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
p = 12
lam1 = np.where(np.arange(p) < 5, 1.4, 0.0)
lam2 = np.where(np.arange(p) < 10, 0.6, 0.0)
V = np.eye(p)
c, s_ = np.cos(0.7), np.sin(0.7)
for i in range(p // 2):
    j = p - 1 - i
    V[i, i], V[i, j], V[j, i], V[j, j] = c, -s_, s_, c
components = np.stack([V @ np.diag(lam1) @ V.T, np.diag(lam2)])
""",
            "call": "sig(evaluate_spectral_functionals(components, 2), 1.0)",
            "gold_call": "sig(_oracle_evaluate_spectral_functionals(components, 2), 1.0)",
        },
        # --- Valid: two general non-commuting symmetric matrices at third order,
        #     where the two orderings of a mixed word differ ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
g = np.arange(6, dtype=float)
A = np.cos(0.3 * (g[:, None] + 2.0 * g[None, :]))
B = np.sin(0.5 + 0.2 * g[:, None] * g[None, :])
components = np.stack([0.5 * (A + A.T), 0.5 * (B + B.T)])
""",
            "call": "sig(evaluate_spectral_functionals(components, 3), 1.0)",
            "gold_call": "sig(_oracle_evaluate_spectral_functionals(components, 3), 1.0)",
        },
        # --- Boundary: order one, where only the two normalised traces exist ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
p = 8
components = np.stack([np.diag(np.where(np.arange(p) < 3, 2.0, 0.0)),
                       np.diag(np.where(np.arange(p) < 6, 0.5, 0.0))])
""",
            "call": "sig(evaluate_spectral_functionals(components, 1), 1.0)",
            "gold_call": "sig(_oracle_evaluate_spectral_functionals(components, 1), 1.0)",
        },
        # --- Edge: two commuting diagonal components, for which every mixed word
        #     reduces to an overlap of the two spectra ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
p = 10
components = np.stack([np.diag(np.where(np.arange(p) < 4, 1.4, 0.0)),
                       np.diag(np.where(np.arange(p) < 8, 0.6, 0.0))])
""",
            "call": "sig(evaluate_spectral_functionals(components, 3), 1.0)",
            "gold_call": "sig(_oracle_evaluate_spectral_functionals(components, 3), 1.0)",
        },
        # --- Edge: a single-trait problem, the smallest square block ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
components = np.stack([np.array([[1.7]]), np.array([[0.4]])])
""",
            "call": "sig(evaluate_spectral_functionals(components, 2), 1.0)",
            "gold_call": "sig(_oracle_evaluate_spectral_functionals(components, 2), 1.0)",
        },
        # --- Invalid: three components instead of two ---
        {
            "setup": """import numpy as np
components = np.stack([np.eye(3), np.eye(3), np.eye(3)])
def run_model():
    try:
        evaluate_spectral_functionals(components, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_spectral_functionals(components, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive word length ---
        {
            "setup": """import numpy as np
components = np.stack([np.eye(3), np.eye(3)])
def run_model():
    try:
        evaluate_spectral_functionals(components, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_spectral_functionals(components, 0)
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
