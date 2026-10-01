"""
Return the monomial coefficients of one member of the eight-member family of degree-eight component polynomials used by the recursive expansion when no acceleration is applied.

When the recursion is given only the trivial spectral bounds 0 and 1, the equioscillation conditions collapse and the family member indexed by L is the unique degree-eight polynomial with p(0) = 0, p(1) = 1 and all seven stationary points at the endpoints, L of them at x = 0 and 7 - L at x = 1. Recover the monomial coefficients from those multiplicity and normalisation conditions; do not rely on a hard-coded table of the eight answers.

Returns
-------
np.ndarray, float, shape (9,): the monomial coefficients b0..b8 of the component polynomial indexed by L, in increasing powers of x.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sp8_family_coefficients(index: int) -> np.ndarray:
    """Return the monomial coefficients of one component polynomial of the family.

    Parameters
    ----------
    index : int
        Member index L of the family, an integer with 0 <= L <= 7.

    Returns
    -------
    coeffs : np.ndarray
        Shape (9,) float array [b0, b1, ..., b8] such that the component
        polynomial is p(x) = b0 + b1*x + ... + b8*x**8.

    Raises
    ------
    ValueError
        If ``index`` is not an integer (a bool counts as not an integer), or
        if it falls outside the range 0 <= index <= 7. The function must
        raise rather than return a placeholder or clamp the index.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(9, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sp8_family_coefficients(index: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np
    from math import comb

    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("index must be an integer")
    index = int(index)
    if index < 0 or index > 7:
        raise ValueError("index must satisfy 0 <= index <= 7")

    # Tabulated members for L = 4, 5, 6, 7, stored as b0..b8.
    table = {
        7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        6: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0, -7.0],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.0, -48.0, 21.0],
        4: [0.0, 0.0, 0.0, 0.0, 0.0, 56.0, -140.0, 120.0, -35.0],
    }
    if index >= 4:
        return np.array(table[index], dtype=float)

    # Reflection: p_L(x) = 1 - p_(7-L)(1 - x). Expanding (1 - x)**k with the
    # binomial theorem turns the mirrored polynomial back into monomials.
    mirror = np.array(table[7 - index], dtype=float)
    coeffs = np.zeros(9, dtype=float)
    for k in range(9):
        if mirror[k] == 0.0:
            continue
        for j in range(k + 1):
            coeffs[j] -= mirror[k] * comb(k, j) * (-1.0) ** j
    coeffs[0] += 1.0
    return coeffs

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values obtained independently of the oracle. The member
        # indexed by L has derivative proportional to x**L * (1 - x)**(7 - L)
        # normalised so that p(1) - p(0) = 1, so its coefficients can be built
        # by integrating that product term by term. Probing the whole vector
        # with the alternating weights 2**j separates every coefficient, so a
        # sign slip, a reversed index or a dropped reflection cannot pass.
        {
            "setup": """import numpy as np
from math import comb
def exact(L):
    c = 8.0 * comb(7, L)
    b = np.zeros(9)
    for j in range(7 - L + 1):
        b[L + j + 1] += c * comb(7 - L, j) * (-1.0) ** j / (L + j + 1)
    return b
w = np.array([(-2.0) ** j for j in range(9)])
EXPECTED = float(sum((L + 1) * np.dot(w, exact(L)) for L in range(8)))
""",
            "call": ("float(sum((L + 1) * np.dot(w, sp8_family_coefficients(L))"
                     " for L in range(8)))"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: the two members that a recursion spends most of its
        # multiplications on (normal scenario) ---
        {
            "setup": """import numpy as np
v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
""",
            "call": ("float(np.dot(v, sp8_family_coefficients(4))"
                     " + 13.0 * np.dot(v, sp8_family_coefficients(3)))"),
            "gold_call": ("float(np.dot(v, _oracle_sp8_family_coefficients(4))"
                          " + 13.0 * np.dot(v, _oracle_sp8_family_coefficients(3)))"),
        },
        # --- Valid: a reflected member evaluated at three interior points ---
        {
            "setup": """import numpy as np
xs = np.array([0.17, 0.4, 0.86])
def value(b, x):
    return float(sum(b[i] * x ** i for i in range(9)))
""",
            "call": "float(sum(value(sp8_family_coefficients(1), x) for x in xs))",
            "gold_call": "float(sum(value(_oracle_sp8_family_coefficients(1), x) for x in xs))",
        },
        # --- Boundary: the two extreme members, where the polynomial degenerates
        # to a pure power and to its reflection ---
        {
            "setup": """import numpy as np
""",
            "call": ("float(np.sum(np.abs(sp8_family_coefficients(7)))"
                     " + 1000.0 * np.sum(np.abs(sp8_family_coefficients(0))))"),
            "gold_call": ("float(np.sum(np.abs(_oracle_sp8_family_coefficients(7)))"
                          " + 1000.0 * np.sum(np.abs(_oracle_sp8_family_coefficients(0))))"),
        },
        # --- Edge: every member must fix both ends of the unit interval, so the
        # constant term vanishes and the coefficients sum to one ---
        {
            "setup": """import numpy as np
def endpoints(b):
    return float(b[0]) + 10.0 * float(np.sum(b))
""",
            "call": "float(sum(endpoints(sp8_family_coefficients(L)) for L in range(8)))",
            "gold_call": "float(sum(endpoints(_oracle_sp8_family_coefficients(L)) for L in range(8)))",
        },
        # --- Invalid: member index outside the family ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        sp8_family_coefficients(8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sp8_family_coefficients(8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-integer member index ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        sp8_family_coefficients(3.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sp8_family_coefficients(3.5)
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
