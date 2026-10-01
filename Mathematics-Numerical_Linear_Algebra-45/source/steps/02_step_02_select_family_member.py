"""
Choose the component polynomial applied at one iteration of the non-accelerated recursion and return the images of the two interior eigenvalues under it.

Each iteration applies the family member that leaves the widest separation between the images of the two interior eigenvalues, which reduces the condition number of the step-function problem as fast as the family allows. Once the lower interior eigenvalue has fallen below kappa = 0.01 and the upper one has risen above 1 - kappa, gap amplification is exhausted and the selection is replaced by a fixed alternation that applies member 3 on odd iterations and member 4 on even ones.

Returns
-------
np.ndarray, float, shape (3,): the selected family index followed by the images of lam_lumo and lam_homo under the selected component polynomial.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def select_family_member(lam_lumo: float, lam_homo: float, iteration: int) -> np.ndarray:
    """Select one component polynomial and advance the interior eigenvalues.

    Parameters
    ----------
    lam_lumo : float
        Largest eigenvalue below the step, with 0 <= lam_lumo < lam_homo.
    lam_homo : float
        Smallest eigenvalue above the step, with lam_homo <= 1.
    iteration : int
        One-based index of the recursion iteration (iteration >= 1).

    Returns
    -------
    state : np.ndarray
        Shape (3,) float array [index, new_lam_lumo, new_lam_homo], where
        ``index`` is the selected family member stored as a float and the
        remaining two entries are the images of the interior eigenvalues
        under the selected component polynomial.

    Raises
    ------
    ValueError
        If ``lam_lumo`` or ``lam_homo`` is not a real number or is not
        finite; if they do not satisfy 0 <= lam_lumo < lam_homo <= 1; if
        ``iteration`` is not an integer (a bool counts as not an integer);
        or if ``iteration`` is less than 1. The function must raise rather
        than return a placeholder or reorder the two eigenvalues itself.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return np.zeros(3, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_family_member(lam_lumo: float, lam_homo: float,
                                 iteration: int) -> np.ndarray:
    import numpy as np
    from math import comb

    for name, val in (("lam_lumo", lam_lumo), ("lam_homo", lam_homo)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(val)):
            raise ValueError(f"{name} must be finite")
    lam_lumo = float(lam_lumo)
    lam_homo = float(lam_homo)
    if not (0.0 <= lam_lumo < lam_homo <= 1.0):
        raise ValueError("require 0 <= lam_lumo < lam_homo <= 1")
    if isinstance(iteration, bool) or not isinstance(iteration, (int, np.integer)):
        raise ValueError("iteration must be an integer")
    iteration = int(iteration)
    if iteration < 1:
        raise ValueError("iteration must be >= 1")

    def family(L):
        # Same construction as sub-problem 01, repeated locally so that this
        # oracle can be executed on its own.
        c = 8.0 * comb(7, L)
        b = np.zeros(9, dtype=float)
        for j in range(7 - L + 1):
            b[L + j + 1] += c * comb(7 - L, j) * (-1.0) ** j / (L + j + 1)
        return b

    def value(b, x):
        acc = 0.0
        for i in range(8, -1, -1):
            acc = acc * x + b[i]
        return float(acc)

    kappa = 0.01
    if lam_lumo < kappa and 1.0 - kappa < lam_homo:
        # Acceleration is exhausted at both ends: alternate the two central
        # members, which together give asymptotic order 20 over two iterations.
        index = 3 if iteration % 2 == 1 else 4
    else:
        gaps = [value(family(L), lam_homo) - value(family(L), lam_lumo)
                for L in range(8)]
        index = int(np.argmax(np.asarray(gaps, dtype=float)))

    b = family(index)
    return np.array([float(index), value(b, lam_lumo), value(b, lam_homo)],
                    dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Pinned values obtained independently of the oracle. With the
        # interior eigenvalues placed just below and just above 1/2 the member
        # indexed 4 wins, and the two images follow from the tabulated closed
        # form -35*x**8 + 120*x**7 - 140*x**6 + 56*x**5 evaluated by hand.
        {
            "setup": """import numpy as np
def p43(x):
    return -35.0 * x ** 8 + 120.0 * x ** 7 - 140.0 * x ** 6 + 56.0 * x ** 5
lo, hi = 0.48, 0.52
EXPECTED = float(4.0 + 100.0 * p43(lo) + 10000.0 * p43(hi))
""",
            "call": ("float(select_family_member(lo, hi, 1)[0]"
                     " + 100.0 * select_family_member(lo, hi, 1)[1]"
                     " + 10000.0 * select_family_member(lo, hi, 1)[2])"),
            "gold_call": "EXPECTED",
        },
        # --- Valid: a gap sitting well below the middle of the interval, where
        # the selection must move away from the central members ---
        {
            "setup": """import numpy as np
lo, hi = 0.2198, 0.2202
""",
            "call": "float(np.dot(select_family_member(lo, hi, 1), np.array([1.0, 100.0, 10000.0])))",
            "gold_call": ("float(np.dot(_oracle_select_family_member(lo, hi, 1),"
                          " np.array([1.0, 100.0, 10000.0])))"),
        },
        # --- Valid: a gap sitting well above the middle of the interval ---
        {
            "setup": """import numpy as np
lo, hi = 0.7642, 0.7688
""",
            "call": "float(np.dot(select_family_member(lo, hi, 4), np.array([1.0, 100.0, 10000.0])))",
            "gold_call": ("float(np.dot(_oracle_select_family_member(lo, hi, 4),"
                          " np.array([1.0, 100.0, 10000.0])))"),
        },
        # --- Boundary: both interior eigenvalues inside kappa of their limits,
        # so the alternation replaces the selection and the parity of the
        # iteration index decides which member is applied ---
        {
            "setup": """import numpy as np
lo, hi = 0.004, 0.997
""",
            "call": ("float(select_family_member(lo, hi, 7)[0]"
                     " + 10.0 * select_family_member(lo, hi, 8)[0]"
                     " + 1000.0 * select_family_member(lo, hi, 7)[2])"),
            "gold_call": ("float(_oracle_select_family_member(lo, hi, 7)[0]"
                          " + 10.0 * _oracle_select_family_member(lo, hi, 8)[0]"
                          " + 1000.0 * _oracle_select_family_member(lo, hi, 7)[2])"),
        },
        # --- Edge: only one end is inside kappa, so the gap-maximising rule
        # still governs the choice ---
        {
            "setup": """import numpy as np
lo, hi = 0.003, 0.61
""",
            "call": "float(np.dot(select_family_member(lo, hi, 3), np.array([1.0, 100.0, 10000.0])))",
            "gold_call": ("float(np.dot(_oracle_select_family_member(lo, hi, 3),"
                          " np.array([1.0, 100.0, 10000.0])))"),
        },
        # --- Invalid: interior eigenvalues in the wrong order ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        select_family_member(0.6, 0.4, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_family_member(0.6, 0.4, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: iteration index below one ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        select_family_member(0.34, 0.36, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_select_family_member(0.34, 0.36, 0)
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
