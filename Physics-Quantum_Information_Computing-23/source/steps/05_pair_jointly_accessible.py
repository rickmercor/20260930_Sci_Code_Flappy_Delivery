"""
Returns 1 if two retained Pauli pairs can be estimated in one preparation–measurement setting.

A grouped fidelity protocol can reuse a setting only when the two output labels are compatible and the two input labels are compatible. That is a property of the pair of Pauli pairs, not of a single Pauli.

Returns
-------
A Python float, 1.0 if the two pairs share a setting and 0.0 otherwise.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def pair_jointly_accessible(pair_a: np.ndarray, pair_b: np.ndarray) -> float:
    """Return 1 if two retained pairs can be estimated together, else 0.
    Each pair is a length-3 row (alpha, beta, coefficient) from the
    retained-support table. The test is the hardware compatibility of the
    two output labels and of the two input labels.
    Parameters
    ----------
    pair_a : np.ndarray
        Length-3 row (alpha, beta, coefficient).
    pair_b : np.ndarray
        Length-3 row (alpha, beta, coefficient).
    Returns
    -------
    flag : float
        1.0 if the two pairs share a setting, otherwise 0.0.
    Raises
    ------
    ValueError
        If either argument is not a finite length-3 array, or if any Pauli
        label is not an integer in {0, ..., 15}.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_pair_jointly_accessible(pair_a: np.ndarray, pair_b: np.ndarray) -> float:
    import numpy as np
    a = np.asarray(pair_a, dtype=float).reshape(-1)
    b = np.asarray(pair_b, dtype=float).reshape(-1)
    if a.size != 3 or b.size != 3:
        raise ValueError("each pair must be a length-3 row")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("pair rows must be finite")
    def _code(x):
        xr = float(x)
        if abs(xr - round(xr)) > 1e-12:
            raise ValueError("Pauli labels must be integers")
        c = int(round(xr))
        if c < 0 or c > 15:
            raise ValueError("Pauli labels must lie in {0, ..., 15}")
        return c
    def _local_ok(p, q):
        return p == 0 or q == 0 or p == q
    a0, a1 = _code(a[0]), _code(a[1])
    b0, b1 = _code(b[0]), _code(b[1])
    out_ok = _local_ok(a0 // 4, b0 // 4) and _local_ok(a0 % 4, b0 % 4)
    inn_ok = _local_ok(a1 // 4, b1 // 4) and _local_ok(a1 % 4, b1 % 4)
    if out_ok and inn_ok:
        return 1.0
    return 0.0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
pair_a = np.array([0.0, 5.0, 1.0])
pair_b = np.array([0.0, 1.0, 0.5])
""",
            "call": "pair_jointly_accessible(pair_a.copy(), pair_b.copy())",
            "gold_call": "_oracle_pair_jointly_accessible(pair_a.copy(), pair_b.copy())",
        },
        {
            "setup": """import numpy as np
pair_a = np.array([5.0, 5.0, 1.0])
pair_b = np.array([10.0, 10.0, 0.5])
""",
            "call": "pair_jointly_accessible(pair_a.copy(), pair_b.copy())",
            "gold_call": "_oracle_pair_jointly_accessible(pair_a.copy(), pair_b.copy())",
        },
        {
            "setup": """import numpy as np
pair_a = np.array([1.0, 4.0, 0.2])
pair_b = np.array([4.0, 1.0, 0.3])
""",
            "call": "pair_jointly_accessible(pair_a.copy(), pair_b.copy())",
            "gold_call": "_oracle_pair_jointly_accessible(pair_a.copy(), pair_b.copy())",
        },
        {
            "setup": """import numpy as np
pair_a = np.array([16.0, 0.0, 1.0])
pair_b = np.array([0.0, 0.0, 1.0])
def run_model():
    try:
        pair_jointly_accessible(pair_a.copy(), pair_b.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_pair_jointly_accessible(pair_a.copy(), pair_b.copy())
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
