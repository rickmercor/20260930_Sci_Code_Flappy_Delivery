"""
Returns the target-dependent finite-shot overhead of the compatible partition of a retained-support table.

Once the supported Pauli pairs are known, the grouped protocol replaces the raw pair count by a single overhead attached to a compatible partition of that table. How the pairs are blocked, and what scalar each block contributes, is the content of the grouped finite-shot bound.

Returns
-------
A Python float, the overhead of the partition; 0 if the table is empty.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def partition_shot_overhead(pair_table: np.ndarray) -> float:
    """Return the target-dependent finite-shot overhead of pair_table.
    The input is the retained-support table. Two rows may share a group
    only when they are jointly accessible and they agree on whether each
    register's Pauli is the identity. An empty table returns 0.
    Order rows by decreasing squared coefficient rounded to twelve decimal
    places, breaking ties by increasing output label, then input label.
    Repeatedly start a group with the first remaining row and scan the
    other remaining rows once in that order. Add a row when it is jointly
    accessible with every current member and agrees with their identity
    status on each register. Repeat on the unassigned rows.

    For each group with unrounded coefficients u, its contribution is
    (sum(abs(u)))**2 / sum(u**2). Return the sum of these contributions
    over groups. Rounding is used only for ordering, never for group costs.

    Parameters
    ----------
    pair_table : np.ndarray
        Array of shape (K, 3) with rows (alpha, beta, coefficient).
    Returns
    -------
    overhead : float
        Nonnegative scalar. Empty input yields 0.
    Raises
    ------
    ValueError
        If pair_table is not a finite array of shape (K, 3), if any Pauli
        label is not an integer in {0, ..., 15}, if any coefficient is
        zero, or if a formed block has vanishing coefficient mass.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_partition_shot_overhead(pair_table: np.ndarray) -> float:
    import numpy as np
    table = np.asarray(pair_table, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3:
        raise ValueError("pair_table must have shape (K, 3)")
    if not np.all(np.isfinite(table)):
        raise ValueError("pair_table must be finite")
    k = int(table.shape[0])
    if k == 0:
        return 0.0
    for i in range(k):
        for j in (0, 1):
            raw = float(table[i, j])
            if abs(raw - round(raw)) > 1e-12:
                raise ValueError("Pauli labels must be integers")
            code = int(round(raw))
            if code < 0 or code > 15:
                raise ValueError("Pauli labels must lie in {0, ..., 15}")
        if float(table[i, 2]) == 0.0:
            raise ValueError("zero coefficients are excluded from support")
    order = sorted(
        range(k),
        key=lambda i: (
            -round(float(table[i, 2]) ** 2, 12),
            float(table[i, 0]),
            float(table[i, 1]),
        ),
    )
    remaining = list(order)
    total = 0.0
    def _same_identity_class(i, j):
        ai, bi = int(round(float(table[i, 0]))), int(round(float(table[i, 1])))
        aj, bj = int(round(float(table[j, 0]))), int(round(float(table[j, 1])))
        return (ai == 0) == (aj == 0) and (bi == 0) == (bj == 0)
    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        keep = []
        for cand in remaining:
            ok = True
            for mem in members:
                if _oracle_pair_jointly_accessible(table[cand], table[mem]) < 0.5:
                    ok = False
                    break
                if not _same_identity_class(cand, mem):
                    ok = False
                    break
            if ok:
                members.append(cand)
            else:
                keep.append(cand)
        u = table[members, 2]
        l1 = float(np.sum(np.abs(u)))
        l2sq = float(np.sum(u * u))
        if l2sq <= 0.0:
            raise ValueError("a block has vanishing coefficient mass")
        total += (l1 * l1) / l2sq
        remaining = keep
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
pair_table = np.array([
    [5.0, 5.0, 1.0],
    [10.0, 10.0, 0.5],
    [0.0, 1.0, 0.25],
], dtype=float)
""",
            "call": "partition_shot_overhead(pair_table.copy())",
            "gold_call": "_oracle_partition_shot_overhead(pair_table.copy())",
        },
        {
            "setup": """import numpy as np
pair_table = np.zeros((0, 3), dtype=float)
""",
            "call": "partition_shot_overhead(pair_table.copy())",
            "gold_call": "_oracle_partition_shot_overhead(pair_table.copy())",
        },
        {
            "setup": """import numpy as np
c2 = (2.0 + np.sqrt(2.0)) / 4.0
s2 = (2.0 - np.sqrt(2.0)) / 4.0
pair_table = np.array([
    [15.0, 15.0, 1.0],
    [3.0, 3.0, c2],
    [12.0, 12.0, c2],
    [3.0, 12.0, s2],
    [12.0, 3.0, s2],
], dtype=float)
""",
            "call": "partition_shot_overhead(pair_table.copy())",
            "gold_call": "_oracle_partition_shot_overhead(pair_table.copy())",
        },
        {
            "setup": """import numpy as np
pair_table = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
def run_model():
    try:
        partition_shot_overhead(pair_table.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_partition_shot_overhead(pair_table.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
pair_table = np.array([[16.0, 0.0, 1.0]], dtype=float)
def run_model():
    try:
        partition_shot_overhead(pair_table.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_partition_shot_overhead(pair_table.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
pair_table = np.array([[0.5, 1.0, 1.0]], dtype=float)
def run_model():
    try:
        partition_shot_overhead(pair_table.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_partition_shot_overhead(pair_table.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
pair_table = np.array([[1.0, 1.0, 0.0]], dtype=float)
def run_model():
    try:
        partition_shot_overhead(pair_table.copy())
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_partition_shot_overhead(pair_table.copy())
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
