"""
Solve the hitting-probability system of the chain to obtain the probability of reaching the product state before falling back.

The probability that a trajectory which has just left the reactant boundary reaches the product state before returning to the reactant state is a hitting probability of the chain, and it obeys a linear system obtained by conditioning on the first step. For every state that is neither of the two boundaries, the probability of eventually hitting the product state equals the average of that same probability over the states reachable in one step. Pinning the reactant excursion at zero and the product state at one turns the remaining equations into a square system whose solution is the vector of hitting probabilities.

The quantity of interest is not a hitting probability but the complement of a return probability: it refers to a trajectory that has already left the reactant excursion and must reach the product state before coming back to it. That value is recovered by taking one obligatory step out of the reactant excursion and averaging the pinned hitting probabilities over the states that step can reach. This is what makes the reactant excursion appear twice with different roles - as the pinned zero of the system and as the starting point of the obligatory first step - and confusing the two returns the trivial answer zero. Solving the system this way is equivalent to the recursive relation between neighbouring interfaces that is traditionally used, because inverting a sub-block of the matrix generates exactly the continued fraction the recursion builds up.

Formulas:

    P[d] = sum_g M[d, g] P[g]   for d not a boundary

    P[reactant] = 0,  P[product] = 1

    P_A(lambda_B | lambda_A) = sum_g M[reactant, g] P[g]

Returns
-------
float: the global crossing probability, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_crossing_probability(transition_matrix, states) -> float:
    """Return the probability of reaching the product state before falling back.

    Parameters
    ----------
    transition_matrix : array_like
        Float array of shape (n_states, n_states) whose rows sum to one, giving
        the probability that one segment type follows another.
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).

    Returns
    -------
    crossing_probability : float
        Probability that a trajectory leaving the reactant excursion reaches the
        product state before returning to the reactant state, as a native Python
        float in (0, 1].

    Raises
    ------
    ValueError
        If transition_matrix is not row-stochastic or if the product state is
        unreachable, so the crossing probability is not in (0, 1].
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_crossing_probability(transition_matrix, states) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    matrix = np.asarray(transition_matrix, dtype=float)
    labels = np.asarray(states)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square")
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    if labels.shape[0] != matrix.shape[0]:
        raise ValueError("states and transition_matrix disagree on the state count")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transition_matrix must be finite")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("transition_matrix must be row-stochastic")

    last = int(labels[:, 0].max())
    index = {(int(a), int(b), int(c)): s for s, (a, b, c) in enumerate(labels)}
    reactant = index.get((-1, 1, 1))
    product = index.get((last, 0, 0))
    if reactant is None or product is None:
        raise ValueError("the reactant excursion and product state must both be present")

    # Both boundaries are pinned, so only the remaining states are unknowns.
    free = [s for s in range(matrix.shape[0]) if s not in (reactant, product)]
    system = np.eye(len(free)) - matrix[np.ix_(free, free)]
    forcing = matrix[np.ix_(free, [product])].ravel()
    if abs(np.linalg.det(system)) < 1e-14:
        raise ValueError("the hitting-probability system is singular")
    solution = np.linalg.solve(system, forcing)

    hitting = np.zeros(matrix.shape[0], dtype=float)
    hitting[free] = solution
    hitting[product] = 1.0                    # the reactant entry stays pinned at zero

    # One obligatory step out of the reactant excursion turns the pinned
    # hitting probabilities into the complement of the return probability.
    crossing_probability = float(matrix[reactant] @ hitting)
    if not 0.0 < crossing_probability <= 1.0:
        raise ValueError("the crossing probability must lie in (0, 1]")

    return crossing_probability

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = """import numpy as np

def make_states(n):
    last = n - 1
    rows = [(-1, 1, 1), (0, -1, -1), (0, -1, 1), (0, 1, -1)]
    for i in range(1, last):
        for k in (-1, 1):
            if i == last - 1 and k == 1:
                continue
            for d in (-1, 1):
                rows.append((i, k, d))
    rows.append((last, 0, 0))
    return np.array(rows, dtype=int)

def make_matrix(n, escape, up_below, up_above):
    states = make_states(n)
    last = n - 1
    index = {tuple(int(v) for v in row): s for s, row in enumerate(states)}
    m = np.zeros((len(states), len(states)))
    for state, row in index.items():
        i, _, d = state
        if i == last:
            m[row, row] = 1.0
            continue
        tgt_i = 0 if i == -1 else i + d
        arr = -1 if i == -1 else -d
        if tgt_i == -1:
            m[row, index[(-1, 1, 1)]] = 1.0
            continue
        if tgt_i == last:
            m[row, index[(last, 0, 0)]] = 1.0
            continue
        if tgt_i == 0:
            up = escape if arr == -1 else 0.0
        else:
            up = up_below if arr == -1 else up_above
        for onward, w in ((-1, 1.0 - up), (1, up)):
            tgt = index.get((tgt_i, arr, onward))
            if tgt is not None:
                m[row, tgt] += w
    return states, m
"""
    return [
        # --- Valid: measured interface set, diffusive interior ---
        {
            "setup": fixture + """states, matrix = make_matrix(8, 0.181066666666667, 0.47, 0.59)
""",
            "call": "compute_crossing_probability(matrix, states)",
            "gold_call": "_oracle_compute_crossing_probability(matrix, states)",
        },
        # --- Valid: shorter interface set with a stiffer reactant boundary ---
        {
            "setup": fixture + """states, matrix = make_matrix(5, 0.06, 0.42, 0.63)
""",
            "call": "compute_crossing_probability(matrix, states)",
            "gold_call": "_oracle_compute_crossing_probability(matrix, states)",
        },
        # --- Boundary: every crossing commits, so the probability saturates ---
        {
            "setup": fixture + """states, matrix = make_matrix(6, 1.0, 1.0, 1.0)
""",
            "call": "compute_crossing_probability(matrix, states)",
            "gold_call": "_oracle_compute_crossing_probability(matrix, states)",
        },
        # --- Edge: a strongly recrossing interior, giving a rare-event value ---
        {
            "setup": fixture + """states, matrix = make_matrix(9, 0.004, 0.11, 0.19)
""",
            "call": "compute_crossing_probability(matrix, states)",
            "gold_call": "_oracle_compute_crossing_probability(matrix, states)",
        },
        # --- Invalid: rows that do not sum to one ---
        {
            "setup": fixture + """states, matrix = make_matrix(8, 0.18, 0.47, 0.59)
matrix[5, :] *= 0.5
def run_model():
    try:
        compute_crossing_probability(matrix, states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_crossing_probability(matrix, states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: the product state is unreachable, so no crossing occurs ---
        {
            "setup": fixture + """states, matrix = make_matrix(8, 0.0, 0.47, 0.59)
def run_model():
    try:
        compute_crossing_probability(matrix, states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_crossing_probability(matrix, states)
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
