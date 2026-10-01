"""
Count how often each segment type is visited on the way from the reactant state to the product state.

A single passage from the reactant state to the product state is, in the chain picture, one walk that starts in the reactant excursion and stops the first time it reaches the product state. Because the interior interfaces are recrossed many times before the walk commits, the walk visits most states repeatedly, and the expected number of those visits is the object that converts per-state measurements into per-passage totals. Deleting the row and column of the absorbing state leaves a sub-stochastic matrix whose powers describe walks that have not yet committed, and summing those powers gives the expected occupation of every state.

Solving a linear system in the transpose is what produces the occupations directly: the expected visit vector is the one that reproduces itself under one step of the chain plus the injection of the single walk that starts in the reactant excursion. The initial visit counts, so the reactant excursion carries an extra unit of occupation that the interior states do not. A useful check falls out for free: the expected number of visits to the reactant excursion is the reciprocal of the probability that a departure from it commits to the product state, because each departure is an independent trial with that success probability. Any wiring error at the boundaries breaks that identity immediately, which makes it a sharper test of the chain than any smooth quantity.

Formulas:

    (I - Q^T) n = e_reactant,  Q the sub-stochastic matrix over transient states

    n[reactant] = 1 / P_A(lambda_B | lambda_A)

Returns
-------
numpy.ndarray of shape (n_states,): expected visits per passage, product state entry zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_visit_counts(transition_matrix, states) -> np.ndarray:
    """Return the expected number of visits to every segment type per passage.

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
    visit_counts : numpy.ndarray
        Float array of shape (n_states,) holding the expected number of visits
        to each state during one passage that starts in the reactant excursion
        and stops on first arrival in the product state. The initial visit is
        counted and the product state entry is zero.

    Raises
    ------
    ValueError
        If the product state is unreachable from the transient states or if
        states and transition_matrix disagree on the state count.
    """
    return np.zeros(0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_visit_counts(transition_matrix, states) -> np.ndarray:
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

    # Everything except the absorbing product state can be revisited.
    transient = [s for s in range(matrix.shape[0]) if s != product]
    sub = matrix[np.ix_(transient, transient)]
    system = np.eye(len(transient)) - sub.T
    if abs(np.linalg.det(system)) < 1e-14:
        raise ValueError("the product state is not reachable from every transient state")

    injection = np.zeros(len(transient), dtype=float)
    injection[transient.index(reactant)] = 1.0
    occupation = np.linalg.solve(system, injection)
    if np.any(occupation < -1e-9):
        raise ValueError("expected visit counts must be non-negative")

    visit_counts = np.zeros(matrix.shape[0], dtype=float)
    visit_counts[transient] = np.maximum(occupation, 0.0)

    return visit_counts

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
        # --- Valid: visits to the reactant excursion on the measured set ---
        {
            "setup": fixture + """states, matrix = make_matrix(8, 0.181066666666667, 0.47, 0.59)
""",
            "call": "float(compute_visit_counts(matrix, states)[0])",
            "gold_call": "float(_oracle_compute_visit_counts(matrix, states)[0])",
        },
        # --- Valid: visits to an ordinary interior segment type ---
        {
            "setup": fixture + """states, matrix = make_matrix(8, 0.181066666666667, 0.47, 0.59)
""",
            "call": "float(compute_visit_counts(matrix, states)[13])",
            "gold_call": "float(_oracle_compute_visit_counts(matrix, states)[13])",
        },
        # --- Boundary: a fully committing chain is traversed once per state ---
        {
            "setup": fixture + """states, matrix = make_matrix(6, 1.0, 1.0, 1.0)
""",
            "call": "float(np.sum(compute_visit_counts(matrix, states)))",
            "gold_call": "float(np.sum(_oracle_compute_visit_counts(matrix, states)))",
        },
        # --- Edge: a strongly recrossing chain, giving large occupations ---
        {
            "setup": fixture + """states, matrix = make_matrix(9, 0.02, 0.21, 0.33)
""",
            "call": "float(np.sum(compute_visit_counts(matrix, states)))",
            "gold_call": "float(np.sum(_oracle_compute_visit_counts(matrix, states)))",
        },
        # --- Invalid: the product state cannot be reached, so no passage ends ---
        {
            "setup": fixture + """states, matrix = make_matrix(8, 0.0, 0.47, 0.59)
def run_model():
    try:
        compute_visit_counts(matrix, states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_visit_counts(matrix, states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: state labels and matrix disagree on the state count ---
        {
            "setup": fixture + """states, matrix = make_matrix(8, 0.18, 0.47, 0.59)
states = states[:-1]
def run_model():
    try:
        compute_visit_counts(matrix, states)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_visit_counts(matrix, states)
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
