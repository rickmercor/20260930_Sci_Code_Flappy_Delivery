"""
Wire the enumerated segment types into the stochastic matrix of the chain, including the two boundary conventions.

Consecutive segments of a long trajectory overlap: the segment that follows one centred on a given interface is centred on the neighbour that the first one departed towards, and it necessarily arrived from the side the first one came from. That geometric fact fixes the sparsity pattern completely. A state may only hand over to the two states of the destination ensemble that share the implied arrival side, so every row carries exactly two non-zero entries, and the numbers filling them are the local crossing probabilities of the destination ensemble, not of the ensemble being left. Reading them off the wrong ensemble is the single easiest way to build a matrix that is stochastic, plausible and wrong.

The two ends need their own conventions. A segment that departs below the first interface enters the reactant state, so it is followed with certainty by a reactant excursion, and a reactant excursion is followed by a segment of the straddling ensemble that arrived from below. A segment that departs above the last interface enters the product state, so it is followed with certainty by the product state; for a measurement that stops at the first arrival in the product state, that state absorbs and its own outgoing statistics are never consulted. The one departure combination that the straddling ensemble cannot supply carries zero probability, so a state that would hand over to it must instead hand over with certainty to the single surviving alternative.

Formulas:

    state (i, k, l)  ->  states (i + l, -l, +-1)

    transition probability = p[i + l, -l, +-1]

Returns
-------
numpy.ndarray of shape (n_states, n_states): the row-stochastic transition matrix of the chain, with the product state absorbing.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_transition_matrix(states, probabilities) -> np.ndarray:
    """Assemble the stochastic matrix of the partial-path chain.

    Parameters
    ----------
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).
    probabilities : array_like
        Float array of shape (N, 2, 2). Element [j, a, b] is the probability
        that a segment of the ensemble centred on lambda_j which arrived from
        the side indexed by a departs towards the side indexed by b, index 0
        meaning below and index 1 meaning above.

    Returns
    -------
    transition_matrix : numpy.ndarray
        Float array of shape (n_states, n_states) whose rows sum to one. Entry
        [s, t] is the probability that the segment following state s is of type
        t. The product state is absorbing.

    Raises
    ------
    ValueError
        If the probability table does not match the interface count implied by
        states or assigns non-zero probability to a segment type absent from
        the straddling ensemble.
    """
    return np.zeros((0, 0))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_transition_matrix(states, probabilities) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    labels = np.asarray(states)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 3 or probs.shape[1:] != (2, 2):
        raise ValueError("probabilities must have shape (n_ensembles, 2, 2)")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probabilities must be finite and lie in [0, 1]")

    n_states = labels.shape[0]
    last = int(labels[:, 0].max())
    if probs.shape[0] != last:
        raise ValueError("probabilities must supply one entry per ensemble lambda_0..lambda_(N-1)")

    index = {(int(a), int(b), int(c)): s for s, (a, b, c) in enumerate(labels)}
    reactant = index.get((-1, 1, 1))
    product = index.get((last, 0, 0))
    if reactant is None or product is None:
        raise ValueError("the reactant excursion and product state must both be present")

    matrix = np.zeros((n_states, n_states), dtype=float)
    for state, row in index.items():
        i, _, departure = state
        if i == last:
            matrix[row, row] = 1.0            # the measurement stops here
            continue
        # A reactant excursion always resumes in the straddling ensemble,
        # arriving from below; every other state steps to the neighbour it
        # departed towards, arriving from the side it just left.
        target_i = 0 if i == -1 else i + departure
        arrival = -1 if i == -1 else -departure
        if target_i == -1:
            matrix[row, reactant] = 1.0
            continue
        if target_i == last:
            matrix[row, product] = 1.0
            continue
        for onward, column in ((-1, 0), (1, 1)):
            weight = float(probs[target_i, 0 if arrival == -1 else 1, column])
            target = index.get((target_i, arrival, onward))
            if target is None:
                if weight > 0.0:
                    raise ValueError("non-zero probability into a segment type that does not exist")
                continue
            matrix[row, target] += weight

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise ValueError("the assembled matrix is not row-stochastic")

    return matrix

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

def make_probs(n, seed):
    last = n - 1
    p = np.zeros((last, 2, 2))
    p[0, 0, 1] = 0.181066666666667
    p[0, 0, 0] = 1.0 - p[0, 0, 1]
    p[0, 1, 0] = 1.0
    for j in range(1, last):
        up_below = 0.40 + 0.03 * j + 0.01 * seed
        up_above = 0.60 + 0.02 * j + 0.01 * seed
        p[j, 0, 1] = up_below
        p[j, 0, 0] = 1.0 - up_below
        p[j, 1, 1] = up_above
        p[j, 1, 0] = 1.0 - up_above
    return p
"""
    return [
        # --- Valid: a progression entry of an ordinary interior ensemble ---
        {
            "setup": fixture + """states = make_states(8)
probs = make_probs(8, 0)
""",
            "call": "float(build_transition_matrix(states, probs)[9, 13])",
            "gold_call": "float(_oracle_build_transition_matrix(states, probs)[9, 13])",
        },
        # --- Valid: whole-matrix fingerprint on the paper-sized interface set ---
        {
            "setup": fixture + """states = make_states(5)
probs = make_probs(5, 2)

def fingerprint(m):
    w = np.arange(1, m.shape[0] + 1, dtype=float)
    return float(w @ m @ w)
""",
            "call": "fingerprint(build_transition_matrix(states, probs))",
            "gold_call": "fingerprint(_oracle_build_transition_matrix(states, probs))",
        },
        # --- Boundary: the reactant excursion is re-entered with certainty
        #     from both straddling types that fall back into it ---
        {
            "setup": fixture + """states = make_states(8)
probs = make_probs(8, 1)
""",
            "call": "float(build_transition_matrix(states, probs)[1, 0] + build_transition_matrix(states, probs)[3, 0])",
            "gold_call": "float(_oracle_build_transition_matrix(states, probs)[1, 0] + _oracle_build_transition_matrix(states, probs)[3, 0])",
        },
        # --- Edge: the product state absorbs, so the trace counts exactly it ---
        {
            "setup": fixture + """states = make_states(6)
probs = make_probs(6, 3)
""",
            "call": "float(np.trace(build_transition_matrix(states, probs)))",
            "gold_call": "float(np.trace(_oracle_build_transition_matrix(states, probs)))",
        },
        # --- Invalid: probability table sized for the wrong interface count ---
        {
            "setup": fixture + """states = make_states(8)
probs = make_probs(6, 0)
def run_model():
    try:
        build_transition_matrix(states, probs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_transition_matrix(states, probs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: the straddling ensemble is given a departure population
        #     it cannot physically have ---
        {
            "setup": fixture + """states = make_states(8)
probs = make_probs(8, 0)
probs[0, 1, 0] = 0.7
probs[0, 1, 1] = 0.3
def run_model():
    try:
        build_transition_matrix(states, probs)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_transition_matrix(states, probs)
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
