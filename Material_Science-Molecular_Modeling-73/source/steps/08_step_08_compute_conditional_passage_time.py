"""
Recover the mean length of a full excursion out of the reactant state, which the partial-path sampling never observes directly.

Partial-path sampling deliberately truncates its trajectories, so the mean duration of a complete excursion above the reactant boundary is absent from its output; the sampled lengths in the ensembles above that boundary are far shorter than the excursions they are pieces of, and using them in place of the full length underestimates the timescale drastically. The chain supplies the missing quantity as a mean first passage time. Conditioning on the first step gives a linear system in which the time to reach either boundary equals the time accumulated in the current state plus the average of the same quantity over the states reachable in one step, with both boundary states carrying zero because a walk that starts there has already arrived.




Two details decide whether the answer is the wanted one. First, the excursion of interest must actually leave: a walk that stops the moment it finds itself in the reactant excursion would return zero, so the reactant excursion is a destination for the system but the reported time is built from one obligatory step out of it. Second, the clock must start at the last crossing of the reactant boundary rather than at the moment the trajectory entered the reactant state, so the middle piece of the reactant excursion - the part spent below the boundary before the final upward crossing - is subtracted from the result. Both modifications shift the answer by amounts comparable to the answer itself, so neither is cosmetic.

Formulas:

    T[d] = tau_m2[d] + sum_g M[d, g] T[g]   for d not a boundary

    T[reactant] = T[product] = 0

    T_prime = tau_m2[reactant] + sum_g M[reactant, g] T[g]

    tau_transit = T_prime - tau_middle[reactant]

Returns
-------
float: mean duration of a full excursion above the reactant boundary, in  phase points, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_conditional_passage_time(transition_matrix, states, overlap_free_times,
                                     reactant_middle_time: float) -> float:
    """Return the mean duration of a full excursion above the reactant boundary.

    Parameters
    ----------
    transition_matrix : array_like
        Float array of shape (n_states, n_states) whose rows sum to one, giving
        the probability that one segment type follows another.
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).
    overlap_free_times : array_like
        Float array of shape (n_states,) holding the non-overlapping time each
        state adds to a stitched trajectory, in phase points.
    reactant_middle_time : float
        Mean time the reactant excursion spends below the reactant boundary
        between its first and last crossing of it, in phase points. Must be
        non-negative.

    Returns
    -------
    conditional_passage_time : float
        Mean duration of one excursion that leaves the reactant boundary and
        ends on returning to the reactant state or on reaching the product
        state, in phase points, as a native Python float.

    Raises
    ------
    ValueError
        If overlap_free_times contains a negative entry or if
        reactant_middle_time is at least the reconstructed stopped time.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_conditional_passage_time(transition_matrix, states, overlap_free_times,
                                             reactant_middle_time: float) -> float:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    matrix = np.asarray(transition_matrix, dtype=float)
    labels = np.asarray(states)
    times = np.asarray(overlap_free_times, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition_matrix must be square")
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    if labels.shape[0] != matrix.shape[0] or times.shape != (matrix.shape[0],):
        raise ValueError("states, transition_matrix and overlap_free_times must agree in size")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(times)):
        raise ValueError("transition_matrix and overlap_free_times must be finite")
    if np.any(times < 0.0):
        raise ValueError("overlap_free_times must be non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("transition_matrix must be row-stochastic")
    if isinstance(reactant_middle_time, bool) or not isinstance(
            reactant_middle_time, (int, float, np.integer, np.floating)):
        raise ValueError("reactant_middle_time must be a real number")
    if not np.isfinite(reactant_middle_time) or float(reactant_middle_time) < 0.0:
        raise ValueError("reactant_middle_time must be finite and non-negative")

    last = int(labels[:, 0].max())
    index = {(int(a), int(b), int(c)): s for s, (a, b, c) in enumerate(labels)}
    reactant = index.get((-1, 1, 1))
    product = index.get((last, 0, 0))
    if reactant is None or product is None:
        raise ValueError("the reactant excursion and product state must both be present")

    # Both the return to the reactant state and the arrival in the product
    # state end the excursion, so both are destinations of the system.
    interior = [s for s in range(matrix.shape[0]) if s not in (reactant, product)]
    system = np.eye(len(interior)) - matrix[np.ix_(interior, interior)]
    if abs(np.linalg.det(system)) < 1e-14:
        raise ValueError("the mean first passage system is singular")
    solution = np.linalg.solve(system, times[interior])

    passage = np.zeros(matrix.shape[0], dtype=float)
    passage[interior] = solution

    # One obligatory step out of the reactant excursion, then move the start of
    # the clock to the last crossing of the reactant boundary.
    stopped = float(times[reactant] + matrix[reactant] @ passage)
    conditional_passage_time = stopped - float(reactant_middle_time)
    if conditional_passage_time <= 0.0:
        raise ValueError("the reactant excursion cannot be longer than the stopped time")

    return conditional_passage_time

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

def make_system(n, escape, up_below, up_above, reactant_middle, tilt):
    states = make_states(n)
    last = n - 1
    index = {tuple(int(v) for v in row): s for s, row in enumerate(states)}
    m = np.zeros((len(states), len(states)))
    times = np.zeros(len(states))
    for state, row in index.items():
        i, k, d = state
        if i == last:
            m[row, row] = 1.0
            continue
        if i == -1:
            times[row] = reactant_middle
        elif i == 0:
            times[row] = 0.0 if (k, d) == (1, -1) else 26.8 + 20.5 * (d < 0)
        else:
            times[row] = 20.0 + tilt * i + 3.0 * (d + 1)
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
    return states, m, times
"""
    return [
        # --- Valid: measured interface set (normal scenario) ---
        {
            "setup": fixture + """states, matrix, times = make_system(8, 0.181066666666667, 0.47, 0.59, 43.7, 6.0)
""",
            "call": "compute_conditional_passage_time(matrix, states, times, 43.7)",
            "gold_call": "_oracle_compute_conditional_passage_time(matrix, states, times, 43.7)",
        },
        # --- Valid: shorter interface set with a stiffer reactant boundary ---
        {
            "setup": fixture + """states, matrix, times = make_system(5, 0.06, 0.42, 0.63, 240.0, 9.0)
""",
            "call": "compute_conditional_passage_time(matrix, states, times, 240.0)",
            "gold_call": "_oracle_compute_conditional_passage_time(matrix, states, times, 240.0)",
        },
        # --- Boundary: no time spent below the reactant boundary ---
        {
            "setup": fixture + """states, matrix, times = make_system(6, 0.25, 0.5, 0.6, 0.0, 4.0)
""",
            "call": "compute_conditional_passage_time(matrix, states, times, 0.0)",
            "gold_call": "_oracle_compute_conditional_passage_time(matrix, states, times, 0.0)",
        },
        # --- Edge: heavy recrossing, so the excursion is far longer than any
        #     sampled segment ---
        {
            "setup": fixture + """states, matrix, times = make_system(9, 0.02, 0.21, 0.33, 55.0, 7.5)
""",
            "call": "compute_conditional_passage_time(matrix, states, times, 55.0)",
            "gold_call": "_oracle_compute_conditional_passage_time(matrix, states, times, 55.0)",
        },
        # --- Invalid: the subtracted reactant piece exceeds the stopped time ---
        {
            "setup": fixture + """states, matrix, times = make_system(8, 0.18, 0.47, 0.59, 43.7, 6.0)
def run_model():
    try:
        compute_conditional_passage_time(matrix, states, times, 1.0e9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_conditional_passage_time(matrix, states, times, 1.0e9)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative measured accumulation ---
        {
            "setup": fixture + """states, matrix, times = make_system(8, 0.18, 0.47, 0.59, 43.7, 6.0)
times[11] = -3.0
def run_model():
    try:
        compute_conditional_passage_time(matrix, states, times, 43.7)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_conditional_passage_time(matrix, states, times, 43.7)
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
