"""
Turn the measured three-part segment lengths into the time each state contributes to a stitched trajectory without double counting.

Consecutive sampled segments share a stretch of trajectory, so their lengths cannot simply be added. Splitting each segment at its first and its last crossing of the interface it is centred on isolates the shared stretch: the piece before the first crossing of that interface duplicates trajectory already carried by the preceding segment, while the piece spanning the crossings and the piece after the last crossing are new. Advancing the chain by one state therefore accumulates the middle piece plus the trailing piece and nothing else, and the leading piece is discarded every time. Discarding it is not an approximation; it is what makes the accumulated time of a stitched walk equal the elapsed time of the trajectory it represents.

Some segment types are missing pieces by construction, and the pattern is not symmetric. A reactant excursion begins and ends on the boundary interface, so it is all middle piece. A straddling segment that begins on the boundary interface and ends on the neighbour above never recrosses the boundary, so it is all trailing piece; one that begins and ends on the boundary interface is all middle piece. The straddling segment that comes down from the neighbour above and drops into the reactant state is the awkward one: everything it does happens before it first touches the boundary interface, so its entire duration is leading piece and it contributes exactly nothing to the accumulated time. Crediting it anyway is a silent error that inflates the reactant residence time without changing any probability. Ordinary interior segments generally carry all three pieces.

Formulas:

    tau_m2[s] = tau_middle[s] + tau_trailing[s]

    type index from (k, l):  (0 if k = -1 else 2) + (0 if l = -1 else 1)

Returns
-------
numpy.ndarray of shape (n_states,): non-overlapping accumulated time per state, in phase points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_overlap_free_times(states, reactant_parts, ensemble_parts) -> np.ndarray:
    """Assemble the non-overlapping time contributed by every segment type.

    Parameters
    ----------
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).
    reactant_parts : array_like
        Float array of shape (3,) holding the mean leading, middle and trailing
        piece of the reactant excursion, in phase points.
    ensemble_parts : array_like
        Float array of shape (N, 4, 3). Element [j, t, :] holds the mean
        leading, middle and trailing piece of path type t of the ensemble
        centred on lambda_j, in phase points. Types are ordered as (arrive
        below / depart below, arrive below / depart above, arrive above /
        depart below, arrive above / depart above).

    Returns
    -------
    overlap_free_times : numpy.ndarray
        Float array of shape (n_states,) holding the time each state adds to a
        stitched trajectory, in phase points. The product state contributes
        zero because the measurement stops on arrival there.

    Raises
    ------
    ValueError
        If a measured path piece is negative or if ensemble_parts does not
        supply one entry per ensemble implied by states.
    """
    return np.zeros(0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_overlap_free_times(states, reactant_parts, ensemble_parts) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    labels = np.asarray(states)
    reactant = np.asarray(reactant_parts, dtype=float)
    parts = np.asarray(ensemble_parts, dtype=float)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    if reactant.shape != (3,):
        raise ValueError("reactant_parts must have shape (3,)")
    if parts.ndim != 3 or parts.shape[1:] != (4, 3):
        raise ValueError("ensemble_parts must have shape (n_ensembles, 4, 3)")
    for name, block in (("reactant_parts", reactant), ("ensemble_parts", parts)):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} must be finite")
        if np.any(block < 0.0):
            raise ValueError(f"{name} must be non-negative")

    last = int(labels[:, 0].max())
    if parts.shape[0] != last:
        raise ValueError("ensemble_parts must supply one entry per ensemble lambda_0..lambda_(N-1)")

    overlap_free_times = np.zeros(labels.shape[0], dtype=float)
    for state, (i, k, departure) in enumerate(labels):
        i, k, departure = int(i), int(k), int(departure)
        if i == last:
            continue                       # the product state stops the clock
        if i == -1:
            pieces = reactant
        else:
            row = (0 if k == -1 else 2) + (0 if departure == -1 else 1)
            pieces = parts[i, row]
        # Only the middle and trailing pieces are new trajectory; the leading
        # piece repeats what the previous segment already accounted for.
        overlap_free_times[state] = float(pieces[1] + pieces[2])

    return overlap_free_times

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

def make_parts(n, spread):
    last = n - 1
    parts = np.zeros((last, 4, 3))
    parts[0] = np.array([[0.0, 47.3, 0.0], [0.0, 0.0, 26.8],
                         [24.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for j in range(1, last):
        base = 18.0 + spread * j
        parts[j] = np.array([[base, base * 1.7, base * 0.98],
                             [base * 1.01, base * 1.2, base * 1.31],
                             [base * 1.16, base * 1.3, base * 0.99],
                             [base * 1.18, base * 1.9, base * 1.34]])
    return parts
"""
    return [
        # --- Valid: an ordinary interior segment carrying all three pieces ---
        {
            "setup": fixture + """states = make_states(8)
reactant = np.array([0.0, 43.7, 0.0])
parts = make_parts(8, 4.5)
""",
            "call": "float(compute_overlap_free_times(states, reactant, parts)[9])",
            "gold_call": "float(_oracle_compute_overlap_free_times(states, reactant, parts)[9])",
        },
        # --- Valid: total accumulated weight of the whole state space ---
        {
            "setup": fixture + """states = make_states(8)
reactant = np.array([0.0, 43.7, 0.0])
parts = make_parts(8, 4.5)
""",
            "call": "float(np.sum(compute_overlap_free_times(states, reactant, parts)))",
            "gold_call": "float(np.sum(_oracle_compute_overlap_free_times(states, reactant, parts)))",
        },
        # --- Boundary: the straddling type whose whole duration precedes its
        #     first boundary crossing, plus the reactant excursion ---
        {
            "setup": fixture + """states = make_states(8)
reactant = np.array([0.0, 43.7, 0.0])
parts = make_parts(8, 4.5)
""",
            "call": "float(compute_overlap_free_times(states, reactant, parts)[3] * 1000.0 + compute_overlap_free_times(states, reactant, parts)[0])",
            "gold_call": "float(_oracle_compute_overlap_free_times(states, reactant, parts)[3] * 1000.0 + _oracle_compute_overlap_free_times(states, reactant, parts)[0])",
        },
        # --- Edge: smallest interface set, and the absorbing product state ---
        {
            "setup": fixture + """states = make_states(4)
reactant = np.array([0.0, 7.25, 0.0])
parts = make_parts(4, 11.0)
""",
            "call": "float(np.sum(compute_overlap_free_times(states, reactant, parts)) + 10.0 * compute_overlap_free_times(states, reactant, parts)[-1])",
            "gold_call": "float(np.sum(_oracle_compute_overlap_free_times(states, reactant, parts)) + 10.0 * _oracle_compute_overlap_free_times(states, reactant, parts)[-1])",
        },
        # --- Invalid: a negative measured piece ---
        {
            "setup": fixture + """states = make_states(8)
reactant = np.array([0.0, 43.7, 0.0])
parts = make_parts(8, 4.5)
parts[3, 1, 2] = -1.0
def run_model():
    try:
        compute_overlap_free_times(states, reactant, parts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_overlap_free_times(states, reactant, parts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a parts table sized for the wrong interface count ---
        {
            "setup": fixture + """states = make_states(8)
reactant = np.array([0.0, 43.7, 0.0])
parts = make_parts(5, 4.5)
def run_model():
    try:
        compute_overlap_free_times(states, reactant, parts)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_overlap_free_times(states, reactant, parts)
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
