"""
Attribute the accumulated time of one passage to the interface each contribution was measured on.

Once the expected occupation of every segment type is known, the mean time a single passage spends in each type is that occupation multiplied by the non-overlapping time the type contributes. Summing those products over the types that share a middle interface gives a profile of accumulated time along the order parameter, which is where the reconstruction stops being a rate calculation and becomes a statement about mechanism: it says where the elapsed time of the transition actually goes. Nothing in the underlying formalism is changed by the grouping, but the grouping is what exposes the region that limits the transition, and it is not read off any single sampled ensemble because it mixes occupation, which is global, with duration, which is local.

The attribution must respect the two boundary conventions. The reactant excursion and every segment of the straddling ensemble are centred on the reactant interface and therefore belong to the same group, even though they come from two different sampled ensembles; keeping them apart would split the residence time of the reactant basin across two entries and misstate both. The product state closes the passage and contributes nothing, so no group is opened for the last interface. Because the leading pieces were already discarded, the profile sums exactly to the mean duration of one passage, which is the check that the attribution has neither lost nor duplicated time.

Formulas:

    dwell[j] = sum over states centred on lambda_j of n[state] * tau_m2[state]

    states with i <= 0 are grouped under lambda_0

    sum_j dwell[j] = mean duration of one passage

Returns
-------
numpy.ndarray of shape (N,): accumulated time per passage grouped by middle  interface, in phase points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_interface_dwell_times(states, visit_counts, overlap_free_times) -> np.ndarray:
    """Group the accumulated time of one passage by middle interface.

    Parameters
    ----------
    states : array_like
        Integer array of shape (n_states, 3) holding the (i, k, l) labels of
        every segment type, with the reactant excursion labelled (-1, +1, +1)
        and the product state labelled (N, 0, 0).
    visit_counts : array_like
        Float array of shape (n_states,) holding the expected number of visits
        to each state during one passage. Entries must be non-negative.
    overlap_free_times : array_like
        Float array of shape (n_states,) holding the non-overlapping time each
        state adds to a stitched trajectory, in phase points. Entries must be
        non-negative.

    Returns
    -------
    dwell_times : numpy.ndarray
        Float array of shape (N,) holding the mean time one passage accumulates
        in segments centred on lambda_0 ... lambda_(N-1), in phase points.

    Raises
    ------
    ValueError
        If visit_counts contains a negative entry or does not have one entry
        per state.
    """
    return np.zeros(0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_interface_dwell_times(states, visit_counts, overlap_free_times) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the
    # module-level import is not in scope here.
    import numpy as np

    labels = np.asarray(states)
    visits = np.asarray(visit_counts, dtype=float)
    times = np.asarray(overlap_free_times, dtype=float)
    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("states must have shape (n_states, 3)")
    if visits.shape != (labels.shape[0],):
        raise ValueError("visit_counts must have one entry per state")
    if times.shape != (labels.shape[0],):
        raise ValueError("overlap_free_times must have one entry per state")
    for name, block in (("visit_counts", visits), ("overlap_free_times", times)):
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{name} must be finite")
        if np.any(block < 0.0):
            raise ValueError(f"{name} must be non-negative")

    last = int(labels[:, 0].max())
    dwell_times = np.zeros(last, dtype=float)
    for state, (i, _, _) in enumerate(labels):
        i = int(i)
        if i == last:
            continue                # the product state closes the passage
        # The reactant excursion and the straddling ensemble are both centred
        # on the reactant interface and form a single group.
        dwell_times[max(i, 0)] += visits[state] * times[state]

    if not np.all(np.isfinite(dwell_times)):
        raise ValueError("the dwell profile is not finite")
    if dwell_times.sum() <= 0.0:
        raise ValueError("a passage must accumulate a positive time")

    return dwell_times

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

def make_profiles(n, tilt):
    states = make_states(n)
    last = n - 1
    visits = np.zeros(states.shape[0])
    times = np.zeros(states.shape[0])
    for s, (i, k, d) in enumerate(states):
        i, k, d = int(i), int(k), int(d)
        if i == last:
            continue
        depth = 0 if i < 0 else i
        visits[s] = 30.0 / (1.0 + depth) + 0.5 * (k + 1) + 0.25 * (d + 1)
        times[s] = 20.0 + tilt * depth + 3.0 * (d + 1)
    times[3] = 0.0
    return states, visits, times
"""
    return [
        # --- Valid: accumulated time at the reactant interface ---
        {
            "setup": fixture + """states, visits, times = make_profiles(8, 6.0)
""",
            "call": "float(compute_interface_dwell_times(states, visits, times)[0])",
            "gold_call": "float(_oracle_compute_interface_dwell_times(states, visits, times)[0])",
        },
        # --- Valid: accumulated time at an interior interface ---
        {
            "setup": fixture + """states, visits, times = make_profiles(8, 6.0)
""",
            "call": "float(compute_interface_dwell_times(states, visits, times)[2])",
            "gold_call": "float(_oracle_compute_interface_dwell_times(states, visits, times)[2])",
        },
        # --- Boundary: the profile sums to the whole duration of a passage ---
        {
            "setup": fixture + """states, visits, times = make_profiles(8, 6.0)
""",
            "call": "float(np.sum(compute_interface_dwell_times(states, visits, times)))",
            "gold_call": "float(np.sum(_oracle_compute_interface_dwell_times(states, visits, times)))",
        },
        # --- Edge: smallest interface set, profile fingerprinted ---
        {
            "setup": fixture + """states, visits, times = make_profiles(4, 15.0)

def fingerprint(profile):
    return float(np.dot(np.arange(1, profile.size + 1, dtype=float), profile))
""",
            "call": "fingerprint(compute_interface_dwell_times(states, visits, times))",
            "gold_call": "fingerprint(_oracle_compute_interface_dwell_times(states, visits, times))",
        },
        # --- Invalid: a negative occupation ---
        {
            "setup": fixture + """states, visits, times = make_profiles(8, 6.0)
visits[7] = -2.0
def run_model():
    try:
        compute_interface_dwell_times(states, visits, times)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_interface_dwell_times(states, visits, times)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: occupation vector of the wrong length ---
        {
            "setup": fixture + """states, visits, times = make_profiles(8, 6.0)
visits = visits[:-2]
def run_model():
    try:
        compute_interface_dwell_times(states, visits, times)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_interface_dwell_times(states, visits, times)
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
