"""
Enumerate the segment types that a long trajectory visits and fix the ordering used by every array downstream.

Viewing a long trajectory as a chain over sampled segments requires deciding what a chain state is. A segment is pinned down by three labels - the interface it is centred on, the neighbour it arrived from and the neighbour it departed to - so a generic interior interface contributes four states. Using three labels rather than the two that a description based on the last pair of interfaces visited would need is exactly what preserves the arrival-side memory of the sampling, and it is what makes the chain faithful to ensembles whose sampling weights depend on the segment type.

Both ends of the interface set break the pattern, and the two ends break it differently. At the reactant end there is an extra state for the excursions that drop below the first interface and come back, while the ensemble straddling that interface carries only three types, because a segment that arrived from the interface above cannot depart towards it without first committing to the reactant state. At the product end the segments that begin beyond the last interface are absorbed into a single product state, which removes the two types of the outermost ordinary ensemble that would have started there; those two are not merely rare, they are unreachable from inside the reactant basin. Counting one reactant excursion, three straddling types, four types for each ordinary interior ensemble, two surviving types at the outermost ensemble and one product state gives the total, and a fixed flattening of that list is what lets the transition matrix, the accumulated times and the boundary conditions be written as plain arrays.


Formulas:

    n_states = 4 * N - 1        for interfaces lambda_0 ... lambda_N

    order: reactant excursion, then the straddling ensemble, then the ordinary

    ensembles in increasing interface index, then the product state

Returns
-------
numpy.ndarray of shape (n_states, 3), dtype int: the flattened state space.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_state_space(n_interfaces: int) -> np.ndarray:
    """Enumerate the segment types of the partial-path chain.

    Parameters
    ----------
    n_interfaces : int
        Total number of interfaces lambda_0 ... lambda_N, so that
        N = n_interfaces - 1. Must be at least 4.

    Returns
    -------
    states : numpy.ndarray
        Integer array of shape (n_states, 3). Row s holds the labels
        (i, k, l) of state s: i is the index of the interface the segment is
        centred on, k is -1 if the segment arrived from below and +1 if it
        arrived from above, and l is -1 if it departed below and +1 if it
        departed above. The reactant excursion is labelled (-1, +1, +1) and the
        product state is labelled (N, 0, 0).

    Raises
    ------
    ValueError
        If n_interfaces is not an integer or is smaller than 4.
    """
    return np.zeros((0, 3), dtype=int)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_state_space(n_interfaces: int) -> np.ndarray:
    # Local import: the harness compiles this oracle on its own, so the module-level import is not in scope here.
    import numpy as np

    if isinstance(n_interfaces, bool) or not isinstance(n_interfaces, (int, np.integer)):
        raise ValueError("n_interfaces must be an integer")
    if int(n_interfaces) < 4:
        raise ValueError("n_interfaces must be at least 4")

    last = int(n_interfaces) - 1
    rows = []

    # The excursions that leave the first interface into the reactant state and come back. They start and end on the same side, hence the (+1, +1) label.
    rows.append((-1, 1, 1))

    # The ensemble straddling the first interface. A segment arriving from the interface above and departing towards it would have to touch the first interface without entering the reactant state, so that type is absent.
    rows.append((0, -1, -1))
    rows.append((0, -1, 1))
    rows.append((0, 1, -1))

    for i in range(1, last):
        for k in (-1, 1):
            # Segments of the outermost ordinary ensemble that arrive from above start beyond the last interface, i.e. inside the product state, and are collected there instead.
            if i == last - 1 and k == 1:
                continue
            for departure in (-1, 1):
                rows.append((i, k, departure))

    # Every segment that begins in the product state is one state.
    rows.append((last, 0, 0))

    states = np.array(rows, dtype=int)
    if states.shape[0] != 4 * last - 1:
        raise ValueError("state enumeration is inconsistent with the interface count")

    return states

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: size of the state space of the measured interface set ---
        {
            "setup": """import numpy as np
n_interfaces = 8
""",
            "call": "float(build_state_space(n_interfaces).shape[0])",
            "gold_call": "float(_oracle_build_state_space(n_interfaces).shape[0])",
        },
        # --- Valid: labels of an ordinary interior segment type ---
        {
            "setup": """import numpy as np
n_interfaces = 8
""",
            "call": "float(np.dot(build_state_space(n_interfaces)[9], np.array([100.0, 10.0, 1.0])))",
            "gold_call": "float(np.dot(_oracle_build_state_space(n_interfaces)[9], np.array([100.0, 10.0, 1.0])))",
        },
        # --- Boundary: the two ends of the flattened list ---
        {
            "setup": """import numpy as np
n_interfaces = 8
""",
            "call": "float(build_state_space(n_interfaces)[0].sum() + build_state_space(n_interfaces)[-1].sum())",
            "gold_call": "float(_oracle_build_state_space(n_interfaces)[0].sum() + _oracle_build_state_space(n_interfaces)[-1].sum())",
        },
        # --- Edge: smallest admissible interface set, whole table fingerprinted ---
        {
            "setup": """import numpy as np
n_interfaces = 4

def fingerprint(table):
    weights = np.arange(1, table.shape[0] + 1, dtype=float)
    return float(np.dot(weights, table @ np.array([100.0, 10.0, 1.0])))
""",
            "call": "fingerprint(build_state_space(n_interfaces))",
            "gold_call": "fingerprint(_oracle_build_state_space(n_interfaces))",
        },
        # --- Invalid: too few interfaces to define an interior ensemble ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_state_space(3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_state_space(3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-integer interface count ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_state_space(8.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_state_space(8.0)
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
