"""
Reduce a discrete trajectory over a loading path to the small set of quantities an experiment on the mark could actually report.

What survives a cycle of the micro-environment, rather than what happens during it, is what an assay of an epigenetic mark measures. The peak level records how far the mark was driven at the height of the stimulus; the residual level records how much remains at the final node, and a residual differing from the starting level is the operational definition of memory. The irreversible cost is the one-homogeneous dissipation potential accumulated along the trajectory. The work supplied by the changing interaction potential is a distinct signed quantity. For the stored energy E(q, ell) = F(q) - q*ell, it is minus the integral of q with respect to ell along the nodal loading path. When the interaction potential completes a closed cycle, this signed line integral is the oriented hysteresis-loop area in the (ell, q) plane. It does not in general equal the accumulated dissipation when the stored energy changes between the initial and final states; instead the energy balance relates supplied work, stored-energy change and dissipation.

Returns
-------
np.ndarray of shape (6,), float: the peak state, the residual state, the accumulated dissipation, the signed supplied work computed as minus the trapezoidal integral of state with respect to interaction potential along the nodal path, the crossing interaction potential and the crossing time, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def summarize_cycle(loading: np.ndarray, trajectory: np.ndarray,
                    rho: float = 0.10) -> np.ndarray:
    """Reduce one cycle of the trajectory to its reportable quantities.

    A crossing of the barrier is detected at the first node whose state is
    strictly positive; the two crossing entries are set to -1 when the state
    never becomes positive, a value the protocol cannot otherwise produce
    since both the time and the interaction potential are non-negative.

    Parameters
    ----------
    loading : np.ndarray
        Array of shape (n_nodes, 3) holding the nodal time, the stimulus and
        the interaction potential; the stimulus column is not used here.
    trajectory : np.ndarray
        Array of shape (n_nodes,) holding the state at each node.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.

    Returns
    -------
    summary : np.ndarray
        Array of shape (6,) holding, in order, the peak state over the path,
        the residual state at the last node, the accumulated dissipation, the
        signed work supplied by the changing interaction potential, computed
        as minus the trapezoidal integral of state with respect to interaction
        potential along the nodal path, the interaction potential at the first
        node beyond the barrier and the time at that node. For a closed cycle
        in interaction potential, the fourth entry is the oriented hysteresis-
        loop area; it is not absolute-valued.

    Raises
    ------
    ValueError
        If loading is not a two-dimensional array of shape (n_nodes, 3) with
        at least two nodes, if trajectory does not hold one state per node of
        that table, if either is not finite throughout, or if rho is not a
        finite number strictly greater than zero.
    """
    return summary  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_summarize_cycle(loading: np.ndarray, trajectory: np.ndarray,
                            rho: float = 0.10) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(loading, dtype=float)
    states = np.asarray(trajectory, dtype=float).ravel()
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 2:
        raise ValueError("loading must be a 2D array of shape (n_nodes, 3) with n_nodes >= 2")
    if states.size != table.shape[0]:
        raise ValueError("trajectory must hold one state per node of the loading table")
    if not (np.all(np.isfinite(table)) and np.all(np.isfinite(states))):
        raise ValueError("loading and trajectory must be finite throughout")
    if not (isinstance(rho, (int, float, np.floating, np.integer))
            and np.isfinite(float(rho)) and float(rho) > 0.0):
        raise ValueError("rho must be a finite number strictly greater than zero")

    time = table[:, 0]
    potential = table[:, 2]
    threshold = float(rho)

    peak = float(np.max(states))
    residual = float(states[-1])

    # A one-homogeneous dissipation potential accumulates as the total
    # variation of the trajectory, weighted by the threshold.
    dissipated = threshold * float(np.sum(np.abs(np.diff(states))))

    # The work the environment supplies is minus the integral of the state
    # against the interaction potential, which for a closed cycle in the
    # potential is the area the trajectory encloses in that plane.
    area = float(-np.trapezoid(states, potential)) if hasattr(np, "trapezoid") \
        else float(-np.trapz(states, potential))

    crossings = np.flatnonzero(states > 0.0)
    if crossings.size == 0:
        crossing_potential, crossing_time = -1.0, -1.0
    else:
        index = int(crossings[0])
        crossing_potential, crossing_time = float(potential[index]), float(time[index])

    return np.array([peak, residual, dissipated, area,
                     crossing_potential, crossing_time], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a bistable response that crosses the barrier and recovers ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
t = np.linspace(0.0, 1.0, 51)
S = 2.5 * (1.0 - np.cos(2.0 * np.pi * t))
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
ell = loading[:, 2]
peak = 0.15 + max(ell.max() - 0.10, 0.0)
q = np.where(ell <= 0.10, -0.15, 0.15 + ell - 0.10)
rising = np.arange(ell.size) <= int(np.argmax(ell))
q = np.where(rising, q, np.minimum(peak, 0.15 + ell + 0.10))
""",
            "call": "digest(summarize_cycle(loading, q, 0.10))",
            "gold_call": "digest(_oracle_summarize_cycle(loading, q, 0.10))",
        },
        # --- Valid: a mark that never crosses the barrier ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
t = np.linspace(0.0, 1.0, 41)
S = 2.5 * (1.0 - np.cos(2.0 * np.pi * t))
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
q = np.full(loading.shape[0], -0.15)
""",
            "call": "digest(summarize_cycle(loading, q, 0.10))",
            "gold_call": "digest(_oracle_summarize_cycle(loading, q, 0.10))",
        },
        # --- Valid: a monotone response with no unloading arm ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
t = np.linspace(0.0, 1.0, 33)
S = 5.0 * t
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
q = np.where(loading[:, 2] <= 0.10, -0.15, 0.15 + loading[:, 2] - 0.10)
""",
            "call": "digest(summarize_cycle(loading, q, 0.10))",
            "gold_call": "digest(_oracle_summarize_cycle(loading, q, 0.10))",
        },
        # --- Boundary: a state that reaches the barrier without passing it ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
t = np.linspace(0.0, 1.0, 21)
S = 2.5 * (1.0 - np.cos(2.0 * np.pi * t))
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
q = np.minimum(0.0, -0.15 + np.maximum(0.0, loading[:, 2] - 0.10))
""",
            "call": "digest(summarize_cycle(loading, q, 0.10))",
            "gold_call": "digest(_oracle_summarize_cycle(loading, q, 0.10))",
        },
        # --- Boundary: the shortest admissible table, a single increment ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
loading = np.array([[0.0, 0.0, 0.0], [0.5, 2.5, 0.45]])
q = np.array([-0.15, 0.5])
""",
            "call": "digest(summarize_cycle(loading, q, 0.10))",
            "gold_call": "digest(_oracle_summarize_cycle(loading, q, 0.10))",
        },
        # --- Invalid: a trajectory of the wrong length ---
        {
            "setup": """import numpy as np
loading = np.zeros((6, 3))
def run_model():
    try:
        summarize_cycle(loading, np.zeros(4))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_summarize_cycle(loading, np.zeros(4))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a machinery with no resistance to remodelling ---
        {
            "setup": """import numpy as np
loading = np.zeros((4, 3))
def run_model():
    try:
        summarize_cycle(loading, np.zeros(4), 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_summarize_cycle(loading, np.zeros(4), 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a trajectory holding a non-finite state ---
        {
            "setup": """import numpy as np
loading = np.zeros((4, 3))
bad = np.zeros(4)
bad[2] = np.inf
def run_model():
    try:
        summarize_cycle(loading, bad, 0.10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_summarize_cycle(loading, bad, 0.10)
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
