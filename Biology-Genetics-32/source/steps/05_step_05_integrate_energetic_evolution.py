"""
Integrate the whole chromatin trajectory over a tabulated loading path by chaining the globally minimizing increment node by node.

Chaining the incremental minimization over a partition produces a piecewise-constant approximation of the evolution whose limit, as the partition is refined, is the solution selected by the global energetic principle. Two properties of the construction matter for what follows. It is unconditionally stable in the energetic sense, because each increment can never do worse than holding the state fixed, so no step size can make the approximation blow up. And it is history-dependent by construction: the objective of each increment is anchored at the state the previous increment returned, which is what encodes memory and makes the response depend on the path of the loading rather than on its instantaneous value. The initial datum is taken as given and not projected, so the trajectory begins exactly where the modeller places it, and the loading is read only at the nodes; whatever the stimulus does strictly between two nodes is invisible to the scheme.

Returns
-------
np.ndarray of shape (n_nodes,), float: the chromatin state at every node of the loading table, starting from the given initial state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def integrate_energetic_evolution(loading: np.ndarray, k: float = 1.0,
                                  a: float = 0.15, rho: float = 0.10,
                                  q_initial: float = -0.15,
                                  b: float = None) -> np.ndarray:
    """Integrate the trajectory selected by the global energetic principle.

    The trajectory starts at q_initial at the first node of the loading table
    and is advanced by one globally minimising increment per subsequent node,
    each increment anchored at the state returned by the previous one.

    Parameters
    ----------
    loading : np.ndarray
        Array of shape (n_nodes, 3) whose third column holds the interaction
        potential at each node; the first two columns are not used here.
    k : float
        Curvature of each well of the landscape, k > 0.
    a : float
        Distance from the barrier to the bottom of the repressed well, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    q_initial : float
        State at the first node, taken as given.
    b : float or None
        Distance from the barrier to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape (n_nodes,) holding the state at every node.

    Raises
    ------
    ValueError
        If loading is not a finite two-dimensional array of shape
        (n_nodes, 3) with at least one node, if q_initial, k, a or rho is not
        a finite number, if k or rho is not strictly greater than zero, if a
        is negative, or if b is neither None nor a finite non-negative
        number.
    """
    return trajectory  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_integrate_energetic_evolution(loading: np.ndarray, k: float = 1.0,
                                          a: float = 0.15, rho: float = 0.10,
                                          q_initial: float = -0.15,
                                          b: float = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(loading, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 1:
        raise ValueError("loading must be a 2D array of shape (n_nodes, 3) with n_nodes >= 1")
    if not np.all(np.isfinite(table)):
        raise ValueError("loading must be finite throughout")
    for name, value in (("q_initial", q_initial), ("k", k), ("a", a), ("rho", rho)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite number")
    if float(k) <= 0.0:
        raise ValueError("k must be strictly greater than zero")
    if float(a) < 0.0:
        raise ValueError("a must be non-negative")
    if float(rho) <= 0.0:
        raise ValueError("rho must be strictly greater than zero")
    if b is not None and not (isinstance(b, (int, float, np.floating, np.integer))
                              and not isinstance(b, bool) and np.isfinite(float(b))
                              and float(b) >= 0.0):
        raise ValueError("b must be None or a finite non-negative number")

    curvature, half_gap, threshold = float(k), float(a), float(rho)
    far_gap = half_gap if b is None else float(b)

    # Continuity of the free energy at the barrier fixes the level of the far
    # well; the near well is the zero of energy.
    level = 0.5 * curvature * (half_gap ** 2 - far_gap ** 2)

    # One increment of the globally minimising scheme. It is written out here
    # rather than imported from sub-problem 03, so that the trajectory this
    # step defines is reproducible from this file alone and cannot be changed
    # by whichever copy of an earlier step happens to be in scope.
    def _advance(previous, load):
        def _objective(state):
            centre = -half_gap if state <= 0.0 else far_gap
            offset = 0.0 if state <= 0.0 else level
            return (0.5 * curvature * (state - centre) ** 2 + offset - state * load
                    + threshold * abs(state - previous))

        knots = sorted(set((0.0, previous)))
        edges = [-np.inf] + knots + [np.inf]
        candidates = list(knots)
        for lower, upper in zip(edges[:-1], edges[1:]):
            if not upper > lower:
                continue
            if np.isneginf(lower):
                probe = upper - 1.0
            elif np.isposinf(upper):
                probe = lower + 1.0
            else:
                probe = 0.5 * (lower + upper)
            centre = -half_gap if probe <= 0.0 else far_gap
            sign = 1.0 if probe > previous else -1.0
            stationary = centre + (load - threshold * sign) / curvature
            if np.isfinite(lower):
                stationary = max(stationary, lower)
            if np.isfinite(upper):
                stationary = min(stationary, upper)
            candidates.append(float(stationary))

        values = [_objective(state) for state in candidates]
        best = min(values)
        tolerance = 1.0e-13 * (1.0 + abs(best))
        tied = [state for state, value in zip(candidates, values) if value <= best + tolerance]
        tied.sort(key=lambda state: (abs(state - previous), state))
        return float(tied[0])

    potential = table[:, 2]
    trajectory = np.empty(potential.size, dtype=float)
    trajectory[0] = float(q_initial)

    # The state is carried forward explicitly: every increment is anchored at
    # the state the previous one returned, which is where the memory lives.
    for index in range(1, potential.size):
        trajectory[index] = _advance(float(trajectory[index - 1]), float(potential[index]))

    return trajectory

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark cycle on a coarse partition ---
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
""",
            "call": "digest(integrate_energetic_evolution(loading, 1.0, 0.15, 0.10, -0.15))",
            "gold_call": "digest(_oracle_integrate_energetic_evolution(loading, 1.0, 0.15, 0.10, -0.15))",
        },
        # --- Valid: a resistant machinery that never lets the mark switch ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
t = np.linspace(0.0, 1.0, 31)
S = 2.5 * (1.0 - np.cos(2.0 * np.pi * t))
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
""",
            "call": "digest(integrate_energetic_evolution(loading, 1.0, 0.15, 0.90, -0.15))",
            "gold_call": "digest(_oracle_integrate_energetic_evolution(loading, 1.0, 0.15, 0.90, -0.15))",
        },
        # --- Valid: a monotonically rising loading with no unloading arm ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
t = np.linspace(0.0, 1.0, 25)
S = 5.0 * t
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
""",
            "call": "digest(integrate_energetic_evolution(loading, 2.0, 0.25, 0.10, -0.25))",
            "gold_call": "digest(_oracle_integrate_energetic_evolution(loading, 2.0, 0.25, 0.10, -0.25))",
        },
        # --- Valid: the asymmetric benchmark landscape over a full cycle ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
t = np.linspace(0.0, 1.0, 61)
S = 2.5 * (1.0 - np.cos(2.0 * np.pi * t))
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
""",
            "call": "digest(integrate_energetic_evolution(loading, 1.07, 0.130, 0.0970, -0.130, 0.075))",
            "gold_call": "digest(_oracle_integrate_energetic_evolution(loading, 1.07, 0.130, 0.0970, -0.130, 0.075))",
        },
        # --- Boundary: a table holding a single node, so nothing is advanced ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
loading = np.array([[0.0, 0.0, 0.0]])
""",
            "call": "digest(integrate_energetic_evolution(loading, 1.0, 0.15, 0.10, -0.15))",
            "gold_call": "digest(_oracle_integrate_energetic_evolution(loading, 1.0, 0.15, 0.10, -0.15))",
        },
        # --- Edge: an initial state placed in the far well instead of the near one ---
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
""",
            "call": "digest(integrate_energetic_evolution(loading, 1.0, 0.15, 0.10, 0.15))",
            "gold_call": "digest(_oracle_integrate_energetic_evolution(loading, 1.0, 0.15, 0.10, 0.15))",
        },
        # --- Invalid: a loading table of the wrong width ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        integrate_energetic_evolution(np.zeros((10, 2)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrate_energetic_evolution(np.zeros((10, 2)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a loading table holding a non-finite entry ---
        {
            "setup": """import numpy as np
bad = np.zeros((5, 3))
bad[2, 2] = np.nan
def run_model():
    try:
        integrate_energetic_evolution(bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrate_energetic_evolution(bad)
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
