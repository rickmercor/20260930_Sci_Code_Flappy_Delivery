"""
Follow the state along the branch of equilibria of the initially occupied well alone, node by node, until that branch stops admitting a state inside its own well.

The alternative to comparing the occupied basin against every other configuration is to refuse the comparison altogether and simply keep satisfying the flow rule on the branch the state is already on. That branch is the family of equilibria of one well, parametrised by the loading, and the state follows it as far as it goes: the well holds the mark while the drive stays inside the elastic range, and pushes it along its own branch once the drive saturates that range. What ends the branch is not a comparison with the far well but the exhaustion of the near one, when the branch equilibrium reaches the barrier and there is no longer any state of that well for the loading to select. Tracking this branch is the natural reading of a locally selected, viscosity-limited evolution, and it is what an experiment probing the occupied phenotype alone would follow.

Returns
-------
np.ndarray of shape (n_nodes,), float: the state at every node, following the branch of the initially occupied well up to and including the first node whose increment carries it beyond the barrier, at which node it is displaced by a + b onto the branch of the far well, and following that branch through every later node.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def track_local_branch(loading: np.ndarray, k: float = 1.0, a: float = 0.15,
                       rho: float = 0.10, q_initial: float = -0.15,
                       b: float = None) -> np.ndarray:
    """Follow the occupied branch, cross when it is exhausted, then follow the far one.

    The initial state must lie in the well centred at -a, that is q_initial
    must not be positive. The branch force of the occupied well is
    k * (q + a), and the state is advanced along it by one predictor-corrector
    increment per node. The branch is exhausted at the first node whose
    increment returns a state beyond the barrier. At that node the state is
    carried across to the far well by a passage that leaves its displacement
    from the well bottom unchanged, so the state recorded at that node is the
    exhausting one moved by exactly a + b. Every later node is advanced along
    the branch of the far well, whose branch force is k * (q - b), by the same
    increment.

    Parameters
    ----------
    loading : np.ndarray
        Array of shape (n_nodes, 3) whose third column holds the interaction
        potential at each node; the first two columns are not used here.
    k : float
        Curvature of the well, k > 0.
    a : float
        Displacement of the well bottom from the barrier, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    q_initial : float
        State at the first node; must satisfy q_initial <= 0.
    b : float or None
        Displacement of the far well bottom from the barrier, b >= 0. None
        takes the two wells to be symmetric, b = a.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape (n_nodes,) holding the state at every node, on the
        occupied branch up to the node that exhausts it and on the far branch
        from that node onwards.

    Raises
    ------
    ValueError
        If loading is not a finite two-dimensional array of shape
        (n_nodes, 3) with at least one node, if k or rho is not a finite
        number strictly greater than zero, if a is not finite and
        non-negative, if b is neither None nor a finite non-negative number,
        or if q_initial is not a finite number no greater than zero.
    """
    return trajectory  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================

import numpy as np
def _oracle_track_local_branch(loading: np.ndarray, k: float = 1.0, a: float = 0.15,
                               rho: float = 0.10, q_initial: float = -0.15,
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
    if float(q_initial) > 0.0:
        raise ValueError("q_initial must lie in the well centred at -a, so it must not be positive")
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

    linear, shift, threshold = float(k), float(a), float(rho)
    far = shift if b is None else float(b)
    tolerance, budget = 1.0e-13, 100

    # One return-map increment on the branch of the occupied well, at the
    # vanishing anharmonicity this step calls for. It is written out here
    # rather than imported from sub-problem 04, so that the branch this step
    # defines is reproducible from this file alone and cannot be changed by
    # whichever copy of an earlier step happens to be in scope.
    def _return_map(previous, load, centre):
        def _force(state):
            return linear * (state - centre)

        # Elastic predictor: the drive the frozen state feels under the new load.
        trial = load - _force(previous)
        if abs(trial) <= threshold:
            return float(previous)

        # Corrector: place the drive exactly on the boundary of the elastic
        # range, on the side the violation points to.
        direction = 1.0 if trial > 0.0 else -1.0
        target = load - direction * threshold

        lower, upper = previous, previous
        span = max(1.0, abs(previous))
        for _ in range(200):
            if direction > 0.0:
                upper = previous + span
                if _force(upper) >= target:
                    break
            else:
                lower = previous - span
                if _force(lower) <= target:
                    break
            span *= 2.0
        else:
            raise ValueError("the branch force could not be bracketed around the target")

        # Safeguarded Newton: take the Newton step when it stays inside the
        # bracket and makes progress, and bisect otherwise.
        state = 0.5 * (lower + upper)
        for _ in range(budget):
            residual = _force(state) - target
            if abs(residual) <= tolerance:
                break
            if residual > 0.0:
                upper = state
            else:
                lower = state
            step = state - residual / linear
            if not (lower < step < upper):
                step = 0.5 * (lower + upper)
            if abs(step - state) <= tolerance * (1.0 + abs(state)):
                state = step
                break
            state = step

        return float(state)

    potential = table[:, 2]
    trajectory = np.empty(potential.size, dtype=float)
    trajectory[0] = float(q_initial)
    crossed = float(q_initial) > 0.0

    for index in range(1, potential.size):
        if crossed:
            # The far branch holds the state for the rest of the cycle.
            trajectory[index] = _return_map(float(trajectory[index - 1]),
                                            float(potential[index]), far)
            continue
        state = _return_map(float(trajectory[index - 1]), float(potential[index]),
                            -shift)
        if state > 0.0:
            # The occupied branch is exhausted. The viscous transient that
            # carries the state across the barrier runs at frozen loading and
            # leaves the displacement from the well bottom unchanged, so the
            # state arrives on the far branch a + b further along.
            state += shift + far
            crossed = True
        trajectory[index] = state

    return trajectory

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================


import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark cycle, whose loading exhausts the branch ---
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
            "call": "digest(track_local_branch(loading, 1.0, 0.15, 0.10, -0.15))",
            "gold_call": "digest(_oracle_track_local_branch(loading, 1.0, 0.15, 0.10, -0.15))",
        },
        # --- Valid: the asymmetric landscape of the task, whose far well sits
        # closer to the barrier than the occupied one ---
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
            "call": "digest(track_local_branch(loading, 1.07, 0.130, 0.0970, -0.130, 0.075))",
            "gold_call": "digest(_oracle_track_local_branch(loading, 1.07, 0.130, 0.0970, -0.130, 0.075))",
        },
        # --- Valid: a deep well the loading cannot exhaust, so the branch survives ---
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
            "call": "digest(track_local_branch(loading, 1.0, 0.80, 0.10, -0.80))",
            "gold_call": "digest(_oracle_track_local_branch(loading, 1.0, 0.80, 0.10, -0.80))",
        },
        # --- Valid: a stiffer well under a monotonically rising loading ---
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
""",
            "call": "digest(track_local_branch(loading, 3.0, 0.05, 0.05, -0.05))",
            "gold_call": "digest(_oracle_track_local_branch(loading, 3.0, 0.05, 0.05, -0.05))",
        },
        # --- Boundary: an initial state placed exactly on the barrier ---
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
""",
            "call": "digest(track_local_branch(loading, 1.0, 0.15, 0.10, 0.0))",
            "gold_call": "digest(_oracle_track_local_branch(loading, 1.0, 0.15, 0.10, 0.0))",
        },
        # --- Boundary: a table holding a single node ---
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
            "call": "digest(track_local_branch(loading, 1.0, 0.15, 0.10, -0.15))",
            "gold_call": "digest(_oracle_track_local_branch(loading, 1.0, 0.15, 0.10, -0.15))",
        },
        # --- Invalid: an initial state outside the well being tracked ---
        {
            "setup": """import numpy as np
loading = np.zeros((5, 3))
def run_model():
    try:
        track_local_branch(loading, 1.0, 0.15, 0.10, 0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_track_local_branch(loading, 1.0, 0.15, 0.10, 0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a loading table of the wrong width ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        track_local_branch(np.zeros((10, 4)))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_track_local_branch(np.zeros((10, 4)))
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
loading = np.zeros((5, 3))
def run_model():
    try:
        track_local_branch(loading, 1.0, 0.15, 0.0, -0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_track_local_branch(loading, 1.0, 0.15, 0.0, -0.15)
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
