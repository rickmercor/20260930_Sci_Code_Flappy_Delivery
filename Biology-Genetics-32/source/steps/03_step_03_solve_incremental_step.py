"""
Advance a bistable chromatin state by one increment of the globally minimizing variational scheme, given the previous state and the interaction potential at the new node.

A rate-independent evolution can be advanced by minimizing, at each new value of the loading, the sum of the stored energy at the trial state and the dissipative cost of reaching that state from the previous one. With a homogeneous dissipation potential, that cost is the potential evaluated on the increment itself, so the objective is smooth except at the previous state, where the dissipation term has a corner, and at the barrier, where the landscape has one. Because the minimization ranges over the whole state space and not merely over a neighbourhood, the scheme compares the occupied basin against every other configuration, including ones no continuous path of equilibria can reach. Where the comparison changes hands, the state is transported discontinuously. The objective is convex on each smooth piece, so the minimizer is found by taking the stationary point of every piece, clipping it to that piece, and comparing the candidates against the corner values.

Returns
-------
float, the state at the new node, the global minimiser over the whole state space of the stored energy plus the dissipation cost of arriving from the previous state, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def solve_incremental_step(q_prev: float, ell: float, k: float = 1.0,
                           a: float = 0.15, rho: float = 0.10,
                           b: float = None) -> float:
    """Advance the state by one globally minimising increment.

    The stored energy of the state q under the interaction potential ell is
    the configurational free energy of the bistable landscape, less q times
    ell; that free energy is k (q + a) ** 2 / 2 on q <= 0 and
    k (q - b) ** 2 / 2 + k (a ** 2 - b ** 2) / 2 on q > 0, so the two
    branches agree in value at the barrier. The dissipation cost of moving
    from q_prev to q is rho times the absolute increment. Ties are resolved
    in favour of the candidate closest to q_prev, and then in favour of the
    smaller state, so that a state which is only marginally displaced by the
    comparison stays where it is.

    Parameters
    ----------
    q_prev : float
        State at the previous node of the partition.
    ell : float
        Interaction potential at the new node.
    k : float
        Curvature of each well of the landscape, k > 0.
    a : float
        Distance from the barrier to the bottom of the repressed well, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    b : float or None
        Distance from the barrier to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    q_next : float
        The state at the new node, as a native Python float.

    Raises
    ------
    ValueError
        If q_prev, ell, k, a or rho is not a finite number, if k or rho is
        not strictly greater than zero, if a is negative, or if b is neither
        None nor a finite non-negative number.
    """
    return q_next  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_incremental_step(q_prev: float, ell: float, k: float = 1.0,
                                   a: float = 0.15, rho: float = 0.10,
                                   b: float = None) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("q_prev", q_prev), ("ell", ell), ("k", k), ("a", a), ("rho", rho)):
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

    previous = float(q_prev)
    load = float(ell)
    curvature, half_gap, threshold = float(k), float(a), float(rho)
    far_gap = half_gap if b is None else float(b)

    # Continuity of the free energy at the barrier fixes the level of the far
    # well; the near well is the zero of energy.
    level = 0.5 * curvature * (half_gap ** 2 - far_gap ** 2)

    def _objective(state):
        centre = -half_gap if state <= 0.0 else far_gap
        offset = 0.0 if state <= 0.0 else level
        return (0.5 * curvature * (state - centre) ** 2 + offset - state * load
                + threshold * abs(state - previous))

    # The objective is smooth except where the landscape has its corner and
    # where the dissipation term changes sign; those two abscissae cut the
    # line into at most three pieces, each carrying a convex quadratic.
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
    # A strict comparison would let rounding decide a genuine tie between the
    # two basins, so near-optimal candidates are gathered and the tie is
    # settled by the stated convention rather than by floating-point noise.
    tolerance = 1.0e-13 * (1.0 + abs(best))
    tied = [state for state, value in zip(candidates, values) if value <= best + tolerance]
    tied.sort(key=lambda state: (abs(state - previous), state))

    return float(tied[0])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a state resting in the left well under a weak potential ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho = -0.15, 0.05, 1.0, 0.15, 0.10
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho))",
        },
        # --- Valid: the same state under a potential well past the comparison ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho = -0.15, 0.30, 1.0, 0.15, 0.10
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho))",
        },
        # --- Valid: a state on the right branch under a falling potential ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho = 0.5466310265, 0.20, 1.0, 0.15, 0.10
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho))",
        },
        # --- Boundary: the potential exactly at the dissipation threshold ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho = -0.15, 0.10, 1.0, 0.15, 0.10
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho))",
        },
        # --- Valid: an asymmetric landscape below its switching load ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho, b = -0.130, 0.120, 1.07, 0.130, 0.0970, 0.075
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho, b))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho, b))",
        },
        # --- Valid: the same asymmetric landscape past its switching load ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho, b = -0.130, 0.135, 1.07, 0.130, 0.0970, 0.075
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho, b))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho, b))",
        },
        # --- Boundary: the raised far well recaptured on an unloading arm ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho, b = 0.16565420560747662, 0.0, 1.07, 0.130, 0.0970, 0.075
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho, b))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho, b))",
        },
        # --- Boundary: a previous state sitting exactly on the barrier ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho = 0.0, 0.22, 1.0, 0.15, 0.10
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho))",
        },
        # --- Edge: a barrier so wide that the far basin never competes ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho = -3.0, 0.45, 1.0, 3.0, 0.10
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho))",
        },
        # --- Edge: no barrier at all, so the landscape has a single well ---
        {
            "setup": """import numpy as np
q_prev, ell, k, a, rho = 0.0, 0.35, 2.0, 0.0, 0.10
""",
            "call": "float(1.0e6 * solve_incremental_step(q_prev, ell, k, a, rho))",
            "gold_call": "float(1.0e6 * _oracle_solve_incremental_step(q_prev, ell, k, a, rho))",
        },
        # --- Invalid: a machinery with no resistance to remodelling ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_incremental_step(-0.15, 0.2, 1.0, 0.15, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_incremental_step(-0.15, 0.2, 1.0, 0.15, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a landscape with no curvature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_incremental_step(-0.15, 0.2, 0.0, 0.15, 0.10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_incremental_step(-0.15, 0.2, 0.0, 0.15, 0.10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-finite interaction potential ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        solve_incremental_step(-0.15, float("inf"), 1.0, 0.15, 0.10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_incremental_step(-0.15, float("inf"), 1.0, 0.15, 0.10)
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
