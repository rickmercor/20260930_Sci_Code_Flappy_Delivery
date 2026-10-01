"""
Advance a state by one increment along a single branch of the landscape, holding it fixed while the drive stays inside the elastic range and otherwise returning it to the yield surface.

Restricting the incremental minimization to a neighbourhood of the current state turns it into a predictor-corrector: freeze the state, measure the drive it feels under the new loading, and act only if that drive leaves the range of forces the machinery can sustain without remodelling. Inside the range nothing moves, and the mark is held by dissipation alone; outside it the state is corrected until the drive sits exactly on the boundary of the range, with the sign of the increment inherited from the sign of the violation. This is the return-mapping construction of computational plasticity, transplanted to chromatin, and it is local by design: it never compares the occupied basin against a distant one. The correction is one scalar equation in the state. It inverts in closed form only when the branch force is linear, so a general branch force, including an anharmonic one that stiffens as the mark saturates, calls for a safeguarded iterative solve.

Returns
-------
float, the state at the new node, equal to the previous state whenever the trial drive lies inside the elastic range and otherwise the unique state placing the drive on the boundary of that range, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def return_map_step(q_prev: float, ell: float, rho: float = 0.10,
                    stiffness: float = 1.0, offset: float = 0.15,
                    curvature: float = 0.0, tol: float = 1.0e-13,
                    max_iter: int = 100) -> float:
    """Advance the state by one increment along a single branch.

    The branch force is stiffness * (q + offset) + curvature * sinh(q), which
    is strictly increasing in q for the admitted parameters, so the corrected
    state is unique whenever a correction is called for.

    Parameters
    ----------
    q_prev : float
        State at the previous node of the partition.
    ell : float
        Interaction potential at the new node.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0; the
        elastic range of sustainable drives is [-rho, rho].
    stiffness : float
        Linear coefficient of the branch force, stiffness > 0.
    offset : float
        Displacement of the well bottom of this branch, so that the branch
        force vanishes at q = -offset.
    curvature : float
        Coefficient of the anharmonic part of the branch force,
        curvature >= 0; zero recovers a linear branch.
    tol : float
        Absolute tolerance on the residual of the corrector, tol > 0.
    max_iter : int
        Maximum number of corrector iterations, max_iter >= 1.

    Returns
    -------
    q_next : float
        The state at the new node, as a native Python float.

    Raises
    ------
    ValueError
        If q_prev, ell, rho, stiffness, offset, curvature or tol is not a
        finite number, if rho, stiffness or tol is not strictly greater than
        zero, if curvature is negative, if max_iter is not an integer of at
        least one, or if the branch force cannot be bracketed around the
        target within the doubling budget.
    """
    return q_next  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_return_map_step(q_prev: float, ell: float, rho: float = 0.10,
                            stiffness: float = 1.0, offset: float = 0.15,
                            curvature: float = 0.0, tol: float = 1.0e-13,
                            max_iter: int = 100) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value in (("q_prev", q_prev), ("ell", ell), ("rho", rho),
                        ("stiffness", stiffness), ("offset", offset),
                        ("curvature", curvature), ("tol", tol)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(float(value))):
            raise ValueError(f"{name} must be a finite number")
    if float(rho) <= 0.0:
        raise ValueError("rho must be strictly greater than zero")
    if float(stiffness) <= 0.0:
        raise ValueError("stiffness must be strictly greater than zero")
    if float(curvature) < 0.0:
        raise ValueError("curvature must be non-negative")
    if float(tol) <= 0.0:
        raise ValueError("tol must be strictly greater than zero")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError("max_iter must be an integer of at least one")

    previous, load, threshold = float(q_prev), float(ell), float(rho)
    linear, shift, anharmonic = float(stiffness), float(offset), float(curvature)
    tolerance, budget = float(tol), int(max_iter)

    def _force(state):
        return linear * (state + shift) + anharmonic * np.sinh(state)

    def _slope(state):
        return linear + anharmonic * np.cosh(state)

    # Elastic predictor: the drive the frozen state feels under the new load.
    trial = load - _force(previous)
    if abs(trial) <= threshold:
        return float(previous)

    # Corrector: place the drive exactly on the boundary of the elastic range,
    # on the side the violation points to.
    direction = 1.0 if trial > 0.0 else -1.0
    target = load - direction * threshold

    # Bracket the root by marching away from the previous state in the
    # direction of the increment; the branch force is strictly increasing, so
    # one-sided expansion always succeeds.
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
        step = state - residual / _slope(state)
        if not (lower < step < upper):
            step = 0.5 * (lower + upper)
        if abs(step - state) <= tolerance * (1.0 + abs(state)):
            state = step
            break
        state = step

    return float(state)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a linear branch held inside its elastic range ---
        {
            "setup": """import numpy as np
q_prev, ell, rho, stiffness, offset = -0.15, 0.05, 0.10, 1.0, 0.15
""",
            "call": "float(1.0e6 * return_map_step(q_prev, ell, rho, stiffness, offset))",
            "gold_call": "float(1.0e6 * _oracle_return_map_step(q_prev, ell, rho, stiffness, offset))",
        },
        # --- Valid: the same branch driven past its yield surface ---
        {
            "setup": """import numpy as np
q_prev, ell, rho, stiffness, offset = -0.15, 0.22, 0.10, 1.0, 0.15
""",
            "call": "float(1.0e6 * return_map_step(q_prev, ell, rho, stiffness, offset))",
            "gold_call": "float(1.0e6 * _oracle_return_map_step(q_prev, ell, rho, stiffness, offset))",
        },
        # --- Valid: an anharmonic branch force, which has no elementary inverse ---
        {
            "setup": """import numpy as np
q_prev, ell, rho, stiffness, offset, curvature = 0.0, 1.4, 0.30, 1.0, 0.0, 1.0
""",
            "call": "float(1.0e6 * return_map_step(q_prev, ell, rho, stiffness, offset, curvature))",
            "gold_call": "float(1.0e6 * _oracle_return_map_step(q_prev, ell, rho, stiffness, offset, curvature))",
        },
        # --- Valid: a reversed increment, the drive violating the lower bound ---
        {
            "setup": """import numpy as np
q_prev, ell, rho, stiffness, offset, curvature = 1.0, 0.5, 0.30, 1.0, 0.0, 1.0
""",
            "call": "float(1.0e6 * return_map_step(q_prev, ell, rho, stiffness, offset, curvature))",
            "gold_call": "float(1.0e6 * _oracle_return_map_step(q_prev, ell, rho, stiffness, offset, curvature))",
        },
        # --- Boundary: the drive sitting exactly on the yield surface ---
        {
            "setup": """import numpy as np
q_prev, ell, rho, stiffness, offset = -0.15, 0.10, 0.10, 1.0, 0.15
""",
            "call": "float(1.0e6 * return_map_step(q_prev, ell, rho, stiffness, offset))",
            "gold_call": "float(1.0e6 * _oracle_return_map_step(q_prev, ell, rho, stiffness, offset))",
        },
        # --- Edge: a strongly anharmonic branch under a very large load ---
        {
            "setup": """import numpy as np
q_prev, ell, rho, stiffness, offset, curvature = 0.0, 500.0, 0.30, 0.5, 0.0, 3.0
""",
            "call": "float(1.0e6 * return_map_step(q_prev, ell, rho, stiffness, offset, curvature))",
            "gold_call": "float(1.0e6 * _oracle_return_map_step(q_prev, ell, rho, stiffness, offset, curvature))",
        },
        # --- Invalid: a negative anharmonic coefficient, which would allow the
        # branch force to lose monotonicity ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        return_map_step(0.0, 1.0, 0.10, 1.0, 0.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_return_map_step(0.0, 1.0, 0.10, 1.0, 0.0, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an elastic range of zero width ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        return_map_step(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_return_map_step(0.0, 1.0, 0.0, 1.0, 0.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a corrector budget of no iterations ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        return_map_step(0.0, 1.0, 0.10, 1.0, 0.0, 0.0, 1.0e-13, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_return_map_step(0.0, 1.0, 0.10, 1.0, 0.0, 0.0, 1.0e-13, 0)
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
