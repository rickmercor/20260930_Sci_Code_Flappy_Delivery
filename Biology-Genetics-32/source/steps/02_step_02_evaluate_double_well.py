"""
Evaluate the configurational free energy of a bistable chromatin landscape and the conjugate force it exerts, at an arbitrary set of states.

Committed cell phenotypes are separated by barriers, the picture Waddington drew and that dynamical systems render as a bistable switch. The minimal landscape with that property is a pair of wells lying on either side of the neutral state, each quadratic in the departure from its own bottom and meeting there in a corner. The corner is a caricature of a smooth barrier, chosen because it keeps the conjugate force piecewise linear so that every yield condition can be inverted elementarily, while retaining the feature that matters: the force is not monotone across the barrier, it drops abruptly on passing from one basin to the other, and no continuous path of equilibria connects the two wells. Placing the two bottoms at unequal distances from the corner, while insisting that the energy itself stay continuous there, raises one well above the other by an amount fixed entirely by that asymmetry, so one phenotype becomes intrinsically preferred without any further parameter.

Returns
-------
np.ndarray of shape (2, q.size), float: the configurational free energy in row zero and its derivative with respect to the state in row one, evaluated at the flattened input states.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def evaluate_double_well(q, k: float = 1.0, a: float = 0.15,
                         b: float = None) -> np.ndarray:
    """Evaluate the bistable landscape and its conjugate force.

    The landscape holds two wells of curvature k whose bottoms sit at -a and
    at +b, meeting at the neutral state q = 0. The repressed well, the one
    centred at -a, is taken as the zero of energy, and the free energy itself
    is continuous at the corner; that continuity fixes the level of the far
    well, so no separate offset is supplied. The state q = 0 belongs to the
    well centred at -a.

    Parameters
    ----------
    q : array_like
        States at which the landscape is evaluated; any shape is accepted and
        the result is flattened in C order.
    k : float
        Curvature of each well, k > 0.
    a : float
        Distance from the corner to the bottom of the repressed well, a >= 0;
        a = 0 collapses the landscape to a single well.
    b : float or None
        Distance from the corner to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    landscape : np.ndarray
        Array of shape (2, q.size) whose first row holds the configurational
        free energy at each state and whose second row holds the conjugate
        force, the derivative of that energy with respect to the state.

    Raises
    ------
    ValueError
        If q holds no state or any non-finite state, if k is not a finite
        number strictly greater than zero, if a is not a finite non-negative
        number, or if b is neither None nor a finite non-negative number.
    """
    return landscape  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_evaluate_double_well(q, k: float = 1.0, a: float = 0.15,
                                 b: float = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    states = np.asarray(q, dtype=float).ravel()
    if states.size < 1:
        raise ValueError("q must hold at least one state")
    if not np.all(np.isfinite(states)):
        raise ValueError("q must hold finite states only")
    if not (isinstance(k, (int, float, np.floating, np.integer))
            and np.isfinite(float(k)) and float(k) > 0.0):
        raise ValueError("k must be a finite number strictly greater than zero")
    if not (isinstance(a, (int, float, np.floating, np.integer))
            and np.isfinite(float(a)) and float(a) >= 0.0):
        raise ValueError("a must be a finite non-negative number")
    if b is not None and not (isinstance(b, (int, float, np.floating, np.integer))
                              and np.isfinite(float(b)) and float(b) >= 0.0):
        raise ValueError("b must be None or a finite non-negative number")

    curvature = float(k)
    repressed = float(a)
    active = repressed if b is None else float(b)

    # The neutral state is assigned to the repressed well, so the centre of
    # the occupied well is -a on {q <= 0} and +b on {q > 0}. This convention
    # only matters at the corner itself, where both branches agree in value.
    centre = np.where(states <= 0.0, -repressed, active)

    # Continuity of the energy at the corner raises the far well by exactly
    # the amount by which the two bottoms are unequally placed; the repressed
    # well is the zero of energy, so only the far branch carries the level.
    level = 0.5 * curvature * (repressed ** 2 - active ** 2)
    offset = np.where(states <= 0.0, 0.0, level)

    energy = 0.5 * curvature * (states - centre) ** 2 + offset
    force = curvature * (states - centre)

    return np.vstack((energy, force)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: states spanning both wells and the corner between them ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
q = np.linspace(-0.6, 0.6, 13)
k, a = 1.0, 0.15
""",
            "call": "digest(evaluate_double_well(q, k, a))",
            "gold_call": "digest(_oracle_evaluate_double_well(q, k, a))",
        },
        # --- Valid: a stiffer landscape with widely separated wells ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
q = np.array([-2.0, -0.9, -0.3, 0.0, 0.3, 0.9, 2.0])
k, a = 4.0, 0.9
""",
            "call": "digest(evaluate_double_well(q, k, a))",
            "gold_call": "digest(_oracle_evaluate_double_well(q, k, a))",
        },
        # --- Boundary: the corner and the two well bottoms exactly ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
q = np.array([-0.15, 0.0, 0.15])
k, a = 1.0, 0.15
""",
            "call": "digest(evaluate_double_well(q, k, a))",
            "gold_call": "digest(_oracle_evaluate_double_well(q, k, a))",
        },
        # --- Edge: a vanishing barrier, where the landscape has one well only ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
q = np.linspace(-1.0, 1.0, 9)
k, a = 2.0, 0.0
""",
            "call": "digest(evaluate_double_well(q, k, a))",
            "gold_call": "digest(_oracle_evaluate_double_well(q, k, a))",
        },
        # --- Edge: a single state supplied as a bare scalar ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
""",
            "call": "digest(evaluate_double_well(0.42, 1.0, 0.15))",
            "gold_call": "digest(_oracle_evaluate_double_well(0.42, 1.0, 0.15))",
        },
        # --- Valid: an asymmetric landscape whose far well is raised ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
q = np.linspace(-0.5, 0.5, 11)
k, a, b = 1.07, 0.130, 0.075
""",
            "call": "digest(evaluate_double_well(q, k, a, b))",
            "gold_call": "digest(_oracle_evaluate_double_well(q, k, a, b))",
        },
        # --- Boundary: the corner approached from both sides of an asymmetric
        # landscape, where the two branches must still agree in value ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
q = np.array([-0.130, -1.0e-12, 0.0, 1.0e-12, 0.075])
k, a, b = 1.07, 0.130, 0.075
""",
            "call": "digest(evaluate_double_well(q, k, a, b))",
            "gold_call": "digest(_oracle_evaluate_double_well(q, k, a, b))",
        },
        # --- Edge: a far well placed beyond the near one, so the level falls ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
q = np.array([-0.4, -0.1, 0.0, 0.1, 0.4, 0.9])
k, a, b = 2.0, 0.20, 0.35
""",
            "call": "digest(evaluate_double_well(q, k, a, b))",
            "gold_call": "digest(_oracle_evaluate_double_well(q, k, a, b))",
        },
        # --- Invalid: a landscape with no curvature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        evaluate_double_well(np.zeros(3), 0.0, 0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_double_well(np.zeros(3), 0.0, 0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative half-separation ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        evaluate_double_well(np.zeros(3), 1.0, -0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_double_well(np.zeros(3), 1.0, -0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative distance to the far well ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        evaluate_double_well(np.zeros(3), 1.0, 0.15, -0.05)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_double_well(np.zeros(3), 1.0, 0.15, -0.05)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an empty set of states ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        evaluate_double_well(np.array([]), 1.0, 0.15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_double_well(np.array([]), 1.0, 0.15)
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
