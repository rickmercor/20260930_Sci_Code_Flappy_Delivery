"""
Tabulate the micro-environmental stimulus protocol and the interaction potential it induces at every node of a uniform partition of the observation window.

In a rate-independent description of epigenetic response, the environment enters only through a scalar loading path and through the interaction potential that path induces. The protocol used here is one full raised-cosine excursion of the stimulus, rising smoothly from rest to a peak at mid-window and returning to rest, so that the cycle is closed in the stimulus even though the chromatin state need not return with it. The coupling to the state saturates: the potential approaches a finite ceiling exponentially as the stimulus grows, because the machinery that reads the environment is itself saturable. That ceiling, and not the peak stimulus, is what decides whether an established mark can be erased at all. Both curves are wanted at the nodes of the partition the time integrator will use, since an incremental scheme reads the loading only at nodes and assembles the discrete work from consecutive nodal values.

Returns
-------
np.ndarray of shape (n_steps + 1, 3), float: the nodal time, the stimulus and the interaction potential at every node of the partition, in that column order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def build_loading_path(n_steps: int, T: float = 1.0, S_max: float = 5.0,
                       ell_inf: float = 0.5, lam: float = 1.0) -> np.ndarray:
    """Tabulate the stimulus protocol and its interaction potential.

    The window [0, T] carries a uniform partition of n_steps intervals, so it
    has n_steps + 1 nodes, the first at t = 0 and the last at t = T. The
    stimulus follows one raised-cosine excursion of amplitude S_max over the
    whole window, and the interaction potential saturates exponentially in the
    stimulus towards the ceiling ell_inf at rate lam.

    Parameters
    ----------
    n_steps : int
        Number of intervals of the partition, n_steps >= 1.
    T : float
        Length of the observation window, T > 0.
    S_max : float
        Peak value of the stimulus, S_max > 0, attained at t = T / 2.
    ell_inf : float
        Saturation ceiling of the interaction potential, ell_inf > 0.
    lam : float
        Saturation rate of the interaction potential, lam > 0.

    Returns
    -------
    loading : np.ndarray
        Array of shape (n_steps + 1, 3) whose columns hold, in order, the
        nodal time, the stimulus at that time and the interaction potential
        at that stimulus.

    Raises
    ------
    ValueError
        If n_steps is not an integer, if n_steps is below one, or if T,
        S_max, ell_inf or lam is not a finite number strictly greater than
        zero.
    """
    return loading  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_build_loading_path(n_steps: int, T: float = 1.0, S_max: float = 5.0,
                               ell_inf: float = 0.5, lam: float = 1.0) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)):
        raise ValueError("n_steps must be an integer")
    if int(n_steps) < 1:
        raise ValueError("n_steps must be at least one")
    for name, value in (("T", T), ("S_max", S_max), ("ell_inf", ell_inf), ("lam", lam)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number strictly greater than zero")

    n_steps = int(n_steps)
    window = float(T)
    amplitude = float(S_max)
    ceiling = float(ell_inf)
    rate = float(lam)

    # The nodes are laid out from the index rather than by accumulation, so the
    # last node lands on T exactly and the spacing carries no drift.
    time = window * np.arange(n_steps + 1, dtype=float) / float(n_steps)

    # One raised-cosine excursion: S(0) = S(T) = 0 with a single peak S_max at
    # mid-window, and a vanishing rate at both ends of the window.
    stimulus = 0.5 * amplitude * (1.0 - np.cos(2.0 * np.pi * time / window))

    # The interaction potential saturates towards its ceiling, so an ever
    # larger stimulus buys ever less drive on the chromatin state.
    potential = ceiling * (1.0 - np.exp(-rate * stimulus))

    return np.column_stack((time, stimulus, potential)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark protocol on a coarse partition ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
n_steps, T, S_max, ell_inf, lam = 8, 1.0, 5.0, 0.5, 1.0
""",
            "call": "digest(build_loading_path(n_steps, T, S_max, ell_inf, lam))",
            "gold_call": "digest(_oracle_build_loading_path(n_steps, T, S_max, ell_inf, lam))",
        },
        # --- Valid: a finer partition over a longer window ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
n_steps, T, S_max, ell_inf, lam = 37, 2.5, 3.0, 0.8, 0.4
""",
            "call": "digest(build_loading_path(n_steps, T, S_max, ell_inf, lam))",
            "gold_call": "digest(_oracle_build_loading_path(n_steps, T, S_max, ell_inf, lam))",
        },
        # --- Boundary: a single interval, so only the two window ends are nodes ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
n_steps, T, S_max, ell_inf, lam = 1, 1.0, 5.0, 0.5, 1.0
""",
            "call": "digest(build_loading_path(n_steps, T, S_max, ell_inf, lam))",
            "gold_call": "digest(_oracle_build_loading_path(n_steps, T, S_max, ell_inf, lam))",
        },
        # --- Edge: a saturation rate so large the potential sits at its ceiling
        # over almost the whole excursion ---
        {
            "setup": """import numpy as np
def digest(block):
    total = 0.0
    for row in np.atleast_2d(np.asarray(block, dtype=float)):
        flat = np.ravel(row)
        total += float(np.sum(flat * 1.0e6 * np.arange(1, flat.size + 1))) + flat.size
    return total
n_steps, T, S_max, ell_inf, lam = 16, 1.0, 5.0, 0.5, 400.0
""",
            "call": "digest(build_loading_path(n_steps, T, S_max, ell_inf, lam))",
            "gold_call": "digest(_oracle_build_loading_path(n_steps, T, S_max, ell_inf, lam))",
        },
        # --- Invalid: a partition with no intervals ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_loading_path(0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_loading_path(0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a window of non-positive length ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_loading_path(10, T=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_loading_path(10, T=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a saturation ceiling that is not positive ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_loading_path(10, ell_inf=-0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_loading_path(10, ell_inf=-0.5)
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
