"""
Step 03: One-cycle propagator of the coupling/decoupling pump.

One-cycle propagator of the sharp coupling/decoupling pump.

The pump alternates two steps. During step 1 (duration t1) the dot is coupled to lead 1 only, g_1 = 1 and g_2 = 0;
during step 2 (duration t2) it is coupled to lead 2 only, g_1 = 0 and g_2 = 1. The switching is instantaneous, so the
drift matrix of the reaction-coordinate step is piecewise constant. The homogeneous part of the Heisenberg evolution of
(d, r_1, r_2) over one full cycle, from the start of step 1 to the end of step 2, is the 3 x 3 matrix that multiplies the
operators at the start of the cycle.

Its eigenvalues all lie inside the unit circle for positive widths and strengths, and their largest modulus fixes how
fast the pump forgets its initial state from one cycle to the next.

Inputs: params as in the drift step, t1 > 0 and t2 > 0. Output: complex ndarray of shape (3, 3). Invalid params or a
duration that is not finite and positive raises ValueError.

Returns
-------
numpy.ndarray of shape (3, 3), complex one-cycle homogeneous propagator from the start of step 1 to the end of step 2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm


def cycle_propagator(params: np.ndarray, t1: float, t2: float) -> np.ndarray:
    '''Homogeneous propagator of (d, r_1, r_2) over one step-1 plus step-2 cycle.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2].
    t1 : float
        Duration of step 1 (dot coupled to lead 1), positive.
    t2 : float
        Duration of step 2 (dot coupled to lead 2), positive.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (3, 3) mapping the operators at the start of step 1 to the end of step 2.

    Raises
    ------
    ValueError
        If params is invalid or a duration is not finite and positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_cycle_propagator(params: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import expm
    t1 = _real_float(t1, "t1")
    t2 = _real_float(t2, "t2")
    if t1 <= 0.0 or t2 <= 0.0:
        raise ValueError("step durations must be positive")
    m1 = _oracle_reaction_coordinate_drift(params, 1.0, 0.0)
    m2 = _oracle_reaction_coordinate_drift(params, 0.0, 1.0)
    return expm(m2 * t2) @ expm(m1 * t1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark device at a short cycle ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "p_g = p.copy()\n",
            "call": "cycle_propagator(p, 1.128, 4.1)",
            "gold_call": "_oracle_cycle_propagator(p_g, 1.128, 4.1)",
            "tol": 1e-10,
        },
        # --- Boundary: very unequal steps, a long step 2 ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.25, 0.0, 0.5, 2.5, 1.25, 0.125, 0.25])\n"
                     "p_g = p.copy()\n",
            "call": "cycle_propagator(p, 0.2, 25.0)",
            "gold_call": "_oracle_cycle_propagator(p_g, 0.2, 25.0)",
            "tol": 1e-10,
        },
        # --- Normal: order of the steps matters; the product is not symmetric in the two drifts ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([-0.4, -1.5, 0.8, 0.6, 2.0, 0.05, 3.0])\n"
                     "p_g = p.copy()\n",
            "call": "cycle_propagator(p, 2.7, 0.9)",
            "gold_call": "_oracle_cycle_propagator(p_g, 2.7, 0.9)",
            "tol": 1e-10,
        },
        # --- Edge: very short steps, the propagator is close to the identity ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([0.9, 0.2, 1.0, 1.0, 0.5, 2.0, 0.4])\n"
                     "p_g = p.copy()\n",
            "call": "cycle_propagator(p, 1e-4, 2e-4)",
            "gold_call": "_oracle_cycle_propagator(p_g, 1e-4, 2e-4)",
            "tol": 1e-12,
        },
        # --- Error: zero duration of step 2 must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([1.0, 0.0, 0.1, 1.0, 1.0, 0.2, 0.1]), 1.0, 0.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(cycle_propagator)",
            "gold_call": "_probe(_oracle_cycle_propagator)",
        },
    ]
