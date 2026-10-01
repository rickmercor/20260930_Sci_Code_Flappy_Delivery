"""
Step 07: Largest pumping efficiency over the step-2 duration (orchestrator).

Largest energy efficiency of the coupling/decoupling pump over the duration of step 2 (orchestrator).

For fixed device parameters, bias and step-1 duration t1, the periodic-steady-state efficiency eta(t2) of the
pumping-performance step oscillates as a function of the step-2 duration: coherence between the dot and lead 1 that
survives step 2 feeds or drains the tunnelling energy available at the next switch, so eta can have several local
maxima of similar height in a window of t2.

Return the global maximum of eta(t2) over t2_min <= t2 <= t2_max to within 1e-5. Because the local maxima can be close
in height, the whole window has to be searched before the best one is refined.

Inputs: params as in the drift step, t1 > 0, 0 < t2_min < t2_max with t1 + t2_max <= 20, bias V > 0. Output: the largest
eta as a float. Invalid inputs, or t2_min >= t2_max, raise ValueError.

Returns
-------
float, largest periodic-state energy efficiency over the step-2 window
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


def maximal_pumping_efficiency(params: np.ndarray, t1: float, t2_min: float, t2_max: float, bias: float) -> float:
    '''Global maximum over t2 in [t2_min, t2_max] of the periodic-steady-state pumping efficiency.

    Parameters
    ----------
    params : np.ndarray
        Shape (7,): [eps, omega_1, lambda_1, Gamma_1, omega_2, lambda_2, Gamma_2].
    t1 : float
        Duration of step 1, positive.
    t2_min : float
        Lower end of the step-2 window, positive.
    t2_max : float
        Upper end of the step-2 window, larger than t2_min.
    bias : float
        V = mu_2 - mu_1 with mu_1 = 0, positive.

    Returns
    -------
    result : float
        The largest efficiency eta over the window.

    Raises
    ------
    ValueError
        If an input is invalid or t2_min >= t2_max.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


def _oracle_maximal_pumping_efficiency(params: np.ndarray, t1: float, t2_min: float, t2_max: float,
                                       bias: float) -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    lo = _real_float(t2_min, "t2_min")
    hi = _real_float(t2_max, "t2_max")
    if not 0.0 < lo < hi:
        raise ValueError("the step-2 window must satisfy 0 < t2_min < t2_max")
    efficiency = lambda t2: _oracle_pumping_performance(params, t1, t2, bias)[3]
    grid = np.linspace(lo, hi, int(np.ceil((hi - lo) / 0.25)) + 1)
    values = np.array([efficiency(x) for x in grid])
    best = int(np.argmax(values))
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, grid.size - 1)]
    res = minimize_scalar(lambda x: -efficiency(x), bounds=(a, b), method="bounded", options={"xatol": 1e-6})
    return float(max(-res.fun, values[best]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark device, full window ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "from scipy.optimize import minimize_scalar\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "p_g = p.copy()\n",
            "call": "maximal_pumping_efficiency(p, 1.128, 1.0, 12.0, 1.0)",
            "gold_call": "_oracle_maximal_pumping_efficiency(p_g, 1.128, 1.0, 12.0, 1.0)",
            "tol": 2e-05,
        },
        # --- Boundary: benchmark device, a narrower window at long step-2 durations ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "from scipy.optimize import minimize_scalar\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "p_g = p.copy()\n",
            "call": "maximal_pumping_efficiency(p, 1.128, 6.0, 12.0, 1.0)",
            "gold_call": "_oracle_maximal_pumping_efficiency(p_g, 1.128, 6.0, 12.0, 1.0)",
            "tol": 2e-05,
        },
        # --- Normal: broader lead 1 and a different step-1 duration ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "from scipy.optimize import minimize_scalar\n"
                     "p = np.array([1.25, 0.0, 0.5, 2.5, 1.25, 0.125, 0.25])\n"
                     "p_g = p.copy()\n",
            "call": "maximal_pumping_efficiency(p, 0.7793, 1.0, 10.0, 1.0)",
            "gold_call": "_oracle_maximal_pumping_efficiency(p_g, 0.7793, 1.0, 10.0, 1.0)",
            "tol": 2e-05,
        },
        # --- Edge: the grid maximum falls at the upper end of the window, so the search must clamp ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "from scipy.optimize import minimize_scalar\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "p_g = p.copy()\n",
            "call": "maximal_pumping_efficiency(p, 1.128, 3.0, 4.2, 1.0)",
            "gold_call": "_oracle_maximal_pumping_efficiency(p_g, 1.128, 3.0, 4.2, 1.0)",
            "tol": 2e-05,
        },
        # --- Error: an empty window must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "from scipy.optimize import minimize_scalar\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1]), 1.128, 5.0, 5.0, 1.0)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(maximal_pumping_efficiency)",
            "gold_call": "_probe(_oracle_maximal_pumping_efficiency)",
        },
    ]
