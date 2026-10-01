"""
Builds the discrete lift weights and mean-reversion speeds that approximate the rough-volatility kernel of the lifted Heston model from the Hurst exponent and the number of lift factors.

Lift weights and mean-reversion speeds.

The lifted Heston model replaces the singular rough fractional kernel by a finite sum of Ornstein-Uhlenbeck factors whose weights and decay speeds are chosen to reproduce the kernel's scaling behaviour across the relevant range of time scales. The construction is controlled by two numbers: a geometric refinement ratio that grows with the number of factors, and the Hurst exponent, which fixes the roughness of the limiting volatility process. The weights set the amplitude with which each factor contributes to the instantaneous variance, while the speeds set how quickly each factor decays back to its long-run level; taken together they define the linear operator that governs the joint evolution of all factors. Because the same weights and speeds enter every later conditional-moment and simulation step, they are computed once, at full double precision, and treated as fixed inputs downstream.

Returns
-------
omega, x : tuple of two (N,) float arrays -- lift weights and mean-reversion speeds, in increasing factor index (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lift_parameters(H: float, N: int) -> tuple:
    """Build the lift weights and mean-reversion speeds of the lifted Heston model.

    Parameters
    ----------
    H : float
        Hurst exponent of the rough-volatility kernel (must satisfy
        0 < H < 0.5).
    N : int
        Number of lift factors (must be an integer >= 1).

    Returns
    -------
    omega : (N,) float array
        Lift weights of the N factors, in the order of increasing index.
    x : (N,) float array
        Mean-reversion speeds of the N factors, in the order of increasing
        index (all strictly positive).

    Raises
    ------
    ValueError
        If H is not a real number with 0 < H < 0.5, or if N is not an
        integer >= 1.
    """
    return omega, x

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math


def _oracle_lift_parameters(H: float, N: int) -> tuple:
    """Reference implementation of lift_parameters."""
    if not isinstance(H, (int, float, np.integer, np.floating)) or not (0.0 < float(H) < 0.5):
        raise ValueError("H must be a real number with 0 < H < 0.5")
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or int(N) < 1:
        raise ValueError("N must be an integer >= 1")
    H = float(H)
    N = int(N)
    rN = 1.0 + 10.0 * N ** (-0.9)
    n = np.arange(1, N + 1)
    omega = ((rN ** (0.5 - H) - 1.0)
             * rN ** ((H - 0.5) * (1.0 + 0.5 * N))
             / (math.gamma(H + 0.5) * math.gamma(1.5 - H))
             * rN ** ((0.5 - H) * n))
    x = ((0.5 - H) / (1.5 - H)
         * (rN ** (1.5 - H) - 1.0) / (rN ** (0.5 - H) - 1.0)
         * rN ** (n - 1.0 - N / 2.0))
    return omega, x

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: normal scenario (benchmark H=0.3, N=5), boundary case
    (single lift factor N=1), edge case (invalid H raises ValueError).
    """
    return [
        {
            # benchmark configuration: weights and speeds of the five lift factors
            "setup": "import numpy as np",
            "call": "float(np.sum(lift_parameters(0.3, 5)[0]) + np.sum(lift_parameters(0.3, 5)[1]))",
            "gold_call": "float(np.sum(_oracle_lift_parameters(0.3, 5)[0]) + np.sum(_oracle_lift_parameters(0.3, 5)[1]))",
        },
        {
            # weights and speeds are strictly positive at the benchmark config
            "setup": "import numpy as np",
            "call": "float(np.min(lift_parameters(0.3, 5)[1]) > 0.0) + float(np.max(lift_parameters(0.3, 5)[1]))",
            "gold_call": "float(np.min(_oracle_lift_parameters(0.3, 5)[1]) > 0.0) + float(np.max(_oracle_lift_parameters(0.3, 5)[1]))",
        },
        {
            # boundary: a single lift factor is the minimal admissible lift
            "setup": "import numpy as np",
            "call": "float(np.sum(lift_parameters(0.1, 1)[0]) + np.sum(lift_parameters(0.1, 1)[1]))",
            "gold_call": "float(np.sum(_oracle_lift_parameters(0.1, 1)[0]) + np.sum(_oracle_lift_parameters(0.1, 1)[1]))",
        },
        {
            # edge: H = 0.5 is outside the admissible roughness range
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: lift_parameters(0.5, 5))",
            "gold_call": "_guard(lambda: _oracle_lift_parameters(0.5, 5))",
        },
        {
            # edge: a non-positive factor count is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: lift_parameters(0.3, 0))",
            "gold_call": "_guard(lambda: _oracle_lift_parameters(0.3, 0))",
        },
    ]
