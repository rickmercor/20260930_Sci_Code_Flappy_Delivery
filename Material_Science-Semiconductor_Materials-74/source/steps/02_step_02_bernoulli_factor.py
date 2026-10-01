"""
Evaluate the Bernoulli factor that carries the drift term inside the flux coefficient.

    B(t) = t / (exp(t) - 1)   for t != 0,

    B(0) = 1.

B is the classical exponential-fitting weight: B(t) -> 0 in the convection-dominated limit t -> +inf and B(0) = 1 in the diffusion-dominated limit. The naive quotient loses all significant digits for small |t| because exp(t) - 1 cancels catastrophically, so the implementation must use expm1 well away from the origin, and the Taylor series

B(t) = 1 - t/2 + t^2/12 - t^4/720 + O(t^6)

near it.

Exponential fitting is what keeps a drift-diffusion discretization stable when the electric field dominates thermal diffusion. The weight that carries it tends to zero in the drift-dominated limit and to one in the diffusion-dominated limit, interpolating smoothly between upwinding and central differencing. Its removable singularity at the origin is a numerical trap rather than a mathematical one: the quotient is perfectly well behaved in exact arithmetic but loses every significant digit in floating-point as the argument approaches zero.

Returns
-------
native Python `float` containing `B(argument)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_bernoulli(argument: float) -> float:
    """Return the Bernoulli factor B(argument), continuous at the origin.

    Parameters
    ----------
    argument : float
        The Bernoulli argument t. Must be finite.

    Returns
    -------
    value : float
        B(t) = t / (exp(t) - 1) for t != 0, and B(0) = 1, evaluated without
        catastrophic cancellation for small |t|.

    Raises
    ------
    ValueError
        If argument is not finite.
    """
    return value  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_evaluate_bernoulli(argument: float) -> float:
    t = float(argument)
    if not np.isfinite(t):
        raise ValueError("argument must be finite")
    if abs(t) < 1.0e-6:
        return float(1.0 - t / 2.0 + t**2 / 12.0 - t**4 / 720.0)
    return float(t / np.expm1(t))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the two primal/dual potential jumps of the task
        {
            "setup": "import numpy as np\n",
            "call": "evaluate_bernoulli(0.6)",
            "gold_call": "_oracle_evaluate_bernoulli(0.6)",
        },
        # Boundary: removable singularity at t = 0, where B(0) = 1
        {
            "setup": "import numpy as np\n",
            "call": "evaluate_bernoulli(0.0)",
            "gold_call": "_oracle_evaluate_bernoulli(0.0)",
        },
        # Edge: near-zero argument where the naive quotient cancels
        {
            "setup": "import numpy as np\n",
            "call": "evaluate_bernoulli(-1.0e-10)",
            "gold_call": "_oracle_evaluate_bernoulli(-1.0e-10)",
        },
        # Edge: strongly convection-dominated argument, B -> 0
        {
            "setup": "import numpy as np\n",
            "call": "evaluate_bernoulli(40.0)",
            "gold_call": "_oracle_evaluate_bernoulli(40.0)",
        },
        # Invalid: non-finite argument
        {
            "setup": """import numpy as np
def run_model():
    try:
        evaluate_bernoulli(np.inf)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_evaluate_bernoulli(np.inf)
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
