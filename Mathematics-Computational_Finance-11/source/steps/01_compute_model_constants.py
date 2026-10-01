"""
Derive the scalar constants that the recursive-utility formulation depends on, given the preference parameters and an interest rate.

Return a 1-D array of length 5, in this fixed order:

  index 0  the recursive-utility parameter combining risk aversion and the elasticity of intertemporal substitution, which is positive under the stated assumptions and whose position relative to one selects the timing regime

  index 1  the coefficient that multiplies the value function on the left-hand side of the stationary Hamilton-Jacobi-Bellman equation once the linear part of the felicity function has been absorbed into the aggregator

  index 2  the exponent carried by the value-function factor in the optimal consumption rule

  index 3  the exponent carried by the value-function factor in the modified aggregator

  index 4  the convenient constant, built from the discount rate, the elasticity of intertemporal substitution and the interest rate, that scales wealth in the supersolution barrier of the problem

Inputs are the risk aversion, the elasticity of intertemporal substitution, the discount rate and the interest rate, all real scalars. Raises ValueError if any input is not a finite real scalar, if risk aversion is not greater than one, if the elasticity of intertemporal substitution is not strictly between zero and one, or if the ordering discount rate greater than interest rate greater than zero fails.

Recursive preferences of the Epstein-Zin type separate aversion to risk across states from willingness to substitute consumption across time, which time-additive utility forces a single parameter to govern. The separation is organised by one derived parameter formed from the risk aversion and the elasticity of intertemporal substitution. Its position relative to one determines whether the agent prefers early or late resolution of uncertainty, and that in turn determines which structural results are available for the associated Hamilton-Jacobi-Bellman equation: in one regime a comparison principle holds and the solution is unique, in the other it does not and existence has to be obtained from monotonicity of an associated fixed-point map instead.

The same parameter reappears in three further places. The felicity function of a recursive utility can be split into a part depending on consumption and the value function and a part linear in the value function alone; absorbing the linear part changes the coefficient multiplying the value function in the stationary equation away from the bare discount rate. The remaining aggregator carries a power of the value function, and the first-order condition for consumption carries a different power of it. These two exponents are not independent, and neither is the coefficient: all three follow from the same combination of risk aversion and elasticity.

Finally, the analysis of the scheme requires explicit sub- and supersolutions that bracket the numerical solution. The supersolution is built from wealth measured in units that include the present value of the higher income stream, scaled by a constant assembled from the discount rate, the elasticity and the interest rate. Under the standing assumptions of the model this constant is strictly between the interest rate and the discount rate, which is what makes the barrier well defined.

Returns
-------
np.ndarray, shape (5,) of dtype float, holding the recursive-utility parameter, the coefficient multiplying the value function in the stationary equation, the consumption-rule exponent, the aggregator exponent, and the supersolution barrier constant, in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_model_constants(gamma: float, psi: float, rho: float, r: float) -> np.ndarray:
    '''Return the five scalar constants of the recursive-utility formulation.

    Parameters
    ----------
    gamma : float
        Risk aversion, strictly greater than one.
    psi : float
        Elasticity of intertemporal substitution, strictly between zero and one.
    rho : float
        Subjective discount rate.
    r : float
        Interest rate, strictly between zero and the discount rate.

    Returns
    -------
    out : np.ndarray
        Shape (5,) float array holding, in order, the recursive-utility parameter, the
        coefficient multiplying the value function in the stationary equation, the exponent
        in the optimal consumption rule, the exponent in the modified aggregator, and the
        constant scaling wealth in the supersolution barrier.
    '''
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_model_constants(gamma: float, psi: float, rho: float, r: float) -> np.ndarray:
    for name, val in (("gamma", gamma), ("psi", psi), ("rho", rho), ("r", r)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float, np.integer, np.floating)):
            raise ValueError(name + " must be a real scalar")
        if not np.isfinite(float(val)):
            raise ValueError(name + " must be finite")
    gamma, psi, rho, r = float(gamma), float(psi), float(rho), float(r)
    if not gamma > 1.0:
        raise ValueError("gamma must be strictly greater than 1")
    if not (0.0 < psi < 1.0):
        raise ValueError("psi must lie strictly between 0 and 1")
    if not (rho > r > 0.0):
        raise ValueError("the ordering rho > r > 0 must hold")

    theta = (1.0 - 1.0 / psi) / (1.0 - gamma)
    lhs_coefficient = rho / theta
    consumption_exponent = (1.0 - gamma * psi) / (1.0 - gamma)
    aggregator_exponent = 1.0 - theta
    b = rho * ((r + psi * (rho - r)) / rho) ** (1.0 / (1.0 - psi))
    return np.array([theta, lhs_coefficient, consumption_exponent,
                     aggregator_exponent, b], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.5, 0.6, 0.045, 0.021410049888
""",
            "call": "compute_model_constants(gamma, psi, rho, r)",
            "gold_call": "_oracle_compute_model_constants(gamma, psi, rho, r)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 2.0, 0.5, 0.05, 0.03
""",
            "call": "compute_model_constants(gamma, psi, rho, r)",
            "gold_call": "_oracle_compute_model_constants(gamma, psi, rho, r)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 8.0, 0.05, 0.09, 1e-6
""",
            "call": "compute_model_constants(gamma, psi, rho, r)",
            "gold_call": "_oracle_compute_model_constants(gamma, psi, rho, r)",
        },
        {
            "setup": """import numpy as np
gamma, psi, rho, r = 1.2, 0.95, 0.06, 0.01
""",
            "call": "compute_model_constants(gamma, psi, rho, r)",
            "gold_call": "_oracle_compute_model_constants(gamma, psi, rho, r)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_model_constants(1.0, 0.6, 0.045, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_model_constants(1.0, 0.6, 0.045, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_model_constants(1.5, 1.0, 0.045, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_model_constants(1.5, 1.0, 0.045, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_model_constants(1.5, 0.6, 0.045, 0.045)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_model_constants(1.5, 0.6, 0.045, 0.045)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_model_constants(1.5, 0.6, float('nan'), 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_model_constants(1.5, 0.6, float('nan'), 0.02)
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
