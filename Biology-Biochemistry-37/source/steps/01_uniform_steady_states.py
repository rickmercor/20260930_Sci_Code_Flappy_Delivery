"""
Return the interior uniform equilibrium and the bistability threshold of the cascade model for a single edge with bias phi and saturation parameter B, as the array [xi, phi_c]. Both follow in closed form from the uniform reduction of the node equation, in which every node sits at the same activity level.

When every node of a long uniform pathway rests at the same activity level, the cascade reduces to one autonomous equation with the fully active and fully inactive states as equilibria and, for moderate bias, a third interior equilibrium that separates their basins. The bias at which that interior equilibrium leaves the physical domain marks the transition between bistable signalling and a monostable, locked pathway.

Returns
-------
ndarray of float64, shape (2,), [xi, phi_c].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def uniform_steady_states(phi: float, B: float) -> "np.ndarray":
    """Return the interior uniform equilibrium and the bistability threshold of the cascade model for a single edge with bias phi and saturation parameter B, as the array [xi, phi_c]. Both follow in closed form from the uniform reduction of the node equation, in which every node sits at the same activity level.

    Parameters
    ----------
    phi : float
        Bias parameter of the edge, in [-1, 1].
    B : float
        Saturation parameter B = 2 beta - 1 of the edge, greater than 1.

    Returns
    -------
    out : np.ndarray
        Array [xi, phi_c]: interior equilibrium and bistability threshold.

    Raises
    ------
    ValueError
        If phi is outside [-1, 1] or B is not greater than 1.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _oracle_uniform_steady_states(phi: float, B: float) -> "np.ndarray":
    """Eqs. (3)-(4): interior equilibrium xi = -phi (2 beta - 1) = -phi B and the bistability
    threshold phi_c = 1 / B of the uniform pathway; returns [xi, phi_c]."""
    if isinstance(phi, bool) or isinstance(B, bool) or not np.isfinite(float(phi)) or not np.isfinite(float(B)):
        raise ValueError("phi and B must be finite numbers")
    p, b = float(phi), float(B)
    if abs(p) > 1.0 or b <= 1.0:
        raise ValueError("need |phi| <= 1 and B > 1")
    return np.array([-p * b, 1.0 / b], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nphi, B = 0.2, 3.0\n",
            "call": "np.asarray(uniform_steady_states(phi, B))",
            "gold_call": "np.asarray(_oracle_uniform_steady_states(phi, B))",
        },
        {
            "setup": "import numpy as np\nphi, B = 0.0, 1.5\n",
            "call": "np.asarray(uniform_steady_states(phi, B))",
            "gold_call": "np.asarray(_oracle_uniform_steady_states(phi, B))",
        },
        {
            "setup": "import numpy as np\nphi, B = -0.3, 8.0\n",
            "call": "np.asarray(uniform_steady_states(phi, B))",
            "gold_call": "np.asarray(_oracle_uniform_steady_states(phi, B))",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        uniform_steady_states(0.2, 1.0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_uniform_steady_states(0.2, 1.0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
