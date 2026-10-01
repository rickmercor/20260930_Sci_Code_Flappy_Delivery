"""
Solve the breakdown condition exactly for its dimensionless field variable.

Substituting the triangular field profile into the ionization integral and integrating by parts leaves a relation between the dimensionless group and a second dimensionless variable equal to the effective field scale divided by the peak field. The relation contains the exponential integral E1(z) = int_z^inf e^-u/u du and cannot be inverted in elementary functions, so it must be solved numerically rather than replaced by an empirical power law. Solve it to full double precision.

Returns
-------
float, the dimensionless field variable (no units)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def avalanche_zeta(phi: float) -> float:
    '''Dimensionless field variable satisfying the exact avalanche condition.

    Parameters
    ----------
    phi : float
        The dimensionless group from the previous step.

    Returns
    -------
    float
        The dimensionless field variable (no units).

    Raises
    ------
    ValueError
        If phi is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import exp1
from scipy.optimize import brentq

def _oracle_avalanche_zeta(phi: float) -> float:
    for _v in (phi,):
        if not np.isfinite(_v) or _v <= 0.0:
            raise ValueError('phi must be finite and strictly positive')
    f = lambda z: np.exp(-z)/z - exp1(z) - 1.0/phi
    return float(brentq(f, 1e-10, 400.0, xtol=1e-15, rtol=8.9e-16, maxiter=300))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np",
            "call": "avalanche_zeta(6.04983e6)",
            "gold_call": "_oracle_avalanche_zeta(6.04983e6)",
        },
        {
            "setup": "import numpy as np",
            "call": "avalanche_zeta(1.1052e4)",
            "gold_call": "_oracle_avalanche_zeta(1.1052e4)",
        },
        {
            "setup": "import numpy as np",
            "call": "avalanche_zeta(2.688e7)",
            "gold_call": "_oracle_avalanche_zeta(2.688e7)",
        },
        {
            "setup": "import numpy as np",
            "call": "avalanche_zeta(50.0)",
            "gold_call": "_oracle_avalanche_zeta(50.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "avalanche_zeta(12.0)",
            "gold_call": "_oracle_avalanche_zeta(12.0)",
        },
        {
            "setup": "import numpy as np",
            "call": "avalanche_zeta(1.0e9)",
            "gold_call": "_oracle_avalanche_zeta(1.0e9)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        avalanche_zeta(-5.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_avalanche_zeta(-5.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
