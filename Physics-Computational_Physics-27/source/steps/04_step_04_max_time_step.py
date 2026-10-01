"""
Convert a target accuracy into the longest single time step a Chebyshev expansion

can take without its accumulated rounding error exceeding that target.



Inputs

------

rho: float, Bernstein radius enclosing the spectrum, >= 1

delta_max: float, target absolute tolerance, > 0

eps: float, machine precision, > 0 (default 1.11e-16)



Returns

-------

t_max: float, the largest single time step meeting the tolerance



Raises

------

ValueError: if rho is below 1, or delta_max or eps is not positive and finite

Truncating the Chebyshev series is not what limits its accuracy away from the interval

[-1, 1]: the series represents an entire function and converges over the whole complex

plane. What limits it is floating-point rounding. Every retained term is stored with a

relative error of order the unit roundoff, and those errors accumulate as the sum is

formed.



How badly they accumulate depends on the time step and on the Bernstein radius only

through their product. The Chebyshev polynomials grow geometrically in that radius while

the Bessel coefficients decay factorially in the step, and the competition between the two

fixes how large the accumulated error can become. The resulting bound grows without limit

as the product does, so for a spectrum far outside [-1, 1] the step is the only lever that

keeps the error in hand.



Read forwards, the bound turns a step and a radius into a worst-case error. Read backwards,

it turns a requested tolerance into a ceiling on the step, which is what this function

returns. The relation is exactly invertible, so evaluating the bound at the ceiling

reproduces the tolerance it was derived from.



The eps used here is the double-precision unit roundoff 2^-53, approximately 1.11e-16,

which is half of numpy's floating-point spacing constant. Using the spacing constant

instead changes the ceiling and every quantity downstream of it.

Returns
-------
float, the largest single time step for which the accumulated rounding error of one Chebyshev expansion stays within delta_max, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def max_time_step(rho: float, delta_max: float, eps: float = 1.11e-16) -> float:
    '''Compute the largest admissible Chebyshev time step for a target tolerance.

    Parameters
    ----------
    rho : float
        Bernstein-ellipse radius enclosing the spectrum, must be >= 1.
    delta_max : float
        Target absolute rounding-error tolerance, must be > 0.
    eps : float
        Machine precision used in the error bound, must be > 0.

    Returns
    -------
    t_max : float
        Largest time step t such that the accumulated rounding error of a single
        Chebyshev expansion stays within delta_max.

    Raises
    ------
    ValueError
        Raised if rho is below 1, or delta_max or eps is not positive and finite.
    '''
    return t_max

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


from scipy.special import lambertw


def _oracle_max_time_step(rho: float, delta_max: float, eps: float = 1.11e-16) -> float:
    r = float(rho)
    dm = float(delta_max)
    ep = float(eps)
    if not np.isfinite(r) or r < 1.0:
        raise ValueError("rho must be finite and >= 1")
    if not np.isfinite(dm) or dm <= 0.0:
        raise ValueError("delta_max must be finite and > 0")
    if not np.isfinite(ep) or ep <= 0.0:
        raise ValueError("eps must be finite and > 0")

    w = float(np.real(lambertw(dm / (4.0 * ep), 0)))

    return float(2.0 * w / r)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the task configuration ---
        {
            "setup": """import numpy as np
rho = 3.0747727084867518
delta_max = 1e-12
eps = 1.11e-16
""",
            "call": "max_time_step(rho, delta_max, eps)",
            "gold_call": "_oracle_max_time_step(rho, delta_max, eps)",
        },
        # --- Normal: a tighter tolerance shortens the admissible step ---
        {
            "setup": """import numpy as np
rho = 3.0747727084867518
delta_max = 1e-14
eps = 1.11e-16
""",
            "call": "max_time_step(rho, delta_max, eps)",
            "gold_call": "_oracle_max_time_step(rho, delta_max, eps)",
        },
        # --- Boundary: rho = 1, the degenerate ellipse [-1, 1] ---
        {
            "setup": """import numpy as np
rho = 1.0
delta_max = 1e-12
eps = 1.11e-16
""",
            "call": "max_time_step(rho, delta_max, eps)",
            "gold_call": "_oracle_max_time_step(rho, delta_max, eps)",
        },
        # --- Edge: the Lambert W argument falls below 1, where W grows slowly ---
        {
            "setup": """import numpy as np
rho = 5.0
delta_max = 1e-16
eps = 1.11e-16
""",
            "call": "max_time_step(rho, delta_max, eps)",
            "gold_call": "_oracle_max_time_step(rho, delta_max, eps)",
        },
        # --- Edge: default eps argument ---
        {
            "setup": """import numpy as np
rho = 2.379795897113
delta_max = 1e-12
""",
            "call": "max_time_step(rho, delta_max)",
            "gold_call": "_oracle_max_time_step(rho, delta_max)",
        },
        # --- Invalid: rho below 1 ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = max_time_step
    try:
        _fn(0.5, 1e-12, 1.11e-16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_max_time_step
    try:
        _fn(0.5, 1e-12, 1.11e-16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: non-positive tolerance ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = max_time_step
    try:
        _fn(2.0, 0.0, 1.11e-16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_max_time_step
    try:
        _fn(2.0, 0.0, 1.11e-16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
