"""
Evaluate the accumulated floating-point rounding-error bound of a truncated

Chebyshev expansion, given the number of terms actually summed.



Inputs

------

n_terms: int, total number of Chebyshev terms summed, >= 1

dt: float, the time step of a single expansion, > 0

rho: float, Bernstein radius enclosing the spectrum, >= 1

eps: float, machine precision, > 0 (default 1.11e-16)



Returns

-------

bound: float, the accumulated rounding-error bound for the stated number of summed terms



Raises

------

ValueError: if n_terms is below 1, dt is not positive, rho is below 1, or eps is not positive

Each retained term of the expansion is a product of a Bessel coefficient and a Chebyshev

polynomial, and both are stored with a relative error of order the unit roundoff. Summing

many such terms accumulates those errors: adding n numbers each carrying eps-level error

contributes a Wilkinson-style factor proportional to how many were added.



Summing the per-term bounds over all orders produces an exponential series in the product

of the time step and the Bernstein radius. The accumulated bound therefore grows

exponentially in that product and only linearly in the number of terms, which is why

shortening the step, rather than truncating earlier, is what controls the error when the

spectrum lies far outside [-1, 1].



Eliminating the term count analytically gives a closed form in the step and radius alone;

keeping the count that a run actually reached gives the realised budget for that run, which

is what this function evaluates.



For a simulation of several successive steps of equal length, the count passed in is the

total summed across all of them, so the bound accumulates linearly in the number of

steps.

Returns
-------
float, the accumulated rounding-error bound of an expansion that summed the stated number of terms at the stated time step and Bernstein radius, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def rounding_error_bound(n_terms: int, dt: float, rho: float, eps: float = 1.11e-16) -> float:
    '''Evaluate the accumulated Chebyshev rounding-error bound.

    Parameters
    ----------
    n_terms : int
        Total number of Chebyshev terms summed, must be >= 1.
    dt : float
        Length of a single time step, must be > 0.
    rho : float
        Bernstein-ellipse radius enclosing the spectrum, must be >= 1.
    eps : float
        Machine precision used in the bound, must be > 0.

    Returns
    -------
    bound : float
        The accumulated rounding-error bound for the stated number of summed terms.

    Raises
    ------
    ValueError
        Raised if n_terms is below 1, dt is not positive, rho is below 1, or eps is not positive.
    '''
    return bound

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rounding_error_bound(n_terms: int, dt: float, rho: float, eps: float = 1.11e-16) -> float:
    if not isinstance(n_terms, (int, np.integer)) or int(n_terms) < 1:
        raise ValueError("n_terms must be an integer >= 1")
    t = float(dt)
    r = float(rho)
    ep = float(eps)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if not np.isfinite(r) or r < 1.0:
        raise ValueError("rho must be finite and >= 1")
    if not np.isfinite(ep) or ep <= 0.0:
        raise ValueError("eps must be finite and > 0")

    return float(4.0 * ep * int(n_terms) * np.exp(t * r / 2.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the totals reached by the task configuration ---
        {
            "setup": """import numpy as np
n_terms = 186
dt = 3.3333333333333335
rho = 3.0747727084867518
eps = 1.11e-16
""",
            "call": "rounding_error_bound(n_terms, dt, rho, eps)",
            "gold_call": "_oracle_rounding_error_bound(n_terms, dt, rho, eps)",
        },
        # --- Normal: a single expansion rather than the whole simulation ---
        {
            "setup": """import numpy as np
n_terms = 31
dt = 3.3333333333333335
rho = 3.0747727084867518
eps = 1.11e-16
""",
            "call": "rounding_error_bound(n_terms, dt, rho, eps)",
            "gold_call": "_oracle_rounding_error_bound(n_terms, dt, rho, eps)",
        },
        # --- Boundary: rho = 1, spectrum confined to the focal interval ---
        {
            "setup": """import numpy as np
n_terms = 40
dt = 1.0
rho = 1.0
eps = 1.11e-16
""",
            "call": "rounding_error_bound(n_terms, dt, rho, eps)",
            "gold_call": "_oracle_rounding_error_bound(n_terms, dt, rho, eps)",
        },
        # --- Edge: a single term, and the default eps ---
        {
            "setup": """import numpy as np
n_terms = 1
dt = 0.25
rho = 2.5
""",
            "call": "rounding_error_bound(n_terms, dt, rho)",
            "gold_call": "_oracle_rounding_error_bound(n_terms, dt, rho)",
        },
        # --- Edge: large t rho where the exponential dominates ---
        {
            "setup": """import numpy as np
n_terms = 200
dt = 8.0
rho = 4.5
eps = 1.11e-16
""",
            "call": "rounding_error_bound(n_terms, dt, rho, eps)",
            "gold_call": "_oracle_rounding_error_bound(n_terms, dt, rho, eps)",
        },
        # --- Invalid: zero terms ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = rounding_error_bound
    try:
        _fn(0, 1.0, 2.0, 1.11e-16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_rounding_error_bound
    try:
        _fn(0, 1.0, 2.0, 1.11e-16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: non-positive time step ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = rounding_error_bound
    try:
        _fn(10, -1.0, 2.0, 1.11e-16)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_rounding_error_bound
    try:
        _fn(10, -1.0, 2.0, 1.11e-16)
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
