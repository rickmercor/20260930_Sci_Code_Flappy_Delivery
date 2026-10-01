"""
Generate the Chebyshev expansion coefficients of the propagator exp(-i t H) up to

a given order, for a bare time step t.



Inputs

------

dt: float, the time step t, > 0

max_order: int, highest expansion order retained, >= 1



Returns

-------

coefficients: (max_order + 1,) complex ndarray, entries c_0 .. c_max_order



Raises

------

ValueError: if dt is not positive and finite, or max_order is below 1

The propagator of a time-independent generator admits an expansion in Chebyshev

polynomials of the first kind whose coefficients are Bessel functions of the first kind.

Because it is built from the Taylor expansion of an entire function it converges for every

argument in the complex plane, not merely on [-1, 1].



The same statement carries over to a matrix argument. Any matrix can be brought to Jordan

normal form, an analytic function of a Jordan block is fixed by the derivatives of that

function at the eigenvalue, and the expansion is valid at every eigenvalue, so it is valid

for the matrix itself. That is what allows the method to be applied to a non-Hermitian

generator with a complex spectrum.



The coefficients depend on the time step alone. They carry an order-dependent phase, and

the lowest order is not scaled the same way as the rest, so the sequence is not simply a

list of Bessel values.



Because the Bessel functions decay factorially once the order exceeds roughly the time

step, the coefficients fall off super-exponentially past that point. That decay is what

lets a truncated sum converge even though the Chebyshev polynomials themselves grow

geometrically off the interval: the factorial in the denominator eventually beats any

geometric growth.

Returns
-------
np.ndarray, complex array of shape (max_order + 1,) holding the Chebyshev expansion coefficients of the propagator for the given time step, ordered from order 0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.special import jv


def chebyshev_coefficients(dt: float, max_order: int) -> np.ndarray:
    '''Build the Chebyshev expansion coefficients of exp(-i t H).

    Parameters
    ----------
    dt : float
        Time step t entering the Bessel functions, must be > 0.
    max_order : int
        Highest Chebyshev order retained, must be >= 1.

    Returns
    -------
    coefficients : np.ndarray
        Complex array of shape (max_order + 1,) holding c_0 .. c_max_order.

    Raises
    ------
    ValueError
        Raised if dt is not positive and finite, or max_order is below 1.
    '''
    return coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import jv


def _oracle_chebyshev_coefficients(dt: float, max_order: int) -> np.ndarray:
    t = float(dt)
    if not np.isfinite(t) or t <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if not isinstance(max_order, (int, np.integer)) or int(max_order) < 1:
        raise ValueError("max_order must be an integer >= 1")

    m = np.arange(int(max_order) + 1)
    coefficients = 2.0 * ((-1j) ** m) * jv(m, t)
    coefficients[0] = jv(0, t)

    return coefficients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the time step used by the task configuration ---
        {
            "setup": """import numpy as np
dt = 3.3333333333333335
max_order = 40
""",
            "call": "chebyshev_coefficients(dt, max_order)",
            "gold_call": "_oracle_chebyshev_coefficients(dt, max_order)",
        },
        # --- Normal: a longer step needs a deeper expansion ---
        {
            "setup": """import numpy as np
dt = 8.0
max_order = 60
""",
            "call": "chebyshev_coefficients(dt, max_order)",
            "gold_call": "_oracle_chebyshev_coefficients(dt, max_order)",
        },
        # --- Boundary: the shortest useful expansion, orders 0 and 1 only ---
        {
            "setup": """import numpy as np
dt = 1.0
max_order = 1
""",
            "call": "chebyshev_coefficients(dt, max_order)",
            "gold_call": "_oracle_chebyshev_coefficients(dt, max_order)",
        },
        # --- Edge: very small step, where all higher coefficients underflow toward zero ---
        {
            "setup": """import numpy as np
dt = 1e-06
max_order = 12
""",
            "call": "chebyshev_coefficients(dt, max_order)",
            "gold_call": "_oracle_chebyshev_coefficients(dt, max_order)",
        },
        # --- Edge: deep truncation far beyond the Bessel turning point ---
        {
            "setup": """import numpy as np
dt = 2.5
max_order = 120
""",
            "call": "chebyshev_coefficients(dt, max_order)",
            "gold_call": "_oracle_chebyshev_coefficients(dt, max_order)",
        },
        # --- Invalid: non-positive time step ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = chebyshev_coefficients
    try:
        _fn(0.0, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_chebyshev_coefficients
    try:
        _fn(0.0, 10)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: max_order below 1 ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = chebyshev_coefficients
    try:
        _fn(1.0, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_chebyshev_coefficients
    try:
        _fn(1.0, 0)
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
