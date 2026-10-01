"""
Step 7: Symmetric finite-difference derivative of a scalar function.

The symmetric difference [f(r0 + dr) - f(r0 - dr)] / (2 dr) has truncation error of order dr^2. With the SDP solved to 1e-11, the estimates at dr = 1e-3, 1e-4, and 1e-5 agree to 1e-7, which confirms convergence.

Returns
-------
# float, the central-difference derivative as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

# =============================================================================
# FUNCTION SIGNATURE
# =============================================================================

import numpy as np
from collections.abc import Callable

def central_difference(
    fn: Callable[[float], float],
    r0: float,
    step: float,
) -> float:
    """Return the symmetric finite-difference derivative of fn at r0.

    Parameters
    ----------
    fn : callable
        Scalar function of one float argument.
    r0 : float
        Finite evaluation point.
    step : float
        Finite positive step; r0 - step must be > 0.

    Returns
    -------
    deriv : float
        (fn(r0 + step) - fn(r0 - step)) / (2 * step) as a native Python float.

    Raises
    ------
    ValueError
        If fn is not callable, r0 or step is non-finite, step <= 0,
        r0 - step <= 0, or fn returns a non-finite value.
    """
    return deriv

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================
import numpy as np
from collections.abc import Callable

def _oracle_central_difference(
    fn: Callable[[float], float],
    r0: float,
    step: float,
) -> float:
    import numpy as np

    r0 = float(r0)
    step = float(step)

    if not callable(fn):
        raise ValueError("fn must be callable")

    if (
        not np.isfinite(r0)
        or not np.isfinite(step)
        or step <= 0.0
        or r0 - step <= 0.0
    ):
        raise ValueError(
            "r0 and step must be finite, step > 0 and r0 - step > 0"
        )

    fp = float(fn(r0 + step))
    fm = float(fn(r0 - step))

    if not np.isfinite(fp) or not np.isfinite(fm):
        raise ValueError("fn returned a non-finite value")

    return float((fp - fm) / (2.0 * step))

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================
def test_cases():
    return [
        # normal: cubic, exact derivative 12 + step^2
        {
            "setup": "import numpy as np\nfn=lambda r: r**3; r0=2.0; step=0.1",
            "call": "central_difference(fn,r0,step)",
            "gold_call": "_oracle_central_difference(fn,r0,step)",
        },

        # boundary: step almost equal to r0
        {
            "setup": "import numpy as np\nfn=np.sin; r0=0.5; step=0.499",
            "call": "central_difference(fn,r0,step)",
            "gold_call": "_oracle_central_difference(fn,r0,step)",
        },

        # edge: constant function, derivative exactly zero
        {
            "setup": "import numpy as np\nfn=lambda r: 7.0; r0=1.0; step=1e-4",
            "call": "central_difference(fn,r0,step)",
            "gold_call": "_oracle_central_difference(fn,r0,step)",
        },
    ]
