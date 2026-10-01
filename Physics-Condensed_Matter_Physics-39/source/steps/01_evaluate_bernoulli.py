"""
Evaluate the Bernoulli transport factor for finite real input values.

Exponentially fitted semiconductor transport uses this scalar factor in its
carrier terms.  This contract fixes elementwise evaluation, shape preservation,
finite-input validation, and numerical accuracy throughout the tested argument
range, including values close to zero.  The factor's mathematical definition is
part of the selected scientific method.

Returns
-------
np.ndarray with the same shape as arguments, containing the Bernoulli transport-factor values.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_bernoulli(arguments: "NDArray[np.float64]") -> "NDArray[np.float64]":
    """Evaluate the Bernoulli function stably and elementwise.

    Parameters
    ----------
    arguments : np.ndarray
        Finite real scalar or array of Bernoulli arguments.

    Returns
    -------
    values : np.ndarray
        Array with the same shape as ``arguments`` containing the corresponding
        Bernoulli transport-factor values.

    Raises
    ------
    ValueError
        If any input value is nonfinite.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import NDArray


def _oracle_evaluate_bernoulli(
    arguments: NDArray[np.float64],
) -> NDArray[np.float64]:
    t = np.asarray(arguments, dtype=np.float64)
    if not np.all(np.isfinite(t)):
        raise ValueError("arguments must contain only finite values")

    values = np.empty_like(t, dtype=np.float64)
    small = np.abs(t) < 1.0e-7
    x = t[small]
    values[small] = 1.0 - x / 2.0 + x * x / 12.0 - x**4 / 720.0 + x**6 / 30240.0
    values[~small] = t[~small] / np.expm1(t[~small])
    return values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return numerically diverse Bernoulli test cases."""
    return [
        {
            "setup": "import numpy as np\nt=np.array([-40.0,-3.0,-0.4,0.0,0.7,4.0,40.0],dtype=float)",
            "call": "np.round(evaluate_bernoulli(t.copy()), 14)",
            "gold_call": "np.round(_oracle_evaluate_bernoulli(t.copy()), 14)",
        },
        {
            "setup": "import numpy as np\nt=np.array([-1e-12,-1e-9,-1e-8,0.0,1e-8,1e-9,1e-12],dtype=float)",
            "call": "np.round(evaluate_bernoulli(t.copy()), 15)",
            "gold_call": "np.round(_oracle_evaluate_bernoulli(t.copy()), 15)",
        },
        {
            "setup": "import numpy as np\nt=np.array([[-12.0,0.25,1e-10],[1.5,-0.75,12.0]],dtype=float)",
            "call": "np.round(evaluate_bernoulli(t.copy()), 13)",
            "gold_call": "np.round(_oracle_evaluate_bernoulli(t.copy()), 13)",
        },
        {
            "setup": "import numpy as np\nt=np.linspace(-5.0,5.0,17,dtype=float).reshape(1,17)",
            "call": "np.round(evaluate_bernoulli(t.copy()), 13)",
            "gold_call": "np.round(_oracle_evaluate_bernoulli(t.copy()), 13)",
        },
    ]
