"""
Evaluate the Bernoulli function B(t) = t/(e^t - 1) element-wise, accurately for every real t.

The Scharfetter-Gummel idea behind every flux in this task is to solve the one-dimensional drift-diffusion equation exactly along an edge on which the electrostatic potential varies linearly. The resulting flux weights the densities at the two ends with the Bernoulli function of the potential drop measured in thermal voltages, B(t) = t/(e^t - 1). Near a PN junction the drop across one edge can be tens of thermal voltages, while in neutral regions it is close to zero, so both limits occur in the same solve. The direct formula fails at both: it is 0/0 at t = 0 and e^t overflows in double precision for t above about 709. The function must return B(0) = 1, stay accurate for |t| of order 1e-9, and return finite values for |t| of several hundred, where B(t) tends to 0 for large positive t and to -t for large negative t. The identity B(-t) = B(t) + t is a useful check, and it is what makes the scheme reproduce thermal equilibrium exactly.

Returns
-------
np.ndarray, same shape as t, float: B(t) = t/(e^t - 1) with B(0) = 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bernoulli(t: "np.ndarray") -> "np.ndarray":
    """Bernoulli function B(t) = t / (exp(t) - 1), applied element-wise.

    Parameters
    ----------
    t : "np.ndarray"
        Real array of any shape (a potential drop divided by the thermal voltage).

    Returns
    -------
    B : "np.ndarray"
        Same shape as t, float. B(0) = 1. Must be accurate near t = 0 and must
        not overflow or return nan for |t| up to at least 800.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return B  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bernoulli(t: "np.ndarray") -> "np.ndarray":
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    small = np.abs(t) < 1e-8
    out[small] = 1.0 - 0.5 * t[small]
    tb = t[~small]
    res = np.empty_like(tb)
    pos = tb > 0
    # t > 0: t e^{-t} / (1 - e^{-t}) cannot overflow; t < 0: t / expm1(t)
    res[pos] = tb[pos] * np.exp(-tb[pos]) / (-np.expm1(-tb[pos]))
    res[~pos] = tb[~pos] / np.expm1(tb[~pos])
    out[~small] = res
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: moderate potential drops across an edge.
        {"setup": "import numpy as np\nt = np.linspace(-5.0, 5.0, 11)\n",
         "call": "bernoulli(t)",
         "gold_call": "_oracle_bernoulli(t)"},
        # Edge: the removable singularity at t = 0 and its neighbourhood.
        {"setup": "import numpy as np\nt = np.array([-1e-12, -1e-9, 0.0, 1e-9, 1e-12, 1e-6, -1e-6])\n",
         "call": "bernoulli(t)",
         "gold_call": "_oracle_bernoulli(t)"},
        # Boundary: very large drops that overflow exp in the naive formula.
        {"setup": "import numpy as np\nt = np.array([-800.0, -60.0, 60.0, 709.5, 800.0])\n",
         "call": "bernoulli(t)",
         "gold_call": "_oracle_bernoulli(t)"},
        # Normal: the identity B(-t) - B(t) = t on a 2D array.
        {"setup": "import numpy as np\nt = np.linspace(-40.0, 40.0, 24).reshape(4, 6)\n",
         "call": "bernoulli(-t) - bernoulli(t)",
         "gold_call": "_oracle_bernoulli(-t) - _oracle_bernoulli(t)"},
    ]
