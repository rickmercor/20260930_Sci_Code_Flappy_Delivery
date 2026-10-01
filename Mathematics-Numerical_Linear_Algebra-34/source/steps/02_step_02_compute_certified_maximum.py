"""
Compute a certified upper estimate of the maximum of the polynomial over the box.

The construction downstream requires a value $\bar g$ with $\bar g \ge \max_B g$, where $B = [-1,1]^n$. Determine such a value from the coefficients alone, using an estimate that depends on their magnitudes and not on their exponents. No optimization, sampling, or grid search is permitted.

The estimate need not be tight. Every bound produced downstream remains valid when $\bar g$ is replaced by any larger value, so a cheap certified over-estimate suffices and is preferred to an expensive exact maximization.

On the centred box every coordinate satisfies $|x_i| \le 1$, so every monomial is bounded by $1$ in absolute value irrespective of its exponent vector. Any estimate derived from that fact alone, together with the coefficient magnitudes, is admissible provided it is genuinely an upper bound for every polynomial with those magnitudes. Such an estimate is attained exactly when the monomials can be made simultaneously extremal at a common corner, and is otherwise strictly loose.

Looseness is inexpensive by design. The quantity enters only through an interval that must contain the support of a certain residual measure; enlarging that interval preserves the sign conditions the construction relies on, and degrades convergence constants without invalidating any result.

A sampled or grid-based maximum would be inadmissible for the opposite reason: it can only underestimate, which is the wrong direction entirely. Sharper certified estimates exist at comparable cost, for instance a single interval-arithmetic evaluation over the box, and a near-optimal value can be obtained by global polynomial optimization at a cost independent of the downstream budget.

Returns
-------
float: a certified upper estimate gbar of the maximum of g over the box B = [-1, 1]^n.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_certified_maximum(coeffs: np.ndarray, exponents: np.ndarray) -> float:
    """Compute a certified upper estimate of the maximum of g over the unit box.

    Parameters
    ----------
    coeffs : np.ndarray
        Array of shape (s,) and dtype float64 holding the coefficients of
        the s monomials of g.
    exponents : np.ndarray
        Integer array of shape (s, n). Row i holds the exponent vector of
        the i-th monomial, matching coeffs[i].

    Returns
    -------
    float
        A value gbar satisfying gbar >= max over B of g, where
        B = [-1, 1]^n. The estimate is certified but need not be tight.

    Raises
    ------
    ValueError
        If the polynomial has no monomials, if the arrays are not
        one- and two-dimensional respectively, or if they disagree in
        length.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_certified_maximum(coeffs: np.ndarray, exponents: np.ndarray) -> float:
    """Reference implementation: bound each monomial by one on the box."""
    c = np.asarray(coeffs, dtype=np.float64)
    E = np.asarray(exponents, dtype=np.int64)

    if c.ndim != 1 or E.ndim != 2:
        raise ValueError("coeffs must be one-dimensional and exponents two-dimensional")
    if c.size != E.shape[0] or c.size == 0:
        raise ValueError("coeffs and exponents must be non-empty and agree in length")

    return float(np.abs(c).sum())

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential test cases: normal, boundary, and edge instances."""
    return [
        {
            # normal: the task instance, where the negative cross term makes
            # the estimate strictly loose against a true maximum of 4
            "setup": "import numpy as np\ncoeffs = np.array([2.0, 2.0, 2.0, -2.0])\nexponents = np.array([[8, 0, 0], [0, 4, 0], [0, 0, 2], [4, 2, 0]])",
            "call": "compute_certified_maximum(coeffs, exponents)",
            "gold_call": "_oracle_compute_certified_maximum(coeffs, exponents)",
        },
        {
            # boundary: all coefficients positive, so the estimate is
            # attained at a corner and is exactly the maximum
            "setup": "import numpy as np\ncoeffs = np.array([1.0, 1.0])\nexponents = np.array([[4, 0], [0, 2]])",
            "call": "compute_certified_maximum(coeffs, exponents)",
            "gold_call": "_oracle_compute_certified_maximum(coeffs, exponents)",
        },
        {
            # edge: a single monomial, the smallest possible polynomial
            "setup": "import numpy as np\ncoeffs = np.array([1.0])\nexponents = np.array([[4, 0]])",
            "call": "compute_certified_maximum(coeffs, exponents)",
            "gold_call": "_oracle_compute_certified_maximum(coeffs, exponents)",
        },
        {
            # edge: coefficients of mixed sign and differing magnitude, where
            # the signed total is 5.0 against a correct estimate of 17.0
            "setup": "import numpy as np\ncoeffs = np.array([5.0, -7.0, 3.0, -2.0])\nexponents = np.array([[8, 0, 0], [0, 4, 0], [0, 0, 2], [4, 2, 0]])",
            "call": "compute_certified_maximum(coeffs, exponents)",
            "gold_call": "_oracle_compute_certified_maximum(coeffs, exponents)",
        },
    ]
