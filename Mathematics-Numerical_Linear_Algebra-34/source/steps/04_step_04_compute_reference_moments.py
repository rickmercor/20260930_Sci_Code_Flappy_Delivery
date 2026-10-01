"""
Compute the constants relating the moments of $g$ over its unit sublevel set to the volume of that set.

Let $K = \{x : g(x) \le 1\}$ for a polynomial $g$ possessing the scaling structure identified earlier. Under that structure there exist constants $z_0, z_1, z_2, \dots$, depending only on the weights and the weighted degree, such that

$$\int_K g(x)^k\,dx = z_k \operatorname{vol}K \qquad \text{for every integer } k \ge 0.$$

Determine $z_0, \dots, z_K$ exactly.

Note what these constants are not. They do not depend on the coefficients of $g$, on the ambient dimension as such, or on the volume itself, which remains unknown at this stage. They are fixed rationals, known in advance of any computation involving the polynomial, and they are what makes the volume recoverable from moment data at all.

The existence of such constants is a consequence of the exact scaling symmetry, and it is a strong statement: it says that although $\operatorname{vol}K$ is unknown, every moment of $g$ over $K$ is pinned to it by a known factor. A single unknown scalar therefore controls the entire moment structure of $g$ on its own sublevel set.

The constants follow from the interaction between the scaling relation satisfied by powers of $g$ and the geometry of the region $K$, whose boundary is the level set $\{g = 1\}$. Positivity of $g$ away from the origin makes every positive value a regular value, so that boundary is a smooth compact hypersurface and the standard integral theorems of vector calculus apply on $K$. Carrying the argument through for each $k$ yields the constants in closed form as ratios of small integers built from the weights and the degree.

Two structural features are worth anticipating. The case $k = 0$ is the identity $\int_K dx = \operatorname{vol}K$, so $z_0 = 1$ necessarily, and any candidate expression failing this is wrong. And the sequence is strictly decreasing in $k$, since $g \le 1$ on $K$ forces $\int_K g^k dx$ to shrink as $k$ grows; the rate of that decay is the quantitative content of the identity.

A caution on the isotropic special case. When all weights are equal, the resulting constants coincide with a classical formula that predates the general result and is stated in terms of the ambient dimension. That coincidence is specific to equal weights. For general weights the classical expression is simply incorrect, and substituting it produces a self-consistent but wrong sequence which every downstream stage silently inherits.

The constants are rational, so they are computed exactly and rounded once.

Returns
-------
np.ndarray of shape (max_order + 1,), dtype float64: the constants [z_0, ..., z_K], with z_0 = 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_reference_moments(weights: np.ndarray, weighted_degree: int, max_order: int) -> np.ndarray:
    """Compute the constants relating moments of g over K to the volume of K.

    Parameters
    ----------
    weights : np.ndarray
        Integer array of shape (n,) holding the positive weights
        w = (w_1, ..., w_n) of the scaling structure.
    weighted_degree : int
        The weighted degree m, a positive integer, satisfying
        <w, alpha> = m for every monomial exponent alpha of g.
    max_order : int
        Largest index K to compute. Must be non-negative.

    Returns
    -------
    np.ndarray
        Array of shape (max_order + 1,) and dtype float64 holding
        [z_0, z_1, ..., z_K], where the integral of g^k over K equals
        z_k times the volume of K. Always z_0 = 1.

    Raises
    ------
    ValueError
        If any weight is not a positive integer, if the weighted degree
        is not a positive integer, or if max_order is negative.
    """
    return np.zeros(max_order + 1, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction


def _oracle_compute_reference_moments(weights: np.ndarray, weighted_degree: int, max_order: int) -> np.ndarray:
    """Reference implementation: exact rational evaluation of the weighted moment constants."""
    w = np.asarray(weights, dtype=np.int64)
    if w.size == 0 or np.any(w <= 0):
        raise ValueError("weights must be strictly positive integers")

    m = int(weighted_degree)
    if m < 1:
        raise ValueError("weighted degree must be a positive integer")

    if int(max_order) < 0:
        raise ValueError("max_order must be non-negative")

    weight_sum = Fraction(int(w.sum()))

    out = []
    for k in range(int(max_order) + 1):
        out.append(float(weight_sum / (weight_sum + Fraction(k * m))))

    return np.array(out, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential test cases: normal, boundary, and edge instances."""
    return [
        {
            # normal: the task instance, w = (1, 2, 4) with m = 8, where the
            # scaling data and the ambient dimension differ
            "setup": "import numpy as np\nweights = np.array([1, 2, 4])\nm = 8\nK = 6",
            "call": "compute_reference_moments(weights, m, K)",
            "gold_call": "_oracle_compute_reference_moments(weights, m, K)",
        },
        {
            # boundary: equal weights, the isotropic case in which the
            # constants coincide with the classical dimension-based formula
            "setup": "import numpy as np\nweights = np.array([1, 1])\nm = 4\nK = 5",
            "call": "compute_reference_moments(weights, m, K)",
            "gold_call": "_oracle_compute_reference_moments(weights, m, K)",
        },
        {
            # edge: repeated weights with no entry equal to 1, so neither the
            # count of weights nor their maximum coincides with the governing
            # quantity; w = (2, 2, 3), m = 6
            "setup": "import numpy as np\nweights = np.array([2, 2, 3])\nm = 6\nK = 5",
            "call": "compute_reference_moments(weights, m, K)",
            "gold_call": "_oracle_compute_reference_moments(weights, m, K)",
        },
        {
            # edge: an extended budget k = 0..14 matching the range used
            # downstream, where the constants become small and any error in
            # the governing quantity is amplified relative to the values
            "setup": "import numpy as np\nweights = np.array([1, 2, 4])\nm = 8\nK = 14",
            "call": "compute_reference_moments(weights, m, K)",
            "gold_call": "_oracle_compute_reference_moments(weights, m, K)",
        },
    ]
