"""
Compute the moments of a polynomial over the centred unit box.

For the uniform probability measure $\mu$ on $B = [-1,1]^n$, with $d\mu = 2^{-n}\mathbf{1}_B\,dx$, compute

$$y_k = \int_B g(x)^k\,d\mu(x) = 2^{-n}\int_B g(x)^k\,dx, \qquad k = 0, 1, \dots, K.$$

These are the only quantities through which $g$ enters the remainder of the pipeline: every later stage consumes $y_0, \dots, y_K$ and never touches the polynomial again. By construction $y_0 = 1$.

The moments must be correct to full double precision at every index in the budget, including indices where the value is large. Accumulated rounding is not acceptable, and the arithmetic strategy needed to avoid it is part of the problem.

Monomial moments over the centred box factorise across coordinates,

$$\int_B x^{\alpha}\,d\mu = \prod_{i=1}^{n} \ell_{\alpha_i}, \qquad \ell_j = \frac{1}{2}\int_{-1}^{1} t^{j}\,dt,$$

where $\ell_j = 1/(j+1)$ when $j$ is even and $\ell_j = 0$ when $j$ is odd, so any box moment reduces to a product of $n$ one-dimensional moments.

This stage is the only place where the ambient dimension enters the cost. The number of monomials of $g^k$ is bounded by $\binom{k+s-1}{s-1}$ for an $s$-term polynomial, so for sparse $g$ the whole sequence is cheap and the cost is essentially independent of $n$. That is what allows the dimension to be confined to a single preprocessing stage rather than propagating into everything downstream.

The moments grow geometrically in $k$, at a rate set by the range of $g$ over the box, and for polynomials with coefficients of mixed sign the intermediate terms of the expansion are far larger than the totals they sum to. Both effects worsen with the budget, and their interaction determines what arithmetic is adequate at a given index.

This stage assumes nothing about $g$ beyond its being a polynomial. The scaling structure, positivity, and containment of the sublevel set play no role, and monomials with odd exponents contribute nothing rather than being excluded in advance.

Returns
-------
np.ndarray of shape (max_order + 1,), dtype float64: the box moments [y_0, ..., y_K], with y_0 = 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_box_moments(coeffs: np.ndarray, exponents: np.ndarray, max_order: int) -> np.ndarray:
    """Compute moments of a polynomial over the centred unit box.

    Parameters
    ----------
    coeffs : np.ndarray
        Array of shape (s,) and dtype float64 holding the coefficients of
        the s monomials of g. Values are assumed exactly representable.
    exponents : np.ndarray
        Integer array of shape (s, n). Row i holds the exponent vector of
        the i-th monomial, matching coeffs[i].
    max_order : int
        Largest power K to compute. Must be non-negative.

    Returns
    -------
    np.ndarray
        Array of shape (max_order + 1,) and dtype float64 holding
        [y_0, y_1, ..., y_K], where y_k is the integral of g^k against the
        uniform probability measure on [-1, 1]^n. Always y_0 = 1.

    Raises
    ------
    ValueError
        If the polynomial has no monomials, if the arrays are not
        one- and two-dimensional respectively, if they disagree in length,
        or if max_order is negative.
    """
    return np.zeros(max_order + 1, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction


def _oracle_compute_box_moments(coeffs: np.ndarray, exponents: np.ndarray, max_order: int) -> np.ndarray:
    """Reference implementation: exact multinomial expansion, then term-by-term integration."""
    c = np.asarray(coeffs, dtype=np.float64)
    E = np.asarray(exponents, dtype=np.int64)

    if c.ndim != 1 or E.ndim != 2:
        raise ValueError("coeffs must be one-dimensional and exponents two-dimensional")
    if c.size != E.shape[0] or c.size == 0:
        raise ValueError("coeffs and exponents must be non-empty and agree in length")
    if int(max_order) < 0:
        raise ValueError("max_order must be non-negative")

    C = [Fraction(float(v)) for v in c]
    n_var = E.shape[1]

    g_poly = {}
    for coeff, row in zip(C, E):
        key = tuple(int(v) for v in row)
        g_poly[key] = g_poly.get(key, Fraction(0)) + coeff

    def _h_integrate(poly):
        """Integrate a monomial dict against the uniform probability measure on the box."""
        total = Fraction(0)
        for a, coeff in poly.items():
            if any(e % 2 for e in a):
                continue
            t = coeff
            for e in a:
                t /= (e + 1)
            total += t
        return total

    def _h_multiply(p, q):
        """Multiply two polynomials represented as exponent-tuple dicts."""
        out = {}
        for a, ca in p.items():
            for b, cb in q.items():
                k = tuple(x + y for x, y in zip(a, b))
                out[k] = out.get(k, Fraction(0)) + ca * cb
        return out

    cur = {tuple([0] * n_var): Fraction(1)}
    ys = [_h_integrate(cur)]
    for _ in range(int(max_order)):
        cur = _h_multiply(cur, g_poly)
        ys.append(_h_integrate(cur))

    return np.array([float(v) for v in ys], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential test cases: normal, boundary, and edge instances."""
    return [
        {
            # normal: the task instance at the pipeline budget k = 0..6
            "setup": "import numpy as np\ncoeffs = np.array([2.0, 2.0, 2.0, -2.0])\nexponents = np.array([[8, 0, 0], [0, 4, 0], [0, 0, 2], [4, 2, 0]])\nK = 6",
            "call": "compute_box_moments(coeffs, exponents, K)",
            "gold_call": "_oracle_compute_box_moments(coeffs, exponents, K)",
        },
        {
            # boundary: separable positive polynomial with no cross term,
            # so no cancellation occurs anywhere in the expansion
            "setup": "import numpy as np\ncoeffs = np.array([1.0, 1.0])\nexponents = np.array([[4, 0], [0, 2]])\nK = 5",
            "call": "compute_box_moments(coeffs, exponents, K)",
            "gold_call": "_oracle_compute_box_moments(coeffs, exponents, K)",
        },
        {
            # edge: monomials with odd exponents present, on a polynomial
            # that is deliberately not quasi-homogeneous, since this stage
            # must not assume any scaling structure
            "setup": "import numpy as np\ncoeffs = np.array([1.0, 1.0, 1.0, -1.0])\nexponents = np.array([[6, 0, 0], [0, 3, 0], [0, 0, 2], [3, 1, 0]])\nK = 4",
            "call": "compute_box_moments(coeffs, exponents, K)",
            "gold_call": "_oracle_compute_box_moments(coeffs, exponents, K)",
        },
        {
            # edge: an extended budget k = 0..14 on the task instance, where
            # the moments reach the order of 10^5 and the mixed-sign
            # expansion produces intermediate terms far larger than the
            # totals; y_14 = 566314.2595399487
            "setup": "import numpy as np\ncoeffs = np.array([2.0, 2.0, 2.0, -2.0])\nexponents = np.array([[8, 0, 0], [0, 4, 0], [0, 0, 2], [4, 2, 0]])\nK = 14",
            "call": "compute_box_moments(coeffs, exponents, K)",
            "gold_call": "_oracle_compute_box_moments(coeffs, exponents, K)",
        },
    ]
