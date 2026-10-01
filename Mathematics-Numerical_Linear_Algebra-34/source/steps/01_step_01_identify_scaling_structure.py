"""
Recover the scaling structure of a polynomial from its exponent pattern alone.

There is a vector of positive integer weights $w$ and a positive integer $m$ for which

$$g(t^{w_1}x_1, \dots, t^{w_n}x_n) = t^{m} g(x), \qquad t > 0.$$

Determine the unique such pair in which the entries share no common factor, and return the weights followed by the degree. The coefficients of $g$ play no role: only the exponent pattern matters.

Raise a ValueError if the exponent array is empty or not two-dimensional, if the solution space is not one-dimensional, or if no strictly positive integer solution exists.

Requiring $\langle w, \alpha \rangle = m$ for every monomial exponent $\alpha$ of $g$ gives a homogeneous linear system in the $n+1$ unknowns $(w_1, \dots, w_n, m)$, with coefficient matrix $[A \mid -\mathbf{1}]$ whose rows are the exponent vectors. When $g$ genuinely possesses such a scaling the solution space is one-dimensional, so the pair is determined only up to a common scalar, and a canonical representative must be selected. The isotropic case $w = (1, \dots, 1)$ arises exactly when every monomial has the same total degree.

The distinction between the weight sum $\sum_i w_i$ and the ambient dimension $n$ becomes important downstream. In the isotropic case the two coincide; in general they do not, and the scaling structure rather than the dimension is what governs the behaviour of sublevel volumes.

The system has integer data, so elimination over $\mathbb{Q}$ is exact and the output is bitwise reproducible on any platform.

Returns
-------
np.ndarray of shape (n + 1,), dtype float64: [w_1, ..., w_n, m].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def identify_scaling_structure(exponents: np.ndarray) -> np.ndarray:
    """Recover the scaling weights and weighted degree of a polynomial.

    Parameters
    ----------
    exponents : np.ndarray
        Integer array of shape (s, n). Row i holds the exponent vector
        alpha_i of the i-th monomial of g. Coefficients are not supplied
        and are not needed.

    Returns
    -------
    np.ndarray
        Array of shape (n + 1,) and dtype float64, holding
        [w_1, ..., w_n, m]: the positive integer weights and weighted
        degree, in the representative whose entries share no common
        factor.

    Raises
    ------
    ValueError
        If the exponent array is empty or not two-dimensional, if the
        solution space is not one-dimensional, or if no strictly positive
        integer solution exists.
    """
    return np.zeros(exponents.shape[1] + 1, dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction
from math import gcd


def _oracle_identify_scaling_structure(exponents: np.ndarray) -> np.ndarray:
    """Reference implementation: exact rational nullspace of [A | -1]."""
    E = np.asarray(exponents, dtype=np.int64)
    if E.ndim != 2 or E.shape[0] == 0:
        raise ValueError("exponents must be a non-empty two-dimensional array")

    n_mono, n_var = E.shape
    rows = [[Fraction(int(E[i, j])) for j in range(n_var)] + [Fraction(-1)]
            for i in range(n_mono)]
    n_col = n_var + 1

    piv, r = [], 0
    for col in range(n_col):
        p = next((i for i in range(r, n_mono) if rows[i][col] != 0), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        pv = rows[r][col]
        rows[r] = [v / pv for v in rows[r]]
        for i in range(n_mono):
            if i != r and rows[i][col] != 0:
                f = rows[i][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        piv.append(col)
        r += 1
        if r == n_mono:
            break

    free = [col for col in range(n_col) if col not in piv]
    if len(free) != 1:
        raise ValueError("scaling structure is not uniquely determined")

    f = free[0]
    sol = [Fraction(0)] * n_col
    sol[f] = Fraction(1)
    for i, col in enumerate(piv):
        sol[col] = -rows[i][f]

    den = 1
    for v in sol:
        den = den * v.denominator // gcd(den, v.denominator)
    ints = [int(v * den) for v in sol]

    g = 0
    for v in ints:
        g = gcd(g, abs(v))
    ints = [v // g for v in ints]

    if ints[-1] < 0:
        ints = [-v for v in ints]
    if any(v <= 0 for v in ints):
        raise ValueError("no strictly positive integer solution exists")

    return np.array(ints, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential test cases: normal, boundary, and edge instances."""
    return [
        {
            # normal: the task instance, anisotropic weights
            "setup": "import numpy as np\nexponents = np.array([[8, 0, 0], [0, 4, 0], [0, 0, 2], [4, 2, 0]])",
            "call": "identify_scaling_structure(exponents)",
            "gold_call": "_oracle_identify_scaling_structure(exponents)",
        },
        {
            # boundary: isotropic scaling, where the weight sum equals the
            # ambient dimension
            "setup": "import numpy as np\nexponents = np.array([[4, 0], [0, 4], [2, 2]])",
            "call": "identify_scaling_structure(exponents)",
            "gold_call": "_oracle_identify_scaling_structure(exponents)",
        },
        {
            # edge: the canonical representative requires reduction by a
            # common factor, and no weight equals 1
            "setup": "import numpy as np\nexponents = np.array([[9, 0, 0], [0, 6, 0], [0, 0, 3], [3, 4, 0]])",
            "call": "identify_scaling_structure(exponents)",
            "gold_call": "_oracle_identify_scaling_structure(exponents)",
        },
        {
            # edge: four variables with a redundant cross monomial, an
            # overdetermined but consistent system
            "setup": "import numpy as np\nexponents = np.array([[12, 0, 0, 0], [0, 6, 0, 0], [0, 0, 4, 0], [0, 0, 0, 3], [6, 3, 0, 0]])",
            "call": "identify_scaling_structure(exponents)",
            "gold_call": "_oracle_identify_scaling_structure(exponents)",
        },
    ]
