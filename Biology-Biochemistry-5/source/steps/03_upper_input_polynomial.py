"""
Construct the polynomial upper bound of the source on the generator applied to the monomial x^mu for the dimerization network, returned as the coefficient vector [c_0, c_1, c_2] of c_0 + c_1 x + c_2 x^2. Use the source's constructive expression, which treats the reaction classes differently; do not return the exact generator action.

The generator applied to a monomial is a polynomial in the copy number whose degree can exceed the monomial's, which is the root of the unclosed moment hierarchy. The source's construction replaces it by a polynomial of the same degree that dominates it on the nonnegative integers, and it is exactly which terms are kept that gives a valid bound.

Returns
-------
ndarray of float64, shape (3,), coefficients [c_0, c_1, c_2] of the upper polynomial.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def upper_input_polynomial(theta: "np.ndarray", mu: int) -> "np.ndarray":
    """Construct the polynomial upper bound of the source on the generator applied to the monomial x^mu for the dimerization network, returned as the coefficient vector [c_0, c_1, c_2] of c_0 + c_1 x + c_2 x^2. Use the source's constructive expression, which treats the reaction classes differently; do not return the exact generator action.

    Parameters
    ----------
    theta : np.ndarray
        The three rate constants (theta_1, theta_2, theta_3).
    mu : int
        Monomial order, 1 or 2.

    Returns
    -------
    coeffs : np.ndarray
        Coefficient vector [c_0, c_1, c_2].

    Raises
    ------
    ValueError
        If theta is not three finite positive rates, or mu is not 1 or 2.
    """
    return coeffs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import comb
from scipy.linalg import expm


def _check_theta(theta):
    th = np.asarray(theta, dtype=np.float64)
    if th.shape != (3,):
        raise ValueError("theta must be the three rate constants (theta_1, theta_2, theta_3)")
    if not np.all(np.isfinite(th)) or np.any(th <= 0.0):
        raise ValueError("every rate constant must be finite and strictly positive")
    return th


def _shift_poly(p, k):
    """Multiply the polynomial coefficient vector p (x^0, x^1, x^2) by x^k, keeping degree <= 2."""
    out = np.zeros(3)
    for i, c in enumerate(p):
        if i + k <= 2:
            out[i + k] += c
    return out


def _oracle_upper_input_polynomial(theta: "np.ndarray", mu: int) -> "np.ndarray":
    """Theorem 2, eq. (19): coefficients [c_0, c_1, c_2] of h+_mu(x) for mu in {1, 2}.

    First-order terms mu * s_j * lambda_j(x) * x^(mu-1) are summed over the
    NON-bimolecular reactions only (s_3 <= 0 makes the dropped term nonpositive);
    the |nu| >= 2 binomial terms C(mu, nu) s_j^nu lambda_j(x) x^(mu-nu) run over
    every reaction. Propensities are polynomials in x, so h+ is a polynomial.
    """
    th = _check_theta(theta)
    if isinstance(mu, bool) or int(mu) != mu or int(mu) not in (1, 2):
        raise ValueError("mu must be 1 or 2")
    mu = int(mu)
    th1, th2, th3 = th
    # lambda_j(x) as polynomial coefficient vectors [x^0, x^1, x^2]
    lam_poly = {1: np.array([th1, 0.0, 0.0]),
                2: np.array([0.0, th2, 0.0]),
                3: np.array([0.0, -th3, th3])}
    stoich = {1: 1.0, 2: -1.0, 3: -2.0}
    bimolecular = {3}

    h = np.zeros(3)
    for j in (1, 2, 3):
        if j not in bimolecular:
            h += mu * stoich[j] * _shift_poly(lam_poly[j], mu - 1)        # first-order, non-bimolecular
        for nu in range(2, mu + 1):                                # |nu| >= 2 terms, all reactions
            h += comb(mu, nu) * stoich[j] ** nu * _shift_poly(lam_poly[j], mu - nu)
    return h

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nmu = 1\n",
            "call": "np.asarray(upper_input_polynomial(theta, mu))",
            "gold_call": "np.asarray(_oracle_upper_input_polynomial(theta, mu))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nmu = 2\n",
            "call": "np.asarray(upper_input_polynomial(theta, mu))",
            "gold_call": "np.asarray(_oracle_upper_input_polynomial(theta, mu))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([12.0, 0.5, 0.3])\nmu = 2\n",
            "call": "np.asarray(upper_input_polynomial(theta, mu))",
            "gold_call": "np.asarray(_oracle_upper_input_polynomial(theta, mu))",
        },
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\ndef run_model():\n    try:\n        upper_input_polynomial(theta, 3)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_upper_input_polynomial(theta, 3)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
