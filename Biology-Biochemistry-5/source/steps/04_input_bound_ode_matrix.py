"""
Assemble the constant 3 x 3 matrix of the closed linear system that governs the upper input bounds for the first and second monomials, on the augmented state [u_1, u_2, 1], from the two polynomial coefficient vectors of the previous step. Apply the source's rule for how each cross coefficient couples the two bounds, and its choice of lower input bound for this network.

Once the generator action on each monomial is sandwiched by polynomials of the same degree, the conditional moments at a boundary state obey a finite closed system of linear differential equations. The structure of that system is fixed by the signs of the polynomial coefficients.

Returns
-------
ndarray of float64, shape (3, 3), the augmented-state matrix of the input-bound system.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def input_bound_ode_matrix(coeffs_mu1: "np.ndarray", coeffs_mu2: "np.ndarray") -> "np.ndarray":
    """Assemble the constant 3 x 3 matrix of the closed linear system that governs the upper input bounds for the first and second monomials, on the augmented state [u_1, u_2, 1], from the two polynomial coefficient vectors of the previous step. Apply the source's rule for how each cross coefficient couples the two bounds, and its choice of lower input bound for this network.

    Parameters
    ----------
    coeffs_mu1 : np.ndarray
        Coefficients [c_0, c_1, c_2] of the first-order upper polynomial.
    coeffs_mu2 : np.ndarray
        Coefficients [c_0, c_1, c_2] of the second-order upper polynomial.

    Returns
    -------
    M : np.ndarray
        Augmented-state matrix of shape (3, 3).

    Raises
    ------
    ValueError
        If either vector is not three finite entries, or the first-order polynomial has an x^2 term.
    """
    return M

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from math import comb
from scipy.linalg import expm


def _oracle_input_bound_ode_matrix(coeffs_mu1: "np.ndarray", coeffs_mu2: "np.ndarray") -> "np.ndarray":
    """Eqs. (17), (22), (23): constant matrix of the closed LTI for z = [u_1+, u_2+, 1].

    Row mu reads d/dt u_mu+ = c_{mu,mu} u_mu+ + {c_{mu,nu}}^+ u_nu+ + {c_{mu,nu}}^- u_nu- + c_{mu,0},
    with u- = 0 (Sec. IV-A), so a NEGATIVE cross coefficient contributes nothing.
    """
    c1 = np.asarray(coeffs_mu1, dtype=np.float64)
    c2 = np.asarray(coeffs_mu2, dtype=np.float64)
    if c1.shape != (3,) or c2.shape != (3,):
        raise ValueError("each coefficient vector must have exactly three entries [c0, c1, c2]")
    if not (np.all(np.isfinite(c1)) and np.all(np.isfinite(c2))):
        raise ValueError("coefficients must be finite")
    if c1[2] != 0.0:
        raise ValueError("h+_1 must be affine in x (no x^2 term)")
    M = np.zeros((3, 3), dtype=np.float64)
    # u_1+: c_{1,1} u_1 + c_{1,0}
    M[0, 0] = c1[1]
    M[0, 2] = c1[0]
    # u_2+: c_{2,2} u_2 + [c_{2,1}]^+ u_1+ + [c_{2,1}]^- u_1- + c_{2,0};  u_1- = 0
    M[1, 1] = c2[2]
    M[1, 0] = max(c2[1], 0.0)
    M[1, 2] = c2[0]
    return M

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ntheta = np.array([5.0, np.log(2.0) / 20.0, 0.02])\nc1 = _oracle_upper_input_polynomial(theta, 1)\nc2 = _oracle_upper_input_polynomial(theta, 2)\n",
            "call": "np.asarray(input_bound_ode_matrix(c1, c2))",
            "gold_call": "np.asarray(_oracle_input_bound_ode_matrix(c1, c2))",
        },
        {
            "setup": "import numpy as np\nc1 = np.array([1.0, -0.5, 0.0])\nc2 = np.array([1.0, -2.0, 0.25])\n",
            "call": "np.asarray(input_bound_ode_matrix(c1, c2))",
            "gold_call": "np.asarray(_oracle_input_bound_ode_matrix(c1, c2))",
        },
        {
            "setup": "import numpy as np\nc1 = np.array([3.0, -0.2, 0.0])\nc2 = np.array([3.0, 0.7, -0.1])\n",
            "call": "np.asarray(input_bound_ode_matrix(c1, c2))",
            "gold_call": "np.asarray(_oracle_input_bound_ode_matrix(c1, c2))",
        },
        {
            "setup": "import numpy as np\nc1 = np.array([1.0, -0.5, 0.1])\nc2 = np.array([1.0, 2.0, 0.5])\ndef run_model():\n    try:\n        input_bound_ode_matrix(c1, c2)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_input_bound_ode_matrix(c1, c2)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
