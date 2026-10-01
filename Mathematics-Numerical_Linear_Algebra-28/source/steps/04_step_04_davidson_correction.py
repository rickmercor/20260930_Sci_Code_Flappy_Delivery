"""
Generate the numerical block of new search directions from the spectral information produced by Step 03. The new directions are constructed in the original problem space and preserve the ordering and number of retained states from the preceding stage. The resulting block is returned appended to the Step 03 state; the next search-space construction stage receives it as (A, S, Vt, Q, W, T).

Subspace eigensolvers improve their approximations by expanding the current search space with directions derived from the quality of the present approximate states. These directions are designed to expose components that are not yet adequately represented in the current subspace, allowing later spectral extractions to access additional information about the desired portion of the spectrum. In the generalized Davidson framework studied in the paper, the correction stage provides this mechanism for progressively enlarging the search space.

Returns
-------
tuple containing the complete Step 03 numerical state (A, S, Vt, Q, W, theta, Y, U, R) together with the Davidson correction block T generated from the retained Ritz residuals
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def davidson_correction(state: tuple) -> tuple:
    """Construct the next block of search directions.

    Parameters
    ----------
    state : tuple
        Numerical state produced by Step 03 containing the operator,
        sketch, search basis, Ritz information, and residuals.

    Returns
    -------
    tuple
        Numerical state extended with the newly constructed search-direction
        information.

    Raises
    ------
    ValueError
        If the chained state is malformed, dimensions are inconsistent,
        required quantities are non-finite, or a new numerical direction
        cannot be constructed.

    Notes
    -----
    The correction for retained pair i is t_i = (D - theta_i I)^(-1) r_i
    with D = diag(A), applied componentwise. A zero denominator gives a
    zero correction when the residual component is zero, and raises
    ValueError otherwise.

    The returned state must preserve all information required by the
    expansion stage.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_davidson_correction(state: tuple) -> tuple:
    """Reference implementation."""
    if not isinstance(state, tuple) or len(state) != 9:
        raise ValueError("state must be the Step 03 tuple")

    A, S, Vt, Q, W, theta, Y, U, R = [
        np.asarray(x, dtype=np.float64) for x in state
    ]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = int(A.shape[0])

    if S.ndim != 2 or S.shape[1] != n:
        raise ValueError("S must have shape (s,n)")

    if Vt.ndim != 2 or Vt.shape[0] != n:
        raise ValueError("Vt must have shape (n,j)")

    if Q.ndim != 2 or Q.shape[1] != Vt.shape[1]:
        raise ValueError("Q and Vt dimensions are inconsistent")

    if W.ndim != 2 or W.shape != Vt.shape:
        raise ValueError("W must have the same shape as Vt")

    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional")

    k = int(theta.size)

    if k < 1:
        raise ValueError("theta must contain at least one value")

    if Y.ndim != 2 or Y.shape != (Vt.shape[1], k):
        raise ValueError("Y has inconsistent shape")

    if U.ndim != 2 or U.shape != (n, k):
        raise ValueError("U has inconsistent shape")

    if R.ndim != 2 or R.shape != (n, k):
        raise ValueError("R has inconsistent shape")

    if not all(
        np.all(np.isfinite(x))
        for x in (A, S, Vt, Q, W, theta, Y, U, R)
    ):
        raise ValueError("state contains non-finite values")

    diagonal = np.diag(A)

    denom = diagonal[:, None] - theta[None, :]
    singular = np.abs(denom) <= 1e-14

    # A zero denominator is admissible only where the corresponding residual
    # component is numerically zero.
    if np.any(singular & (np.abs(R) > 1e-14)):
        raise ValueError(
            "singular correction denominator for a nonzero residual"
        )

    T = np.zeros_like(R, dtype=np.float64)
    valid = ~singular
    T[valid] = R[valid] / denom[valid]

    if not np.all(np.isfinite(T)):
        raise ValueError("correction block contains non-finite values")

    return A, S, Vt, Q, W, theta, Y, U, R, T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    """Return normal, boundary, edge, and invalid test specifications."""

    return [

        # Normal: two retained Ritz states with ordinary, nonzero denominators
        # and residuals generated from the supplied Ritz data.
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "A = np.diag([4.0, 5.0, 6.0, 7.0])\n"
                "S = np.eye(4, dtype=float)\n"
                "V = np.eye(4, dtype=float)[:, :2]\n"
                "Q = V.copy()\n"
                "W = A @ V\n"
                "theta = np.array([2.5, 3.5])\n"
                "Y = np.eye(2, dtype=float)\n"
                "U = V.copy()\n"
                "R = W @ Y - U * theta[None, :]\n"
                "state = (A, S, V, Q, W, theta, Y, U, R)\n"
            ),
            "call": "davidson_correction(copy.deepcopy(state))",
            "gold_call": "_oracle_davidson_correction(state)",
        },

        # Boundary: two retained states lie exactly on diagonal entries of A.
        # The corresponding singular-denominator residual components are zero,
        # while other residual components remain nonzero and must be corrected.
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "A = np.diag([2.0, 3.0, 5.0, 7.0])\n"
                "S = np.eye(4, dtype=float)\n"
                "c = 1.0 / np.sqrt(2.0)\n"
                "V = np.array([[c, 0.0], [0.0, c], [c, 0.0], [0.0, c]])\n"
                "Q = V.copy()\n"
                "W = A @ V\n"
                "theta = np.array([2.0, 3.0])\n"
                "Y = np.eye(2, dtype=float)\n"
                "U = V.copy()\n"
                "R = W @ Y - U * theta[None, :]\n"
                "state = (A, S, V, Q, W, theta, Y, U, R)\n"
            ),
            "call": "davidson_correction(copy.deepcopy(state))",
            "gold_call": "_oracle_davidson_correction(state)",
        },

        # Edge: a two-column correction block with mixed signs and unequal
        # diagonal shifts, exercising componentwise correction values that are
        # nonzero in both retained directions.
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "A = np.diag([1.0, 2.0, 4.0, 6.0])\n"
                "S = np.eye(4, dtype=float)\n"
                "V = np.array([\n"
                "    [1.0, 0.0],\n"
                "    [0.0, 1.0],\n"
                "    [1.0, 1.0],\n"
                "    [0.0, -1.0],\n"
                "], dtype=float) / np.sqrt(2.0)\n"
                "Q = V.copy()\n"
                "W = A @ V\n"
                "theta = np.array([3.0, 5.0])\n"
                "Y = np.eye(2, dtype=float)\n"
                "U = V.copy()\n"
                "R = W @ Y - U * theta[None, :]\n"
                "state = (A, S, V, Q, W, theta, Y, U, R)\n"
            ),
            "call": "davidson_correction(copy.deepcopy(state))",
            "gold_call": "_oracle_davidson_correction(state)",
        },

        # Invalid: the correction denominator is singular for a nonzero
        # residual component, so both model and oracle must reject the state.
        {
            "setup": (
                "import copy\nimport numpy as np\n"
                "A = np.diag([2.0, 3.0, 5.0, 7.0])\n"
                "S = np.eye(4, dtype=float)\n"
                "c = 1.0 / np.sqrt(2.0)\n"
                "V = np.array([[c, 0.0], [0.0, c], [c, 0.0], [0.0, c]])\n"
                "Q = V.copy()\n"
                "W = A @ V\n"
                "theta = np.array([2.0, 3.0])\n"
                "Y = np.eye(2, dtype=float)\n"
                "U = V.copy()\n"
                "R = W @ Y - U * theta[None, :]\n"
                "R[0, 0] = 1.0\n"
                "state = (A, S, V, Q, W, theta, Y, U, R)\n"
                "def run_model():\n"
                "    try:\n"
                "        davidson_correction(copy.deepcopy(state))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_davidson_correction(state)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },

    ]
