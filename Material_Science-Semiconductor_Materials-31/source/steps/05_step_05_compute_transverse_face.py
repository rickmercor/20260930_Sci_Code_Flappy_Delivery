"""
Evaluate the signed transverse contribution to the integrated normal flux.

The transverse contribution is

$$

J^t=*\int_*{-\ell}^{\ell}Q(s)\left[j_t(s,H/2)-j_t(s,-H/2)\right]\,ds,

\qquad j_t=\alpha\partial_tu_h-\beta_tu_h.

$$

For $u_h=*\sum_*{a,b}c_{ab}s^at^b$, the endpoint difference equals

$H\sum_a(2\alpha c_{a2}-*\beta_tc_*{a1})s^a$.

Consequently $J^t=H*\sum_*{a=0}^2I_a(2\alpha c_{a2}-*\beta_tc_*{a1})$.

The quadratic transverse curvature contributes even when the source is zero.

Returns
-------
A nine-entry float array representing the transverse contribution to the positive-normal face flux.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_transverse_face(
    coefficients: np.ndarray,
    face_length: float,
    diffusion: float,
    tangent_drift: float,
    moments: np.ndarray,
) -> np.ndarray:
    r"""Evaluate the signed transverse contribution to the integrated normal flux.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape $(9,3,3)$; local basis polynomial tensor.
    face_length : float
        Positive face length $H$.
    diffusion : float
        Positive diffusion coefficient $\alpha$.
    tangent_drift : float
        Finite signed tangent drift $\beta_t$ in the oriented frame.
    moments : np.ndarray
        Shape $(3,)$, signed moments $I_0,I_1,I_2$.

    Returns
    -------
    np.ndarray
        Shape $(9,)$; transverse contribution to the positive-normal flux.

    Raises
    ------
    ValueError
        If shapes or finite-value requirements fail, or face length or diffusion is non-positive.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_transverse_face(
    coefficients: np.ndarray,
    face_length: float,
    diffusion: float,
    tangent_drift: float,
    moments: np.ndarray,
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    coefficients = _finite_array(coefficients, "coefficients", (9, 3, 3))
    length = _positive_scalar(face_length, "face_length")
    alpha = _positive_scalar(diffusion, "diffusion")
    beta = _finite_scalar(tangent_drift, "tangent_drift")
    moments = _finite_array(moments, "moments", (3,))
    return length * (
        (2 * alpha * coefficients[:, :, 2] - beta * coefficients[:, :, 1]) @ moments
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nc = np.arange(81, dtype=float).reshape(9,3,3) / 30\nlength, alpha, beta = .4, .03, -.9\nmoments = np.array([-.04, .001, -.0003])\n",
            "call": "compute_transverse_face(c.copy(), length, alpha, beta, moments.copy())",
            "gold_call": "_oracle_compute_transverse_face(c.copy(), length, alpha, beta, moments.copy())",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nc = np.zeros((9,3,3)); c[:,1,2] = np.arange(1.,10.)\nlength, alpha, beta = .2, .1, 0.\nmoments = np.array([0., .003, 0.])\n",
            "call": "compute_transverse_face(c.copy(), length, alpha, beta, moments.copy())",
            "gold_call": "_oracle_compute_transverse_face(c.copy(), length, alpha, beta, moments.copy())",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nc = np.ones((9,3,3)); c[:,0,1] = np.arange(9.)\nlength, alpha, beta = .6, .001, -3.\nmoments = np.array([.04, .001, .0002])\n",
            "call": "compute_transverse_face(c.copy(), length, alpha, beta, moments.copy())",
            "gold_call": "_oracle_compute_transverse_face(c.copy(), length, alpha, beta, moments.copy())",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nc = np.ones((8,3,3)); length, alpha, beta = .4, .03, 1.\nmoments = np.ones(3)\ndef _exception_code(function):\n    try:\n        function(c.copy(), length, alpha, beta, moments.copy())\n    except ValueError:\n        return 1\n    return 0\n\n",
            "call": "_exception_code(compute_transverse_face)",
            "gold_call": "_exception_code(_oracle_compute_transverse_face)",
        },
    ]
