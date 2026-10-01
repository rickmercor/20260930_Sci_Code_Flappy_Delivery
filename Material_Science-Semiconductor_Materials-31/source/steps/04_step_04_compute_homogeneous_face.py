"""
Evaluate the homogeneous integrated normal-flux functional of a quadratic element.

For a face of length $H$, let $T_-=*\int_*{-H/2}^{H/2}u_h(-\ell,t)\,dt$ and $T_+=*\int_*{-H/2}^{H/2}u_h(\ell,t)\,dt$.

The homogeneous flux in the positive normal direction is

$$

J^h=*\frac*{\alpha}{2\ell}\left[B(z)T_+-B(-z)T_-\right].

$$

The tangent-power integrals are $(H,0,H^3/12)$.

Return the nine linear-functional coefficients, so their dot product with the element's nodal vector gives $J^h$.

Returns
-------
A nine-entry float array representing the homogeneous face flux as a functional of local nodal values.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_homogeneous_face(
    coefficients: np.ndarray,
    half_width: float,
    face_length: float,
    diffusion: float,
    kernel: np.ndarray,
) -> np.ndarray:
    r"""Evaluate the homogeneous integrated normal-flux functional of a quadratic element.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape $(9,3,3)$; basis coefficients in powers of normal and tangent coordinates.
    half_width : float
        Positive normal half-width $\ell$.
    face_length : float
        Positive tangent length $H$.
    diffusion : float
        Positive diffusion coefficient $\alpha$.
    kernel : np.ndarray
        Shape $(5,)$, $[B(z),B(-z),I_0,I_1,I_2]$ for the same normal interval.

    Returns
    -------
    np.ndarray
        Shape $(9,)$; coefficients of the positive-normal homogeneous flux.

    Raises
    ------
    ValueError
        If any shape or finite-value requirement fails, a length or diffusion is non-positive, or either trace weight is negative.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _face_inputs(coefficients, half_width, face_length, diffusion, kernel):
    coefficients = _finite_array(coefficients, "coefficients", (9, 3, 3))
    ell = _positive_scalar(half_width, "half_width")
    length = _positive_scalar(face_length, "face_length")
    alpha = _positive_scalar(diffusion, "diffusion")
    kernel = _finite_array(kernel, "kernel", (5,))
    if np.any(kernel[:2] < 0):
        raise ValueError("trace weights must be nonnegative")
    return coefficients, ell, length, alpha, kernel


def _oracle_compute_homogeneous_face(
    coefficients: np.ndarray,
    half_width: float,
    face_length: float,
    diffusion: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    coefficients, ell, length, alpha, kernel = _face_inputs(
        coefficients, half_width, face_length, diffusion, kernel
    )
    tangent_integrals = np.array([length, 0.0, length**3 / 12])
    left = np.einsum(
        "kab,a,b->k", coefficients, np.array([1.0, -ell, ell**2]), tangent_integrals
    )
    right = np.einsum(
        "kab,a,b->k", coefficients, np.array([1.0, ell, ell**2]), tangent_integrals
    )
    return alpha / (2 * ell) * (kernel[0] * right - kernel[1] * left)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nc = np.arange(81, dtype=float).reshape(9,3,3) / 40\nell, length, alpha = .1, .4, .03\nkernel = _oracle_compute_normal_kernel(ell, alpha, .3)\n",
            "call": "compute_homogeneous_face(c.copy(), ell, length, alpha, kernel.copy())",
            "gold_call": "_oracle_compute_homogeneous_face(c.copy(), ell, length, alpha, kernel.copy())",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nc = np.zeros((9,3,3)); c[:,1,0] = np.arange(9.)\nell, length, alpha = .2, .5, .1\nkernel = np.array([1., 1., 0., .2**2/6, 0.])\n",
            "call": "compute_homogeneous_face(c.copy(), ell, length, alpha, kernel.copy())",
            "gold_call": "_oracle_compute_homogeneous_face(c.copy(), ell, length, alpha, kernel.copy())",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nc = np.zeros((9,3,3)); c[:,0,2] = np.arange(1.,10.)\nell, length, alpha = .01, .8, .001\nkernel = _oracle_compute_normal_kernel(ell, alpha, 45.)\n",
            "call": "compute_homogeneous_face(c.copy(), ell, length, alpha, kernel.copy())",
            "gold_call": "_oracle_compute_homogeneous_face(c.copy(), ell, length, alpha, kernel.copy())",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nc = np.ones((9,3,3)); ell, length, alpha = .1, 0., .03\nkernel = np.ones(5)\ndef _exception_code(function):\n    try:\n        function(c.copy(), ell, length, alpha, kernel.copy())\n    except ValueError:\n        return 1\n    return 0\n\n",
            "call": "_exception_code(compute_homogeneous_face)",
            "gold_call": "_exception_code(_oracle_compute_homogeneous_face)",
        },
    ]
