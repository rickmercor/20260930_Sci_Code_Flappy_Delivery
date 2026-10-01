"""
Construct the coupled TM operators from the reciprocal-permittivity convolution matrix and its supplied directional derivative.



Form P = C inverse, Q = K C K - I, and the ordered product M = P Q, where K is the diagonal matrix of the supplied transverse wavevectors. Compute the directional derivatives of all three matrices while holding K fixed.



Preserve matrix multiplication order: P and Q need not commute. Evaluate inverse actions through linear solves. This step receives numerical arrays directly and does not call the first step internally.

In the finite-dimensional TM formulation, reciprocal-permittivity coupling mixes the retained Fourier harmonics. The inverse coupling and the transverse-wavevector coupling define noncommuting operators whose multiplication order determines the modal generator. When the fill fraction changes, the Fourier coupling varies while the harmonic wavevector matrix remains fixed. Propagate that tangent through the required linear solves and ordered matrix products without assuming that any factors commute.

Returns
-------
Return (P,dP,Q,dQ,product,product_tangent), six complex128 arrays of shape (N,N), preserving harmonic ordering. P and Q are the TM operators defined in the main problem, product is their prescribed ordered product, and the three tangent outputs are their directional derivatives induced by tangent with kappa fixed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tm_operators(
    coefficients: np.ndarray,
    tangent: np.ndarray,
    kappa: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Form the ordered TM operators and their tangents.

    Parameters
    ----------
    coefficients : np.ndarray
        Nonsingular convolution matrix C of shape (N, N).
    tangent : np.ndarray
        Directional derivative of C with shape (N, N).
    kappa : np.ndarray
        Fixed real wavevector of shape (N,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray]
        P, dP, Q, dQ, P Q, and d(P Q), each complex128 (N, N).

    Raises
    ------
    np.linalg.LinAlgError
        If the coefficient matrix is singular.
    """
    return None, None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve


def _constant(value):
    array = np.asarray(value, dtype=np.complex128)
    return array, np.zeros_like(array)


def _subtract(left, right):
    return left[0] - right[0], left[1] - right[1]


def _multiply(left, right):
    return (
        left[0] @ right[0],
        left[1] @ right[0] + left[0] @ right[1],
    )


def _solve(left, right):
    value = solve(left[0], right[0])
    tangent = solve(
        left[0],
        right[1] - left[1] @ value,
    )
    return value, tangent


def _oracle_tm_operators(
    coefficients: np.ndarray,
    tangent: np.ndarray,
    kappa: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return P, dP, Q, dQ, PQ, d(PQ), each of shape (N, N).

    C and dC are finite arrays with consistent shapes.
    C is nonsingular, and kappa is a real vector of shape (N,).
    kappa is held fixed under differentiation.
    """
    identity = _constant(np.eye(len(kappa)))
    wavevector = _constant(np.diag(kappa))

    inverse = _solve(
        (coefficients, tangent),
        identity,
    )

    coupling = _subtract(
        _multiply(
            _multiply(
                wavevector,
                (coefficients, tangent),
            ),
            wavevector,
        ),
        identity,
    )

    product = _multiply(inverse, coupling)

    return (
        inverse[0],
        inverse[1],
        coupling[0],
        coupling[1],
        product[0],
        product[1],
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = (
        "import numpy as np\n"
        "base = np.array([[0.7, 0.1j], [-0.1j, 0.9]], dtype=np.complex128)\n"
        "direction = np.array([[0.2, 0.04], [0.04, -0.1]], dtype=np.complex128)"
    )
    return [
        {
            "setup": setup,
            "call": "tm_operators(base, direction, np.array([0.2, 1.4]))",
            "gold_call": "_oracle_tm_operators(base, direction, np.array([0.2, 1.4]))",
        },
        {
            "setup": setup,
            "call": "tm_operators(base, np.zeros_like(base), np.array([0.2, 1.4]))",
            "gold_call": "_oracle_tm_operators(base, np.zeros_like(base), np.array([0.2, 1.4]))",
        },
        {
            "setup": setup,
            "call": "tm_operators(base, np.eye(2), np.array([0.2, 1.4]))",
            "gold_call": "_oracle_tm_operators(base, np.eye(2), np.array([0.2, 1.4]))",
        },
    ]
