"""
Recover generalized spectral parameters and physical pole/zero locations.

Solve the ordered generalized eigenproblem C_{r+1} v = lambda C_r v. The source
identifies each physical location as s = 1/lambda. Only finite eigenvalues whose
imaginary part has magnitude at most imag_tol are accepted. Return the values
ordered by increasing physical location, preserving lambda-location pairing.

Parameters
----------
hankel_current : np.ndarray
    Finite real square matrix C_r^(d).
hankel_shifted : np.ndarray
    Finite real square matrix C_{r+1}^(d) with the same shape.
imag_tol : float
    Nonnegative finite absolute tolerance for the imaginary part of each
    generalized eigenvalue.

Returns
-------
spectral_data : np.ndarray
    Real array with shape (d, 2). Column 0 contains lambda_n and column 1
    contains s_n=1/lambda_n, ordered by increasing s_n.

Raises
------
ValueError
    If the matrices are invalid, imag_tol is invalid, the generalized
    eigenproblem contains a nonfinite or effectively zero eigenvalue, or a
    recovered eigenvalue is not real within imag_tol.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def recover_spectral_locations(
    hankel_current: "np.ndarray",
    hankel_shifted: "np.ndarray",
    imag_tol: float,
) -> "np.ndarray":
    """Solve the shifted Hankel pencil and return [lambda, 1/lambda] pairs."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigvals


def _oracle_recover_spectral_locations(
    hankel_current: "np.ndarray",
    hankel_shifted: "np.ndarray",
    imag_tol: float,
) -> "np.ndarray":
    """Solve the shifted Hankel pencil and return [lambda, 1/lambda] pairs."""
    try:
        current = np.asarray(hankel_current, dtype=float)
        shifted = np.asarray(hankel_shifted, dtype=float)
        tol = float(imag_tol)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("matrices and imag_tol must be real numeric values") from exc
    if (
        current.ndim != 2
        or current.shape[0] != current.shape[1]
        or current.shape[0] == 0
        or shifted.shape != current.shape
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(shifted))
    ):
        raise ValueError("the Hankel matrices must be finite nonempty square arrays of equal shape")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("imag_tol must be nonnegative and finite")

    values = eigvals(shifted, current)
    if np.any(~np.isfinite(values)):
        raise ValueError("the generalized eigenproblem returned a nonfinite value")
    if np.any(np.abs(values.imag) > tol):
        raise ValueError("a generalized eigenvalue is not real within imag_tol")

    lambdas = values.real
    scale = max(1.0, float(np.max(np.abs(lambdas))))
    if np.any(np.abs(lambdas) <= np.finfo(float).eps * scale):
        raise ValueError("a generalized eigenvalue is numerically zero")
    locations = 1.0 / lambdas
    order = np.argsort(locations)
    return np.column_stack((lambdas[order], locations[order]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    return [
        {
            "setup": "import numpy as np\nc=np.array([0.060771,-0.060655960547,-0.175767949961623479,-0.254210533421592269403791,-0.301159493447749769008969911999,-0.326106315744614322485936659263474047,-0.336471851249179010399929864467206626537119,-0.337243559154630177357962321708726123322431250431,-0.331672505490807096899101287418561408111207028195519199]); C=np.array([[c[1+i+j] for j in range(4)] for i in range(4)]); Cs=np.array([[c[2+i+j] for j in range(4)] for i in range(4)])",
            "call": "np.round(recover_spectral_locations(C,Cs,1e-10),10)",
            "gold_call": "np.round(_oracle_recover_spectral_locations(C,Cs,1e-10),10)",
        },
        {
            "setup": "import numpy as np\nC=np.array([[0.25]]); Cs=np.array([[0.125]])",
            "call": "np.round(recover_spectral_locations(C,Cs,1e-10),10)",
            "gold_call": "np.round(_oracle_recover_spectral_locations(C,Cs,1e-10),10)",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.1,0.12,0.10825,0.087,0.065700625]); C=np.array([[c[1+i+j] for j in range(2)] for i in range(2)]); Cs=np.array([[c[2+i+j] for j in range(2)] for i in range(2)])",
            "call": "np.round(recover_spectral_locations(C,Cs,1e-10),10)",
            "gold_call": "np.round(_oracle_recover_spectral_locations(C,Cs,1e-10),10)",
        },
    ]
