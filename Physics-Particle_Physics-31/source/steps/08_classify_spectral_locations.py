"""
Classify recovered locations as poles or zeros with the odd-r Vandermonde sign rule.

Given C_r^(d) and the ordered generalized spectral data, form the Vandermonde
matrix V_{in}=lambda_n^i and reconstruct d_r = V^{-1} C_r V^{-T}. For odd r,
the sign of [d_r]_{nn} is sigma_n: +1 for a pole and -1 for a zero. Return both
the classified locations and the ordered diagonal weights used for the
classification so that the source-specific classification certificate remains
available to the final reconstruction.

Parameters
----------
hankel_current : np.ndarray
    Finite real square matrix C_r^(d).
r : int
    Positive odd source shift.
spectral_data : np.ndarray
    Real array with shape (d, 2), with columns [lambda_n, location_n] in a
    common ordering.

Returns
-------
classification : tuple[np.ndarray, np.ndarray]
    First entry: real array with shape (d, 2), columns [location_n, sigma_n],
    where sigma_n is +1 for a pole and -1 for a zero. Second entry: real array
    of length d containing the corresponding ordered diagonal weights
    diag(d_r).

Raises
------
ValueError
    If r is not positive and odd, inputs are malformed or nonfinite, the
    Vandermonde matrix is singular, or a reconstructed diagonal weight is
    numerically zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def classify_spectral_locations(
    hankel_current: "np.ndarray", r: int, spectral_data: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """Return classified locations and ordered diagonal classification weights."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_classify_spectral_locations(
    hankel_current: "np.ndarray", r: int, spectral_data: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """Return classified locations and ordered diagonal classification weights."""
    try:
        current = np.asarray(hankel_current, dtype=float)
        spectrum = np.asarray(spectral_data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("hankel_current and spectral_data must be real numeric arrays") from exc
    if not isinstance(r, (int, np.integer)) or int(r) <= 0 or int(r) % 2 != 1:
        raise ValueError("r must be a positive odd integer")
    if (
        current.ndim != 2
        or current.shape[0] != current.shape[1]
        or current.shape[0] == 0
        or spectrum.shape != (current.shape[0], 2)
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(spectrum))
    ):
        raise ValueError("inputs have incompatible shapes or contain nonfinite values")

    lambdas = spectrum[:, 0]
    locations = spectrum[:, 1]
    d = current.shape[0]
    vandermonde = np.vstack([lambdas ** i for i in range(d)])
    try:
        left = np.linalg.solve(vandermonde, current)
        d_r = np.linalg.solve(vandermonde, left.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("the Vandermonde matrix is singular") from exc
    d_r = 0.5 * (d_r + d_r.T)
    diagonal = np.diag(d_r).copy()
    scale = max(1.0, float(np.max(np.abs(diagonal))))
    if np.any(np.abs(diagonal) <= 1.0e-12 * scale):
        raise ValueError("a reconstructed diagonal weight is numerically zero")

    signs = np.where(diagonal > 0.0, 1.0, -1.0)
    classified = np.column_stack((locations, signs))
    return classified, diagonal

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    pack = "lambda out: (np.round(out[0],10),np.round(out[1],10))"
    return [
        {
            "setup": "import numpy as np\nc=np.array([0.060771,-0.060655960547,-0.175767949961623479,-0.254210533421592269403791,-0.301159493447749769008969911999,-0.326106315744614322485936659263474047,-0.336471851249179010399929864467206626537119,-0.337243559154630177357962321708726123322431250431]); C=np.array([[c[1+i+j] for j in range(4)] for i in range(4)]); spec=np.array([[0.919798,1.0871952319965906],[0.802520,1.2460748641778398],[0.485921,2.0579476910855880],[0.307872,3.2481031077850535]]); pack=" + pack,
            "call": "pack(classify_spectral_locations(C,1,spec))",
            "gold_call": "pack(_oracle_classify_spectral_locations(C,1,spec))",
        },
        {
            "setup": "import numpy as np\nC=np.array([[0.25]]); spec=np.array([[0.5,2.0]]); pack=" + pack,
            "call": "pack(classify_spectral_locations(C,1,spec))",
            "gold_call": "pack(_oracle_classify_spectral_locations(C,1,spec))",
        },
        {
            "setup": "import numpy as np\nc=np.array([0.1,0.12,0.10825,0.087]); C=np.array([[c[1+i+j] for j in range(2)] for i in range(2)]); spec=np.array([[0.65,1.0/0.65],[0.55,1.0/0.55]]); pack=" + pack,
            "call": "pack(classify_spectral_locations(C,1,spec))",
            "gold_call": "pack(_oracle_classify_spectral_locations(C,1,spec))",
        },
    ]
