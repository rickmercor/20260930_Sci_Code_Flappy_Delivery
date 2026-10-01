"""
Reconstruct the normalized rational amplitude and verify its EFT re-expansion.

Each classified zero at rho contributes a numerator factor (1-s/rho), each pole at mu
contributes a denominator factor (1-s/mu), the denominator is normalized by D(0)=1,
and the numerator carries the overall normalization b_0. Re-expand N(s)/D(s) through
the supplied EFT order and return the maximum absolute coefficient residual.

Parameters
----------
b : np.ndarray
    One-dimensional finite real EFT coefficient array with nonzero b[0].
classified_locations : np.ndarray
    Finite real array with shape (d, 2), columns [location, sigma], where
    sigma=+1 denotes a pole and sigma=-1 denotes a zero. Locations must be
    nonzero and signs must be exactly +/-1.

Returns
-------
reconstruction : tuple[np.ndarray, np.ndarray, float]
    Numerator coefficients in ascending powers, denominator coefficients in
    ascending powers, and max absolute EFT re-expansion residual.

Raises
------
ValueError
    If either input is malformed, nonfinite, has a zero location, or contains
    a sign other than +/-1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reconstruct_rational_amplitude(
    b: "np.ndarray", classified_locations: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray, float]":
    """Return numerator, denominator, and EFT re-expansion residual."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reconstruct_rational_amplitude(
    b: "np.ndarray", classified_locations: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray, float]":
    """Return numerator, denominator, and EFT re-expansion residual."""
    try:
        coeff = np.asarray(b, dtype=float)
        classified = np.asarray(classified_locations, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("inputs must be real numeric arrays") from exc
    if (
        coeff.ndim != 1
        or coeff.size < 1
        or not np.all(np.isfinite(coeff))
        or coeff[0] == 0.0
    ):
        raise ValueError("b must be a finite one-dimensional array with nonzero b[0]")
    if (
        classified.ndim != 2
        or classified.shape[1] != 2
        or classified.shape[0] < 1
        or not np.all(np.isfinite(classified))
        or np.any(classified[:, 0] == 0.0)
        or np.any(~np.isin(classified[:, 1], (-1.0, 1.0)))
    ):
        raise ValueError("classified_locations must contain nonzero locations and +/-1 signs")

    numerator = np.array([coeff[0]], dtype=float)
    denominator = np.array([1.0], dtype=float)
    for location, sign in classified:
        factor = np.array([1.0, -1.0 / location], dtype=float)
        if sign < 0.0:
            numerator = np.polynomial.polynomial.polymul(numerator, factor)
        else:
            denominator = np.polynomial.polynomial.polymul(denominator, factor)

    reconstructed = np.zeros_like(coeff)
    for n in range(coeff.size):
        rhs = numerator[n] if n < numerator.size else 0.0
        for k in range(1, min(n, denominator.size - 1) + 1):
            rhs -= denominator[k] * reconstructed[n - k]
        reconstructed[n] = rhs / denominator[0]

    residual = float(np.max(np.abs(reconstructed - coeff)))
    return numerator, denominator, residual

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    pack = "lambda out: (np.round(out[0],10),np.round(out[1],10),round(float(out[2]),12))"
    return [
        {
            "setup": "import numpy as np\nb=np.array([1.255458,0.076295438118,-0.035757230423273274,-0.075823351552458039254394,-0.083749978089300131915797567674,-0.078338731191377881605205437737333754,-0.068275581072225571993077829698251907397434,-0.057419982857692864178695909526227899110314926714,-0.047357424351643017636639948800385481733980392431793594,-0.038625674826665615587987604470468827548796529186628134990074,-0.031299339343808186683923366066895834773625422979666139953711748154]); cl=np.array([[1.0871952319965906,-1.0],[1.2460748641778398,1.0],[2.0579476910855880,1.0],[3.2481031077850535,-1.0]]); pack=" + pack,
            "call": "pack(reconstruct_rational_amplitude(b,cl))",
            "gold_call": "pack(_oracle_reconstruct_rational_amplitude(b,cl))",
        },
        {
            "setup": "import numpy as np\nb=np.array([2.0,1.0,0.5,0.25,0.125,0.0625,0.03125]); cl=np.array([[2.0,1.0]]); pack=" + pack,
            "call": "pack(reconstruct_rational_amplitude(b,cl))",
            "gold_call": "pack(_oracle_reconstruct_rational_amplitude(b,cl))",
        },
        {
            "setup": "import numpy as np\nb=np.array([1.25,-0.5,0.0,0.0,0.0,0.0,0.0]); cl=np.array([[2.5,-1.0]]); pack=" + pack,
            "call": "pack(reconstruct_rational_amplitude(b,cl))",
            "gold_call": "pack(_oracle_reconstruct_rational_amplitude(b,cl))",
        },
    ]
