"""
Construct the balanced residual polynomial from the Balance Method 2 root set and return its coefficients in ascending order of polynomial degree.

First obtain the balanced root set $\\{r_1, \\ldots, r_m\\}$ using Balance Method 2.



The associated residual polynomial is normalized by $\\pi(0) = 1$ and is defined from its roots by



$$

\\pi(\\alpha) = \\prod_{j=1}^{m} \\left(1 - \\frac{\\alpha}{r_j}\\right).

$$



Represent the polynomial in ascending powers,



$$

\\pi(\\alpha) = c_0 + c_1 \\alpha + c_2 \\alpha^2 + \\cdots + c_m \\alpha^m,

$$



and return $[c_0, c_1, \\ldots, c_m]$.



The balanced roots may contain complex-conjugate pairs. For a valid conjugate-symmetric root set, the coefficients of the complete product are real up to numerical roundoff.



Use apply_balance_method2 to construct the balanced root set before forming the polynomial.



Do not reorder or discard roots solely because they are complex; every retained root contributes one factor to the product, and a repeated root contributes one factor per occurrence.



The returned coefficient array must be real-valued. If the imaginary part remaining after the complete conjugate-symmetric product is larger than numerical tolerance, the input is invalid and ValueError must be raised. Empty, non-one-dimensional, or non-finite input must also raise ValueError.

Returns
-------
A real one-dimensional NumPy array [c_0, ..., c_m] containing the balanced residual-polynomial coefficients in ascending powers.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_balanced_residual_polynomial(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Construct the balanced residual polynomial.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.

    Returns
    -------
    np.ndarray
        Real one-dimensional coefficient array [c_0, ..., c_m]
        representing pi(alpha) in ascending powers.

    Raises
    ------
    ValueError
        If the array is empty or not one-dimensional, contains non-finite
        entries, or the product coefficients are not real to tolerance.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_balanced_residual_polynomial(
    harmonic_ritz_roots,
):
    roots = np.asarray(
        harmonic_ritz_roots,
        dtype=complex,
    )

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real)
        & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    balanced_roots = _oracle_apply_balance_method2(
        roots
    )

    tol = 1.0e-12

    if np.any(np.abs(balanced_roots) <= tol):
        raise ValueError(
            "Balanced polynomial roots must be nonzero."
        )

    coefficients = np.array(
        [1.0 + 0.0j],
        dtype=complex,
    )

    for root in balanced_roots:
        factor = np.array(
            [
                1.0 + 0.0j,
                -1.0 / root,
            ],
            dtype=complex,
        )

        coefficients = np.convolve(
            coefficients,
            factor,
        )

    scale = max(
        1.0,
        float(np.max(np.abs(coefficients))),
    )

    if np.max(np.abs(coefficients.imag)) > tol * scale:
        raise ValueError(
            "The balanced polynomial coefficients must be real for a conjugate-symmetric root set."
        )

    return coefficients.real.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    -5.0,
    -1.3,
    0.9,
    2.0 + 1.5j,
    2.0 - 1.5j
], dtype=complex)
""",
            "call": """build_balanced_residual_polynomial(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_build_balanced_residual_polynomial(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    -6.0,
    -2.0,
    0.8,
    3.5
], dtype=complex)
""",
            "call": """build_balanced_residual_polynomial(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_build_balanced_residual_polynomial(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    1.0,
    -1.1111111111111112,
    -20.0 + 20.0j,
    -20.0 - 20.0j
], dtype=complex)
""",
            "call": """build_balanced_residual_polynomial(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_build_balanced_residual_polynomial(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    2.0,
    2.0,
    -0.8
], dtype=complex)
""",
            "call": """build_balanced_residual_polynomial(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_build_balanced_residual_polynomial(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np


def _run_invalid(fn, *args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

import numpy as np

harmonic_ritz_roots = np.array([
    -5.0,
    np.nan,
    0.9
], dtype=complex)
""",
            "call": """_run_invalid(
    build_balanced_residual_polynomial, harmonic_ritz_roots
)""",
            "gold_call": """_run_invalid(
    _oracle_build_balanced_residual_polynomial, harmonic_ritz_roots
)""",
        },
    ]
