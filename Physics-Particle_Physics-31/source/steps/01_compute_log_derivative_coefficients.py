"""
Compute the logarithmic-derivative EFT coefficients used by the finite-spectrum inverse-EFT construction.

For an amplitude A(s) = sum_{j>=0} b_j s^j with b_0 != 0, define
Q(s) = A'(s)/A(s) = sum_{k>=0} c_k s^k. Matching A'(s) = Q(s) A(s)
gives the recursion
(n+1)b_{n+1} = sum_{k=0}^n c_k b_{n-k}.
Return c_0,...,c_{m-2} when b contains m coefficients.

Parameters
----------
b : np.ndarray
    One-dimensional finite real array of at least two EFT coefficients with
    nonzero first entry.

Returns
-------
c : np.ndarray
    One-dimensional float array of length len(b)-1 containing the transformed
    coefficients.

Raises
------
ValueError
    If b is not a finite real one-dimensional array of length at least two, or
    if b[0] is zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_log_derivative_coefficients(b: "np.ndarray") -> "np.ndarray":
    """Compute c_k in Q(s)=A'(s)/A(s) from the EFT coefficients b_j."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from decimal import Decimal, localcontext
import numpy as np


def _oracle_compute_log_derivative_coefficients(b: "np.ndarray") -> "np.ndarray":
    """Compute c_k in Q(s)=A'(s)/A(s) from the EFT coefficients b_j."""
    raw = np.asarray(b)
    if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0.0):
        raise ValueError("b must be real-valued")
    try:
        coeff = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("b must contain real numeric values") from exc
    if coeff.ndim != 1 or coeff.size < 2 or not np.all(np.isfinite(coeff)):
        raise ValueError("b must be a finite one-dimensional array of length at least two")
    if coeff[0] == 0.0:
        raise ValueError("b[0] must be nonzero")

    # Carry the logarithmic-derivative recurrence at high precision so that
    # small roundoff errors do not move the final benchmark's tenth decimal.
    with localcontext() as ctx:
        ctx.prec = 60
        b_dec = [Decimal(str(float(value))) for value in coeff]
        c_dec = []

        for n in range(coeff.size - 1):
            previous = sum(
                (c_dec[k] * b_dec[n - k] for k in range(n)),
                Decimal(0),
            )
            current = (
                Decimal(n + 1) * b_dec[n + 1] - previous
            ) / b_dec[0]
            c_dec.append(current)

    return np.asarray([float(value) for value in c_dec], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    return [
        {
            "setup": "import numpy as np\nb=np.array([1.255458,0.076295438118,-0.035757230423273274,-0.075823351552458039254394,-0.083749978089300131915797567674,-0.078338731191377881605205437737333754,-0.068275581072225571993077829698251907397434,-0.057419982857692864178695909526227899110314926714,-0.047357424351643017636639948800385481733980392431793594,-0.038625674826665615587987604470468827548796529186628134990074,-0.031299339343808186683923366066895834773625422979666139953711748154],dtype=float)",
            "call": "np.round(compute_log_derivative_coefficients(b),12)",
            "gold_call": "np.round(_oracle_compute_log_derivative_coefficients(b),12)",
        },
        {
            "setup": "import numpy as np\nb=np.array([2.0,1.0,0.5,0.25,0.125,0.0625,0.03125],dtype=float)",
            "call": "np.round(compute_log_derivative_coefficients(b),12)",
            "gold_call": "np.round(_oracle_compute_log_derivative_coefficients(b),12)",
        },
        {
            "setup": "import numpy as np\nb=np.array([1.25,-0.5,0.0,0.0,0.0,0.0,0.0],dtype=float)",
            "call": "np.round(compute_log_derivative_coefficients(b),12)",
            "gold_call": "np.round(_oracle_compute_log_derivative_coefficients(b),12)",
        },
    ]
