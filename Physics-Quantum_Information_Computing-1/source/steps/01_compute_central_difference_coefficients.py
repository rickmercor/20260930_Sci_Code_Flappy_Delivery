"""
Construct the symmetric central finite-difference coefficient rows for derivative orders $q=1,\ldots,q_{\max}$, where $J=\texttt{degree}$ and $q_{\max}=\texttt{max\_order}\le 2J$, on the integer stencil $j=-J,\ldots,J$. Each row must differentiate every polynomial of degree at most $2J$ exactly at the origin and obey centered-stencil parity: odd derivative rows are antisymmetric and even derivative rows are symmetric.

The time-shift Krylov reconstruction obtains powers of a Hamiltonian from derivatives of a unitary propagator at zero time. On a symmetric stencil, the required weights are the derivatives at the origin of the Lagrange cardinal polynomials, equivalently the unique solution of the polynomial-exactness moment equations. Odd derivative rows are antisymmetric and even derivative rows are symmetric; enforcing that parity removes roundoff-scale violations without changing the finite-difference order.

Returns
-------
np.ndarray of shape (max_order, 2 * degree + 1), with row q - 1 containing the float64 weights for derivative order q on nodes -degree,...,degree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral

import numpy as np

STEP_NAME = "compute_central_difference_coefficients"
STEP_ID = "01"
STEP_DESCRIPTION = r"""Construct the symmetric central finite-difference coefficient rows for derivative orders 1 through `max_order` on the integer stencil from `-degree` through `degree`. The row for derivative order `q` must differentiate every polynomial of degree at most `2 * degree` exactly at the origin and must obey the parity of a centered derivative stencil."""
STEP_SCIENTIFIC_BACKGROUND = r"""The time-shift Krylov reconstruction obtains powers of a Hamiltonian from derivatives of a unitary propagator at zero time. On a symmetric stencil, the required weights are the derivatives at the origin of the Lagrange cardinal polynomials, equivalently the unique solution of the polynomial-exactness moment equations. Odd derivative rows are antisymmetric and even derivative rows are symmetric; enforcing that parity removes roundoff-scale violations without changing the finite-difference order."""
EXPECTED_RETURN_LINE = "np.ndarray of shape (max_order, 2 * degree + 1), with row q - 1 containing the float64 weights for derivative order q on nodes -degree,...,degree"
IS_FINAL_ORCHESTRATOR = False


def compute_central_difference_coefficients(
    degree: int,
    max_order: int,
) -> np.ndarray:
    """Construct centered finite-difference coefficient rows.

    Parameters
    ----------
    degree : int
        Positive half-width of the symmetric stencil.
    max_order : int
        Largest derivative order, from 1 through ``2 * degree``.

    Returns
    -------
    coefficients : np.ndarray
        Float64 array of shape ``(max_order, 2 * degree + 1)``. Row
        ``q - 1`` contains the weights for derivative order ``q`` on nodes
        ``-degree, ..., degree``.

    Raises
    ------
    ValueError
        If ``degree`` or ``max_order`` is not an integer in the stated range.
    """
    return np.empty((max_order, 2 * degree + 1), dtype=np.float64)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_central_difference_coefficients(
    degree: int,
    max_order: int,
) -> np.ndarray:
    import math
    from fractions import Fraction
    from numbers import Integral

    import numpy as np

    if isinstance(degree, bool) or not isinstance(degree, Integral) or int(degree) < 1:
        raise ValueError("degree must be an integer >= 1")
    if (
        isinstance(max_order, bool)
        or not isinstance(max_order, Integral)
        or int(max_order) < 1
        or int(max_order) > 2 * int(degree)
    ):
        raise ValueError("max_order must be an integer in [1, 2 * degree]")

    degree = int(degree)
    max_order = int(max_order)
    nodes = list(range(-degree, degree + 1))
    rows = np.zeros((max_order, 2 * degree + 1), dtype=np.float64)

    # Build each Lagrange cardinal polynomial exactly in ascending powers.
    for column, node in enumerate(nodes):
        polynomial = [Fraction(1, 1)]
        for other in nodes:
            if other == node:
                continue
            denominator = node - other
            updated = [Fraction(0, 1)] * (len(polynomial) + 1)
            for power, coefficient in enumerate(polynomial):
                updated[power] += -Fraction(other, denominator) * coefficient
                updated[power + 1] += Fraction(1, denominator) * coefficient
            polynomial = updated

        for derivative_order in range(1, max_order + 1):
            rows[derivative_order - 1, column] = float(
                math.factorial(derivative_order) * polynomial[derivative_order]
            )

    return rows

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "degree = 3\nmax_order = 6",
            "call": "compute_central_difference_coefficients(degree, max_order)",
            "gold_call": "_oracle_compute_central_difference_coefficients(degree, max_order)",
        },
        {
            "setup": "degree = 1\nmax_order = 2",
            "call": "compute_central_difference_coefficients(degree, max_order)",
            "gold_call": "_oracle_compute_central_difference_coefficients(degree, max_order)",
        },
        {
            "setup": "degree = 4\nmax_order = 3",
            "call": "compute_central_difference_coefficients(degree, max_order)",
            "gold_call": "_oracle_compute_central_difference_coefficients(degree, max_order)",
        },
        {
            "setup": """def run_model():
    try:
        compute_central_difference_coefficients(0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_central_difference_coefficients(0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """def run_model():
    try:
        compute_central_difference_coefficients(2, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_compute_central_difference_coefficients(2, 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
