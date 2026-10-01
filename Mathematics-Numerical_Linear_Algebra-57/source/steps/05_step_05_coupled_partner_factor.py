"""
*The iteration carries a pair, and every step so far has reported only the residual-bearing member `$M$`. The partner `$X$` is the object the method is actually computing: it converges to the inverse `$p$$-th root of the input, while$$M$` only certifies how far it has to go. This step returns the partner after a whole run of steps whose coefficients are already known.*



The two members start from the same positive scaling constant, the one that normalises the starting `$M$`. `$M$` starts from the input divided by the `$p$`-th power of that constant, as an earlier step returns; `$X$` starts from the identity divided by the constant itself, to the first power. That asymmetry between the two starting divisions is exactly what makes the tie `$M = X ** p A$` hold at the start.



Over one accelerated step `$M$` is multiplied by the `$p$$-th power of$$I + alpha R$`, where `$R = I - M$` is the residual carried into that step, while `$X$` is multiplied by the same factor only once. Because every iterate is a polynomial in the input, the factor commutes with `$X$`, and the tie is preserved step by step: raising the once-multiplied `$X$` to the `$p$$-th power reproduces the$$p$$-times-multiplied$$M$`. The residual driving each step must therefore be recomputed from the current `$M$`, not from the current `$X$`, and the two members have to advance in lockstep within a single loop.



The coefficients are supplied in the order they were fitted, one per step, and the number of steps is their count.



The two scaling operations have different homogeneity. Rescaling `$A$` leaves the carried `$M$` unchanged but rescales the starting partner by the inverse `$p$`-th root of that factor. Both must remain representable whenever the true float64 result is representable, even if a naive Frobenius norm of `$A$` is zero or infinite. As in the starting-matrix step, an `$A$` admitted by the symmetry tolerance represents its symmetric projection; the same projected matrix must be used for both members or the coupled tie is already broken at the start.

Returns
-------
np.ndarray of shape (n, n), the partner member of the coupled pair after the run
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coupled_partner_factor(A: np.ndarray, alphas: np.ndarray, p: int) -> np.ndarray:
    """Return the partner member of the coupled pair after the supplied steps.

    Raises ``ValueError`` unless every one of the following holds: ``A`` and
    ``alphas`` are real rather than complex; ``A`` is a nonempty square
    two-dimensional array; every entry of ``A`` is finite; ``A`` is symmetric to
    an absolute tolerance of ``1e-12`` and is then projected to its symmetric
    part; that symmetric part is nonzero, including when a naive Frobenius norm
    underflows or overflows;
    ``alphas`` is a nonempty one-dimensional array of finite values, so a run of
    no steps is rejected rather than returning the starting partner; and ``p`` is
    an integer, not a bool, with ``p >= 1``.

    Parameters
    ----------
    A : np.ndarray
        Nonzero finite real symmetric matrix of shape ``(n, n)``.
    alphas : np.ndarray
        Finite real fitted coefficients, one per step, in the order applied.
    p : int
        Root order. The iteration targets the inverse ``p``-th root of ``A``.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n, n)`` holding the partner after every
        supplied step has been applied.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_coupled_partner_factor(
    A: np.ndarray,
    alphas: np.ndarray,
    p: int,
) -> np.ndarray:
    """Reference partner iterate, advanced in lockstep with the earlier oracles."""
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
        raise ValueError("p must be an integer")
    if int(p) < 1:
        raise ValueError("p must be at least one")
    order = int(p)
    coefficients = np.asarray(alphas)
    if np.iscomplexobj(coefficients):
        raise ValueError("alphas must be real")
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("alphas must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("alphas must contain only finite values")
    carried = _oracle_scaled_starting_matrix(A, order)  # noqa: F821
    matrix = np.asarray(A, dtype=float)
    projected = matrix.copy()
    for row in range(matrix.shape[0]):
        for column in range(row + 1, matrix.shape[1]):
            left = float(matrix[row, column])
            right = float(matrix[column, row])
            if np.signbit(left) == np.signbit(right):
                midpoint = left + 0.5 * (right - left)
            else:
                midpoint = 0.5 * (left + right)
            projected[row, column] = midpoint
            projected[column, row] = midpoint
    matrix = projected
    magnitude = float(np.max(np.abs(matrix)))
    scaled = matrix / magnitude
    scaled_norm = float(np.linalg.norm(scaled, ord="fro"))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        frobenius = float(np.linalg.norm(matrix, ord="fro"))
    if (
        np.isfinite(frobenius)
        and frobenius > 0.0
        and frobenius <= np.finfo(float).max / 2.0
    ):
        scale = float((2.0 * frobenius / (order + 1.0)) ** (1.0 / order))
        inverse_scale = 1.0 / scale
    else:
        log_scale = (
            np.log(2.0 / (order + 1.0)) + np.log(magnitude) + np.log(scaled_norm)
        ) / order
        inverse_scale = float(np.exp(-log_scale))
    identity = np.eye(carried.shape[0], dtype=float)
    partner = identity * inverse_scale
    for alpha in coefficients:
        partner = partner @ (identity + float(alpha) * (identity - carried))
        carried = _oracle_advance_inverse_newton(carried, float(alpha), order)  # noqa: F821
    return partner

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return ordinary, extreme-scale, projected-symmetry and invalid cases."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
alphas = np.array([0.45, 0.6125, 0.375])
p = 3
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[4.0]])
alphas = np.array([0.5, 0.5])
p = 2
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
alphas = np.array([0.2, 0.35, 0.275, 0.1875])
p = 5
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.75, -0.5], [-0.5, 2.25]])
alphas = np.array([0.4])
p = 1
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.75, -0.5], [-0.5, 2.25]])
alphas = np.array([0.0, 0.0, 0.0])
p = 3
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
v = np.array([[1.0], [-2.0], [0.5]])
A = v @ v.T + 2.0 * np.eye(3)
alphas = np.array([0.25, 0.5, 0.75, 1.0, 0.125])
p = 4
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = 1e300 * np.array([[2.0, 0.5, -0.25], [0.5, 1.5, 0.125], [-0.25, 0.125, 3.0]])
p = 401
alphas = np.array([1.0 / p])
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.6e308, 3e307], [3e307, 1.2e308]])
p = 401
alphas = np.array([1.0 / p])
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = 1e-300 * np.array([[2.0, -0.25], [-0.25, 1.0]])
alphas = np.array([0.25, 0.4, 0.3])
p = 7
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[np.nextafter(0.0, 1.0)]])
alphas = np.array([1.0 / 401.0])
p = 401
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2e-12, 9e-13], [1e-13, 1e-12]])
alphas = np.array([0.3, 0.55])
p = 3
""",
            "call": "coupled_partner_factor(A, alphas, p)",
            "gold_call": "_oracle_coupled_partner_factor(A, alphas, p)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5], [0.5, 1.5]])
alphas = np.array([])
p = 3
def run_model_empty():
    try:
        coupled_partner_factor(A, alphas, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_empty():
    try:
        _oracle_coupled_partner_factor(A, alphas, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_empty()",
            "gold_call": "run_oracle_empty()",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 2.0], [0.0, 1.0]])
alphas = np.array([0.5])
p = 3
def run_model_asym():
    try:
        coupled_partner_factor(A, alphas, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_asym():
    try:
        _oracle_coupled_partner_factor(A, alphas, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_asym()",
            "gold_call": "run_oracle_asym()",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0, 0.5], [0.5, 1.5]])
alphas = np.array([0.5 + 0.0j, 0.25 + 1e-18j])
p = 3
def run_model_complex():
    try:
        coupled_partner_factor(A, alphas, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_complex():
    try:
        _oracle_coupled_partner_factor(A, alphas, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_complex()",
            "gold_call": "run_oracle_complex()",
        },
    ]
