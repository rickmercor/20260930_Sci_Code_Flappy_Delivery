"""
Bound and evaluate the first right-preconditioned residual reduction.

One zero-start step of a right-preconditioned minimal-residual method searches

a single direction, so its optimal reduction can be written down in closed form

rather than iterated for. That exact reduction is controlled from above by a

field-of-values estimate built from two scalars of the preconditioned operator:

how coercive its symmetric part is, and how much the operator can inflate a

vector. The second is a norm, not an eigenvalue; for a nonnormal operator the

largest eigenvalue modulus understates the inflation, and an estimate built

from it is not a bound at all. The estimate must also stay real and stay

meaningful when the symmetric part loses definiteness, in which case it

degenerates to the statement that no reduction is guaranteed. Report the exact

reduction, the coefficient attaining it, the estimate, and the two scalars it

was built from, and reject data on which the estimate fails.

Returns
-------
tuple of five finite floats: exact reduction in [0, 1], optimal coefficient, field-of-values estimate in [0, 1], coercivity, and positive inflation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def certify_residual_bounds(
    full_operator: np.ndarray,
    right_inverse: np.ndarray,
    right_hand_side: np.ndarray,
) -> tuple[float, float, float, float, float]:
    r"""Return the exact one-step reduction, its coefficient, and the estimate.

    Let $\tilde{A}=AW$ be the right-preconditioned operator. The exact
    reduction is $\min_{\alpha}\lVert b-\alpha\tilde{A}b\rVert_2/\lVert
    b\rVert_2$ and the coefficient is the minimizing $\alpha$; when
    $\tilde{A}b$ vanishes to working precision the coefficient is exactly 0.0
    and the reduction exactly 1.0, the threshold being
    ``np.finfo(float).eps * max(1.0, inflation) * np.linalg.norm(b)``. The
    coercivity is the least eigenvalue of the symmetric part of $\tilde{A}$
    and the inflation is the operator 2-norm of $\tilde{A}$; the estimate is
    the sharpest bound on the reduction that those two scalars supply
    simultaneously for every right-hand side.

    Raises ValueError unless every input is finite; full_operator and
    right_inverse are square with the same nonempty shape (n, n);
    right_hand_side has shape (n,) and nonzero norm; the inflation is
    positive; and the exact reduction does not exceed the estimate by more
    than 1e-9.

    Parameters
    ----------
    full_operator : np.ndarray
        Complete, potentially nonsymmetric operator A of shape (n, n).
    right_inverse : np.ndarray
        Right-preconditioning action W of shape (n, n).
    right_hand_side : np.ndarray
        Nonzero right-hand side b of shape (n,).

    Returns
    -------
    tuple[float, float, float, float, float]
        Exact one-step reduction, optimal coefficient, field-of-values
        estimate, coercivity, and inflation.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_certify_residual_bounds(
    full_operator: np.ndarray,
    right_inverse: np.ndarray,
    right_hand_side: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Reference one-step field-of-values certificate."""
    operator = np.asarray(full_operator, dtype=float)
    inverse = np.asarray(right_inverse, dtype=float)
    rhs = np.asarray(right_hand_side, dtype=float)
    if (
        operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or operator.shape[0] == 0
    ):
        raise ValueError("full_operator must be a nonempty square matrix")
    n_dof = operator.shape[0]
    if inverse.shape != (n_dof, n_dof):
        raise ValueError("right_inverse must have shape (n, n)")
    if rhs.shape != (n_dof,):
        raise ValueError("right_hand_side must have shape (n,)")
    if not all(np.all(np.isfinite(x)) for x in (operator, inverse, rhs)):
        raise ValueError("all inputs must be finite")
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm == 0.0:
        raise ValueError("right_hand_side must have nonzero norm")

    preconditioned = operator @ inverse
    symmetric = 0.5 * (preconditioned + preconditioned.T)
    symmetric_low = float(np.linalg.eigvalsh(symmetric)[0])
    spectral_norm = float(np.linalg.svd(preconditioned, compute_uv=False)[0])
    if spectral_norm <= 0.0:
        raise ValueError("the preconditioned operator must be nonzero")
    clamped = max(symmetric_low, 0.0)
    estimate = float(np.sqrt(max(0.0, 1.0 - (clamped / spectral_norm) ** 2)))

    image = preconditioned @ rhs
    image_norm = float(np.linalg.norm(image))
    zero_tolerance = np.finfo(float).eps * max(1.0, spectral_norm) * rhs_norm
    if image_norm <= zero_tolerance:
        coefficient = 0.0
        reduction = 1.0
    else:
        coefficient = float(rhs @ image) / float(image @ image)
        reduction = float(np.linalg.norm(rhs - coefficient * image) / rhs_norm)
    if not all(
        np.isfinite(x)
        for x in (reduction, coefficient, estimate, symmetric_low, spectral_norm)
    ):
        raise ValueError("the certificate quantities must be finite")
    if reduction > estimate + 1e-9:
        raise ValueError("the exact reduction violates the field-of-values estimate")
    return reduction, coefficient, estimate, symmetric_low, spectral_norm

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return task-scale, unpreconditioned, nonnormal, degenerate, invalid."""
    return [
        {
            "setup": """import numpy as np
A = np.array([[104.58,-1.48,.2,0,.25],[-1.51,28.626666666666665,-.603333333333333,.433333333333333,-.25],[.2,-.653333333333333,7.226666666666666,-.306666666666667,.22],[0,.433333333333333,-.346666666666667,4.136666666666667,1.74],[.26,-.25,.22,1.68,2.395]])
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
R = np.array([[-.331419211787616,.086034102962501,.822780175278025,1.4677952822772,1.115852830670655],[7.607543403265093,1.165070606175298,.207739110938568,.012975632230369,-.00628049304442]])
W = np.linalg.solve(M + R.T @ R, np.eye(5))
b = np.array([1.0, -.5, .75, .2, -1.1])
""",
            "call": "np.array(certify_residual_bounds(A, W, b), dtype=float)",
            "gold_call": "np.array(_oracle_certify_residual_bounds(A, W, b), dtype=float)",
        },
        {
            "setup": """import numpy as np
A = np.array([[4.0, .5, 0.0], [.5, 3.0, -.25], [0.0, -.25, 1.5]])
W = np.eye(3)
b = np.array([1.0, -2.0, .5])
""",
            "call": "np.array(certify_residual_bounds(A, W, b), dtype=float)",
            "gold_call": "np.array(_oracle_certify_residual_bounds(A, W, b), dtype=float)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 8.0, 0.0], [-.5, 1.0, 3.0], [0.0, -.25, 1.0]])
W = np.diag([.5, 2.0, 1.5])
b = np.array([.3, 1.0, -.7])
""",
            "call": "np.array(certify_residual_bounds(A, W, b), dtype=float)",
            "gold_call": "np.array(_oracle_certify_residual_bounds(A, W, b), dtype=float)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, 1.0], [0.0, 0.0]])
W = np.eye(2)
b = np.array([1.0, 0.0])
""",
            "call": "np.array(certify_residual_bounds(A, W, b), dtype=float)",
            "gold_call": "np.array(_oracle_certify_residual_bounds(A, W, b), dtype=float)",
        },
        {
            "setup": """import numpy as np
A = np.array([[2.0]])
W = np.array([[0.5]])
b = np.array([3.0])
""",
            "call": "np.array(certify_residual_bounds(A, W, b), dtype=float)",
            "gold_call": "np.array(_oracle_certify_residual_bounds(A, W, b), dtype=float)",
        },
        {
            "setup": """import numpy as np
A = np.array([[3.0, -1.0, .2, 0.0], [.4, 2.0, -.3, .1], [0.0, .25, 1.5, -.2], [.1, 0.0, .3, 1.0]])
W = np.linalg.inv(np.array([[3.0, -1.0, .2, 0.0], [.4, 2.0, -.3, .1], [0.0, .25, 1.5, -.2], [.1, 0.0, .3, 1.0]]))
b = np.array([1.0, .5, -1.5, .25])
""",
            "call": "np.array(certify_residual_bounds(A, W, b), dtype=float)",
            "gold_call": "np.array(_oracle_certify_residual_bounds(A, W, b), dtype=float)",
        },
        {
            "setup": """import numpy as np
A = np.array([[1.0, 12.0, 0.0], [0.0, 1.0, 12.0], [0.0, 0.0, 1.0]])
W = np.eye(3)
b = np.array([1.0, 1.0, 1.0])
""",
            "call": "np.array(certify_residual_bounds(A, W, b), dtype=float)",
            "gold_call": "np.array(_oracle_certify_residual_bounds(A, W, b), dtype=float)",
        },
        {
            "setup": """import numpy as np
A = np.eye(2)
W = np.eye(2)
b = np.zeros(2)
def run_model():
    try:
        certify_residual_bounds(A, W, b)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_residual_bounds(A, W, b)
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
            "setup": """import numpy as np
A = np.eye(2)
W = np.zeros((2, 2))
b = np.ones(2)
def run_model():
    try:
        certify_residual_bounds(A, W, b)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_residual_bounds(A, W, b)
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
            "setup": """import numpy as np
A = np.eye(3)
W = np.eye(2)
b = np.ones(3)
def run_model():
    try:
        certify_residual_bounds(A, W, b)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_certify_residual_bounds(A, W, b)
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
