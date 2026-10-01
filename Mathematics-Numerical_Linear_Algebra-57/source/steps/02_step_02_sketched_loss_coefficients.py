"""
One accelerated iteration replaces `$M$$by$$(I + alpha R) ** p M$`, where `$R = I - M$` is the residual carried into the step and ``alpha`` is the single free scalar the accelerated method refits at that iteration. The residual after the step is read off the new `$M$` by the same defining relation. Every iterate of the coupled pair is a polynomial in the original matrix, so `$R$` commutes with `$M$` and no ordering convention is needed anywhere below.



Eliminating `$M$` in favour of `$R$$turns the post-step residual into a matrix polynomial in$$R$` alone whose scalar coefficients are polynomials in ``alpha``: one degree higher in `$R$` than the order of the step, and of degree exactly the order of the step in ``alpha``. The constant term in ``alpha`` is the residual carried in, unchanged, which is why setting ``alpha`` to zero freezes the iteration rather than restarting it. Deriving that expansion is the first half of this step; it depends on `$p$` alone and forms no matrix.



The free scalar is then chosen by minimising how large that post-step residual is, but measuring it in full would cost a cubic number of operations in the matrix dimension at every iteration. The published fit instead compresses the post-step residual from the left with a fixed sketch `$S$` and minimises the squared Frobenius norm of the compressed matrix, so the quantity being minimised is a scalar polynomial in ``alpha``. This step returns its coefficients in ascending order.



Because `$R$$is symmetric, so is the post-step residual at every$`alpha``, and the squared Frobenius norm of `$S$` against it collapses onto a trace of `$S$` against a single power of `$R$`. Each coefficient of the loss is therefore a linear combination of the scalar sketched moments ``trace(S @ R ** i @ S.T)``, and how far `$i$` has to run is fixed by the degree of the expansion alone. Accumulating those scalars, rather than the matrix powers behind them, is what reduces the fit from cubic to sketch-dependent quadratic scaling in the matrix dimension; it is also why the sketch must be applied before the norm is squared rather than after.



The derivation assumes an exactly symmetric residual. An input admitted by the absolute symmetry tolerance therefore represents ``(R + R.T) / 2`` and must be projected onto that invariant subspace before any moment is accumulated. This is observable when a small residual is paired with a large sketch, and is not the same as silently using the accepted but nonsymmetric entries. The construction is also degree-general: no coefficient table for one particular `$p$` may be hard-coded, and the returned length and highest moment must continue to follow from the input order when the derivative degree is well above five.

Returns
-------
np.ndarray of shape (2 * p + 1,), sketched loss coefficients in ascending alpha order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sketched_loss_coefficients(R: np.ndarray, S: np.ndarray, p: int) -> np.ndarray:
    """Reduce the sketched post-step residual to a scalar polynomial in ``alpha``.

    Raises ``ValueError`` unless every one of the following holds: ``R`` and
    ``S`` are real rather than complex; ``R`` is a nonempty square
    two-dimensional array; ``R`` is symmetric to an absolute tolerance of
    ``1e-12``, so a residual that is asymmetric only at the ``1e-13`` level is
    accepted and projected to ``(R + R.T) / 2`` rather than rejected; ``S`` is
    two-dimensional with at least one row and with its column count equal to the
    dimension of ``R``, so a sketch of the wrong width is rejected rather than
    broadcast; every entry of both is finite; and ``p`` is an integer, not a
    bool, with ``p >= 1``.

    Parameters
    ----------
    R : np.ndarray
        Nonempty finite real symmetric residual of shape ``(n, n)``.
    S : np.ndarray
        Fixed finite real sketch of shape ``(m, n)`` with ``m >= 1``.
    p : int
        Root order of the accelerated step.

    Returns
    -------
    np.ndarray
        Float64 vector of shape ``(2 * p + 1,)`` holding the coefficients of the
        sketched loss in ascending powers of ``alpha``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sketched_loss_coefficients(
    R: np.ndarray,
    S: np.ndarray,
    p: int,
) -> np.ndarray:
    """Reference sketched loss, assembled from scalar trace moments."""
    if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
        raise ValueError("p must be an integer")
    if int(p) < 1:
        raise ValueError("p must be at least one")
    order = int(p)
    residual = np.asarray(R)
    sketch = np.asarray(S)
    if np.iscomplexobj(residual) or np.iscomplexobj(sketch):
        raise ValueError("R and S must be real")
    residual = np.asarray(residual, dtype=float)
    sketch = np.asarray(sketch, dtype=float)
    if (
        residual.ndim != 2
        or residual.shape[0] != residual.shape[1]
        or residual.shape[0] == 0
    ):
        raise ValueError("R must be a nonempty square matrix")
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(sketch)):
        raise ValueError("R and S must contain only finite values")
    if not np.allclose(residual, residual.T, rtol=0.0, atol=1e-12):
        raise ValueError("R must be symmetric")
    if sketch.ndim != 2 or sketch.shape[0] == 0 or sketch.shape[1] != residual.shape[0]:
        raise ValueError("S must have shape (m, n) with m positive")
    projected = residual.copy()
    for row in range(residual.shape[0]):
        for column in range(row + 1, residual.shape[1]):
            left = float(residual[row, column])
            right = float(residual[column, row])
            if np.signbit(left) == np.signbit(right):
                midpoint = left + 0.5 * (right - left)
            else:
                midpoint = 0.5 * (left + right)
            projected[row, column] = midpoint
            projected[column, row] = midpoint
    residual = projected
    width = order + 2
    expansion = np.zeros((order + 1, width), dtype=float)
    expansion[0, 1] = 1.0
    weight = 1.0
    for index in range(1, order + 1):
        weight = weight * (order - index + 1) / index
        expansion[index, index] -= weight
        expansion[index, index + 1] += weight
    moments = np.empty(2 * (width - 1) + 1, dtype=float)
    rolling = sketch.copy()
    for power in range(moments.size):
        moments[power] = float(np.tensordot(rolling, sketch, axes=([0, 1], [0, 1])))
        rolling = rolling @ residual
    gram = np.array(
        [[moments[a + b] for b in range(width)] for a in range(width)],
        dtype=float,
    )
    coefficients = np.zeros(2 * order + 1, dtype=float)
    for left in range(order + 1):
        for right in range(order + 1):
            coefficients[left + right] += float(
                expansion[left] @ gram @ expansion[right]
            )
    return coefficients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return low/high-order, projected-symmetry, zero-loss and invalid cases."""
    return [
        {
            "setup": """import numpy as np
R = np.array([[0.4, 0.1, 0.0], [0.1, -0.7, 0.2], [0.0, 0.2, 0.35]])
S = np.array([[1.0, -0.5, 0.25], [0.2, 0.7, -0.4]])
p = 3
""",
            "call": "sketched_loss_coefficients(R, S, p)",
            "gold_call": "_oracle_sketched_loss_coefficients(R, S, p)",
        },
        {
            "setup": """import numpy as np
R = np.array([[0.5]])
S = np.array([[2.0]])
p = 1
""",
            "call": "sketched_loss_coefficients(R, S, p)",
            "gold_call": "_oracle_sketched_loss_coefficients(R, S, p)",
        },
        {
            "setup": """import numpy as np
R = np.array([[0.4, 0.1, 0.0], [0.1, -0.7, 0.2], [0.0, 0.2, 0.35]])
S = np.array([[1.0, -0.5, 0.25], [0.2, 0.7, -0.4]])
p = 5
""",
            "call": "sketched_loss_coefficients(R, S, p)",
            "gold_call": "_oracle_sketched_loss_coefficients(R, S, p)",
        },
        {
            "setup": """import numpy as np
R = np.array([[-0.25, 0.4], [0.4, 0.6]])
S = np.array([[0.5, -1.25], [1.0, 0.75], [-0.6, 0.2], [0.1, 0.9]])
p = 2
""",
            "call": "sketched_loss_coefficients(R, S, p)",
            "gold_call": "_oracle_sketched_loss_coefficients(R, S, p)",
        },
        {
            "setup": """import numpy as np
R = np.array([[0.4, 0.1, 0.0], [0.1, -0.7, 0.2], [0.0, 0.2, 0.35]])
R[0, 1] += 4e-13
S = np.array([[1.0, -0.5, 0.25], [0.2, 0.7, -0.4]])
p = 4
""",
            "call": "sketched_loss_coefficients(R, S, p)",
            "gold_call": "_oracle_sketched_loss_coefficients(R, S, p)",
        },
        {
            "setup": """import numpy as np
R = np.array([[2e-12, 9e-13], [1e-13, -1e-12]])
S = 1e12 * np.array([[1.0, -0.5], [0.25, 0.75]])
p = 6
""",
            "call": "sketched_loss_coefficients(R, S, p)",
            "gold_call": "_oracle_sketched_loss_coefficients(R, S, p)",
        },
        {
            "setup": """import numpy as np
R = np.array([[0.72, -0.11, 0.04], [-0.11, -0.63, 0.19], [0.04, 0.19, 0.31]])
S = np.array([[0.8, -0.2, 0.5], [-0.4, 0.9, 0.3], [0.1, -0.7, 1.2]])
p = np.int64(8)
""",
            "call": "sketched_loss_coefficients(R, S, p)",
            "gold_call": "_oracle_sketched_loss_coefficients(R, S, p)",
        },
        {
            "setup": """import numpy as np
R = np.zeros((3, 3))
S = np.array([[1.0, -0.5, 0.25], [0.2, 0.7, -0.4]])
p = 3
""",
            "call": "sketched_loss_coefficients(R, S, p)",
            "gold_call": "_oracle_sketched_loss_coefficients(R, S, p)",
        },
        {
            "setup": """import numpy as np
R = np.array([[0.4, 0.9, 0.0], [0.1, -0.7, 0.2], [0.0, 0.2, 0.35]])
S = np.array([[1.0, -0.5, 0.25]])
p = 3
def run_model_asym():
    try:
        sketched_loss_coefficients(R, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_asym():
    try:
        _oracle_sketched_loss_coefficients(R, S, p)
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
R = np.array([[0.4, 0.1], [0.1, -0.7]])
S = np.array([[1.0, -0.5, 0.25]])
p = 3
def run_model_width():
    try:
        sketched_loss_coefficients(R, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_width():
    try:
        _oracle_sketched_loss_coefficients(R, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_width()",
            "gold_call": "run_oracle_width()",
        },
        {
            "setup": """import numpy as np
R = np.array([[0.4, 0.1], [0.1, -0.7]])
S = np.array([[1.0, -0.5]])
p = True
def run_model_bool():
    try:
        sketched_loss_coefficients(R, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_bool():
    try:
        _oracle_sketched_loss_coefficients(R, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_bool()",
            "gold_call": "run_oracle_bool()",
        },
        {
            "setup": """import numpy as np
R = np.array([[0.4, 0.1], [0.1, np.inf]])
S = np.array([[1.0, -0.5]])
p = 3
def run_model_inf():
    try:
        sketched_loss_coefficients(R, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_inf():
    try:
        _oracle_sketched_loss_coefficients(R, S, p)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_inf()",
            "gold_call": "run_oracle_inf()",
        },
    ]
