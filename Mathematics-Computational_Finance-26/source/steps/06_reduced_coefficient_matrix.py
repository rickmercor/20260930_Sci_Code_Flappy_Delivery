"""
Combine the two reduced operator blocks into the coefficient matrix that drives the reduced evolution. The combination is fixed by the source paper's reduction, and it is the single place in the pipeline where the direction of time is encoded.

The pricing equation has three terms, and the reduction turns each into a contribution to this matrix: the second-order term through the diffusion block, the first-order term through the convection block, and the zeroth-order discounting term through something simpler, because the projection of multiplication by a constant onto the basis is not an integral at all. The coefficients that multiply the three contributions, and the signs they carry, come from writing the equation in the direction the reconstruction runs rather than the direction pricing normally runs. Every sign reverses between the two, and the third contribution is the one most often dropped, because in the ordinary direction it reads as bookkeeping rather than as part of the operator.

The result is a square matrix of the same size as its two inputs, non-symmetric, and time-dependent through the diffusion block alone. One of its columns collapses to something trivial for reasons that involve only the lowest mode, which is the cheapest available check on the whole assembly chain: if that column comes out dense, the error is upstream in one of the two operator blocks or in an index order, not here.

The step takes both blocks as flattened square arrays and returns the combination in the same layout, so it can be applied to whichever time level the caller needs without rebuilding anything.

A parabolic equation integrated in its natural direction smooths, and the same equation integrated the other way amplifies. The generator is the same object in both cases up to a sign, but the sign is everything: it decides whether the eigenvalues of the reduced coefficient matrix have negative real parts, and the solution decays, or positive ones, and it grows. An implementation that carries the signs of the ordinary direction will produce a perfectly well-behaved computation whose answer solves a different problem.

The zeroth-order term deserves separate attention because it behaves differently from the other two. The diffusion and drift contributions are genuine differential operators whose projections mix modes; discounting is multiplication by a constant, and the projection of multiplication by a constant onto an orthonormal basis is that constant times the identity. It therefore shifts the whole spectrum uniformly rather than redistributing anything, which is precisely why it is easy to overlook.

Structural facts about the assembled matrix come from the grading of the basis. The lowest mode is constant, so both of its derivatives vanish identically and it contributes nothing to any integral involving a derivative. Only a term that does not differentiate its argument can reach the part of the matrix that mode governs.

Returns
-------
np.ndarray of shape ((N+1)**2,), the coefficient matrix flattened row-major in the same layout as its two inputs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reduced_coefficient_matrix(a_flat, b_flat, r):
    """Coefficient matrix of the reduced forward-time evolution.

    Combines the reduced diffusion block A and the reduced convection block B
    into the matrix C that satisfies the reduced system u'(t) = C(t) u(t) for
    the forward-time pricing equation, in the combination the source paper's
    reduction gives.

    Args:
        a_flat (np.ndarray): reduced diffusion matrix at the required time,
            flattened row-major, square.
        b_flat (np.ndarray): reduced convection matrix, flattened row-major,
            square and of the same size.
        r (float): risk-free rate.

    Expected return:
        np.ndarray of shape ((N+1)**2,), the coefficient matrix flattened
        row-major in the same layout as the inputs.

    Raises:
        ValueError: if either input is not a flattened square matrix, or if
        the two do not have the same shape.
    """
    return np.zeros(np.asarray(a_flat).size)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reduced_coefficient_matrix(a_flat, b_flat, r):
    def _square(v):
        v = np.asarray(v, dtype=float).reshape(-1)
        k = int(round(float(np.sqrt(v.size))))
        if k * k != v.size or v.size == 0:
            raise ValueError("flattened input is not a square matrix")
        return v.reshape(k, k)

    A = _square(a_flat)
    B = _square(b_flat)
    r = float(r)
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    C = -0.5 * A - r * B + r * np.eye(A.shape[0])
    return C.reshape(-1)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "pass",
            "call": "reduced_coefficient_matrix(np.arange(9.0), np.ones(9), 0.5)",
            "gold_call": "_oracle_reduced_coefficient_matrix(np.arange(9.0), np.ones(9), 0.5)",
        },
        {
            "setup": "pass",
            "call": "reduced_coefficient_matrix(np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0]), 0.0)",
            "gold_call": "_oracle_reduced_coefficient_matrix(np.array([1.0, 2.0, 3.0, 4.0]), np.array([5.0, 6.0, 7.0, 8.0]), 0.0)",
        },
        {
            "setup": "pass",
            "call": "reduced_coefficient_matrix(np.zeros(16), np.zeros(16), 0.05)",
            "gold_call": "_oracle_reduced_coefficient_matrix(np.zeros(16), np.zeros(16), 0.05)",
        },
        {
            "setup": "rng = np.random.default_rng(3)\nA = rng.standard_normal((5, 5))\nB = rng.standard_normal((5, 5))\nA[:, 0] = 0.0\nB[:, 0] = 0.0",
            "call": "reduced_coefficient_matrix(A.reshape(-1), B.reshape(-1), 0.05)",
            "gold_call": "_oracle_reduced_coefficient_matrix(A.reshape(-1), B.reshape(-1), 0.05)",
        },
        {
            "setup": "pass",
            "call": "reduced_coefficient_matrix(np.zeros(1), np.zeros(1), 2.0)",
            "gold_call": "_oracle_reduced_coefficient_matrix(np.zeros(1), np.zeros(1), 2.0)",
        },
        {
            "setup": "def run_model():\n    try:\n        reduced_coefficient_matrix(np.zeros(5), np.zeros(5), 0.05)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduced_coefficient_matrix(np.zeros(5), np.zeros(5), 0.05)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        reduced_coefficient_matrix(np.zeros(9), np.zeros(4), 0.05)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_reduced_coefficient_matrix(np.zeros(9), np.zeros(4), 0.05)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
