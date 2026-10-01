"""
Solve the assembled least-squares system and return the coefficient trajectory that minimises the discretised Tikhonov functional. The input is the packed matrix and right-hand side produced by the previous step; the output is the stacked trajectory, one block of coefficients per time level.

The system is tall and thin, because the residual, data and regularisation terms each contribute their own rows while the unknowns are only the trajectory itself. It has no exact solution and is not meant to, since the data are noisy and the penalty actively pulls the answer away from a perfect fit. The residual at the minimiser is therefore small but non-zero, and a computation that reports a residual at the level of machine precision has solved a square system somewhere instead of the overdetermined one.

Solve by least squares rather than by forming and inverting the normal equations. The regularisation makes the system well conditioned enough that both routes agree far inside the required tolerance here, but the normal equations square the condition number, and the margin that buys is worth keeping for the harder configurations the same code has to serve.

Only the number of columns is passed in. The number of rows is whatever the assembly produced, and it is recoverable from the length of the packed input, since the packing is a matrix of that width followed by one right-hand-side entry per row. A length that is not consistent with the stated width means an inconsistent call rather than a recoverable situation.

An overdetermined linear system is solved in the least-squares sense by finding the vector that minimises the Euclidean norm of the residual. Geometrically the answer projects the right-hand side onto the column space of the matrix, and it is characterised by the residual being orthogonal to every column, which is the normal-equation condition. That characterisation is the natural way to check a solution without re-deriving it.

Whether to solve via the normal equations or via an orthogonal factorisation is a numerical-stability question with a standard answer. Forming the normal-equation matrix squares the condition number, so a system that is merely awkward becomes badly conditioned and digits are lost that no amount of care later recovers. Orthogonal factorisation works with the original matrix and loses roughly half as many. When a problem is well conditioned the two agree; the discipline is worth keeping anyway, because the condition number is a property of the data and not of the day.

Regularisation is itself a conditioning device. The penalty rows add a multiple of the identity to the normal-equation matrix, which lifts its smallest singular values away from zero and bounds the condition number by roughly the ratio of the largest to the regularisation scale. That is what makes an ill-posed problem computable at all, and it is why the penalty weight controls both how much the answer is biased and how accurately it can be computed.

Returns
-------
np.ndarray of shape (ncol,), the minimising stacked trajectory; entries k*(N+1) to (k+1)*(N+1) are the coefficient vector at time level k.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_tikhonov_system(sys_flat, ncol):
    """Least-squares solution of the assembled Tikhonov system.

    Unpacks the augmented matrix and right-hand side from the packed array.
    The matrix has ncol columns; its row count is recovered from the length
    of sys_flat, which holds nrow*ncol matrix entries followed by nrow
    right-hand-side entries.

    Args:
        sys_flat (np.ndarray): the packed system, matrix flattened row-major
            followed by the right-hand side.
        ncol (int): number of columns, that is (nt+1)*(N+1).

    Expected return:
        np.ndarray of shape (ncol,), the minimising stacked trajectory;
        entries k*(N+1) to (k+1)*(N+1) are the coefficient vector at time
        level k.

    Raises:
        ValueError: if ncol < 1, if the length of sys_flat is not a multiple
        of ncol+1, or if the implied system is not overdetermined.
    """
    return np.zeros(ncol)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_tikhonov_system(sys_flat, ncol):
    def _unpack(v, nrow, ncol):
        return v[:nrow * ncol].reshape(nrow, ncol), v[nrow * ncol:]

    sys_flat = np.asarray(sys_flat, dtype=float).reshape(-1)
    ncol = int(ncol)
    if ncol < 1:
        raise ValueError("ncol must be at least 1")
    if sys_flat.size == 0 or sys_flat.size % (ncol + 1) != 0:
        raise ValueError("sys_flat length is not a multiple of ncol + 1")
    nrow = sys_flat.size // (ncol + 1)
    if nrow < ncol:
        raise ValueError("the implied system is not overdetermined")
    M, rhs = _unpack(sys_flat, nrow, ncol)
    sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return np.asarray(sol, dtype=float).reshape(-1)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "rng = np.random.default_rng(5)\nM = rng.standard_normal((51, 15))\nrhs = rng.standard_normal(51)\ns = np.concatenate((M.reshape(-1), rhs))",
            "call": "solve_tikhonov_system(s, 15)",
            "gold_call": "_oracle_solve_tikhonov_system(s, 15)",
        },
        {
            "setup": "M = np.zeros((51, 15))\nM[:15, :] = np.eye(15)\nrhs = np.zeros(51)\nrhs[:15] = np.arange(15, dtype=float)\ns = np.concatenate((M.reshape(-1), rhs))",
            "call": "solve_tikhonov_system(s, 15)",
            "gold_call": "_oracle_solve_tikhonov_system(s, 15)",
        },
        {
            "setup": "M = np.zeros((26, 8))\nM[:8, :] = np.eye(8)\ns = np.concatenate((M.reshape(-1), np.zeros(26)))",
            "call": "solve_tikhonov_system(s, 8)",
            "gold_call": "_oracle_solve_tikhonov_system(s, 8)",
        },
        {
            "setup": "M = np.zeros((51, 15))\nM[:15, :] = 2.0 * np.eye(15)\nM[15:30, :] = np.eye(15)\nrhs = np.zeros(51)\nrhs[:15] = 10.0\ns = np.concatenate((M.reshape(-1), rhs))",
            "call": "solve_tikhonov_system(s, 15)",
            "gold_call": "_oracle_solve_tikhonov_system(s, 15)",
            "tol": 1e-08,
        },
        {
            "setup": "M = np.ones((9, 1))\nrhs = np.full(9, 2.0)\ns = np.concatenate((M.reshape(-1), rhs))",
            "call": "solve_tikhonov_system(s, 1)",
            "gold_call": "_oracle_solve_tikhonov_system(s, 1)",
            "tol": 1e-08,
        },
        {
            "setup": "def run_model():\n    try:\n        solve_tikhonov_system(np.zeros(100), 7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_solve_tikhonov_system(np.zeros(100), 7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        solve_tikhonov_system(np.zeros(24), 11)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_solve_tikhonov_system(np.zeros(24), 11)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
