"""
Turn the Tikhonov minimisation into one overdetermined linear least-squares problem and return the assembled matrix and right-hand side packed into a single flat array. This is where the regularisation is actually chosen, because the rows that implement the penalty are the only record of which norm it is taken in, and that norm is the source paper's.

Minimising a sum of squared quadratic terms is the same as stacking the discretised operator of each term into one tall matrix and solving in the least-squares sense, provided each block carries the square root of its own weight. A term that is an integral over the horizon and a term that is a single point evaluation do not scale the same way with the time step, and a term carrying a regularisation weight picks that up as well. Getting one of those square roots wrong does not produce a wrong-looking system, it produces a system that is regularised by the wrong amount.

The residual term contributes one block per time interval, evaluated at the midpoint, with the difference quotient standing for the derivative and the average of the two endpoint states standing for the state. The data term is a single identity block and is the only source of a non-zero right-hand side. The regularisation term contributes one block per grid point at which each of its constituents is defined, which is why those blocks do not all have the same count: an undifferentiated quantity is defined everywhere, a first difference loses one point and a second difference loses two. How many such groups there are is determined by the norm the paper specifies, so the height of the assembled system is not something the caller dictates.

The number of unknowns is the number of grid points times the number of modes. Row ordering has no effect on the solution, since permuting the rows permutes the terms of a sum.

Tikhonov regularisation replaces an unstable problem with a nearby stable one by adding a penalty that forbids the wild solutions. Which penalty is chosen is a modelling decision, not a detail: penalising the size of the solution suppresses large answers, penalising the size of its derivative suppresses rough answers, and penalising a second derivative suppresses answers that bend sharply. For a trajectory reconstructed against an unstable evolution, the instability expresses itself as oscillation in time, so how far up the derivatives the penalty reaches decides what the method can and cannot control. The source states which norm it uses.

The equivalence between minimising a quadratic functional and solving a stacked least-squares problem is what makes any of this computable. A sum of squared norms of linear expressions in the unknown is itself the squared norm of one taller linear expression, formed by stacking. The only subtlety is the weights: a term multiplied by a constant in the functional corresponds to a block multiplied by the square root of that constant in the matrix, because the block gets squared when the norm is taken.

Discretising an integral term contributes the same square-root logic. Approximating an integral by a sum over intervals with a common width means each summand carries that width, so the corresponding matrix block carries its square root. A term that is a point evaluation rather than an integral carries no such factor at all, which is why a data block sits at weight one and does not shrink as the time grid is refined.

Returns
-------
np.ndarray of shape (nrow*ncol + nrow,) with ncol = (nt+1)*(N+1). The first nrow*ncol entries are the augmented matrix flattened row-major, the remaining nrow the right-hand side. The row count nrow follows from the terms of the functional and the time grid and is not supplied by the caller.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_tikhonov_system(u0d, c_stack, T, alpha):
    """Assemble the Tikhonov minimisation as one least-squares system.

    Builds the augmented matrix and right-hand side whose least-squares
    solution minimises the discretised Tikhonov functional over the stacked
    coefficient trajectory v(t_0), ..., v(t_nt) on a uniform grid of nt
    intervals.  The residual blocks are evaluated at the nt interval
    midpoints with the forward difference quotient for the derivative and the
    average of the two endpoint states for the state, scaled by sqrt(dt); the
    data block is a single identity of weight one; the penalty blocks are
    scaled by sqrt(alpha*dt), using the forward difference quotient wherever
    a first time derivative appears in the paper's norm and the three-point
    central quotient wherever a second one does.

    Args:
        u0d (np.ndarray): shape (N+1,), Legendre coefficients of the data.
        c_stack (np.ndarray): the nt coefficient matrices at the interval
            midpoints, each (N+1) by (N+1), flattened row-major and
            concatenated in time order.
        T (float): horizon.
        alpha (float): regularisation weight.

    Expected return:
        np.ndarray of shape (nrow*ncol + nrow,) with ncol = (nt+1)*(N+1).
        The first nrow*ncol entries are the augmented matrix flattened
        row-major, the remaining nrow the right-hand side.  The row count
        nrow follows from the terms of the functional.

    Raises:
        ValueError: if u0d is empty, if T <= 0, if alpha <= 0, if the length
        of c_stack is not a multiple of (N+1)**2, or if it implies fewer
        than two time intervals.
    """
    return np.zeros(1)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_tikhonov_system(u0d, c_stack, T, alpha):
    def _build(nt, m, dt, alpha, C, data):
        nvar = (nt + 1) * m
        nrow = m * (4 * nt + 1)
        M = np.zeros((nrow, nvar))
        rhs = np.zeros(nrow)
        eye = np.eye(m)
        sdt = float(np.sqrt(dt))
        sa = float(np.sqrt(alpha * dt))
        p = 0
        for k in range(nt):
            M[p:p + m, k * m:(k + 1) * m] = sdt * (-eye / dt - 0.5 * C[k])
            M[p:p + m, (k + 1) * m:(k + 2) * m] = sdt * (eye / dt - 0.5 * C[k])
            p += m
        M[p:p + m, 0:m] = eye
        rhs[p:p + m] = data
        p += m
        for k in range(nt + 1):
            M[p:p + m, k * m:(k + 1) * m] = sa * eye
            p += m
        for k in range(nt):
            M[p:p + m, k * m:(k + 1) * m] = -sa * eye / dt
            M[p:p + m, (k + 1) * m:(k + 2) * m] = sa * eye / dt
            p += m
        for k in range(1, nt):
            M[p:p + m, (k - 1) * m:k * m] = sa * eye / dt ** 2
            M[p:p + m, k * m:(k + 1) * m] = -2.0 * sa * eye / dt ** 2
            M[p:p + m, (k + 1) * m:(k + 2) * m] = sa * eye / dt ** 2
            p += m
        return M, rhs

    u0d = np.asarray(u0d, dtype=float).reshape(-1)
    c_stack = np.asarray(c_stack, dtype=float).reshape(-1)
    T = float(T)
    alpha = float(alpha)
    m = u0d.size
    if m < 1:
        raise ValueError("u0d must be non-empty")
    if T <= 0.0:
        raise ValueError("T must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if c_stack.size % (m * m) != 0 or c_stack.size == 0:
        raise ValueError("c_stack length is not a multiple of (N+1)**2")
    nt = c_stack.size // (m * m)
    if nt < 2:
        raise ValueError("at least two time intervals are required")
    C = c_stack.reshape(nt, m, m)
    M, rhs = _build(nt, m, T / nt, alpha, C, u0d)
    return np.concatenate((M.reshape(-1), rhs))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "u0 = np.array([1.0, 2.0, 3.0])\ncs = np.tile(np.eye(3).reshape(-1), 4)\ndef run_model():\n    return float(assemble_tikhonov_system(u0, cs, 1.0, 1e-03).size)\ndef run_gold():\n    return float(_oracle_assemble_tikhonov_system(u0, cs, 1.0, 1e-03).size)",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "u0 = np.array([1.0, 2.0, 3.0])\ncs = np.tile(np.eye(3).reshape(-1), 4)\nT, alpha, ncol = 1.0, 1e-03, 15\ndef run_model():\n    s = assemble_tikhonov_system(u0, cs, T, alpha)\n    nrow = s.size // (ncol + 1)\n    M = s[:nrow * ncol].reshape(nrow, ncol)\n    b = s[nrow * ncol:]\n    return np.concatenate(((M.T @ M).reshape(-1), M.T @ b))\ndef run_gold():\n    s = _oracle_assemble_tikhonov_system(u0, cs, T, alpha)\n    nrow = s.size // (ncol + 1)\n    M = s[:nrow * ncol].reshape(nrow, ncol)\n    b = s[nrow * ncol:]\n    return np.concatenate(((M.T @ M).reshape(-1), M.T @ b))",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-07,
        },
        {
            "setup": "rng = np.random.default_rng(8)\nu0 = rng.standard_normal(3)\ncs = rng.standard_normal(27)\nT, alpha, ncol = 2.0, 0.3, 12\ndef run_model():\n    s = assemble_tikhonov_system(u0, cs, T, alpha)\n    nrow = s.size // (ncol + 1)\n    M = s[:nrow * ncol].reshape(nrow, ncol)\n    b = s[nrow * ncol:]\n    return np.concatenate(((M.T @ M).reshape(-1), M.T @ b))\ndef run_gold():\n    s = _oracle_assemble_tikhonov_system(u0, cs, T, alpha)\n    nrow = s.size // (ncol + 1)\n    M = s[:nrow * ncol].reshape(nrow, ncol)\n    b = s[nrow * ncol:]\n    return np.concatenate(((M.T @ M).reshape(-1), M.T @ b))",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-07,
        },
        {
            "setup": "u0 = np.array([1.0, 2.0, 3.0])\ncs = np.tile(np.eye(3).reshape(-1), 4)\ndef run_model():\n    s = assemble_tikhonov_system(u0, cs, 1.0, 1e-03)\n    nrow = s.size // 16\n    return float(np.linalg.norm(s[:nrow * 15]))\ndef run_gold():\n    s = _oracle_assemble_tikhonov_system(u0, cs, 1.0, 1e-03)\n    nrow = s.size // 16\n    return float(np.linalg.norm(s[:nrow * 15]))",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-08,
        },
        {
            "setup": "u0 = np.array([0.7])\ncs = np.array([0.4, -0.2])\nT, alpha, ncol = 1.0, 1e-03, 3\ndef run_model():\n    s = assemble_tikhonov_system(u0, cs, T, alpha)\n    nrow = s.size // (ncol + 1)\n    M = s[:nrow * ncol].reshape(nrow, ncol)\n    b = s[nrow * ncol:]\n    return np.concatenate(((M.T @ M).reshape(-1), M.T @ b))\ndef run_gold():\n    s = _oracle_assemble_tikhonov_system(u0, cs, T, alpha)\n    nrow = s.size // (ncol + 1)\n    M = s[:nrow * ncol].reshape(nrow, ncol)\n    b = s[nrow * ncol:]\n    return np.concatenate(((M.T @ M).reshape(-1), M.T @ b))",
            "call": "run_model()",
            "gold_call": "run_gold()",
            "tol": 1e-06,
        },
        {
            "setup": "u0 = np.array([1.0, 2.0, 3.0])\ncs = np.tile(np.eye(3).reshape(-1), 4)\ndef run_model():\n    try:\n        assemble_tikhonov_system(u0, cs, 1.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_assemble_tikhonov_system(u0, cs, 1.0, 0.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "u0 = np.array([1.0, 2.0, 3.0])\ndef run_model():\n    try:\n        assemble_tikhonov_system(u0, np.ones(10), 1.0, 1e-03)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_assemble_tikhonov_system(u0, np.ones(10), 1.0, 1e-03)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
