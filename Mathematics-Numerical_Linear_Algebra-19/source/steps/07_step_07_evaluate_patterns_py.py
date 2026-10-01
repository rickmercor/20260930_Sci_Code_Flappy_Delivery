"""
Assemble the two inverse approximations on fixed supports and evaluate their original-system relative residuals after the prescribed restarted GMRES solves.

Equation (2) distinguishes left and right preconditioning. The transpose construction targets the left product, while right-preconditioned Krylov variables must be mapped to physical iterates before evaluating the original residual.




$$

M_R=S_A,\ M_L=S_{A^{\mathsf T}}^{\mathsf T};\quad AM_Ry=b,\ x_R=M_Ry;\quad M_LAx_L=M_Lb;\quad R_{q,j}=|b^{(j)}-Ax_{q,j}|_2/|b^{(j)}|_2.

$$

Returns
-------
np.ndarray, shape (2, number_of_rhs), containing unrounded left and right true relative residuals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def evaluate_patterns(
    A: np.ndarray,
    rhs: np.ndarray,
    supports: tuple,
    restart: int = 3,
    cycles: int = 2,
) -> np.ndarray:
    """Return true relative residuals for fixed left and right patterns.

    Parameters
    ----------
    A : np.ndarray
        Finite real nonsingular square matrix of size n with nonzero diagonal;
        entries are converted to binary64 before exact column fitting.
    rhs : np.ndarray
        Finite real array of shape (n, m), m >= 1, with no zero column.
        These are the original right-hand sides, with no extra diagonal scaling.
    supports : tuple
        Exactly 2*n nonempty increasing index tuples. Tuple k contains k
        for the kth column of A; tuple n+k contains k for the kth column of A.T.
    restart, cycles : int
        Positive non-Boolean integers, with restart <= n.

    Returns
    -------
    residuals : np.ndarray
        Float64 array of shape (2, m): row 0 is left, row 1 is right.
        Column fits use exact rational arithmetic; Krylov minimization is
        evaluated numerically without rounding reported intermediate values.
        Each solve starts at zero and retains its endpoint at each restart.
        No convergence tolerance is used. An invariant Krylov subspace uses
        its available basis; a zero starting residual leaves the iterate fixed.

    Raises
    ------
    ValueError
        If A fails its conditions; rhs has the wrong shape or a zero column;
        supports is malformed, missing a required diagonal index, or has the
        wrong length; restart/cycles are invalid; or binary64 evaluation cannot
        produce finite coefficients, iterates, or relative residuals.
    """
    return np.empty((0, 0), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _nr_norm(v):
    scale = float(np.max(np.abs(v)))
    return 0.0 if scale == 0 else scale * float(np.linalg.norm(v / scale))


def _nr_gmres(H, g, restart, cycles):
    n = len(g)
    x = np.zeros(n, dtype=float)
    for _ in range(cycles):
        residual = g - H @ x
        beta = _nr_norm(residual)
        if beta == 0:
            continue
        V = np.zeros((n, restart + 1), dtype=float)
        hessenberg = np.zeros((restart + 1, restart), dtype=float)
        V[:, 0] = residual / beta
        dimension = restart
        for j in range(restart):
            w = H @ V[:, j]
            before = _nr_norm(w)
            # Two passes reduce loss of orthogonality in this nonsymmetric problem.
            for _pass in range(2):
                for i in range(j + 1):
                    projection = float(V[:, i] @ w)
                    hessenberg[i, j] += projection
                    w -= projection * V[:, i]
            length = _nr_norm(w)
            hessenberg[j + 1, j] = length
            # Numerical invariant-subspace detection, not a residual tolerance.
            if length <= 32 * np.finfo(float).eps * before:
                dimension = j + 1
                break
            V[:, j + 1] = w / length
        target = np.zeros(dimension + 1)
        target[0] = beta
        correction = np.linalg.lstsq(
            hessenberg[:dimension + 1, :dimension], target, rcond=None
        )[0]
        x += V[:, :dimension] @ correction
    return x


def _oracle_evaluate_patterns(
    A: np.ndarray,
    rhs: np.ndarray,
    supports: tuple,
    restart: int = 3,
    cycles: int = 2,
) -> np.ndarray:
    data = _nr_matrix(A)
    n = data['n']
    A = _nr_real_array(A, 'A', 2)
    rhs = _nr_real_array(rhs, 'rhs', 2)
    if rhs.shape[0] != n or any(not np.any(b) for b in rhs.T):
        raise ValueError("rhs must have n rows and no zero column")
    restart = _nr_integer(restart, 'restart', 1, n)
    cycles = _nr_integer(cycles, 'cycles', 1)
    if not isinstance(supports, (tuple, list)) or len(supports) != 2 * n:
        raise ValueError("supports must contain exactly 2*n columns")
    matrices = []
    for side, C in enumerate((A, A.T)):
        matrix = np.zeros((n, n))
        for k in range(n):
            J = supports[side * n + k]
            v, _, _ = _oracle_fit_column(C, k, J)
            try:
                matrix[list(J), k] = [float(_nr_fraction(z)) for z in v]
            except (OverflowError, FloatingPointError) as exc:
                raise ValueError('inverse coefficients exceed binary64 range') from exc
        matrices.append(matrix)
    MR, ML = matrices[0], matrices[1].T
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            if not np.all(np.isfinite(MR)) or not np.all(np.isfinite(ML)):
                raise ValueError("nonfinite inverse coefficients")
            HR, HL = A @ MR, ML @ A
            result = np.empty((2, rhs.shape[1]))
            for j, b in enumerate(rhs.T):
                right_y = _nr_gmres(HR, b, restart, cycles)
                left_x = _nr_gmres(HL, ML @ b, restart, cycles)
                result[0, j] = _nr_norm(b - A @ left_x) / _nr_norm(b)
                result[1, j] = _nr_norm(b - A @ (MR @ right_y)) / _nr_norm(b)
            if not np.all(np.isfinite(result)):
                raise ValueError("nonfinite true residuals")
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError("binary64 Krylov evaluation failed") from exc
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic setup/call/gold_call test specifications."""
    return [
        # normal: nonsymmetric left/right residuals
        {
            "setup": """import numpy as np
n=6
i=np.arange(n)
A=np.diag(2.0+i)
A[i,(i+1)%n]=-1.0
rhs=np.column_stack((1.0+i,(-1.0)**i))
supports=tuple((k,) for k in range(n))*2

expected = np.array([[0.0024359691350374375, 0.00047512640436924445],
 [0.001995041657854204, 0.00047694985321254993]], dtype=float)

def _check_result(value):
    return int(isinstance(value, np.ndarray) and value.shape == expected.shape
               and np.all(np.isfinite(value))
               and np.allclose(value, expected, rtol=1e-09, atol=1e-11))
""",
            "call": '_check_result(evaluate_patterns(A, rhs, supports, 2, 2))',
            "gold_call": '_check_result(_oracle_evaluate_patterns(A, rhs, supports, 2, 2))',
        },
        # boundary: invariant subspace and exact initial inverse
        {
            "setup": """import numpy as np
A=np.diag([2.0,3.0])
rhs=np.array([[1.0,2.0],[2.0,-1.0]])
supports=((0,),(1,),(0,),(1,))

expected = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float)

def _check_result(value):
    return int(isinstance(value, np.ndarray) and value.shape == expected.shape
               and np.all(np.isfinite(value))
               and np.allclose(value, expected, rtol=0.0, atol=1e-12))
""",
            "call": '_check_result(evaluate_patterns(A, rhs, supports, 2, 2))',
            "gold_call": '_check_result(_oracle_evaluate_patterns(A, rhs, supports, 2, 2))',
        },
        # edge: scaled coordinates with a restart
        {
            "setup": """import numpy as np
n = 8
i = np.arange(n)
B = 3.0 * np.eye(n)
B[i, (i + 1) % n] = -1.0
dl = 2.0**((3*i) % 7 - 3)
dr = 1.0 / dl
A = dl[:, None] * B * dr[None, :]
rhs = np.column_stack((1.0 + i/8, (-1.0)**i + i/16))
box = (((1, 4), (1, 4), 1, 1), ((1, 16), (1, 16), 1, 1))
supports=((0,6,7),(0,1,7),(0,1,2),(1,2,3),(2,3,4),(3,4,5),(4,5,6),(5,6,7),
          (0,1,2),(1,2,3),(2,3,4),(3,4,5),(4,5,6),(5,6,7),(0,6,7),(0,1,7))

expected = np.array([[0.0002098899572169274, 3.172552613611506e-05],
 [3.4216277236422044e-05, 2.0421296190033545e-05]], dtype=float)

def _check_result(value):
    return int(isinstance(value, np.ndarray) and value.shape == expected.shape
               and np.all(np.isfinite(value))
               and np.allclose(value, expected, rtol=1e-08, atol=1e-10))
""",
            "call": '_check_result(evaluate_patterns(A, rhs, supports, 2, 2))',
            "gold_call": '_check_result(_oracle_evaluate_patterns(A, rhs, supports, 2, 2))',
        },
        # invalid: zero original right-hand side
        {
            "setup": """import numpy as np
A=np.eye(2)
rhs=np.zeros((2,1))
supports=((0,),(1,),(0,),(1,))

def _capture_value_error(fn):
    try:
        fn()
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": '_capture_value_error(lambda: evaluate_patterns(A, rhs, supports, 1, 1))',
            "gold_call": '_capture_value_error(lambda: _oracle_evaluate_patterns(A, rhs, supports, 1, 1))',
        },
    ]
