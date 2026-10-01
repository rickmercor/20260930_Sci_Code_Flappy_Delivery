"""
Solve the deterministically sketched GMRES least-squares problem.

Sketched GMRES minimizes a compressed residual while retaining the original Krylov search space. The resulting approximation must therefore be judged by its residual in the full ambient system rather than by the compressed residual alone.

Returns
-------
tuple[np.ndarray,float], coefficient vector y and full-space squared residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def deterministic_sgmres_solve(V: np.ndarray, M: np.ndarray, b: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray,float]:
    r"""Solve the deterministically sketched GMRES least-squares problem.

    Parameters
    ----------
    V : np.ndarray
        Finite full-column-rank Krylov basis of shape $n\times m$.
    M : np.ndarray
        Finite full-column-rank matrix image $M=AV$ of shape $n\times m$,
        compatible with V.
    b : np.ndarray
        Finite right-hand-side vector of length $n$.
    indices : np.ndarray
        One-dimensional integer array containing at least $m$ distinct
        one-based sampled row indices.

    Returns
    -------
    y : np.ndarray
        Float64 Krylov coefficient vector $y$ of length $m$.
    residual_sq : float
        Full-space squared residual $\lVert b-My\rVert_2^2$ as a native
        Python float.

    Raises
    ------
    ValueError
        If V or M is invalid, nonfinite, incompatible, or rank
        deficient; if b has invalid shape or values; if indices are
        invalid or repeated; or if the sampled matrix image is rank
        deficient.
    """
    return y, residual_sq

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_deterministic_sgmres_solve(V, M, b, indices):
    V = np.asarray(V, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    idx = np.asarray(indices)
    if V.ndim != 2 or M.shape != V.shape or not np.all(np.isfinite(V)) or not np.all(np.isfinite(M)):
        raise ValueError("V and M must be finite and same shape")
    n, m = V.shape
    if np.linalg.matrix_rank(V) != m or np.linalg.matrix_rank(M) != m:
        raise ValueError("V and M must be full column rank")
    if b.ndim != 1 or b.shape[0] != n or not np.all(np.isfinite(b)):
        raise ValueError("invalid b")
    if idx.ndim != 1 or idx.size < m or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("invalid indices")
    idx = idx.astype(np.int64)
    if np.unique(idx).size != idx.size or np.any(idx < 1) or np.any(idx > n):
        raise ValueError("invalid indices")
    SM = M[idx - 1, :]
    if np.linalg.matrix_rank(SM) != m:
        raise ValueError("sampled image must be full rank")
    Q, R = np.linalg.qr(SM, mode='reduced')
    y = np.linalg.solve(R, Q.T @ b[idx - 1])
    r = b - M @ y
    return y.astype(np.float64), float(r @ r)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
V=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,-.5]])
M=np.array([[2.,.2],[.1,1.8],[2.1,2.],[3.9,-.7]])
b=np.array([1.,-.3,.8,.2]); indices=np.array([1,2,4])""",'call':'deterministic_sgmres_solve(V,M,b,indices)','gold_call':'_oracle_deterministic_sgmres_solve(V,M,b,indices)'},
        {'setup':"""import numpy as np
V=np.array([[1.],[2.],[3.]])
M=np.array([[2.],[1.],[4.]])
b=np.array([1.,0.,2.]); indices=np.array([1])""",'call':'deterministic_sgmres_solve(V,M,b,indices)','gold_call':'_oracle_deterministic_sgmres_solve(V,M,b,indices)'},
        {'setup':"""import numpy as np
V=np.array([[1.,0.],[0.,1.],[1.,1.]])
M=np.array([[1.,.2],[.3,1.1],[1.4,1.2]])
b=np.array([.2,.5,-.1]); indices=np.array([1,2,3])""",'call':'deterministic_sgmres_solve(V,M,b,indices)','gold_call':'_oracle_deterministic_sgmres_solve(V,M,b,indices)'},
        {'setup':"""import numpy as np
V=np.eye(2); M=np.eye(2); b=np.ones(2); indices=np.array([1,1])
def run_model():
    try: deterministic_sgmres_solve(V,M,b,indices); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_deterministic_sgmres_solve(V,M,b,indices); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
V=np.eye(2); M=np.array([[1.,0.],[2.,0.]]); b=np.ones(2); indices=np.array([1,2])
def run_model():
    try: deterministic_sgmres_solve(V,M,b,indices); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_deterministic_sgmres_solve(V,M,b,indices); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
V=np.eye(2); M=np.eye(2); b=np.array([1.,np.nan]); indices=np.array([1,2])
def run_model():
    try: deterministic_sgmres_solve(V,M,b,indices); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_deterministic_sgmres_solve(V,M,b,indices); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
