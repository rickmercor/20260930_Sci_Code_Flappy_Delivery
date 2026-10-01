"""
Compute residual inflation relative to the best reduced-space solve.

Residual inflation measures the loss caused by solving the sketched least-squares problem instead of the best unsketched problem available in the same reduced search space. It separates sketching error from the approximation limitation imposed by the Krylov dimension itself.

Returns
-------
tuple[float,float], residual-inflation factor and optimal full-space squared residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def deterministic_residual_inflation(M: np.ndarray, b: np.ndarray, residual_sq: float) -> tuple[float,float]:
    r"""Compute residual inflation relative to the best reduced-space solve.

    Parameters
    ----------
    M : np.ndarray
        Finite full-column-rank reduced design matrix of shape $n\times m$.
    b : np.ndarray
        Finite right-hand-side vector of length $n$.
    residual_sq : float
        Finite nonnegative squared residual from the sketched solve.

    Returns
    -------
    rho_det : float
        Deterministic residual-inflation factor
        $\rho_{\mathrm{det}}=\mathrm{residual\_sq}/\min_x\lVert b-Mx\rVert_2^2$.
    optimal_residual_sq : float
        Minimum squared residual attainable over the same reduced
        search space.

    Raises
    ------
    ValueError
        If M is invalid or rank deficient, if b is invalid, if
        residual_sq is not finite and nonnegative, or if the optimal
        reduced-space residual is zero.
    """
    return rho_det, optimal_residual_sq

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_deterministic_residual_inflation(M, b, residual_sq):
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if M.ndim != 2 or min(M.shape) < 1 or not np.all(np.isfinite(M)):
        raise ValueError("invalid M")
    n, m = M.shape
    if np.linalg.matrix_rank(M) != m:
        raise ValueError("M must be full column rank")
    if b.ndim != 1 or b.shape[0] != n or not np.all(np.isfinite(b)):
        raise ValueError("invalid b")
    if not isinstance(residual_sq, (int, float, np.integer, np.floating)) or not np.isfinite(residual_sq) or float(residual_sq) < 0:
        raise ValueError("invalid residual_sq")
    x = np.linalg.lstsq(M, b, rcond=None)[0]
    r = b - M @ x
    opt = float(r @ r)
    if opt <= 0:
        raise ValueError("optimal residual must be positive")
    return float(residual_sq) / opt, opt

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
M=np.array([[1.,0.],[0.,1.],[1.,1.]])
b=np.array([1.,2.,4.]); residual_sq=2.0""",'call':'deterministic_residual_inflation(M,b,residual_sq)','gold_call':'_oracle_deterministic_residual_inflation(M,b,residual_sq)'},
        {'setup':"""import numpy as np
M=np.array([[1.],[2.],[4.]])
b=np.array([0.,1.,0.]); residual_sq=.9""",'call':'deterministic_residual_inflation(M,b,residual_sq)','gold_call':'_oracle_deterministic_residual_inflation(M,b,residual_sq)'},
        {'setup':"""import numpy as np
M=np.array([[1.,.2],[.3,1.],[.5,-.4],[.2,.8]])
b=np.array([.4,-.2,.9,.1]); residual_sq=1.2""",'call':'deterministic_residual_inflation(M,b,residual_sq)','gold_call':'_oracle_deterministic_residual_inflation(M,b,residual_sq)'},
        {'setup':"""import numpy as np
M=np.eye(2); b=np.array([1.,2.]); residual_sq=0.
def run_model():
    try: deterministic_residual_inflation(M,b,residual_sq); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_deterministic_residual_inflation(M,b,residual_sq); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
M=np.array([[1.],[2.]]); b=np.array([0.,1.]); residual_sq=-1.
def run_model():
    try: deterministic_residual_inflation(M,b,residual_sq); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_deterministic_residual_inflation(M,b,residual_sq); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
M=np.array([[1.,0.],[2.,0.],[3.,0.]]); b=np.ones(3); residual_sq=1.
def run_model():
    try: deterministic_residual_inflation(M,b,residual_sq); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_deterministic_residual_inflation(M,b,residual_sq); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
