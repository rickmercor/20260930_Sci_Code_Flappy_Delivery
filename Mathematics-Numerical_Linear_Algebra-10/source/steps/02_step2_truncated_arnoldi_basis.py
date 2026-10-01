"""
Generate a truncated-Arnoldi basis and its matrix image.

Incomplete orthogonalization reduces the cost of basis generation but permits correlations with older Krylov vectors. Those correlations are scientifically important here because the deterministic row-selection methods act on the basis that is actually generated, not on an idealized orthogonal replacement.

Returns
-------
tuple[np.ndarray,np.ndarray], float64 arrays $V\in\mathbb{R}^{n\times m}$ and $M=AV\in\mathbb{R}^{n\times m}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def truncated_arnoldi_basis(A: np.ndarray, b: np.ndarray, m: int, k: int) -> tuple[np.ndarray,np.ndarray]:
    r"""Generate a truncated-Arnoldi basis and its matrix image.

    Parameters
    ----------
    A : np.ndarray
        Finite nonempty square matrix of shape $n\times n$.
    b : np.ndarray
        Finite nonzero starting vector of length n.
    m : int
        Number of Krylov basis vectors, satisfying $1\le m\le n$.
    k : int
        Positive truncation length, satisfying $1\le k\le m$; the new
        vector is orthogonalized against the $k$ most recent basis vectors
        only, with $v_1=b/\lVert b\rVert_2$. For each new column, compute
        every projection coefficient against the original image
        $Av_j$, then subtract the corresponding basis components
        (one pass of classical, not modified, Gram--Schmidt).
        Breakdown means that the new norm is at most binary64 machine
        epsilon times $\lVert Av_j\rVert_2$.

    Returns
    -------
    V : np.ndarray
        Float64 truncated-Arnoldi basis of shape $n\times m$.
    M : np.ndarray
        Float64 matrix image $M=AV$ of shape $n\times m$.

    Raises
    ------
    ValueError
        If A is not a finite nonempty square matrix, if b is not a
        finite nonzero vector of compatible length, if m or k is
        outside its admissible range, or if Arnoldi breakdown occurs.
    """
    return V, M

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_truncated_arnoldi_basis(A, b, m, k):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be nonempty square")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    n = A.shape[0]
    if b.ndim != 1 or b.shape[0] != n or not np.all(np.isfinite(b)):
        raise ValueError("b must be finite length n")
    if not isinstance(m, (int, np.integer)) or not (1 <= int(m) <= n):
        raise ValueError("invalid m")
    if not isinstance(k, (int, np.integer)) or not (1 <= int(k) <= int(m)):
        raise ValueError("invalid k")
    if np.linalg.norm(b) == 0:
        raise ValueError("b must be nonzero")
    m = int(m)
    k = int(k)
    V = np.empty((n, m), dtype=np.float64)
    M = np.empty((n, m), dtype=np.float64)
    V[:, 0] = b / np.linalg.norm(b)
    M[:, 0] = A @ V[:, 0]
    for j in range(1, m):
        w = M[:, j - 1].copy()
        for i in range(max(0, j - k), j):
            w -= V[:, i] * float(V[:, i] @ M[:, j - 1])
        nw = float(np.linalg.norm(w))
        if not np.isfinite(nw) or nw <= np.finfo(np.float64).eps * np.linalg.norm(M[:, j - 1]):
            raise ValueError("Arnoldi breakdown")
        V[:, j] = w / nw
        M[:, j] = A @ V[:, j]
    return V, M

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
A=np.array([[2.,1.,0.,0.],[.1,2.2,.8,0.],[0.,.1,2.4,.7],[0.,0.,.1,2.6]])
b=np.array([1.,-.3,.7,.4]); m=3; k=1""",'call':'truncated_arnoldi_basis(A,b,m,k)','gold_call':'_oracle_truncated_arnoldi_basis(A,b,m,k)'},
        {'setup':"""import numpy as np
A=np.diag([1.,2.,4.]); b=np.array([1.,1.,1.]); m=1; k=1""",'call':'truncated_arnoldi_basis(A,b,m,k)','gold_call':'_oracle_truncated_arnoldi_basis(A,b,m,k)'},
        {'setup':"""import numpy as np
A=np.array([[1.,.7,.2,0.],[.05,1.1,.6,.1],[0,.05,1.2,.5],[0,0,.05,1.3]])
b=np.array([1.,.2,-.5,.7]); m=4; k=2""",'call':'truncated_arnoldi_basis(A,b,m,k)','gold_call':'_oracle_truncated_arnoldi_basis(A,b,m,k)'},
        {'setup':"""import numpy as np
A=np.eye(3); b=np.zeros(3); m=2; k=1
def run_model():
    try: truncated_arnoldi_basis(A,b,m,k); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_truncated_arnoldi_basis(A,b,m,k); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.ones((2,3)); b=np.ones(2); m=2; k=1
def run_model():
    try: truncated_arnoldi_basis(A,b,m,k); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_truncated_arnoldi_basis(A,b,m,k); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.eye(3); b=np.array([1.,0.,0.]); m=2; k=1
def run_model():
    try: truncated_arnoldi_basis(A,b,m,k); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_truncated_arnoldi_basis(A,b,m,k); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
