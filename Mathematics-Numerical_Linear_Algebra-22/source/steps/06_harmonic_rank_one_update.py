"""
Repair a truncated Krylov decomposition by a rank-one modification of its projected matrix, enforcing the harmonic orthogonality condition.

The projected evaluation of a matrix function from a Krylov decomposition is justified only when the trailing basis vector satisfies an orthogonality condition, which a truncated recurrence does not provide. Decompose the trailing vector into a component in the span of the approximation basis and a remainder satisfying the required condition, then absorb the first component into the projected matrix.

Here the condition to enforce is that the corrected trailing vector be orthogonal to the image of the approximation basis under A. Determine the coefficient vector of the component of the trailing vector in the basis span that achieves this, and form the corrected projected matrix by adding to the leading m by m block of the recurrence matrix a rank-one term built from that coefficient vector scaled by the trailing recurrence coefficient, acting only on the last column.

The coefficient vector is the solution of the resulting square linear system of order m, computed by direct LU factorization with partial pivoting.

The inputs are the matrix A and the packed decomposition produced by the previous step. Return the corrected projected matrix as a square array of order m.

The function raises ValueError if A is not a square two-dimensional finite array, if the packed array is not two-dimensional with at least two columns and a row count exceeding its column count by at least one, if the leading block width does not match the order of A, if the packed array is not finite, or if the square system determining the coefficient vector is singular.

Several projection methods for matrix functions share the same closed form: the function of a small projected matrix, applied to the first coordinate vector and lifted back by the basis. Which projection a given decomposition realizes is decided by a Petrov-Galerkin condition on the residual of the associated shifted linear systems, and each condition corresponds to a different orthogonality requirement on the vector that terminates the recurrence. Requiring the residual to be orthogonal to the Krylov subspace itself gives one method; requiring it to be orthogonal to the image of that subspace under the matrix gives the harmonic variant.



The two are not interchangeable. Their projected matrices differ, their eigenvalues differ, and the convergence guarantees attached to them hold under different hypotheses: the plain condition is analyzed for Hermitian positive definite arguments, while the harmonic condition is analyzed for the broader class of matrices with numerical range in the open right half-plane; both analyses concern functions of Stieltjes type. The harmonic eigenvalues also play a direct role in the error analysis, since the residual of the associated shifted systems is proportional to a product of factors indexed by them, and that product is bounded by one when they lie in the open right half-plane.



Because a truncated recurrence satisfies neither condition, the decomposition must be modified after the fact. The modification is algebraically exact rather than approximate: it changes which vector terminates the recurrence and compensates in the projected matrix, leaving the identity relating the matrix, the basis and the projected matrix intact.

Returns
-------
np.ndarray, the corrected projected matrix as a float64 array of shape (m, m)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def harmonic_rank_one_update(A: np.ndarray, packed: np.ndarray) -> np.ndarray:
    '''Apply the harmonic rank-one correction to a truncated Krylov decomposition.

    Parameters
    ----------
    A : np.ndarray
        Square finite matrix of shape (N, N).
    packed : np.ndarray
        Packed decomposition of shape (N + m + 1, m + 1): its first N rows hold
        the m+1 basis vectors as columns, and its remaining m+1 rows hold the
        (m+1, m) recurrence matrix in their first m columns.

    Returns
    -------
    h_tilde : np.ndarray
        Corrected projected matrix, a float array of shape (m, m).

    Raises
    ------
    ValueError
        If A is not a square two-dimensional finite array, if the packed array is
        not two-dimensional with at least two columns and a row count exceeding its
        column count by at least one, if the leading block width does not match the
        order of A, if the packed array is not finite, or if the square system
        determining the coefficient vector is singular.
    '''
    return h_tilde  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_harmonic_rank_one_update(A: np.ndarray, packed: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    A = np.asarray(A, dtype=float)
    packed = np.asarray(packed, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square two-dimensional array")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    if packed.ndim != 2 or packed.shape[1] < 2:
        raise ValueError("packed must be two-dimensional with at least two columns")
    if not np.all(np.isfinite(packed)):
        raise ValueError("packed must be finite")

    m = packed.shape[1] - 1
    ndim = packed.shape[0] - m - 1
    if ndim < 1:
        raise ValueError("packed row count is inconsistent with its column count")
    if ndim != A.shape[0]:
        raise ValueError("packed leading block width does not match the order of A")

    basis = packed[:ndim, :m]
    trailing = packed[:ndim, m]
    hess = packed[ndim:, :m]

    image = A @ basis
    gram = image.T @ basis
    rhs = image.T @ trailing
    if not np.all(np.isfinite(gram)) or np.linalg.matrix_rank(gram) < m:
        raise ValueError("the square system determining the coefficient vector is singular")
    try:
        c_m = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the square system determining the coefficient vector is singular") from exc

    h_tilde = hess[:m, :m].copy()
    h_tilde[:, m - 1] = h_tilde[:, m - 1] + c_m * hess[m, m - 1]
    return h_tilde

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: production decomposition at the production threshold ---
        {
            "setup": """import numpy as np
def _ta(A, b, tr, tau):
    N = A.shape[0]; be = np.linalg.norm(b); B = [b/be]; H = np.zeros((N+1, N))
    for j in range(N):
        w = A @ B[j]
        for i in range(max(0, j-tr+1), j+1):
            H[i, j] = B[i] @ w; w = w - H[i, j]*B[i]
        H[j+1, j] = np.linalg.norm(w); B.append(w/H[j+1, j])
        if np.linalg.cond(np.array(B).T) > tau:
            m = j+1; Bm = np.array(B).T
            P = np.zeros((N+m+1, m+1)); P[:N, :] = Bm; P[N:, :m] = H[:m+1, :m]
            return P
    raise ValueError
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
packed = _ta(A, b, 1, 1.0e4)
""",
            "call": "harmonic_rank_one_update(A, packed)",
            "gold_call": "_oracle_harmonic_rank_one_update(A, packed)",
            "tol": 1e-6,
        },
        # --- normal: production matrix, looser threshold and larger subspace ---
        {
            "setup": """import numpy as np
def _ta(A, b, tr, tau):
    N = A.shape[0]; be = np.linalg.norm(b); B = [b/be]; H = np.zeros((N+1, N))
    for j in range(N):
        w = A @ B[j]
        for i in range(max(0, j-tr+1), j+1):
            H[i, j] = B[i] @ w; w = w - H[i, j]*B[i]
        H[j+1, j] = np.linalg.norm(w); B.append(w/H[j+1, j])
        if np.linalg.cond(np.array(B).T) > tau:
            m = j+1; Bm = np.array(B).T
            P = np.zeros((N+m+1, m+1)); P[:N, :] = Bm; P[N:, :m] = H[:m+1, :m]
            return P
    raise ValueError
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
packed = _ta(A, b, 1, 2.0e4)
""",
            "call": "harmonic_rank_one_update(A, packed)",
            "gold_call": "_oracle_harmonic_rank_one_update(A, packed)",
            "tol": 1e-4,
        },
        # --- normal: small nonsymmetric matrix ---
        {
            "setup": """import numpy as np
def _ta(A, b, tr, tau):
    N = A.shape[0]; be = np.linalg.norm(b); B = [b/be]; H = np.zeros((N+1, N))
    for j in range(N):
        w = A @ B[j]
        for i in range(max(0, j-tr+1), j+1):
            H[i, j] = B[i] @ w; w = w - H[i, j]*B[i]
        H[j+1, j] = np.linalg.norm(w); B.append(w/H[j+1, j])
        if np.linalg.cond(np.array(B).T) > tau:
            m = j+1; Bm = np.array(B).T
            P = np.zeros((N+m+1, m+1)); P[:N, :] = Bm; P[N:, :m] = H[:m+1, :m]
            return P
    raise ValueError
A = (np.diag(np.arange(1.0, 9.0)) + np.diag(0.5*np.ones(7), 1)
     + np.diag(0.25*np.ones(7), -1))
b = np.arange(1.0, 9.0)
b = b/np.linalg.norm(b)
packed = _ta(A, b, 1, 1.0e2)
""",
            "call": "harmonic_rank_one_update(A, packed)",
            "gold_call": "_oracle_harmonic_rank_one_update(A, packed)",
        },
        # --- boundary: very small subspace ---
        {
            "setup": """import numpy as np
def _ta(A, b, tr, tau):
    N = A.shape[0]; be = np.linalg.norm(b); B = [b/be]; H = np.zeros((N+1, N))
    for j in range(N):
        w = A @ B[j]
        for i in range(max(0, j-tr+1), j+1):
            H[i, j] = B[i] @ w; w = w - H[i, j]*B[i]
        H[j+1, j] = np.linalg.norm(w); B.append(w/H[j+1, j])
        if np.linalg.cond(np.array(B).T) > tau:
            m = j+1; Bm = np.array(B).T
            P = np.zeros((N+m+1, m+1)); P[:N, :] = Bm; P[N:, :m] = H[:m+1, :m]
            return P
    raise ValueError
A = (np.diag(np.arange(1.0, 9.0)) + np.diag(0.5*np.ones(7), 1)
     + np.diag(0.25*np.ones(7), -1))
b = np.arange(1.0, 9.0)
b = b/np.linalg.norm(b)
packed = _ta(A, b, 1, 5.0)
""",
            "call": "harmonic_rank_one_update(A, packed)",
            "gold_call": "_oracle_harmonic_rank_one_update(A, packed)",
        },
        # --- structural: correction confined to the last column, spectrum in the right half-plane ---
        {
            "setup": """import numpy as np
def _ta(A, b, tr, tau):
    N = A.shape[0]; be = np.linalg.norm(b); B = [b/be]; H = np.zeros((N+1, N))
    for j in range(N):
        w = A @ B[j]
        for i in range(max(0, j-tr+1), j+1):
            H[i, j] = B[i] @ w; w = w - H[i, j]*B[i]
        H[j+1, j] = np.linalg.norm(w); B.append(w/H[j+1, j])
        if np.linalg.cond(np.array(B).T) > tau:
            m = j+1; Bm = np.array(B).T
            P = np.zeros((N+m+1, m+1)); P[:N, :] = Bm; P[N:, :m] = H[:m+1, :m]
            return P
    raise ValueError
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
packed = _ta(A, b, 1, 1.0e4)
def probe(fn):
    Ht = fn(A, packed)
    m = packed.shape[1] - 1
    N = packed.shape[0] - m - 1
    H0 = packed[N:, :m][:m, :m]
    ev = np.linalg.eigvals(Ht)
    return [int(Ht.shape[0]), int(Ht.shape[1]),
            int(np.allclose(Ht[:, :m-1], H0[:, :m-1], rtol=0.0, atol=0.0)),
            int(np.all(ev.real > 0.0)),
            int(np.max(np.abs(ev.imag)) == 0.0)]
""",
            "call": "probe(harmonic_rank_one_update)",
            "gold_call": "probe(_oracle_harmonic_rank_one_update)",
        },
        # --- invalid: packed leading block inconsistent with A ---
        {
            "setup": """import numpy as np
A = np.eye(5)*3.0
packed = np.ones((10, 4))
def run_model():
    try:
        harmonic_rank_one_update(A, packed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_harmonic_rank_one_update(A, packed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-square matrix ---
        {
            "setup": """import numpy as np
A = np.ones((3, 4))
packed = np.ones((7, 3))
def run_model():
    try:
        harmonic_rank_one_update(A, packed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_harmonic_rank_one_update(A, packed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: singular coefficient system ---
        {
            "setup": """import numpy as np
A = np.zeros((4, 4))
packed = np.zeros((7, 3))
packed[:4, 0] = np.array([1.0, 0.0, 0.0, 0.0])
packed[:4, 1] = np.array([0.0, 1.0, 0.0, 0.0])
packed[:4, 2] = np.array([0.0, 0.0, 1.0, 0.0])
packed[4:, :2] = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
def run_model():
    try:
        harmonic_rank_one_update(A, packed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_harmonic_rank_one_update(A, packed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: packed array with a single column ---
        {
            "setup": """import numpy as np
A = np.eye(3)*2.0
packed = np.ones((5, 1))
def run_model():
    try:
        harmonic_rank_one_update(A, packed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_harmonic_rank_one_update(A, packed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-finite entry in the packed array ---
        {
            "setup": """import numpy as np
A = np.eye(4)*2.0
packed = np.ones((7, 3))
packed[0, 0] = np.nan
def run_model():
    try:
        harmonic_rank_one_update(A, packed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_harmonic_rank_one_update(A, packed)
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
