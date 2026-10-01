"""
Build a Krylov decomposition of a matrix by a truncated orthogonalization recurrence, stopping adaptively on the conditioning of the generated basis.

Starting from the vector b divided by its Euclidean norm, generate successive vectors by applying A to the most recent basis vector and subtracting its components along the trunc most recently generated basis vectors, one at a time, each subtraction applied before the next coefficient is computed. The coefficient used at each subtraction, together with the norm of the remainder, populates the rectangular upper Hessenberg recurrence matrix H of shape (j+1, j) satisfying that A applied to the first j basis vectors equals the first j+1 basis vectors multiplied by H.

After each new vector is appended, form the 2-norm condition number of the basis assembled so far, including that newest vector. Continue while it remains at or below the threshold; the first time it exceeds the threshold, stop and take the subspace dimension m to be the number of vectors preceding the newest one.

Return a single packed array of shape (N + m + 1, m + 1), where N is the dimension of A. Its first N rows hold the m + 1 basis vectors as columns, the first m of which are the approximation basis and the last of which is the trailing vector. Its remaining m + 1 rows hold the recurrence matrix of shape (m + 1, m) in their first m columns, with the final column of that block set to zero.

The function raises ValueError if A is not a square two-dimensional finite array, if b is not a finite one-dimensional array of matching length with nonzero norm, if trunc is not a positive integer, if tau is not finite and greater than one, if the recurrence breaks down with a zero remainder, or if the threshold is not exceeded before the basis dimension reaches the size of A.

A full orthogonalization recurrence produces an orthonormal Krylov basis, but its cost and storage grow with the subspace dimension because every new vector must be orthogonalized against all its predecessors. Restricting the orthogonalization to a fixed window of recent vectors reduces both to a constant per step, at the cost of losing global orthogonality: components along vectors outside the window are never removed and accumulate, so the generated basis becomes progressively more nearly linearly dependent, in the worst case at a rate exponential in the dimension.

This degradation is the binding constraint on how far the recurrence may usefully be run. It is not detected by the recurrence itself, which continues to produce unit vectors and finite coefficients regardless, so it must be monitored externally through the conditioning of the accumulated basis. Choosing the subspace dimension by that measurement rather than fixing it in advance adapts the work to the problem: matrices and starting vectors that degrade slowly earn larger subspaces, and those that degrade quickly are cut short before the projected quantities become meaningless.

The window size interacts with the symmetry of the matrix. For a symmetric argument, orthogonality against two recent vectors is already sufficient in exact arithmetic, so windows of two or more leave the basis nearly orthonormal for a long time and the point at which conditioning degrades is decided by rounding rather than by the recurrence. A window of one removes that coincidence and produces a basis whose conditioning grows immediately and reproducibly.

Returns
-------
np.ndarray, a float64 array of shape (N + m + 1, m + 1) packing the m+1 basis vectors in its first N rows and the (m+1, m) upper Hessenberg recurrence matrix in its remaining rows, the final column of that lower block being zero
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def truncated_arnoldi_basis(A: np.ndarray, b: np.ndarray, trunc: int,
                            tau: float) -> np.ndarray:
    '''Truncated Krylov decomposition with conditioning-based adaptive stopping.

    Parameters
    ----------
    A : np.ndarray
        Square finite matrix of shape (N, N).
    b : np.ndarray
        Finite starting vector of length N with nonzero norm.
    trunc : int
        Number of most recent basis vectors to orthogonalize against.
        Must be a positive integer.
    tau : float
        Basis condition number threshold. Must be finite and greater than one.

    Returns
    -------
    packed : np.ndarray
        Float array of shape (N + m + 1, m + 1). Rows 0..N-1 hold the m+1 basis
        vectors as columns; rows N..N+m hold the (m+1, m) recurrence matrix in
        their first m columns, with the final column of that block zero.

    Raises
    ------
    ValueError
        If A is not a square two-dimensional finite array, if b is not a finite one-
        dimensional array of matching length with nonzero norm, if trunc is not a
        positive integer, if tau is not finite and greater than one, if the
        recurrence breaks down with a zero remainder, or if the threshold is not
        exceeded before the basis dimension reaches the size of A.
    '''
    return packed  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_truncated_arnoldi_basis(A: np.ndarray, b: np.ndarray, trunc: int,
                                    tau: float) -> np.ndarray:
    """Reference implementation."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1:
        raise ValueError("A must be a square two-dimensional array")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must be finite")
    if b.ndim != 1 or b.size != A.shape[0]:
        raise ValueError("b must be one-dimensional of length matching A")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must be finite")
    if isinstance(trunc, bool) or not isinstance(trunc, (int, np.integer)) or int(trunc) < 1:
        raise ValueError("trunc must be a positive integer")
    trunc = int(trunc)
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 1.0:
        raise ValueError("tau must be finite and greater than one")

    ndim = A.shape[0]
    beta = float(np.linalg.norm(b))
    if beta == 0.0:
        raise ValueError("b must have nonzero norm")

    basis = [b / beta]
    hess = np.zeros((ndim + 1, ndim), dtype=float)

    for j in range(ndim):
        w = A @ basis[j]
        for i in range(max(0, j - trunc + 1), j + 1):
            hess[i, j] = float(basis[i] @ w)
            w = w - hess[i, j] * basis[i]
        nrm = float(np.linalg.norm(w))
        if nrm == 0.0:
            raise ValueError("recurrence broke down with a zero remainder")
        hess[j + 1, j] = nrm
        basis.append(w / nrm)

        bmat = np.array(basis).T
        if np.linalg.cond(bmat) > tau:
            m = j + 1
            packed = np.zeros((ndim + m + 1, m + 1), dtype=float)
            packed[:ndim, :] = bmat
            packed[ndim:, :m] = hess[:m + 1, :m]
            return packed

    raise ValueError("condition threshold not exceeded before basis filled the space")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: production matrix, vector and threshold ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
trunc = 1
tau = 1.0e4
""",
            "call": "truncated_arnoldi_basis(A, b, trunc, tau)",
            "gold_call": "_oracle_truncated_arnoldi_basis(A, b, trunc, tau)",
        },
        # --- normal: production matrix, looser threshold ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
trunc = 1
tau = 1.0e6
""",
            "call": "truncated_arnoldi_basis(A, b, trunc, tau)",
            "gold_call": "_oracle_truncated_arnoldi_basis(A, b, trunc, tau)",
        },
        # --- normal: production matrix, different starting vector ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.sin(0.11*k) + 1.5
b = b/np.linalg.norm(b)
trunc = 1
tau = 1.0e4
""",
            "call": "truncated_arnoldi_basis(A, b, trunc, tau)",
            "gold_call": "_oracle_truncated_arnoldi_basis(A, b, trunc, tau)",
        },
        # --- normal: small nonsymmetric matrix ---
        {
            "setup": """import numpy as np
A = (np.diag(np.arange(1.0, 9.0)) + np.diag(0.5*np.ones(7), 1)
     + np.diag(0.25*np.ones(7), -1))
b = np.arange(1.0, 9.0)
b = b/np.linalg.norm(b)
trunc = 1
tau = 1.0e2
""",
            "call": "truncated_arnoldi_basis(A, b, trunc, tau)",
            "gold_call": "_oracle_truncated_arnoldi_basis(A, b, trunc, tau)",
        },
        # --- boundary: threshold so tight the recurrence stops almost at once ---
        {
            "setup": """import numpy as np
A = (np.diag(np.arange(1.0, 9.0)) + np.diag(0.5*np.ones(7), 1)
     + np.diag(0.25*np.ones(7), -1))
b = np.arange(1.0, 9.0)
b = b/np.linalg.norm(b)
trunc = 1
tau = 5.0
""",
            "call": "truncated_arnoldi_basis(A, b, trunc, tau)",
            "gold_call": "_oracle_truncated_arnoldi_basis(A, b, trunc, tau)",
        },
        # --- boundary: unnormalized starting vector, scaling must not change the basis ---
        {
            "setup": """import numpy as np
A = (np.diag(np.arange(1.0, 9.0)) + np.diag(0.5*np.ones(7), 1)
     + np.diag(0.25*np.ones(7), -1))
b = 37.5*np.arange(1.0, 9.0)
trunc = 1
tau = 1.0e2
""",
            "call": "truncated_arnoldi_basis(A, b, trunc, tau)",
            "gold_call": "_oracle_truncated_arnoldi_basis(A, b, trunc, tau)",
        },
        # --- structural: stopping index brackets the threshold ---
        {
            "setup": """import numpy as np
L1 = np.diag(2.0*np.ones(24)) + np.diag(-np.ones(23), 1) + np.diag(-np.ones(23), -1)
M = np.kron(L1, np.eye(24)) + np.kron(np.eye(24), L1)
k = np.arange(576)
dv = 1.0 + 3.0*np.cos(0.7*k)**2
A = M*np.sqrt(dv)[:, None]*np.sqrt(dv)[None, :]
b = np.cos(0.3*k) + 0.5
b = b/np.linalg.norm(b)
trunc = 1
tau = 1.0e4
def probe(fn):
    P = fn(A, b, trunc, tau)
    m = P.shape[1] - 1
    N = P.shape[0] - m - 1
    Bfull = P[:N, :]
    return [m, N,
            int(np.linalg.cond(Bfull[:, :m]) <= tau),
            int(np.linalg.cond(Bfull) > tau),
            int(np.all(P[N:, m] == 0.0))]
""",
            "call": "probe(truncated_arnoldi_basis)",
            "gold_call": "probe(_oracle_truncated_arnoldi_basis)",
        },
        # --- invalid: threshold never exceeded ---
        {
            "setup": """import numpy as np
A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
b = np.array([1.0, 0.0, 0.0])
trunc = 1
tau = 1.0e3
def run_model():
    try:
        truncated_arnoldi_basis(A, b, trunc, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_truncated_arnoldi_basis(A, b, trunc, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: zero starting vector ---
        {
            "setup": """import numpy as np
A = np.eye(4)*2.0
b = np.zeros(4)
trunc = 1
tau = 1.0e2
def run_model():
    try:
        truncated_arnoldi_basis(A, b, trunc, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_truncated_arnoldi_basis(A, b, trunc, tau)
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
b = np.ones(3)
trunc = 1
tau = 1.0e2
def run_model():
    try:
        truncated_arnoldi_basis(A, b, trunc, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_truncated_arnoldi_basis(A, b, trunc, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: threshold not above one ---
        {
            "setup": """import numpy as np
A = np.eye(4)*2.0 + np.diag(np.ones(3), 1)
b = np.ones(4)
trunc = 1
tau = 1.0
def run_model():
    try:
        truncated_arnoldi_basis(A, b, trunc, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_truncated_arnoldi_basis(A, b, trunc, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: nonpositive truncation window ---
        {
            "setup": """import numpy as np
A = np.eye(4)*2.0 + np.diag(np.ones(3), 1)
b = np.ones(4)
trunc = 0
tau = 1.0e2
def run_model():
    try:
        truncated_arnoldi_basis(A, b, trunc, tau)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_truncated_arnoldi_basis(A, b, trunc, tau)
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
