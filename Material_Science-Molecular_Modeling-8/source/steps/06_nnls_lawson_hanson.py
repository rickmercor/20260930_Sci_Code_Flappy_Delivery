"""
Solve the nonnegative least-squares problem A y = b, y >= 0 with the source's active-set algorithm under the declared deterministic conventions: dual tolerance 1e-10 on the stopping test, the smallest index among gradient maximizers, least-squares subproblems by QR factorization, the source's step-length rule and passive-set drop rule, and a 200-iteration guard raising ValueError.

The source solves its scaled system with the classical active-set nonnegative least-squares algorithm; the structure of the outer and inner loops is the source's, and the declared tolerances pin the arithmetic deterministically.

Returns
-------
return (N,) float64: nonnegative least-squares solution
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def nnls_lawson_hanson(A, b):
    """A: (m, N); b: (m,). Returns (N,) float64: the nonnegative
    least-squares solution computed by the source's active-set algorithm
    under the declared conventions (dual tolerance 1e-10, smallest index
    among gradient maximizers, QR-based least-squares subproblems, the
    source's step-length and drop rules, 200-iteration guard).
    Raises ValueError if an iteration limit is exceeded."""
    return np.zeros(np.asarray(A).shape[1])

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 6: active-set nonnegative least squares with declared conventions."""

import numpy as np


def _ls_solve(A, b):
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)


def _oracle_nnls_lawson_hanson(A, b):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = A.shape[1]
    P = []
    Z = list(range(n))
    w = np.zeros(n)
    it = 0
    while True:
        grad = -(A.T @ (A @ w - b))
        if not Z:
            break
        gz = np.full(n, -np.inf)
        gz[Z] = grad[Z]
        if np.max(gz) <= 1e-10:
            break
        it += 1
        if it > 200:
            raise ValueError("NNLS iteration limit exceeded")
        tau = int(np.argmax(gz))
        Z.remove(tau)
        P.append(tau)
        z = np.zeros(n)
        z[P] = _ls_solve(A[:, P], b)
        inner = 0
        while P and np.min(z[P]) <= 0:
            inner += 1
            if inner > 200:
                raise ValueError("NNLS inner iteration limit exceeded")
            Qset = [i for i in P if z[i] <= 0]
            alpha = min(w[i] / (w[i] - z[i]) for i in Qset)
            w = w + alpha * (z - w)
            drop = [i for i in P if w[i] <= 0]
            for i in drop:
                P.remove(i)
                Z.append(i)
            Z.sort()
            z = np.zeros(n)
            if P:
                z[P] = _ls_solve(A[:, P], b)
        w = z.copy()
    return w

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+3)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+3)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*3+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+3)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*3)%31)/25.0-0.6) for c in range(3)],axis=1)\nM=scaled_moment_system(w, v, x)\nR=rate_rows(w, v, w2, v2)\nA0=_n.vstack([M[:, :24], R[:, :24]])\nb=_n.concatenate([M[:, 24], R[:, 24]])\ns=column_scaling(A0)\nA=A0*s[None, :]', "call": "nnls_lawson_hanson(A, b)", "gold_call": "_oracle_nnls_lawson_hanson(A, b)", "tol": 1e-08},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+5)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+5)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*5+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+5)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*5)%31)/25.0-0.6) for c in range(3)],axis=1)\nM=scaled_moment_system(w, v, x)\nR=rate_rows(w, v, w2, v2)\nA0=_n.vstack([M[:, :24], R[:, :24]])\nb=_n.concatenate([M[:, 24], R[:, 24]])\ns=column_scaling(A0)\nA=A0*s[None, :]', "call": "nnls_lawson_hanson(A, b)", "gold_call": "_oracle_nnls_lawson_hanson(A, b)", "tol": 1e-08},
        {"setup": 'import numpy as _n\nI=_n.arange(24)\nw=0.5+(((I+1)*(I+3)+4)%7)/10.0\nv=_n.stack([((((I+1)*(I+2)+(c+2)*(I+5)+4)%47)/23.5-1.0) for c in range(3)],axis=1)\nx=(((I+2)*(I+4)*4+1)%41)/41.0\nK=_n.arange(10)\nw2=0.8+(((K+1)*(K+2)+4)%5)/10.0\nv2=_n.stack([((((K+2)*(K+3)+(c+3)*(K+1)+2*4)%31)/25.0-0.6) for c in range(3)],axis=1)\nM=scaled_moment_system(w, v, x)\nA0=M[:, :24]\nb=M[:, 24]\ns=column_scaling(A0)\nA=A0*s[None, :]', "call": "nnls_lawson_hanson(A, b)", "gold_call": "_oracle_nnls_lawson_hanson(A, b)", "tol": 1e-08},
        {"setup": 'import numpy as _n\nA=_n.array([[1.0],[2.0],[0.5]])\nb=_n.array([1.0,2.2,0.4])', "call": "nnls_lawson_hanson(A, b)", "gold_call": "_oracle_nnls_lawson_hanson(A, b)", "tol": 1e-08},
        {"setup": 'import numpy as _n\nA=_n.array([[1.0,0.5],[0.2,1.0],[0.3,0.4]])\nb=_n.array([-1.0,-0.5,-0.7])', "call": "nnls_lawson_hanson(A, b)", "gold_call": "_oracle_nnls_lawson_hanson(A, b)", "tol": 1e-08},
    ]
