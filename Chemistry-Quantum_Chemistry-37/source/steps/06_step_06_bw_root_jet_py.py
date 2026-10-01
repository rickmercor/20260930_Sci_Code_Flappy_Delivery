"""
Implement bw_root_jet. The second-order Brillouin-Wigner energy is an implicit root. Its response includes the changing poles, couplings and diagonal matrix element; vanishing baseline coupling can still have a nonzero mixed contribution.

The second-order Brillouin-Wigner energy is an implicit root. Its response includes the changing poles, couplings and diagonal matrix element; vanishing baseline coupling can still have a nonzero mixed contribution.

Returns
-------
np.ndarray of shape (4,), the selected energy root and its derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bw_root_jet(H: np.ndarray, e: np.ndarray, k: int) -> np.ndarray:
    """Solve and analytically differentiate the selected second-order BW root.

    Parameters
    ----------
    H, e : np.ndarray
        Finite real jets (4,n,n) and (4,n), n>=1; each H slot symmetric
        to atol=1e-12. Slots contain actual derivatives (value,a,b,ab).
    k : int
        Integer index 0<=k<n, not bool. Solve
        F(z)=z-H_kk-sum_{j!=k} H_kj**2/(z-e_j)=0 in the open pole interval
        containing e_k. The interval may be unbounded; all e_j, j!=k,
        are treated as denominator poles even for zero coupling.
        Include derivatives of H, e and the implicit root. No finite
        differences. With n=1, return H[:,0,0].

    Returns
    -------
    result : np.ndarray
        Four-vector (root, root_a, root_b, root_ab).

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes and symmetry, k is invalid, e_k lies within 1e-10 of another
        reference energy, the selected interval does not contain exactly
        one real root separated from both finite bounds by >1e-10,
        the baseline eigensolver fails, or the output is nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bw_root_jet(H: np.ndarray, e: np.ndarray, k: int) -> np.ndarray:
    """Solve and analytically differentiate the selected second-order BW root.

    Parameters
    ----------
    H, e : np.ndarray
        Finite real jets (4,n,n) and (4,n), n>=1; each H slot symmetric
        to atol=1e-12. Slots contain actual derivatives (value,a,b,ab).
    k : int
        Integer index 0<=k<n, not bool. Solve
        F(z)=z-H_kk-sum_{j!=k} H_kj**2/(z-e_j)=0 in the open pole interval
        containing e_k. The interval may be unbounded; all e_j, j!=k,
        are treated as denominator poles even for zero coupling.
        Include derivatives of H, e and the implicit root. No finite
        differences. With n=1, return H[:,0,0].

    Returns
    -------
    result : np.ndarray
        Four-vector (root, root_a, root_b, root_ab).

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real arrays of the stated
        shapes and symmetry, k is invalid, e_k lies within 1e-10 of another
        reference energy, the selected interval does not contain exactly
        one real root separated from both finite bounds by >1e-10,
        the baseline eigensolver fails, or the output is nonfinite.
    """
    try:
        if np.iscomplexobj(H) or np.iscomplexobj(e):
            raise ValueError('complex input')
        H,e = np.asarray(H,float),np.asarray(e,float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid arrays') from exc
    if H.ndim != 3 or H.shape[0] != 4 or H.shape[1] < 1 or H.shape[1] != H.shape[2]:
        raise ValueError('invalid H shape')
    n = H.shape[1]
    if e.shape != (4,n) or not np.isfinite(H).all() or not np.isfinite(e).all() or not np.allclose(H,H.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('invalid or asymmetric jet')
    if isinstance(k,(bool,np.bool_)) or not isinstance(k,(int,np.integer)) or not 0<=k<n:
        raise ValueError('invalid candidate')
    J = [j for j in range(n) if j != k]
    poles = e[:,J]
    if np.any(np.abs(e[0,k]-poles[0])<=1e-10):
        raise ValueError('reference at a pole')
    lo = max((p for p in poles[0] if p<e[0,k]),default=-np.inf)
    hi = min((p for p in poles[0] if p>e[0,k]),default=np.inf)
    arrow = np.diag(np.r_[H[0,k,k],poles[0]])
    arrow[0,1:] = H[0,k,J]; arrow[1:,0] = H[0,k,J]
    try:
        roots = np.linalg.eigvalsh(arrow)
    except np.linalg.LinAlgError as exc:
        raise ValueError('root eigensolver failed') from exc
    candidates = roots[(roots>lo+1e-10)&(roots<hi-1e-10)]
    if len(candidates) != 1:
        raise ValueError('no unique interior root')
    C = np.zeros((4,4,4))
    for r,s,t in [(0,0,0),(1,1,0),(1,0,1),(2,2,0),(2,0,2),
                  (3,3,0),(3,1,2),(3,2,1),(3,0,3)]:
        C[r,s,t] = 1
    w = H[:,k,J]
    c = np.einsum('rst,sj,tj->rj',C,w,w)
    result = np.array([candidates[0],0.,0.,0.])
    fz = 1+np.sum(c[0]/(result[0]-poles[0])**2)
    # First evaluate partial a,b terms, then the total mixed residual
    # with the known root_a/root_b but root_ab set to zero.
    for stage in range(2):
        d = result[:,None]-poles
        reciprocal = np.array([1/d[0],-d[1]/d[0]**2,-d[2]/d[0]**2,
                                2*d[1]*d[2]/d[0]**3-d[3]/d[0]**2])
        residual = result-H[:,k,k]-np.einsum('rst,sj,tj->rj',C,c,reciprocal).sum(axis=1)
        if stage == 0:
            result[1:3] = -residual[1:3]/fz
        else:
            result[3] = -residual[3]/fz
    if not np.isfinite(result).all():
        raise ValueError('nonfinite root jet')
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
H = np.array([[[0.2, 0.3], [0.3, 2]], [[0.1, 0.2], [0.2, -0.1]], [[0, 0.1], [0.1, 0.4]], [[0.03, -0.02], [-0.02, 0.01]]])
e = np.array([[0, 2], [0, -0.1], [0, 0.4], [0, 0.01]], float)
k = 0


def run(fn):
    return fn(H.copy(), e.copy(), k)
""",
            "call": 'run(bw_root_jet)',
            "gold_call": 'run(_oracle_bw_root_jet)',
        },
        {
            "setup": """import numpy as np
H = np.array([2.0, 0.3, -0.2, 0.5]).reshape(4, 1, 1)
e = H[:, :, 0].copy()
k = 0


def run(fn):
    return fn(H.copy(), e.copy(), k)
""",
            "call": 'run(bw_root_jet)',
            "gold_call": 'run(_oracle_bw_root_jet)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 2, 2))
H[0] = np.diag([0.2, 2])
H[1, 0, 1] = H[1, 1, 0] = 0.3
H[2, 0, 1] = H[2, 1, 0] = -0.4
e = np.zeros((4, 2))
e[0] = [0, 2]
k = 0


def run(fn):
    return fn(H.copy(), e.copy(), k)
""",
            "call": 'run(bw_root_jet)',
            "gold_call": 'run(_oracle_bw_root_jet)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 2, 2))
H[0] = [[0, 0.2], [0.2, 0]]
e = np.zeros((4, 2))
k = 0


def run(fn):
    try:
        fn(H.copy(), e.copy(), k)
    except ValueError:
        return 1
    return 0
""",
            "call": 'run(bw_root_jet)',
            "gold_call": 'run(_oracle_bw_root_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3709)
X = rng.normal(size=(4, 5, 5))
H = 0.07 * (X + X.transpose(0, 2, 1))
e = 0.09 * rng.normal(size=(4, 5))
e[0] = [-3.0, -1.0, 0.2, 1.7, 4.0]
k = 2
H[0, k, k] = e[0, k] + 0.1
for j in range(5):
    if j != k:
        H[0, k, j] = H[0, j, k] = 0.16 + 0.03 * j


def run(fn):
    return fn(H.copy(), e.copy(), k)
""",
            "call": 'run(bw_root_jet)',
            "gold_call": 'run(_oracle_bw_root_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3710)
X = rng.normal(size=(4, 5, 5))
H = 0.07 * (X + X.transpose(0, 2, 1))
e = 0.09 * rng.normal(size=(4, 5))
e[0] = [-2.0, -2.0, 0.1, 1.5, 3.0]
k = 2
H[0, k, k] = e[0, k] + 0.1
for j in range(5):
    if j != k:
        H[0, k, j] = H[0, j, k] = 0.16 + 0.03 * j


def run(fn):
    return fn(H.copy(), e.copy(), k)
""",
            "call": 'run(bw_root_jet)',
            "gold_call": 'run(_oracle_bw_root_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3711)
X = rng.normal(size=(4, 5, 5))
H = 0.07 * (X + X.transpose(0, 2, 1))
e = 0.09 * rng.normal(size=(4, 5))
e[0] = [-3.0, -1.0, 1.0, 3.0, 5.0]
k = 0
H[0, k, k] = e[0, k] + 0.1
for j in range(5):
    if j != k:
        H[0, k, j] = H[0, j, k] = 0.16 + 0.03 * j


def run(fn):
    return fn(H.copy(), e.copy(), k)
""",
            "call": 'run(bw_root_jet)',
            "gold_call": 'run(_oracle_bw_root_jet)',
        },
        {
            "setup": """import numpy as np
e = np.array([[-2.0, 0.0, 1.5, 4.0], [0.1, 0.4, -0.2, 0.3], [-0.3, 0.2, 0.1, -0.4], [0.2, -0.1, 0.3, 0.4]])
k = 1
H = np.zeros((4, 4, 4))
H[:, 1, 1] = [0.2, 0.1, -0.2, 0.3]
for j, jet in [(0, [0.25, 0.1, -0.1, 0.07]), (2, [0.0, 0.3, -0.4, 0.1]), (3, [-0.2, -0.05, 0.2, -0.1])]:
    H[:, 1, j] = H[:, j, 1] = jet


def run(fn):
    return fn(H.copy(), e.copy(), 1)
""",
            "call": 'run(bw_root_jet)',
            "gold_call": 'run(_oracle_bw_root_jet)',
        },
    ]
