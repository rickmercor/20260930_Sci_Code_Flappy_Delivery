"""
Implement effective_jet. The effective Hamiltonian incorporates external-state perturbative contributions. Differentiation must include both coupling products and changing reference denominators before symmetrization.

The effective Hamiltonian incorporates external-state perturbative contributions. Differentiation must include both coupling products and changing reference denominators before symmetrization.

Returns
-------
np.ndarray of shape (4,p,p), the symmetrized effective-Hamiltonian jet.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_jet(H: np.ndarray, e: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Construct the symmetrized second-order effective-Hamiltonian jet.

    Parameters
    ----------
    H : np.ndarray
        Finite real (4,n,n) jet, n>=1, each slot symmetric to atol=1e-12.
    e : np.ndarray
        Finite real reference-energy jet (4,n).
    P : np.ndarray
        Nonempty, strictly increasing integer index array in [0,n).
        Q is its complement. Slots are (value,a,b,ab), actual derivatives.
        For i,j in P the matrix is H_ij plus one half the sum over q in Q
        of H_iq H_qj [(e_i-e_q)^(-1)+(e_j-e_q)^(-1)]. Differentiate this
        expression through mixed order with P fixed. A full P gives H.

    Returns
    -------
    result : np.ndarray
        Shape (4,len(P),len(P)), the effective matrix and its derivatives.

    Raises
    ------
    ValueError
        If arrays cannot be converted to finite real arrays, shapes or P
        indices are invalid, any H slot fails symmetry at atol=1e-12,
        any baseline P-Q reference-energy separation is <=1e-10 in absolute
        value, or the calculated result is nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_effective_jet(H: np.ndarray, e: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Construct the symmetrized second-order effective-Hamiltonian jet.

    Parameters
    ----------
    H : np.ndarray
        Finite real (4,n,n) jet, n>=1, each slot symmetric to atol=1e-12.
    e : np.ndarray
        Finite real reference-energy jet (4,n).
    P : np.ndarray
        Nonempty, strictly increasing integer index array in [0,n).
        Q is its complement. Slots are (value,a,b,ab), actual derivatives.
        For i,j in P the matrix is H_ij plus one half the sum over q in Q
        of H_iq H_qj [(e_i-e_q)^(-1)+(e_j-e_q)^(-1)]. Differentiate this
        expression through mixed order with P fixed. A full P gives H.

    Returns
    -------
    result : np.ndarray
        Shape (4,len(P),len(P)), the effective matrix and its derivatives.

    Raises
    ------
    ValueError
        If arrays cannot be converted to finite real arrays, shapes or P
        indices are invalid, any H slot fails symmetry at atol=1e-12,
        any baseline P-Q reference-energy separation is <=1e-10 in absolute
        value, or the calculated result is nonfinite.
    """
    try:
        if any(np.iscomplexobj(x) for x in (H,e,P)):
            raise ValueError('complex input')
        H, e, P = np.asarray(H,float), np.asarray(e,float), np.asarray(P)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid arrays') from exc
    if H.ndim != 3 or H.shape[0] != 4 or H.shape[1] < 1 or H.shape[1] != H.shape[2]:
        raise ValueError('invalid H shape')
    n = H.shape[1]
    if e.shape != (4,n) or not np.isfinite(H).all() or not np.isfinite(e).all():
        raise ValueError('invalid energy jet')
    if not np.allclose(H,H.transpose(0,2,1),atol=1e-12,rtol=0):
        raise ValueError('asymmetric jet')
    if P.ndim != 1 or P.size == 0 or P.dtype.kind not in 'iu' or np.any(P<0) or np.any(P>=n) or np.any(np.diff(P.astype(int))<=0):
        raise ValueError('invalid P')
    Q = [q for q in range(n) if q not in P]
    # Product tensor for actual value/a/b/ab derivatives.
    C = np.zeros((4,4,4))
    for r,s,t in [(0,0,0),(1,1,0),(1,0,1),(2,2,0),(2,0,2),
                  (3,3,0),(3,1,2),(3,2,1),(3,0,3)]:
        C[r,s,t] = 1
    result = H[:,P,:][:,:,P].copy()
    for ii,i in enumerate(P):
        for jj,j in enumerate(P):
            for q in Q:
                d = e[:,j] - e[:,q]
                if abs(d[0]) <= 1e-10:
                    raise ValueError('singular reference denominator')
                reciprocal = np.array([1/d[0], -d[1]/d[0]**2,
                    -d[2]/d[0]**2, 2*d[1]*d[2]/d[0]**3-d[3]/d[0]**2])
                product = np.einsum('rst,s,t->r', C, H[:,i,q], H[:,q,j])
                result[:,ii,jj] += np.einsum('rst,s,t->r', C, product, reciprocal)
    result = (result + result.transpose(0,2,1))/2
    if not np.isfinite(result).all():
        raise ValueError('nonfinite effective jet')
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
H = np.zeros((4, 2, 2))
H[0] = [[0, 0.3], [0.3, 2]]
H[1] = [[0.1, 0.2], [0.2, -0.1]]
H[2] = [[0, -0.1], [-0.1, 0.4]]
H[3] = [[0.02, 0.03], [0.03, 0.01]]
e = np.array([[0, 2], [0.1, -0.1], [0, 0.4], [0.02, 0.01]])
P = np.array([0])


def run(fn):
    return fn(H.copy(), e.copy(), P.copy())
""",
            "call": 'run(effective_jet)',
            "gold_call": 'run(_oracle_effective_jet)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 2, 2))
H[0] = [[0, 0.3], [0.3, 2]]
H[3] = np.eye(2)
e = np.array([[0, 2], [0, 0], [0, 0], [1, 1]], float)
P = np.array([0, 1])


def run(fn):
    return fn(H.copy(), e.copy(), P.copy())
""",
            "call": 'run(effective_jet)',
            "gold_call": 'run(_oracle_effective_jet)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 3, 3))
H[0] = np.diag([0, 1, 3])
H[1, 1, 2] = H[1, 2, 1] = 0.2
H[2, 1, 2] = H[2, 2, 1] = 0.3
e = np.zeros((4, 3))
e[0] = [0, 1, 3]
P = np.array([2])


def run(fn):
    return fn(H.copy(), e.copy(), P.copy())
""",
            "call": 'run(effective_jet)',
            "gold_call": 'run(_oracle_effective_jet)',
        },
        {
            "setup": """import numpy as np
H = np.zeros((4, 2, 2))
e = np.zeros((4, 2))
P = np.array([0])


def run(fn):
    try:
        fn(H.copy(), e.copy(), P.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": 'run(effective_jet)',
            "gold_call": 'run(_oracle_effective_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3703)
X = rng.normal(size=(4, 5, 5))
H = 0.08 * (X + X.transpose(0, 2, 1))
H[0] += np.diag([-3.0, -0.4, 1.2, 3.4, 5.0])
e = np.zeros((4, 5))
e[0] = [-3.2, -0.2, 1.5, 3.0, 5.4]
e[1:] = 0.12 * rng.normal(size=(3, 5))
P = np.array([0, 2, 4])


def run(fn):
    return fn(H.copy(), e.copy(), P.copy())
""",
            "call": 'run(effective_jet)',
            "gold_call": 'run(_oracle_effective_jet)',
        },
        {
            "setup": """import numpy as np
rng = np.random.default_rng(3704)
X = rng.normal(size=(4, 4, 4))
H = 0.08 * (X + X.transpose(0, 2, 1))
H[0] += np.diag([-1.0, 0.0, 2.0, 4.0])
e = np.zeros((4, 4))
e[0] = [-1.2, 0.02, 0.11, 4.2]
e[1:] = 0.12 * rng.normal(size=(3, 4))
P = np.array([0, 2])


def run(fn):
    return fn(H.copy(), e.copy(), P.copy())
""",
            "call": 'run(effective_jet)',
            "gold_call": 'run(_oracle_effective_jet)',
        },
    ]
