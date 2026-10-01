"""
Return gauge-independent EVB and order-two valence MEB projectors for a two-orbital impurity.

Bath observables depend on the retained subspace, not signs or rotations of its basis. The MEB powers refer to the full one-spin density. Rank-zero and rank-one cases are valid and must not be padded with unrelated orbitals.

Returns
-------
tuple[np.ndarray, np.ndarray], EVB and MEB environment projectors, each shape (n,n).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bath_projectors(D: "np.ndarray", A: "np.ndarray", cutoff: float = 1e-10) -> "tuple[np.ndarray, np.ndarray]":
    """Construct extended-valence and moment-expansion bath projectors.

    Parameters
    ----------
    D : np.ndarray
        Finite real symmetric (n,n) one-spin occupation matrix, n >= 2,
        eigenvalues in [-1e-10,1+1e-10]; do not renormalize its trace.
    A : np.ndarray
        Finite real (n,2) impurity coefficients, A.T@A=I within 1e-10
        entrywise. Both columns remain impurity; only the second is valence.
    cutoff : float
        Real finite scalar, 0 < cutoff < 1. Absolute singular-value cutoff.

    Returns
    -------
    (P_evb, P_meb) : tuple[np.ndarray, np.ndarray]
        Two (n,n) orthogonal environment projectors for the extended-valence
        and second-order moment-expansion bath prescriptions in Section
        II.2.1, equations (4)-(5), of
        https://arxiv.org/html/2601.01641v2.
        Use both impurity columns for the extended-valence bath and only
        the second impurity column as the moment-expansion seed.
        For each construction, concatenate its raw generating columns
        and retain left singular directions with singular values strictly
        greater than cutoff. Apply this cutoff to the concatenated raw
        matrix, without prior column normalization or sequential
        residual-vector thresholding. Retain at most two bath directions.
        Return the orthogonal projector onto the retained subspace.
        An empty retained span returns the zero projector. Symmetrize D.

    Raises
    ------
    ValueError
        If inputs are not convertible to finite real arrays/scalar, shapes
        violate the above, D is asymmetric beyond 1e-10 or violates its
        spectral bounds, A is not orthonormal within 1e-10, or cutoff is
        not a real scalar strictly between zero and one.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_bath_projectors(D: "np.ndarray", A: "np.ndarray", cutoff: float = 1e-10) -> "tuple[np.ndarray, np.ndarray]":
    """Deterministic reference implementation."""
    try:
        if np.iscomplexobj(D) or np.iscomplexobj(A) or np.iscomplexobj(cutoff) or np.ndim(cutoff)!=0: raise ValueError('real inputs')
        D=np.asarray(D,dtype=float); A=np.asarray(A,dtype=float); cutoff=float(cutoff)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('numeric inputs') from exc
    if D.ndim!=2 or D.shape[0]!=D.shape[1] or len(D)<2 or A.shape!=(len(D),2): raise ValueError('shape')
    if not np.isfinite(D).all() or not np.isfinite(A).all() or not np.isfinite(cutoff) or not 0<cutoff<1: raise ValueError('finite inputs')
    if np.max(np.abs(D-D.T))>1e-10 or np.max(np.abs(A.T@A-np.eye(2)))>1e-10: raise ValueError('symmetry or metric')
    D=(D+D.T)/2
    eig=np.linalg.eigvalsh(D)
    if eig.min()<-1e-10 or eig.max()>1+1e-10: raise ValueError('occupations')
    R=np.eye(len(D))-A@A.T
    matrices=[R@D@A,np.column_stack([R@D@A[:,1],R@D@D@A[:,1]])]
    out=[]
    for X in matrices:
        B,s,_=np.linalg.svd(X,full_matrices=False)
        B=B[:,s>cutoff]
        out.append(B@B.T)
    return out[0],out[1]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = """import numpy as np
F = np.array([
    [-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],
    [.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],
    [.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])
M = np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],
              [.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])
U = np.array([2.4,1.7,2.1,1.3,1.9,1.5])
"""
    s1 = base + """
def _run(fn, iao, thermal, bath=None, embed=None):
    f, m, u = F.copy(), M.copy(), U.copy()
    a = iao(f.copy(), m.copy(), 2)[:, :2].copy()
    d, _ = thermal(f.copy(), 2.3, 4.0)
    if bath is None:
        return fn(d.copy(), a.copy())
    p, _ = bath(d.copy(), a.copy())
    if embed is None:
        return fn(f.copy(), u.copy(), a.copy(), p.copy())
    h, na, nt = embed(f.copy(), u.copy(), a.copy(), p.copy())
    targets = np.array([3.6965052718310805, 2.834230591270832])
    return fn(h.copy(), na.copy(), nt.copy(), 2.3, targets)
"""
    s2 = """import numpy as np
D = np.eye(4) * 0.5
A = np.eye(4)[:, :2]
"""
    s3 = """import numpy as np
D = np.eye(4) * 0.5
D[1, 2] = D[2, 1] = 0.12
A = np.eye(4)[:, :2]
"""
    s4 = """import numpy as np
def _exception_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

D = np.eye(4) * 1.2
A = np.eye(4)[:, :2]
"""
    s5 = """import numpy as np
def _exception_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

D = np.eye(4) * 0.5
A = np.ones((4, 2))
"""
    s6 = """import numpy as np
rng = np.random.default_rng(731)
Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
D = (Q * np.array([0.97, 0.81, 0.56, 0.31, 0.12, 0.02])) @ Q.T
A = np.eye(6)[:, :2]
"""
    return [
        {"setup": s1,
         "call": '_run(bath_projectors, intrinsic_orbitals, thermal_reference)',
         "gold_call": '_run(_oracle_bath_projectors, _oracle_intrinsic_orbitals, _oracle_thermal_reference)'},
        {"setup": s2,
         "call": 'bath_projectors(D.copy(), A.copy())',
         "gold_call": '_oracle_bath_projectors(D.copy(), A.copy())'},
        {"setup": s3,
         "call": 'bath_projectors(D.copy(), A.copy())',
         "gold_call": '_oracle_bath_projectors(D.copy(), A.copy())'},
        {"setup": s4,
         "call": '_exception_code(bath_projectors, D.copy(), A.copy())',
         "gold_call": '_exception_code(_oracle_bath_projectors, D.copy(), A.copy())'},
        {"setup": s5,
         "call": '_exception_code(bath_projectors, D.copy(), A.copy())',
         "gold_call": '_exception_code(_oracle_bath_projectors, D.copy(), A.copy())'},
        {"setup": s6,
         "call": 'bath_projectors(D.copy(), A.copy(), 1e-10)',
         "gold_call": '_oracle_bath_projectors(D.copy(), A.copy(), 1e-10)'},
    ]
