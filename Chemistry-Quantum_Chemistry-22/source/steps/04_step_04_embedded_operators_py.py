"""
Project the interacting spatial-orbital Hamiltonian into an impurity-plus-bath space and construct full-Fock number operators.

Projection of original onsite opposite-spin interactions produces a general four-index interaction in the embedding. A deterministic bath-coordinate convention is used solely to make matrix-valued outputs comparable; physical results are invariant under bath rotations.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray], full-Fock H, impurity-number NA, and total-number NT matrices.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def embedded_operators(h: "np.ndarray", U: "np.ndarray", A: "np.ndarray", P: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Build H, impurity number NA, and total number NT on the full Fock space.

    Parameters
    ----------
    h : np.ndarray
        Finite real symmetric (n,n) original one-body matrix, n >= 2.
    U : np.ndarray
        Finite real (n,) onsite opposite-spin interaction coefficients.
        Zero and negative coefficients are allowed.
    A : np.ndarray
        Finite real (n,2) orthonormal impurity columns, kept in this order.
    P : np.ndarray
        Finite real symmetric (n,n) bath orthogonal projector, orthogonal
        to A, rank r in {0,1,2}. All matrix tolerances below are 1e-9.

    Returns
    -------
    (H, NA, NT) : tuple[np.ndarray, np.ndarray, np.ndarray]
        Real matrices of size 2**(2*k), k=2+r. Define a bath basis by
        repeatedly taking a column of residual projector R, selecting the
        smallest index whose R[j,j] is within 1e-12 of its maximum,
        normalizing R[:,j] in Euclidean norm, then replacing R by
        sym(R-b b.T). Start R=sym(P), perform r iterations, E=[A,B].
        Basis states are ascending integers 0..2**(2*k)-1, with bits
        0..k-1 for up spin and k..2*k-1 for down spin. Apply canonical
        fermionic signs in that order. Project all original interactions
        U_i n_i_up n_i_down, not just embedding onsite terms. Excluded
        orbitals are empty; add no frozen-core, mean-field or constant term.

    Raises
    ------
    ValueError
        If arrays are not convertible to finite real floats; dimensions
        violate the above; h/P is asymmetric beyond 1e-9; A.T@A differs
        from I beyond 1e-9; P@P differs from P beyond 1e-9; P@A differs
        from zero beyond 1e-9; or the number of eigenvalues of sym(P)
        greater than 0.5 exceeds two. Symmetrize accepted h/P.
    
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_embedded_operators(h: "np.ndarray", U: "np.ndarray", A: "np.ndarray", P: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Deterministic reference implementation."""
    try:
        if any(np.iscomplexobj(x) for x in [h,U,A,P]): raise ValueError('real inputs')
        h=np.asarray(h,dtype=float); U=np.asarray(U,dtype=float); A=np.asarray(A,dtype=float); P=np.asarray(P,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('numeric inputs') from exc
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or len(h)<2: raise ValueError('h shape')
    n=len(h)
    if U.shape!=(n,) or A.shape!=(n,2) or P.shape!=(n,n) or not all(np.isfinite(x).all() for x in [h,U,A,P]): raise ValueError('shape or finiteness')
    if np.max(np.abs(h-h.T))>1e-9 or np.max(np.abs(P-P.T))>1e-9 or np.max(np.abs(A.T@A-np.eye(2)))>1e-9 or np.max(np.abs(P@P-P))>1e-9 or np.max(np.abs(P@A))>1e-9: raise ValueError('matrix conditions')
    h=(h+h.T)/2; P=(P+P.T)/2
    r=int(np.count_nonzero(np.linalg.eigvalsh(P)>.5))
    if r>2: raise ValueError('rank')
    R=P.copy(); cols=[]
    for _ in range(r):
        diagonal=np.diag(R); j=int(np.flatnonzero(diagonal>=diagonal.max()-1e-12)[0])
        b=R[:,j]/np.linalg.norm(R[:,j]); cols.append(b)
        R-=np.outer(b,b); R=(R+R.T)/2
    E=np.column_stack([A]+cols); k=E.shape[1]; dim=1<<(2*k)
    ht=E.T@h@E
    H=np.zeros((dim,dim)); nt=np.zeros(dim); na=np.zeros(dim)
    # Second quantization of each original-site density, separately by spin.
    densities=[]
    for spin in range(2):
        site=np.zeros((n,dim,dim))
        for state in range(dim):
            if spin==0:
                nt[state]=state.bit_count()
                na[state]=sum((state>>p)&1 for p in [0,1,k,k+1])
            for q in range(k):
                qb=q+spin*k
                if not (state>>qb)&1: continue
                t=state^(1<<qb); sq=(-1)**((state&((1<<qb)-1)).bit_count())
                for p in range(k):
                    pb=p+spin*k
                    if (t>>pb)&1: continue
                    dest=t|(1<<pb); sign=sq*(-1)**((t&((1<<pb)-1)).bit_count())
                    H[dest,state]+=ht[p,q]*sign
                    site[:,dest,state]+=E[:,p]*E[:,q]*sign
        densities.append(site)
    for i in range(n): H+=U[i]*(densities[0][i]@densities[1][i])
    return (H+H.T)/2,np.diag(na),np.diag(nt)

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
h = np.diag([-0.4, 0.7])
U = np.array([1.0, 0.0])
A = np.eye(2)
P = np.zeros((2, 2))
"""
    s3 = base + """
A = np.eye(6)[:, :2]
v = np.array([0.0, 0.0, 1.0, 2.0, -1.0, 0.0])
v /= np.linalg.norm(v)
P = np.outer(v, v)
"""
    s4 = base + """
def _exception_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

A = np.eye(6)[:, :2]
P = np.eye(6)
"""
    s5 = base + """
def _exception_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

A = np.eye(6)[:, :2]
P = np.eye(6) * 0.3
"""
    s6 = """import numpy as np
rng = np.random.default_rng(947)
Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
A = Q[:, :2]
B = Q[:, 2:4]
P = B @ B.T
h = np.diag([-1.4, -0.7, -0.1, 0.4, 0.9, 1.5])
h[0, 3] = h[3, 0] = 0.23
h[1, 5] = h[5, 1] = -0.17
U = np.array([2.7, 0.4, 1.9, 3.1, 0.8, 1.3])
"""
    return [
        {"setup": s1,
         "call": '_run(embedded_operators, intrinsic_orbitals, thermal_reference, bath_projectors)',
         "gold_call": '_run(_oracle_embedded_operators, _oracle_intrinsic_orbitals, _oracle_thermal_reference, _oracle_bath_projectors)'},
        {"setup": s2,
         "call": 'embedded_operators(h.copy(), U.copy(), A.copy(), P.copy())',
         "gold_call": '_oracle_embedded_operators(h.copy(), U.copy(), A.copy(), P.copy())'},
        {"setup": s3,
         "call": 'embedded_operators(F.copy(), -U.copy(), A.copy(), P.copy())',
         "gold_call": '_oracle_embedded_operators(F.copy(), -U.copy(), A.copy(), P.copy())'},
        {"setup": s4,
         "call": '_exception_code(embedded_operators, F.copy(), U.copy(), A.copy(), P.copy())',
         "gold_call": '_exception_code(_oracle_embedded_operators, F.copy(), U.copy(), A.copy(), P.copy())'},
        {"setup": s5,
         "call": '_exception_code(embedded_operators, F.copy(), U.copy(), A.copy(), P.copy())',
         "gold_call": '_exception_code(_oracle_embedded_operators, F.copy(), U.copy(), A.copy(), P.copy())'},
        {"setup": s6,
         "call": 'embedded_operators(h.copy(), U.copy(), A.copy(), P.copy())',
         "gold_call": '_oracle_embedded_operators(h.copy(), U.copy(), A.copy(), P.copy())'},
    ]
