"""
Solve the eliminated temporal gauge field for globally neutral colour sources.

Equation (3) of the source paper is K A0^a + Q^a = 0. The full K is singular because the global gauge transformation at the tree root remains unfixed, but every supplied colour row has zero total charge. Solve in the orthogonal complement of the constant mode and return the unique zero-mean representative A0^a = -K^+ Q^a. Use all colour rows independently and preserve their ordering. Work in the full site space: do not ground a lattice site or add a regulator, and ensure every returned colour row has zero mean. For this numerical task, N >= 2 and C >= 1. On the constant-orthogonal subspace, require lambda_min > 1e-11 * lambda_max > 0; reject kernels at or below this task-defined relative cutoff. Require `max(abs(K-K.T)) <= 1e-11*norm(K,2)` and `norm(K@ones(N)) <= 1e-10*norm(K,2)*sqrt(N)`. Each source row q must satisfy `abs(sum(q)) <= 1e-10*max(1,norm(q))*sqrt(N)`. Interpret accepted roundoff using `Keff=P@((K+K.T)/2)@P` and `Qeff=Q@P`, where `P=I-ones((N,N))/N`: the cutoff and field equation refer to these projected, symmetric inputs.

Returns
-------
temporal_field : np.ndarray, shape (C, N), float Unique zero-mean solution rows satisfying Keff A0^a = -Qeff^a.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def solve_temporal_gauge_field(kernel, charges) -> np.ndarray:
    '''Zero-mean temporal field solving K A0 + Q = 0 in every colour channel.

    Parameters
    ----------
    kernel : array-like of shape (N, N)
        Finite real symmetric kernel on N >= 2 sites with one constant zero mode.
        On its orthogonal complement, lambda_min > 1e-11 * lambda_max > 0.
    charges : array-like of shape (C, N)
        Finite real globally neutral source in each of C >= 1 colour rows.

    Returns
    -------
    temporal_field : np.ndarray, shape (C, N), float
        Unique zero-mean solution rows satisfying Keff A0^a = -Qeff^a.

    Raises
    ------
    ValueError
        If the inputs violate the shape, realness, finiteness, symmetry, constant-mode,
        relative spectral-cutoff, or source-neutrality conditions stated above and
        in the background.
    '''
    return np.zeros_like(np.asarray(charges, dtype=float))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_temporal_gauge_field(kernel, charges):
    import numpy as np

    if np.iscomplexobj(kernel) or np.iscomplexobj(charges):
        raise ValueError("inputs must be real")
    try:
        K = np.asarray(kernel, dtype=float)
        Q = np.asarray(charges, dtype=float)
    except Exception as exc:
        raise ValueError("invalid input") from exc
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] < 2:
        raise ValueError("kernel must be square")
    N = K.shape[0]
    if Q.ndim != 2 or Q.shape[1] != N or Q.shape[0] < 1:
        raise ValueError("charges must have shape (C,N)")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(Q)):
        raise ValueError("inputs must be finite")
    matrix_scale = float(np.max(np.abs(K)))
    if matrix_scale == 0.0:
        raise ValueError("kernel is not positive on the singlet subspace")
    Kn = K / matrix_scale
    scale_k = float(np.linalg.norm(Kn, ord=2))
    if np.max(np.abs(Kn-Kn.T)) > 1e-11*scale_k:
        raise ValueError("kernel must be symmetric")
    if np.linalg.norm(Kn @ np.ones(N)) > 1e-10 * scale_k * np.sqrt(N):
        raise ValueError("constant vector is not the zero mode")
    row_scale = np.maximum(1.0, np.max(np.abs(Q), axis=1))
    scaled_q = Q / row_scale[:, None]
    norm_q = np.maximum(1.0 / row_scale, np.linalg.norm(scaled_q, axis=1))
    if np.any(np.abs(np.sum(scaled_q, axis=1)) > 1e-10 * norm_q * np.sqrt(N)):
        raise ValueError("charges must be globally neutral")

    U = np.zeros((N, N - 1), dtype=float)
    for j in range(1, N):
        s = np.sqrt(j * (j + 1.0))
        U[:j, j - 1] = 1.0 / s
        U[j, j - 1] = -j / s
    reduced = U.T @ ((Kn+Kn.T)/2) @ U
    reduced = (reduced+reduced.T)/2
    lam = np.linalg.eigvalsh(reduced)
    if lam[0] <= 1e-11 * float(lam[-1]):
        raise ValueError("kernel is not positive on the singlet subspace")
    solve_scale = np.max(np.abs(Q), axis=1)
    solve_scale = np.where(solve_scale == 0.0, 1.0, solve_scale)
    solve_q = Q / solve_scale[:, None]
    coeff = np.linalg.solve(reduced, (solve_q @ U).T)
    field = -(U @ coeff).T
    # Combine exponents so representable fields survive extreme input scales.
    field_m, field_e = np.frexp(field)
    charge_m, charge_e = np.frexp(solve_scale[:, None])
    kernel_m, kernel_e = np.frexp(matrix_scale)
    return np.ldexp(field_m * charge_m / kernel_m, field_e + charge_e - kernel_e)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '# case: normal\n'
           'import numpy as np\n'
           'def _lap(n,edges,w=1.0):\n'
           '    K=np.zeros((n,n))\n'
           '    for u,v in edges:\n'
           '        K[u,u]+=w; K[v,v]+=w; K[u,v]-=w; K[v,u]-=w\n'
           '    return K\n'
           'K=_lap(5,[(0,1),(1,2),(1,3),(3,4)],2.3)\n'
           'Q=np.array([[1.,-2.,.5,.25,.25],[-1.,0.,1.,0.,0.]])\n',
  'call': 'solve_temporal_gauge_field(K,Q)',
  'gold_call': '_oracle_solve_temporal_gauge_field(K,Q)'},
 {'setup': '# case: boundary\n'
           'import numpy as np\n'
           'def _lap(n,edges,w=1.0):\n'
           '    K=np.zeros((n,n))\n'
           '    for u,v in edges:\n'
           '        K[u,u]+=w; K[v,v]+=w; K[u,v]-=w; K[v,u]-=w\n'
           '    return K\n'
           'K=_lap(2,[(0,1)],.7)\n'
           'Q=np.array([[2.,-2.]])\n',
  'call': 'solve_temporal_gauge_field(K,Q)',
  'gold_call': '_oracle_solve_temporal_gauge_field(K,Q)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'def _lap(n,edges,w=1.0):\n'
           '    K=np.zeros((n,n))\n'
           '    for u,v in edges:\n'
           '        K[u,u]+=w; K[v,v]+=w; K[u,v]-=w; K[v,u]-=w\n'
           '    return K\n'
           'K=_lap(9,[(0,i) for i in range(1,9)],4.1)\n'
           'Q=np.arange(27,dtype=float).reshape(3,9)*np.array([[1.],[2.],[-3.]]); '
           'Q-=Q.mean(axis=1,keepdims=True)\n',
  'call': 'solve_temporal_gauge_field(K,Q)',
  'gold_call': '_oracle_solve_temporal_gauge_field(K,Q)'},
 {'setup': '# case: normal\n'
           'import numpy as np\n'
           'def _lap(n,edges,w=1.0):\n'
           '    K=np.zeros((n,n))\n'
           '    for u,v in edges:\n'
           '        K[u,u]+=w; K[v,v]+=w; K[u,v]-=w; K[v,u]-=w\n'
           '    return K\n'
           'K=_lap(7,[(i,i+1) for i in range(6)],.13)\n'
           'Q=np.array([[.2,-.1,.7,-.4,.3,-.2,-.5]])\n',
  'call': 'solve_temporal_gauge_field(K,Q)',
  'gold_call': '_oracle_solve_temporal_gauge_field(K,Q)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'def _mt(L,a):\n'
           '    V=L**3; K=np.zeros((V,V)); w=1.0/(a*a)\n'
           '    f=lambda x,y,z: x+L*(y+L*z)\n'
           '    E=[(f(x,y,z),f(x,y,z+1)) for z in range(L-1) for y in range(L) for x in range(L)]\n'
           '    E+=[(f(x,y,0),f(x,y+1,0)) for y in range(L-1) for x in range(L)]\n'
           '    E+=[(f(x,0,0),f(x+1,0,0)) for x in range(L-1)]\n'
           '    for u,v in E:\n'
           '        K[u,u]+=w; K[v,v]+=w; K[u,v]-=w; K[v,u]-=w\n'
           '    return K\n'
           'def _q(L,C):\n'
           '    V=L**3; i=np.arange(V,dtype=float)\n'
           '    Q=np.stack([np.cos(0.7*(c+1)*i+0.3*c)+0.5*np.sin(0.21*i*(c+2)) for c in '
           'range(C)])\n'
           '    return Q-Q.mean(axis=1,keepdims=True)\n'
           'K=_mt(4,0.83)\n'
           'Q=_q(4,3)\n',
  'call': 'solve_temporal_gauge_field(K,Q)',
  'gold_call': '_oracle_solve_temporal_gauge_field(K,Q)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except Exception: return 2\n'
           'K=np.array([[1.,-1.],[-1.,1.]])\n'
           'Q=np.array([[1.,-1.]])\n',
  'call': 'tuple((status(fn) for fn in (lambda: solve_temporal_gauge_field(np.ones((2, 3)), Q), '
          'lambda: solve_temporal_gauge_field(K, np.array([1.0, -1.0])), lambda: '
          'solve_temporal_gauge_field(K, np.array([[np.inf, -np.inf]])), lambda: '
          'solve_temporal_gauge_field(np.array([[1.0, 0.0], [-1.0, 1.0]]), Q), lambda: '
          'solve_temporal_gauge_field(np.eye(2), Q), lambda: solve_temporal_gauge_field(K, '
          'np.array([[1.0, 0.0]])), lambda: solve_temporal_gauge_field(np.zeros((2, 2)), Q))))',
  'gold_call': 'tuple((status(fn) for fn in (lambda: '
               '_oracle_solve_temporal_gauge_field(np.ones((2, 3)), Q), lambda: '
               '_oracle_solve_temporal_gauge_field(K, np.array([1.0, -1.0])), lambda: '
               '_oracle_solve_temporal_gauge_field(K, np.array([[np.inf, -np.inf]])), lambda: '
               '_oracle_solve_temporal_gauge_field(np.array([[1.0, 0.0], [-1.0, 1.0]]), Q), '
               'lambda: _oracle_solve_temporal_gauge_field(np.eye(2), Q), lambda: '
               '_oracle_solve_temporal_gauge_field(K, np.array([[1.0, 0.0]])), lambda: '
               '_oracle_solve_temporal_gauge_field(np.zeros((2, 2)), Q))))'}]
