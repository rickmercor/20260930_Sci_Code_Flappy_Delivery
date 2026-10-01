"""
Invert the singular Gauss kernel only on the global singlet subspace.

The paper's maximal-tree kernel has a constant zero mode corresponding to the residual global gauge transformation. Equation (5) restricts the physical states to the global charge singlet, and the paragraph carrying it states that with that mode excluded the kernel is positive definite and therefore invertible, so Gauss's law is solved as A_0 = -K^-1 Q. The nonlocal term of Equation (8) is the consumer of that inverse. Return the unique symmetric Moore-Penrose Green matrix that annihilates the normalized constant vector, inverts the kernel on its orthogonal complement, and has zero row and column sums. The input may be a connected weighted graph Laplacian on N >= 2 sites with exactly one constant zero mode. For this numerical task, the smallest eigenvalue of K restricted to the constant-orthogonal subspace must exceed 1e-11 times its largest eigenvalue; reject kernels at or below this relative cutoff. This conditioning restriction is task-defined and invariant under positive rescaling. Require `max(abs(K-K.T)) <= 1e-11*norm(K,2)` and `norm(K@ones(N)) <= 1e-10*norm(K,2)*sqrt(N)`. Interpret accepted roundoff using `Keff=P@((K+K.T)/2)@P`, where `P=I-ones((N,N))/N`; the cutoff and inverse refer to Keff.

Returns
-------
green : np.ndarray, shape (N, N), float Symmetric Moore-Penrose inverse of Keff with zero row and column sums.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def invert_gauss_kernel_on_singlet(kernel) -> np.ndarray:
    '''Green matrix for the positive Gauss kernel with only its constant mode removed.

    Parameters
    ----------
    kernel : array-like of shape (N, N)
        Finite real symmetric kernel on N >= 2 sites with one constant zero mode.
        On its orthogonal complement, lambda_min > 1e-11 * lambda_max > 0.

    Returns
    -------
    green : np.ndarray, shape (N, N), float
        Symmetric Moore-Penrose inverse of Keff with zero row and column sums.

    Raises
    ------
    ValueError
        If kernel violates the shape, realness, finiteness, symmetry, constant-mode,
        or relative spectral-cutoff conditions stated above and in the background.
    '''
    return np.zeros_like(np.asarray(kernel, dtype=float))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_invert_gauss_kernel_on_singlet(kernel):
    import numpy as np

    if np.iscomplexobj(kernel):
        raise ValueError("kernel must be real")
    try:
        K = np.asarray(kernel, dtype=float)
    except Exception as exc:
        raise ValueError("invalid kernel") from exc
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] < 2:
        raise ValueError("kernel must be square")
    if not np.all(np.isfinite(K)):
        raise ValueError("kernel must be finite")
    N = K.shape[0]
    matrix_scale = float(np.max(np.abs(K)))
    if matrix_scale == 0.0:
        raise ValueError("kernel is not positive on the singlet subspace")
    Kn = K / matrix_scale
    scale = float(np.linalg.norm(Kn, ord=2))
    if np.max(np.abs(Kn-Kn.T)) > 1e-11*scale:
        raise ValueError("kernel must be symmetric")
    if np.linalg.norm(Kn @ np.ones(N)) > 1e-10 * scale * np.sqrt(N):
        raise ValueError("constant vector is not the zero mode")

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
    G = U @ np.linalg.solve(reduced, U.T)
    return (0.5 * (G + G.T)) / matrix_scale

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
           'K=_lap(5,[(0,1),(1,2),(1,3),(3,4)],2.3)\n',
  'call': 'invert_gauss_kernel_on_singlet(K)',
  'gold_call': '_oracle_invert_gauss_kernel_on_singlet(K)'},
 {'setup': '# case: boundary\n'
           'import numpy as np\n'
           'def _lap(n,edges,w=1.0):\n'
           '    K=np.zeros((n,n))\n'
           '    for u,v in edges:\n'
           '        K[u,u]+=w; K[v,v]+=w; K[u,v]-=w; K[v,u]-=w\n'
           '    return K\n'
           'K=_lap(2,[(0,1)],0.7)\n',
  'call': 'invert_gauss_kernel_on_singlet(K)',
  'gold_call': '_oracle_invert_gauss_kernel_on_singlet(K)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'def _lap(n,edges,w=1.0):\n'
           '    K=np.zeros((n,n))\n'
           '    for u,v in edges:\n'
           '        K[u,u]+=w; K[v,v]+=w; K[u,v]-=w; K[v,u]-=w\n'
           '    return K\n'
           'K=_lap(9,[(0,i) for i in range(1,9)],4.1)\n',
  'call': 'invert_gauss_kernel_on_singlet(K)',
  'gold_call': '_oracle_invert_gauss_kernel_on_singlet(K)'},
 {'setup': '# case: normal\n'
           'import numpy as np\n'
           'def _lap(n,edges,w=1.0):\n'
           '    K=np.zeros((n,n))\n'
           '    for u,v in edges:\n'
           '        K[u,u]+=w; K[v,v]+=w; K[u,v]-=w; K[v,u]-=w\n'
           '    return K\n'
           'K=_lap(7,[(i,i+1) for i in range(6)],0.13)\n',
  'call': 'invert_gauss_kernel_on_singlet(K)',
  'gold_call': '_oracle_invert_gauss_kernel_on_singlet(K)'},
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
           'K=_mt(4,0.83)\n',
  'call': 'invert_gauss_kernel_on_singlet(K)',
  'gold_call': '_oracle_invert_gauss_kernel_on_singlet(K)'},
 {'setup': '# case: edge\n'
           'import numpy as np\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except Exception: return 2\n',
  'call': 'tuple((status(fn) for fn in (lambda: '
          'invert_gauss_kernel_on_singlet(np.ones((2, 3))), lambda: '
          'invert_gauss_kernel_on_singlet(np.array([[np.nan, 0.0], [0.0, 0.0]])), lambda: '
          'invert_gauss_kernel_on_singlet(np.array([[1.0, 0.0], [-1.0, 1.0]])), lambda: '
          'invert_gauss_kernel_on_singlet(np.eye(2)), lambda: '
          'invert_gauss_kernel_on_singlet(np.zeros((2, 2))))))',
  'gold_call': 'tuple((status(fn) for fn in (lambda: '
               '_oracle_invert_gauss_kernel_on_singlet(np.ones((2, 3))), lambda: '
               '_oracle_invert_gauss_kernel_on_singlet(np.array([[np.nan, 0.0], [0.0, 0.0]])), '
               'lambda: _oracle_invert_gauss_kernel_on_singlet(np.array([[1.0, 0.0], [-1.0, '
               '1.0]])), lambda: _oracle_invert_gauss_kernel_on_singlet(np.eye(2)), lambda: '
               '_oracle_invert_gauss_kernel_on_singlet(np.zeros((2, 2))))))'}]
