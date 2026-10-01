"""
Infer the unique net-flux optimum for every design and independent condition.

Reaction-attuned directional entropy uses the supplied reaction weights in both directions. Organic source balances, membrane-current balance and one-way net constraints define the feasible phenotype; dependent conservation equations do not provide additional information.

Returns
-------
A finite ndarray of shape (C,P,N).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phenotype_panel(balanced: "np.ndarray", demand: "np.ndarray",

                    weights: "np.ndarray", factors: "np.ndarray",

                    oneway: "np.ndarray") -> "np.ndarray":

    """

    balanced has shape (P,M+3,N), with M organic rows, two buffered H+

    rows and one charge row. demand is length M; the charge demand is

    zero. weights is positive finite (C,N), factors positive finite

    (P,N); multiply factors independently into the original weights.

    oneway lists distinct zero-based net-nonnegative reaction indices.

    Return net fluxes (C,P,N) for the source directional-entropy optimum:

    with effective weights w = weights * factors, maximise

    -sum_j [f_j ln(f_j / w_j) + r_j ln(r_j / w_j)] over strictly positive

    forward and reverse variables whose differences f_j - r_j are the net

    fluxes, subject to the balance rows and the one-way constraints.

    All other net fluxes are unconstrained in sign; no normalization or

    extra linear flux reward is added. Finite aligned arrays and feasible

    balances are required; tested shape/value violations or infeasibility

    raise ValueError. An unresolved numerical solve raises RuntimeError.

    Duplicate conservation rows are allowed. Absolute accuracy 0.000002

    is sufficient; numerical fluxes below 2e-10 may be set to zero.

    """

    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_phenotype_panel(balanced: "np.ndarray", demand: "np.ndarray",
                    weights: "np.ndarray", factors: "np.ndarray",
                    oneway: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy.linalg import null_space, qr
    from itertools import combinations
    bal, b, g, fac = map(lambda x: np.asarray(x, dtype=float),
                         (balanced, demand, weights, factors))
    one = np.asarray(oneway, dtype=int)
    if (bal.ndim != 3 or g.ndim != 2 or fac.shape != (bal.shape[0], bal.shape[2])
            or b.shape != (bal.shape[1]-3,) or g.shape[1] != bal.shape[2]
            or g.shape[0] == 0 or np.any(g <= 0) or np.any(fac <= 0)
            or any(not np.isfinite(x).all() for x in (bal, b, g, fac))
            or one.ndim != 1 or np.any((one < 0) | (one >= g.shape[1]))
            or len(np.unique(one)) != len(one)):
        raise ValueError('Aligned positive phenotype inputs required')
    out = np.empty((g.shape[0], fac.shape[0], g.shape[1]))
    for p in range(fac.shape[0]):
        full = np.vstack([bal[p, :-3], bal[p, -1]])
        rhs = np.r_[b, 0.]
        initial = np.linalg.lstsq(full, rhs, rcond=1e-12)[0]
        if np.max(np.abs(full @ initial - rhs)) > 1e-8:
            raise ValueError('Inconsistent source balances')
        _, _, piv = qr(full.T, pivoting=True, mode='economic')
        rank = np.linalg.matrix_rank(full, tol=1e-10)
        basis, target = full[piv[:rank]], rhs[piv[:rank]]
        for c in range(g.shape[0]):
            scale = 2. * g[c] * fac[p] / np.e
            best, best_cost = None, np.inf
            for size in range(one.size+1):
                for active in combinations(one, size):
                    a = np.vstack([basis, np.eye(g.shape[1])[list(active)]])
                    d = np.r_[target, np.zeros(size)]
                    v = np.linalg.lstsq(a, d, rcond=1e-12)[0]
                    if np.max(np.abs(a @ v-d), initial=0.) > 1e-8:
                        continue
                    null = null_space(a, rcond=1e-12)
                    def _objective(n):
                        return float(np.sum(n*np.arcsinh(n/scale)-np.hypot(n, scale)))
                    converged = False
                    for iteration in range(150):
                        gradient = null.T @ np.arcsinh(v/scale)
                        if np.max(np.abs(gradient), initial=0.) < 2e-11:
                            converged = True
                            break
                        hessian = (null.T / np.hypot(v, scale)) @ null
                        direction = np.linalg.solve(hessian, gradient)
                        step, alpha, old = -null @ direction, 1., _objective(v)
                        while (_objective(v+alpha*step) > old-1e-4*alpha*(gradient@direction)+2e-13
                               and alpha > 1e-10):
                            alpha *= .5
                        v += alpha * step
                    if not converged:
                        raise RuntimeError('Entropy face solve did not converge')
                    if np.min(v[one], initial=0.) < -2e-8:
                        continue
                    cost = _objective(v)
                    if cost < best_cost:
                        best_cost, best = cost, v.copy()
            if best is None:
                raise ValueError('Source balances and one-way evidence are infeasible')
            best[np.abs(best) < 2e-10] = 0.
            out[c, p] = best
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               's=np.array([[-1.,-1.,0.],[1.,1.,0.]])\n'
               'bal=np.zeros((2,5,3)); bal[:,:2]=s; bal[:,-1,:]=[-1.,0.,1.]\n'
               'b=np.array([-2.,2.]); g=np.array([[1.,4.,.03],[3.,2.,.1]])\n'
               'fac=np.array([[1.,1.,1.],[.4,2.,1.]])\n'
               'one=np.array([2],dtype=int)\n',
      'call': 'phenotype_panel(bal,b,g,fac,one)',
      'gold_call': '_oracle_phenotype_panel(bal,b,g,fac,one)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               's=np.array([[-1.,-1.,0.],[1.,1.,0.]])\n'
               'bal=np.zeros((2,5,3)); bal[:,:2]=s; bal[:,-1,:]=[-1.,0.,1.]\n'
               'b=np.array([-2.,2.]); g=np.array([[1.,4.,.03],[3.,2.,.1]])\n'
               'fac=np.array([[1.,1.,1.],[.4,2.,1.]])\n'
               'one=np.array([2],dtype=int)\n'
               'b*=0.',
      'call': 'phenotype_panel(bal,b,g,fac,one)',
      'gold_call': '_oracle_phenotype_panel(bal,b,g,fac,one)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               's=np.array([[-1.,-1.,0.],[1.,1.,0.]])\n'
               'bal=np.zeros((2,5,3)); bal[:,:2]=s; bal[:,-1,:]=[-1.,0.,1.]\n'
               'b=np.array([-2.,2.]); g=np.array([[1.,4.,.03],[3.,2.,.1]])\n'
               'fac=np.array([[1.,1.,1.],[.4,2.,1.]])\n'
               'one=np.array([2],dtype=int)\n'
               'one=np.array([0,2]); g=np.array([[.02,8.,.01]]); fac[1]=[3.,.3,2.]',
      'call': 'phenotype_panel(bal,b,g,fac,one)',
      'gold_call': '_oracle_phenotype_panel(bal,b,g,fac,one)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               's=np.array([[-1.,-1.,0.],[1.,1.,0.]])\n'
               'bal=np.zeros((2,5,3)); bal[:,:2]=s; bal[:,-1,:]=[-1.,0.,1.]\n'
               'b=np.array([-2.,2.]); g=np.array([[1.,4.,.03],[3.,2.,.1]])\n'
               'fac=np.array([[1.,1.,1.],[.4,2.,1.]])\n'
               'one=np.array([2],dtype=int)\n'
               'b=np.array([-2.,3.])\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(phenotype_panel, (bal,b,g,fac,one,))',
      'gold_call': '_raises_value_error(_oracle_phenotype_panel, (bal,b,g,fac,one,))',
      'tol': 0.0}]
