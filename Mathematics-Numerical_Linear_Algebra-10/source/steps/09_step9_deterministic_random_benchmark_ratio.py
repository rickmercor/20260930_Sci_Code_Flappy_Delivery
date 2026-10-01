"""
Run the complete deterministic-versus-random benchmark pipeline.

The final ratio compares the worst inflation on a prescribed residual uncertainty ellipsoid with the exact random-orthonormal expectation. The basis and row selection stay fixed, while the full residual and its optimal reference both change with the perturbation.

Returns
-------
float, the unrounded robust-deterministic-to-random residual-inflation ratio.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def deterministic_random_benchmark_ratio(K, b: np.ndarray, m: int = 5, k: int = 1, s: int = 7, tie_tol: float = 1e-12, tau: float = 0.65, weights=None) -> float:
    r"""Run the complete deterministic-versus-random benchmark pipeline.

    Parameters
    ----------
    K : np.ndarray or int
        Finite real square matrix defining the linear system, or a
        positive integer $n$, in which case the deterministic banded
        operator of `construct_krylov_operator(n)` is used.
    b : np.ndarray
        Finite nonzero right-hand-side and Krylov starting vector.
    m : int, optional
        Krylov subspace dimension. The default is 5.
    k : int, optional
        Truncated-Arnoldi orthogonalization length. The default is 1.
    s : int, optional
        Deterministic sketch size and random embedding dimension.
        The default is 7.
    tie_tol : float, optional
        Finite nonnegative tolerance used in deterministic row-selection
        tie breaking. The default is 1e-12.

    tau : float, optional
        Finite nonnegative uncertainty radius, default $0.65$.
    weights : np.ndarray or None, optional
        Positive finite length-$n$ diagonal entries of $W$. If omitted,
        $W_{ii}=1+0.1i$ with one-based $i$.

    Notes
    -----
    Freeze the basis and selected rows from the original right-hand side.
    Put $M=KV$, $u=(b-MM^\dagger b)/\lVert b-MM^\dagger b\rVert_2$,
    and let $S$ extract the selected rows. The robust inflation is
    $\rho_{\mathrm{rob}}=\max_e
    \lVert (I-M(SM)^\dagger S)(u+e)\rVert_2^2/\lVert u+e\rVert_2^2$,
    subject to $M^\top e=0$, $u^\top e=0$, and $e^\top We=\tau^2$.
    The last constraint is equality. At zero radius use the nominal
    inflation. For positive radius require $n-m-1\ge1$. Whiten the
    metric only on this constrained subspace; Euclidean and weighted
    orthogonality are different. The denominator varies with direction.
    Use the fractional-sphere stage to obtain the global maximum.
    No Krylov regeneration or row reselection is performed for a perturbation.

    Returns
    -------
    value : float
        Unrounded ratio $\rho_{\mathrm{rob}}/\rho_{\mathrm{orth}}$ as a
        native Python float.

    Raises
    ------
    ValueError
        If K is neither a two-dimensional array nor a positive integer,
        or if any input violates the requirements of the Krylov-basis,
        Q-DEIM, GappyPOD+E, sketched-GMRES, residual-inflation, or
        random-orthonormal benchmark stages, if $\tau$ or the weights are invalid,
        or if positive $\tau$ has an empty tangent space.
    """
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_deterministic_random_benchmark_ratio(K, b, m=5, k=1, s=7, tie_tol=1e-12, tau=0.65, weights=None):
    if isinstance(K, (bool, np.bool_)):
        raise ValueError("K must be a square matrix or a positive integer")
    if isinstance(K, (int, np.integer)):
        K = _oracle_construct_krylov_operator(int(K))
    elif np.asarray(K).ndim != 2:
        raise ValueError("K must be a square matrix or a positive integer")
    V, M = _oracle_truncated_arnoldi_basis(K, b, m, k)
    p0 = _oracle_qdeim_indices(V, tie_tol)
    p = _oracle_gappypod_e_oversample(V, p0, s, tie_tol)
    _, residual_sq = _oracle_deterministic_sgmres_solve(V, M, b, p)
    rho_det, opt = _oracle_deterministic_residual_inflation(M, b, residual_sq)
    rho_orth = _oracle_random_orthonormal_benchmark(V.shape[0], V.shape[1], s, "real")
    if not np.isscalar(tau) or not np.isfinite(tau) or tau < 0:
        raise ValueError("invalid uncertainty radius")
    n = V.shape[0]
    weights = 1.0 + 0.1 * np.arange(1, n + 1) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be finite positive length n")
    if tau == 0:
        return float(_oracle_fractional_sphere_maximum(np.zeros((1, 1)), np.zeros(1), rho_det, np.zeros((1, 1)), 0.0) / rho_orth)
    if n - m - 1 < 1:
        raise ValueError("positive radius requires a nonempty tangent space")
    b = np.asarray(b, dtype=float)
    x = np.linalg.lstsq(M, b, rcond=None)[0]
    u = (b - M @ x) / np.sqrt(opt)
    Q = np.linalg.qr(np.column_stack((M, u)), mode="complete")[0]
    Z = Q[:, m + 1:]
    C = np.linalg.cholesky(Z.T @ (weights[:, None] * Z))
    T = np.linalg.solve(C, Z.T).T
    rows = p - 1
    # F maps residual perturbations to their oblique fitted component.
    F = M @ np.linalg.lstsq(M[rows], T[rows], rcond=None)[0]
    a = M @ np.linalg.lstsq(M[rows], u[rows], rcond=None)[0]
    B = T.T @ T
    A = B + F.T @ F
    g = F.T @ a
    rho_robust = _oracle_fractional_sphere_maximum(A, g, rho_det, B, tau)
    return float(rho_robust / rho_orth)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    def _base_test_cases():
        return [{'setup': 'import numpy as np\nK=np.zeros((14,14))\nfor i1 in range(1,15):\n    i=i1-1; K[i,i]=1+.05*i1\n    if i+1<14: K[i,i+1]=.8; K[i+1,i]=.01\n    if i+2<14: K[i,i+2]=.5\nb=np.array([-1.75034599,-2.39261019,-0.66844122,2.54142790,2.37457273,1.51299239,-0.75570191,-0.83104218,-1.29937175,-1.23666168,0.61959633,1.73510860,-0.25227877,-0.54794458])', 'call': 'deterministic_random_benchmark_ratio(K,b,5,1,7,1e-12)', 'gold_call': '_oracle_deterministic_random_benchmark_ratio(K,b,5,1,7,1e-12)'}, {'setup': 'import numpy as np\nb=np.array([-1.75034599,-2.39261019,-0.66844122,2.54142790,2.37457273,1.51299239,-0.75570191,-0.83104218,-1.29937175,-1.23666168,0.61959633,1.73510860,-0.25227877,-0.54794458])', 'call': 'deterministic_random_benchmark_ratio(14,b,5,1,7,1e-12)', 'gold_call': '_oracle_deterministic_random_benchmark_ratio(14,b,5,1,7,1e-12)'}, {'setup': 'import numpy as np\nK=np.diag(np.linspace(1.,3.,9))+np.diag(np.full(8,.7),1)+np.diag(np.full(7,.15),2)\nb=np.array([1.,-.4,.8,-.2,.6,.3,-.7,.5,-.1])', 'call': 'deterministic_random_benchmark_ratio(K,b,3,1,5,1e-12)', 'gold_call': '_oracle_deterministic_random_benchmark_ratio(K,b,3,1,5,1e-12)'}, {'setup': 'import numpy as np\nK=np.diag(np.linspace(1.,4.,10))+np.diag(np.full(9,.5),1)+np.diag(np.full(8,.2),2)\nb=np.array([.7,-.2,.4,.9,-.6,.3,.8,-.5,.1,.2])', 'call': 'deterministic_random_benchmark_ratio(K,b,4,2,7,0.0)', 'gold_call': '_oracle_deterministic_random_benchmark_ratio(K,b,4,2,7,0.0)'}, {'setup': 'import numpy as np\nK=np.eye(6); b=np.zeros(6)\ndef run_model():\n    try: deterministic_random_benchmark_ratio(K,b,2,1,4,1e-12); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_deterministic_random_benchmark_ratio(K,b,2,1,4,1e-12); return 0\n    except ValueError: return 1\n    except Exception: return 2', 'call': 'run_model()', 'gold_call': 'run_gold()'}, {'setup': 'import numpy as np\nK=np.diag(np.linspace(1.,2.5,8))+np.diag(np.full(7,.6),1)+np.diag(np.full(6,.15),2)\nb=np.array([1.,-.4,.7,.2,-.6,.5,-.3,.8])\ndef run_model():\n    try: deterministic_random_benchmark_ratio(K,b,3,1,4,1e-12); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_deterministic_random_benchmark_ratio(K,b,3,1,4,1e-12); return 0\n    except ValueError: return 1\n    except Exception: return 2', 'call': 'run_model()', 'gold_call': 'run_gold()'}, {'setup': 'import numpy as np\nK=np.ones((3,4)); b=np.ones(3)\ndef run_model():\n    try: deterministic_random_benchmark_ratio(K,b,2,1,3,1e-12); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_deterministic_random_benchmark_ratio(K,b,2,1,3,1e-12); return 0\n    except ValueError: return 1\n    except Exception: return 2', 'call': 'run_model()', 'gold_call': 'run_gold()'}]
    cases = _base_test_cases()
    base = 'import numpy as np\nb=np.array([.8,-.3,.7,.4,-.6,.9,-.2,.1,.5,-.4])'
    for tail in ['tau=0.', 'tau=.2,weights=np.geomspace(.05,20.,10)', 'tau=1.4,weights=np.array([2.,1.,4.,.5,3.,7.,.4,2.,1.,5.])']:
        call = f'deterministic_random_benchmark_ratio(10,b,3,1,5,1e-12,{tail})'
        cases.append({'setup': base, 'call': call, 'gold_call': '_oracle_' + call})
    call = 'deterministic_random_benchmark_ratio(10,b,3,1,10,tau=.8)'
    cases.append({'setup': base, 'call': call, 'gold_call': '_oracle_' + call})
    for tail in ['tau=-.1', 'weights=np.ones(9)', 'weights=np.zeros(10)', 'weights=np.full(10,np.inf)']:
        setup = base
        for label, prefix in [('model', ''), ('gold', '_oracle_')]:
            setup += f'\ndef run_{label}():\n    try:\n        {prefix}deterministic_random_benchmark_ratio(10,b,3,1,5,{tail})\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n'
        cases.append({'setup': setup, 'call': 'run_model()', 'gold_call': 'run_gold()'})
    return cases
