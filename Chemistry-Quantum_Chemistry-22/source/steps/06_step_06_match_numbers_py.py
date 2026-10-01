"""
Solve simultaneous grand-canonical total and impurity number constraints using analytic thermal response.

Both chemical potentials enter the same Gibbs state. Solving the total constraint once and then shifting the impurity generally breaks the first constraint. An analytic susceptibility supports damped simultaneous Newton updates.

Returns
-------
tuple[np.ndarray, float, np.ndarray], fitted (gc,imp) chemical potentials, native-float impurity variance, and matched (total,impurity) means.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def match_numbers(H: "np.ndarray", NA: "np.ndarray", NT: "np.ndarray", beta: float, targets: "np.ndarray") -> "tuple[np.ndarray, float, np.ndarray]":
    """Find a jointly number-matched interacting thermal state.

    Parameters
    ----------
    H, NA, NT, beta
        Same real, finite, dimensional, symmetry and beta contracts as
        thermal_response: nonempty equal square operators, symmetry
        tolerance 1e-9, scalar 0 < beta <= 100. The traceless parts of
        NT and NA must be linearly independent: the smaller eigenvalue
        of their 2x2 Frobenius Gram matrix must exceed 1e-12.
    targets : np.ndarray
        Finite real (2,), ordered (total,impurity). Each target must be
        strictly inside the corresponding symmetrized operator's spectral
        interval. Inputs must admit a finite simultaneous solution.

    Returns
    -------
    (mu, variance, means) : tuple[np.ndarray, float, np.ndarray]
        Ordered chemical potentials (gc,imp), impurity variance, and
        ordered means. Both residuals must be <= 1e-9 in absolute value.
        Use analytic thermal response in the fit, not finite differences. Do not freeze a potential after
        satisfying only one constraint. No numerical output clipping.

    Raises
    ------
    ValueError
        On any invalid thermal_response input, invalid targets, failure of
        the stated operator-independence/spectral conditions, or inability
        to obtain a finite joint fit meeting absolute residual 1e-9.
    
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_match_numbers(H: "np.ndarray", NA: "np.ndarray", NT: "np.ndarray", beta: float, targets: "np.ndarray") -> "tuple[np.ndarray, float, np.ndarray]":
    """Deterministic reference implementation."""
    try:
        if np.iscomplexobj(targets): raise ValueError('real targets')
        targets=np.asarray(targets,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('targets') from exc
    if targets.shape!=(2,) or not np.isfinite(targets).all(): raise ValueError('targets')
    mu=np.zeros(2)
    means,var,J=_oracle_thermal_response(H,NA,NT,beta,mu)
    operators=[np.asarray(NT,dtype=float),np.asarray(NA,dtype=float)]
    centered=[]
    for op,target in zip(operators,targets):
        op=(op+op.T)/2; e=np.linalg.eigvalsh(op)
        if not e[0]<target<e[-1]: raise ValueError('target outside open spectrum')
        centered.append(op-np.trace(op)/len(op)*np.eye(len(op)))
    gram=np.array([[np.sum(a*b) for b in centered] for a in centered])
    if np.linalg.eigvalsh(gram)[0]<=1e-12: raise ValueError('dependent fields')
    for iteration in range(100):
        r=means-targets
        if np.max(np.abs(r))<=2e-11: break
        d,V=np.linalg.eigh(J)
        step=V@((V.T@r)/np.maximum(d,1e-10))
        size=np.max(np.abs(step))
        if size>10: step*=10/size
        accepted=False
        for backtrack in range(30):
            trial=mu-step*(.5**backtrack)
            tm,tv,tj=_oracle_thermal_response(H,NA,NT,beta,trial)
            if np.linalg.norm(tm-targets)<np.linalg.norm(r):
                mu=trial; means=tm; var=tv; J=tj; accepted=True; break
        if not accepted: break
    if np.max(np.abs(means-targets))>1e-9 or not np.isfinite(mu).all(): raise ValueError('joint fit did not converge')
    return mu,float(var),means

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
H = np.zeros((4, 4))
NA = np.diag([0.0, 1.0, 0.0, 1.0])
NT = np.diag([0.0, 0.0, 1.0, 1.0])
beta = 1.0
targets = np.array([0.5, 0.5])
"""
    s3 = """import numpy as np
H = np.array([
    [0.0, 0.3, 0.1],
    [0.3, 0.7, -0.2],
    [0.1, -0.2, 1.2],
])
NA = np.diag([0.0, 1.0, 2.0])
NT = np.diag([0.0, 2.0, 3.0])
beta = 1.8
targets = np.array([1.1101298803979867, 0.6148810052264091])
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

H = np.zeros((4, 4))
NA = np.diag([0.0, 1.0, 0.0, 1.0])
NT = np.diag([0.0, 0.0, 1.0, 1.0])
beta = 1.0
targets = np.array([0.0, 0.5])
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

H = np.eye(2)
NA = np.diag([0.0, 1.0])
NT = np.diag([0.0, 1.0])
beta = 1.0
targets = np.array([0.5, 0.5])
"""
    s6 = """import numpy as np
H = np.array([
    [0.0, 0.7, 0.2, 0.0],
    [0.7, 1.1, -0.4, 0.3],
    [0.2, -0.4, 0.4, 0.6],
    [0.0, 0.3, 0.6, 1.8],
])
NA = np.diag([0.0, 1.0, 1.0, 2.0])
NT = np.diag([0.0, 1.0, 2.0, 3.0])
beta = 3.5
targets = np.array([1.5313618580494572, 0.8469777351286496])
"""
    return [
        {"setup": s1,
         "call": '_run(match_numbers, intrinsic_orbitals, thermal_reference, bath_projectors, embedded_operators)',
         "gold_call": '_run(_oracle_match_numbers, _oracle_intrinsic_orbitals, _oracle_thermal_reference, _oracle_bath_projectors, _oracle_embedded_operators)'},
        {"setup": s2,
         "call": 'match_numbers(H.copy(), NA.copy(), NT.copy(), beta, targets.copy())',
         "gold_call": '_oracle_match_numbers(H.copy(), NA.copy(), NT.copy(), beta, targets.copy())'},
        {"setup": s3,
         "call": 'match_numbers(H.copy(), NA.copy(), NT.copy(), beta, targets.copy())',
         "gold_call": '_oracle_match_numbers(H.copy(), NA.copy(), NT.copy(), beta, targets.copy())'},
        {"setup": s4,
         "call": '_exception_code(match_numbers, H.copy(), NA.copy(), NT.copy(), beta, targets.copy())',
         "gold_call": '_exception_code(_oracle_match_numbers, H.copy(), NA.copy(), NT.copy(), beta, targets.copy())'},
        {"setup": s5,
         "call": '_exception_code(match_numbers, H.copy(), NA.copy(), NT.copy(), beta, targets.copy())',
         "gold_call": '_exception_code(_oracle_match_numbers, H.copy(), NA.copy(), NT.copy(), beta, targets.copy())'},
        {"setup": s6,
         "call": 'match_numbers(H.copy(), NA.copy(), NT.copy(), beta, targets.copy())',
         "gold_call": '_oracle_match_numbers(H.copy(), NA.copy(), NT.copy(), beta, targets.copy())'},
    ]
