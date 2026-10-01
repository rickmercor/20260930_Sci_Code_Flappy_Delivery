"""
Compare interacting impurity variances using separate EVB and MEB number-matched ensembles.

The impurity target is shared, but each embedding target depends on the retained bath projector. The requested difference is MEB minus EVB. All projected interactions and all grand-canonical sectors contribute.

Returns
-------
tuple[float, float, float], native Python EVB variance, MEB variance, and MEB-minus-EVB difference.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compare_baths(h: "np.ndarray", U: "np.ndarray", A: "np.ndarray", D: "np.ndarray", beta: float) -> tuple[float, float, float]:
    """Evaluate the two bath variances with independent chemical-potential fits.

    Parameters
    ----------
    h, U, A
        Same contracts as embedded_operators for these inputs: finite real
        h (n,n), U (n,), A (n,2), n >= 2, h symmetry and A orthonormality
        to 1e-9. A must also satisfy bath_projectors' stricter 1e-10 metric
        tolerance. No restriction on the signs of U.
    D : np.ndarray
        Same (n,n) one-spin occupation density contract as bath_projectors:
        real finite, symmetry 1e-10, spectral bounds [-1e-10,1+1e-10].
    beta : float
        Finite real scalar, 0 < beta <= 100.

    Returns
    -------
    (var_evb, var_meb, difference) : tuple[float, float, float]
        Native floats, with difference=var_meb-var_evb. Generate baths
        using cutoff=1e-10; both must have rank two. The impurity target
        is 2*trace(A.T D A); the target for bath projector P is
        2*trace((A A.T+P) D).
        Obtain each pair of chemical potentials by match_numbers.

    Raises
    ------
    ValueError
        On any invalid input under bath_projectors/embedded_operators/
        match_numbers contracts; if either bath rank is not two; or if
        either target is outside its open operator spectral interval or
        no finite simultaneous fit reaches absolute residual 1e-9.
    
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compare_baths(h: "np.ndarray", U: "np.ndarray", A: "np.ndarray", D: "np.ndarray", beta: float) -> tuple[float, float, float]:
    """Deterministic reference implementation."""
    Pev,Pme=_oracle_bath_projectors(D,A,1e-10)
    A=np.asarray(A,dtype=float); D=np.asarray(D,dtype=float)
    out=[]
    for P in [Pev,Pme]:
        if np.count_nonzero(np.linalg.eigvalsh(P)>.5)!=2: raise ValueError('two bath orbitals required')
        H,NA,NT=_oracle_embedded_operators(h,U,A,P)
        targets=np.array([2*np.trace((A@A.T+P)@D),2*np.trace(A.T@D@A)])
        _,var,_=_oracle_match_numbers(H,NA,NT,beta,targets)
        out.append(float(var))
    return out[0],out[1],float(out[1]-out[0])

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
def _run(fn, iao, thermal, beta=2.3, nelec=4.0, scale=1.0, fit_beta=None):
    f, m, u = F.copy(), M.copy(), U.copy()
    a = iao(f.copy(), m.copy(), 2)[:, :2].copy()
    d, _ = thermal(f.copy(), beta, nelec)
    return fn(f.copy(), scale*u, a.copy(), d.copy(),
              beta if fit_beta is None else fit_beta)
"""
    s2 = base + """
def _exception_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
"""
    s3 = base + """
def _run(fn, iao, thermal, beta=2.3, nelec=4.0, scale=1.0, fit_beta=None):
    f, m, u = F.copy(), M.copy(), U.copy()
    a = iao(f.copy(), m.copy(), 2)[:, :2].copy()
    d, _ = thermal(f.copy(), beta, nelec)
    return fn(f.copy(), scale*u, a.copy(), d.copy(),
              beta if fit_beta is None else fit_beta)

def _exception_code(fn):
    try:
        fn()
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
"""
    return [
        {"setup": s1,
         "call": '_run(compare_baths, intrinsic_orbitals, thermal_reference)',
         "gold_call": '_run(_oracle_compare_baths, _oracle_intrinsic_orbitals, _oracle_thermal_reference)'},
        {"setup": s1,
         "call": '_run(compare_baths, intrinsic_orbitals, thermal_reference, scale=0.0)',
         "gold_call": '_run(_oracle_compare_baths, _oracle_intrinsic_orbitals, _oracle_thermal_reference, scale=0.0)'},
        {"setup": s1,
         "call": '_run(compare_baths, intrinsic_orbitals, thermal_reference, beta=1.1, nelec=3.3, scale=-0.3)',
         "gold_call": '_run(_oracle_compare_baths, _oracle_intrinsic_orbitals, _oracle_thermal_reference, beta=1.1, nelec=3.3, scale=-0.3)'},
        {"setup": s2,
         "call": '_exception_code(compare_baths, F.copy(),U.copy(),np.eye(6)[:,:2],np.eye(6)*.5,2.)',
         "gold_call": '_exception_code(_oracle_compare_baths, F.copy(),U.copy(),np.eye(6)[:,:2],np.eye(6)*.5,2.)'},
        {"setup": s3,
         "call": '_exception_code(lambda: _run(compare_baths, intrinsic_orbitals, thermal_reference, fit_beta=0.0))',
         "gold_call": '_exception_code(lambda: _run(_oracle_compare_baths, _oracle_intrinsic_orbitals, _oracle_thermal_reference, fit_beta=0.0))'},
    ]
