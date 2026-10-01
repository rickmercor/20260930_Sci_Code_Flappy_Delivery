"""
Calculate a spin-restricted thermal reference density at a prescribed mean total electron number.

The density has one-spin occupations between zero and one and is not normalized to unit trace. A common chemical potential enforces spin-doubled electron number; stable Fermi occupations are needed when beta times an energy gap is large.

Returns
-------
tuple[np.ndarray, float], the one-spin density D and native Python chemical potential mu.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thermal_reference(F: "np.ndarray", beta: float, ne: float) -> "tuple[np.ndarray, float]":
    """Return the one-spin Fermi density and number-matching chemical potential.

    Parameters
    ----------
    F : np.ndarray
        Finite real symmetric (n,n) matrix, n >= 1; symmetry tolerance 1e-10.
    beta : float
        Finite inverse temperature, 0 < beta <= 100.
    ne : float
        Finite mean total electron number, 1e-6 <= ne <= 2*n-1e-6.

    Returns
    -------
    (D, mu) : tuple[np.ndarray, float]
        One-spin density D with 2*trace(D)=ne and native-float mu.
        Use the fixed F without a self-consistency update. Accepted
        near-symmetric F is symmetrized. Degenerate spectra are allowed.

    Raises
    ------
    ValueError
        If F is not convertible to a real finite square nonempty array,
        is asymmetric beyond 1e-10, or beta/ne are not real scalar numbers
        convertible to floats, are nonfinite, or violate the stated bounds.
    
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_thermal_reference(F: "np.ndarray", beta: float, ne: float) -> "tuple[np.ndarray, float]":
    """Deterministic reference implementation."""
    try:
        if np.iscomplexobj(F) or np.iscomplexobj(beta) or np.iscomplexobj(ne) or np.ndim(beta)!=0 or np.ndim(ne)!=0: raise ValueError('real scalar inputs')
        F=np.asarray(F,dtype=float); beta=float(beta); ne=float(ne)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError('numeric inputs') from exc
    if F.ndim!=2 or F.shape[0]!=F.shape[1] or F.shape[0]<1 or not np.isfinite(F).all(): raise ValueError('F')
    n=len(F)
    if np.max(np.abs(F-F.T))>1e-10 or not np.isfinite([beta,ne]).all() or not 0<beta<=100 or not 1e-6<=ne<=2*n-1e-6: raise ValueError('parameters')
    e,V=np.linalg.eigh((F+F.T)/2)
    lo=e.min()-50/beta; hi=e.max()+50/beta
    for _ in range(180):
        mu=(lo+hi)/2
        f=np.exp(-np.logaddexp(0,beta*(e-mu)))
        if 2*f.sum()<ne: lo=mu
        else: hi=mu
    mu=float((lo+hi)/2)
    f=np.exp(-np.logaddexp(0,beta*(e-mu)))
    return (V*f)@V.T,mu

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary, edge and invalid cases."""
    return [{'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n',
  'call': 'thermal_reference(F,2.3,4.)',
  'gold_call': '_oracle_thermal_reference(F,2.3,4.)'},
 {'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n',
  'call': 'thermal_reference(np.zeros((3,3)),1.,3.)',
  'gold_call': '_oracle_thermal_reference(np.zeros((3,3)),1.,3.)'},
 {'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n',
  'call': 'thermal_reference(np.diag([-5.,-.01,.01,5.]),80.,3.7)',
  'gold_call': '_oracle_thermal_reference(np.diag([-5.,-.01,.01,5.]),80.,3.7)'},
 {'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n'
           '\n'
           'def _exception_code(function, *args, **kwargs):\n'
           '    try:\n'
           '        function(*args, **kwargs)\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n',
  'call': '_exception_code(thermal_reference, F,0.,4.)',
  'gold_call': '_exception_code(_oracle_thermal_reference, F,0.,4.)'},
 {'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n'
           '\n'
           'def _exception_code(function, *args, **kwargs):\n'
           '    try:\n'
           '        function(*args, **kwargs)\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n',
  'call': '_exception_code(thermal_reference, F,2.,12.)',
  'gold_call': '_exception_code(_oracle_thermal_reference, F,2.,12.)'}]
