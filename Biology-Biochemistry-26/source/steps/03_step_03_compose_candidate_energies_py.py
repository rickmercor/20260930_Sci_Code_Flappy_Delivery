"""
Compose baseline and mixed geometric-topological energy vectors in supplied candidate order.

Combine aligned geometric and topological state summaries according to the matching treatment. geometry is finite shape (N,), persistence is finite nonnegative shape (N,3), and records is finite nonempty shape (K,5), with columns [lambda0,lambda1,lambda2,mu,temperature]. N>=1; each mu lies in [0,1] and each temperature is positive. Return a float array of shape (K+1,N), with the pure geometric setup first, followed by the records in their input order. Temperatures do not alter a state's energy. Invalid inputs raise ValueError. Do not mutate inputs.

Returns
-------
energy_panel : np.ndarray — float array of shape (K+1, N), with the pure geometric setup first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def compose_candidate_energies(
    geometry: np.ndarray,
    persistence: np.ndarray,
    records: np.ndarray,
) -> np.ndarray:
    """Return pure-geometric and mixed candidate energies for all states.
 
    Parameters
    ----------
    geometry
        Finite geometric energy vector of shape (N,).
    persistence
        Nonnegative total-persistence array of shape (N, 3).
    records
        Finite candidate records of shape (K, 5).
 
    Returns
    -------
    np.ndarray
        Float array of shape (K+1, N).
    """
    return energy_panel

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compose_candidate_energies(
    geometry: np.ndarray,
    persistence: np.ndarray,
    records: np.ndarray,
) -> np.ndarray:
    import numpy as np
    g=np.asarray(geometry,dtype=float); p=np.asarray(persistence,dtype=float)
    r=np.asarray(records,dtype=float)
    if g.ndim!=1 or not len(g) or p.shape!=(len(g),3) or r.ndim!=2 or r.shape[1]!=5 or not len(r):
        raise ValueError('expected g[N], persistence[N,3], records[K,5]')
    if not all(np.all(np.isfinite(a)) for a in [g,p,r]) or np.any(p<0) or np.any((r[:,3]<0)|(r[:,3]>1)) or np.any(r[:,4]<=0):
        raise ValueError('invalid energies, persistence, mixture or temperature')
    return np.vstack([g,r[:,3,None]*g[None,:]+(1-r[:,3,None])*(r[:,:3]@p.T)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'g=np.array([1.,-3.,5.]);p=np.array([[4.,2.,1.],[2.,7.,3.],[5.,1.,6.]])\n'
               'r=np.array([[1.,-.2,-.8,.4,2.],[.3,.1,-.7,.7,4.]])\n',
      'call': 'compose_candidate_energies(g,p,r)',
      'gold_call': '_oracle_compose_candidate_energies(g,p,r)'},
     {'setup': 'import numpy as np\n'
               'g=np.array([2.]);p=np.array([[1.,2.,3.]]);r=np.array([[1.,2.,3.,0.,1.],[1.,2.,3.,1.,7.]])\n',
      'call': 'compose_candidate_energies(g,p,r)',
      'gold_call': '_oracle_compose_candidate_energies(g,p,r)'},
     {'setup': 'import numpy as np\ng=np.array([0.,0.]);p=np.zeros((2,3));r=np.array([[1.,-1.,1.,.5,.1]])\n',
      'call': 'compose_candidate_energies(g,p,r)',
      'gold_call': '_oracle_compose_candidate_energies(g,p,r)'},
     {'setup': 'import numpy as np\n'
               'g=np.array([1.]);p=np.array([[-1.,0.,0.]]);r=np.array([[1.,1.,1.,.5,2.]])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        compose_candidate_energies(g,p,r)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_compose_candidate_energies(g,p,r)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
