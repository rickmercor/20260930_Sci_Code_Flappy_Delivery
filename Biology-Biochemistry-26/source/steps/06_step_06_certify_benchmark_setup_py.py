"""
Convert one setup's replay representatives into success bounds and a catalogue-wide soundness flag.

Certify a single setup using its finite energy vector, aligned nonnegative assembly distances and nonempty integer vector of run-representative indices. cutoff and z are finite positive scalars. A state is assembled exactly when its distance is strictly below cutoff. Return a float array [successful_run_count, lower_bound, upper_bound, soundness_flag], where the bounds use the matching treatment's binomial-interval convention with the supplied normal quantile z. The task-defined soundness flag is one exactly when the global minimum-energy state over the entire supplied state catalogue is assembled, and zero otherwise. Exact energy ties choose the smallest state index. Invalid inputs raise ValueError. Do not mutate inputs

Returns
-------
certificate : np.ndarray — float array [successful_run_count, lower_bound, upper_bound, soundness_flag].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def certify_benchmark_setup(
    energies: np.ndarray,
    distances: np.ndarray,
    run_minima: np.ndarray,
    cutoff: float,
    z: float,
) -> np.ndarray:
    """Return replay success bounds and global-minimum soundness.
 
    Parameters
    ----------
    energies, distances
        Aligned finite state vectors.
    run_minima
        Nonempty integer vector of representative state indices.
    cutoff, z
        Positive assembly cutoff and normal quantile.
 
    Returns
    -------
    np.ndarray
        Float certificate of shape (4,).
    """
    return certificate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_certify_benchmark_setup(
    energies: np.ndarray,
    distances: np.ndarray,
    run_minima: np.ndarray,
    cutoff: float,
    z: float,
) -> np.ndarray:
    import numpy as np
    e=np.asarray(energies,dtype=float); d=np.asarray(distances,dtype=float); m=np.asarray(run_minima)
    if e.ndim!=1 or not len(e) or d.shape!=e.shape or m.ndim!=1 or not len(m) or not np.issubdtype(m.dtype,np.integer):
        raise ValueError('nonempty aligned energy/distance and integer run-minimum vectors required')
    if not all(np.all(np.isfinite(x)) for x in [e,d,[cutoff,z]]) or np.any(d<0) or cutoff<=0 or z<=0 or np.any((m<0)|(m>=len(e))):
        raise ValueError('invalid certification domain')
    count=int(np.sum(d[m]<cutoff)); n=len(m); phat=count/n; den=1+z*z/n
    center=(phat+z*z/(2*n))/den
    half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
    sound=float(d[int(np.argmin(e))]<cutoff)
    return np.array([count,max(0.,center-half),min(1.,center+half),sound])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'e=np.array([2.,-1.,0.]);d=np.array([3.,.7,1.8]);m=np.array([1,1,0,2,1])\n',
      'call': 'certify_benchmark_setup(e,d,m,2.,1.8)',
      'gold_call': '_oracle_certify_benchmark_setup(e,d,m,2.,1.8)'},
     {'setup': 'import numpy as np\ne=np.array([0.,1.]);d=np.array([2.,3.]);m=np.zeros(12,dtype=int)\n',
      'call': 'certify_benchmark_setup(e,d,m,2.,1.96)',
      'gold_call': '_oracle_certify_benchmark_setup(e,d,m,2.,1.96)'},
     {'setup': 'import numpy as np\ne=np.array([-1.,-1.]);d=np.array([4.,0.]);m=np.ones(7,dtype=int)\n',
      'call': 'certify_benchmark_setup(e,d,m,1.,2.)',
      'gold_call': '_oracle_certify_benchmark_setup(e,d,m,1.,2.)'},
     {'setup': 'import numpy as np\n'
               'e=np.zeros(2);d=np.zeros(2);m=np.array([2])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        certify_benchmark_setup(e,d,m,1.,2.)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_certify_benchmark_setup(e,d,m,1.,2.)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
