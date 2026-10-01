"""
Continuous normalization data for a separable broadband realization.

Continuous normalization data for a separable broadband realization.

For phase array p of shape (3,m), S_a(t)=sum_{k=1}^m cos(k t+p[a,k-1]).
Let P=S_x(x)S_y(y) and T=P S_z(z). All extrema refer to the full
periodic square or cube [0, 2 pi]^d, independently of any observation slab.

Returns
-------
Float array [[min(P),max(P)],[min(T),max(T)]].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seed_bounds(phases):
    """Return float array [[min(P),max(P)],[min(T),max(T)]].

    phases is a finite real (3,m) array, m>=1, in radians; mode numbers
    are 1,...,m and all amplitudes equal one. Extrema are continuous.

    Raises
    ------
    ValueError
        If phases has the wrong shape, is empty, or carries nonfinite values.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _check_phases(phases):
    p=np.asarray(phases,dtype=float)
    if p.ndim!=2 or p.shape[0]!=3 or p.shape[1]<1 or not np.isfinite(p).all():
        raise ValueError('phases must be finite with shape (3,m), m>=1')
    return p


def _check_bounds(bounds):
    b=np.asarray(bounds,dtype=float)
    if b.shape!=(2,2) or not np.isfinite(b).all() or np.any(b[:,1]<=b[:,0]):
        raise ValueError('bounds must contain two finite increasing pairs')
    return b


def _canonical_phases():
    return np.array([[5.93,2.26,4.93,3.72,1.85],
                     [5.80,5.46,2.29,6.11,1.41],
                     [5.06,4.28,2.96,0.19,5.62]])


def _wave(t, phases, derivative=0):
    k=np.arange(1,len(phases)+1)
    return np.sum(k**derivative*np.cos(np.asarray(t)[...,None]*k+phases+derivative*np.pi/2),axis=-1)


def _oracle_seed_bounds(phases):
    phases=_check_phases(phases);m=phases.shape[1];ranges=[]
    for row in phases:
        c=np.zeros(2*m+1,dtype=complex)
        for k,angle in enumerate(row,1):
            c[m+k]+=1j*k*np.exp(1j*angle)/2
            c[m-k]-=1j*k*np.exp(-1j*angle)/2
        roots=np.polynomial.polynomial.polyroots(c)
        angles=np.angle(roots[np.abs(np.abs(roots)-1)<1e-7])%(2*np.pi)
        if len(angles)<2:
            raise ValueError('stationary roots unresolved')
        values=_wave(np.r_[angles,0.],row)
        ranges.append([values.min(),values.max()])
    pp=np.array([x*y for x in ranges[0] for y in ranges[1]])
    tt=np.array([v*z for v in pp for z in ranges[2]])
    return np.array([[pp.min(),pp.max()],[tt.min(),tt.max()]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\np = np.random.default_rng(41).uniform(-2, 2, (3, 4))\n',
      'call': 'seed_bounds(p.copy())',
      'gold_call': '_oracle_seed_bounds(p.copy())'},
     {'setup': 'import numpy as np\np = np.array([[0.3], [1.2], [-0.7]])\n',
      'call': 'seed_bounds(p.copy())',
      'gold_call': '_oracle_seed_bounds(p.copy())'},
     {'setup': 'import numpy as np\n'
               'p = np.zeros((2, 3))\n'
               '\n'
               'def case():\n'
               '    try:\n'
               '        seed_bounds(p.copy())\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n'
               '\n'
               'def gold_case():\n'
               '    try:\n'
               '        _oracle_seed_bounds(p.copy())\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': 'case()',
      'gold_call': 'gold_case()'}]
