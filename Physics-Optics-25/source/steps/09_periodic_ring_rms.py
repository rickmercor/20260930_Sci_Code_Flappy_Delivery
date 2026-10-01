"""
Root mean square of a real response around an irregularly sampled closed ring.

Periodic trapezoid weights give every azimuth half of each adjacent gap,
including the gap that closes the ring through 2*pi.

Returns
-------
The weighted root mean square as a nonnegative float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def periodic_ring_rms(values: np.ndarray, azimuths_rad: np.ndarray) -> float:
    """Return the periodic-trapezoid root mean square of one ring response.

    values and azimuths_rad must be finite real one-dimensional arrays of the
    same length n>=3, with 0<=azimuths_rad[0]<...<azimuths_rad[-1]<2*pi.
    With phi_n = phi_0 + 2*pi, Delta_j = phi_(j+1) - phi_j for j=0..n-1 and
    w_j = (Delta_(j-1) + Delta_j)/2 using cyclic indices, return
    sqrt(sum_j w_j*values_j**2 / (2*pi)). Raise ValueError for any contract
    violation.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_periodic_ring_rms(values: np.ndarray, azimuths_rad: np.ndarray) -> float:
    response = np.asarray(values, dtype=float)
    azimuths = np.asarray(azimuths_rad, dtype=float)
    if response.ndim != 1 or azimuths.ndim != 1 or response.size != azimuths.size or response.size < 3:
        raise ValueError("values and azimuths_rad must be one-dimensional with equal length n >= 3")
    if not np.all(np.isfinite(response)) or not np.all(np.isfinite(azimuths)):
        raise ValueError("inputs must be finite")
    if azimuths[0] < 0 or azimuths[-1] >= 2 * np.pi or np.any(np.diff(azimuths) <= 0):
        raise ValueError("azimuths_rad must increase strictly inside [0, 2*pi)")
    gaps = np.diff(np.append(azimuths, azimuths[0] + 2 * np.pi))
    weights = 0.5 * (np.roll(gaps, 1) + gaps)
    return float(np.sqrt(np.sum(weights * response ** 2) / (2 * np.pi)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'periodic_ring_rms(f,p)',
      'gold_call': '_oracle_periodic_ring_rms(f,p)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'periodic_ring_rms(g, three)',
      'gold_call': '_oracle_periodic_ring_rms(g, three)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'periodic_ring_rms(flat, p)',
      'gold_call': '_oracle_periodic_ring_rms(flat, p)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'periodic_ring_rms(f[::2].copy(), p[::2].copy())',
      'gold_call': '_oracle_periodic_ring_rms(f[::2].copy(), p[::2].copy())'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'periodic_ring_rms((np.abs(f) ** 0.5).copy(), p.copy())',
      'gold_call': '_oracle_periodic_ring_rms((np.abs(f) ** 0.5).copy(), p.copy())'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(periodic_ring_rms, f, p[:-1])',
      'gold_call': 'rejects(_oracle_periodic_ring_rms, f, p[:-1])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(periodic_ring_rms, f[:2], p[:2])',
      'gold_call': 'rejects(_oracle_periodic_ring_rms, f[:2], p[:2])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(periodic_ring_rms, f[None], p[None])',
      'gold_call': 'rejects(_oracle_periodic_ring_rms, f[None], p[None])'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(periodic_ring_rms, nanf, p)',
      'gold_call': 'rejects(_oracle_periodic_ring_rms, nanf, p)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(periodic_ring_rms, f, badp)',
      'gold_call': 'rejects(_oracle_periodic_ring_rms, f, badp)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(periodic_ring_rms, f, late)',
      'gold_call': 'rejects(_oracle_periodic_ring_rms, f, late)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(periodic_ring_rms, f, early)',
      'gold_call': 'rejects(_oracle_periodic_ring_rms, f, early)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.07,.62,1.1,1.71,2.19,2.83,3.3,3.92,4.37,5.02,5.47,6.1])\n'
               'f=.2*np.cos(2*p+.4)-.07*np.sin(p)+.03\n'
               'three=np.array([.4,2.9,4.1])\n'
               'g=np.array([.5,-1.2,.3])\n'
               'flat=np.full(p.size,-.37)\n'
               'badp=p.copy(); badp[6]=badp[5]\n'
               'late=p.copy(); late[-1]=2*np.pi\n'
               'early=p.copy(); early[0]=-.01\n'
               'nanf=f.copy(); nanf[3]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(periodic_ring_rms, f, p[::-1])',
      'gold_call': 'rejects(_oracle_periodic_ring_rms, f, p[::-1])'}]
