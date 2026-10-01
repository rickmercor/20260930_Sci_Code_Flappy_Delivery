"""
Closed winding of a planar vector sampled around a ring.

A planar vector sampled in azimuthal order around a closed ring accumulates a
total rotation. Principal angle increments between consecutive samples,
including the increment from the last sample back to the first, measure it.

Returns
-------
The signed winding in full turns.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def closed_vector_winding(planar_vectors: np.ndarray, magnitude_floor: float = 1e-10) -> float:
    """Return the closed winding of an ordered planar-vector sequence.

    planar_vectors must be a finite real array of shape (n,2) with n>=3,
    whose rows (x_j, y_j) are consecutive samples around one closed ring.
    magnitude_floor must be a finite positive real number and not a Boolean.
    With z_j = x_j + i*y_j, sum the principal arguments in (-pi,pi] of
    z_(j+1)/z_j for j=0..n-1 with z_n = z_0, and divide the sum by 2*pi.
    Raise ValueError for any contract violation or when any |z_j| is at most
    magnitude_floor. Return the winding as a float in full turns.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_closed_vector_winding(planar_vectors: np.ndarray, magnitude_floor: float = 1e-10) -> float:
    vectors = np.asarray(planar_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2 or vectors.shape[0] < 3:
        raise ValueError("planar_vectors must have shape (n, 2) with n >= 3")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("planar_vectors must be finite")
    if isinstance(magnitude_floor, (bool, np.bool_)) or not isinstance(magnitude_floor, (int, float, np.integer, np.floating)):
        raise ValueError("magnitude_floor must be a real number")
    if not np.isfinite(magnitude_floor) or magnitude_floor <= 0:
        raise ValueError("magnitude_floor must be finite and positive")
    z = vectors[:, 0] + 1j * vectors[:, 1]
    if np.any(np.abs(z) <= magnitude_floor):
        raise ValueError("vector magnitude is at or below magnitude_floor")
    increments = np.angle(np.roll(z, -1) / z)
    increments = np.where(increments <= -np.pi, np.pi, increments)
    return float(np.sum(increments) / (2.0 * np.pi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nv=np.array([[-1.,0.],[1.,0.],[0.,1.]])\n',
      'call': 'closed_vector_winding(v)',
      'gold_call': '_oracle_closed_vector_winding(v)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'closed_vector_winding(two)',
      'gold_call': '_oracle_closed_vector_winding(two)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'closed_vector_winding(minus_two)',
      'gold_call': '_oracle_closed_vector_winding(minus_two)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'closed_vector_winding(static)',
      'gold_call': '_oracle_closed_vector_winding(static)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'closed_vector_winding(three.copy())',
      'gold_call': '_oracle_closed_vector_winding(three.copy())'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'closed_vector_winding(three[::-1].copy())',
      'gold_call': '_oracle_closed_vector_winding(three[::-1].copy())'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'closed_vector_winding(one)',
      'gold_call': '_oracle_closed_vector_winding(one)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'closed_vector_winding(3.0 * two, 0.002)',
      'gold_call': '_oracle_closed_vector_winding(3.0 * two, 0.002)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'closed_vector_winding(np.roll(minus_two, 5, axis=0))',
      'gold_call': '_oracle_closed_vector_winding(np.roll(minus_two, 5, axis=0))'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(closed_vector_winding, two[:, 0].copy())',
      'gold_call': 'rejects(_oracle_closed_vector_winding, two[:, 0].copy())'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(closed_vector_winding, np.hstack([two, two[:, :1]]))',
      'gold_call': 'rejects(_oracle_closed_vector_winding, np.hstack([two, two[:, :1]]))'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(closed_vector_winding, two[:2].copy())',
      'gold_call': 'rejects(_oracle_closed_vector_winding, two[:2].copy())'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(closed_vector_winding, nan.copy())',
      'gold_call': 'rejects(_oracle_closed_vector_winding, nan.copy())'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(closed_vector_winding, small.copy())',
      'gold_call': 'rejects(_oracle_closed_vector_winding, small.copy())'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(closed_vector_winding, two.copy(), 0.0)',
      'gold_call': 'rejects(_oracle_closed_vector_winding, two.copy(), 0.0)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(closed_vector_winding, two.copy(), np.nan)',
      'gold_call': 'rejects(_oracle_closed_vector_winding, two.copy(), np.nan)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(closed_vector_winding, two.copy(), True)',
      'gold_call': 'rejects(_oracle_closed_vector_winding, two.copy(), True)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([.03,.51,1.08,1.55,2.13,2.66,3.18,3.75,4.24,4.81,5.35,5.88])\n'
               'def ring(m,amp=.3,mod=.1,phase=.2,wob=0.):\n'
               ' t=m*p+phase+wob*np.sin(p)\n'
               ' r=amp*(1+mod*np.cos(p))\n'
               ' return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)\n'
               'two=ring(2); minus_two=ring(-2,phase=2.8)\n'
               'static=ring(0,phase=3.,wob=1.2)\n'
               'q=np.array([.1,.8,1.4,2.2,2.9,3.6,4.4,5.1,5.7])\n'
               'three=np.stack([np.cos(3*q+2.9),np.sin(3*q+2.9)],axis=1)*(.2+.05*np.sin(q))[:,None]\n'
               'one=np.array([[1.,0.],[-.4,.9],[-.5,-.8]])\n'
               'small=two.copy(); small[4]=[0.,0.]\n'
               'nan=two.copy(); nan[2,1]=np.nan\n'
               'def rejects(fn,*a):\n'
               ' try: fn(*a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(closed_vector_winding, two.copy(), 0.5)',
      'gold_call': 'rejects(_oracle_closed_vector_winding, two.copy(), 0.5)'}]
