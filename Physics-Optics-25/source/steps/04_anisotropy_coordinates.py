"""
Six simultaneous anisotropy coordinates from material generators.

The public entry map fixes every sign and position for linear and circular
diattenuation and birefringence in the material generator.

Returns
-------
A finite (n,6) array ordered (LD, LD_prime, LB, LB_prime, CD, CB).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def anisotropy_coordinates(material_stack: np.ndarray) -> np.ndarray:
    """Read all six coordinates with the exact public entry map.

    material_stack must be a finite real array of shape (n,4,4) with n>=1,
    and every matrix must match the public material-generator entry
    structure within absolute tolerance 1e-8. Raise ValueError for any
    contract violation. Read LD=-M[0,1], LD_prime=-M[0,2], LB=M[3,2],
    LB_prime=M[1,3], CD=M[0,3] and CB=M[1,2], and return them as an (n,6)
    array in that order, in radians per unit path.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_anisotropy_coordinates(material_stack: np.ndarray) -> np.ndarray:
    material = np.asarray(material_stack, dtype=float)
    if material.ndim != 3 or material.shape[0] < 1 or material.shape[1:] != (4, 4):
        raise ValueError("material_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(material)):
        raise ValueError("material_stack must be finite")
    coordinates = np.stack(
        (-material[:, 0, 1], -material[:, 0, 2],
         material[:, 3, 2], material[:, 1, 3],
         material[:, 0, 3], material[:, 1, 2]),
        axis=-1,
    )
    ld, ldp, lb, lbp, cd, cb = np.moveaxis(coordinates, -1, 0)
    rebuilt = np.zeros_like(material)
    rebuilt[:, 0, 1] = rebuilt[:, 1, 0] = -ld
    rebuilt[:, 0, 2] = rebuilt[:, 2, 0] = -ldp
    rebuilt[:, 0, 3] = rebuilt[:, 3, 0] = cd
    rebuilt[:, 1, 2] = cb
    rebuilt[:, 2, 1] = -cb
    rebuilt[:, 1, 3] = lbp
    rebuilt[:, 3, 1] = -lbp
    rebuilt[:, 2, 3] = -lb
    rebuilt[:, 3, 2] = lb
    if not np.allclose(material, rebuilt, rtol=0.0, atol=1e-8):
        raise ValueError("material_stack violates the public entry map")
    return coordinates

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'def make(q):\n'
               ' x=np.zeros((len(q),4,4))\n'
               ' for i,(ld,ldp,lb,lbp,cd,cb) in enumerate(q):\n'
               '  x[i]=[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]\n'
               ' return x\n'
               'q=np.array([[.11,-.23,.37,-.05,.017,-.061],[-.29,.04,-.13,.52,-.008,.093],[.07,.19,-.44,-.31,.026,.0],[0.,-.15,.08,.21,-.033,-.12]])\n'
               'x=make(q); one=make(q[2:3])\n'
               'tiny=make(q[:2]); tiny[0,2,3]+=4e-9\n'
               'bad=x.copy(); bad[1,2,3]+=2e-5\n'
               'diag=x.copy(); diag[3,1,1]=-.02\n'
               'nan=x.copy(); nan[0,0,3]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'anisotropy_coordinates(x)',
      'gold_call': '_oracle_anisotropy_coordinates(x)'},
     {'setup': 'import numpy as np\n'
               'def make(q):\n'
               ' x=np.zeros((len(q),4,4))\n'
               ' for i,(ld,ldp,lb,lbp,cd,cb) in enumerate(q):\n'
               '  x[i]=[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]\n'
               ' return x\n'
               'q=np.array([[.11,-.23,.37,-.05,.017,-.061],[-.29,.04,-.13,.52,-.008,.093],[.07,.19,-.44,-.31,.026,.0],[0.,-.15,.08,.21,-.033,-.12]])\n'
               'x=make(q); one=make(q[2:3])\n'
               'tiny=make(q[:2]); tiny[0,2,3]+=4e-9\n'
               'bad=x.copy(); bad[1,2,3]+=2e-5\n'
               'diag=x.copy(); diag[3,1,1]=-.02\n'
               'nan=x.copy(); nan[0,0,3]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'anisotropy_coordinates(one)',
      'gold_call': '_oracle_anisotropy_coordinates(one)'},
     {'setup': 'import numpy as np\n'
               'def make(q):\n'
               ' x=np.zeros((len(q),4,4))\n'
               ' for i,(ld,ldp,lb,lbp,cd,cb) in enumerate(q):\n'
               '  x[i]=[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]\n'
               ' return x\n'
               'q=np.array([[.11,-.23,.37,-.05,.017,-.061],[-.29,.04,-.13,.52,-.008,.093],[.07,.19,-.44,-.31,.026,.0],[0.,-.15,.08,.21,-.033,-.12]])\n'
               'x=make(q); one=make(q[2:3])\n'
               'tiny=make(q[:2]); tiny[0,2,3]+=4e-9\n'
               'bad=x.copy(); bad[1,2,3]+=2e-5\n'
               'diag=x.copy(); diag[3,1,1]=-.02\n'
               'nan=x.copy(); nan[0,0,3]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'anisotropy_coordinates(tiny)',
      'gold_call': '_oracle_anisotropy_coordinates(tiny)'},
     {'setup': 'import numpy as np\n'
               'def make(q):\n'
               ' x=np.zeros((len(q),4,4))\n'
               ' for i,(ld,ldp,lb,lbp,cd,cb) in enumerate(q):\n'
               '  x[i]=[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]\n'
               ' return x\n'
               'q=np.array([[.11,-.23,.37,-.05,.017,-.061],[-.29,.04,-.13,.52,-.008,.093],[.07,.19,-.44,-.31,.026,.0],[0.,-.15,.08,.21,-.033,-.12]])\n'
               'x=make(q); one=make(q[2:3])\n'
               'tiny=make(q[:2]); tiny[0,2,3]+=4e-9\n'
               'bad=x.copy(); bad[1,2,3]+=2e-5\n'
               'diag=x.copy(); diag[3,1,1]=-.02\n'
               'nan=x.copy(); nan[0,0,3]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(anisotropy_coordinates, x[0])',
      'gold_call': 'rejects(_oracle_anisotropy_coordinates, x[0])'},
     {'setup': 'import numpy as np\n'
               'def make(q):\n'
               ' x=np.zeros((len(q),4,4))\n'
               ' for i,(ld,ldp,lb,lbp,cd,cb) in enumerate(q):\n'
               '  x[i]=[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]\n'
               ' return x\n'
               'q=np.array([[.11,-.23,.37,-.05,.017,-.061],[-.29,.04,-.13,.52,-.008,.093],[.07,.19,-.44,-.31,.026,.0],[0.,-.15,.08,.21,-.033,-.12]])\n'
               'x=make(q); one=make(q[2:3])\n'
               'tiny=make(q[:2]); tiny[0,2,3]+=4e-9\n'
               'bad=x.copy(); bad[1,2,3]+=2e-5\n'
               'diag=x.copy(); diag[3,1,1]=-.02\n'
               'nan=x.copy(); nan[0,0,3]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(anisotropy_coordinates, x[:, :3])',
      'gold_call': 'rejects(_oracle_anisotropy_coordinates, x[:, :3])'},
     {'setup': 'import numpy as np\n'
               'def make(q):\n'
               ' x=np.zeros((len(q),4,4))\n'
               ' for i,(ld,ldp,lb,lbp,cd,cb) in enumerate(q):\n'
               '  x[i]=[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]\n'
               ' return x\n'
               'q=np.array([[.11,-.23,.37,-.05,.017,-.061],[-.29,.04,-.13,.52,-.008,.093],[.07,.19,-.44,-.31,.026,.0],[0.,-.15,.08,.21,-.033,-.12]])\n'
               'x=make(q); one=make(q[2:3])\n'
               'tiny=make(q[:2]); tiny[0,2,3]+=4e-9\n'
               'bad=x.copy(); bad[1,2,3]+=2e-5\n'
               'diag=x.copy(); diag[3,1,1]=-.02\n'
               'nan=x.copy(); nan[0,0,3]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(anisotropy_coordinates, x[:0])',
      'gold_call': 'rejects(_oracle_anisotropy_coordinates, x[:0])'},
     {'setup': 'import numpy as np\n'
               'def make(q):\n'
               ' x=np.zeros((len(q),4,4))\n'
               ' for i,(ld,ldp,lb,lbp,cd,cb) in enumerate(q):\n'
               '  x[i]=[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]\n'
               ' return x\n'
               'q=np.array([[.11,-.23,.37,-.05,.017,-.061],[-.29,.04,-.13,.52,-.008,.093],[.07,.19,-.44,-.31,.026,.0],[0.,-.15,.08,.21,-.033,-.12]])\n'
               'x=make(q); one=make(q[2:3])\n'
               'tiny=make(q[:2]); tiny[0,2,3]+=4e-9\n'
               'bad=x.copy(); bad[1,2,3]+=2e-5\n'
               'diag=x.copy(); diag[3,1,1]=-.02\n'
               'nan=x.copy(); nan[0,0,3]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(anisotropy_coordinates, bad)',
      'gold_call': 'rejects(_oracle_anisotropy_coordinates, bad)'},
     {'setup': 'import numpy as np\n'
               'def make(q):\n'
               ' x=np.zeros((len(q),4,4))\n'
               ' for i,(ld,ldp,lb,lbp,cd,cb) in enumerate(q):\n'
               '  x[i]=[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]\n'
               ' return x\n'
               'q=np.array([[.11,-.23,.37,-.05,.017,-.061],[-.29,.04,-.13,.52,-.008,.093],[.07,.19,-.44,-.31,.026,.0],[0.,-.15,.08,.21,-.033,-.12]])\n'
               'x=make(q); one=make(q[2:3])\n'
               'tiny=make(q[:2]); tiny[0,2,3]+=4e-9\n'
               'bad=x.copy(); bad[1,2,3]+=2e-5\n'
               'diag=x.copy(); diag[3,1,1]=-.02\n'
               'nan=x.copy(); nan[0,0,3]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(anisotropy_coordinates, diag)',
      'gold_call': 'rejects(_oracle_anisotropy_coordinates, diag)'},
     {'setup': 'import numpy as np\n'
               'def make(q):\n'
               ' x=np.zeros((len(q),4,4))\n'
               ' for i,(ld,ldp,lb,lbp,cd,cb) in enumerate(q):\n'
               '  x[i]=[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]\n'
               ' return x\n'
               'q=np.array([[.11,-.23,.37,-.05,.017,-.061],[-.29,.04,-.13,.52,-.008,.093],[.07,.19,-.44,-.31,.026,.0],[0.,-.15,.08,.21,-.033,-.12]])\n'
               'x=make(q); one=make(q[2:3])\n'
               'tiny=make(q[:2]); tiny[0,2,3]+=4e-9\n'
               'bad=x.copy(); bad[1,2,3]+=2e-5\n'
               'diag=x.copy(); diag[3,1,1]=-.02\n'
               'nan=x.copy(); nan[0,0,3]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(anisotropy_coordinates, nan)',
      'gold_call': 'rejects(_oracle_anisotropy_coordinates, nan)'}]
