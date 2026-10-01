"""
Conventional circular-birefringence reading under plus 45 degree illumination.

A conventional chiroptical reading interprets the azimuth change of plus 45
degree linear light as if circular birefringence alone produced it.

Returns
-------
A finite (n,) array of principal readings in radians.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def conventional_circular_reading(normalized_stack: np.ndarray) -> np.ndarray:
    """Read the conventional circular retardance of every normalized matrix.

    normalized_stack must be a finite real array of shape (n,4,4) with n>=1
    whose [0,0] entries each lie within 1e-8 of one. Illuminate with the
    Stokes vector (1,0,1,0). The reading is twice the signed rotation from
    the output azimuth back to the plus 45 degree input azimuth, equal to
    zero for an unchanged plus 45 degree output and reported as a principal
    value in (-pi,pi]. Raise ValueError for any contract violation or when
    the output Q and U satisfy hypot(Q,U)<=1e-12. Return shape (n,).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_conventional_circular_reading(normalized_stack: np.ndarray) -> np.ndarray:
    matrices = np.asarray(normalized_stack, dtype=float)
    if matrices.ndim != 3 or matrices.shape[0] < 1 or matrices.shape[1:] != (4, 4):
        raise ValueError("normalized_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("normalized_stack must be finite")
    if np.any(np.abs(matrices[:, 0, 0] - 1.0) > 1e-8):
        raise ValueError("matrices must be intensity-normalized")
    output = matrices @ np.array([1.0, 0.0, 1.0, 0.0])
    q = output[:, 1]
    u = output[:, 2]
    if np.any(np.hypot(q, u) <= 1e-12):
        raise ValueError("output linear polarization must be nonzero")
    values = np.arctan2(q, u)
    return np.where(values <= -np.pi, np.pi, values)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'from scipy.linalg import expm\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'def norm(ms):\n'
               ' ms=np.array(ms); return ms/ms[:,:1,:1]\n'
               'rot=norm([expm(gen(cb=v)) for v in (-2.9,-.8,0.,.35,1.1,3.05)])\n'
               'mixed=norm([expm(gen(ld=.21,ldp=-.07,lb=.46,lbp=.12,cd=.02,cb=.05)),expm(gen(ld=-.15,ldp=.18,lb=-.3,lbp=.41,cd=-.01,cb=-.2)),expm(gen(ldp=.3,lbp=.9))])\n'
               'cut=np.eye(4); cut[1,1]=cut[2,2]=-1.; cut[1,2]=0.; cut[2,1]=0.; cut=cut[None]\n'
               'bad=rot.copy(); bad[0,0,0]=.9\n'
               'circular=norm([expm(gen(lb=np.pi/2))])\n'
               'nan=rot.copy(); nan[2,3,1]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'conventional_circular_reading(rot)',
      'gold_call': '_oracle_conventional_circular_reading(rot)'},
     {'setup': 'import numpy as np\n'
               'from scipy.linalg import expm\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'def norm(ms):\n'
               ' ms=np.array(ms); return ms/ms[:,:1,:1]\n'
               'rot=norm([expm(gen(cb=v)) for v in (-2.9,-.8,0.,.35,1.1,3.05)])\n'
               'mixed=norm([expm(gen(ld=.21,ldp=-.07,lb=.46,lbp=.12,cd=.02,cb=.05)),expm(gen(ld=-.15,ldp=.18,lb=-.3,lbp=.41,cd=-.01,cb=-.2)),expm(gen(ldp=.3,lbp=.9))])\n'
               'cut=np.eye(4); cut[1,1]=cut[2,2]=-1.; cut[1,2]=0.; cut[2,1]=0.; cut=cut[None]\n'
               'bad=rot.copy(); bad[0,0,0]=.9\n'
               'circular=norm([expm(gen(lb=np.pi/2))])\n'
               'nan=rot.copy(); nan[2,3,1]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'conventional_circular_reading(mixed)',
      'gold_call': '_oracle_conventional_circular_reading(mixed)'},
     {'setup': 'import numpy as np\n'
               'from scipy.linalg import expm\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'def norm(ms):\n'
               ' ms=np.array(ms); return ms/ms[:,:1,:1]\n'
               'rot=norm([expm(gen(cb=v)) for v in (-2.9,-.8,0.,.35,1.1,3.05)])\n'
               'mixed=norm([expm(gen(ld=.21,ldp=-.07,lb=.46,lbp=.12,cd=.02,cb=.05)),expm(gen(ld=-.15,ldp=.18,lb=-.3,lbp=.41,cd=-.01,cb=-.2)),expm(gen(ldp=.3,lbp=.9))])\n'
               'cut=np.eye(4); cut[1,1]=cut[2,2]=-1.; cut[1,2]=0.; cut[2,1]=0.; cut=cut[None]\n'
               'bad=rot.copy(); bad[0,0,0]=.9\n'
               'circular=norm([expm(gen(lb=np.pi/2))])\n'
               'nan=rot.copy(); nan[2,3,1]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'conventional_circular_reading(np.concatenate([cut,mixed[:1]]))',
      'gold_call': '_oracle_conventional_circular_reading(np.concatenate([cut,mixed[:1]]))'},
     {'setup': 'import numpy as np\n'
               'from scipy.linalg import expm\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'def norm(ms):\n'
               ' ms=np.array(ms); return ms/ms[:,:1,:1]\n'
               'rot=norm([expm(gen(cb=v)) for v in (-2.9,-.8,0.,.35,1.1,3.05)])\n'
               'mixed=norm([expm(gen(ld=.21,ldp=-.07,lb=.46,lbp=.12,cd=.02,cb=.05)),expm(gen(ld=-.15,ldp=.18,lb=-.3,lbp=.41,cd=-.01,cb=-.2)),expm(gen(ldp=.3,lbp=.9))])\n'
               'cut=np.eye(4); cut[1,1]=cut[2,2]=-1.; cut[1,2]=0.; cut[2,1]=0.; cut=cut[None]\n'
               'bad=rot.copy(); bad[0,0,0]=.9\n'
               'circular=norm([expm(gen(lb=np.pi/2))])\n'
               'nan=rot.copy(); nan[2,3,1]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(conventional_circular_reading, rot[0])',
      'gold_call': 'rejects(_oracle_conventional_circular_reading, rot[0])'},
     {'setup': 'import numpy as np\n'
               'from scipy.linalg import expm\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'def norm(ms):\n'
               ' ms=np.array(ms); return ms/ms[:,:1,:1]\n'
               'rot=norm([expm(gen(cb=v)) for v in (-2.9,-.8,0.,.35,1.1,3.05)])\n'
               'mixed=norm([expm(gen(ld=.21,ldp=-.07,lb=.46,lbp=.12,cd=.02,cb=.05)),expm(gen(ld=-.15,ldp=.18,lb=-.3,lbp=.41,cd=-.01,cb=-.2)),expm(gen(ldp=.3,lbp=.9))])\n'
               'cut=np.eye(4); cut[1,1]=cut[2,2]=-1.; cut[1,2]=0.; cut[2,1]=0.; cut=cut[None]\n'
               'bad=rot.copy(); bad[0,0,0]=.9\n'
               'circular=norm([expm(gen(lb=np.pi/2))])\n'
               'nan=rot.copy(); nan[2,3,1]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(conventional_circular_reading, rot[:, :3])',
      'gold_call': 'rejects(_oracle_conventional_circular_reading, rot[:, :3])'},
     {'setup': 'import numpy as np\n'
               'from scipy.linalg import expm\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'def norm(ms):\n'
               ' ms=np.array(ms); return ms/ms[:,:1,:1]\n'
               'rot=norm([expm(gen(cb=v)) for v in (-2.9,-.8,0.,.35,1.1,3.05)])\n'
               'mixed=norm([expm(gen(ld=.21,ldp=-.07,lb=.46,lbp=.12,cd=.02,cb=.05)),expm(gen(ld=-.15,ldp=.18,lb=-.3,lbp=.41,cd=-.01,cb=-.2)),expm(gen(ldp=.3,lbp=.9))])\n'
               'cut=np.eye(4); cut[1,1]=cut[2,2]=-1.; cut[1,2]=0.; cut[2,1]=0.; cut=cut[None]\n'
               'bad=rot.copy(); bad[0,0,0]=.9\n'
               'circular=norm([expm(gen(lb=np.pi/2))])\n'
               'nan=rot.copy(); nan[2,3,1]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(conventional_circular_reading, rot[:0])',
      'gold_call': 'rejects(_oracle_conventional_circular_reading, rot[:0])'},
     {'setup': 'import numpy as np\n'
               'from scipy.linalg import expm\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'def norm(ms):\n'
               ' ms=np.array(ms); return ms/ms[:,:1,:1]\n'
               'rot=norm([expm(gen(cb=v)) for v in (-2.9,-.8,0.,.35,1.1,3.05)])\n'
               'mixed=norm([expm(gen(ld=.21,ldp=-.07,lb=.46,lbp=.12,cd=.02,cb=.05)),expm(gen(ld=-.15,ldp=.18,lb=-.3,lbp=.41,cd=-.01,cb=-.2)),expm(gen(ldp=.3,lbp=.9))])\n'
               'cut=np.eye(4); cut[1,1]=cut[2,2]=-1.; cut[1,2]=0.; cut[2,1]=0.; cut=cut[None]\n'
               'bad=rot.copy(); bad[0,0,0]=.9\n'
               'circular=norm([expm(gen(lb=np.pi/2))])\n'
               'nan=rot.copy(); nan[2,3,1]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(conventional_circular_reading, bad)',
      'gold_call': 'rejects(_oracle_conventional_circular_reading, bad)'},
     {'setup': 'import numpy as np\n'
               'from scipy.linalg import expm\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'def norm(ms):\n'
               ' ms=np.array(ms); return ms/ms[:,:1,:1]\n'
               'rot=norm([expm(gen(cb=v)) for v in (-2.9,-.8,0.,.35,1.1,3.05)])\n'
               'mixed=norm([expm(gen(ld=.21,ldp=-.07,lb=.46,lbp=.12,cd=.02,cb=.05)),expm(gen(ld=-.15,ldp=.18,lb=-.3,lbp=.41,cd=-.01,cb=-.2)),expm(gen(ldp=.3,lbp=.9))])\n'
               'cut=np.eye(4); cut[1,1]=cut[2,2]=-1.; cut[1,2]=0.; cut[2,1]=0.; cut=cut[None]\n'
               'bad=rot.copy(); bad[0,0,0]=.9\n'
               'circular=norm([expm(gen(lb=np.pi/2))])\n'
               'nan=rot.copy(); nan[2,3,1]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(conventional_circular_reading, circular)',
      'gold_call': 'rejects(_oracle_conventional_circular_reading, circular)'},
     {'setup': 'import numpy as np\n'
               'from scipy.linalg import expm\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'def norm(ms):\n'
               ' ms=np.array(ms); return ms/ms[:,:1,:1]\n'
               'rot=norm([expm(gen(cb=v)) for v in (-2.9,-.8,0.,.35,1.1,3.05)])\n'
               'mixed=norm([expm(gen(ld=.21,ldp=-.07,lb=.46,lbp=.12,cd=.02,cb=.05)),expm(gen(ld=-.15,ldp=.18,lb=-.3,lbp=.41,cd=-.01,cb=-.2)),expm(gen(ldp=.3,lbp=.9))])\n'
               'cut=np.eye(4); cut[1,1]=cut[2,2]=-1.; cut[1,2]=0.; cut[2,1]=0.; cut=cut[None]\n'
               'bad=rot.copy(); bad[0,0,0]=.9\n'
               'circular=norm([expm(gen(lb=np.pi/2))])\n'
               'nan=rot.copy(); nan[2,3,1]=np.nan\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(conventional_circular_reading, nan)',
      'gold_call': 'rejects(_oracle_conventional_circular_reading, nan)'}]
