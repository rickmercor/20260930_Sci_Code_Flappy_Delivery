"""
Physical differential decomposition of logarithmic Mueller generators.

The differential Mueller study cited by the task separates each logarithmic
generator into a material-anisotropy generator and a depolarizing generator
using the metric structure of Stokes space.

Returns
-------
A (2,n,4,4) array with material generators first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lorentz_decomposition(logarithm_stack: np.ndarray) -> np.ndarray:
    """Separate material and depolarizing differential generators.

    logarithm_stack must be a finite real array of shape (n,4,4) with n>=1.
    Apply the physically valid differential decomposition used by the cited
    study to every generator. Return a finite (2,n,4,4) array in lane order
    (material, depolarization) whose two lanes sum to the input. Raise
    ValueError for any contract violation.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_lorentz_decomposition(logarithm_stack: np.ndarray) -> np.ndarray:
    generators = np.asarray(logarithm_stack, dtype=float)
    if generators.ndim != 3 or generators.shape[0] < 1 or generators.shape[1:] != (4, 4):
        raise ValueError("logarithm_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(generators)):
        raise ValueError("logarithm_stack must be finite")
    metric = np.diag([1.0, -1.0, -1.0, -1.0])
    adjoint = np.einsum("ab,ncb,cd->nad", metric, generators, metric)
    material = 0.5 * (generators - adjoint)
    depolarization = generators - material
    return np.stack((material, depolarization))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3003)\n'
               'x=rng.normal(scale=.2,size=(6,4,4))\n'
               'diag=np.diag([0.,-.11,-.23,-.31])[None]\n'
               'ld,ldp,lb,lbp,cd,cb=.13,-.07,.42,.18,.03,-.09\n'
               'pure=np.array([[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]],dtype=float)\n'
               'mixed=pure+diag+np.array([[[0,.02,-.01,.03],[-.02,0,.04,0],[.01,.04,0,-.05],[-.03,0,-.05,0]]])\n'
               'bad=x.copy(); bad[1,2,0]=np.inf\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'lorentz_decomposition(x)',
      'gold_call': '_oracle_lorentz_decomposition(x)'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3003)\n'
               'x=rng.normal(scale=.2,size=(6,4,4))\n'
               'diag=np.diag([0.,-.11,-.23,-.31])[None]\n'
               'ld,ldp,lb,lbp,cd,cb=.13,-.07,.42,.18,.03,-.09\n'
               'pure=np.array([[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]],dtype=float)\n'
               'mixed=pure+diag+np.array([[[0,.02,-.01,.03],[-.02,0,.04,0],[.01,.04,0,-.05],[-.03,0,-.05,0]]])\n'
               'bad=x.copy(); bad[1,2,0]=np.inf\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'lorentz_decomposition(np.concatenate([pure,diag]))',
      'gold_call': '_oracle_lorentz_decomposition(np.concatenate([pure,diag]))'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3003)\n'
               'x=rng.normal(scale=.2,size=(6,4,4))\n'
               'diag=np.diag([0.,-.11,-.23,-.31])[None]\n'
               'ld,ldp,lb,lbp,cd,cb=.13,-.07,.42,.18,.03,-.09\n'
               'pure=np.array([[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]],dtype=float)\n'
               'mixed=pure+diag+np.array([[[0,.02,-.01,.03],[-.02,0,.04,0],[.01,.04,0,-.05],[-.03,0,-.05,0]]])\n'
               'bad=x.copy(); bad[1,2,0]=np.inf\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'lorentz_decomposition(mixed)',
      'gold_call': '_oracle_lorentz_decomposition(mixed)'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3003)\n'
               'x=rng.normal(scale=.2,size=(6,4,4))\n'
               'diag=np.diag([0.,-.11,-.23,-.31])[None]\n'
               'ld,ldp,lb,lbp,cd,cb=.13,-.07,.42,.18,.03,-.09\n'
               'pure=np.array([[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]],dtype=float)\n'
               'mixed=pure+diag+np.array([[[0,.02,-.01,.03],[-.02,0,.04,0],[.01,.04,0,-.05],[-.03,0,-.05,0]]])\n'
               'bad=x.copy(); bad[1,2,0]=np.inf\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(lorentz_decomposition, x[0])',
      'gold_call': 'rejects(_oracle_lorentz_decomposition, x[0])'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3003)\n'
               'x=rng.normal(scale=.2,size=(6,4,4))\n'
               'diag=np.diag([0.,-.11,-.23,-.31])[None]\n'
               'ld,ldp,lb,lbp,cd,cb=.13,-.07,.42,.18,.03,-.09\n'
               'pure=np.array([[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]],dtype=float)\n'
               'mixed=pure+diag+np.array([[[0,.02,-.01,.03],[-.02,0,.04,0],[.01,.04,0,-.05],[-.03,0,-.05,0]]])\n'
               'bad=x.copy(); bad[1,2,0]=np.inf\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(lorentz_decomposition, x[:, :3])',
      'gold_call': 'rejects(_oracle_lorentz_decomposition, x[:, :3])'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3003)\n'
               'x=rng.normal(scale=.2,size=(6,4,4))\n'
               'diag=np.diag([0.,-.11,-.23,-.31])[None]\n'
               'ld,ldp,lb,lbp,cd,cb=.13,-.07,.42,.18,.03,-.09\n'
               'pure=np.array([[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]],dtype=float)\n'
               'mixed=pure+diag+np.array([[[0,.02,-.01,.03],[-.02,0,.04,0],[.01,.04,0,-.05],[-.03,0,-.05,0]]])\n'
               'bad=x.copy(); bad[1,2,0]=np.inf\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(lorentz_decomposition, x[:0])',
      'gold_call': 'rejects(_oracle_lorentz_decomposition, x[:0])'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(3003)\n'
               'x=rng.normal(scale=.2,size=(6,4,4))\n'
               'diag=np.diag([0.,-.11,-.23,-.31])[None]\n'
               'ld,ldp,lb,lbp,cd,cb=.13,-.07,.42,.18,.03,-.09\n'
               'pure=np.array([[[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]]],dtype=float)\n'
               'mixed=pure+diag+np.array([[[0,.02,-.01,.03],[-.02,0,.04,0],[.01,.04,0,-.05],[-.03,0,-.05,0]]])\n'
               'bad=x.copy(); bad[1,2,0]=np.inf\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(lorentz_decomposition, bad)',
      'gold_call': 'rejects(_oracle_lorentz_decomposition, bad)'}]
