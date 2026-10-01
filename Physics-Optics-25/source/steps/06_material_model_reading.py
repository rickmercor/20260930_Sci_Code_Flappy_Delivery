"""
Conventional reading of homogeneous nondepolarizing material media.

At every point the model is the homogeneous unit-path medium whose
differential matrix is the material generator, so all material anisotropies
act simultaneously and no depolarizing generator is present.

Returns
-------
A finite (n,) array of model readings in radians.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def material_model_reading(material_stack: np.ndarray) -> np.ndarray:
    """Read the conventional circular retardance of each material model medium.

    material_stack must be a finite real array of shape (n,4,4) with n>=1.
    For every generator construct the homogeneous unit-path Mueller medium
    it generates, divide that matrix by its own [0,0] entry, and apply the
    step-05 plus 45 degree reading. Raise ValueError for a contract
    violation, a nonfinite model matrix, a model [0,0] entry at most 1e-12,
    or an output with hypot(Q,U)<=1e-12. Return principal values in
    (-pi,pi] with shape (n,).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_material_model_reading(material_stack: np.ndarray) -> np.ndarray:
    import warnings

    from scipy.linalg import expm

    material = np.asarray(material_stack, dtype=float)
    if material.ndim != 3 or material.shape[0] < 1 or material.shape[1:] != (4, 4):
        raise ValueError("material_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(material)):
        raise ValueError("material_stack must be finite")
    model = np.empty_like(material)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for index in range(material.shape[0]):
            model[index] = expm(material[index])
    if not np.all(np.isfinite(model)):
        raise ValueError("model matrices must be finite")
    scales = model[:, 0, 0]
    if np.any(scales <= 1e-12):
        raise ValueError("model M[0,0] must exceed 1e-12")
    model = model / scales[:, None, None]
    return _oracle_conventional_circular_reading(model)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'material_model_reading(x)',
      'gold_call': '_oracle_material_model_reading(x)'},
     {'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'material_model_reading(strong)',
      'gold_call': '_oracle_material_model_reading(strong)'},
     {'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'material_model_reading(pure)',
      'gold_call': '_oracle_material_model_reading(pure)'},
     {'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(material_model_reading, x[0])',
      'gold_call': 'rejects(_oracle_material_model_reading, x[0])'},
     {'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(material_model_reading, x[:, :3])',
      'gold_call': 'rejects(_oracle_material_model_reading, x[:, :3])'},
     {'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(material_model_reading, x[:0])',
      'gold_call': 'rejects(_oracle_material_model_reading, x[:0])'},
     {'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(material_model_reading, bad)',
      'gold_call': 'rejects(_oracle_material_model_reading, bad)'},
     {'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(material_model_reading, zero_linear)',
      'gold_call': 'rejects(_oracle_material_model_reading, zero_linear)'},
     {'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(material_model_reading, zero_scale)',
      'gold_call': 'rejects(_oracle_material_model_reading, zero_scale)'},
     {'setup': 'import numpy as np\n'
               'def gen(ld=0.,ldp=0.,lb=0.,lbp=0.,cd=0.,cb=0.):\n'
               ' return '
               'np.array([[0,-ld,-ldp,cd],[-ld,0,cb,lbp],[-ldp,-cb,0,-lb],[cd,-lbp,lb,0]],dtype=float)\n'
               'x=np.array([gen(.08*np.cos(.4*j),.05*np.sin(.3*j),.5+.03*j,-.2+.02*j,.01,-.3+.12*j) '
               'for j in range(6)])\n'
               'strong=np.array([gen(.31,-.12,1.02,.44,.004,-.002),gen(-.27,.2,-.93,.61,0.,.0019)])\n'
               'pure=np.array([gen(cb=.7),gen(ld=.3),gen(lbp=.6)])\n'
               'bad=x.copy(); bad[0,1,2]=np.nan\n'
               'zero_linear=gen(lb=np.pi/2)[None]\n'
               'zero_scale=np.zeros((1,4,4)); zero_scale[0,0,1]=np.pi/2; zero_scale[0,1,0]=-np.pi/2\n'
               'huge=np.zeros((1,4,4)); huge[0,0,0]=1e308\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(material_model_reading, huge)',
      'gold_call': 'rejects(_oracle_material_model_reading, huge)'}]
