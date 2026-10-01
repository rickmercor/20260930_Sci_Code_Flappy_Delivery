"""
Intensity normalization of a stack of complete Mueller matrices.

Every measured Mueller matrix is divided by its own M[0,0] element before a
differential generator is formed.

Returns
-------
A finite normalized array with the same (n,4,4) shape.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normalize_mueller_stack(mueller_stack: np.ndarray) -> np.ndarray:
    """Divide every complete Mueller matrix by its own M[0,0] element.

    mueller_stack must be a finite real array of shape (n,4,4) with n>=1.
    Raise ValueError for any other shape, for a nonfinite entry, when any
    M[0,0] is at most 1e-12, or when the normalized stack is not finite.
    Return a float array of shape (n,4,4) whose [0,0] entries equal one.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_normalize_mueller_stack(mueller_stack: np.ndarray) -> np.ndarray:
    matrices = np.asarray(mueller_stack, dtype=float)
    if matrices.ndim != 3 or matrices.shape[0] < 1 or matrices.shape[1:] != (4, 4):
        raise ValueError("mueller_stack must have shape (n, 4, 4) with n >= 1")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("mueller_stack must be finite")
    scales = matrices[:, 0, 0]
    if np.any(scales <= 1e-12):
        raise ValueError("every M[0,0] must exceed 1e-12")
    with np.errstate(over="ignore", invalid="ignore"):
        result = matrices / scales[:, None, None]
    if not np.all(np.isfinite(result)):
        raise ValueError("normalized stack must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'normalize_mueller_stack(x)',
      'gold_call': '_oracle_normalize_mueller_stack(x)'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'normalize_mueller_stack(one)',
      'gold_call': '_oracle_normalize_mueller_stack(one)'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'normalize_mueller_stack(wide)',
      'gold_call': '_oracle_normalize_mueller_stack(wide)'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(normalize_mueller_stack, x[0])',
      'gold_call': 'rejects(_oracle_normalize_mueller_stack, x[0])'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(normalize_mueller_stack, x[:, :3])',
      'gold_call': 'rejects(_oracle_normalize_mueller_stack, x[:, :3])'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(normalize_mueller_stack, x[:0])',
      'gold_call': 'rejects(_oracle_normalize_mueller_stack, x[:0])'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(normalize_mueller_stack, nan)',
      'gold_call': 'rejects(_oracle_normalize_mueller_stack, nan)'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(normalize_mueller_stack, inf)',
      'gold_call': 'rejects(_oracle_normalize_mueller_stack, inf)'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(normalize_mueller_stack, zero)',
      'gold_call': 'rejects(_oracle_normalize_mueller_stack, zero)'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(normalize_mueller_stack, negative)',
      'gold_call': 'rejects(_oracle_normalize_mueller_stack, negative)'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(normalize_mueller_stack, floor)',
      'gold_call': 'rejects(_oracle_normalize_mueller_stack, floor)'},
     {'setup': 'import numpy as np\n'
               'base=np.array([[1.,-.12,.07,.02],[-.11,.93,.05,-.21],[.06,-.04,.88,.17],[.03,.24,-.15,.86]])\n'
               'twist=np.array([[0.,.03,-.02,.01],[.02,-.05,.04,.03],[-.01,.02,.06,-.04],[.02,-.03,.01,.05]])\n'
               'x=np.stack([(.62+.07*i)*(base+i*twist) for i in range(5)])\n'
               'one=np.array([[[3.5e-12,-1.1e-12,4e-13,2e-13],[-1e-12,3e-12,1e-13,-6e-13],[5e-13,-2e-13,2.9e-12,5e-13],[3e-13,7e-13,-4e-13,2.8e-12]]])\n'
               'nan=x.copy(); nan[3,2,1]=np.nan\n'
               'inf=x.copy(); inf[1,0,3]=np.inf\n'
               'zero=x.copy(); zero[2,0,0]=0.\n'
               'negative=x.copy(); negative[4,0,0]=-.7\n'
               'floor=x.copy(); floor[0,0,0]=1e-12\n'
               'overflow=x.copy(); overflow[1,0,0]=2e-12; overflow[1,2,2]=1e300\n'
               'wide=np.stack([x[0]*1e-3,x[1]*4.2e5,x[2]])\n'
               'def rejects(fn,a):\n'
               ' try: fn(a)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(normalize_mueller_stack, overflow)',
      'gold_call': 'rejects(_oracle_normalize_mueller_stack, overflow)'}]
