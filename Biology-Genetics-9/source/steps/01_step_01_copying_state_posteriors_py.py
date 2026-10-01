"""
Copying-state posterior evidence

A composite reference state retains its identity along the chromosome even when its reference-panel label changes. Obtain the ancestry-agnostic state and same-state adjacent-marker posterior probabilities under the source model and the task numerical conventions. Inputs are a finite strictly increasing cM vector g of length M, a binary mismatch array (M,H), H >= 2, integer nref >= 2, and ne > 0. M may equal 1. Return gamma and same, each (M,H): gamma[m,h] is the smoothed state probability, and same[m,h] is the joint probability of state h at markers m-1 and m. At m=0, same[0] equals gamma[0]. Raise ValueError outside this domain.

Returns
-------
Tuple of two binary64 arrays, each shape (M,H).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def copying_state_posteriors(g: np.ndarray, mismatch: np.ndarray, nref: int, ne: float \
    ) -> tuple[np.ndarray, np.ndarray]:
    (
        'Tuple of two binary64 arrays, each shape (M,H). Raises Val'
        'ueError for inputs outside the stated domain.'
    )
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def _oracle_copying_state_posteriors(g: np.ndarray, mismatch: np.ndarray, nref: int, \
    ne: float) -> tuple[np.ndarray, np.ndarray]:
    import math
    import numpy as np
    from scipy.special import logsumexp
    g = _array(g, 1); z = _array(mismatch, 2)
    nref = float(_array(nref, 0)); ne = float(_array(ne, 0))
    if not (len(g) == z.shape[0] and z.shape[1] >= 2):
        raise ValueError('input outside the declared domain')
    if not (np.all(np.diff(g) > 0) and np.all((z == 0) | (z == 1))):
        raise ValueError('input outside the declared domain')
    if not (np.isfinite(nref) and nref >= 2 and int(nref) == nref):
        raise ValueError('input outside the declared domain')
    if not (np.isfinite(ne) and ne > 0):
        raise ValueError('input outside the declared domain')
    nref = int(nref)

    g = np.asarray(g, float)
    z = np.asarray(mismatch, int)
    mcount, hcount = z.shape
    lam = 1.0 / (math.log(nref) + 0.5)
    theta = lam / (2.0 * (lam + nref))
    em = np.where(z, theta, 1.0-theta)
    r = -np.expm1(-0.04*ne/nref*np.r_[0.,np.diff(g)])
    f = np.empty_like(em); b = np.ones_like(em)
    f[0] = em[0]/em[0].sum()
    for m in range(1,mcount):
        f[m] = em[m]*((1-r[m])*f[m-1]+r[m]/hcount)
        f[m] /= f[m].sum()
    for m in range(mcount-2,-1,-1):
        v = b[m+1]*em[m+1]
        b[m] = (1-r[m+1])*v+r[m+1]/hcount*v.sum()
        b[m] /= b[m].sum()
    gamma = f*b
    gamma /= gamma.sum(axis=1)[:,None]
    same = np.empty_like(gamma)
    same[0] = gamma[0]
    for m in range(1,mcount):
        v = b[m]*em[m]
        den = np.dot((1-r[m])*f[m-1]+r[m]/hcount,v)
        same[m] = f[m-1]*((1-r[m])+r[m]/hcount)*v/den
    return gamma, same

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def _pack(value):
    values = value if isinstance(value, tuple) else (value,)
    pieces = []
    for v in values:
        a = np.asarray(v, dtype=float)
        pieces.append(np.r_[float(a.ndim), a.shape, a.ravel()])
    return np.concatenate(pieces)

def _error_code(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

def test_cases():
    return [{'setup': (
        'import numpy as np\ng=np.array([0.,.03,.17,.22]); z=np.array([[0,1,0],[1,0,1],'
        '[0,0,1],[1,0,0]])'
    ), 'call': '_pack(copying_state_posteriors(g,z,120,20000))', 'gold_call': \
        '_pack(_oracle_copying_state_posteriors(g,z,120,20000))'}, {'setup': \
        'import numpy as np\ng=np.array([0.]); z=np.array([[0,1]])', 'call': \
        '_pack(copying_state_posteriors(g,z,12,10000))', 'gold_call': \
        '_pack(_oracle_copying_state_posteriors(g,z,12,10000))'}, {'setup': (
        'import numpy as np\ng=np.array([0.,1e-5,.2]); z=np.array([[1,1,1,1],[0,1,1,0],'
        '[1,1,0,1]])'
    ), 'call': '_pack(copying_state_posteriors(g,z,80,100000))', 'gold_call': \
        '_pack(_oracle_copying_state_posteriors(g,z,80,100000))'}, {'setup': \
        'import numpy as np\ng=np.array([0.,0.]); z=np.array([[0,1],[1,0]])', 'call': \
        '_error_code(copying_state_posteriors,g,z,100,10000)', 'gold_call': \
        '_error_code(_oracle_copying_state_posteriors,g,z,100,10000)'}]
