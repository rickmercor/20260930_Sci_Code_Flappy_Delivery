"""
Recalibrated copying matrix

Apply the copying matrix update to smoothed ancestry state evidence from all supplied targets.

Gammas and panels are equally long, nonempty sequences. Every gamma entry is a finite nonnegative array with shape (M,K,H) whose joint probability mass sums to one at each marker. Its aligned panels entry is an integer array with shape (M,H) and values from 0 through J minus 1. old_p is a finite nonnegative row stochastic array with shape (K,J). Preserve the corresponding row of old_p when an ancestry has zero total posterior support.

Return a binary64 row stochastic array with shape (K,J). Raise ValueError if any input is outside the stated domain.

Returns
-------
Binary64 row-stochastic array (K,J).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def update_copying(gammas: list[np.ndarray], panels: list[np.ndarray], old_p: np. \
    ndarray) -> np.ndarray:
    (
        'Binary64 row-stochastic array (K,J). Raises ValueError for'
        ' inputs outside the stated domain.'
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

def _oracle_update_copying(gammas: list[np.ndarray], panels: list[np.ndarray], old_p: \
    np.ndarray) -> np.ndarray:
    import math
    import numpy as np
    from scipy.special import logsumexp
    old_p = _array(old_p, 2)
    if not (len(gammas) == len(panels) and len(gammas) > 0):
        raise ValueError('input outside the declared domain')
    if not (np.all(old_p >= 0) and np.allclose(old_p.sum(1), 1, atol=1e-9)):
        raise ValueError('input outside the declared domain')
    for a0, j0 in zip(gammas, panels):
        a0 = _array(a0, 3); j0 = _array(j0, 2)
        if not (a0.shape[0] == j0.shape[0] and a0.shape[2] == j0.shape[1]):
            raise ValueError('input outside the declared domain')
        if not (a0.shape[1] == len(old_p) and np.all(a0 >= 0)):
            raise ValueError('input outside the declared domain')
        if not (np.allclose(a0.sum((1,2)), 1, atol=1e-9)):
            raise ValueError('input outside the declared domain')
        if not (np.all(j0 == np.floor(j0)) and np.all((j0 >= 0) & (j0 < old_p.shape[1] \
            ))):
            raise ValueError('input outside the declared domain')

    out = np.zeros_like(np.asarray(old_p,float))
    for gamma, panel in zip(gammas,panels):
        gamma = np.asarray(gamma,float); panel = np.asarray(panel,int)
        for m in range(len(gamma)):
            for h in range(gamma.shape[2]):
                out[:,panel[m,h]] += gamma[m,:,h]
    for i in range(len(out)):
        total = out[i].sum()
        out[i] = old_p[i] if total == 0 else out[i]/total
    return out

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
        'import numpy as np\nga=[np.array([[[.1,.2],[.3,.4]],[[.2,.1'
        '],[.6,.1]]])]; js=[np.array([[0,1],[1,0]])]; old=np.array([[.4,.6],'
        '[.2,.8]])'
    ), 'call': '_pack(update_copying(ga,js,old))', 'gold_call': \
        '_pack(_oracle_update_copying(ga,js,old))'}, {'setup': (
        'import numpy as np\nga=[np.array([[[.4,.6],[0.,0.]]])]; js='
        '[np.array([[0,1]])]; old=np.array([[.4,.6],[.3,.7]])'
    ), 'call': '_pack(update_copying(ga,js,old))', 'gold_call': \
        '_pack(_oracle_update_copying(ga,js,old))'}, {'setup': (
        'import numpy as np\nga=[np.array([[[.8,.1],[.02,.08]]]),np.'
        'array([[[.01,.09],[.2,.7]],[[.05,.15],[.7,.1]]])]; js=[np.'
        'array([[2,0]]),np.array([[0,1],[2,1]])]; old=np.array([[.2,.3,.5],['
        '.4,.2,.4]])'
    ), 'call': '_pack(update_copying(ga,js,old))', 'gold_call': \
        '_pack(_oracle_update_copying(ga,js,old))'}, {'setup': (
        'import numpy as np\nga=[np.array([[[.1,.1],[.1,.1]]])]; js='
        '[np.array([[0,1]])]; old=np.array([[.4,.6],[.2,.8]])'
    ), 'call': '_error_code(update_copying,ga,js,old)', 'gold_call': \
        '_error_code(_oracle_update_copying,ga,js,old)'}]
