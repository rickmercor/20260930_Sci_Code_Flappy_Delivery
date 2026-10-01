"""
Reported local copying records

Convert aligned marker level panel probabilities and switching increments into retained local records using the task window, marker selection, rounding, precision, and chromosome start conventions.

Inputs are a finite, strictly increasing cM vector g with shape (M,), a nonnegative post array with shape (Q,M,J) whose panel probability rows sum to one, a finite nonnegative tau array with shape (Q,M), a finite width greater than 0 in cM, and an integer min_markers greater than or equal to 2. Retained windows must contain at least min_markers and have positive genetic span.

Order retained records by genomic window and then by target. Return profiles with shape (W,Q,J), rates with shape (W,Q) in units per Morgan, original window identifiers with shape (W,), and half open marker spans with shape (W,2).

Raise ValueError if any input is outside the stated domain or no window survives.

Returns
-------
Tuple: profiles (W,Q,J), rates (W,Q) per Morgan, window IDs (W,), spans (W,2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def window_records(g: np.ndarray, post: np.ndarray, tau: np.ndarray, width: float, \
    min_markers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        'Tuple: profiles (W,Q,J), rates (W,Q) per Morgan, window ID'
        's (W,), spans (W,2). Raises ValueError for inputs outside '
        'the stated domain.'
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

def _oracle_window_records(g: np.ndarray, post: np.ndarray, tau: np.ndarray, width: \
    float, min_markers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import math
    import numpy as np
    from scipy.special import logsumexp
    g = _array(g, 1); post = _array(post, 3); tau = _array(tau, 2)
    width = float(_array(width, 0)); min_markers = float(_array(min_markers, 0))
    if not (post.shape[:2] == tau.shape and post.shape[1] == len(g)):
        raise ValueError('input outside the declared domain')
    if not (np.all(np.diff(g) > 0) and len(g) >= 2):
        raise ValueError('input outside the declared domain')
    if not (np.all(post >= 0) and np.allclose(post.sum(2), 1, atol=1e-9)):
        raise ValueError('input outside the declared domain')
    if not (np.all(tau >= -1e-9) and np.isfinite(width) and width > 0):
        raise ValueError('input outside the declared domain')
    if not (min_markers >= 2 and int(min_markers) == min_markers):
        raise ValueError('input outside the declared domain')
    min_markers = int(min_markers)

    g = np.asarray(g, float)
    post, tau = np.asarray(post), np.asarray(tau)
    ends = []
    threshold = float(g[0]+width)
    for m in range(len(g)):
        if g[m] > threshold:
            ends.append(m)
            threshold += width
    ends.append(len(g))
    probs, rates, ids, spans = [], [], [], []
    start = 0
    for w, end in enumerate(ends):
        if end-start >= min_markers:
            pp, rr = [], []
            for q in range(post.shape[0]):
                v = [sum(float(x) for x in post[q,start:end,j])/(end-start) for j in \
                    range(post.shape[2])]
                pp.append([float(np.rint(x*1000.))/1000. for x in v])
                rv = sum(float(x) for x in tau[q,start:end])*100./(g[end-1]-g[start])
                # The benchmark keeps the rate in binary64.
                rr.append(float(rv))
            probs.append(pp); rates.append(rr); ids.append(w); spans.append((start,end))
        start = end
    if not (len(probs) > 0):
        raise ValueError('input outside the declared domain')
    return np.array(probs), np.array(rates), np.array(ids,int), np.array(spans,int)

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
        'import numpy as np\ng=np.arange(9)*.15; post=np.tile([.2234'
        ',.3123,.4643],(2,9,1)); tau=np.tile(np.r_[0.,np.arange(1,9'
        ')*.03],(2,1))'
    ), 'call': '_pack(window_records(g,post,tau,.5,2))', 'gold_call': \
        '_pack(_oracle_window_records(g,post,tau,.5,2))'}, {'setup': (
        'import numpy as np\ng=np.array([0.,.25,.5,.75,1.]); post=np'
        '.tile([.3335,.6665],(1,5,1)); tau=np.array([[0.,.2,.3,.4,.'
        '5]])'
    ), 'call': '_pack(window_records(g,post,tau,.5,2))', 'gold_call': \
        '_pack(_oracle_window_records(g,post,tau,.5,2))'}, {'setup': (
        'import numpy as np\ng=np.array([0.,.1,.6,1.1,1.2,1.3,1.8,1.'
        '9]); post=np.tile([.7,.2,.1],(1,8,1)); tau=np.array([[0.,.'
        '1,.2,.3,.4,.5,.6,.7]])'
    ), 'call': '_pack(window_records(g,post,tau,.5,2))', 'gold_call': \
        '_pack(_oracle_window_records(g,post,tau,.5,2))'}, {'setup': (
        'import numpy as np\ng=np.array([0.,.1]); post=np.array([[[.4,.6],[.4,'
        '.6]]]); tau=np.array([[0.,.1]])'
    ), 'call': '_error_code(window_records,g,post,tau,.5,3)', 'gold_call': \
        '_error_code(_oracle_window_records,g,post,tau,.5,3)'}]
