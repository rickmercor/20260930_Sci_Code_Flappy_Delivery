"""
Clustering continuity evidence

Obtain the lag one ancestry continuity diagnostics from integer hard assignments. The calculation must preserve target identity across adjacent windows.

labels contains W times Q entries in window major, target minor order. W is at least 3, Q is at least 1, K is at least 2, and every label is an integer from 0 through K minus 1.

Return a binary64 array of length K plus 1. The first K entries are the ancestry diagnostics in component order, and the last entry is their minimum.

Raise ValueError if any input is outside the stated domain or if zero variance makes any diagnostic undefined.

Returns
-------
Binary64 array of length K+1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def ancestry_continuity(labels: np.ndarray, n_windows: int, n_targets: int, k: int) -> \
    np.ndarray:
    (
        'Binary64 array of length K+1. Raises ValueError for inputs'
        ' outside the stated domain.'
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

def _oracle_ancestry_continuity(labels: np.ndarray, n_windows: int, n_targets: int, k: \
    int) -> np.ndarray:
    import math
    import numpy as np
    from scipy.special import logsumexp
    values = _array(labels)
    n_windows, n_targets, k = [float(_array(v, 0)) for v in (n_windows, n_targets, k)]
    if not (n_windows >= 3 and n_targets >= 1 and k >= 2):
        raise ValueError('input outside the declared domain')
    if not (all(int(v) == v for v in (n_windows, n_targets, k))):
        raise ValueError('input outside the declared domain')
    if not (values.size == n_windows*n_targets and np.all(values == np.floor(values))):
        raise ValueError('input outside the declared domain')
    if not (np.all((values >= 0) & (values < k))):
        raise ValueError('input outside the declared domain')
    n_windows, n_targets, k = map(int, (n_windows, n_targets, k))

    lab = np.asarray(labels,int).reshape(n_windows,n_targets)
    out = []
    for a in range(k):
        x = (lab[:-1].T.ravel()==a).astype(float)
        y = (lab[1:].T.ravel()==a).astype(float)
        xx=x-x.mean(); yy=y-y.mean()
        den=math.sqrt(float(xx@xx)*float(yy@yy))
        if den == 0.: raise ValueError('undefined continuity statistic')
        out.append(float(xx@yy)/den)
    return np.r_[out,min(out)]

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
    return [{'setup': 'import numpy as np\nlab=np.array([0,1,0,1, 0,1,0,1, 1,1,0,0, 1,0,1,0])', \
        'call': '_pack(ancestry_continuity(lab,4,4,2))', 'gold_call': \
        '_pack(_oracle_ancestry_continuity(lab,4,4,2))'}, {'setup': \
        'import numpy as np\nlab=np.array([0,1,0,1, 1,0,1,0, 0,1,0,1])', 'call': \
        '_pack(ancestry_continuity(lab,3,4,2))', 'gold_call': \
        '_pack(_oracle_ancestry_continuity(lab,3,4,2))'}, {'setup': (
        'import numpy as np\nlab=np.array([0,1,2,0,1,2, 0,2,2,1,0,1, 1,2,0,1,'
        '0,2, 2,1,0,1,2,0])'
    ), 'call': '_pack(ancestry_continuity(lab,4,6,3))', 'gold_call': \
        '_pack(_oracle_ancestry_continuity(lab,4,6,3))'}, {'setup': \
        'import numpy as np\nlab=np.array([0]*6)', 'call': \
        '_error_code(ancestry_continuity,lab,3,2,2)', 'gold_call': \
        '_error_code(_oracle_ancestry_continuity,lab,3,2,2)'}]
