"""
Fitted ancestry representation

Fit the clustering representation to the rounded local profiles using the finite full covariance mixture schedule supplied with the task.

profiles is a finite array with shape (W,Q,J), where J is at least 2. Every entry lies from 0 through 1, and each rounded panel probability row is within J times 0.0005 of unit total. starts has shape (R,K), where R is at least 1 and K is at least 2. Every row contains K distinct valid flattened record indices. cycles is a positive integer, and ridge is finite and greater than 0. Use the stated initialization, update cycles, regularization, and restart tie conventions. A component with effective mass less than or equal to 1e-12 is invalid.

Return fitted coordinate means with shape (K,J-1), record responsibilities with shape (W times Q,K), all final log likelihoods with shape (R,), the selected start index, and the omitted original coordinate index. Preserve the component order of the selected start. An exact tie for the omitted coordinate selects the lowest coordinate index.

Raise ValueError if any input is outside the stated domain.

Returns
-------
Tuple: means (K,J-1), responsibilities (W*Q,K), likelihoods (R,), winner int, omitted coordinate int.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_profiles(profiles: np.ndarray, starts: np.ndarray, cycles: int = 80, ridge: \
    float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    (
        'Tuple: means (K,J-1), responsibilities (W*Q,K), likelihood'
        's (R,), winner int, omitted coordinate int. Raises ValueEr'
        'ror for inputs outside the stated domain.'
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

def _coordinates(profiles):
    a = np.asarray(profiles, float).reshape(-1, np.shape(profiles)[-1])
    drop = int(a.sum(axis=0).argmax())
    return np.delete(a, drop, axis=1), drop

def _fit(x, starts, cycles=60, ridge=1e-6):
    x = np.asarray(x, float)
    starts = np.asarray(starts, int)
    n, d = x.shape; k = starts.shape[1]
    v = x-x.mean(axis=0)
    base = v.T@v/n+ridge*np.eye(d)
    def evaluate(pi, means, covs):
        logp = np.empty((n,k))
        for a in range(k):
            v = x-means[a]
            sign, ld = np.linalg.slogdet(covs[a])
            if sign <= 0: raise ValueError('non-positive covariance')
            quad = np.einsum('ij,ji->i',v,np.linalg.solve(covs[a],v.T))
            logp[:,a] = math.log(pi[a])-.5*(d*math.log(2*math.pi)+ld+quad)
        norm = logsumexp(logp,axis=1)
        return np.exp(logp-norm[:,None]),float(norm.sum())
    fits = []
    for start in starts:
        pi = np.full(k,1./k); means = x[start].copy()
        covs = np.repeat(base[None],k,axis=0)
        for _ in range(cycles):
            resp, _ = evaluate(pi,means,covs)
            nk = resp.sum(axis=0)
            if np.any(nk <= 1e-12): raise ValueError('empty numerical component')
            pi = nk/n; means = resp.T@x/nk[:,None]
            for a in range(k):
                v = x-means[a]
                covs[a] = (v.T*resp[:,a])@v/nk[a]+ridge*np.eye(d)
        resp,ll = evaluate(pi,means,covs)
        fits.append((ll,means.copy(),resp.copy()))
    ll = np.array([v[0] for v in fits])
    # Task tie tolerance makes numerically identical restarts reproducible.
    eligible = np.flatnonzero(ll >= ll.max()-1e-9)
    win = int(eligible[0]); _, means, resp = fits[win]
    return means, resp, ll, win

def _oracle_fit_profiles(profiles: np.ndarray, starts: np.ndarray, cycles: int = 80, \
    ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    import math
    import numpy as np
    from scipy.special import logsumexp
    profiles = _array(profiles, 3); st = _array(starts, 2)
    cycles = float(_array(cycles, 0)); ridge = float(_array(ridge, 0))
    if not (profiles.shape[2] >= 2 and np.all((profiles >= 0) & (profiles <= 1))):
        raise ValueError('input outside the declared domain')
    if not (np.allclose(profiles.sum(2), 1, atol=.0005*profiles.shape[2]+1e-9)):
        raise ValueError('input outside the declared domain')
    if not (st.shape[1] >= 2 and np.all(st == np.floor(st))):
        raise ValueError('input outside the declared domain')
    if not (np.all((st >= 0) & (st < profiles.shape[0]*profiles.shape[1]))):
        raise ValueError('input outside the declared domain')
    if not (all(len(set(row)) == len(row) for row in st)):
        raise ValueError('input outside the declared domain')
    if not (cycles >= 1 and int(cycles) == cycles and np.isfinite(ridge) and ridge > 0):
        raise ValueError('input outside the declared domain')
    cycles = int(cycles)

    x, drop = _coordinates(profiles)
    means, resp, ll, win = _fit(x,starts,cycles,ridge)
    return means, resp, ll, win, drop

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
        'import numpy as np\np=np.array([[[.10,.55,.35],[.12,.56,.32'
        '],[.35,.50,.15]],[[.32,.51,.17],[.11,.60,.29],[.34,.49,.17'
        ']]]); st=np.array([[0,2],[1,3]])'
    ), 'call': '_pack(fit_profiles(p,st,12,1e-4))', 'gold_call': \
        '_pack(_oracle_fit_profiles(p,st,12,1e-4))'}, {'setup': (
        'import numpy as np\np=np.array([[[.2,.8],[.3,.7],[.6,.4],[.'
        '7,.3]]]); st=np.array([[0,3]])'
    ), 'call': '_pack(fit_profiles(p,st,1,1e-3))', 'gold_call': \
        '_pack(_oracle_fit_profiles(p,st,1,1e-3))'}, {'setup': (
        'import numpy as np\np=np.array([[[.55,.20,.25],[.50,.25,.25'
        '],[.45,.30,.25]],[[.25,.50,.25],[.20,.55,.25],[.30,.45,.25'
        ']]]); st=np.array([[0,4],[2,3]])'
    ), 'call': '_pack(fit_profiles(p,st,25,1e-6))', 'gold_call': \
        '_pack(_oracle_fit_profiles(p,st,25,1e-6))'}, {'setup': (
        'import numpy as np\np=np.array([[[.2,.8],[.3,.7],[.6,.4]]])'
        '; st=np.array([[0,9]])'
    ), 'call': '_error_code(fit_profiles,p,st,2,1e-5)', 'gold_call': \
        '_error_code(_oracle_fit_profiles,p,st,2,1e-5)'}]
