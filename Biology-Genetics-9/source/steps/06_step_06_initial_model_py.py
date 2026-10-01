"""
Initial ancestry parameters

Obtain the initial ancestry specific copying matrix and switching rates from fitted component means, record responsibilities, and aligned local rate records.

means is a finite nonnegative array with shape (K,J-1), and every row sums to at most one. resp is a finite nonnegative array with shape (N,K), and every row sums to one. rates contains N finite values greater than 0 in record order. drop is an integer from 0 through J minus 1. Every hard component must be represented. The reconstructed copying matrix must contain only values greater than 0. Apply the ancestry label convention stated in the task Scientific Background to both returned arrays.

Return the row stochastic copying matrix with shape (K,J) and the ancestry rate array with shape (K,) in units per Morgan.

Raise ValueError if any input is outside the stated domain

Returns
-------
Tuple: copying matrix (K,J), ancestry rates (K,) per Morgan.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def initial_model(means: np.ndarray, resp: np.ndarray, drop: int, rates: np.ndarray) \
    -> tuple[np.ndarray, np.ndarray]:
    (
        'Tuple: copying matrix (K,J), ancestry rates (K,) per Morga'
        'n. Raises ValueError for inputs outside the stated domain.'
    )
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_initial_model(
    means: np.ndarray,
    resp: np.ndarray,
    drop: int,
    rates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    import math
    import numpy as np
    from scipy.special import logsumexp

    means = _array(means, 2)
    resp = _array(resp, 2)
    rv = _array(rates)
    drop = float(_array(drop, 0))

    if not (
        means.shape[0] == resp.shape[1]
        and resp.shape[0] == rv.size
    ):
        raise ValueError('input outside the declared domain')

    if not (
        np.all(means >= 0)
        and np.all(means.sum(axis=1) <= 1)
    ):
        raise ValueError('input outside the declared domain')

    if not (
        np.all(resp >= 0)
        and np.allclose(resp.sum(axis=1), 1, atol=1e-9)
    ):
        raise ValueError('input outside the declared domain')

    if not (
        np.all(rv > 0)
        and 0 <= drop <= means.shape[1]
        and int(drop) == drop
    ):
        raise ValueError('input outside the declared domain')

    if len(set(resp.argmax(axis=1))) != len(means):
        raise ValueError('input outside the declared domain')

    drop = int(drop)

    means = np.asarray(means)
    resp = np.asarray(resp)

    p = np.insert(
        means,
        drop,
        1.0 - means.sum(axis=1),
        axis=1
    )

    labels = resp.argmax(axis=1)
    rv = np.asarray(rates).ravel()

    rho = np.array([
        rv[labels == i].mean()
        for i in range(len(p))
    ])

    if (
        not np.isfinite(rho).all()
        or np.any(rho <= 0)
        or np.any(p <= 0)
    ):
        raise ValueError('invalid model parameters')

    # Ancestry IDs are canonicalized by complete P rows.
    order = np.array(
        sorted(
            range(len(p)),
            key=lambda i: tuple(p[i])
        ),
        dtype=int
    )

    return p[order], rho[order]

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
    return [
        {
            'setup': (
                'import numpy as np\n'
                'me=np.array([[.1,.2],[.4,.3]])\n'
                're=np.array([[.8,.2],[.4,.6],[.51,.49],[.1,.9]])\n'
                'rates=np.array([100.,700.,300.,900.])'
            ),
            'call': '_pack(initial_model(me,re,1,rates))',
            'gold_call': '_pack(_oracle_initial_model(me,re,1,rates))',
        },
        {
            'setup': (
                'import numpy as np\n'
                'me=np.array([[.4],[.2]])\n'
                're=np.array([[.5,.5],[.2,.8]])\n'
                'rates=np.array([200.,800.])'
            ),
            'call': '_pack(initial_model(me,re,0,rates))',
            'gold_call': '_pack(_oracle_initial_model(me,re,0,rates))',
        },
        {
            'setup': (
                'import numpy as np\n'
                'me=np.array([[.2,.3],[.2,.1],[.1,.7]])\n'
                're=np.eye(3)\n'
                'rates=np.array([[250.,1250.,650.]])'
            ),
            'call': '_pack(initial_model(me,re,2,rates))',
            'gold_call': '_pack(_oracle_initial_model(me,re,2,rates))',
        },
        {
            'setup': (
                'import numpy as np\n'
                'me=np.array([[.2],[.7]])\n'
                're=np.array([[.9,.1],[.8,.2]])\n'
                'rates=np.array([5.,6.])'
            ),
            'call': '_error_code(initial_model,me,re,0,rates)',
            'gold_call': '_error_code(_oracle_initial_model,me,re,0,rates)',
        },
    ]
