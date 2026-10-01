"""
Step 7: importance-reweight the proposal cloud for three populations.

Evaluate products of normalized Beta densities at nuisance columns `4:8`, using shape rows `((2,5),(5,2),(2,2),(2,5))`, `((2,2),(2,2),(2,2),(2,2))`, and `((5,2),(2,5),(5,2),(5,2))`. Repeat each run's density over all descendant rows and normalize separately over all impact rows. The uniform proposal density is one, so it contributes no varying denominator. The default retains fragment multiplicity. Only $balance_realizations=True$ subtracts the log count of each realization before normalization.

Returns
-------
Return a finite positive `(3,n_impacts)` array whose rows each sum to one, ordered by hypotheses `K=0,1,2` and the input impact-row order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 7: importance-reweight the proposal cloud for three populations."""
import math
import numpy as np
from scipy.special import betaln, logsumexp

def compute_population_weights(impact_cloud, balance_realizations=False):
    """Compute three normalized per-impact population-weight rows.

    Parameters
    ----------
    impact_cloud : array_like, shape (n,p), n>=1, p>=8
        Finite rows with nonnegative integer run labels in column 3 and
        nuisance coordinates strictly in (0,1) in columns 4:8.
    balance_realizations : bool
        A Python or NumPy boolean. If True, divide each run's row weights
        by its descendant count.

    Returns
    -------
    ndarray, shape (3,n)
        Positive normalized weights for K=0,1,2 in input impact order.

    Raises
    ------
    ValueError
        For malformed or nonfinite cloud, invalid run labels, or nuisance
        coordinates outside the open unit interval, or a nonboolean switch.
    RuntimeError
        If normalized weights underflow or become nonfinite.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 7: importance-reweight the proposal cloud for three populations."""
import math
import numpy as np
from scipy.special import betaln, logsumexp
_BETA_SHAPES = (((2.0, 5.0), (5.0, 2.0), (2.0, 2.0), (2.0, 5.0)), ((2.0, 2.0), (2.0, 2.0), (2.0, 2.0), (2.0, 2.0)), ((5.0, 2.0), (2.0, 5.0), (5.0, 2.0), (5.0, 2.0)))

def _oracle_compute_population_weights(impact_cloud, balance_realizations=False):
    if not isinstance(balance_realizations, (bool, np.bool_)):
        raise ValueError('balance_realizations must be boolean')
    cloud = np.asarray(impact_cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] < 8 or len(cloud) < 1:
        raise ValueError('impact_cloud must have at least eight columns and one row')
    if not np.all(np.isfinite(cloud)):
        raise ValueError('impact_cloud must be finite')
    nuisance = cloud[:, 4:8]
    if np.any(nuisance <= 0.0) or np.any(nuisance >= 1.0):
        raise ValueError('nuisance coordinates must lie strictly inside (0,1)')
    labels = cloud[:, 3]
    rounded = np.rint(labels).astype(int)
    if np.any(labels != rounded) or np.any(rounded < 0):
        raise ValueError('realization labels must be nonnegative integers')
    unique, inverse, counts = np.unique(rounded, return_inverse=True, return_counts=True)
    del unique
    rows = []
    for shapes in _BETA_SHAPES:
        log_weight = np.zeros(len(cloud), dtype=float)
        for coordinate, (a, b) in enumerate(shapes):
            value = nuisance[:, coordinate]
            log_weight += (a - 1.0) * np.log(value) + (b - 1.0) * np.log1p(-value) - betaln(a, b)
        if bool(balance_realizations):
            log_weight -= np.log(counts[inverse])
        rows.append(np.exp(log_weight - logsumexp(log_weight)))
    result = np.asarray(rows)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise RuntimeError('importance weights are not finite and positive')
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import math
import numpy as np
from scipy.special import betaln, logsumexp

def test_cases():
    """Exercise multiplicity retention, run balancing, and invalid support."""
    setup = 'cloud=np.array([[10.,-2.,.03,0.,.10,.20,.30,.40,2.],[11.,-1.,.08,0.,.10,.20,.30,.40,2.],[-4.,3.,.02,1.,.80,.70,.60,.50,5.],[7.,9.,.15,2.,.35,.65,.25,.75,3.]])'
    return [{'setup': setup, 'call': 'compute_population_weights(cloud)', 'gold_call': '_oracle_compute_population_weights(cloud)'}, {'setup': setup, 'call': 'compute_population_weights(cloud, True)', 'gold_call': '_oracle_compute_population_weights(cloud, True)'}, {'setup': 'def catches(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ncloud=np.array([[0.,0.,.1,0.,0.,.3,.4,.5,1.]])', 'call': 'catches(lambda: compute_population_weights(cloud))', 'gold_call': 'catches(lambda: _oracle_compute_population_weights(cloud))'}, {'setup': 'c=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,0.,.2,.3,.4,.5],[-5.,-2.,.09,1.,.7,.6,.5,.4]])\nc[:,3]=np.arange(3)\n', 'call': 'compute_population_weights(c,True)', 'gold_call': '_oracle_compute_population_weights(c,True)'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: compute_population_weights(np.ones((2,7))))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(np.ones((2,7))))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: compute_population_weights(np.ones((0,8))))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(np.ones((0,8))))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,0.,.2,.3,.4,.5],[-5.,-2.,.09,1.,.7,.6,.5,.4]])\nc[0,0]=np.nan\n', 'call': 'rejects(lambda: compute_population_weights(c))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,0.,.2,.3,.4,.5],[-5.,-2.,.09,1.,.7,.6,.5,.4]])\nc[0,0]=np.inf\n', 'call': 'rejects(lambda: compute_population_weights(c))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,0.,.2,.3,.4,.5],[-5.,-2.,.09,1.,.7,.6,.5,.4]])\nc[0,0]=-np.inf\n', 'call': 'rejects(lambda: compute_population_weights(c))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,0.,.2,.3,.4,.5],[-5.,-2.,.09,1.,.7,.6,.5,.4]])\nc[0,3]=-1.\n', 'call': 'rejects(lambda: compute_population_weights(c))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,0.,.2,.3,.4,.5],[-5.,-2.,.09,1.,.7,.6,.5,.4]])\nc[0,3]=.5\n', 'call': 'rejects(lambda: compute_population_weights(c))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,0.,.2,.3,.4,.5],[-5.,-2.,.09,1.,.7,.6,.5,.4]])\nc[0,4]=0.\n', 'call': 'rejects(lambda: compute_population_weights(c))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,0.,.2,.3,.4,.5],[-5.,-2.,.09,1.,.7,.6,.5,.4]])\nc[0,4]=1.\n', 'call': 'rejects(lambda: compute_population_weights(c))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,1.,.7,.6,.5,.4]])', 'call': 'rejects(lambda: compute_population_weights(c,balance_realizations=np.nan))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c,balance_realizations=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,1.,.7,.6,.5,.4]])', 'call': 'rejects(lambda: compute_population_weights(c,balance_realizations=np.inf))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c,balance_realizations=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.\n    return 0.\nc=np.array([[1.,2.,.03,0.,.2,.3,.4,.5],[3.,4.,.07,1.,.7,.6,.5,.4]])', 'call': 'rejects(lambda: compute_population_weights(c,balance_realizations=1))', 'gold_call': 'rejects(lambda: _oracle_compute_population_weights(c,balance_realizations=1))'}]
