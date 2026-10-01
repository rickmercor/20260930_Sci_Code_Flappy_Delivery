"""
Step 8: fit a weighted three-dimensional mass-location Gaussian mixture.

Fit a normalized weighted full-covariance Gaussian mixture in `z=((east-198000)/4000,(north-5000)/2000,ln(mass)-ln(.05))`. Deterministic initialization uses a stable mergesort by `.72*z0-.28*z1+.35*z2`, $array_split$ blocks, block-weighted means, uniform component weights, and the global weighted population covariance plus `diag(regularization)`. Run exactly `iterations` standard weighted EM updates, use determinant-normalized Gaussian densities and population covariances, add `diag(regularization)` after every covariance update, preserve all cross covariances and component order, and do not stop on convergence.

Returns
-------
Return a finite vector of length `13*components`, flattened as all component weights, all three-vector means in component-major order, then all `3x3` covariance matrices in component-major row-major order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 8: fit a weighted three-dimensional mass-location Gaussian mixture."""
import math
import numpy as np
from scipy.special import logsumexp

def fit_ground_gmm(impact_cloud, sample_weights, components=4, iterations=120, regularization=(0.0064, 0.01, 0.0049)):
    """Fit the deterministic weighted full-covariance Gaussian mixture.

    Parameters
    ----------
    impact_cloud : array_like, shape (n,p), n>=1, p>=3
        Finite rows starting with east, north, and positive mass.
    sample_weights : array_like, shape (n,)
        Finite nonnegative weights with positive total and positive weight
        in every deterministic initialization block.
    components, iterations : int
        Finite integers, not bool, with 1<=components<=n and iterations>=1.
    regularization : array_like, shape (3,)
        Finite positive diagonal covariance additions in standardized units.

    Returns
    -------
    ndarray, shape (13*components,)
        All weights, all means (component-major), then all covariances
        (component-major, row-major), preserving initialization order.

    Raises
    ------
    ValueError
        For any shape, finiteness, mass, weight, initialization-block, or
        numeric-control domain violation above.
    RuntimeError
        If a component becomes empty or a covariance is not positive definite.
    Notes
    -----
    Return the stated deterministic result to rtol=1e-10 and atol=1e-12.
    The supplied numerical controls define the discrete target.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8: fit a weighted three-dimensional mass-location Gaussian mixture."""
import math
import numpy as np
from scipy.special import logsumexp
_FEATURE_CENTER = np.array([198000.0, 5000.0, math.log(0.05)])
_FEATURE_SCALE = np.array([4000.0, 2000.0, 1.0])

def _oracle_fit_ground_gmm(impact_cloud, sample_weights, components=4, iterations=120, regularization=(0.0064, 0.01, 0.0049)):
    cloud = np.asarray(impact_cloud, dtype=float)
    weights_in = np.asarray(sample_weights, dtype=float)
    diagonal = np.asarray(regularization, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] < 3 or len(cloud) < 1:
        raise ValueError('impact_cloud must contain east, north, and mass')
    if not np.all(np.isfinite(cloud)) or np.any(cloud[:, 2] <= 0.0):
        raise ValueError('impact rows must be finite with positive mass')
    if weights_in.shape != (len(cloud),) or not np.all(np.isfinite(weights_in)):
        raise ValueError('sample_weights must be one finite value per impact')
    if np.any(weights_in < 0.0) or np.sum(weights_in) <= 0.0:
        raise ValueError('sample_weights must be nonnegative with positive sum')
    if not np.all(np.isfinite([components, iterations])):
        raise ValueError('GMM controls must be finite')
    if isinstance(components, (bool, np.bool_)) or int(components) != components:
        raise ValueError('components must be an integer')
    if isinstance(iterations, (bool, np.bool_)) or int(iterations) != iterations:
        raise ValueError('iterations must be an integer')
    components = int(components)
    iterations = int(iterations)
    if components < 1 or len(cloud) < components or iterations < 1:
        raise ValueError('GMM controls are invalid')
    if diagonal.shape != (3,) or not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError('regularization must be three positive diagonal entries')
    physical = np.column_stack((cloud[:, 0], cloud[:, 1], np.log(cloud[:, 2])))
    points = (physical - _FEATURE_CENTER) / _FEATURE_SCALE
    weights_in = weights_in / np.sum(weights_in)
    score = 0.72 * points[:, 0] - 0.28 * points[:, 1] + 0.35 * points[:, 2]
    groups = np.array_split(np.argsort(score, kind='mergesort'), components)
    means = np.empty((components, 3), dtype=float)
    for index, group in enumerate(groups):
        local_weight = weights_in[group]
        if np.sum(local_weight) <= 0.0:
            raise ValueError('every deterministic initialization block needs positive weight')
        means[index] = local_weight @ points[group] / np.sum(local_weight)
    global_mean = weights_in @ points
    centered = points - global_mean
    common_covariance = centered.T * weights_in @ centered
    regularizer = np.diag(diagonal)
    covariances = np.repeat((common_covariance + regularizer)[None, :, :], components, axis=0)
    mixture_weights = np.full(components, 1.0 / components)

    def log_density(values, mean, covariance):
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0:
            raise RuntimeError('non-positive GMM covariance')
        delta = values - mean
        quadratic = np.einsum('ni,ij,nj->n', delta, np.linalg.inv(covariance), delta)
        return -0.5 * (3.0 * math.log(2.0 * math.pi) + logdet + quadratic)
    for _ in range(iterations):
        log_components = np.column_stack([math.log(mixture_weights[j]) + log_density(points, means[j], covariances[j]) for j in range(components)])
        responsibilities = np.exp(log_components - logsumexp(log_components, axis=1)[:, None])
        effective = weights_in[:, None] * responsibilities
        counts = np.sum(effective, axis=0)
        if np.any(counts <= 0.0):
            raise RuntimeError('empty GMM component')
        mixture_weights = counts / np.sum(counts)
        means = effective.T @ points / counts[:, None]
        for j in range(components):
            delta = points - means[j]
            covariances[j] = delta.T * effective[:, j] @ delta / counts[j] + regularizer
    if np.any(np.linalg.eigvalsh(covariances) <= 0.0):
        raise RuntimeError('fitted covariance is not positive definite')
    return np.concatenate((mixture_weights, means.ravel(), covariances.ravel()))

# =============================================================================
# TEST CASES
# =============================================================================

import math
import numpy as np
from scipy.special import logsumexp

def test_cases():
    """Exercise asymmetric weighted fits and domain rejection."""
    first = 'cloud=np.array([[194000.,4300.,.018],[195500.,4700.,.031],[197000.,5200.,.047],[198600.,4500.,.082],[200200.,5600.,.14],[202400.,4900.,.23]]); w=np.array([.07,.11,.19,.23,.17,.23])'
    second = 'cloud=np.array([[193000.,4100.,.015],[194200.,5500.,.021],[196800.,4600.,.038],[198100.,5300.,.061],[199500.,4400.,.093],[201000.,5700.,.17],[203000.,4800.,.29]]); w=np.array([.03,.09,.14,.22,.18,.16,.18])'
    return [{'setup': first, 'call': 'fit_ground_gmm(cloud,w,2,35,(.02,.03,.04))', 'gold_call': '_oracle_fit_ground_gmm(cloud,w,2,35,(.02,.03,.04))'}, {'setup': second, 'call': 'fit_ground_gmm(cloud, w, 3, 27, (0.01, 0.015, 0.02))', 'gold_call': '_oracle_fit_ground_gmm(cloud, w, 3, 27, (0.01, 0.015, 0.02))'}, {'setup': 'def catches(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\ncloud=np.array([[1.,2.,-.1],[2.,3.,.2]]); w=np.array([.4,.6])', 'call': 'catches(lambda: fit_ground_gmm(cloud,w,2))', 'gold_call': 'catches(lambda: _oracle_fit_ground_gmm(cloud,w,2))'}, {'setup': 'c=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'fit_ground_gmm(c,w,1,1,(.03,.04,.05))', 'gold_call': '_oracle_fit_ground_gmm(c,w,1,1,(.03,.04,.05))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(np.ones((2,2)),[.4,.6]))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(np.ones((2,2)),[.4,.6]))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(np.ones((0,3)),[]))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(np.ones((0,3)),[]))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,[1.,2.]))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,[1.,2.]))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,[0.,0.,0.,0.]))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,[0.,0.,0.,0.]))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,[-1.,1.,1.,1.]))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,[-1.,1.,1.,1.]))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,[np.nan,1.,1.,1.]))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,[np.nan,1.,1.,1.]))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,[0.,0.,.4,.6],2))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,[0.,0.,.4,.6],2))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,5))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,5))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,0))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,0))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,True))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,True))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,2,0))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,2,0))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,2,True))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,2,True))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,2,3,[.1,.2]))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,2,3,[.1,.2]))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,2,3,[.1,0.,.2]))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,2,3,[.1,0.,.2]))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\nc[0,0]=np.nan\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,components=np.nan))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,components=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,iterations=np.nan))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,iterations=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,regularization=(.1,.2,np.nan)))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,regularization=(.1,.2,np.nan)))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\nc[0,0]=np.inf\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,components=np.inf))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,components=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,iterations=np.inf))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,iterations=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,regularization=(.1,.2,np.inf)))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,regularization=(.1,.2,np.inf)))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\nc[0,0]=-np.inf\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,components=-np.inf))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,components=-np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,iterations=-np.inf))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,iterations=-np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,regularization=(.1,.2,-np.inf)))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,regularization=(.1,.2,-np.inf)))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,components=1.5))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,components=1.5))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nc=np.array([[193000.,4300.,.02],[197000.,4900.,.04],[199000.,5500.,.08],[202000.,6100.,.17]]); w=np.array([.1,.2,.3,.4])\n', 'call': 'rejects(lambda: fit_ground_gmm(c,w,iterations=1.5))', 'gold_call': 'rejects(lambda: _oracle_fit_ground_gmm(c,w,iterations=1.5))'}]
