"""
Obtain the source method's weak-association residual correlation estimate from aligned, marginally standardized Z scores, with the supplied weak-association cutoff.

weak_z is a finite M x R matrix and cutoff is positive. Follow the source's all-trait weak-association selection and standardized-trait second-moment convention. The selected rows must yield a positive-definite correlation matrix. A zero selected-row count or zero marginal second moment is invalid. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
R x R positive-definite correlation matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def estimate_residual_correlation(weak_z, cutoff) -> np.ndarray:
    """Obtain the source method's weak-association residual correlation estimate from aligned, marginally standardized Z scores, with the supplied weak-association cutoff.

    Return R x R positive-definite correlation matrix."""
    # Placeholder only; implement the operation described above.
    return np.zeros((np.shape(weak_z)[1], np.shape(weak_z)[1]), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, ndtr

def _bg11_array(value, ndim):
    a = np.asarray(value, dtype=float)
    if a.ndim != ndim or not np.all(np.isfinite(a)) or any(s == 0 for s in a.shape):
        raise ValueError('invalid array dimensions or nonfinite entries')
    return a

def _bg11_cov(value, size=None, pd=True):
    a = _bg11_array(value, 2)
    if a.shape[0] != a.shape[1] or (size is not None and a.shape != (size, size)):
        raise ValueError('invalid covariance shape')
    if not np.allclose(a, a.T, rtol=0, atol=1e-10):
        raise ValueError('covariance is not symmetric')
    eigen = np.linalg.eigvalsh(a)
    if (pd and eigen[0] <= 0) or (not pd and eigen[0] < -1e-10):
        raise ValueError('invalid covariance definiteness')
    return a

def _bg11_prob(value, size):
    a = _bg11_array(value, 1)
    if a.shape != (size,) or np.any(a <= 0) or not np.isclose(a.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError('expected a strictly positive probability vector')
    return a / a.sum()

def _bg11_prior(prior, r):
    a = _bg11_array(prior, 2)
    if a.shape[1] != 1+r*r:
        raise ValueError('invalid packed prior')
    w = _bg11_prob(a[:, 0], len(a))
    u = a[:, 1:].reshape(-1, r, r)
    for h in u:
        _bg11_cov(h, r, pd=False)
    return w, u

def _bg11_integer(value, lower=1):
    if not np.isscalar(value) or not np.isfinite(value) or value != int(value) or value < lower:
        raise ValueError('invalid integer')
    return int(value)

def _bg11_state(state):
    s = _bg11_array(state, 3)
    l, rows, width = s.shape
    r = (width-3)//2
    j = rows-1
    if r < 1 or width != 2*r+3 or j < 1:
        raise ValueError('invalid posterior state dimensions')
    a = s[:, :j, 0]
    f = s[:, :j, r+1:2*r+1]
    if np.any(a < 0) or not np.allclose(a.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError('invalid posterior probabilities')
    if np.any(f < -1e-12) or np.any(f > 1+1e-12) or np.any(s[:, j, 0] < 0):
        raise ValueError('invalid sign probabilities or scales')
    return s, l, j, r

def _oracle_estimate_residual_correlation(weak_z, cutoff):
    z = _bg11_array(weak_z, 2)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('cutoff must be positive')
    retained = z[np.max(np.abs(z), axis=1) < cutoff]
    if not len(retained):
        raise ValueError('empty weak association set')
    moment = retained.T @ retained / len(retained)
    if np.any(np.diag(moment) <= 0):
        raise ValueError('zero marginal second moment')
    v = moment / np.sqrt(np.outer(np.diag(moment), np.diag(moment)))
    return _bg11_cov(v, z.shape[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'weak=np.array([[.3,.8],[-.9,.2],[.7,-.6],[1.1,.4],[-.4,-.7],[2.,.1]])\n'
               'training=np.array([[2.7,1.8],[-1.8,-2.1],[3.4,-.4],[.1,2.8],[-2.4,1.1],[.5,-2.9]])\n'
               'v=np.array([[1.,.27],[.27,1.]])\n'
               'u=np.array([[[1.,0.],[0.,1.]],[[1.,1.],[1.,1.]],[[1.,-.7],[-.7,.49]]])\n'
               'prior=np.column_stack(([.21,.43,.36],u.reshape(3,4)))\n'
               'xx=np.array([[24.,11.,-3.],[11.,30.,2.],[-3.,2.,21.]])\n'
               'xy=np.array([[11.,7.],[8.,4.],[-1.,8.]])\n'
               'pi=np.array([.23,.51,.26]);d=np.diag(xx)\n'
               'grid=np.array([0.,.015,.06,.2,.7]);priority=np.array([30.,10.,20.])\n',
      'call': 'estimate_residual_correlation(weak, 2.)',
      'gold_call': '_oracle_estimate_residual_correlation(weak, 2.)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'weak=np.array([[.3,.8],[-.9,.2],[.7,-.6],[1.1,.4],[-.4,-.7],[2.,.1]])\n'
               'training=np.array([[2.7,1.8],[-1.8,-2.1],[3.4,-.4],[.1,2.8],[-2.4,1.1],[.5,-2.9]])\n'
               'v=np.array([[1.,.27],[.27,1.]])\n'
               'u=np.array([[[1.,0.],[0.,1.]],[[1.,1.],[1.,1.]],[[1.,-.7],[-.7,.49]]])\n'
               'prior=np.column_stack(([.21,.43,.36],u.reshape(3,4)))\n'
               'xx=np.array([[24.,11.,-3.],[11.,30.,2.],[-3.,2.,21.]])\n'
               'xy=np.array([[11.,7.],[8.,4.],[-1.,8.]])\n'
               'pi=np.array([.23,.51,.26]);d=np.diag(xx)\n'
               'grid=np.array([0.,.015,.06,.2,.7]);priority=np.array([30.,10.,20.])\n'
               'weak=weak[:-1]*.7\n',
      'call': 'estimate_residual_correlation(weak, 1.4)',
      'gold_call': '_oracle_estimate_residual_correlation(weak, 1.4)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'weak=np.array([[.3,.8],[-.9,.2],[.7,-.6],[1.1,.4],[-.4,-.7],[2.,.1]])\n'
               'training=np.array([[2.7,1.8],[-1.8,-2.1],[3.4,-.4],[.1,2.8],[-2.4,1.1],[.5,-2.9]])\n'
               'v=np.array([[1.,.27],[.27,1.]])\n'
               'u=np.array([[[1.,0.],[0.,1.]],[[1.,1.],[1.,1.]],[[1.,-.7],[-.7,.49]]])\n'
               'prior=np.column_stack(([.21,.43,.36],u.reshape(3,4)))\n'
               'xx=np.array([[24.,11.,-3.],[11.,30.,2.],[-3.,2.,21.]])\n'
               'xy=np.array([[11.,7.],[8.,4.],[-1.,8.]])\n'
               'pi=np.array([.23,.51,.26]);d=np.diag(xx)\n'
               'grid=np.array([0.,.015,.06,.2,.7]);priority=np.array([30.,10.,20.])\n'
               'weak=np.array([[.4,.7],[.4,-.7]])\n',
      'call': 'estimate_residual_correlation(weak, 2.)',
      'gold_call': '_oracle_estimate_residual_correlation(weak, 2.)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'weak=np.array([[.3,.8],[-.9,.2],[.7,-.6],[1.1,.4],[-.4,-.7],[2.,.1]])\n'
               'training=np.array([[2.7,1.8],[-1.8,-2.1],[3.4,-.4],[.1,2.8],[-2.4,1.1],[.5,-2.9]])\n'
               'v=np.array([[1.,.27],[.27,1.]])\n'
               'u=np.array([[[1.,0.],[0.,1.]],[[1.,1.],[1.,1.]],[[1.,-.7],[-.7,.49]]])\n'
               'prior=np.column_stack(([.21,.43,.36],u.reshape(3,4)))\n'
               'xx=np.array([[24.,11.,-3.],[11.,30.,2.],[-3.,2.,21.]])\n'
               'xy=np.array([[11.,7.],[8.,4.],[-1.,8.]])\n'
               'pi=np.array([.23,.51,.26]);d=np.diag(xx)\n'
               'grid=np.array([0.,.015,.06,.2,.7]);priority=np.array([30.,10.,20.])\n'
               'weak=np.array([[.2,1.1],[-.8,.5],[2.,.3],[-.4,-1.1],[-2.,.7]])\n',
      'call': 'estimate_residual_correlation(weak, 2.)',
      'gold_call': '_oracle_estimate_residual_correlation(weak, 2.)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'weak=np.array([[.3,.8],[-.9,.2],[.7,-.6],[1.1,.4],[-.4,-.7],[2.,.1]])\n'
               'training=np.array([[2.7,1.8],[-1.8,-2.1],[3.4,-.4],[.1,2.8],[-2.4,1.1],[.5,-2.9]])\n'
               'v=np.array([[1.,.27],[.27,1.]])\n'
               'u=np.array([[[1.,0.],[0.,1.]],[[1.,1.],[1.,1.]],[[1.,-.7],[-.7,.49]]])\n'
               'prior=np.column_stack(([.21,.43,.36],u.reshape(3,4)))\n'
               'xx=np.array([[24.,11.,-3.],[11.,30.,2.],[-3.,2.,21.]])\n'
               'xy=np.array([[11.,7.],[8.,4.],[-1.,8.]])\n'
               'pi=np.array([.23,.51,.26]);d=np.diag(xx)\n'
               'grid=np.array([0.,.015,.06,.2,.7]);priority=np.array([30.,10.,20.])\n'
               'weak=weak[:,[1,0]]\n',
      'call': 'estimate_residual_correlation(weak, 2.)',
      'gold_call': '_oracle_estimate_residual_correlation(weak, 2.)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'weak=np.array([[.3,.8],[-.9,.2],[.7,-.6],[1.1,.4],[-.4,-.7],[2.,.1]])\n'
               'training=np.array([[2.7,1.8],[-1.8,-2.1],[3.4,-.4],[.1,2.8],[-2.4,1.1],[.5,-2.9]])\n'
               'v=np.array([[1.,.27],[.27,1.]])\n'
               'u=np.array([[[1.,0.],[0.,1.]],[[1.,1.],[1.,1.]],[[1.,-.7],[-.7,.49]]])\n'
               'prior=np.column_stack(([.21,.43,.36],u.reshape(3,4)))\n'
               'xx=np.array([[24.,11.,-3.],[11.,30.,2.],[-3.,2.,21.]])\n'
               'xy=np.array([[11.,7.],[8.,4.],[-1.,8.]])\n'
               'pi=np.array([.23,.51,.26]);d=np.diag(xx)\n'
               'grid=np.array([0.,.015,.06,.2,.7]);priority=np.array([30.,10.,20.])\n'
               '\n'
               'def _case_raises():\n'
               '    try:\n'
               '        estimate_residual_correlation(weak, -1.)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        _oracle_estimate_residual_correlation(weak, -1.)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
