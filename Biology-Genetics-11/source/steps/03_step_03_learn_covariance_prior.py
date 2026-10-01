"""
Fit the source's zero-centered noisy training mixture for exactly iterations updates, then apply the supplied terminal covariance ridge.

iterations is a nonnegative integer; ridge is nonnegative. Use the complete mixture state from each update as the next initial state, without component sorting, pruning, splitting, or merging. After the last update, add ridge times the identity to every covariance once. For zero iterations, apply only this terminal ridge. A packed prior is K x (1+R*R): each row contains its positive mixture probability followed by the row-major entries of its symmetric positive-semidefinite R x R covariance. Mixture probabilities sum to one; a covariance may be singular. Covariance-pattern means are fixed at zero. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
Final K x (1+R*R) packed prior.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def learn_covariance_prior(
    training_z, residual_covariance, initial_prior, iterations, ridge,
) -> np.ndarray:
    """Fit the source's zero-centered noisy training mixture for exactly iterations updates, then apply the supplied terminal covariance ridge.

    Return Final K x (1+R*R) packed prior."""
    # Placeholder only; implement the operation described above.
    return np.zeros_like(initial_prior, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, ndtr

def _oracle_learn_covariance_prior(training_z, residual_covariance, initial_prior, iterations, ridge):
    x = _bg11_array(training_z, 2)
    r = x.shape[1]
    _bg11_cov(residual_covariance, r)
    _bg11_prior(initial_prior, r)
    count = _bg11_integer(iterations, 0)
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError('ridge must be nonnegative')
    result = np.array(initial_prior, dtype=float, copy=True)
    for _ in range(count):
        result = _oracle_update_covariance_mixture(x, residual_covariance, result)
    u = result[:, 1:].reshape(-1, r, r).copy()
    u += ridge*np.eye(r)
    result[:, 1:] = u.reshape(len(u), r*r)
    return result

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
      'call': 'learn_covariance_prior(training,v,prior,4,.03)',
      'gold_call': '_oracle_learn_covariance_prior(training,v,prior,4,.03)',
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
               'grid=np.array([0.,.015,.06,.2,.7]);priority=np.array([30.,10.,20.])\n',
      'call': 'learn_covariance_prior(training,v,prior,0,.05)',
      'gold_call': '_oracle_learn_covariance_prior(training,v,prior,0,.05)',
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
               'grid=np.array([0.,.015,.06,.2,.7]);priority=np.array([30.,10.,20.])\n',
      'call': 'learn_covariance_prior(training,v,prior,1,0.)',
      'gold_call': '_oracle_learn_covariance_prior(training,v,prior,1,0.)',
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
               'training=np.zeros_like(training)\n',
      'call': 'learn_covariance_prior(training,v,prior,2,.1)',
      'gold_call': '_oracle_learn_covariance_prior(training,v,prior,2,.1)',
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
               'training=training[::-1]\n',
      'call': 'learn_covariance_prior(training,v,prior,3,.015)',
      'gold_call': '_oracle_learn_covariance_prior(training,v,prior,3,.015)',
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
               '        learn_covariance_prior(training,v,prior,-1,.01)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        _oracle_learn_covariance_prior(training,v,prior,-1,.01)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
