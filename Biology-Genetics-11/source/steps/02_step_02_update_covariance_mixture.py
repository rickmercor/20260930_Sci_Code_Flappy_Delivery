"""
Carry out one simultaneous expectation/maximization update of the zero-centered prior mixture under the source's noisy lead-score training model.

training_z is T x R; each row is an already selected lead score from a distinct training region. Its latent zero-mean effect has the packed mixture prior and its independent observation error has the positive-definite covariance residual_covariance. Update both weights and covariance patterns, retaining zero component means. Do not add a ridge in this operation. A packed prior is K x (1+R*R): each row contains its positive mixture probability followed by the row-major entries of its symmetric positive-semidefinite R x R covariance. Mixture probabilities sum to one; a covariance may be singular. Covariance-pattern means are fixed at zero. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
Updated K x (1+R*R) packed prior.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def update_covariance_mixture(training_z, residual_covariance, prior) -> np.ndarray:
    """Carry out one simultaneous expectation/maximization update of the zero-centered prior mixture under the source's noisy lead-score training model.

    Return Updated K x (1+R*R) packed prior."""
    # Placeholder only; implement the operation described above.
    return np.zeros_like(prior, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp

def _oracle_update_covariance_mixture(training_z, residual_covariance, prior):
    x = _bg11_array(training_z, 2)
    n, r = x.shape
    v = _bg11_cov(residual_covariance, r)
    w, u = _bg11_prior(prior, r)
    k = len(w)
    log_q = np.empty((n, k))
    moments = np.empty((n, k, r, r))
    for h in range(k):
        total = u[h] + v
        solved = np.linalg.solve(total, x.T).T
        log_q[:, h] = np.log(w[h]) - 0.5*(np.linalg.slogdet(total)[1] + np.sum(x*solved, axis=1))
        mean = solved @ u[h]
        covariance = u[h] - u[h] @ np.linalg.solve(total, u[h])
        moments[:, h] = covariance + np.einsum('ni,nj->nij', mean, mean)
    q = np.exp(log_q - logsumexp(log_q, axis=1)[:, None])
    counts = q.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError('mixture component has zero numerical mass')
    updated = np.einsum('nk,nkij->kij', q, moments) / counts[:, None, None]
    updated = (updated + updated.transpose(0, 2, 1))/2
    return np.column_stack((counts/n, updated.reshape(k, r*r)))

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
      'call': 'update_covariance_mixture(training,v,prior)',
      'gold_call': '_oracle_update_covariance_mixture(training,v,prior)',
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
               'training=training[:1]\n',
      'call': 'update_covariance_mixture(training,v,prior)',
      'gold_call': '_oracle_update_covariance_mixture(training,v,prior)',
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
      'call': 'update_covariance_mixture(training,v,prior)',
      'gold_call': '_oracle_update_covariance_mixture(training,v,prior)',
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
               'prior=np.array([[1.,0.,0.,0.,0.]])\n',
      'call': 'update_covariance_mixture(training,v,prior)',
      'gold_call': '_oracle_update_covariance_mixture(training,v,prior)',
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
               'prior=prior[[2,0,1]]\n',
      'call': 'update_covariance_mixture(training,v,prior)',
      'gold_call': '_oracle_update_covariance_mixture(training,v,prior)',
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
               'prior[0,1]=-1\n'
               '\n'
               'def _case_raises():\n'
               '    try:\n'
               '        update_covariance_mixture(training,v,prior)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        _oracle_update_covariance_mixture(training,v,prior)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
