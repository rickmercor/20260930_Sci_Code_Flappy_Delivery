"""
Determine the fixed-scale single-effect posterior, retaining SNP, covariance-pattern, and trait-sign uncertainty.

information is a positive length-J vector d; residual_crossproduct is J x R. Conditional on a selected SNP j and effect b, residual_crossproduct[j]/d[j] has distribution N(b, residual_covariance/d[j]). One latent SNP is drawn from the strictly positive length-J probability vector variant_prior. Conditional on covariance pattern k, its effect has prior N(0, scale*U[k]); scale is nonnegative. Other SNP effect rows are zero. Local false sign rates condition on the SNP being selected, marginalize the pattern, and use the source's strict positive/negative sign events; an exactly zero trait is a point mass. A packed prior is K x (1+R*R): each row contains its positive mixture probability followed by the row-major entries of its symmetric positive-semidefinite R x R covariance. Mixture probabilities sum to one; a covariance may be singular. Covariance-pattern means are fixed at zero. Posterior state: a (J+1) x (2R+3) numeric array. In the J SNP rows, column 0 is the SNP selection probability, columns 1:R+1 are unconditional posterior mean effects, columns R+1:2R+1 are trait-wise conditional local false sign rates, column 2R+1 is the SNP log mixture Bayes factor, and the last column is zero. The final row has [scale, log marginal Bayes factor, delta, previous_delta, sweeps] in columns 0:5; remaining entries are zero. Delta, previous_delta and sweeps are zero until the full additive fit reports them. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
Packed (J+1) x (2R+3) posterior state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fit_single_effect(
    information, residual_crossproduct, residual_covariance, prior, variant_prior,
    scale,
) -> np.ndarray:
    """Determine the fixed-scale single-effect posterior, retaining SNP, covariance-pattern, and trait-sign uncertainty.

    Return Packed (J+1) x (2R+3) posterior state."""
    # Placeholder only; implement the operation described above.
    return np.zeros(
        (np.size(information) + 1, 2 * np.shape(residual_crossproduct)[1] + 3),
        dtype=float,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, ndtr

def _oracle_fit_single_effect(information, residual_crossproduct, residual_covariance, prior, variant_prior, scale):
    d = _bg11_array(information, 1)
    xy = _bg11_array(residual_crossproduct, 2)
    j, r = xy.shape
    if d.shape != (j,) or np.any(d <= 0) or not np.isfinite(scale) or scale < 0:
        raise ValueError('invalid information or scale')
    v = _bg11_cov(residual_covariance, r)
    w, u = _bg11_prior(prior, r)
    pi = _bg11_prob(variant_prior, j)
    b = xy/d[:, None]
    k = len(w)
    log_bf = np.empty((j, k))
    means = np.empty((j, k, r))
    positive = np.zeros((j, k, r))
    negative = np.zeros_like(positive)
    for a in range(j):
        noise = v/d[a]
        for h in range(k):
            covariance = scale*u[h]
            total = noise+covariance
            solved = np.linalg.solve(total, b[a])
            mean = covariance @ solved
            posterior_covariance = covariance - covariance @ np.linalg.solve(total, covariance)
            log_bf[a, h] = 0.5*(np.linalg.slogdet(noise)[1] - np.linalg.slogdet(total)[1]
                                  + b[a] @ (np.linalg.solve(noise, b[a])-solved))
            means[a, h] = mean
            # A zero diagonal of a PSD prior is an exact zero-valued trait.
            supported = np.diag(covariance) > 0
            variance = np.diag(posterior_covariance)
            if np.any(variance[supported] <= 0):
                raise ValueError('posterior variance lost positive precision')
            z = mean[supported]/np.sqrt(variance[supported])
            positive[a, h, supported] = ndtr(z)
            negative[a, h, supported] = ndtr(-z)
    local_bf = logsumexp(log_bf+np.log(w), axis=1)
    log_evidence = float(logsumexp(local_bf+np.log(pi)))
    alpha = np.exp(local_bf+np.log(pi)-log_evidence)
    omega = np.exp(log_bf+np.log(w)-local_bf[:, None])
    out = np.zeros((j+1, 2*r+3))
    out[:j, 0] = alpha
    out[:j, 1:r+1] = alpha[:, None]*np.einsum('jk,jkr->jr', omega, means)
    out[:j, r+1:2*r+1] = 1-np.maximum(np.einsum('jk,jkr->jr', omega, positive), np.einsum('jk,jkr->jr', omega, negative))
    out[:j, 2*r+1] = local_bf
    out[j, :2] = [scale, log_evidence]
    return out

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
      'call': 'fit_single_effect(d,xy,v,prior,pi,.06)',
      'gold_call': '_oracle_fit_single_effect(d,xy,v,prior,pi,.06)',
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
      'call': 'fit_single_effect(d,xy,v,prior,pi,0.)',
      'gold_call': '_oracle_fit_single_effect(d,xy,v,prior,pi,0.)',
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
               'prior=np.array([[1.,1.,0.,0.,0.]])\n',
      'call': 'fit_single_effect(d,xy,v,prior,pi,.2)',
      'gold_call': '_oracle_fit_single_effect(d,xy,v,prior,pi,.2)',
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
               'xy*=11\n',
      'call': 'fit_single_effect(d,xy,v,prior,pi,.06)',
      'gold_call': '_oracle_fit_single_effect(d,xy,v,prior,pi,.06)',
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
               'prior=np.array([[.8,0.,0.,0.,0.],[.2,1.,0.,0.,0.]])\n',
      'call': 'fit_single_effect(d,xy,v,prior,pi,.08)',
      'gold_call': '_oracle_fit_single_effect(d,xy,v,prior,pi,.08)',
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
               'pi=np.array([.2,.2,.2])\n'
               '\n'
               'def _case_raises():\n'
               '    try:\n'
               '        fit_single_effect(d,xy,v,prior,pi,.06)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        _oracle_fit_single_effect(d,xy,v,prior,pi,.06)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
