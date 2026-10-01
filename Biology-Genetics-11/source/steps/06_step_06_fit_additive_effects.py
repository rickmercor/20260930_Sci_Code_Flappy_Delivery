"""
Fit the source model with effects latent single-effect components using the supplied exact sufficient statistics and fixed learned prior.

genotype_crossproduct is positive-definite J x J; genotype_trait_crossproduct is J x R. Use zero initial component means, initial SNP probabilities variant_prior, and zero initial scales. scale_grid starts at zero and is strictly increasing. Fit components sequentially in increasing index order using the newest states. The terminal sweep is the first completed sweep with the largest absolute change in any unconditional component mean at most tolerance and with every scale unchanged from its value at the beginning of that sweep. effects and max_sweeps are positive integers and tolerance is positive. Raise ValueError if the cap is reached without satisfying both conditions. An optional first_component_state supplies the already computed first component update from the all-zero initial means, under these same inputs; it replaces only component 0 of sweep 1 and does not increment the sweep count. It may equivalently be recomputed. No residual-covariance updating or additional hyperparameter fitting occurs here. Posterior state: a (J+1) x (2R+3) numeric array. In the J SNP rows, column 0 is the SNP selection probability, columns 1:R+1 are unconditional posterior mean effects, columns R+1:2R+1 are trait-wise conditional local false sign rates, column 2R+1 is the SNP log mixture Bayes factor, and the last column is zero. The final row has [scale, log marginal Bayes factor, delta, previous_delta, sweeps] in columns 0:5; remaining entries are zero. Delta, previous_delta and sweeps are zero until the full additive fit reports them. Return effects copies stacked on a new leading axis. Repeat terminal delta, preceding-sweep delta (zero before sweep 1), and sweep count in every metadata row. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
L x (J+1) x (2R+3) terminal state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fit_additive_effects(
    genotype_crossproduct, genotype_trait_crossproduct, residual_covariance, prior,
    variant_prior, scale_grid, effects, tolerance, max_sweeps,
    first_component_state=None,
) -> np.ndarray:
    """Fit the source model with effects latent single-effect components using the supplied exact sufficient statistics and fixed learned prior.

    Return L x (J+1) x (2R+3) terminal state."""
    # Placeholder only; implement the operation described above.
    return np.zeros(
        (int(effects), np.shape(genotype_trait_crossproduct)[0] + 1,
         2 * np.shape(genotype_trait_crossproduct)[1] + 3),
        dtype=float,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, ndtr

def _oracle_fit_additive_effects(genotype_crossproduct, genotype_trait_crossproduct, residual_covariance, prior, variant_prior, scale_grid, effects, tolerance, max_sweeps, first_component_state=None):
    xy = _bg11_array(genotype_trait_crossproduct, 2)
    j, r = xy.shape
    xx = _bg11_cov(genotype_crossproduct, j)
    _bg11_cov(residual_covariance, r)
    _bg11_prior(prior, r)
    pi = _bg11_prob(variant_prior, j)
    l = _bg11_integer(effects)
    cap = _bg11_integer(max_sweeps)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive')
    grid = _bg11_array(scale_grid, 1)
    if grid[0] != 0 or np.any(np.diff(grid) <= 0):
        raise ValueError('IBSS scale grid must start at zero and be strictly increasing')
    state = np.zeros((l, j+1, 2*r+3))
    state[:, :j, 0] = pi
    state[:, :j, r+1:2*r+1] = 1
    previous_delta = 0.0
    for sweep in range(1, cap+1):
        old_means = state[:, :j, 1:r+1].copy()
        old_scales = state[:, j, 0].copy()
        for a in range(l):
            means = state[:, :j, 1:r+1]
            residual = xy - xx @ (means.sum(axis=0)-means[a])
            if sweep == 1 and a == 0 and first_component_state is not None:
                supplied = _bg11_array(first_component_state, 2)
                if supplied.shape != state[a].shape:
                    raise ValueError('invalid first component state')
                state[a] = supplied
            else:
                state[a] = _oracle_select_effect_scale(np.diag(xx), residual, residual_covariance, prior, pi, grid)
        delta = float(np.max(np.abs(state[:, :j, 1:r+1]-old_means)))
        if delta <= tolerance and np.array_equal(state[:, j, 0], old_scales):
            state[:, j, 2:5] = [delta, previous_delta, sweep]
            return state
        previous_delta = delta
    raise ValueError('no convergence within max_sweeps')

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
      'call': 'fit_additive_effects(xx,xy,v,prior,pi,grid,2,1e-9,160)',
      'gold_call': '_oracle_fit_additive_effects(xx,xy,v,prior,pi,grid,2,1e-9,160)',
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
               'xy*=0\n',
      'call': 'fit_additive_effects(xx,xy,v,prior,pi,grid,3,1e-9,160)',
      'gold_call': '_oracle_fit_additive_effects(xx,xy,v,prior,pi,grid,3,1e-9,160)',
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
      'call': 'fit_additive_effects(xx,xy,v,prior,pi,grid,1,1e-9,160)',
      'gold_call': '_oracle_fit_additive_effects(xx,xy,v,prior,pi,grid,1,1e-9,160)',
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
               'xx=np.diag([24.,30.,21.]);xy[0]*=-1\n',
      'call': 'fit_additive_effects(xx,xy,v,prior,pi,grid,2,1e-9,160)',
      'gold_call': '_oracle_fit_additive_effects(xx,xy,v,prior,pi,grid,2,1e-9,160)',
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
               'xy*=.2\n',
      'call': 'fit_additive_effects(xx,xy,v,prior,pi,grid,4,1e-9,160)',
      'gold_call': '_oracle_fit_additive_effects(xx,xy,v,prior,pi,grid,4,1e-9,160)',
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
               '        fit_additive_effects(xx,xy,v,prior,pi,grid,2,1e-15,1)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        _oracle_fit_additive_effects(xx,xy,v,prior,pi,grid,2,1e-15,1)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
