"""
Select the component scale by the source's marginal-evidence objective over a supplied finite grid and return its posterior.

Inputs and posterior layout are as in the preceding single-effect operation. scale_grid is a nonempty strictly increasing vector of nonnegative values. An optional initial_candidate is a previously computed posterior for the first scale, under exactly the same inputs; it may be reused. Compare the full single-effect marginal likelihood, integrating both SNP and pattern uncertainty. Choose the smaller scale on an exact tie. This discrete search is a benchmark restriction on the source objective, not a claim about a software default. Posterior state: a (J+1) x (2R+3) numeric array. In the J SNP rows, column 0 is the SNP selection probability, columns 1:R+1 are unconditional posterior mean effects, columns R+1:2R+1 are trait-wise conditional local false sign rates, column 2R+1 is the SNP log mixture Bayes factor, and the last column is zero. The final row has [scale, log marginal Bayes factor, delta, previous_delta, sweeps] in columns 0:5; remaining entries are zero. Delta, previous_delta and sweeps are zero until the full additive fit reports them. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
Selected packed (J+1) x (2R+3) posterior state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_effect_scale(
    information, residual_crossproduct, residual_covariance, prior, variant_prior,
    scale_grid, initial_candidate=None,
) -> np.ndarray:
    """Select the component scale by the source's marginal-evidence objective over a supplied finite grid and return its posterior.

    Return Selected packed (J+1) x (2R+3) posterior state."""
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

def _oracle_select_effect_scale(information, residual_crossproduct, residual_covariance, prior, variant_prior, scale_grid, initial_candidate=None):
    grid = _bg11_array(scale_grid, 1)
    if grid[0] < 0 or np.any(np.diff(grid) <= 0):
        raise ValueError('scale grid must be nonnegative and strictly increasing')
    best = None
    for index, scale in enumerate(grid):
        if index == 0 and initial_candidate is not None:
            candidate = _bg11_array(initial_candidate, 2).copy()
            j, r = np.asarray(residual_crossproduct).shape
            if candidate.shape != (j+1, 2*r+3) or candidate[-1, 0] != scale:
                raise ValueError('invalid initial scale candidate')
        else:
            candidate = _oracle_fit_single_effect(information, residual_crossproduct, residual_covariance, prior, variant_prior, float(scale))
        if best is None or candidate[-1, 1] > best[-1, 1]:
            best = candidate
    return best

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
      'call': 'select_effect_scale(d,xy,v,prior,pi,grid)',
      'gold_call': '_oracle_select_effect_scale(d,xy,v,prior,pi,grid)',
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
               'xy=np.zeros_like(xy)\n',
      'call': 'select_effect_scale(d,xy,v,prior,pi,grid)',
      'gold_call': '_oracle_select_effect_scale(d,xy,v,prior,pi,grid)',
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
      'call': 'select_effect_scale(d,xy,v,prior,pi,grid)',
      'gold_call': '_oracle_select_effect_scale(d,xy,v,prior,pi,grid)',
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
               'grid=np.array([.01,.03,.1,.3]);pi=np.array([.01,.04,.95]);xy[2]=[2.,-1.]\n',
      'call': 'select_effect_scale(d,xy,v,prior,pi,grid)',
      'gold_call': '_oracle_select_effect_scale(d,xy,v,prior,pi,grid)',
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
               'xy=np.array([[9.334277356709858,6.910918725502524],[-1.8668574602360501,1.6992179726812442],[8.693709041476758,-.3667149394787321]]);pi=np.array([.007884225701745837,.7439771719985124,.24813860229974155])\n',
      'call': 'select_effect_scale(d,xy,v,prior,pi,grid)',
      'gold_call': '_oracle_select_effect_scale(d,xy,v,prior,pi,grid)',
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
               'grid=np.array([0.,.1,.1])\n'
               '\n'
               'def _case_raises():\n'
               '    try:\n'
               '        select_effect_scale(d,xy,v,prior,pi,grid)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        _oracle_select_effect_scale(d,xy,v,prior,pi,grid)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
