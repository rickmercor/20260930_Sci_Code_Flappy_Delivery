"""
Orchestrate the complete source-grounded training, regional fit, uncertainty summary, and follow-up policy; return the one requested probability.

Use every preceding scientific stage: infer residual correlation with weak_cutoff, train the packed initial_prior for ed_iterations with terminal ridge, fit effects components to the supplied sufficient statistics, resolve coverage/min_purity sets, obtain source-defined average sign uncertainty, and apply trait_index/sign_threshold/priority. Input and boundary conventions are those declared for the individual stages. Derive LD by normalizing genotype_crossproduct to unit diagonal. Return zero if the policy produces no candidate. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
One finite float in [0,1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_prioritized_pip(
    weak_z, training_z, initial_prior, genotype_crossproduct,
    genotype_trait_crossproduct, variant_prior, scale_grid, priority, effects,
    ed_iterations, ridge, weak_cutoff, coverage, min_purity, trait_index,
    sign_threshold, tolerance, max_sweeps,
) -> float:
    """Orchestrate the complete source-grounded training, regional fit, uncertainty summary, and follow-up policy; return the one requested probability.

    Return One finite float in [0,1]."""
    # Placeholder only; implement the operation described above.
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, ndtr

def _oracle_compute_prioritized_pip(weak_z, training_z, initial_prior, genotype_crossproduct, genotype_trait_crossproduct, variant_prior, scale_grid, priority, effects, ed_iterations, ridge, weak_cutoff, coverage, min_purity, trait_index, sign_threshold, tolerance, max_sweeps):
    weak = _bg11_array(weak_z, 2)
    training = _bg11_array(training_z, 2)
    if weak.shape[1] != training.shape[1]:
        raise ValueError('training panels must have the same aligned traits')
    count = _bg11_integer(ed_iterations, 0)
    v = _oracle_estimate_residual_correlation(weak, weak_cutoff)
    # Reuse completed work at the three natural stage boundaries.
    current = initial_prior
    if count:
        current = _oracle_update_covariance_mixture(training, v, current)
    prior = _oracle_learn_covariance_prior(training, v, current, max(count-1, 0), ridge)
    d = np.diag(np.asarray(genotype_crossproduct, dtype=float))
    initial = _oracle_fit_single_effect(d, genotype_trait_crossproduct, v, prior, variant_prior, float(_bg11_array(scale_grid, 1)[0]))
    first = _oracle_select_effect_scale(d, genotype_trait_crossproduct, v, prior, variant_prior, scale_grid, initial_candidate=initial)
    state = _oracle_fit_additive_effects(genotype_crossproduct, genotype_trait_crossproduct, v, prior, variant_prior, scale_grid, effects, tolerance, max_sweeps, first_component_state=first)
    xx = np.asarray(genotype_crossproduct, dtype=float)
    ld = xx / np.sqrt(np.outer(np.diag(xx), np.diag(xx)))
    cs = _oracle_resolve_credible_sets(state, ld, coverage, min_purity)
    average = _oracle_average_trait_sign_uncertainty(state)
    result = _oracle_prioritize_variant(state, cs, average, trait_index, sign_threshold, priority)
    return float(result[1])

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
      'call': 'compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,3,.02,2.,.9,.2,0,1.,1e-9,160)',
      'gold_call': '_oracle_compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,3,.02,2.,.9,.2,0,1.,1e-9,160)',
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
      'call': 'compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,2,.02,2.,.9,0.,0,1.,1e-9,160)',
      'gold_call': '_oracle_compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,2,.02,2.,.9,0.,0,1.,1e-9,160)',
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
               'priority=np.array([10.,40.,20.])\n',
      'call': 'compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,0,.06,2.,.95,0.,1,1.,1e-9,160)',
      'gold_call': '_oracle_compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,0,.06,2.,.95,0.,1,1.,1e-9,160)',
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
               'xy[0]*=-1;xx=np.diag([24.,30.,21.])\n',
      'call': 'compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,2,.04,2.,.8,0.,1,.2,1e-9,160)',
      'gold_call': '_oracle_compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,2,.04,2.,.8,0.,1,.2,1e-9,160)',
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
               'pi=np.array([.6,.1,.3]);priority=np.array([10.,20.,30.])\n',
      'call': 'compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,3,2,.05,2.,.95,0.,1,.2,1e-9,160)',
      'gold_call': '_oracle_compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,3,2,.05,2.,.95,0.,1,.2,1e-9,160)',
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
               '        '
               'compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,3,.02,-2.,.9,.2,0,1.,1e-9,160)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        '
               '_oracle_compute_prioritized_pip(weak,training,prior,xx,xy,pi,grid,priority,2,3,.02,-2.,.9,.2,0,1.,1e-9,160)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
