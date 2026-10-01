"""
Select a follow-up SNP under the declared credible-set and trait-sign policy and report its overall probability of a nonzero effect in any trait

state, credible_sets and average_lfsr use the prior return layouts. A component is eligible when its set is retained and its average local false sign rate in the zero-based trait_index is strictly below sign_threshold. Candidates are the union of eligible memberships. Choose the candidate with the greatest supplied priority, breaking priority ties by lower SNP index. The requested PIP concerns the fitted nonzero effect across all latent components, including components whose credible set is discarded. A zero-scale component represents the exact null. If there is no eligible candidate, return [-1,0]. priority is a finite length-J vector; sign_threshold lies in [0,1]. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
Length-2 numeric vector [selected zero-based SNP index, cross-trait PIP].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def prioritize_variant(
    state, credible_sets, average_lfsr, trait_index, sign_threshold, priority,
) -> np.ndarray:
    """Select a follow-up SNP under the declared credible-set and trait-sign policy and report its overall probability of a nonzero effect in any trait.

    Return Length-2 numeric vector [selected zero-based SNP index, cross-trait PIP]."""
    # Placeholder only; implement the operation described above.
    return np.zeros(2, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, ndtr

def _oracle_prioritize_variant(state, credible_sets, average_lfsr, trait_index, sign_threshold, priority):
    s, l, j, r = _bg11_state(state)
    cs = _bg11_array(credible_sets, 2)
    averages = _bg11_array(average_lfsr, 2)
    ranking = _bg11_array(priority, 1)
    trait = _bg11_integer(trait_index, 0)
    if cs.shape != (l, j+3) or averages.shape != (l, r) or ranking.shape != (j,) or trait >= r:
        raise ValueError('incompatible prioritization dimensions')
    if not np.isfinite(sign_threshold) or not 0 <= sign_threshold <= 1:
        raise ValueError('invalid sign threshold')
    if np.any((cs[:, 0] != 0) & (cs[:, 0] != 1)) or np.any((cs[:, 3:] != 0) & (cs[:, 3:] != 1)) or np.any(averages < 0) or np.any(averages > 1):
        raise ValueError('invalid membership or average-lfsr state')
    eligible = (cs[:, 0] == 1) & (averages[:, trait] < sign_threshold)
    candidates = np.flatnonzero(np.any(cs[eligible, 3:] == 1, axis=0))
    if not len(candidates):
        return np.array([-1.0, 0.0])
    chosen = int(candidates[np.argmax(ranking[candidates])])
    active = s[:, j, 0] > 0
    pip = float(1-np.prod(1-s[active, chosen, 0]))
    return np.array([float(chosen), pip])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n',
      'call': 'prioritize_variant(s,cs,avg,0,.07,ranking)',
      'gold_call': '_oracle_prioritize_variant(s,cs,avg,0,.07,ranking)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n',
      'call': 'prioritize_variant(s,cs,avg,0,.06,ranking)',
      'gold_call': '_oracle_prioritize_variant(s,cs,avg,0,.06,ranking)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               'ranking[:2]=20\n',
      'call': 'prioritize_variant(s,cs,avg,0,.07,ranking)',
      'gold_call': '_oracle_prioritize_variant(s,cs,avg,0,.07,ranking)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               's[2,-1,0]=.4\n',
      'call': 'prioritize_variant(s,cs,avg,0,.07,ranking)',
      'gold_call': '_oracle_prioritize_variant(s,cs,avg,0,.07,ranking)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               'cs[1,0]=1;ranking=np.array([20.,20.,30.,40.])\n',
      'call': 'prioritize_variant(s,cs,avg,1,.095,ranking)',
      'gold_call': '_oracle_prioritize_variant(s,cs,avg,1,.095,ranking)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               '\n'
               'def _case_raises():\n'
               '    try:\n'
               '        prioritize_variant(s,cs,avg,2,.07,ranking)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        _oracle_prioritize_variant(s,cs,avg,2,.07,ranking)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
