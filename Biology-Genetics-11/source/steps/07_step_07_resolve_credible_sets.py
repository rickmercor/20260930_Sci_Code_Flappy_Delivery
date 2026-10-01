"""
Determine source-style per-component credible sets and retain those meeting the supplied scale, purity, and deduplication policy.

state is a stacked posterior state; ld is a positive-definite J x J correlation matrix. For each component, take the shortest descending-probability prefix attaining coverage, using lower SNP index on probability ties and all SNPs if coverage is one. Purity is the minimum absolute pairwise LD within the set (one for a singleton). Retain positive-scale sets whose purity is at least min_purity; if an identical membership set was retained earlier, retain only that first component. Keep diagnostics for discarded and zero-scale sets. coverage lies in (0,1] and min_purity in [0,1]. Return L x (J+3): columns 0,1,2 are retained flag, achieved coverage and purity, followed by J binary membership flags. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
L x (J+3) numeric credible-set diagnostics.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def resolve_credible_sets(state, ld, coverage, min_purity) -> np.ndarray:
    """Determine source-style per-component credible sets and retain those meeting the supplied scale, purity, and deduplication policy.

    Return L x (J+3) numeric credible-set diagnostics."""
    # Placeholder only; implement the operation described above.
    return np.zeros((np.shape(state)[0], np.shape(ld)[0] + 3), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, ndtr

def _oracle_resolve_credible_sets(state, ld, coverage, min_purity):
    s, l, j, r = _bg11_state(state)
    correlation = _bg11_cov(ld, j)
    if not np.allclose(np.diag(correlation), 1, rtol=0, atol=1e-10):
        raise ValueError('LD must be a correlation matrix')
    if not np.isfinite(coverage) or not 0 < coverage <= 1 or not np.isfinite(min_purity) or not 0 <= min_purity <= 1:
        raise ValueError('invalid credible-set thresholds')
    out = np.zeros((l, j+3))
    seen = set()
    for a in range(l):
        probabilities = s[a, :j, 0]
        order = np.argsort(-probabilities, kind='stable')
        count = j if coverage == 1 else min(j, int(np.searchsorted(np.cumsum(probabilities[order]), coverage, side='left'))+1)
        members = order[:count]
        pure = 1.0 if count == 1 else float(np.min(np.abs(correlation[np.ix_(members, members)][np.triu_indices(count, 1)])))
        key = tuple(sorted(members.tolist()))
        retained = s[a, j, 0] > 0 and pure >= min_purity and key not in seen
        if retained:
            seen.add(key)
        out[a, :3] = [float(retained), probabilities[members].sum(), pure]
        out[a, 3+members] = 1
    return out

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
      'call': 'resolve_credible_sets(s,ld,.9,.95)',
      'gold_call': '_oracle_resolve_credible_sets(s,ld,.9,.95)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n',
      'call': 'resolve_credible_sets(s,ld,1.,0.)',
      'gold_call': '_oracle_resolve_credible_sets(s,ld,1.,0.)',
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
      'call': 'resolve_credible_sets(s,ld,.9,.94)',
      'gold_call': '_oracle_resolve_credible_sets(s,ld,.9,.94)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               's[0,:4,0]=[.9,.05,.03,.02]\n',
      'call': 'resolve_credible_sets(s,ld,.9,1.)',
      'gold_call': '_oracle_resolve_credible_sets(s,ld,.9,1.)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               's[0,:4,0]=[1.,0.,0.,0.]\n',
      'call': 'resolve_credible_sets(s,ld,1.,0.)',
      'gold_call': '_oracle_resolve_credible_sets(s,ld,1.,0.)',
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
               '        resolve_credible_sets(s,ld,0.,.9)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        _oracle_resolve_credible_sets(s,ld,0.,.9)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
