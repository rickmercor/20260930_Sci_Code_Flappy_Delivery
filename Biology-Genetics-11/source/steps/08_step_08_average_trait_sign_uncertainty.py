"""
Compute the source-defined component average local false sign rate for each trait.

state is a valid L x (J+1) x (2R+3) posterior state, using the layout defined above. Apply the source's averaging convention for the trait-specific single-effect summary. Include a row for every component, including zero-scale components. Use binary64 arithmetic without intermediate rounding. Raise ValueError for incompatible dimensions, nonfinite inputs, invalid probability vectors or thresholds, and covariances outside the stated domain. No particular linear-algebra factorization is required.

Returns
-------
L x R component average local false sign rates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def average_trait_sign_uncertainty(state) -> np.ndarray:
    """Compute the source-defined component average local false sign rate for each trait.

    Return L x R component average local false sign rates."""
    # Placeholder only; implement the operation described above.
    return np.zeros(
        (np.shape(state)[0], (np.shape(state)[2] - 3) // 2),
        dtype=float,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import logsumexp, ndtr

def _oracle_average_trait_sign_uncertainty(state):
    if np.asarray(state).ndim != 3:
        raise ValueError('expected a stacked posterior state')
    s, l, j, r = _bg11_state(state)
    return np.einsum('lj,ljr->lr', s[:, :j, 0], s[:, :j, r+1:2*r+1])

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
      'call': 'average_trait_sign_uncertainty(s)',
      'gold_call': '_oracle_average_trait_sign_uncertainty(s)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               's[:,:4,3:5]=1\n',
      'call': 'average_trait_sign_uncertainty(s)',
      'gold_call': '_oracle_average_trait_sign_uncertainty(s)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               's[:,:4,0]=.25\n',
      'call': 'average_trait_sign_uncertainty(s)',
      'gold_call': '_oracle_average_trait_sign_uncertainty(s)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               's=s[:1,:2,:].copy();s[:,0,0]=1\n',
      'call': 'average_trait_sign_uncertainty(s)',
      'gold_call': '_oracle_average_trait_sign_uncertainty(s)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               's[0,:4,3:5]=[[.6,.4],[.3,.8],[.2,.9],[.8,.6]]\n',
      'call': 'average_trait_sign_uncertainty(s)',
      'gold_call': '_oracle_average_trait_sign_uncertainty(s)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               's=np.zeros((3,5,7));s[:,:4,0]=[[.46,.45,.06,.03],[.44,.44,.08,.04],[.03,.04,.46,.47]]\n'
               's[:,:4,3:5]=[[[.01,.12],[.02,.03],[.45,.21],[.38,.24]],[[.03,.02],[.01,.04],[.35,.4],[.42,.31]],[[.4,.32],[.45,.29],[.04,.02],[.03,.01]]]\n'
               's[:,-1,0]=[.2,.3,0.]\n'
               'ld=np.array([[1.,-.97,0.,0.],[-.97,1.,0.,0.],[0.,0.,1.,.94],[0.,0.,.94,1.]])\n'
               'cs=np.array([[1.,.91,.97,1.,1.,0.,0.],[0.,.88,.97,1.,1.,0.,0.],[0.,.93,.94,0.,0.,1.,1.]])\n'
               'avg=np.array([[.06,.08],[.07,.09],[1.,1.]])\n'
               'ranking=np.array([10.,20.,40.,30.])\n'
               's[0,0,0]=-.1\n'
               '\n'
               'def _case_raises():\n'
               '    try:\n'
               '        average_trait_sign_uncertainty(s)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n'
               '\n'
               'def _reference_raises():\n'
               '    try:\n'
               '        _oracle_average_trait_sign_uncertainty(s)\n'
               '    except ValueError:\n'
               '        return True\n'
               '    return False\n',
      'call': '_case_raises()',
      'gold_call': '_reference_raises()',
      'tol': 1e-08}]
