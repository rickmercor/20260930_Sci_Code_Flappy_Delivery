"""
Compute the full local/global proposal transition matrix.

Each matrix row gives destination probabilities from its current cell.

Returns
-------
np.ndarray of shape (N, N): the row-stochastic complete mixture proposal matrix, one row per current cell.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compose_proposal(local: "np.ndarray", global_prob: "np.ndarray", large_step: float) -> "np.ndarray":
    """Parameters
    ----------
    local : np.ndarray
        Nonnegative row-stochastic (N, N) small-step proposal matrix.
    global_prob : np.ndarray
        Nonnegative (N,) destination probabilities summing to one.
    large_step : float
        Probability in $[0,1]$ of selecting the large-step proposal.

    Returns
    -------
    result : np.ndarray
        Row-stochastic (N, N) complete mixture proposal matrix.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compose_proposal(local: "np.ndarray", global_prob: "np.ndarray", large_step: float) -> "np.ndarray":
    return (1.0 - large_step) * local + large_step * global_prob[None, :]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test configurations."""
    return [{'setup': 'import numpy as np\n'
               'flux = np.array([[.2,0],[.8,.1],[4,.5],[.1,2],[.3,7],[1.2,.2],[2.5,1],[.05,.9]])\n'
               'volumes = np.array([.08,.12,.09,.16,.14,.11,.17,.13])\n'
               'local = np.zeros((8,8))\n'
               'for i in range(8):\n'
               '    local[i,i] = .1\n'
               '    local[i,(i+1)%8] = .55\n'
               '    local[i,(i-1)%8] = .35\n'
               'global_prob = volumes.copy()\n'
               'large_step = .3\n'
               'pixel = 1\n',
      'call': 'compose_proposal(local.copy(), global_prob.copy(), large_step)',
      'gold_call': '_oracle_compose_proposal(local.copy(), global_prob.copy(), large_step)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'flux = np.array([[2.,3.]])\n'
               'volumes = np.array([1.])\n'
               'local = np.ones((1,1))\n'
               'global_prob = np.ones(1)\n'
               'large_step = 0.\n'
               'pixel = 1\n',
      'call': 'compose_proposal(local.copy(), global_prob.copy(), large_step)',
      'gold_call': '_oracle_compose_proposal(local.copy(), global_prob.copy(), large_step)',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'flux = np.array([[.01,0.],[0.,9.],[.2,.1]])\n'
               'volumes = np.array([.2,.5,.3])\n'
               'local = np.array([[.1,.9,0.],[0.,.1,.9],[.9,0.,.1]])\n'
               'global_prob = np.array([.2,.5,.3])\n'
               'large_step = 1.\n'
               'pixel = 1\n',
      'call': 'compose_proposal(local.copy(), global_prob.copy(), large_step)',
      'gold_call': '_oracle_compose_proposal(local.copy(), global_prob.copy(), large_step)',
      'tol': 1e-10}]
