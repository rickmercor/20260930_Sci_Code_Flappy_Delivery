"""
Compute the Metropolis–Hastings acceptance probabilities for the complete proposal.

The target entries are probability masses on a finite set, and the proposal entries are transition probabilities. Self-proposals are allowed.

Returns
-------
np.ndarray of shape (N, N): entry $(i,j)$ is the Metropolis–Hastings acceptance probability for proposing cell $j$ from cell $i$, equal to one where the forward proposal probability is zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_acceptance(masses: "np.ndarray", proposal: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    masses : np.ndarray
        Positive (N,) unnormalized target masses.
    proposal : np.ndarray
        Nonnegative row-stochastic (N, N) proposal probabilities.

    Returns
    -------
    result : np.ndarray
        Shape (N, N), entry $(i,j)$ is the acceptance probability for $j$ from $i$.
        Convention: entries for a zero forward proposal probability equal one.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_acceptance(masses: "np.ndarray", proposal: "np.ndarray") -> "np.ndarray":
    forward = masses[:, None] * proposal
    reverse = forward.T
    ratio = np.divide(reverse, forward, out=np.ones_like(forward), where=forward > 0)
    return np.minimum(1.0, ratio)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test configurations."""
    return [{'setup': 'import numpy as np\n'
               'masses = np.array([.2,1.7,.6])\n'
               'proposal = np.array([[.1,.7,.2],[.2,.2,.6],[.5,.1,.4]])',
      'call': 'compute_acceptance(masses.copy(), proposal.copy())',
      'gold_call': '_oracle_compute_acceptance(masses.copy(), proposal.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\nmasses = np.ones(2)\nproposal = np.array([[.3,.7],[.7,.3]])',
      'call': 'compute_acceptance(masses.copy(), proposal.copy())',
      'gold_call': '_oracle_compute_acceptance(masses.copy(), proposal.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'masses = np.array([.01,9.,.3])\n'
               'proposal = np.array([[.1,.9,0.],[0.,.1,.9],[.9,0.,.1]])',
      'call': 'compute_acceptance(masses.copy(), proposal.copy())',
      'gold_call': '_oracle_compute_acceptance(masses.copy(), proposal.copy())',
      'tol': 1e-10}]
