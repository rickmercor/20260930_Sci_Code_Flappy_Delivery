"""
Compute the statewise acceptance mean and second rejection-probability moment.

Moments refer to a fresh proposal at the given current cell, before the accept/reject decision.

Returns
-------
np.ndarray of shape (N, 2): mean acceptance probability in column 0 and second moment of the rejection probability in column 1, in cell order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_rejection_moments(proposal: "np.ndarray", acceptance: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    proposal : np.ndarray
        Nonnegative row-stochastic (N, N) proposal matrix.
    acceptance : np.ndarray
        Shape (N, N) acceptance probabilities in $[0,1]$.

    Returns
    -------
    result : np.ndarray
        Shape (N, 2), with mean acceptance in column zero and the second
        moment of the rejection probability in column one; cell order is preserved.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_rejection_moments(proposal: "np.ndarray", acceptance: "np.ndarray") -> "np.ndarray":
    rho = np.sum(proposal * acceptance, axis=1)
    r2 = np.sum(proposal * (1.0 - acceptance) ** 2, axis=1)
    return np.column_stack((rho, r2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test configurations."""
    return [{'setup': 'import numpy as np\n'
               'proposal = np.array([[.2,.8],[.6,.4]])\n'
               'acceptance = np.array([[1.,.3],[1.,1.]])',
      'call': 'compute_rejection_moments(proposal.copy(), acceptance.copy())',
      'gold_call': '_oracle_compute_rejection_moments(proposal.copy(), acceptance.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\nproposal = np.ones((1,1))\nacceptance = np.ones((1,1))',
      'call': 'compute_rejection_moments(proposal.copy(), acceptance.copy())',
      'gold_call': '_oracle_compute_rejection_moments(proposal.copy(), acceptance.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'proposal = np.array([[.1,.9],[.9,.1]])\n'
               'acceptance = np.array([[1.,.001],[1.,1.]])',
      'call': 'compute_rejection_moments(proposal.copy(), acceptance.copy())',
      'gold_call': '_oracle_compute_rejection_moments(proposal.copy(), acceptance.copy())',
      'tol': 1e-10}]
