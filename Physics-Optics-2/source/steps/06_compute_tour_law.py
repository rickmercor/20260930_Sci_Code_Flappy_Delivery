"""
Compute the stationary cell probabilities of the accepted-tour sequence.

Every accepted proposal starts a new tour, even when it returns the same cell. The accepted-tour chain is assumed irreducible.

Returns
-------
np.ndarray of shape (N,): the normalized stationary probability vector of the accepted-tour chain.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_tour_law(proposal: "np.ndarray", acceptance: "np.ndarray", moments: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    proposal : np.ndarray
        Row-stochastic (N, N) proposal matrix.
    acceptance : np.ndarray
        Matching (N, N) acceptance probabilities.
    moments : np.ndarray
        Shape (N, 2), with positive mean acceptance in column zero and
        second rejection-probability moment in column one.

    Returns
    -------
    result : np.ndarray
        Shape (N,), the normalized stationary probability vector of the
        accepted-tour transition matrix.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_tour_law(proposal: "np.ndarray", acceptance: "np.ndarray", moments: "np.ndarray") -> "np.ndarray":
    transition = proposal * acceptance / moments[:, 0, None]
    n = len(transition)
    system = transition.T - np.eye(n)
    system[-1, :] = 1.0
    rhs = np.zeros(n)
    rhs[-1] = 1.0
    stationary = np.linalg.solve(system, rhs)
    return stationary

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test configurations."""
    return [{'setup': 'import numpy as np\n'
               'proposal = np.array([[.2,.8],[.6,.4]])\n'
               'acceptance = np.array([[1.,.3],[1.,1.]])\n'
               'moments = np.array([[.44,.392],[1.,0.]])',
      'call': 'compute_tour_law(proposal.copy(), acceptance.copy(), moments.copy())',
      'gold_call': '_oracle_compute_tour_law(proposal.copy(), acceptance.copy(), moments.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'proposal = np.ones((1,1))\n'
               'acceptance = np.ones((1,1))\n'
               'moments = np.array([[1.,0.]])',
      'call': 'compute_tour_law(proposal.copy(), acceptance.copy(), moments.copy())',
      'gold_call': '_oracle_compute_tour_law(proposal.copy(), acceptance.copy(), moments.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'proposal = np.array([[.1,.9],[.9,.1]])\n'
               'acceptance = np.array([[1.,.001],[1.,1.]])\n'
               'moments = np.array([[.1009,.8982009],[1.,0.]])',
      'call': 'compute_tour_law(proposal.copy(), acceptance.copy(), moments.copy())',
      'gold_call': '_oracle_compute_tour_law(proposal.copy(), acceptance.copy(), moments.copy())',
      'tol': 1e-10}]
