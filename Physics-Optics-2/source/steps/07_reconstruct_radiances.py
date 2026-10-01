"""
Compute the exact pixel radiance and long-run vanilla radiance for the finite transport model.

The transport observable is normalized by the scalar importance density. Radiances use the original cell-volume reference measure.

Returns
-------
np.ndarray of shape (2,): the exact radiance, then the almost-sure limiting vanilla radiance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reconstruct_radiances(quantities: "np.ndarray", mean_weights: "np.ndarray", stationary: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    quantities : np.ndarray
        Shape (N, 2), unnormalized target masses then selected pixel observables.
    mean_weights : np.ndarray
        Positive (N,) conditional mean operational holding weights.
    stationary : np.ndarray
        Shape (N,), stationary probabilities of the accepted-tour chain.

    Returns
    -------
    result : np.ndarray
        Shape (2,), ordered as exact radiance, then almost-sure limiting
        vanilla radiance over infinitely many completed acceptance tours.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_reconstruct_radiances(quantities: "np.ndarray", mean_weights: "np.ndarray", stationary: "np.ndarray") -> "np.ndarray":
    masses = quantities[:, 0]
    observable = quantities[:, 1]
    weighted_law = stationary * mean_weights
    exact = np.dot(masses, observable)
    vanilla = masses.sum() * np.dot(weighted_law, observable) / weighted_law.sum()
    return np.array([exact, vanilla])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test configurations."""
    return [{'setup': 'import numpy as np\n'
               'quantities = np.array([[.3,.1],[1.2,.9]])\n'
               'mean_weights = np.array([1.,1.8])\n'
               'stationary = np.array([1/3,2/3])',
      'call': 'reconstruct_radiances(quantities.copy(), mean_weights.copy(), stationary.copy())',
      'gold_call': '_oracle_reconstruct_radiances(quantities.copy(), mean_weights.copy(), stationary.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'quantities = np.array([[5.,.6]])\n'
               'mean_weights = np.ones(1)\n'
               'stationary = np.ones(1)',
      'call': 'reconstruct_radiances(quantities.copy(), mean_weights.copy(), stationary.copy())',
      'gold_call': '_oracle_reconstruct_radiances(quantities.copy(), mean_weights.copy(), stationary.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\n'
               'quantities = np.array([[.01,0.],[9.,1.]])\n'
               'mean_weights = np.array([1.,5.])\n'
               'stationary = np.array([1/91,90/91])',
      'call': 'reconstruct_radiances(quantities.copy(), mean_weights.copy(), stationary.copy())',
      'gold_call': '_oracle_reconstruct_radiances(quantities.copy(), mean_weights.copy(), stationary.copy())',
      'tol': 1e-10}]
