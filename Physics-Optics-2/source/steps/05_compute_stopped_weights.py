"""
Compute the conditional mean of the operational vanilla holding weight at each cell.

The estimator is the acceptance-terminated vanilla MH estimator selected in the main problem. A terminal accepted proposal participates in its weight update, including accepted self-proposals. The operational definition governs any inconsistency with an ancillary theoretical identity.

Returns
-------
np.ndarray of shape (N,): the conditional mean of one completed tour's operational vanilla holding weight at each cell.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_stopped_weights(moments: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    moments : np.ndarray
        Shape (N, 2), with mean acceptance in column zero and second moment
        of rejection probability in column one, for independent proposals at
        each cell. Mean acceptance is positive and at most one.

    Returns
    -------
    result : np.ndarray
        Shape (N,), the exact conditional expectation of one completed
        tour's operational vanilla holding weight at each cell.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_stopped_weights(moments: "np.ndarray") -> "np.ndarray":
    rho = moments[:, 0]
    r2 = moments[:, 1]
    return 1.0 + (1.0 - rho) / (1.0 - r2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test configurations."""
    return [{'setup': 'import numpy as np\nmoments = np.array([[.44,.392],[.75,.0625]])',
      'call': 'compute_stopped_weights(moments.copy())',
      'gold_call': '_oracle_compute_stopped_weights(moments.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\nmoments = np.array([[1.,0.]])',
      'call': 'compute_stopped_weights(moments.copy())',
      'gold_call': '_oracle_compute_stopped_weights(moments.copy())',
      'tol': 1e-10},
     {'setup': 'import numpy as np\nmoments = np.array([[.1009,.8982009],[.5,.25]])',
      'call': 'compute_stopped_weights(moments.copy())',
      'gold_call': '_oracle_compute_stopped_weights(moments.copy())',
      'tol': 1e-10}]
