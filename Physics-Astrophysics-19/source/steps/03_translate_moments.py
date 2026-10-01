"""
Translate arbitrary raw multipoles between fixed expansion centers.

The translation applies to monopoles and to higher moments.

Returns
-------
np.ndarray, same shape as moments
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def translate_moments(moments: "np.ndarray", displacement: "np.ndarray", p: int) -> "np.ndarray":
    """Translate raw source moments from old center to new center.

    Parameters
    ----------
    moments : np.ndarray
        Finite raw moments through total degree p, in kernel_derivatives order.
    displacement : np.ndarray
        Old center minus new center, shape (3,).
    p : int
        Total order, 0 <= p <= 6.

    Returns
    -------
    result : np.ndarray
        Raw moments of the same source distribution about the new center,
        with the same shape and kernel_derivatives order.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_translate_moments(moments: "np.ndarray", displacement: "np.ndarray", p: int) -> "np.ndarray":
    indices = _indices(p)
    result = np.zeros(len(indices))
    for i,n in enumerate(indices):
        for j,k in enumerate(indices):
            if all(k[a]<=n[a] for a in range(3)):
                result[i] += _binomial(n,k)*_power(displacement, np.subtract(n,k))*moments[j]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [{'setup': 'import numpy as np\nq=np.arange(20,dtype=float)/13\nd=np.array([.2,-.1,.3])', 'call': 'translate_moments(q.copy(), d.copy(), 3)', 'gold_call': '_oracle_translate_moments(q.copy(), d.copy(), 3)', 'tol': 1e-10}, {'setup': 'import numpy as np\nq=np.array([2.])\nd=np.array([.3,.4,.5])', 'call': 'translate_moments(q.copy(), d.copy(), 0)', 'gold_call': '_oracle_translate_moments(q.copy(), d.copy(), 0)', 'tol': 1e-10}, {'setup': 'import numpy as np\nq=np.linspace(-.4,.6,35)\nd=np.zeros(3)', 'call': 'translate_moments(q.copy(), d.copy(), 4)', 'gold_call': '_oracle_translate_moments(q.copy(), d.copy(), 4)', 'tol': 1e-10}]
