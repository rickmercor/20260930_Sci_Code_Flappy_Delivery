"""
Shift factorial-normalized locals toward a destination particle.

Local coefficients are the factorial-normalized Taylor coefficients of the potential about their expansion center.

Returns
-------
np.ndarray, same shape as locals_in
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def translate_locals(locals_in: "np.ndarray", displacement: "np.ndarray", p: int) -> "np.ndarray":
    """Translate a local polynomial from an old center to a new center.

    Parameters
    ----------
    locals_in : np.ndarray
        Normalized local coefficients through p, in kernel_derivatives order.
    displacement : np.ndarray
        New center minus old center, shape (3,).
    p : int
        Total polynomial degree, 0 <= p <= 6.

    Returns
    -------
    result : np.ndarray
        Same-shape normalized local coefficients L_n = (1/n!) d^n P at the
        new center, where P is the degree-p local polynomial that locals_in
        represents about the old center.
        Coefficient 000 is potential and minus the three first-degree
        coefficients is acceleration per unit target mass.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_translate_locals(locals_in: "np.ndarray", displacement: "np.ndarray", p: int) -> "np.ndarray":
    indices = _indices(p)
    result = np.zeros(len(indices))
    for i,n in enumerate(indices):
        for j,k in enumerate(indices):
            if all(k[a]>=n[a] for a in range(3)):
                result[i] += _binomial(k,n)*_power(displacement,np.subtract(k,n))*locals_in[j]
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [{'setup': 'import numpy as np\nl=np.linspace(-.3,.2,35)\nd=np.array([.1,.2,-.1])', 'call': 'translate_locals(l.copy(), d.copy(), 4)', 'gold_call': '_oracle_translate_locals(l.copy(), d.copy(), 4)', 'tol': 1e-10}, {'setup': 'import numpy as np\nl=np.array([-.8])\nd=np.array([.2,.3,.4])', 'call': 'translate_locals(l.copy(), d.copy(), 0)', 'gold_call': '_oracle_translate_locals(l.copy(), d.copy(), 0)', 'tol': 1e-10}, {'setup': 'import numpy as np\nl=np.arange(20,dtype=float)/11\nd=np.zeros(3)', 'call': 'translate_locals(l.copy(), d.copy(), 3)', 'gold_call': '_oracle_translate_locals(l.copy(), d.copy(), 3)', 'tol': 1e-10}]
