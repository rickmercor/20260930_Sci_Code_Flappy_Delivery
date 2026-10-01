"""
Aggregate monopole particles into raw Cartesian source moments.

Equation (18) uses unnormalized moments: factorial factors enter the
interaction operator rather than this accumulation.

Returns
-------
np.ndarray, shape ((p+1)(p+2)(p+3)/6,)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def particle_moments(x: "np.ndarray", masses: "np.ndarray", center: "np.ndarray", p: int) -> "np.ndarray":
    """Accumulate Q_n=sum_j masses_j (x_j-center)^n.

    Parameters
    ----------
    x : np.ndarray
        Finite positions, shape (N,3), including N=0.
    masses : np.ndarray
        Finite signed weights, shape (N,).
    center : np.ndarray
        Fixed expansion center, shape (3,).
    p : int
        Total order, 0 <= p <= 6.

    Returns
    -------
    result : np.ndarray
        Raw moment vector, ordered as kernel_derivatives. Empty input gives
        zeros. The zeroth component is the sum of the weights.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_particle_moments(x: "np.ndarray", masses: "np.ndarray", center: "np.ndarray", p: int) -> "np.ndarray":
    offsets = np.asarray(x, dtype=float) - np.asarray(center)
    masses = np.asarray(masses, dtype=float)
    return np.array([np.sum(masses*np.prod(offsets**np.array(n),axis=1)) for n in _indices(p)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [{'setup': 'import numpy as np\nx=np.array([[.2,-.1,.3],[-.4,.5,.1]])\nm=np.array([.8,1.2])\nc=np.array([.1,.1,0.])', 'call': 'particle_moments(x.copy(), m.copy(), c.copy(), 4)', 'gold_call': '_oracle_particle_moments(x.copy(), m.copy(), c.copy(), 4)', 'tol': 1e-10}, {'setup': 'import numpy as np\nx=np.empty((0,3))\nm=np.empty(0)\nc=np.ones(3)', 'call': 'particle_moments(x.copy(), m.copy(), c.copy(), 2)', 'gold_call': '_oracle_particle_moments(x.copy(), m.copy(), c.copy(), 2)', 'tol': 1e-10}, {'setup': 'import numpy as np\nx=np.array([[1.,0.,0.],[-1.,0.,0.]])\nm=np.array([1.,-1.])\nc=np.zeros(3)', 'call': 'particle_moments(x.copy(), m.copy(), c.copy(), 3)', 'gold_call': '_oracle_particle_moments(x.copy(), m.copy(), c.copy(), 3)', 'tol': 1e-10}]
