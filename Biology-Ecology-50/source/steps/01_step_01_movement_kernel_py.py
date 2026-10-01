"""
Construct a zero-inflated, barrier-adjusted daily movement matrix.

Adult mosquitoes may remain at their current site or disperse to another site on a given day. The zero-inflated dispersal model assigns a separate staying probability \(p_0\); distance and barrier permeability determine how the remaining probability is distributed among other sites.

Returns
-------
An \((N,N)\) nonnegative, row-stochastic NumPy array of daily adult movement probabilities. Every diagonal entry equals \(p_0\).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def movement_kernel(coords: "np.ndarray", lam: float, barrier_x: float, delta: float, p0: float) -> "np.ndarray":
    """Construct a zero-inflated, barrier-adjusted daily movement matrix.

Parameters
----------
coords : float array of shape (N,2), at least two distinct locations
lam : positive inverse-distance scale conditional on leaving a site
barrier_x : vertical barrier x coordinate
delta : crossing factor in (0,1]
p0 : daily probability of staying at the origin, in [0,1)

Returns
-------
(N,N) nonnegative row-stochastic array, with diagonal entries equal to p0.

Notes
-----
For j != i, use distance-dependent exponential weights with a factor delta on cross-barrier movements, normalize only over j != i, then allocate total probability 1-p0 to those destinations. A node on the barrier lies on the right. This is the zero-inflated kernel in article Eq. 2 together with the barrier adjustment; p0 is exactly the diagonal probability, so do not renormalize the diagonal with off-diagonal weights.
"""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_movement_kernel(coords: "np.ndarray", lam: float, barrier_x: float, delta: float, p0: float) -> "np.ndarray":
    pts=np.asarray(coords,dtype=float)
    dist=np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2)
    sides=pts[:,0]<barrier_x
    weights=np.exp(-lam*dist)*np.where(sides[:,None]!=sides[None,:],delta,1.)
    np.fill_diagonal(weights,0.)
    M=(1.-p0)*weights/weights.sum(axis=1,keepdims=True)
    np.fill_diagonal(M,p0)
    return M

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Three independent source-method scenarios."""
    return [{'setup': 'import numpy as np\ncoords=np.array([[0.0, 0.0], [0.0, 20.0], [20.0, 0.0], [20.0, 20.0]],dtype=float)\n', 'call': 'movement_kernel(coords,0.06535947712418301,10.,0.47,0.73)', 'gold_call': '_oracle_movement_kernel(coords,0.06535947712418301,10.,0.47,0.73)', 'tol': 1e-09}, {'setup': 'import numpy as np\ncoords=np.array([[0.0, 0.0], [0.0, 20.0], [20.0, 0.0], [20.0, 20.0]],dtype=float)\n', 'call': 'movement_kernel(coords,0.06535947712418301,10.,1.0,0.25)', 'gold_call': '_oracle_movement_kernel(coords,0.06535947712418301,10.,1.0,0.25)', 'tol': 1e-09}, {'setup': 'import numpy as np\ncoords=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],dtype=float)\n', 'call': 'movement_kernel(coords,0.05,10.,0.2,0.91)', 'gold_call': '_oracle_movement_kernel(coords,0.05,10.,0.2,0.91)', 'tol': 1e-09}]
