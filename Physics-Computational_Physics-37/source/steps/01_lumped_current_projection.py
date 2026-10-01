"""
Recover the ordered nodal current proxy from a lumped L2 load.

Use the diagonal mass-lumped L2 projection. For node i with positive lumped mass m_i and Cartesian weak current load b_i, compute



J_i = b_i / m_i.



Division is componentwise across the three Cartesian components. Return the n-by-3 array J in the original node order.

Returns
-------
Return one n-by-3 real NumPy array in node order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def lumped_current_projection(lumped_mass, current_load):
    """Project a vector load with the lumped mass.

    Parameters
    ----------
    lumped_mass : array_like, shape (n,)
        Positive nodal lumped masses.
    current_load : array_like, shape (n,3)
        Cartesian weak curl loads in node order.
    Returns
    -------
    ndarray, shape (n,3)
        Projected Cartesian nodal current in the same order.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_lumped_current_projection(lumped_mass, current_load):
    """Apply the paper's lumped L2 projection (30) to vector curl loads."""
    import numpy as np
    mass = np.asarray(lumped_mass, dtype=float)
    load = np.asarray(current_load, dtype=float)
    if mass.ndim != 1 or mass.size == 0 or load.shape != (mass.size, 3):
        raise ValueError("mass must have shape (n,) and curl_load shape (n,3)")
    if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(load)) or np.any(mass <= 0):
        raise ValueError("inputs must be finite and masses positive")
    return (load / mass[:, None]).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nm=np.array([.2,.8]); b=np.array([[.1,.2,.3],[.8,-.4,.0]])', 'call': 'lumped_current_projection(m,b)', 'gold_call': '_oracle_lumped_current_projection(m,b)'}, {'setup': 'import numpy as np\nm=np.array([1.]); b=np.array([[3.,-2.,5.]])', 'call': 'lumped_current_projection(m,b)', 'gold_call': '_oracle_lumped_current_projection(m,b)'}, {'setup': 'import numpy as np\nm=np.array([.1,.3,.6]); b=np.arange(9.,dtype=float).reshape(3,3)', 'call': 'lumped_current_projection(m,b)', 'gold_call': '_oracle_lumped_current_projection(m,b)'}, {'setup': 'import numpy as np\nm=np.array([.25,.25,.5]); b=np.eye(3)', 'call': 'lumped_current_projection(m,b)', 'gold_call': '_oracle_lumped_current_projection(m,b)'}, {'setup': 'import numpy as np\nm=np.array([2.,3.]); b=np.array([[-1.,4.,2.],[6.,0.,-3.]])', 'call': 'lumped_current_projection(m,b)', 'gold_call': '_oracle_lumped_current_projection(m,b)'}, {'setup': 'import numpy as np\nm=np.array([.4,.35,.25]); j=np.array([[1.,2.,3.],[-1.,0.,2.],[.5,.25,-.5]]); b=m[:,None]*j', 'call': 'lumped_current_projection(m,b)', 'gold_call': '_oracle_lumped_current_projection(m,b)'}, {'setup': 'import numpy as np\nm=np.array([.05,.15,.3,.5]); b=np.linspace(-.6,.7,12).reshape(4,3)', 'call': 'lumped_current_projection(m,b)', 'gold_call': '_oracle_lumped_current_projection(m,b)'}]
