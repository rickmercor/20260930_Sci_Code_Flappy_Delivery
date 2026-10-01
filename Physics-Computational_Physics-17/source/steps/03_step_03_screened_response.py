"""
Return the isotropic Gaussian-averaged force response of nonreciprocal screened Coulomb interactions: force on j is minus the sum of B[j,i] times the gradient with respect to r_j of exp(-distance/ell)/distance. The squared separation in the averaged response is regularized by adding radius**2. couplings is a finite square matrix with zero diagonal; its first index is the responding bead. Exclude self forces and impose zero row sums. radius and screening_length are positive. Raise ValueError for invalid inputs.

Return the isotropic Gaussian-averaged force response of nonreciprocal screened Coulomb interactions: force on j is minus the sum of B[j,i] times the gradient with respect to r_j of exp(-distance/ell)/distance. The squared separation in the averaged response is regularized by adding radius**2. couplings is a finite square matrix with zero diagonal; its first index is the responding bead. Exclude self forces and impose zero row sums. radius and screening_length are positive. Raise ValueError for invalid inputs.

Returns
-------
return response
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def screened_response(squared_separations, radius, screening_length, couplings):
    """Return the isotropic Gaussian-averaged force response of nonreciprocal screened Coulomb interactions: force on j is minus the sum of B[j,i] times the gradient with respect to r_j of exp(-distance/ell)/distance. The squared separation in the averaged response is regularized by adding radius**2. couplings is a finite square matrix with zero diagonal; its first index is the responding bead. Exclude self forces and impose zero row sums. radius and screening_length are positive. Raise ValueError for invalid inputs."""
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_screened_response(squared_separations, radius, screening_length, couplings):
    import numpy as np
    from scipy.special import erfcx
    d=np.asarray(squared_separations,dtype=float); b=np.asarray(couplings,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or b.shape!=d.shape or not np.all(np.isfinite(d)) or not np.all(np.isfinite(b)) or np.min(d)<0 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0) or not np.allclose(np.diag(b),0) or not np.isfinite(radius) or radius<=0 or not np.isfinite(screening_length) or screening_length<=0:
        raise ValueError('invalid screening inputs')
    z=(d+radius**2)/(6*screening_length**2)
    k=-(4*np.pi*z**3)**(-0.5)+(np.pi*z)**(-0.5)-erfcx(np.sqrt(z))
    u=b*k/(3*screening_length**3)
    np.fill_diagonal(u,0)
    u-=np.diag(u.sum(axis=1))
    return u

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np', 'call': 'screened_response(np.array([[0.,1.],[1.,0.]]),0.5,2.,np.array([[0.,1.],[-3.,0.]]))', 'gold_call': '_oracle_screened_response(np.array([[0.,1.],[1.,0.]]),0.5,2.,np.array([[0.,1.],[-3.,0.]]))'}, {'setup': 'import numpy as np', 'call': 'screened_response(np.zeros((2,2)),0.5,2.,np.array([[0.,1.],[-3.,0.]]))', 'gold_call': '_oracle_screened_response(np.zeros((2,2)),0.5,2.,np.array([[0.,1.],[-3.,0.]]))'}, {'setup': 'import numpy as np', 'call': 'screened_response(np.array([[0.,6.],[6.,0.]]),0.5,2.,np.array([[0.,2.],[-0.5,0.]]))', 'gold_call': '_oracle_screened_response(np.array([[0.,6.],[6.,0.]]),0.5,2.,np.array([[0.,2.],[-0.5,0.]]))'}, {'setup': 'import numpy as np\ndef _invalid_result(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1', 'call': '_invalid_result(lambda: screened_response(np.zeros((2,2)),0.5,0.,np.zeros((2,2))))', 'gold_call': '_invalid_result(lambda: _oracle_screened_response(np.zeros((2,2)),0.5,0.,np.zeros((2,2))))'}]
