"""
Return the Gaussian-averaged contact force response matrix for a normalized isotropic three-dimensional Gaussian pair potential with squared width radius**2/3 and amplitude strength. The input is a symmetric nonnegative squared-separation matrix with zero diagonal; radius is positive and strength nonnegative. Return minus one third of the Gaussian-averaged trace of the pair Hessian, so a repulsive kernel gives negative off-diagonal entries. Diagonal response ensures zero row sums, with self interactions excluded. Raise ValueError for invalid inputs.

Return the Gaussian-averaged contact force response matrix for a normalized isotropic three-dimensional Gaussian pair potential with squared width radius**2/3 and amplitude strength. The input is a symmetric nonnegative squared-separation matrix with zero diagonal; radius is positive and strength nonnegative. Return minus one third of the Gaussian-averaged trace of the pair Hessian, so a repulsive kernel gives negative off-diagonal entries. Diagonal response ensures zero row sums, with self interactions excluded. Raise ValueError for invalid inputs.

Returns
-------
return response
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contact_response(squared_separations, radius, strength):
    """Return the Gaussian-averaged contact force response matrix for a normalized isotropic three-dimensional Gaussian pair potential with squared width radius**2/3 and amplitude strength. The input is a symmetric nonnegative squared-separation matrix with zero diagonal; radius is positive and strength nonnegative. Return minus one third of the Gaussian-averaged trace of the pair Hessian, so a repulsive kernel gives negative off-diagonal entries. Diagonal response ensures zero row sums, with self interactions excluded. Raise ValueError for invalid inputs."""
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_contact_response(squared_separations, radius, strength):
    import numpy as np
    d=np.asarray(squared_separations,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or not np.all(np.isfinite(d)) or np.min(d)<0 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0) or not np.isfinite(radius) or radius<=0 or not np.isfinite(strength) or strength<0:
        raise ValueError('invalid contact inputs')
    p=(2*np.pi*(radius**2+d)/3)**(-1.5)
    s=-3*strength*p/(radius**2+d)
    np.fill_diagonal(s,0)
    s-=np.diag(s.sum(axis=1))
    return s

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np', 'call': 'contact_response(np.array([[0.,1.],[1.,0.]]),0.5,1.)', 'gold_call': '_oracle_contact_response(np.array([[0.,1.],[1.,0.]]),0.5,1.)'}, {'setup': 'import numpy as np', 'call': 'contact_response(np.zeros((2,2)),0.5,1.)', 'gold_call': '_oracle_contact_response(np.zeros((2,2)),0.5,1.)'}, {'setup': 'import numpy as np', 'call': 'contact_response(np.array([[0.,6.],[6.,0.]]),0.5,2.5)', 'gold_call': '_oracle_contact_response(np.array([[0.,6.],[6.,0.]]),0.5,2.5)'}, {'setup': 'import numpy as np\ndef _invalid_result(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1', 'call': '_invalid_result(lambda: contact_response(np.eye(2),-1.,1.))', 'gold_call': '_invalid_result(lambda: _oracle_contact_response(np.eye(2),-1.,1.))'}]
