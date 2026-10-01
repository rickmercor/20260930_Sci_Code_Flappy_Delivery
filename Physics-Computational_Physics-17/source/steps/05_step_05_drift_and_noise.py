"""
Return a numeric array of shape (2,n,n) containing the preaveraged drift and the monomer force-noise contribution to the dot-product covariance injection for the isotropic Gaussian polymer. mobility is symmetric; spring, contact, chemical are force-response matrices; force_covariance uses the convention that Cartesian noise covariance is force_covariance/3. Every input must be a same-size finite square matrix. Raise ValueError for invalid inputs.

Return a numeric array of shape (2,n,n) containing the preaveraged drift and the monomer force-noise contribution to the dot-product covariance injection for the isotropic Gaussian polymer. mobility is symmetric; spring, contact, chemical are force-response matrices; force_covariance uses the convention that Cartesian noise covariance is force_covariance/3. Every input must be a same-size finite square matrix. Raise ValueError for invalid inputs.

Returns
-------
return response
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def drift_and_noise(mobility, spring, contact, chemical, force_covariance):
    """Return a numeric array of shape (2,n,n) containing the preaveraged drift and the monomer force-noise contribution to the dot-product covariance injection for the isotropic Gaussian polymer. mobility is symmetric; spring, contact, chemical are force-response matrices; force_covariance uses the convention that Cartesian noise covariance is force_covariance/3. Every input must be a same-size finite square matrix. Raise ValueError for invalid inputs."""
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_drift_and_noise(mobility, spring, contact, chemical, force_covariance):
    import numpy as np
    m,k,s,u,c=[np.asarray(v,dtype=float) for v in (mobility,spring,contact,chemical,force_covariance)]
    if m.ndim!=2 or m.shape[0]!=m.shape[1] or any(v.shape!=m.shape or not np.all(np.isfinite(v)) for v in (m,k,s,u,c)) or not np.allclose(m,m.T) or not np.allclose(c,c.T):
        raise ValueError('invalid coefficient inputs')
    return np.stack((m@(k+s+u),m@c@m))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np', 'call': 'drift_and_noise(np.eye(2),-np.eye(2),np.zeros((2,2)),np.zeros((2,2)),np.eye(2))', 'gold_call': '_oracle_drift_and_noise(np.eye(2),-np.eye(2),np.zeros((2,2)),np.zeros((2,2)),np.eye(2))'}, {'setup': 'import numpy as np', 'call': 'drift_and_noise(np.array([[1.,0.2],[0.2,1.]]),-np.eye(2),np.zeros((2,2)),np.array([[1.,-1.],[0.,0.]]),np.diag([4.,1.]))', 'gold_call': '_oracle_drift_and_noise(np.array([[1.,0.2],[0.2,1.]]),-np.eye(2),np.zeros((2,2)),np.array([[1.,-1.],[0.,0.]]),np.diag([4.,1.]))'}, {'setup': 'import numpy as np', 'call': 'drift_and_noise(np.array([[1.,0.25],[0.25,1.5]]),np.array([[-0.5,0.5],[0.5,-0.5]]),np.array([[0.7,-0.7],[-0.7,0.7]]),np.array([[-0.2,0.2],[-0.4,0.4]]),np.diag([2.,0.5]))', 'gold_call': '_oracle_drift_and_noise(np.array([[1.,0.25],[0.25,1.5]]),np.array([[-0.5,0.5],[0.5,-0.5]]),np.array([[0.7,-0.7],[-0.7,0.7]]),np.array([[-0.2,0.2],[-0.4,0.4]]),np.diag([2.,0.5]))'}, {'setup': 'import numpy as np\ndef _invalid_result(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1', 'call': '_invalid_result(lambda: drift_and_noise(np.eye(2),np.eye(3),np.eye(2),np.eye(2),np.eye(2)))', 'gold_call': '_invalid_result(lambda: _oracle_drift_and_noise(np.eye(2),np.eye(3),np.eye(2),np.eye(2),np.eye(2)))'}]
