"""
Return the equal-time dot-product covariance at end_time, starting at time zero. coefficient_function(X) returns a (2,n,n) array containing the current preaveraged drift and covariance injection for a supplied covariance X. Use the continuous Gaussian covariance dynamics with relative numerical accuracy 1e-9 or better and absolute accuracy 1e-11 or better. The initial covariance is finite symmetric positive semidefinite; end_time is finite and nonnegative. Raise ValueError for invalid inputs.

Return the equal-time dot-product covariance at end_time, starting at time zero. coefficient_function(X) returns a (2,n,n) array containing the current preaveraged drift and covariance injection for a supplied covariance X. Use the continuous Gaussian covariance dynamics with relative numerical accuracy 1e-9 or better and absolute accuracy 1e-11 or better. The initial covariance is finite symmetric positive semidefinite; end_time is finite and nonnegative. Raise ValueError for invalid inputs.

Returns
-------
return response
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evolve_covariance(initial_covariance, end_time, coefficient_function):
    """Return the equal-time dot-product covariance at end_time, starting at time zero. coefficient_function(X) returns a (2,n,n) array containing the current preaveraged drift and covariance injection for a supplied covariance X. Use the continuous Gaussian covariance dynamics with relative numerical accuracy 1e-9 or better and absolute accuracy 1e-11 or better. The initial covariance is finite symmetric positive semidefinite; end_time is finite and nonnegative. Raise ValueError for invalid inputs."""
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evolve_covariance(initial_covariance, end_time, coefficient_function):
    import numpy as np
    from scipy.integrate import solve_ivp
    x=np.asarray(initial_covariance,dtype=float)
    if x.ndim!=2 or x.shape[0]!=x.shape[1] or not np.all(np.isfinite(x)) or not np.allclose(x,x.T) or np.linalg.eigvalsh(x).min()<-1e-10 or not np.isfinite(end_time) or end_time<0 or not callable(coefficient_function):
        raise ValueError('invalid evolution inputs')
    if end_time==0: return x.copy()
    def rhs(t,y):
        state=y.reshape(x.shape)
        j,q=coefficient_function(state)
        return (j@state+state@j.T+q).ravel()
    sol=solve_ivp(rhs,(0.,end_time),x.ravel(),method='DOP853',rtol=1e-11,atol=1e-13)
    if not sol.success: raise ValueError('integration failed')
    return sol.y[:,-1].reshape(x.shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ncf=lambda x: np.stack((-np.eye(2),np.eye(2)))', 'call': 'evolve_covariance(np.eye(2),1.,cf)', 'gold_call': '_oracle_evolve_covariance(np.eye(2),1.,cf)'}, {'setup': 'import numpy as np\ncf=lambda x: np.stack((-np.eye(2),np.eye(2)))', 'call': 'evolve_covariance(np.eye(2),0.,cf)', 'gold_call': '_oracle_evolve_covariance(np.eye(2),0.,cf)'}, {'setup': 'import numpy as np\ncf=lambda x: np.stack((-np.diag(1+np.diag(x)),np.eye(2)))', 'call': 'evolve_covariance(np.zeros((2,2)),2.,cf)', 'gold_call': '_oracle_evolve_covariance(np.zeros((2,2)),2.,cf)'}, {'setup': 'import numpy as np\ncf=lambda x: np.stack((np.array([[-1.,0.3],[0.,-2.]]),np.array([[1.,0.2],[0.2,0.6]])))', 'call': 'evolve_covariance(np.array([[2.,0.4],[0.4,1.]]),0.7,cf)', 'gold_call': '_oracle_evolve_covariance(np.array([[2.,0.4],[0.4,1.]]),0.7,cf)'}, {'setup': 'import numpy as np\ndef _invalid_result(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1', 'call': '_invalid_result(lambda: evolve_covariance(np.eye(2),-1.,lambda x: np.stack((-np.eye(2),np.eye(2)))))', 'gold_call': '_invalid_result(lambda: _oracle_evolve_covariance(np.eye(2),-1.,lambda x: np.stack((-np.eye(2),np.eye(2)))))'}]
