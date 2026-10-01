"""
Return the unequal-time dot-product cross-covariance with the earlier time held fixed. earlier_covariance is the equal-time covariance at that earlier time. coefficient_function(X) returns the current (drift, covariance injection) as a numeric (2,n,n) array. The same-time covariance continues evolving during lag, and the unequal-time covariance follows the causal Gaussian dynamics. Return only the final unequal-time matrix, with relative accuracy 1e-9 or better and absolute accuracy 1e-11 or better. lag is finite and nonnegative and earlier_covariance is finite symmetric positive semidefinite. Raise ValueError for invalid inputs.

Return the unequal-time dot-product cross-covariance with the earlier time held fixed. earlier_covariance is the equal-time covariance at that earlier time. coefficient_function(X) returns the current (drift, covariance injection) as a numeric (2,n,n) array. The same-time covariance continues evolving during lag, and the unequal-time covariance follows the causal Gaussian dynamics. Return only the final unequal-time matrix, with relative accuracy 1e-9 or better and absolute accuracy 1e-11 or better. lag is finite and nonnegative and earlier_covariance is finite symmetric positive semidefinite. Raise ValueError for invalid inputs.

Returns
-------
return response
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evolve_cross_covariance(earlier_covariance, lag, coefficient_function):
    """Return the unequal-time dot-product cross-covariance with the earlier time held fixed. earlier_covariance is the equal-time covariance at that earlier time. coefficient_function(X) returns the current (drift, covariance injection) as a numeric (2,n,n) array. The same-time covariance continues evolving during lag, and the unequal-time covariance follows the causal Gaussian dynamics. Return only the final unequal-time matrix, with relative accuracy 1e-9 or better and absolute accuracy 1e-11 or better. lag is finite and nonnegative and earlier_covariance is finite symmetric positive semidefinite. Raise ValueError for invalid inputs."""
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evolve_cross_covariance(earlier_covariance, lag, coefficient_function):
    import numpy as np
    from scipy.integrate import solve_ivp
    x=np.asarray(earlier_covariance,dtype=float)
    if x.ndim!=2 or x.shape[0]!=x.shape[1] or not np.all(np.isfinite(x)) or not np.allclose(x,x.T) or np.linalg.eigvalsh(x).min()<-1e-10 or not np.isfinite(lag) or lag<0 or not callable(coefficient_function):
        raise ValueError('invalid cross-covariance inputs')
    if lag==0: return x.copy()
    def rhs(t,y):
        same,cross=y.reshape((2,)+x.shape)
        j,q=coefficient_function(same)
        return np.stack((j@same+same@j.T+q,j@cross)).ravel()
    sol=solve_ivp(rhs,(0.,lag),np.stack((x,x)).ravel(),method='DOP853',rtol=1e-11,atol=1e-13)
    if not sol.success: raise ValueError('integration failed')
    return sol.y[:,-1].reshape((2,)+x.shape)[1]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ncf=lambda x: np.stack((np.array([[-1.,0.3],[0.,-2.]]),np.eye(2)))', 'call': 'evolve_cross_covariance(np.eye(2),1.,cf)', 'gold_call': '_oracle_evolve_cross_covariance(np.eye(2),1.,cf)'}, {'setup': 'import numpy as np\ncf=lambda x: np.stack((np.array([[-1.,0.3],[0.,-2.]]),np.eye(2)))', 'call': 'evolve_cross_covariance(np.eye(2),0.,cf)', 'gold_call': '_oracle_evolve_cross_covariance(np.eye(2),0.,cf)'}, {'setup': 'import numpy as np\ncf=lambda x: np.stack((-np.diag(1+np.diag(x)),np.eye(2)))', 'call': 'evolve_cross_covariance(np.diag([1.,2.]),2.,cf)', 'gold_call': '_oracle_evolve_cross_covariance(np.diag([1.,2.]),2.,cf)'}, {'setup': 'import numpy as np\ncf=lambda x: np.stack((np.array([[-1.,0.3],[0.,-2.]]),np.array([[1.,0.2],[0.2,0.6]])))', 'call': 'evolve_cross_covariance(np.array([[2.,0.4],[0.4,1.]]),0.7,cf)', 'gold_call': '_oracle_evolve_cross_covariance(np.array([[2.,0.4],[0.4,1.]]),0.7,cf)'}, {'setup': 'import numpy as np\ndef _invalid_result(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1', 'call': '_invalid_result(lambda: evolve_cross_covariance(np.eye(2),-1.,lambda x: np.stack((-np.eye(2),np.eye(2)))))', 'gold_call': '_invalid_result(lambda: _oracle_evolve_cross_covariance(np.eye(2),-1.,lambda x: np.stack((-np.eye(2),np.eye(2)))))'}]
