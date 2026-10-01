"""
Final orchestrator: compute Q for the twelve-bead task configuration. activity replaces C[3,3] in one-based notation; attraction replaces B[9,3]; all other parameters are those of the task. Initial independent bonds have unit mean squared length, radius 0.5, spring stiffness 0.5, contact strength 1, self-mobility 1, screening length 2, B[3,9]=1, earlier time 1 and later time 2. An independent spatial Gaussian velocity field has amplitude 4 and correlation length 1.5; its Cartesian covariance is amplitude/3 times exp(-r*r/(2*length**2)) times the identity and temporal delta function. activity is finite nonnegative and attraction finite. Raise ValueError for invalid inputs.

Final orchestrator: compute Q for the twelve-bead task configuration. activity replaces C[3,3] in one-based notation; attraction replaces B[9,3]; all other parameters are those of the task. Initial independent bonds have unit mean squared length, radius 0.5, spring stiffness 0.5, contact strength 1, self-mobility 1, screening length 2, B[3,9]=1, earlier time 1 and later time 2. An independent spatial Gaussian velocity field has amplitude 4 and correlation length 1.5; its Cartesian covariance is amplitude/3 times exp(-r*r/(2*length**2)) times the identity and temporal delta function. activity is finite nonnegative and attraction finite. Raise ValueError for invalid inputs.

Returns
-------
return response
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def correlation_asymmetry(activity=4.0, attraction=-3.0):
    """Final orchestrator: compute Q for the twelve-bead task configuration. activity replaces C[3,3] in one-based notation; attraction replaces B[9,3]; all other parameters are those of the task. Initial independent bonds have unit mean squared length, radius 0.5, spring stiffness 0.5, contact strength 1, self-mobility 1, screening length 2, B[3,9]=1, earlier time 1 and later time 2. An independent spatial Gaussian velocity field has amplitude 4 and correlation length 1.5; its Cartesian covariance is amplitude/3 times exp(-r*r/(2*length**2)) times the identity and temporal delta function. activity is finite nonnegative and attraction finite. Raise ValueError for invalid inputs."""
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_correlation_asymmetry(activity=4.0, attraction=-3.0):
    import numpy as np
    if not np.isfinite(activity) or activity<0 or not np.isfinite(attraction):
        raise ValueError('invalid instance parameters')
    n=12
    p=np.eye(n)-np.ones((n,n))/n
    d=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    x0=-0.5*p@d@p
    k=0.5*(np.eye(n,k=1)+np.eye(n,k=-1)); k-=np.diag(k.sum(axis=1))
    c=np.eye(n); c[2,2]=activity
    b=np.zeros((n,n)); b[8,2]=attraction; b[2,8]=1.
    def coefficients(x):
        d=np.maximum(np.diag(x)[:,None]+np.diag(x)[None,:]-x-x.T,0)
        np.fill_diagonal(d,0)
        s=_oracle_contact_response(d,0.5,1.)
        m=_oracle_averaged_mobility(d,0.5,1.)
        u=_oracle_screened_response(d,0.5,2.,b)
        coefficients=_oracle_drift_and_noise(m,k,s,u,c)
        coefficients[1]+=_oracle_averaged_spatial_covariance(d,4.,1.5)
        return coefficients
    x1=_oracle_evolve_covariance(x0,1.,coefficients)
    y=_oracle_evolve_cross_covariance(x1,1.,coefficients)
    u=np.eye(n)[2]-np.eye(n)[1]; v=np.eye(n)[8]-np.eye(n)[7]
    return float(u@y@v-v@y@u)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'correlation_asymmetry(4.,-3.)', 'gold_call': '_oracle_correlation_asymmetry(4.,-3.)'}, {'setup': '', 'call': 'correlation_asymmetry(1.,-3.)', 'gold_call': '_oracle_correlation_asymmetry(1.,-3.)'}, {'setup': '', 'call': 'correlation_asymmetry(4.,0.)', 'gold_call': '_oracle_correlation_asymmetry(4.,0.)'}, {'setup': 'import numpy as np\ndef _invalid_result(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1', 'call': '_invalid_result(lambda: correlation_asymmetry(-1.,-3.))', 'gold_call': '_invalid_result(lambda: _oracle_correlation_asymmetry(-1.,-3.))'}]
