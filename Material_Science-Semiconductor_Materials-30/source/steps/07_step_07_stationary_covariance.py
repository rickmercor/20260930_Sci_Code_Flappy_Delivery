"""
Solve the stationary covariance of the cycle-start Gaussian state.

Intrinsic fluctuations persist between cycles and are transported by a nonnormal transition matrix.

Returns
-------
return covariance
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stationary_covariance(M, Qxx):
    """M is a real square matrix of dimension d>=1, with spectral radius<=.98.
    Qxx is real symmetric positive semidefinite with shape(d,d). Return symmetric
    P(d,d) satisfying P=M P M.T+Qxx. Zero and singular Qxx are supported. M may be
    nonnormal and may have negative eigenvalues. P is the covariance of the scaled
    cycle-start population fluctuation, with no extra population-size factor.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return covariance

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_stationary_covariance(M, Qxx):
    import numpy as np
    from scipy.linalg import solve_discrete_lyapunov
    P = solve_discrete_lyapunov(np.asarray(M, float), np.asarray(Qxx, float))
    return (P+P.T)/2.

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'dense',
      'setup': 'import numpy as np\n'
               'M=np.array([[.5,.1],[-.2,.4]])\n'
               'Qxx=np.array([[.3,.06],[.06,.2]])\n',
      'call': 'stationary_covariance(M,Qxx)',
      'gold_call': '_oracle_stationary_covariance(M,Qxx)',
      'tol': 2e-07},
     {'name': 'nonnormal',
      'setup': 'import numpy as np\nM=np.array([[.95,1.2],[0.,.7]])\nQxx=np.diag([.02,.01])\n',
      'call': 'stationary_covariance(M,Qxx)',
      'gold_call': '_oracle_stationary_covariance(M,Qxx)',
      'tol': 2e-07},
     {'name': 'zero_noise',
      'setup': 'import numpy as np\nM=np.array([[.5,.7],[0.,.9]])\nQxx=np.zeros((2,2))\n',
      'call': 'stationary_covariance(M,Qxx)',
      'gold_call': '_oracle_stationary_covariance(M,Qxx)',
      'tol': 2e-07},
     {'name': 'scalar',
      'setup': 'import numpy as np\nM=np.array([[.98]])\nQxx=np.array([[.2]])\n',
      'call': 'stationary_covariance(M,Qxx)',
      'gold_call': '_oracle_stationary_covariance(M,Qxx)',
      'tol': 2e-07},
     {'name': 'negative_modes',
      'setup': 'import numpy as np\nM=np.array([[-.8,.1],[0.,-.3]])\nQxx=np.array([[.4,.1],[.1,.2]])\n',
      'call': 'stationary_covariance(M,Qxx)',
      'gold_call': '_oracle_stationary_covariance(M,Qxx)',
      'tol': 2e-07}]
