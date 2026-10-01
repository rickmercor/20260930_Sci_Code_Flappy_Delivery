"""
Expand the family-resolved capacitance difference to quadratic order.

The benchmark retains a centered quadratic detector acting on the Gaussian gate-charge process.

Returns
-------
return linear, quadratic
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def detector_coefficients(mean_gates, weights, beta, population):
    """mean_gates is length 2*m in gate-major order, weights lengthm positive summing1.
    beta>=0 is a scalar; population>0 is N. Every 1+beta*mean_gates entry is positive.
    For c(q)=(1+beta*q)^(-1/2), the measured family difference is
    weights[i]*(c(q_first_i)-c(q_second_i)). Let z be the sqrt(N)-scaled raw
    gate-charge fluctuation. Return L(m,2*m), H(m,2*m,2*m) defining
     y_i=L[i]@z+0.5*(z.T@H[i]@z-tr(H[i]@Cov(z))).
    L is the gradient of that difference at mean_gates. H is its Hessian divided
    by sqrt(population). Derivatives are ordinary derivatives, so do not put the
    Taylor factor 1/2 inside H. This specified quadratic model is used exactly;
    it is not a request for higher-order corrections to population dynamics.
    beta=0 gives zero coefficients. H[i] is symmetric.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return linear, quadratic

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_detector_coefficients(mean_gates, weights, beta, population):
    import numpy as np
    mean_gates = np.asarray(mean_gates, float)
    weights = np.asarray(weights, float)
    m = weights.size
    mean = mean_gates.reshape(2, m)
    first = -.5*beta*(1.+beta*mean)**(-1.5)
    second = .75*beta**2*(1.+beta*mean)**(-2.5)
    L = np.zeros((m, 2*m))
    H = np.zeros((m, 2*m, 2*m))
    for i in range(m):
        L[i, i], L[i, m+i] = weights[i]*first[0, i], -weights[i]*first[1, i]
        H[i, i, i] = weights[i]*second[0, i]/np.sqrt(population)
        H[i, m+i, m+i] = -weights[i]*second[1, i]/np.sqrt(population)
    return L, H

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'three_families',
      'setup': 'import numpy as np\n'
               'means=np.array([.4,.65,1.02,.36,.59,.94])\n'
               'weights=np.array([.46,.31,.23])\n'
               'beta=2.\n'
               'N=500.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(detector_coefficients(means,weights,beta,N))',
      'gold_call': '_check_result(_oracle_detector_coefficients(means,weights,beta,N))',
      'tol': 2e-07},
     {'name': 'zero_beta',
      'setup': 'import numpy as np\n'
               'means=np.array([.4,.65,1.02,.36,.59,.94])\n'
               'weights=np.array([.46,.31,.23])\n'
               'beta=2.\n'
               'N=500.\n'
               'beta=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(detector_coefficients(means,weights,beta,N))',
      'gold_call': '_check_result(_oracle_detector_coefficients(means,weights,beta,N))',
      'tol': 2e-07},
     {'name': 'one_family',
      'setup': 'import numpy as np\n'
               'means=np.array([.8,.3])\n'
               'weights=[1.]\n'
               'beta=.7\n'
               'N=100.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(detector_coefficients(means,weights,beta,N))',
      'gold_call': '_check_result(_oracle_detector_coefficients(means,weights,beta,N))',
      'tol': 2e-07},
     {'name': 'population_scaling',
      'setup': 'import numpy as np\n'
               'means=np.array([.4,.65,1.02,.36,.59,.94])\n'
               'weights=np.array([.46,.31,.23])\n'
               'beta=2.\n'
               'N=500.\n'
               'N=2000.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(detector_coefficients(means,weights,beta,N))',
      'gold_call': '_check_result(_oracle_detector_coefficients(means,weights,beta,N))',
      'tol': 2e-07}]
