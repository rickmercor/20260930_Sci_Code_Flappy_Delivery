"""
Return the cubic-to-linear HT phase contrast.

The contrast measures the signed phase interference of the cubic HT contribution relative to the linear contribution.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_ht(nodes: "np.ndarray", weights: "np.ndarray", frequencies: "np.ndarray", waits: "np.ndarray", damping: float, model: dict) -> float:
    """Return the cubic-to-linear HT phase contrast.
    
    Parameters
    ----------
    nodes, weights, frequencies, waits : array-like
        Transform inputs defined by spectral_polynomial.
    damping : float
        Nonnegative damping coefficient.
    model : dict
        Vibronic model dictionary specified in the task.
    
    Returns
    -------
    answer : float
        Im(S_3 conjugate(S_1))/|S_1|^2 for the transformed polynomial coefficients.
    
    Raises
    ------
    ValueError
        If |S_1|<=1e-14.
    
    Notes
    -----
    This is a coefficient ratio, not the response evaluated at lambda=1. The underlying response is the single all-ket correlation specified in the task, with no i^M prefactor, commutator sum, orientational average or factorial normalization. Earlier public functions are available under public names.
    """
    return answer

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_ht(nodes: "np.ndarray", weights: "np.ndarray", frequencies: "np.ndarray", waits: "np.ndarray", damping: float, model: dict) -> float:
    import numpy as np
    coeff = _oracle_spectral_polynomial(nodes,weights,frequencies,waits,damping,model)
    coeff = coeff[:,0]+1j*coeff[:,1]
    if abs(coeff[1]) <= 1e-14:
        raise ValueError('linear HT coefficient is zero')
    answer = float((coeff[3]*coeff[1].conjugate()).imag / abs(coeff[1])**2)
    return answer

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():return [{'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'omega': [0.7, 1.15], 'coordinate': [1.0, -0.6], 'beta': 1.8, 'energies': [0.0, 1.8, "
               "3.05], 'shifts': [[0.0, 0.0], [0.42, -0.26], [-0.31, 0.37]], 'mu0': [[0.0, 1.0, 0.35], [1.0, "
               "0.0, 0.8], [0.35, 0.8, 0.0]], 'mu1': [[0.0, 0.18, -0.11], [0.18, 0.0, 0.14], [-0.11, 0.14, "
               '0.0]]}\n'
               'nodes=[0.17, 0.49, 0.96, 1.51]\n'
               'weights=[0.22, 0.38, 0.47, 0.31]\n'
               'frequencies=[1.4, -0.9]\n'
               'waits=[0.23, 0.41, 0.19]\n'
               'damping=0.08\n'
               '\n',
      'call': 'solve_ht(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'gold_call': '_oracle_solve_ht(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'omega': [0.7, 1.15], 'coordinate': [1.0, -0.6], 'beta': 1.8, 'energies': [0.0, 1.8, "
               "3.05], 'shifts': [[0.0, 0.0], [0.42, -0.26], [-0.31, 0.37]], 'mu0': [[0.0, 1.0, 0.35], [1.0, "
               "0.0, 0.8], [0.35, 0.8, 0.0]], 'mu1': [[0.0, 0.18, -0.11], [0.18, 0.0, 0.14], [-0.11, 0.14, "
               '0.0]]}\n'
               'nodes=[0.17, 0.49, 0.96, 1.51]\n'
               'weights=[0.22, 0.38, 0.47, 0.31]\n'
               'frequencies=[1.4, -0.9]\n'
               'waits=[0.23, 0.41, 0.19]\n'
               'damping=0.08\n'
               "model['coordinate']=[-1.,.6]\n",
      'call': 'solve_ht(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'gold_call': '_oracle_solve_ht(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'omega': [0.7, 1.15], 'coordinate': [1.0, -0.6], 'beta': 1.8, 'energies': [0.0, 1.8, "
               "3.05], 'shifts': [[0.0, 0.0], [0.42, -0.26], [-0.31, 0.37]], 'mu0': [[0.0, 1.0, 0.35], [1.0, "
               "0.0, 0.8], [0.35, 0.8, 0.0]], 'mu1': [[0.0, 0.18, -0.11], [0.18, 0.0, 0.14], [-0.11, 0.14, "
               '0.0]]}\n'
               'nodes=[0.17, 0.49, 0.96, 1.51]\n'
               'weights=[0.22, 0.38, 0.47, 0.31]\n'
               'frequencies=[1.4, -0.9]\n'
               'waits=[0.23, 0.41, 0.19]\n'
               'damping=0.08\n'
               "model['mu0']=2*np.array(model['mu0'])\n",
      'call': 'solve_ht(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'gold_call': '_oracle_solve_ht(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'tol': 2e-08},
     {'setup': 'import numpy as np\n'
               'import copy\n'
               "model={'omega': [0.7, 1.15], 'coordinate': [1.0, -0.6], 'beta': 1.8, 'energies': [0.0, 1.8, "
               "3.05], 'shifts': [[0.0, 0.0], [0.42, -0.26], [-0.31, 0.37]], 'mu0': [[0.0, 1.0, 0.35], [1.0, "
               "0.0, 0.8], [0.35, 0.8, 0.0]], 'mu1': [[0.0, 0.18, -0.11], [0.18, 0.0, 0.14], [-0.11, 0.14, "
               '0.0]]}\n'
               'nodes=[0.17, 0.49, 0.96, 1.51]\n'
               'weights=[0.22, 0.38, 0.47, 0.31]\n'
               'frequencies=[1.4, -0.9]\n'
               'waits=[0.23, 0.41, 0.19]\n'
               'damping=0.08\n'
               "model['mu1']=np.zeros((3,3))\n"
               'def _raises_value_error(fn, *args, **kwargs):\n'
               '    try:\n'
               '        fn(*args, **kwargs)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('required ValueError was not raised')\n"
               '\n',
      'call': '_raises_value_error(solve_ht, *copy.deepcopy((nodes, weights, frequencies, waits, damping, '
              'model)))',
      'gold_call': '_raises_value_error(_oracle_solve_ht, *copy.deepcopy((nodes, weights, frequencies, waits, '
                   'damping, model)))',
      'tol': 2e-08}]
