"""
Sum the closed electronic pathways.

The same vibronic Hamiltonian contributes through different electronic-state sequences between optical interactions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def response_polynomial(times: "np.ndarray", model: dict) -> "np.ndarray":
    """Return the all-ket HT response polynomial for the supplied times.
    
    Parameters
    ----------
    times : array-like
        Real chronological intervals with shape (M,), M>=1; zero and negative intervals are allowed.
    model : dict
        Contains omega, coordinate, beta, energies, shifts, mu0 and mu1 as specified in the task.
    
    Returns
    -------
    coeff : ndarray
        Packed complex coefficients [lambda^k]G(lambda) with shape (M+2,2).
    
    Notes
    -----
    Use hbar=1, H_e=E_e+sum_k omega_k(a_k^dagger+d_ek)(a_k+d_ek), Q=sum_k coordinate_k(a_k+a_k^dagger), the normalized thermal state of H_0, and mu(lambda)=mu0+lambda mu1 Q. G is the single all-ket correlation from the task, without i^M, commutator sums, orientational averaging or factorial normalization. Include every closed electronic path from state 0 back to state 0. Complex outputs use final axis [real, imaginary]. Earlier public functions are available under public names.
    """
    return coeff

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_response_polynomial(times: "np.ndarray", model: dict) -> "np.ndarray":
    import numpy as np
    from itertools import product
    mu0, mu1 = np.asarray(model['mu0']), np.asarray(model['mu1'])
    order, n = len(times), len(mu0)
    coeff = np.zeros((order+2, 2))
    for path in product(range(n), repeat=order):
        states = (0,) + path + (0,)
        if any(mu0[b,a] == 0 and mu1[b,a] == 0 for a,b in zip(states[:-1], states[1:])):
            continue
        u,v,phase = _oracle_contour_sources(path, times, model['energies'], model['shifts'], model['omega'], model['coordinate'])
        c,l,B = _oracle_gaussian_generator(u,v,phase,model['omega'],model['beta'])
        moments = _oracle_squarefree_moments(c,l,B)
        coeff += _oracle_pathway_polynomial(path,mu0,mu1,moments)
    return coeff

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
      'call': 'response_polynomial(*copy.deepcopy(([0.71], model)))',
      'gold_call': '_oracle_response_polynomial(*copy.deepcopy(([0.71], model)))',
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
               '\n',
      'call': 'response_polynomial(*copy.deepcopy(([0.3, -0.6, 0.8], model)))',
      'gold_call': '_oracle_response_polynomial(*copy.deepcopy(([0.3, -0.6, 0.8], model)))',
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
               '\n',
      'call': 'response_polynomial(*copy.deepcopy(([0.49, *waits, 0.96], model)))',
      'gold_call': '_oracle_response_polynomial(*copy.deepcopy(([0.49, *waits, 0.96], model)))',
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
               '\n',
      'call': 'response_polynomial(*copy.deepcopy(([0.0, 0.0, 0.0, 0.0, 0.0], model)))',
      'gold_call': '_oracle_response_polynomial(*copy.deepcopy(([0.0, 0.0, 0.0, 0.0, 0.0], model)))',
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
               "model['mu1']=np.zeros((3,3))\n",
      'call': 'response_polynomial(*copy.deepcopy(([0.4, 0.1, 0.8], model)))',
      'gold_call': '_oracle_response_polynomial(*copy.deepcopy(([0.4, 0.1, 0.8], model)))',
      'tol': 2e-08}]
