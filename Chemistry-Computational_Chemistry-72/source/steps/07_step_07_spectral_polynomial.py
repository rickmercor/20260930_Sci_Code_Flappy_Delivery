"""
Apply the specified two-time discrete transform.

The selected contour is resolved with respect to its first and last waiting intervals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spectral_polynomial(nodes: "np.ndarray", weights: "np.ndarray", frequencies: "np.ndarray", waits: "np.ndarray", damping: float, model: dict) -> "np.ndarray":
    """Return the exact finite two-time transformed HT polynomial.
    
    Parameters
    ----------
    nodes : array-like
        Nonnegative transform nodes with shape (q,).
    weights : array-like
        Real node weights with shape (q,).
    frequencies : array-like
        Two signed transform frequencies.
    waits : array-like
        Fixed nonnegative internal delays.
    damping : float
        Nonnegative exponential damping coefficient.
    model : dict
        Vibronic model dictionary specified in the task.
    
    Returns
    -------
    spectrum : ndarray
        Packed complex transformed coefficients with shape (len(waits)+4,2).
    
    Notes
    -----
    For times=(nodes[i],*waits,nodes[j]), sum weights[i] weights[j] exp(i(frequencies[0] nodes[i]+frequencies[1] nodes[j])-damping(nodes[i]+nodes[j])) G(lambda;times). The finite sum defines the observable exactly; add no FFT normalization, time-step factor, 2*pi factor or continuum extrapolation. Complex outputs use final axis [real, imaginary]. Earlier public functions are available under public names.
    """
    return spectrum

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_spectral_polynomial(nodes: "np.ndarray", weights: "np.ndarray", frequencies: "np.ndarray", waits: "np.ndarray", damping: float, model: dict) -> "np.ndarray":
    import numpy as np
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    frequencies, waits = np.asarray(frequencies), np.asarray(waits)
    total = np.zeros(len(waits)+4, complex)
    for i, t1 in enumerate(nodes):
        for j, tlast in enumerate(nodes):
            times = np.r_[t1, waits, tlast]
            c = _oracle_response_polynomial(times,model)
            c = c[:,0]+1j*c[:,1]
            factor = weights[i]*weights[j]*np.exp(1j*(frequencies[0]*t1+frequencies[1]*tlast)-damping*(t1+tlast))
            total += factor*c
    return np.stack((total.real, total.imag), -1)

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
      'call': 'spectral_polynomial(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'gold_call': '_oracle_spectral_polynomial(*copy.deepcopy((nodes, weights, frequencies, waits, damping, '
                   'model)))',
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
               'nodes=[.3];weights=[.7]\n',
      'call': 'spectral_polynomial(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'gold_call': '_oracle_spectral_polynomial(*copy.deepcopy((nodes, weights, frequencies, waits, damping, '
                   'model)))',
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
               'weights=[.2,-.3,.4,-.2];frequencies=[-.4,1.1]\n',
      'call': 'spectral_polynomial(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'gold_call': '_oracle_spectral_polynomial(*copy.deepcopy((nodes, weights, frequencies, waits, damping, '
                   'model)))',
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
               'waits=[.5]\n',
      'call': 'spectral_polynomial(*copy.deepcopy((nodes, weights, frequencies, waits, damping, model)))',
      'gold_call': '_oracle_spectral_polynomial(*copy.deepcopy((nodes, weights, frequencies, waits, damping, '
                   'model)))',
      'tol': 2e-08}]
