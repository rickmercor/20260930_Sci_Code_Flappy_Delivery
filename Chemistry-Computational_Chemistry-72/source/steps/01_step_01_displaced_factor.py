"""
Factor a displaced oscillator propagator.

Electronic excitation shifts the vibrational minimum. The displacement phase remains physically relevant even when the net phase-space displacement closes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def displaced_factor(omega: "np.ndarray", shift: "np.ndarray", duration: float, energy: float) -> tuple:
    """Return the continuous displaced-oscillator factorization.
    
    Parameters
    ----------
    omega : array-like
        Positive mode frequencies with shape (m,).
    shift : array-like
        Real displacement vector with shape (m,).
    duration : float
        Signed propagation time.
    energy : float
        Electronic energy offset.
    
    Returns
    -------
    phase : float
        Unwrapped scalar phase with phase(0)=0.
    angle : ndarray
        Rotation angles with shape (m,).
    alpha : ndarray
        Displacements with shape (m,2), packed as [real, imaginary].
    
    Notes
    -----
    The Hamiltonian is H=energy+sum_k omega_k(a_k^dagger+shift_k)(a_k+shift_k), with no zero-point term. The result satisfies U(duration)=exp(i phase) prod_k D(alpha_k) exp(-i angle_k a_k^dagger a_k), with D(alpha)=exp(alpha a^dagger-alpha* a). Do not wrap the phase to a principal interval. NumPy and SciPy are available and imports belong inside the implementation.
    """
    return phase, angle, alpha

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_displaced_factor(omega: "np.ndarray", shift: "np.ndarray", duration: float, energy: float) -> tuple:
    import numpy as np
    omega, shift = np.asarray(omega, float), np.asarray(shift, float)
    angle = omega * duration
    alpha = shift * np.expm1(-1j * angle)
    phase = float(-energy * duration - np.sum(shift**2 * np.sin(angle)))
    return phase, angle, np.stack((alpha.real, alpha.imag), -1)

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
      'call': "displaced_factor(*copy.deepcopy((model['omega'], model['shifts'][1], 0.73, 1.8)))",
      'gold_call': "_oracle_displaced_factor(*copy.deepcopy((model['omega'], model['shifts'][1], 0.73, 1.8)))",
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
      'call': "displaced_factor(*copy.deepcopy((model['omega'], model['shifts'][1], 0.0, 1.8)))",
      'gold_call': "_oracle_displaced_factor(*copy.deepcopy((model['omega'], model['shifts'][1], 0.0, 1.8)))",
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
      'call': "displaced_factor(*copy.deepcopy((model['omega'], model['shifts'][2], -2.4, 3.05)))",
      'gold_call': "_oracle_displaced_factor(*copy.deepcopy((model['omega'], model['shifts'][2], -2.4, 3.05)))",
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
      'call': 'displaced_factor(*copy.deepcopy(([1.0], [0.9], 31.0, 2.0)))',
      'gold_call': '_oracle_displaced_factor(*copy.deepcopy(([1.0], [0.9], 31.0, 2.0)))',
      'tol': 2e-08}]
