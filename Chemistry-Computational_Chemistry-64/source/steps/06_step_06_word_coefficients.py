"""
Sum the exact electronic-path contributions to one dipole correlation word.

Electronic paths interfere at the amplitude level. A path with a vanishing Condon factor may still contribute through HT terms. Resolving powers of lambda before summation preserves these terms and their relative signs

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def word_coefficients(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, times: np.ndarray, beta: float = np.inf) -> np.ndarray:
    """Sum the exact electronic-path contributions to one dipole correlation word.
    
    Parameters
    ----------
    energies, omega, displacements, mu0, mu1 : ndarrays
        Shapes (N,), (F,), (N,F), (N,N), (F,N,N), respectively. Hbar=1, omega>0,
        energies[0]=0, displacements[0]=0, and dipole matrices are Hermitian.
    times : real ndarray, shape (J,)
        LEFT-TO-RIGHT times in the operator word; 2 <= J <= 8, N <= 4, F <= 24.
    
    beta : positive float, default np.inf
        Inverse vibrational temperature, with k_B=hbar=1. The initial
        vibrational density is the normalized product of exp(-beta*omega_f*n_f).
        np.inf is the vacuum limit. The electronic state is prepared in state 0;
        there is no Boltzmann average over electronic states.
    
    Returns
    -------
    result : complex ndarray, shape (J+1,)
        Coefficients of lambda in the thermal expectation of the ordered product
        of J Heisenberg dipoles mu(t;lambda). The electronic initial state is 0.
        The Hamiltonian and dipole convention are those in pathway_coefficients.
        Sum supported closed paths coherently before taking any real part.
        Return zeros if no supported path exists. No response prefactor or
        commutator sign is included here. Earlier electronic_paths and
        pathway_coefficients functions are available. Inputs are valid.
        Numeric outputs are tested with absolute and relative tolerances
        of 1e-9; integer outputs must be exact.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_word_coefficients(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, times: np.ndarray, beta: float = np.inf) -> np.ndarray:
    times = np.asarray(times)
    paths = _oracle_electronic_paths(mu0, mu1, len(times))
    result = np.zeros(len(times)+1, dtype=complex)
    for path in paths:
        result += _oracle_pathway_coefficients(
            energies, omega, displacements, mu0, mu1, times, path, beta)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'energies = np.array([0.0, 1.23, 2.17], dtype=float)\n'
               'omega = np.array([0.73, 1.11], dtype=float)\n'
               'displacements = np.array([[0.0, 0.0], [0.42, -0.31], [-0.36, 0.57]], dtype=float)\n'
               'mu0 = np.array([[0.12, 0.83, -0.21], [0.83, -0.17, 0.64], [-0.21, 0.64, 0.09]], '
               'dtype=float)\n'
               'mu1 = np.array([[[0.04, 0.19, 0.11], [0.19, -0.06, -0.14], [0.11, -0.14, 0.03]], '
               '[[-0.02, -0.13, 0.08], [-0.13, 0.05, 0.16], [0.08, 0.16, -0.04]]], dtype=float)\n'
               'waits = np.array([0.37, 0.82, 0.53, 1.14, 0.61], dtype=float)\n'
               'times=np.array([0.37,0.])',
      'call': 'word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times)))',
      'gold_call': '_oracle_word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times)))',
      'name': 'legacy_1',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'energies = np.array([0.0, 1.23, 2.17], dtype=float)\n'
               'omega = np.array([0.73, 1.11], dtype=float)\n'
               'displacements = np.array([[0.0, 0.0], [0.42, -0.31], [-0.36, 0.57]], dtype=float)\n'
               'mu0 = np.array([[0.12, 0.83, -0.21], [0.83, -0.17, 0.64], [-0.21, 0.64, 0.09]], '
               'dtype=float)\n'
               'mu1 = np.array([[[0.04, 0.19, 0.11], [0.19, -0.06, -0.14], [0.11, -0.14, 0.03]], '
               '[[-0.02, -0.13, 0.08], [-0.13, 0.05, 0.16], [0.08, 0.16, -0.04]]], dtype=float)\n'
               'waits = np.array([0.37, 0.82, 0.53, 1.14, 0.61], dtype=float)\n'
               'times=np.array([0.37,1.72,1.19,0.])',
      'call': 'word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times)))',
      'gold_call': '_oracle_word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times)))',
      'name': 'legacy_2',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'energies = np.array([0.0, 1.23, 2.17], dtype=float)\n'
               'omega = np.array([0.73, 1.11], dtype=float)\n'
               'displacements = np.array([[0.0, 0.0], [0.42, -0.31], [-0.36, 0.57]], dtype=float)\n'
               'mu0 = np.array([[0.12, 0.83, -0.21], [0.83, -0.17, 0.64], [-0.21, 0.64, 0.09]], '
               'dtype=float)\n'
               'mu1 = np.array([[[0.04, 0.19, 0.11], [0.19, -0.06, -0.14], [0.11, -0.14, 0.03]], '
               '[[-0.02, -0.13, 0.08], [-0.13, 0.05, 0.16], [0.08, 0.16, -0.04]]], dtype=float)\n'
               'waits = np.array([0.37, 0.82, 0.53, 1.14, 0.61], dtype=float)\n'
               'times=np.array([0.,1.19,3.47,2.86,1.72,0.37])',
      'call': 'word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times)))',
      'gold_call': '_oracle_word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times)))',
      'name': 'legacy_3',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'energies = np.array([0.0, 1.23, 2.17], dtype=float)\n'
               'omega = np.array([0.73, 1.11], dtype=float)\n'
               'displacements = np.array([[0.0, 0.0], [0.42, -0.31], [-0.36, 0.57]], dtype=float)\n'
               'mu0 = np.array([[0.12, 0.83, -0.21], [0.83, -0.17, 0.64], [-0.21, 0.64, 0.09]], '
               'dtype=float)\n'
               'mu1 = np.array([[[0.04, 0.19, 0.11], [0.19, -0.06, -0.14], [0.11, -0.14, 0.03]], '
               '[[-0.02, -0.13, 0.08], [-0.13, 0.05, 0.16], [0.08, 0.16, -0.04]]], dtype=float)\n'
               'waits = np.array([0.37, 0.82, 0.53, 1.14, 0.61], dtype=float)\n'
               'mu0[:]=0\n'
               'mu1[:]=0\n'
               'times=np.array([0.37,0.])',
      'call': 'word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times)))',
      'gold_call': '_oracle_word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times)))',
      'name': 'legacy_4',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'energies = np.array([0.0, 1.23, 2.17], dtype=float)\n'
               'omega = np.array([0.73, 1.11], dtype=float)\n'
               'displacements = np.array([[0.0, 0.0], [0.42, -0.31], [-0.36, 0.57]], dtype=float)\n'
               'mu0 = np.array([[0.12, 0.83, -0.21], [0.83, -0.17, 0.64], [-0.21, 0.64, 0.09]], '
               'dtype=float)\n'
               'mu1 = np.array([[[0.04, 0.19, 0.11], [0.19, -0.06, -0.14], [0.11, -0.14, 0.03]], '
               '[[-0.02, -0.13, 0.08], [-0.13, 0.05, 0.16], [0.08, 0.16, -0.04]]], dtype=float)\n'
               'waits = np.array([0.37, 0.82, 0.53, 1.14, 0.61], dtype=float)\n'
               'times=np.array([0.37,0.])\n'
               'beta = 0.85\n',
      'call': 'word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times, beta)))',
      'gold_call': '_oracle_word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times, beta)))',
      'name': 'finite_temperature',
      'tol': 1e-09},
     {'name': 'twenty_four_mode_benchmark',
      'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'f = np.arange(1, 25, dtype=float)\n'
               'energies = np.array([0.0, 8.23, 13.17])\n'
               'omega = 0.47 + 0.041*f + 0.0007*f**2\n'
               'displacements = np.array([np.zeros(24),\n'
               '    0.16*np.cos(0.41*f) + 0.04*np.sin(0.17*f),\n'
               '    0.14*np.sin(0.37*f) - 0.05*np.cos(0.23*f)])\n'
               'mu0 = np.array([[0.12, 0.83, -0.21], [0.83, -0.17, 0.64], [-0.21, 0.64, 0.09]])\n'
               'mu1 = np.zeros((24, 3, 3))\n'
               'mu1[:,0,0] = 0.018*np.cos(0.29*f)\n'
               'mu1[:,1,1] = -0.024*np.sin(0.31*f)\n'
               'mu1[:,2,2] = 0.015*np.cos(0.43*f)\n'
               'mu1[:,0,1] = mu1[:,1,0] = 0.055*np.cos(0.37*f) + 0.021*np.sin(0.13*f)\n'
               'mu1[:,0,2] = mu1[:,2,0] = -0.047*np.sin(0.23*f) + 0.019*np.cos(0.53*f)\n'
               'mu1[:,1,2] = mu1[:,2,1] = 0.051*np.cos(0.19*f) - 0.023*np.sin(0.41*f)\n'
               'waits = np.array([0.37, 0.82, 0.53, 1.14, 0.61])\n'
               'beta = 0.85\n'
               '\n'
               'times=np.array([0.37,1.72,1.19,0.])',
      'call': 'word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times, beta)))',
      'gold_call': '_oracle_word_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times, beta)))',
      'tol': 1e-09}]
