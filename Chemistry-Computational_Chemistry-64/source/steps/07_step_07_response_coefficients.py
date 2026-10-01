"""
Evaluate the causal M-th order response as a polynomial in HT scale lambda.

The full causal response includes every left and right action of the field. Hermiticity makes the final coefficients real after the factor $$i^M$$ is included. Response order M and HT polynomial order r are different indices.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def response_coefficients(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, waits: np.ndarray, beta: float = np.inf) -> np.ndarray:
    """Evaluate the causal M-th order response as a polynomial in HT scale lambda.
    
    Parameters
    ----------
    energies, omega, displacements, mu0, mu1 : ndarrays
        Shapes (N,), (F,), (N,F), (N,N), (F,N,N). Hbar=1, omega>0, energies[0]=0,
        displacements[0]=0. Dipole matrices are Hermitian; N<=4, F<=24.
    waits : real ndarray, shape (M,)
        Nonnegative consecutive waiting times; 1 <= M <= 5.
        Define T_0=0 and T_k=sum(waits[:k]), k=1,...,M.
    
    beta : positive float, default np.inf
        Inverse vibrational temperature, with k_B=hbar=1. The initial
        vibrational density is the normalized product of exp(-beta*omega_f*n_f).
        np.inf is the vacuum limit. The electronic state is prepared in state 0;
        there is no Boltzmann average over electronic states.
    
    Returns
    -------
    result : real ndarray, shape (M+2,)
        Coefficients c_r of lambda**r, r=0,...,M+1, in
        R(lambda)=i**M*Tr[mu(T_M;lambda) [mu(T_(M-1);lambda),
           [... [mu(T_0;lambda), rho0] ...]]],
        rho0=|0><0| tensor rho_beta, with rho_beta defined by beta above. The Hamiltonian and dipole are those in
        pathway_coefficients. There is no factorial, rotating-wave approximation,
        orientational average, damping, or additional sign. Every commutator is
        [A,B]=A*B-B*A. Earlier commutator_words and word_coefficients functions
        are available. Sum complex amplitudes with their signs, multiply by i**M,
        then take the real part; exact coefficients are real. Inputs are valid.
        Numeric outputs are tested with absolute and relative tolerances
        of 1e-9; integer outputs must be exact.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_response_coefficients(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, waits: np.ndarray, beta: float = np.inf) -> np.ndarray:
    waits = np.asarray(waits)
    order = len(waits)
    times = np.concatenate(([0.], np.cumsum(waits)))
    result = np.zeros(order+2, dtype=complex)
    for row in _oracle_commutator_words(order):
        result += row[0]*_oracle_word_coefficients(
            energies, omega, displacements, mu0, mu1, times[row[1:]], beta)
    return np.real((1j**order)*result)

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
               'waits=np.array([0.37])',
      'call': 'response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, waits)))',
      'gold_call': '_oracle_response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'waits)))',
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
               'waits=np.array([0.37,0.82])',
      'call': 'response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, waits)))',
      'gold_call': '_oracle_response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'waits)))',
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
               'waits=np.array([0.37,0.82,0.53])',
      'call': 'response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, waits)))',
      'gold_call': '_oracle_response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'waits)))',
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
               'waits = np.array([0.37, 0.82, 0.53, 1.14, 0.61], dtype=float)',
      'call': 'response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, waits)))',
      'gold_call': '_oracle_response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'waits)))',
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
               'waits=np.zeros(3)',
      'call': 'response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, waits)))',
      'gold_call': '_oracle_response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'waits)))',
      'name': 'legacy_5',
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
               'waits=np.array([0.37])\n'
               'beta = 0.85\n',
      'call': 'response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, waits, '
              'beta)))',
      'gold_call': '_oracle_response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'waits, beta)))',
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
               'beta = 0.85\n',
      'call': 'response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, waits, '
              'beta)))',
      'gold_call': '_oracle_response_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'waits, beta)))',
      'tol': 1e-09}]
