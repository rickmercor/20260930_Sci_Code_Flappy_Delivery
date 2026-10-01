"""
Compute the contribution omitted by a second-order HT truncation.

Truncating the dipole expansion at its linear nuclear term does not truncate the response at second order in HT coupling. The requested scalar isolates the cubic through sixth-order HT contribution to a fifth-order optical response. The HT input consists of dipole derivatives with respect to the dimensionless coordinate of the specified source method.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, dmu_dX: np.ndarray, waits: np.ndarray, beta: float = np.inf) -> float:
    """Compute the contribution omitted by a second-order HT truncation.
    
    Parameters
    ----------
    energies, omega, displacements, mu0 : ndarrays
        Shapes (N,), (F,), (N,F), (N,N); same valid-domain Hamiltonian and
        Condon dipole as response_coefficients, with hbar=1.
    dmu_dX : real ndarray, shape (F,N,N)
        Hermitian dipole derivatives D_f = d(mu)/d(X_f) with respect to the
        dimensionless coordinate X_f = (a_f + a_f.dagger)/2. The HT dipole is
        therefore mu(lambda) = mu0 + lambda*sum_f D_f*X_f
        = mu0 + lambda*sum_f (D_f/2)*(a_f + a_f.dagger), so the mu1 argument
        of response_coefficients is dmu_dX/2. This is not the
        (a_f + a_f.dagger)/sqrt(2) convention.
    waits : real ndarray, shape (M,)
        Nonnegative consecutive waiting times, 1 <= M <= 5. M=5 in the benchmark.
    
    beta : positive float, default np.inf
        Inverse vibrational temperature, with k_B=hbar=1. The initial
        vibrational density is the normalized product of exp(-beta*omega_f*n_f).
        np.inf is the vacuum limit. The electronic state is prepared in state 0;
        there is no Boltzmann average over electronic states.
    
    Returns
    -------
    result : float
        R(1)-[c_0+c_1+c_2]=sum(c_r for r=3,...,M+1), where c_r are the coefficients
        returned by response_coefficients with mu1 = dmu_dX/2 and the SAME
        Hamiltonian, Condon dipole and beta.
        Second-order here means polynomial order in lambda, not optical response
        order. There is no absolute value, percentage, division, or rounding.
        Return 0.0 when M=1. Call the earlier response_coefficients step to
        integrate the full pipeline. Inputs are valid.
        Numeric outputs are tested with absolute and relative tolerances
        of 1e-9; integer outputs must be exact.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_solve(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, dmu_dX: np.ndarray, waits: np.ndarray, beta: float = np.inf) -> float:
    mu1 = 0.5*np.asarray(dmu_dX)
    coefficients = _oracle_response_coefficients(
        energies, omega, displacements, mu0, mu1, waits, beta)
    return float(np.sum(coefficients[3:]))

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
               'dmu_dX = 2*mu1',
      'call': 'solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
      'gold_call': '_oracle_solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
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
               'mu1[:]=0\n'
               'dmu_dX = 2*mu1',
      'call': 'solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
      'gold_call': '_oracle_solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
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
               'waits=np.array([0.37,0.82,0.53])\n'
               'dmu_dX = 2*mu1',
      'call': 'solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
      'gold_call': '_oracle_solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
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
               'waits=np.array([0.37])\n'
               'dmu_dX = 2*mu1',
      'call': 'solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
      'gold_call': '_oracle_solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
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
               'mu1*=0.7\n'
               'waits=np.array([0.61,0.29,0.73])\n'
               'dmu_dX = 2*mu1',
      'call': 'solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
      'gold_call': '_oracle_solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits)))',
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
               'beta = 0.85\n'
               '\n'
               'dmu_dX = 2*mu1',
      'call': 'solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits, beta)))',
      'gold_call': '_oracle_solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits, '
                   'beta)))',
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
               'dmu_dX = np.zeros((24, 3, 3))\n'
               'dmu_dX[:,0,0] = 0.036*np.cos(0.29*f)\n'
               'dmu_dX[:,1,1] = -0.048*np.sin(0.31*f)\n'
               'dmu_dX[:,2,2] = 0.030*np.cos(0.43*f)\n'
               'dmu_dX[:,0,1] = dmu_dX[:,1,0] = 0.110*np.cos(0.37*f) + 0.042*np.sin(0.13*f)\n'
               'dmu_dX[:,0,2] = dmu_dX[:,2,0] = -0.094*np.sin(0.23*f) + 0.038*np.cos(0.53*f)\n'
               'dmu_dX[:,1,2] = dmu_dX[:,2,1] = 0.102*np.cos(0.19*f) - 0.046*np.sin(0.41*f)\n'
               'waits = np.array([0.37, 0.82, 0.53, 1.14, 0.61])\n'
               'beta = 0.85\n',
      'call': 'solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits, beta)))',
      'gold_call': '_oracle_solve(*deepcopy((energies, omega, displacements, mu0, dmu_dX, waits, '
                   'beta)))',
      'tol': 1e-09}]
