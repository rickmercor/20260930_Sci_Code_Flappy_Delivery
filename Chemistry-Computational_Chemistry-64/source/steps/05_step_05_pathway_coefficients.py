"""
Evaluate one closed electronic path in an ordered dipole correlation.

A Heisenberg operator word gives signed propagation intervals after adjacent evolution operators are combined. Negative intervals represent the reversed parts of the contour. The electronic phase multiplies the vibrational generating kernel before electronic paths are summed.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pathway_coefficients(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, times: np.ndarray, path: np.ndarray, beta: float = np.inf) -> np.ndarray:
    """Evaluate one closed electronic path in an ordered dipole correlation.
    
    Parameters
    ----------
    energies : real ndarray, shape (N,)
        Electronic energies, energies[0]=0; hbar=1.
    omega : real ndarray, shape (F,)
        Positive mode frequencies, F=1,...,24.
    displacements : real ndarray, shape (N,F)
        State displacements, displacements[0,:]=0.
    mu0 : complex ndarray, shape (N,N)
        Hermitian Condon dipole matrix.
    mu1 : complex ndarray, shape (F,N,N)
        Hermitian coefficient matrices multiplying a_f+a_f.dagger.
    times : real ndarray, shape (J,)
        Times in LEFT-TO-RIGHT product order; they need not be chronological.
    path : integer ndarray, shape (J+1,)
        Electronic indices from left to right, path[0]=path[J]=0; 2 <= J <= 8.
    
    beta : positive float, default np.inf
        Inverse vibrational temperature, with k_B=hbar=1. The initial
        vibrational density is the normalized product of exp(-beta*omega_f*n_f).
        np.inf is the vacuum limit. The electronic state is prepared in state 0;
        there is no Boltzmann average over electronic states.
    
    Returns
    -------
    result : complex ndarray, shape (J+1,)
        Coefficients of lambda for the path contribution to
        Tr[(|0><0| tensor rho_beta)*mu(times[0];lambda)...mu(times[J-1];lambda)].
        mu(t;lambda)=exp(i*H*t)*[mu0+lambda*sum_f mu1[f]*(a_f+a_f.dagger)]*exp(-i*H*t),
        H_j=energies[j]+sum_f omega[f]*(a_f.dagger+z[j,f])*(a_f+z[j,f]).
        Use signed intervals [-times[0], times[0]-times[1], ...,
        times[J-2]-times[J-1], times[J-1]], including the endpoint intervals.
        Include electronic phases and the thermal Gaussian normalization.
        Earlier contour_gaussian and ht_coefficients functions are available.
        The infinite oscillator-space result is intended. Inputs are valid.
        Numeric outputs are tested with absolute and relative tolerances
        of 1e-9; integer outputs must be exact.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np


def _oracle_pathway_coefficients(energies: np.ndarray, omega: np.ndarray, displacements: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, times: np.ndarray, path: np.ndarray, beta: float = np.inf) -> np.ndarray:
    energies, omega, displacements, mu0, mu1, times, path = map(
        np.asarray, (energies, omega, displacements, mu0, mu1, times, path))
    dt = np.concatenate(([-times[0]], times[:-1]-times[1:], [times[-1]]))
    z = displacements[path]
    mu = mu0[path[:-1], path[1:]]
    h = mu1[:, path[:-1], path[1:]].T
    g, linear, quadratic = _oracle_contour_gaussian(omega, z, dt, h, beta)
    coefficients = _oracle_ht_coefficients(mu, linear, quadratic)
    phase = np.exp(-1j*np.dot(energies[path], dt))
    return phase*g*coefficients

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
               'times=np.array([0.37,0.])\n'
               'path=np.array([0,1,0])',
      'call': 'pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times, '
              'path)))',
      'gold_call': '_oracle_pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times, path)))',
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
               'times=np.array([0.37,1.72,1.19,0.])\n'
               'path=np.array([0,1,2,1,0])',
      'call': 'pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times, '
              'path)))',
      'gold_call': '_oracle_pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times, path)))',
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
               'times=np.array([0.,1.19,3.47,2.86,1.72,0.37])\n'
               'path=np.array([0,1,2,0,2,1,0])',
      'call': 'pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times, '
              'path)))',
      'gold_call': '_oracle_pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times, path)))',
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
               'times=np.array([0.7,0.2,0.9,0.])\n'
               'path=np.array([0,1,0,2,0])',
      'call': 'pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times, '
              'path)))',
      'gold_call': '_oracle_pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times, path)))',
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
               'path=np.array([0,1,0])\n'
               'beta = 0.85\n',
      'call': 'pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times, path, '
              'beta)))',
      'gold_call': '_oracle_pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times, path, beta)))',
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
               'times=np.array([0.,1.19,3.47,2.86,1.72,0.37])\n'
               'path=np.array([0,1,2,0,2,1,0])',
      'call': 'pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, times, path, '
              'beta)))',
      'gold_call': '_oracle_pathway_coefficients(*deepcopy((energies, omega, displacements, mu0, mu1, '
                   'times, path, beta)))',
      'tol': 1e-09}]
