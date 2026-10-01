"""
Compute the exact thermal generating kernel of ordered HT insertions.

Ordered displaced harmonic propagators and linear-coordinate insertions form a Gaussian operator algebra. Thermal averaging changes the symmetric fluctuation term; the commutator phase is fixed by operator order. The source Hessian is symmetric under transpose, not Hermitian.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def contour_gaussian(omega: np.ndarray, z: np.ndarray, dt: np.ndarray, h: np.ndarray, beta: float = np.inf) -> tuple:
    """Compute the exact thermal generating kernel of ordered HT insertions.
    
    Parameters
    ----------
    omega : real ndarray, shape (F,)
        Positive mode frequencies; hbar=1; 1 <= F <= 24.
    z : real ndarray, shape (J+1,F)
        Displacements for each propagation interval, including both endpoints.
    dt : real ndarray, shape (J+1,)
        Signed interval lengths, sum(dt)=0 within 1e-12; 1 <= J <= 8.
    h : complex ndarray, shape (J,F)
        At insertion i, Q_i=sum_f h[i,f]*(a_f+a_f.dagger). These are coefficients
        of a+a.dagger, not derivatives with respect to a normalized coordinate.
    
    beta : positive float, default np.inf
        Inverse vibrational temperature, with k_B=hbar=1. The initial
        vibrational density is the normalized product of exp(-beta*omega_f*n_f).
        np.inf is the vacuum limit. The electronic state is prepared in state 0;
        there is no Boltzmann average over electronic states.
    
    Returns
    -------
    result : tuple (g, L, K)
        g is complex, L has shape (J,), K has shape (J,J) and is complex symmetric.
        Define U_l=exp[-i*dt[l]*sum_f omega[f]*(a_f.dagger+z[l,f])*(a_f+z[l,f])].
        For independent formal sources s_i, define
        Z(s)=Tr[rho_beta U_0 exp(s_0 Q_0) U_1 ... exp(s_(J-1) Q_(J-1)) U_J],
        rho_beta=prod_f[(1-exp(-beta*omega_f))*exp(-beta*omega_f*n_f)].
        At beta=np.inf, interpret this density as the vacuum projector.
        Return the coefficients in the exact identity
        Z(s)=g*exp(sum_i L_i*s_i + 0.5*sum_ij K_ij*s_i*s_j).
        Include diagonal entries of K. Do not conjugate formal source variables
        or h coefficients. The full infinite oscillator space is intended;
        a finite Fock cutoff or finite differences are not part of this contract.
        All inputs are valid; zero durations and negative durations are allowed.
        Numeric outputs are tested with absolute and relative tolerances
        of 1e-9; integer outputs must be exact.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_contour_gaussian(omega: np.ndarray, z: np.ndarray, dt: np.ndarray, h: np.ndarray, beta: float = np.inf) -> tuple:
    omega, z, dt, h = map(np.asarray, (omega, z, dt, h))
    count, modes = h.shape
    log_matrix = np.zeros((count+1, count+1), dtype=complex)
    rotation = np.ones(modes, dtype=complex)
    total_a = np.zeros((modes, count+1), dtype=complex)
    total_b = np.zeros_like(total_a)
    phase = 0j
    for ell in range(count+1):
        angle = omega*dt[ell]
        r = np.exp(-1j*angle)
        d = z[ell]*(r-1)
        a = np.zeros_like(total_a)
        b = np.zeros_like(total_b)
        a[:,0] = rotation*d
        b[:,0] = -np.conj(rotation*d)
        phase -= 1j*np.sum(z[ell]**2*np.sin(angle))
        log_matrix += 0.5*(total_b.T@a-total_a.T@b)
        total_a += a
        total_b += b
        rotation *= r
        if ell < count:
            a = np.zeros_like(total_a)
            b = np.zeros_like(total_b)
            a[:,ell+1] = rotation*h[ell]
            b[:,ell+1] = np.conj(rotation)*h[ell]
            log_matrix += 0.5*(total_b.T@a-total_a.T@b)
            total_a += a
            total_b += b
    thermal_width = 1/np.tanh(0.5*beta*omega)
    log_matrix += 0.5*total_a.T@(thermal_width[:,None]*total_b)
    g = np.exp(phase+log_matrix[0,0])
    linear = log_matrix[0,1:]+log_matrix[1:,0]
    quadratic = log_matrix[1:,1:]+log_matrix[1:,1:].T
    return complex(g), linear, quadratic

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'omega=np.array([0.8])\n'
               'z=np.array([[0.],[0.6],[0.]])\n'
               'dt=np.array([0.,0.7,-0.7])\n'
               'h=np.ones((2,1))',
      'call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
              'np.ravel(r[2]))))(contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'gold_call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
                   'np.ravel(r[2]))))(_oracle_contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'name': 'legacy_1',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'omega=np.array([0.7,1.1])\n'
               'z=np.zeros((4,2))\n'
               'dt=np.zeros(4)\n'
               'h=np.array([[0.2,0.5],[-0.3,0.4],[0.7,-0.2]])',
      'call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
              'np.ravel(r[2]))))(contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'gold_call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
                   'np.ravel(r[2]))))(_oracle_contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'name': 'legacy_2',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'omega=np.array([0.8])\n'
               'z=np.zeros((3,1))\n'
               'dt=np.array([-0.4,1.1,-0.7])\n'
               'h=np.array([[1+0.3j],[0.2-0.7j]])',
      'call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
              'np.ravel(r[2]))))(contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'gold_call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
                   'np.ravel(r[2]))))(_oracle_contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'name': 'legacy_3',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'omega=np.array([0.73,1.11])\n'
               'z=np.array([[0.,0.],[0.42,-0.31],[-0.36,0.57],[0.42,-0.31],[0.,0.]])\n'
               'dt=np.array([-1.7,0.8,1.4,-0.5,0.])\n'
               'h=np.array([[0.19,-0.13],[-0.14,0.16],[0.11,0.08],[0.04,-0.02]])',
      'call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
              'np.ravel(r[2]))))(contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'gold_call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
                   'np.ravel(r[2]))))(_oracle_contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'name': 'legacy_4',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'omega=np.array([1.2])\n'
               'z=np.array([[0.],[-0.4],[0.]])\n'
               'dt=np.array([-0.3,0.9,-0.6])\n'
               'h=np.zeros((2,1))',
      'call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
              'np.ravel(r[2]))))(contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'gold_call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
                   'np.ravel(r[2]))))(_oracle_contour_gaussian(*deepcopy((omega, z, dt, h))))',
      'name': 'legacy_5',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'omega=np.array([0.8])\n'
               'z=np.array([[0.],[0.6],[0.]])\n'
               'dt=np.array([0.,0.7,-0.7])\n'
               'h=np.ones((2,1))\n'
               'beta = 0.85\n',
      'call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
              'np.ravel(r[2]))))(contour_gaussian(*deepcopy((omega, z, dt, h, beta))))',
      'gold_call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
                   'np.ravel(r[2]))))(_oracle_contour_gaussian(*deepcopy((omega, z, dt, h, beta))))',
      'name': 'finite_temperature',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
               'import numpy as np\n'
               'omega=np.array([0.8])\n'
               'z=np.zeros((3,1))\n'
               'dt=np.array([-0.4,1.1,-0.7])\n'
               'h=np.array([[1+0.3j],[0.2-0.7j]])',
      'call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
              'np.ravel(r[2]))))(contour_gaussian(*deepcopy((omega, z, dt, h, 0.6))))',
      'gold_call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
                   'np.ravel(r[2]))))(_oracle_contour_gaussian(*deepcopy((omega, z, dt, h, 0.6))))',
      'name': 'complex_sources_thermal',
      'tol': 1e-09},
     {'setup': 'from copy import deepcopy\n'
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
               'z=displacements[[0,1,2,0]]\n'
               'dt=np.array([-0.3,0.9,-0.2,-0.4])\n'
               'h=mu1[:,[0,1,2],[1,2,0]].T\n',
      'call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
              'np.ravel(r[2]))))(contour_gaussian(*deepcopy((omega, z, dt, h, beta))))',
      'gold_call': '(lambda r: np.concatenate(([r[0]], np.shape(r[1]), np.ravel(r[1]), np.shape(r[2]), '
                   'np.ravel(r[2]))))(_oracle_contour_gaussian(*deepcopy((omega, z, dt, h, beta))))',
      'name': 'twenty_four_modes',
      'tol': 1e-09}]
