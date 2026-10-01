"""
Compute the thermal generating quadratic form.

A thermal harmonic state is Gaussian, while the ordered Weyl product retains the quantum commutator phase.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gaussian_generator(u: "np.ndarray", v: "np.ndarray", phase: float, omega: "np.ndarray", beta: float) -> tuple:
    """Return the thermal Gaussian generating form for an ordered Weyl product.
    
    Parameters
    ----------
    u, v : array-like
        Packed complex affine coefficients with shape (f,m,r+1,2).
    phase : float
        Continuous scalar phase accumulated by the contour.
    omega : array-like
        Positive mode frequencies with shape (m,).
    beta : float
        Positive inverse temperature.
    
    Returns
    -------
    c : ndarray
        Packed complex constant logarithm with shape (2,).
    linear : ndarray
        Packed complex linear coefficients with shape (r,2).
    quadratic : ndarray
        Packed complex symmetric quadratic coefficients with shape (r,r,2).
    
    Notes
    -----
    The returned coefficients satisfy Tr[rho_beta exp(i phase) prod_f exp(sum_k(u_fk a_k^dagger+v_fk a_k))]=exp(c+l^T s+0.5 s^T B s) for the normalized product thermal state rho_beta proportional to exp(-beta sum_k omega_k a_k^dagger a_k). B is complex symmetric, not Hermitian. Use the logarithm defined continuously by the ordered exponential algebra rather than a principal logarithm of the trace. Transposes in the polynomial are ordinary transposes. Complex outputs use final axis [real, imaginary].
    """
    return c, linear, quadratic

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_gaussian_generator(u: "np.ndarray", v: "np.ndarray", phase: float, omega: "np.ndarray", beta: float) -> tuple:
    import numpy as np
    u, v = np.asarray(u, float), np.asarray(v, float)
    u, v = u[..., 0]+1j*u[..., 1], v[..., 0]+1j*v[..., 1]
    nu = .5 / np.tanh(.5*beta*np.asarray(omega, float))
    dimension = u.shape[-1]
    A = np.zeros((dimension, dimension), complex)
    for i in range(len(u)):
        for j in range(i+1, len(u)):
            A += .5*(v[i].T @ u[j] - u[i].T @ v[j])
    U, V = u.sum(axis=0), v.sum(axis=0)
    A += U.T @ (nu[:, None] * V)
    c = A[0, 0] + 1j*phase
    linear = A[0, 1:] + A[1:, 0]
    quadratic = A[1:, 1:] + A[1:, 1:].T
    return np.array([c.real, c.imag]), np.stack((linear.real, linear.imag), -1), np.stack((quadratic.real, quadratic.imag), -1)

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
               "u_c,v_c,phase_c=contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               "u_g,v_g,phase_g=_oracle_contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               '\n',
      'call': "gaussian_generator(*copy.deepcopy((u_c, v_c, phase_c, model['omega'], 1.8)))",
      'gold_call': "_oracle_gaussian_generator(*copy.deepcopy((u_g, v_g, phase_g, model['omega'], 1.8)))",
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
               "u_c,v_c,phase_c=contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               "u_g,v_g,phase_g=_oracle_contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               '\n',
      'call': "gaussian_generator(*copy.deepcopy((u_c, v_c, phase_c, model['omega'], 0.13)))",
      'gold_call': "_oracle_gaussian_generator(*copy.deepcopy((u_g, v_g, phase_g, model['omega'], 0.13)))",
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
               "u_c,v_c,phase_c=contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               "u_g,v_g,phase_g=_oracle_contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               '\n',
      'call': "gaussian_generator(*copy.deepcopy((u_c, v_c, phase_c, model['omega'], 80.0)))",
      'gold_call': "_oracle_gaussian_generator(*copy.deepcopy((u_g, v_g, phase_g, model['omega'], 80.0)))",
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
               '\n'
               'def pack(x):\n'
               '    x=np.asarray(x,complex)\n'
               '    return np.stack((x.real,x.imag),-1)\n'
               'rng=np.random.default_rng(73)\n'
               'u=pack(.1*(rng.normal(size=(3,2,5))+1j*rng.normal(size=(3,2,5))))\n'
               'v=pack(.1*(rng.normal(size=(3,2,5))+1j*rng.normal(size=(3,2,5))))\n',
      'call': 'gaussian_generator(*copy.deepcopy((u, v, -0.3, [0.7, 1.2], 1.4)))',
      'gold_call': '_oracle_gaussian_generator(*copy.deepcopy((u, v, -0.3, [0.7, 1.2], 1.4)))',
      'tol': 2e-08}]
