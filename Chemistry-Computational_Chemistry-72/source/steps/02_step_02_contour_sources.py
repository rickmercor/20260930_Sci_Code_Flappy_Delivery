"""
Resolve the ordered contour into affine Weyl factors.

Coordinate insertions lie between displaced evolutions, so the resulting source coefficients depend on their chronological positions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contour_sources(path: "np.ndarray", times: "np.ndarray", energies: "np.ndarray", shifts: "np.ndarray", omega: "np.ndarray", coordinate: "np.ndarray") -> tuple:
    """Return affine Weyl factors for the ordered contour.
    
    Parameters
    ----------
    path : array-like
        Electronic states occupied during the M chronological intervals.
    times : array-like
        Signed interval durations with shape (M,), M>=1.
    energies : array-like
        Electronic energies with energies[0]=0.
    shifts : array-like
        State-dependent real displacements with shifts[0]=0.
    omega : array-like
        Positive vibrational frequencies.
    coordinate : array-like
        Real coefficients defining Q=sum_k coordinate_k(a_k+a_k^dagger).
    
    Returns
    -------
    u : ndarray
        Packed complex affine creation coefficients with shape (2M+2,m,M+2,2).
    v : ndarray
        Packed complex affine annihilation coefficients with the same shape.
    phase : float
        Sum of the continuous displaced-propagator phases.
    
    Notes
    -----
    For formal sources s_0,...,s_M, the ordered contour is F=U_0(-sum t) exp(s_M Q) U_path[M-1](t_M)...U_path[0](t_1) exp(s_0 Q). The coefficient-axis index 0 is constant and index j+1 multiplies s_j. Keep the written factor order, use one displacement factor and one source factor per propagator/source pair, move rotations to the far right where the total angle closes, and do not combine adjacent Weyl factors. Complex outputs use final axis [real, imaginary]. Earlier public functions are available under public names.
    """
    return u, v, phase

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_contour_sources(path: "np.ndarray", times: "np.ndarray", energies: "np.ndarray", shifts: "np.ndarray", omega: "np.ndarray", coordinate: "np.ndarray") -> tuple:
    import numpy as np
    path, times = np.asarray(path, int), np.asarray(times, float)
    energies, shifts = np.asarray(energies, float), np.asarray(shifts, float)
    omega, coordinate = np.asarray(omega, float), np.asarray(coordinate, float)
    order, modes = len(times), len(omega)
    r, count = order + 1, 2 * order + 2
    u = np.zeros((count, modes, r+1), complex)
    v = np.zeros_like(u)
    angle = np.zeros(modes)
    phase, row = 0., 0
    for j in range(order, -1, -1):
        state = 0 if j == order else path[j]
        duration = -sum(times) if j == order else times[j]
        ph, theta, packed = _oracle_displaced_factor(omega, shifts[state], duration, energies[state])
        alpha = packed[:, 0] + 1j * packed[:, 1]
        u[row, :, 0] = alpha * np.exp(-1j * angle)
        v[row, :, 0] = -alpha.conj() * np.exp(1j * angle)
        phase += ph
        angle += theta
        row += 1
        u[row, :, j+1] = coordinate * np.exp(-1j * angle)
        v[row, :, j+1] = coordinate * np.exp(1j * angle)
        row += 1
    return np.stack((u.real, u.imag), -1), np.stack((v.real, v.imag), -1), float(phase)

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
      'call': 'contour_sources(*copy.deepcopy(([1, 2, 1, 2, 1], [0.17, 0.23, 0.41, 0.19, 0.96], '
              "model['energies'], model['shifts'], model['omega'], model['coordinate'])))",
      'gold_call': '_oracle_contour_sources(*copy.deepcopy(([1, 2, 1, 2, 1], [0.17, 0.23, 0.41, 0.19, 0.96], '
                   "model['energies'], model['shifts'], model['omega'], model['coordinate'])))",
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
      'call': "contour_sources(*copy.deepcopy(([2], [-0.4], model['energies'], model['shifts'], model['omega'], "
              "model['coordinate'])))",
      'gold_call': "_oracle_contour_sources(*copy.deepcopy(([2], [-0.4], model['energies'], model['shifts'], "
                   "model['omega'], model['coordinate'])))",
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
      'call': "contour_sources(*copy.deepcopy(([1, 0, 2], [0.0, 0.0, 0.0], model['energies'], model['shifts'], "
              "model['omega'], model['coordinate'])))",
      'gold_call': "_oracle_contour_sources(*copy.deepcopy(([1, 0, 2], [0.0, 0.0, 0.0], model['energies'], "
                   "model['shifts'], model['omega'], model['coordinate'])))",
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
               "model['coordinate']=[0.,0.]\n",
      'call': "contour_sources(*copy.deepcopy(([2, 1, 2], [0.3, -0.7, 0.2], model['energies'], model['shifts'], "
              "model['omega'], model['coordinate'])))",
      'gold_call': "_oracle_contour_sources(*copy.deepcopy(([2, 1, 2], [0.3, -0.7, 0.2], model['energies'], "
                   "model['shifts'], model['omega'], model['coordinate'])))",
      'tol': 2e-08}]
