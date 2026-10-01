"""
Evaluate all distinct-source Gaussian derivatives.

Each optical interaction contributes at most one coordinate insertion, and contractions can connect distinct insertion positions.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def squarefree_moments(c: "np.ndarray", linear: "np.ndarray", quadratic: "np.ndarray") -> "np.ndarray":
    """Return all distinct-source derivatives of a Gaussian generating function.

    Parameters
    ----------
    c : array-like
        Packed complex constant with shape (2,).
    linear : array-like
        Packed complex linear coefficients with shape (r,2), 1<=r<=10.
    quadratic : array-like
        Packed complex symmetric quadratic coefficients with shape (r,r,2).

    Returns
    -------
    moments : ndarray
        Packed complex array with shape (2**r,2). Entry mask is the
        derivative at zero with respect to each source whose bit is set,
        with every selected source differentiated exactly once.

    Notes
    -----
    The input convention defines the generating function as
    exp(c+l^T s+0.5 s^T B s). Bit j denotes source s_j. Return ordinary
    derivatives without factorial normalization, including the empty
    derivative at mask 0. Complex outputs use final axis [real, imaginary].
    """
    return moments

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_squarefree_moments(c: "np.ndarray", linear: "np.ndarray", quadratic: "np.ndarray") -> "np.ndarray":
    import numpy as np
    c, linear, quadratic = np.asarray(c), np.asarray(linear), np.asarray(quadratic)
    c = c[0]+1j*c[1]
    linear = linear[..., 0]+1j*linear[..., 1]
    quadratic = quadratic[..., 0]+1j*quadratic[..., 1]
    r = len(linear)
    values = np.zeros(1 << r, complex)
    values[0] = np.exp(c)
    for mask in range(1, len(values)):
        bit = mask & -mask
        i = bit.bit_length()-1
        rest = mask ^ bit
        value = linear[i] * values[rest]
        for j in range(r):
            if rest & (1 << j):
                value += quadratic[i, j] * values[rest ^ (1 << j)]
        values[mask] = value
    return np.stack((values.real, values.imag), -1)

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
               "c_c,l_c,B_c=gaussian_generator(copy.deepcopy(u_c),copy.deepcopy(v_c),phase_c,copy.deepcopy(model['omega']),model['beta'])\n"
               "u_g,v_g,phase_g=_oracle_contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               "c_g,l_g,B_g=_oracle_gaussian_generator(copy.deepcopy(u_g),copy.deepcopy(v_g),phase_g,copy.deepcopy(model['omega']),model['beta'])\n"
               '\n',
      'call': 'squarefree_moments(*copy.deepcopy((c_c, l_c, B_c)))',
      'gold_call': '_oracle_squarefree_moments(*copy.deepcopy((c_g, l_g, B_g)))',
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
               '\n',
      'call': 'squarefree_moments(*copy.deepcopy((pack(0.1 + 0.3j), pack([0.7 - 0.2j]), pack([[3.0 + 4j]]))))',
      'gold_call': '_oracle_squarefree_moments(*copy.deepcopy((pack(0.1 + 0.3j), pack([0.7 - 0.2j]), pack([[3.0 '
                   '+ 4j]]))))',
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
               "c_c,l_c,B_c=gaussian_generator(copy.deepcopy(u_c),copy.deepcopy(v_c),phase_c,copy.deepcopy(model['omega']),model['beta'])\n"
               "u_g,v_g,phase_g=_oracle_contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               "c_g,l_g,B_g=_oracle_gaussian_generator(copy.deepcopy(u_g),copy.deepcopy(v_g),phase_g,copy.deepcopy(model['omega']),model['beta'])\n"
               'B_c[np.arange(6),np.arange(6),0]+=100.\n'
               'B_g[np.arange(6),np.arange(6),0]+=100.\n',
      'call': 'squarefree_moments(*copy.deepcopy((c_c, l_c, B_c)))',
      'gold_call': '_oracle_squarefree_moments(*copy.deepcopy((c_g, l_g, B_g)))',
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
               '\n',
      'call': 'squarefree_moments(*copy.deepcopy((pack(0), pack(np.zeros(4)), pack(np.array([[0.0, 1.0, 0.2, '
              '0.3], [1.0, 0.0, -0.4, 0.6], [0.2, -0.4, 0.0, 0.8], [0.3, 0.6, 0.8, 0.0]])))))',
      'gold_call': '_oracle_squarefree_moments(*copy.deepcopy((pack(0), pack(np.zeros(4)), pack(np.array([[0.0, '
                   '1.0, 0.2, 0.3], [1.0, 0.0, -0.4, 0.6], [0.2, -0.4, 0.0, 0.8], [0.3, 0.6, 0.8, 0.0]])))))',
      'tol': 2e-08}]
