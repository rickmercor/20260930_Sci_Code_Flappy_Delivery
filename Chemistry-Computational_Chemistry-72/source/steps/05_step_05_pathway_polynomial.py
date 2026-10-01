"""
Contract insertion moments with electronic dipoles.

Franck–Condon and Herzberg–Teller amplitudes interfere along the same chronological electronic pathway.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pathway_polynomial(path: "np.ndarray", mu0: "np.ndarray", mu1: "np.ndarray", moments: "np.ndarray") -> "np.ndarray":
    """Return the HT polynomial for one electronic pathway.

    Parameters
    ----------
    path : array-like
        Electronic states during the M chronological intervals.
    mu0, mu1 : array-like
        Real symmetric electronic dipole matrices with shape (n,n),
        defining the dipole mu(lambda)=mu0+lambda*mu1*Q.
    moments : array-like
        Packed complex ordered coordinate-insertion moments with shape
        (2**(M+1),2). Bit j of the moment index specifies a coordinate
        insertion at chronological optical interaction j; an unset bit
        specifies no coordinate insertion at that position.

    Returns
    -------
    coeff : ndarray
        Packed complex polynomial coefficients of lambda^k for this
        electronic pathway, with shape (M+2,2), ordered by increasing k.

    Notes
    -----
    The complete chronological state sequence is (0,path[0],...,path[M-1],0).
    Dipole matrix indices follow the convention [next state, previous state].
    Coefficients have ordinary polynomial normalization, with no factorial
    or additional phase. Complex outputs use final axis [real, imaginary].
    """
    return coeff

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_pathway_polynomial(path: "np.ndarray", mu0: "np.ndarray", mu1: "np.ndarray", moments: "np.ndarray") -> "np.ndarray":
    import numpy as np
    path, mu0, mu1 = np.asarray(path, int), np.asarray(mu0), np.asarray(mu1)
    moments = np.asarray(moments)
    moments = moments[:, 0]+1j*moments[:, 1]
    states = [0] + list(path) + [0]
    r = len(states)-1
    coeff = np.zeros(r+1, complex)
    for mask, moment in enumerate(moments):
        factor = 1.
        for j in range(r):
            matrix = mu1 if mask & (1 << j) else mu0
            factor *= matrix[states[j+1], states[j]]
        coeff[mask.bit_count()] += factor * moment
    return np.stack((coeff.real, coeff.imag), -1)

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
               'moments_c=squarefree_moments(copy.deepcopy(c_c),copy.deepcopy(l_c),copy.deepcopy(B_c))\n'
               "u_g,v_g,phase_g=_oracle_contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               "c_g,l_g,B_g=_oracle_gaussian_generator(copy.deepcopy(u_g),copy.deepcopy(v_g),phase_g,copy.deepcopy(model['omega']),model['beta'])\n"
               'moments_g=_oracle_squarefree_moments(copy.deepcopy(c_g),copy.deepcopy(l_g),copy.deepcopy(B_g))\n'
               '\n',
      'call': "pathway_polynomial(*copy.deepcopy(([1, 2, 1, 2, 1], model['mu0'], model['mu1'], moments_c)))",
      'gold_call': "_oracle_pathway_polynomial(*copy.deepcopy(([1, 2, 1, 2, 1], model['mu0'], model['mu1'], "
                   'moments_g)))',
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
               'moments_c=squarefree_moments(copy.deepcopy(c_c),copy.deepcopy(l_c),copy.deepcopy(B_c))\n'
               "u_g,v_g,phase_g=_oracle_contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               "c_g,l_g,B_g=_oracle_gaussian_generator(copy.deepcopy(u_g),copy.deepcopy(v_g),phase_g,copy.deepcopy(model['omega']),model['beta'])\n"
               'moments_g=_oracle_squarefree_moments(copy.deepcopy(c_g),copy.deepcopy(l_g),copy.deepcopy(B_g))\n'
               '\n',
      'call': "pathway_polynomial(*copy.deepcopy(([1, 2, 1, 2, 1], model['mu0'], np.zeros((3, 3)), moments_c)))",
      'gold_call': "_oracle_pathway_polynomial(*copy.deepcopy(([1, 2, 1, 2, 1], model['mu0'], np.zeros((3, 3)), "
                   'moments_g)))',
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
               'moments_c=squarefree_moments(copy.deepcopy(c_c),copy.deepcopy(l_c),copy.deepcopy(B_c))\n'
               "u_g,v_g,phase_g=_oracle_contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               "c_g,l_g,B_g=_oracle_gaussian_generator(copy.deepcopy(u_g),copy.deepcopy(v_g),phase_g,copy.deepcopy(model['omega']),model['beta'])\n"
               'moments_g=_oracle_squarefree_moments(copy.deepcopy(c_g),copy.deepcopy(l_g),copy.deepcopy(B_g))\n'
               '\n',
      'call': "pathway_polynomial(*copy.deepcopy(([1, 2, 1, 2, 1], np.zeros((3, 3)), model['mu1'], moments_c)))",
      'gold_call': "_oracle_pathway_polynomial(*copy.deepcopy(([1, 2, 1, 2, 1], np.zeros((3, 3)), model['mu1'], "
                   'moments_g)))',
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
               'moments_c=squarefree_moments(copy.deepcopy(c_c),copy.deepcopy(l_c),copy.deepcopy(B_c))\n'
               "u_g,v_g,phase_g=_oracle_contour_sources([1,2,1,2,1],[.17,.23,.41,.19,.96],copy.deepcopy(model['energies']),copy.deepcopy(model['shifts']),copy.deepcopy(model['omega']),copy.deepcopy(model['coordinate']))\n"
               "c_g,l_g,B_g=_oracle_gaussian_generator(copy.deepcopy(u_g),copy.deepcopy(v_g),phase_g,copy.deepcopy(model['omega']),model['beta'])\n"
               'moments_g=_oracle_squarefree_moments(copy.deepcopy(c_g),copy.deepcopy(l_g),copy.deepcopy(B_g))\n'
               "model['mu1']=np.array(model['mu1']);model['mu1'][1,2]=model['mu1'][2,1]=0.\n",
      'call': "pathway_polynomial(*copy.deepcopy(([1, 2, 1, 2, 1], model['mu0'], model['mu1'], moments_c)))",
      'gold_call': "_oracle_pathway_polynomial(*copy.deepcopy(([1, 2, 1, 2, 1], model['mu0'], model['mu1'], "
                   'moments_g)))',
      'tol': 2e-08}]
