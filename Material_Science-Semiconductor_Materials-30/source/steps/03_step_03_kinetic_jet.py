"""
Construct the drift, coupled Jacobian and intrinsic diffusion matrix.

The two reduced probabilities share one conserved three-state population. Capture transitions generate anticorrelated increments between charge states.

Returns
-------
return drift, jacobian, diffusion
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def kinetic_lna(p, cfg, density):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    density>=0 is the electron density in cm^-3. Let c,e be Step 1 coefficients,
    psi Step 2 barrier and u=c*density*exp(-psi). For each family,
    F_i1=u_i0*p_i0-(e_i0+u_i1)*p_i1+e_i1*p_i2;
    F_i2=u_i1*p_i1-e_i1*p_i2. A=dF/dp includes the barrier dependence of u.
    The fluctuation coordinate is xi=sqrt(N)*(p_random-p_mean), with N_i=N*weights[i].
    The four channels have reduced stoichiometries (1,0),(-1,0),(-1,1),(1,-1)
    and per-trap rates u_i0*p_i0,e_i0*p_i1,u_i1*p_i1,e_i1*p_i2 respectively.
    D is the sum of rate*outer(stoichiometry,stoichiometry)/weights[i] embedded
    in each family's block. Distinct reaction channels have independent noise.
    Return F(d,), A(d,d), D(d,d). D is for xi, so it contains no factor 1/N.
    Emission is not suppressed by the barrier. No clipping or renormalization.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return drift, jacobian, diffusion

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_kinetic_lna(p, cfg, density):
    import numpy as np
    p = np.asarray(p, float)
    d = p.size
    c, e = _oracle_emission_coefficients(cfg['T'], cfg['sigma'], cfg['depth'], cfg['degeneracy'])
    psi, normal = _oracle_barrier_state(p, cfg['weights'], cfg['alpha'], cfg['eta'])
    u = c*density*np.exp(-psi)
    F = np.zeros(d)
    A = np.zeros((d, d))
    D = np.zeros((d, d))
    for i in range(d//2):
        j = 2*i
        p0, p1, p2 = 1.-p[j]-p[j+1], p[j], p[j+1]
        forward0, reverse0 = u[i, 0]*p0, e[i, 0]*p1
        forward1, reverse1 = u[i, 1]*p1, e[i, 1]*p2
        F[j:j+2] = [forward0-reverse0-forward1+reverse1, forward1-reverse1]
        A[j:j+2, j:j+2] = [[-u[i, 0]-e[i, 0]-u[i, 1], -u[i, 0]+e[i, 1]],
                            [u[i, 1], -e[i, 1]]]
        A[j:j+2] -= np.outer([forward0-forward1, forward1], normal)
        D[j:j+2, j:j+2] = (((forward0+reverse0)*np.array([[1., 0.], [0., 0.]])
                            +(forward1+reverse1)*np.array([[1., -1.], [-1., 1.]]))
                           /cfg['weights'][i])
    return F, A, D

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'coupled',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==3, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(kinetic_lna(p,cfg,protocol["fill_density"]))',
      'gold_call': '_check_result(_oracle_kinetic_lna(p,cfg,protocol["fill_density"]))',
      'tol': 2e-07},
     {'name': 'emission_only',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               '\n'
               'for key in ("sigma","depth","degeneracy"):\n'
               '    cfg[key]=np.asarray(cfg[key])[:1].copy()\n'
               'cfg["weights"]=[1.]\n'
               'p=np.array([.2,.3])\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==3, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(kinetic_lna(p,cfg,0.))',
      'gold_call': '_check_result(_oracle_kinetic_lna(p,cfg,0.))',
      'tol': 2e-07},
     {'name': 'empty',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               '\n'
               'p=np.zeros(2*len(cfg["weights"]))\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==3, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(kinetic_lna(p,cfg,protocol["fill_density"]))',
      'gold_call': '_check_result(_oracle_kinetic_lna(p,cfg,protocol["fill_density"]))',
      'tol': 2e-07},
     {'name': 'two_species',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               '\n'
               'for key in ("sigma","depth","degeneracy"):\n'
               '    cfg[key]=np.asarray(cfg[key])[:2].copy()\n'
               'cfg["weights"]=[.58,.42]\n'
               'p=np.array([.10,.20,.15,.24])\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==3, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(kinetic_lna(p,cfg,protocol["fill_density"]))',
      'gold_call': '_check_result(_oracle_kinetic_lna(p,cfg,protocol["fill_density"]))',
      'tol': 2e-07},
     {'name': 'uncoupled',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               'cfg["alpha"]=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==3, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(kinetic_lna(p,cfg,protocol["fill_density"]))',
      'gold_call': '_check_result(_oracle_kinetic_lna(p,cfg,protocol["fill_density"]))',
      'tol': 2e-07}]
