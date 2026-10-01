"""
Propagate the mean, transition matrix and accumulated process covariance.

Noise generated during a segment is transported by the time-dependent coupled population dynamics.

Returns
-------
return endpoint, transition, process_covariance
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gaussian_flow(p, cfg, density, duration):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    density>=0; duration>=0 seconds. Along q'=F(q), use A(q),D(q) from kinetic_lna.
    The Gaussian fluctuation solves dxi=A(q)xi dt+B(q)dW, with B B^T=D(q).
    Return q_end(d,), Phi(d,d), V(d,d) such that xi_end=Phi*xi_start+epsilon,
    Cov(epsilon)=V and epsilon is independent of xi_start. Coefficients vary along q.
    At duration 0 return p.copy(), identity, zeros. Phi initially identity; V initially 0.
    Raises
    ------
    ValueError: if numerical integration fails.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return endpoint, transition, process_covariance

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_gaussian_flow(p, cfg, density, duration):
    import numpy as np
    from scipy.integrate import solve_ivp
    p = np.asarray(p, float)
    d = p.size
    if duration == 0:
        return p.copy(), np.eye(d), np.zeros((d, d))
    initial = np.r_[p, np.eye(d).ravel(), np.zeros(d*d)]
    def _rhs(t, value):
        q = value[:d]
        Phi = value[d:d+d*d].reshape(d, d)
        V = value[d+d*d:].reshape(d, d)
        f, A, D = _oracle_kinetic_lna(q, cfg, density)
        return np.r_[f, (A@Phi).ravel(), (A@V+V@A.T+D).ravel()]
    sol = solve_ivp(_rhs, (0., duration), initial, method='DOP853', rtol=3e-12, atol=3e-14)
    if not sol.success:
        raise ValueError('Gaussian flow integration failed')
    end = sol.y[:, -1]
    V = end[d+d*d:].reshape(d, d)
    return end[:d], end[d:d+d*d].reshape(d, d), (V+V.T)/2.

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'capture_segment',
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
      'call': '_check_result(gaussian_flow(p,cfg,protocol["fill_density"],.003))',
      'gold_call': '_check_result(_oracle_gaussian_flow(p,cfg,protocol["fill_density"],.003))',
      'tol': 2e-07},
     {'name': 'zero_duration',
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
      'call': '_check_result(gaussian_flow(p,cfg,1e8,0.))',
      'gold_call': '_check_result(_oracle_gaussian_flow(p,cfg,1e8,0.))',
      'tol': 2e-07},
     {'name': 'emission_segment',
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
      'call': '_check_result(gaussian_flow(p,cfg,0.,.009))',
      'gold_call': '_check_result(_oracle_gaussian_flow(p,cfg,0.,.009))',
      'tol': 2e-07},
     {'name': 'short_segment',
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
               'p=np.zeros(2*len(cfg["weights"]))\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==3, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(gaussian_flow(p,cfg,2e10,1e-8))',
      'gold_call': '_check_result(_oracle_gaussian_flow(p,cfg,2e10,1e-8))',
      'tol': 2e-07},
     {'name': 'two_species_read',
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
      'call': '_check_result(gaussian_flow(p,cfg,1e8,.006))',
      'gold_call': '_check_result(_oracle_gaussian_flow(p,cfg,1e8,.006))',
      'tol': 2e-07}]
