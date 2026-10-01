"""
Find the attracting mean state at the start of the fill phase.

The cycle starts from the stationary periodic branch, not an empty reset after each pulse.

Returns
-------
return periodic_population
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def periodic_state(cfg, protocol, gates):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    protocol and gates have the meanings and domain from hybrid_cycle. alpha>0.
    Return pstar(d,) satisfying hybrid_cycle(pstar,...)[0]=pstar. Select the unique
    attracting branch reached by repeated cycles from empty populations. All startup
    cycles and a neighbourhood of this fixed point have unique interior transversal
    events; the cycle Jacobian has spectral radius<=.98. Return the fill-start phase.
    The infinity-norm cycle residual must be <=2e-9 on the unrounded state.
    Raises
    ------
    ValueError: if a required hybrid cycle fails, a physical periodic state cannot
    be obtained, or the fixed-point computation does not converge.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return periodic_population

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_periodic_state(cfg, protocol, gates):
    import numpy as np
    from scipy.optimize import root
    d = 2*len(cfg['weights'])
    p = np.zeros(d)
    for _ in range(4):
        p = _oracle_hybrid_cycle(p, cfg, protocol, gates)[0]
    cache = {}
    def _value(v):
        if 'p' not in cache or not np.array_equal(v, cache['p']):
            cycle = _oracle_hybrid_cycle(v, cfg, protocol, gates)
            cache.update(p=v.copy(), f=cycle[0]-v, J=cycle[1]-np.eye(d))
        return cache['f'], cache['J']
    result = root(lambda v: _value(v)[0], p, jac=lambda v: _value(v)[1], tol=1e-10)
    p = result.x
    if np.max(np.abs(_value(p)[0])) > 2e-10:
        raise ValueError('periodic state did not converge')
    if np.min(p) < -1e-10 or np.max(p.reshape(-1, 2).sum(axis=1)) > 1.+1e-10:
        raise ValueError('periodic state outside probability simplex')
    return p

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'main',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n',
      'call': 'periodic_state(cfg,protocol,gates)',
      'gold_call': '_oracle_periodic_state(cfg,protocol,gates)',
      'tol': 2e-07},
     {'name': 'one_species',
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
               'p=np.array([.2,.3])\n',
      'call': 'periodic_state(cfg,protocol,gates)',
      'gold_call': '_oracle_periodic_state(cfg,protocol,gates)',
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
               'p=np.array([.10,.20,.15,.24])\n',
      'call': 'periodic_state(cfg,protocol,gates)',
      'gold_call': '_oracle_periodic_state(cfg,protocol,gates)',
      'tol': 2e-07},
     {'name': 'changed_protocol',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               'protocol["period"]=.02\n'
               'protocol["level"]=.85\n',
      'call': 'periodic_state(cfg,protocol,gates)',
      'gold_call': '_oracle_periodic_state(cfg,protocol,gates)',
      'tol': 2e-07}]
