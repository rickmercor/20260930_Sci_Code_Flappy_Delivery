"""
Solve the electrostatic barrier and its population gradient.

The gradient couples every defect family through the total trapped charge.

Returns
-------
return psi, normal
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def barrier_state(p, weights, alpha, eta):
    """Return float psi and its population gradient normal of shape(d,).
    p has d=2*m entries; weights has m positive entries summing to1. alpha>=0,
    eta>=0. Finite off-simplex p is allowed in this step. Q=sum_i weights[i]*
    (p[2*i]+2*p[2*i+1]); psi is the unique real root psi+eta*psi^3=alpha*Q.
    normal[j]=dpsi/dp[j] in ambient coordinates. Include alpha=0 and eta=0 limits.
    Only population derivatives are returned; there are no x,y coordinates.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return psi, normal

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_barrier_state(p, weights, alpha, eta):
    import numpy as np
    from scipy.optimize import brentq
    p = np.asarray(p, float)
    w = np.repeat(np.asarray(weights, float), 2)*np.tile([1., 2.], len(weights))
    r = alpha*(w@p)
    if eta == 0 or r == 0:
        psi = r
    else:
        psi = brentq(lambda v: v+eta*v**3-r, min(0., r), max(0., r), xtol=5e-15)
    normal = alpha*w/(1.+3.*eta*psi**2)
    return float(psi), normal

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'interior',
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
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
      'gold_call': '_check_result(_oracle_barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
      'tol': 2e-07},
     {'name': 'linear',
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
               'cfg["eta"]=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
      'gold_call': '_check_result(_oracle_barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
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
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
      'gold_call': '_check_result(_oracle_barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
      'tol': 2e-07},
     {'name': 'zero_coupling',
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
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
      'gold_call': '_check_result(_oracle_barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
      'tol': 2e-07},
     {'name': 'negative_ambient_charge',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               'p=-p\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==2, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
      'gold_call': '_check_result(_oracle_barrier_state(p,cfg["weights"],cfg["alpha"],cfg["eta"]))',
      'tol': 2e-07}]
