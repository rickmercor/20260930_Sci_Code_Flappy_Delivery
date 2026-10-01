"""
Form one cycle with joint fluctuations at two moving read gates.

An occupancy-triggered switch changes the pulse duration. Its timing fluctuation is shared by both observations and the next cycle state.

Returns
-------
return endpoint, M, C, Qxx, Qxz, Qzz, mean_gates, tau
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hybrid_cycle(p, cfg, protocol, gates):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    Here alpha>0. protocol has period>0 seconds, level>0 dimensionless,
    fill_density>0 and read_density>=0 in cm^-3. gates is exactly two fractions
    0<=r1<=r2<=1. Fill until the first upward crossing psi=level at tau; read until
    the fixed end time period. The mean state is continuous at the switch.
    Observe each family's raw charge p_i1+2*p_i2 at
     t_j=tau+r_j*(period-tau).
    These observation times follow the perturbed event, not the nominal clock.
    Use the leading Gaussian linearization of this hybrid process: on each smooth
    segment xi obeys the SDE of gaussian_flow; linearize the threshold crossing
    and the observation times with respect to initial fluctuations and all earlier
    noise. The threshold, total period and fractions are fixed. There is no state
    reset, added switch noise or independent timing-noise source. Noise increments
    on disjoint nominal smooth intervals are independent. Retain correlations
    caused by their shared history and shared event-time fluctuation.
    For arbitrary deterministic initial p, define
     xi_next=M*xi_start+epsilon_x,
     z=C*xi_start+epsilon_z,
    where z has 2*m entries: all first-gate charge fluctuations followed by all
    second-gate charge fluctuations, scaled by sqrt(N); no weights in raw charges.
    The joint innovation is independent of xi_start. Qxx=Cov(epsilon_x),
    Qxz=Cov(epsilon_x,epsilon_z), Qzz=Cov(epsilon_z). Gate-state noise and endpoint
    noise are generally correlated. Return exactly eight values in this order:
     endpoint(d,), M(d,d), C(2*m,d), Qxx(d,d), Qxz(d,2*m), Qzz(2*m,2*m),
     mean_gates(2*m,), tau(float seconds).
    Coincident gates and gate endpoints 0,1 are supported; covariance may be singular.
    For normal inputs the first crossing is unique, transversal and interior.
    Raises
    ------
    ValueError: if initial psi>=level, no upward event occurs strictly before period,
    the event is not transversal/upward, or a required integration fails.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return endpoint, M, C, Qxx, Qxz, Qzz, mean_gates, tau

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_hybrid_cycle(p, cfg, protocol, gates):
    import numpy as np
    from scipy.integrate import solve_ivp
    p = np.asarray(p, float)
    gates = np.asarray(gates, float)
    d = p.size
    m = d//2
    period, level = protocol['period'], protocol['level']
    nf, nr = protocol['fill_density'], protocol['read_density']
    def _event(t, q):
        return _oracle_barrier_state(q, cfg['weights'], cfg['alpha'], cfg['eta'])[0]-level
    _event.terminal = True
    _event.direction = 1
    if _event(0., p) >= 0:
        raise ValueError('initial barrier must be below level')
    sol = solve_ivp(lambda t, q: _oracle_kinetic_lna(q, cfg, nf)[0],
                    (0., period), p, events=_event, method='DOP853', rtol=3e-12, atol=3e-14)
    if not sol.success or len(sol.t_events[0]) == 0:
        raise ValueError('no interior upward event')
    tau = float(sol.t_events[0][0])
    if not 0. < tau < period:
        raise ValueError('event must be strictly interior')
    q, Pf, Vf = _oracle_gaussian_flow(p, cfg, nf, tau)
    fminus = _oracle_kinetic_lna(q, cfg, nf)[0]
    normal = _oracle_barrier_state(q, cfg['weights'], cfg['alpha'], cfg['eta'])[1]
    speed = normal@fminus
    if speed <= 0:
        raise ValueError('event must be transversal and upward')
    clock = normal/speed  # b=-sqrt(N)*delta_tau
    total = d+1+2*m
    E = np.zeros((total, d))
    E[:d] = np.eye(d)-np.outer(fminus, clock)
    E[d] = clock
    B = E@Pf
    V = E@Vf@E.T
    charge = np.zeros((m, d))
    for i in range(m):
        charge[i, 2*i:2*i+2] = [1., 2.]
    mean_gates = np.zeros(2*m)
    previous = 0.
    for j, fraction in enumerate([*gates, 1.]):
        q, Phi, W = _oracle_gaussian_flow(q, cfg, nr, (fraction-previous)*(period-tau))
        transition = np.eye(total)
        transition[:d, :d] = Phi
        B = transition@B
        V = transition@V@transition.T
        V[:d, :d] += W
        f = _oracle_kinetic_lna(q, cfg, nr)[0]
        if j < 2:
            rows = slice(d+1+j*m, d+1+(j+1)*m)
            sample = np.eye(total)
            sample[rows] = 0.
            sample[rows, :d] = charge
            sample[rows, d] = fraction*(charge@f)
            B = sample@B
            V = sample@V@sample.T
            mean_gates[j*m:(j+1)*m] = charge@q
        else:
            sample = np.eye(total)
            sample[:d, d] = f
            B = sample@B
            V = sample@V@sample.T
        previous = fraction
    V = (V+V.T)/2.
    keep = slice(d+1, total)
    return (q, B[:d], B[keep], V[:d, :d], V[:d, keep], V[keep, keep],
            mean_gates, tau)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'name': 'empty_cycle',
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
               '    assert len(value)==8, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(hybrid_cycle(p,cfg,protocol,gates))',
      'gold_call': '_check_result(_oracle_hybrid_cycle(p,cfg,protocol,gates))',
      'tol': 2e-07},
     {'name': 'interior_cycle',
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
               '    assert len(value)==8, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(hybrid_cycle(p,cfg,protocol,gates))',
      'gold_call': '_check_result(_oracle_hybrid_cycle(p,cfg,protocol,gates))',
      'tol': 2e-07},
     {'name': 'endpoint_gates',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               'gates=np.array([0.,1.])\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==8, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(hybrid_cycle(p,cfg,protocol,gates))',
      'gold_call': '_check_result(_oracle_hybrid_cycle(p,cfg,protocol,gates))',
      'tol': 2e-07},
     {'name': 'coincident_gates',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               'gates=np.array([.5,.5])\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==8, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(hybrid_cycle(p,cfg,protocol,gates))',
      'gold_call': '_check_result(_oracle_hybrid_cycle(p,cfg,protocol,gates))',
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
               '    assert len(value)==8, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(hybrid_cycle(p,cfg,protocol,gates))',
      'gold_call': '_check_result(_oracle_hybrid_cycle(p,cfg,protocol,gates))',
      'tol': 2e-07},
     {'name': 'emission_read',
      'setup': 'import numpy as np\n'
               "cfg={'T': 220.0, 'sigma': [[2e-15, 1.3e-15], [1.1e-15, 2.4e-15], [3.2e-15, 8e-16]], "
               "'depth': [[0.4, 0.46], [0.42, 0.48], [0.44, 0.45]], 'degeneracy': [[2.0, 0.5], [1.0, "
               "2.0], [0.5, 1.0]], 'weights': [0.46, 0.31, 0.23], 'alpha': 1.7, 'eta': 0.12}\n"
               "protocol={'period': 0.016, 'level': 1.0, 'fill_density': 20000000000.0, "
               "'read_density': 100000000.0}\n"
               'gates=np.array([1./12.,.75])\n'
               'p=np.array([.03,.12,.06,.16,.2,.18])\n'
               'protocol["read_density"]=0.\n'
               '\n'
               'def _check_result(value):\n'
               '    assert len(value)==8, "incorrect number of returned values"\n'
               '    return value\n',
      'call': '_check_result(hybrid_cycle(p,cfg,protocol,gates))',
      'gold_call': '_check_result(_oracle_hybrid_cycle(p,cfg,protocol,gates))',
      'tol': 2e-07},
     {'name': 'initial_on_surface',
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
               'cfg["alpha"]=2.\n'
               'p=np.array([.25,0.])\n'
               'protocol["level"]=.5\n'
               '\n'
               'def _capture_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('expected ValueError')\n",
      'call': '_capture_value_error(lambda: hybrid_cycle(p,cfg,protocol,gates))',
      'gold_call': '_capture_value_error(lambda: _oracle_hybrid_cycle(p,cfg,protocol,gates))',
      'tol': 2e-07},
     {'name': 'event_not_reached',
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
               'protocol["period"]=1e-7\n'
               '\n'
               'def _capture_value_error(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('expected ValueError')\n",
      'call': '_capture_value_error(lambda: hybrid_cycle(p,cfg,protocol,gates))',
      'gold_call': '_capture_value_error(lambda: _oracle_hybrid_cycle(p,cfg,protocol,gates))',
      'tol': 2e-07}]
