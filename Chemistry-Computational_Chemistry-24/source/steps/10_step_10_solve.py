"""
Return the requested energy-efficiency benchmark for one complete kinetic instance.

The benchmark combines turnover and stationary energetic quantities for the supplied kinetic instance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve(data: "dict") -> float:
    """Compute the requested source-defined scalar for one inhibited dynamic catalyst.

    ``data`` must contain finite positive values for ``u0,u1,u2,w0,w1,beta0,x,y,
    gamma1,gamma2`` satisfying the contracts of the preceding steps. All tested
    instances have catalytic efficiency greater than one.

    Returns
    -------
    float
        Source-defined energy-efficiency scalar.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve(data: "dict") -> float:
    rates = _oracle_parallel_rates(data['u0'], data['u1'], data['u2'], data['w1'], data['beta0'], data['x'], data['y'])
    q = _oracle_master_generator(data['u0'], data['u1'], data['u2'], data['w0'], data['w1'], data['beta0'], data['x'], data['y'], data['gamma1'], data['gamma2'])
    fp = _oracle_product_first_passage(q, data['u2'], float(rates[2]))
    turnover = _oracle_nonrenewal_turnover(fp)
    tau_static = _oracle_static_turnover(data['u0'], data['u1'], data['u2'], data['w0'], data['w1'])
    ef = _oracle_catalytic_efficiency(tau_static, float(turnover[0]))
    stationary = _oracle_stationary_distribution(q)
    diss = _oracle_cycle_dissipation(stationary, data['u0'], data['u1'], data['u2'], data['w0'], data['w1'], data['beta0'], data['x'], data['y'], data['gamma1'], data['gamma2'])
    return _oracle_energy_efficiency(float(diss[-1]), ef)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary and edge cases with isolated inputs."""
    return [
        {
            'setup': 'data=dict(u0=10.0,u1=3.0,u2=1.0,w0=4.0,w1=1/3,beta0=1.0,x=.1,y=.1,gamma1=2.3,gamma2=.9)',
            'call': 'solve(data.copy())',
            'gold_call': '_oracle_solve(data.copy())',
            'tol': 1e-10,
        },
        {
            'setup': "data=dict(u0=10.0,u1=3.0,u2=1.0,w0=4.0,w1=1/3,beta0=1.0,x=.1,y=.1,gamma1=2.3,gamma2=.9)\ndata['gamma1']=4.1;data['gamma2']=1.7",
            'call': 'solve(data.copy())',
            'gold_call': '_oracle_solve(data.copy())',
            'tol': 1e-10,
        },
        {
            'setup': "data=dict(u0=10.0,u1=3.0,u2=1.0,w0=4.0,w1=1/3,beta0=1.0,x=.1,y=.1,gamma1=2.3,gamma2=.9)\ndata['x']=.2",
            'call': 'solve(data.copy())',
            'gold_call': '_oracle_solve(data.copy())',
            'tol': 1e-10,
        },
        {
            'setup': "data=dict(u0=10.0,u1=3.0,u2=1.0,w0=4.0,w1=1/3,beta0=1.0,x=.1,y=.1,gamma1=2.3,gamma2=.9)\ndata['y']=.2",
            'call': 'solve(data.copy())',
            'gold_call': '_oracle_solve(data.copy())',
            'tol': 1e-10,
        },
        {
            'setup': 'data = dict(u0=10., u1=3., u2=1., w0=4., w1=1/3, beta0=1., x=.1, y=.1, gamma1=1e-4, gamma2=.9)',
            'call': 'solve(data.copy())',
            'gold_call': '_oracle_solve(data.copy())',
            'tol': 1e-09,
        },
    ]
