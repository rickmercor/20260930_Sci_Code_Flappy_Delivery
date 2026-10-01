"""
Propagate the centre and classical action with a specified fourth-order composition.

Single-Hessian propagation retains the full anharmonic force. The action contains both kinetic and potential contributions; it supplies the dynamical phase.

Returns
-------
positions, momenta, actions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def centre_action(mass, q0, p0, origin, powers, coefficients, dt, nsteps):
    r"""Propagate dq/dt=M^{-1}p, dp/dt=-grad V(q), dS/dt=p.T M^{-1}p/2-V(q).
    Start S=0. V is the monomial polynomial defined in potential_jet.
    One VTV(h) step applies the exact potential flow for h/2, the exact kinetic
    flow for h, and the potential flow for h/2, updating S in every flow.
    One requested step is VTV(gamma*dt), VTV((1-2*gamma)*dt), VTV(gamma*dt),
    in that chronological order, where gamma=1/(2-2**(1/3)).
    Return initial values and values after each full requested step.
    Earlier dependency: potential_jet(q, origin, powers, coefficients).

    Parameters
    ----------
    mass : real SPD (D,D) array
    q0, p0, origin : real (D,) arrays
    powers, coefficients : polynomial arrays with the contract of potential_jet
    dt : finite nonzero float; negative dt is allowed
    nsteps : nonnegative integer

    Returns
    -------
    positions, momenta : real arrays, shape (nsteps+1,D)
    actions : real array, shape (nsteps+1,)

    Raises
    ------
    ValueError
        For invalid polynomial data as defined in potential_jet; invalid shapes,
        empty or nonfinite state; mass not symmetric within 1e-12 or not SPD;
        zero or nonfinite dt; or nsteps not a nonnegative integer.
        The supported data have finite trajectories over the requested interval.
    """
    return positions, momenta, actions

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_centre_action(mass, q0, p0, origin, powers, coefficients, dt, nsteps):
    import numpy as np
    mass = np.asarray(mass, dtype=float)
    q = np.asarray(q0, dtype=float).copy()
    p = np.asarray(p0, dtype=float).copy()
    if (q.ndim != 1 or q.size == 0 or p.shape != q.shape
            or mass.shape != (q.size, q.size)
            or not all(np.all(np.isfinite(a)) for a in (mass, q, p))
            or not np.allclose(mass, mass.T, atol=1e-12, rtol=0)
            or np.min(np.linalg.eigvalsh(mass)) <= 0
            or not np.isfinite(dt) or dt == 0
            or not isinstance(nsteps, (int, np.integer)) or nsteps < 0):
        raise ValueError("Invalid trajectory inputs")
    _oracle_potential_jet(q, origin, powers, coefficients)
    inverse_mass = np.linalg.inv(mass)
    qs = np.empty((nsteps + 1, q.size))
    ps = np.empty_like(qs)
    actions = np.empty(nsteps + 1)
    qs[0], ps[0], actions[0] = q, p, 0.0
    action = 0.0
    gamma = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    for step in range(nsteps):
        for scale in (gamma, 1.0 - 2.0 * gamma, gamma):
            h = dt * scale
            value, gradient, _ = _oracle_potential_jet(q, origin, powers, coefficients)
            p -= 0.5 * h * gradient
            action -= 0.5 * h * value
            action += 0.5 * h * (p @ inverse_mass @ p)
            q += h * (inverse_mass @ p)
            value, gradient, _ = _oracle_potential_jet(q, origin, powers, coefficients)
            p -= 0.5 * h * gradient
            action -= 0.5 * h * value
        qs[step + 1], ps[step + 1], actions[step + 1] = q, p, action
    return qs, ps, actions

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'case_1',
            "setup": """import numpy as np
data={'mass': [[1.2, 0.12, -0.08], [0.12, 1.5, 0.1], [-0.08, 0.1, 0.9]], 'q_initial': [-0.35, 0.22, 0.18], 'origin': [0.1, -0.15, 0.08], 'potential_powers': [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [2, 1, 0], [1, 1, 1], [4, 0, 0], [0, 4, 0], [0, 0, 4], [2, 2, 0], [0, 2, 2], [2, 0, 2]], 'potential_coefficients': [1.6, 0.5, 0.925, 0.65, 0.21, -0.16, 0.27, 0.06, -0.04, 0.03, 0.08, -0.055, 0.018, 0.024, 0.02, 0.012, 0.016, 0.01]}
""",
            "call": "centre_action(data['mass'], data['q_initial'], np.zeros(3), data['origin'], data['potential_powers'], data['potential_coefficients'], 0.04, 25)",
            "gold_call": "_oracle_centre_action(data['mass'], data['q_initial'], np.zeros(3), data['origin'], data['potential_powers'], data['potential_coefficients'], 0.04, 25)",
        },
        {
            "name": 'case_2',
            "setup": """import numpy as np
M=np.eye(1);q=np.array([.2]);p=np.array([.3]);origin=np.zeros(1);powers=np.array([[2]]);a=np.array([.5])
""",
            "call": 'centre_action(M, q, p, origin, powers, a, 0.04, 0)',
            "gold_call": '_oracle_centre_action(M, q, p, origin, powers, a, 0.04, 0)',
        },
        {
            "name": 'case_3',
            "setup": """import numpy as np
M=np.array([[2.,.1],[.1,1.]]);q=np.array([.2,-.3]);p=np.array([.1,.4]);origin=np.zeros(2);powers=np.array([[0,0]]);a=np.array([.7])
""",
            "call": 'centre_action(M, q, p, origin, powers, a, 0.03, 7)',
            "gold_call": '_oracle_centre_action(M, q, p, origin, powers, a, 0.03, 7)',
        },
        {
            "name": 'case_4',
            "setup": """import numpy as np
M=np.eye(1);q=np.array([.2]);p=np.array([.3]);origin=np.zeros(1);powers=np.array([[2],[4]]);a=np.array([.5,.1])
""",
            "call": 'centre_action(M, q, p, origin, powers, a, -0.04, 11)',
            "gold_call": '_oracle_centre_action(M, q, p, origin, powers, a, -0.04, 11)',
        },
        {
            "name": 'case_5',
            "setup": """import numpy as np
M=np.eye(1);q=np.zeros(1);p=q.copy();origin=q.copy();powers=np.array([[2]]);a=np.ones(1)

def _status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
""",
            "call": '_status(centre_action, M, q, p, origin, powers, a, 0.0, 2)',
            "gold_call": '_status(_oracle_centre_action, M, q, p, origin, powers, a, 0.0, 2)',
        },
    ]
