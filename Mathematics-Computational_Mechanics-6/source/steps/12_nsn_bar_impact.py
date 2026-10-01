"""
Run the whole scheme for a bar launched at a rigid obstacle, with a degradable interface at its midpoint, and report the end-of-run diagnostics. Call the earlier step functions in sequence; do not re-implement them. Conventions this driver fixes: the constrained node is index 0 and the bar moves toward the obstacle at negative velocity; the released element has zero-based index n_elements // 2 counted from the constrained end, so the interface opening is u[mid + 1] - u[mid]; the interface carries the same secant traction in compression as in tension; the traction evaluated at the end-of-step configuration, times the cross-sectional area, enters the end-of-step acceleration of the two interface nodes and hence this step's trapezoidal velocity update; a contact onset is a step whose active set is nonempty after a step whose active set was empty.

End-to-end driver. The bar starts a small distance from the obstacle, all nodes moving toward it. One node carries the unilateral constraint.

Returns
-------
ndarray of shape (5,): far-end nodal velocity, accumulated impulse, kinetic energy, number of contact onsets, final interface damage.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def nsn_bar_impact(n_elements, length, area, youngs, density, impact_speed, restitution, time_step_fraction, initial_gap_fraction, bounce_cycles, cohesive_strength, critical_opening, beta, alpha, tangential_offset):
    """Run the whole scheme for a bar launched at a rigid obstacle, with a degradable interface at its midpoint, and report the end-of-run diagnostics. Call the earlier step functions in sequence; do not re-implement them.

    Returns
    -------
    ndarray of shape (5,): far-end nodal velocity, accumulated impulse, kinetic energy, number of contact onsets, final interface damage.

    Raises
    ------
    ValueError: if bounce_cycles is not positive, or any value it forwards to an earlier step is invalid.
    """
    return np.zeros(1)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_nsn_bar_impact(n_elements, length, area, youngs, density,
                           impact_speed, restitution, time_step_fraction,
                           initial_gap_fraction, bounce_cycles,
                           cohesive_strength, critical_opening, beta,
                           alpha, tangential_offset):
    if bounce_cycles <= 0:
        raise ValueError("bounce_cycles must be positive")
    sysmat = _oracle_bar_system_matrices(n_elements, length, area, youngs, density)
    n = int(n_elements) + 1
    K = sysmat[:n, :]
    Md = sysmat[n, :]
    h_el = float(length) / int(n_elements)
    dtc = _oracle_critical_time_step(n_elements, length, youngs, density)
    dt = float(time_step_fraction) * dtc
    c = np.sqrt(float(youngs) / float(density))
    t_bounce = 2.0 * float(length) / c
    nstep = int(round(float(bounce_cycles) * t_bounce / dt))
    H = np.zeros((1, n), dtype=float); H[0, 0] = 1.0
    v0 = -abs(float(impact_speed))
    u = np.full(n, float(initial_gap_fraction) * abs(v0) * dt, dtype=float)
    v = np.full(n, v0, dtype=float)
    a = (1.0 / Md) * (-(K @ u))
    total_impulse = 0.0
    onsets = 0
    in_contact = False
    # An extrinsic cohesive interface sits at the midpoint. The element spanning it is
    # released so the two halves can separate; the cohesive traction is the only thing
    # holding them together, and it degrades irreversibly as the interface opens.
    mid = int(n_elements) // 2
    K = K.copy()
    kel = float(youngs) * float(area) / h_el
    K[mid:mid + 2, mid:mid + 2] -= kel * np.array([[1.0, -1.0], [-1.0, 1.0]])
    cap = _oracle_cohesive_stiffness_cap(youngs, h_el, alpha, cohesive_strength,
                                         critical_opening)
    d_tilde = float(cap[1])
    damage = np.zeros(1, dtype=float)
    dt_off = np.array([[float(tangential_offset)]], dtype=float)
    a = (1.0 / Md) * (-(K @ u))
    for _ in range(nstep):
        u_pred = _oracle_smooth_predictor(u, v, a, dt)
        active = _oracle_active_contact_set(u, v, a, dt, H)
        if active.any():
            W = _oracle_contact_response_operator(K, Md, H, active, dt)
            p = _oracle_contact_impulse(W, K, Md, H, active, u_pred, v, a, dt, restitution)
            total_impulse += float(np.sum(p))
            if not in_contact:
                onsets += 1
            in_contact = True
        else:
            p = np.zeros(0, dtype=float)
            in_contact = False
        state = _oracle_nonsmooth_state_update(K, Md, H, active, u_pred, v, a, dt, p)
        u, v, a = state[0], state[1], state[2]
        # cohesive interface: opening -> irreversible damage -> traction -> internal force
        dn = np.array([u[mid + 1] - u[mid]], dtype=float)
        eff = _oracle_cohesive_effective_opening(dn, dt_off, beta)
        damage = _oracle_cohesive_damage_update(eff, critical_opening, damage)
        tvec = _oracle_cohesive_traction(dn, dt_off, beta, damage, cohesive_strength,
                                         critical_opening, d_tilde)
        fcoh = float(tvec[0, 0]) * float(area)
        # The traction evaluated at the end-of-step configuration is part of the
        # end-of-step acceleration a_{n+1} (paper, eq. 41a with K(d)), so it enters the
        # trapezoidal velocity update of this step, not only the next step's predictor.
        # The impulse touches only the constrained node, so u_{n+1} and the smooth
        # prediction coincide at the interface nodes and the force is the same on both.
        acoh = (1.0 / Md) * np.eye(n)[mid] * fcoh - (1.0 / Md) * np.eye(n)[mid + 1] * fcoh
        a = a + acoh
        v = v + 0.5 * dt * acoh
    kinetic = 0.5 * float(np.sum(Md * v * v))
    return np.array([float(v[-1]), total_impulse, kinetic, float(onsets),
                     float(damage[0])], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "nsn_bar_impact(50, 0.254, 6.45e-4, 211e9, 7847.0, 5.0, 0.85, 0.40, 3.17, 5.0, 6.0e8, 6e-4, 1.4, 4.0, 2e-5)",
         "gold_call": "_oracle_nsn_bar_impact(50, 0.254, 6.45e-4, 211e9, 7847.0, 5.0, 0.85, 0.40, 3.17, 5.0, 6.0e8, 6e-4, 1.4, 4.0, 2e-5)"},   # normal
        {"setup": "import numpy as np",
         "call": "nsn_bar_impact(10, 0.254, 6.45e-4, 211e9, 7847.0, 5.0, 0.0, 0.40, 3.17, 1.2, 6.0e8, 6e-4, 1.4, 4.0, 2e-5)",
         "gold_call": "_oracle_nsn_bar_impact(10, 0.254, 6.45e-4, 211e9, 7847.0, 5.0, 0.0, 0.40, 3.17, 1.2, 6.0e8, 6e-4, 1.4, 4.0, 2e-5)"},   # boundary
        {"setup": "import numpy as np",
         "call": "nsn_bar_impact(80, 0.254, 6.45e-4, 211e9, 7847.0, 5.0, 1.0, 0.25, 3.17, 3.0, 6.0e8, 6e-4, 1.4, 4.0, 2e-5)",
         "gold_call": "_oracle_nsn_bar_impact(80, 0.254, 6.45e-4, 211e9, 7847.0, 5.0, 1.0, 0.25, 3.17, 3.0, 6.0e8, 6e-4, 1.4, 4.0, 2e-5)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        nsn_bar_impact(10, 0.254, 6.45e-4, 211e9, 7847.0, 5.0, 0.5, 0.4, 3.17, 0.0, 6.0e8, 6e-4, 1.4, 4.0, 2e-5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_nsn_bar_impact(10, 0.254, 6.45e-4, 211e9, 7847.0, 5.0, 0.5, 0.4, 3.17, 0.0, 6.0e8, 6e-4, 1.4, 4.0, 2e-5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
