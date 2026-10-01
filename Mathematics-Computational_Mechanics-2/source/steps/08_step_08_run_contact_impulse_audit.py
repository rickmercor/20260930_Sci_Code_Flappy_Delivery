"""
Differentiate the fully discrete contact impulse with respect to log stiffness.



For the three fixed experiments, replace every stiffness by $e^\eta k_r$.

The returned scalar is $S(\lambda)=dJ(\eta)/d\eta$ at

$\eta=\log\lambda$, where $J$ sums the norms of normalized final generalized momentum changes.

Differentiate the source's finite-step kick-drift-kick map, with initial

positions and velocities held fixed, and rebuild the active contact geometry

at each force evaluation. This is a sensitivity of the discrete trajectory,

not an integral of partial force derivatives at frozen positions.

The moving state is $q=(\theta,\psi)$: body 2 has center

$c_2+\theta d$ and angle $\phi_2+\psi$, and body 1 stays fixed.

The generalized mass is $\operatorname{diag}(M,I)$, initial $\psi=0$,

and initial angular velocities are $0.15,-0.12,0.08$ radians per second.

The corresponding inertias are $0.18,0.21,0.095$ kilograms square metres.

Normalize linear and angular momentum changes by one SI unit of their

respective dimensions before taking their Euclidean norm.

The sample fibers remain fixed. Differentiate locally through each unique

active endpoint; do not differentiate the integer owner labels or insert

continuous-time event corrections. At exact owner ties or tangencies the

underlying derivative contract rejects the state.



Fixed experiment data (SI units; angles in radians):



| $r$ | box $(x_{\min},x_{\max},y_{\min},y_{\max})$ | $(n_h,n_v)$ | $c_1$; $(a_1,b_1)$; $\phi_1$ | $c_2$; $(a_2,b_2)$; $\phi_2$ | $d$ | $(\theta_0,v_0,M,k,\Delta t,N_t)$ |

|---:|---|---|---|---|---|---|

| 1 | $(-1.4,1.4,-1.1,1.1)$ | $(64,64)$ | $(-0.22,0)$; $(0.82,0.55)$; $0.20$ | $(0.38,0.07)$; $(0.70,0.46)$; $-0.32$ | $(1,0)$ | $(0,-0.35,1.7,480,0.002,50)$ |

| 2 | $(-1.6,1.6,-1.2,1.2)$ | $(48,48)$ | $(-0.30,-0.04)$; $(0.94,0.48)$; $-0.28$ | $(0.46,0.12)$; $(0.63,0.51)$; $0.41$ | $(0.9659258263,0.2588190451)$ | $(0,-0.29,2.1,620,0.0015,60)$ |

| 3 | $(-1.2,1.2,-1.2,1.2)$ | $(80,80)$ | $(-0.18,0)$; $(0.68,0.52)$; $0.15$ | $(0.69,0.03)$; $(0.44,0.36)$; $-0.20$ | $(1,0)$ | $(0,-0.22,1.3,700,0.001,80)$ |



Fibers are midpoint horizontal lines followed by midpoint vertical lines.

The two families each carry half of the sampling-density weight; use the

stated root and transversality tolerances $10^{-12}$ and $10^{-10}$.

The moving body translates and rotates, with $q=(\theta,\psi)$ and generalized mass $D=\operatorname{diag}(M,I)$. The force evaluation must return both generalized force $F$ and its full state Jacobian $K$ from the current energy-force matrix, including the mixed translation-rotation terms. Under $k_r(\eta)=e^\eta k_r$, the fibers and initial data stay fixed. Differentiate the complete finite-step trajectory, then the norm of each terminal generalized momentum change normalized by its stated SI units. The local derivative follows unique active endpoints at each force evaluation; no continuous-time event correction is included. Derive the sensitivity transport through the source dynamics. Analytic, automatic, and numerical differentiation are acceptable when they resolve the same local derivative to the comparison tolerance.

Returns
-------
One signed finite float: the derivative of the summed normalized generalized momentum norms with respect to log stiffness, evaluated at the supplied positive scale.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_contact_impulse_audit(force_scale: float = 1.0) -> float:
    r"""Return the log-stiffness derivative of the discrete impulse audit.

    Parameters
    ----------
    force_scale : float, optional
        Positive finite multiplier applied to every stated contact stiffness.
        The task instance uses $\lambda=1$. The derivative is evaluated at
        $\eta=\log\lambda$, not at zero for every input scale.

    Returns
    -------
    float
        Signed finite dimensionless derivative $S(\lambda)$. Momentum is
        components normalized by $1\,\mathrm{kg\,m\,s^{-1}}$ for translation
        and $1\,\mathrm{kg\,m^2\,s^{-1}}$ for rotation. Use the Euclidean
        norm of these two dimensionless components in each experiment. Differentiate both kicks
        and the drift, including the force dependence on the evolving state.
        Use the current matrix contract of compute_contact_energy_force;
        its state Jacobian is required. Numerical differentiation is allowed
        if it resolves this same local derivative to the test tolerance.

    Raises
    ------
    ValueError
        If ``force_scale`` is non-positive or non-finite, or a terminal
        generalized momentum change is exactly zero (undefined norm derivative).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _axis_fibers(box, n_horizontal, n_vertical):
    """Construct midpoint horizontal fibers followed by vertical fibers."""
    x_min, x_max, y_min, y_max = np.asarray(box, dtype=np.float64)
    y_values = y_min + (np.arange(n_horizontal) + 0.5) * (
        y_max - y_min
    ) / n_horizontal
    x_values = x_min + (np.arange(n_vertical) + 0.5) * (
        x_max - x_min
    ) / n_vertical
    starts = np.vstack(
        [
            np.column_stack([np.full(n_horizontal, x_min), y_values]),
            np.column_stack([x_values, np.full(n_vertical, y_min)]),
        ]
    )
    ends = np.vstack(
        [
            np.column_stack([np.full(n_horizontal, x_max), y_values]),
            np.column_stack([x_values, np.full(n_vertical, y_max)]),
        ]
    )
    weights = np.concatenate(
        [
            np.full(n_horizontal, (y_max - y_min) / (2.0 * n_horizontal)),
            np.full(n_vertical, (x_max - x_min) / (2.0 * n_vertical)),
        ]
    )
    return starts, ends, weights


def _experiment_data():
    """Return the three immutable experiment dictionaries."""
    return [
        {
            "box": (-1.4, 1.4, -1.1, 1.1),
            "counts": (64, 64),
            "center1": (-0.22, 0.0),
            "axes1": (0.82, 0.55),
            "angle1": 0.20,
            "center2": (0.38, 0.07),
            "axes2": (0.70, 0.46),
            "angle2": -0.32,
            "direction": (1.0, 0.0),
            "theta": 0.0,
            "velocity": -0.35,
            "mass": 1.7,
            "inertia": 0.18,
            "omega": 0.15,
            "stiffness": 480.0,
            "dt": 0.002,
            "steps": 50,
        },
        {
            "box": (-1.6, 1.6, -1.2, 1.2),
            "counts": (48, 48),
            "center1": (-0.30, -0.04),
            "axes1": (0.94, 0.48),
            "angle1": -0.28,
            "center2": (0.46, 0.12),
            "axes2": (0.63, 0.51),
            "angle2": 0.41,
            "direction": (0.9659258263, 0.2588190451),
            "theta": 0.0,
            "velocity": -0.29,
            "mass": 2.1,
            "inertia": 0.21,
            "omega": -0.12,
            "stiffness": 620.0,
            "dt": 0.0015,
            "steps": 60,
        },
        {
            "box": (-1.2, 1.2, -1.2, 1.2),
            "counts": (80, 80),
            "center1": (-0.18, 0.0),
            "axes1": (0.68, 0.52),
            "angle1": 0.15,
            "center2": (0.69, 0.03),
            "axes2": (0.44, 0.36),
            "angle2": -0.20,
            "direction": (1.0, 0.0),
            "theta": 0.0,
            "velocity": -0.22,
            "mass": 1.3,
            "inertia": 0.095,
            "omega": 0.08,
            "stiffness": 700.0,
            "dt": 0.001,
            "steps": 80,
        },
    ]


def _force_at_state(
    experiment, theta, starts, ends, weights, force_scale, pipeline
):
    """Compose all geometric and energetic steps at one state."""
    (
        field_function,
        root_function,
        interval_function,
        area_function,
        root_derivative_function,
        area_derivative_function,
        energy_force_function,
    ) = pipeline
    fixed_state = np.zeros(2, dtype=np.float64)
    fixed_directions = np.zeros((2, 2), dtype=np.float64)
    rotation_rates = np.zeros(2, dtype=np.float64)
    direction = np.asarray(experiment["direction"], dtype=np.float64)
    moving_state = np.asarray(theta, dtype=np.float64)
    moving_directions = np.vstack([direction, np.zeros(2)])
    moving_rotation_rates = np.array([0.0, 1.0])
    roots1 = root_function(
        starts,
        ends,
        experiment["center1"],
        experiment["axes1"],
        experiment["angle1"],
        fixed_state,
        fixed_directions,
        rotation_rates,
    )
    roots2 = root_function(
        starts,
        ends,
        experiment["center2"],
        experiment["axes2"],
        experiment["angle2"],
        moving_state,
        moving_directions,
        moving_rotation_rates,
    )
    intervals = interval_function(np.stack([roots1, roots2]))
    overlap = area_function(intervals, starts, ends, weights)

    finite1 = np.isfinite(roots1)
    finite2 = np.isfinite(roots2)
    segments = ends - starts
    points1 = starts[:, None, :] + roots1[:, :, None] * segments[:, None, :]
    points2 = starts[:, None, :] + roots2[:, :, None] * segments[:, None, :]
    values1 = field_function(
        points1[finite1],
        experiment["center1"],
        experiment["axes1"],
        experiment["angle1"],
        fixed_state,
        fixed_directions,
        rotation_rates,
    )[:, 0]
    values2 = field_function(
        points2[finite2],
        experiment["center2"],
        experiment["axes2"],
        experiment["angle2"],
        moving_state,
        moving_directions,
        moving_rotation_rates,
    )[:, 0]
    if np.any(np.abs(values1) > 1e-9) or np.any(np.abs(values2) > 1e-9):
        raise ValueError("computed roots do not satisfy the body fields")

    derivatives1 = root_derivative_function(
        starts,
        ends,
        roots1,
        experiment["center1"],
        experiment["axes1"],
        experiment["angle1"],
        fixed_state,
        fixed_directions,
        rotation_rates,
    )
    derivatives2 = root_derivative_function(
        starts,
        ends,
        roots2,
        experiment["center2"],
        experiment["axes2"],
        experiment["angle2"],
        moving_state,
        moving_directions,
        moving_rotation_rates,
    )
    overlap_sensitivities = area_derivative_function(
        intervals, np.stack([derivatives1, derivatives2]), starts, ends, weights
    )
    energy_force = np.asarray(energy_force_function(
        overlap, overlap_sensitivities,
        force_scale * experiment["stiffness"], 1.0,
    ), dtype=np.float64)
    if energy_force.shape != (3, 3) or not np.all(np.isfinite(energy_force)):
        raise ValueError("energy-force result must have shape (3, 3)")
    return energy_force[0, 1:].copy(), energy_force[1:, 1:].copy()


def _oracle_run_contact_impulse_audit(force_scale: float = 1.0) -> float:
    """Reference end-to-end three-experiment computation."""
    if not np.isfinite(force_scale) or force_scale <= 0.0:
        raise ValueError("force_scale must be positive and finite")
    pipeline = (
        _oracle_evaluate_ellipse_field,  # noqa: F821
        _oracle_compute_fiber_roots,  # noqa: F821
        _oracle_intersect_inside_intervals,  # noqa: F821
        _oracle_estimate_overlap_area,  # noqa: F821
        _oracle_differentiate_boundary_roots,  # noqa: F821
        _oracle_differentiate_overlap_area,  # noqa: F821
        _oracle_compute_contact_energy_force,  # noqa: F821
    )
    total_sensitivity = 0.0
    for experiment in _experiment_data():
        starts, ends, weights = _axis_fibers(
            experiment["box"], *experiment["counts"]
        )
        theta = np.array([experiment["theta"], 0.0], dtype=np.float64)
        initial_velocity = np.array([experiment["velocity"], experiment["omega"]], dtype=np.float64)
        velocity = initial_velocity.copy()
        position_tangent = np.zeros(2)
        velocity_tangent = np.zeros(2)
        mass = np.array([experiment["mass"], experiment["inertia"]])
        time_step = float(experiment["dt"])
        for _ in range(int(experiment["steps"])):
            old_force, old_jacobian = _force_at_state(
                experiment,
                theta,
                starts,
                ends,
                weights,
                float(force_scale),
                pipeline,
            )
            half_velocity_tangent = velocity_tangent + 0.5 * time_step * (
                old_force + old_jacobian @ position_tangent
            ) / mass
            half_velocity = velocity + 0.5 * time_step * old_force / mass
            position_tangent += time_step * half_velocity_tangent
            theta = theta + time_step * half_velocity
            new_force, new_jacobian = _force_at_state(
                experiment,
                theta,
                starts,
                ends,
                weights,
                float(force_scale),
                pipeline,
            )
            velocity_tangent = half_velocity_tangent + 0.5 * time_step * (
                new_force + new_jacobian @ position_tangent
            ) / mass
            velocity = half_velocity + 0.5 * time_step * new_force / mass
        momentum = mass * (velocity - initial_velocity)
        momentum_norm = np.linalg.norm(momentum)
        if momentum_norm == 0.0:
            raise ValueError("absolute impulse is not differentiable at zero")
        total_sensitivity += float(momentum @ (mass * velocity_tangent) / momentum_norm)
    if not np.isfinite(total_sensitivity):
        raise ValueError("audit scalar must be finite")
    return float(total_sensitivity)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three consequential integrations and one invalid case."""
    return [
        {
            "setup": "force_scale = 1.0",
            "call": "run_contact_impulse_audit(force_scale)",
            "gold_call": "_oracle_run_contact_impulse_audit(force_scale)",
        },
        {
            "setup": "force_scale = 0.8",
            "call": "run_contact_impulse_audit(force_scale)",
            "gold_call": "_oracle_run_contact_impulse_audit(force_scale)",
        },
        {
            "setup": "force_scale = 1.731",
            "call": "run_contact_impulse_audit(force_scale)",
            "gold_call": "_oracle_run_contact_impulse_audit(force_scale)",
        },
        {
            "setup": """
force_scale = 0.0
def run_model():
    try:
        run_contact_impulse_audit(force_scale)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_contact_impulse_audit(force_scale)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
