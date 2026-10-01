"""
Compose every earlier step to obtain the largest Lyapunov exponent of the long-time collective state of a heterogeneous gene population whose chromatin marks have settled.

Orchestrator: It locates each class's equilibria (solve_epigenetic_equilibria) and settles its starting marks into constant inputs with population shares (settle_starting_marks); brackets and solves the time-independent self-consistency (average_correlation_overlap, solve_frozen_correlation) and tests its stability with the slope overlap (average_gain_overlap); when that state is unstable, locates the decaying autocorrelation (locate_correlation_hilltop, average_potential_overlap, solve_decaying_correlation), traces it in lag time (trace_decay_trajectory) and converts the fluctuation potential along it into the exponent (solve_fluctuation_ground_state).

Step scientific background: A settled population either freezes into a time-independent collective state or, when that state is unstable, keeps fluctuating with an autocorrelation that decays onto a static part; the largest Lyapunov exponent follows from the lowest eigenvalue of the fluctuation operator built on whichever state the network occupies.

Returns
-------
float: largest Lyapunov exponent of the settled collective state, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_settled_lyapunov_exponent(
    inputs: tuple = (-0.32, -0.1, 0.04, 0.11, 0.35),
    fractions: tuple = (0.14, 0.22, 0.27, 0.21, 0.16),
    alpha: float = 0.5,
    beta: float = 4.0,
    n_marks: int = 1000,
    mark_span: float = 2.5,
    nodes: int = 241,
    t_max: float = 40.0,
    n_tau: int = 4001,
) -> float:
    """Return the largest Lyapunov exponent of the settled collective state.

    Class ``k`` has external input ``inputs[k]`` and population fraction
    ``fractions[k]``; its marks start at the ``n_marks`` equally spaced values
    from ``-mark_span`` to ``+mark_span`` inclusive and settle onto the
    attractors of the class's slow flow. Each occupied attractor contributes one
    constant input weighted by the class fraction times its basin share, and the
    weights are rescaled to sum to one exactly.

    Select the largest physical time-independent self-consistency root in
    ``[0, 1]`` and test it with the fluctuation potential
    ``W = 1 - average_gain_overlap([d_star], d_star, ...)``. If ``W >= 0`` the
    root is stable and its constant-potential ground energy gives the exponent.
    Otherwise use the earlier solvers to find the nonconstant autocorrelation:
    its long-lag value must independently be a hilltop returned by
    ``locate_correlation_hilltop`` and its effective potential, evaluated with
    ``average_potential_overlap``, must equal that at the release point.

    Trace this physical branch on the nested uniform lag grids containing
    ``n_tau`` and ``2 * n_tau - 1`` points over ``[0, t_max]``. On each grid form
    ``1 - average_gain_overlap`` along the trajectory and solve the tail-matched
    continuous fluctuation problem. Remove the leading second-order error of
    the piecewise-linear potential using ``(4 * E_fine - E_coarse) / 3`` with the
    two nested energies before applying ``lambda = -1 + sqrt(1 - E0)``. The
    defaults reproduce the problem statement. Numerical bracketing and
    interpolation choices must be inferred from the contracts of the earlier
    steps rather than duplicated here.

    Parameters
    ----------
    inputs : tuple
        Non-empty sequence of finite external inputs, one per gene class.
    fractions : tuple
        Positive population fractions of the same length as ``inputs``, summing
        to one within ``1e-9``.
    alpha : float
        Nonnegative finite feedback strength.
    beta : float
        Positive finite response gain.
    n_marks : int
        Number of starting marks per class, at least 2 (booleans are rejected).
    mark_span : float
        Positive finite half-width of the starting mark interval.
    nodes : int
        Number of abscissae per standard-normal variable, at least 3.
    t_max : float
        Positive finite length of the lag-time window.
    n_tau : int
        Number of lag-time grid points of the coarser trajectory, at least 3.

    Returns
    -------
    float
        The largest Lyapunov exponent, a native Python float.

    Raises
    ------
    ValueError
        If ``inputs`` is not a non-empty one-dimensional sequence of finite
        numbers, if ``fractions`` does not match ``inputs`` in length or is not
        positive or does not sum to one within ``1e-9``, if ``n_marks`` or
        ``n_tau`` is too small, if ``mark_span`` or ``t_max`` is not positive and
        finite, if no time-independent state is found on ``[0, 1]``, if an
        unstable time-independent state admits no decaying autocorrelation in its
        bracket, if the lowest eigenvalue exceeds one, or if an argument fails the
        contract of the step that consumes it.
    RuntimeError
        If the continuous fluctuation eigenproblem does not converge to a
        nodeless bound state.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_settled_lyapunov_exponent(
    inputs: tuple = (-0.32, -0.1, 0.04, 0.11, 0.35),
    fractions: tuple = (0.14, 0.22, 0.27, 0.21, 0.16),
    alpha: float = 0.5,
    beta: float = 4.0,
    n_marks: int = 1000,
    mark_span: float = 2.5,
    nodes: int = 241,
    t_max: float = 40.0,
    n_tau: int = 4001,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    classes = _epi_require_vector(inputs, "inputs")
    shares = _epi_require_vector(fractions, "fractions")
    if shares.size != classes.size:
        raise ValueError("inputs and fractions must have the same length")
    if bool(np.any(shares <= 0.0)) or abs(float(np.sum(shares)) - 1.0) > 1e-9:
        raise ValueError("fractions must be positive and sum to one within 1e-9")
    span = _epi_require_scalar(mark_span, "mark_span", 0.0, True)
    horizon = _epi_require_scalar(t_max, "t_max", 0.0, True)
    marks = _epi_require_count(n_marks, "n_marks", 2)
    coarse = _epi_require_count(n_tau, "n_tau", 3)

    starting = np.linspace(-span, span, marks)
    fields, weights = [], []
    for external, share in zip(classes, shares):
        table = _oracle_solve_epigenetic_equilibria(alpha, beta, float(external))
        for value, basin in _oracle_settle_starting_marks(starting, table, float(external)):
            fields.append(float(value))
            weights.append(float(share) * float(basin))
    fields = np.asarray(fields, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / float(np.sum(weights))  # removes accumulated rounding only

    grid = np.linspace(0.0, 1.0, 41)
    residual = np.array([float(_oracle_average_correlation_overlap(
        np.array([node]), node, fields, weights, beta, nodes)[0]) - node for node in grid])
    crossings = np.nonzero(residual[:-1] * residual[1:] < 0.0)[0]
    if crossings.size > 0:
        last = int(crossings[-1])
        d_star = _oracle_solve_frozen_correlation(fields, weights, beta, float(grid[last]),
                                                  float(grid[last + 1]), nodes)
    else:
        exact = np.nonzero(residual == 0.0)[0]
        if exact.size == 0:
            raise ValueError("no time-independent state on [0, 1]")
        d_star = float(grid[int(exact[-1])])

    static = 1.0 - float(_oracle_average_gain_overlap(np.array([d_star]), d_star, fields,
                                                      weights, beta, nodes)[0])
    if static >= 0.0:
        # A stable frozen state: constant fluctuation potential, lowest eigenvalue W itself.
        return float(-1.0 + np.sqrt(1.0 - static))

    delta_zero, delta_inf = _oracle_solve_decaying_correlation(
        fields, weights, beta, 0.5 * d_star, d_star * (1.0 - 1e-9), nodes)
    checked_top = _oracle_locate_correlation_hilltop(delta_zero, fields, weights, beta, nodes)
    if not np.isfinite(checked_top) or abs(float(checked_top) - float(delta_inf)) > 5e-10:
        raise ValueError("the decaying solution does not end on the selected hilltop")
    endpoints = np.array([delta_inf, delta_zero])
    endpoint_potential = (_oracle_average_potential_overlap(
        endpoints, delta_zero, fields, weights, beta, nodes) - 0.5 * endpoints ** 2)
    if abs(float(endpoint_potential[1] - endpoint_potential[0])) > 5e-10:
        raise ValueError("the decaying solution does not conserve endpoint energy")
    fluctuation = _epi_chebyshev_fit(
        lambda lags: 1.0 - _oracle_average_gain_overlap(np.atleast_1d(lags), delta_zero, fields,
                                                        weights, beta, nodes),
        delta_inf, delta_zero, 64)
    energies = []
    for points in (coarse, 2 * coarse - 1):
        profile = _oracle_trace_decay_trajectory(delta_zero, delta_inf, fields, weights, beta,
                                                 horizon, points, nodes)
        potential = _epi_chebyshev_eval(fluctuation, profile, delta_inf, delta_zero)
        energies.append(_oracle_solve_fluctuation_ground_state(potential, horizon / (points - 1)))
    ground = (4.0 * energies[1] - energies[0]) / 3.0  # leading O(h^2) interpolation error
    if ground > 1.0:
        raise ValueError("the fluctuation operator admits no real Lyapunov exponent")
    return float(-1.0 + np.sqrt(1.0 - ground))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    header = "import numpy as np\n"
    status = (
        "import numpy as np\n"
        "def _status(action):\n"
        "    try:\n"
        "        action()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        # --- Normal: the problem's population at a coarse resolution, a chaotic settled state ---
        {
            "setup": header,
            "call": "compute_settled_lyapunov_exponent((-0.32, -0.1, 0.04, 0.11, 0.35), (0.14, 0.22, 0.27, 0.21, 0.16), 0.5, 4.0, 200, 2.5, 61, 30.0, 401)",
            "gold_call": "_oracle_compute_settled_lyapunov_exponent((-0.32, -0.1, 0.04, 0.11, 0.35), (0.14, 0.22, 0.27, 0.21, 0.16), 0.5, 4.0, 200, 2.5, 61, 30.0, 401)",
        },
        # --- Edge: feedback switched off, bare inputs and stronger chaos ---
        {
            "setup": header,
            "call": "compute_settled_lyapunov_exponent((-0.32, -0.1, 0.04, 0.11, 0.35), (0.14, 0.22, 0.27, 0.21, 0.16), 0.0, 4.0, 50, 2.5, 61, 25.0, 401)",
            "gold_call": "_oracle_compute_settled_lyapunov_exponent((-0.32, -0.1, 0.04, 0.11, 0.35), (0.14, 0.22, 0.27, 0.21, 0.16), 0.0, 4.0, 50, 2.5, 61, 25.0, 401)",
        },
        # --- Boundary: a strongly saturated population whose frozen state is stable ---
        {
            "setup": header,
            "call": "compute_settled_lyapunov_exponent((-1.45, -0.55, -0.1, 0.35, 1.1), (0.14, 0.22, 0.27, 0.21, 0.16), 1.6, 2.0, 100, 2.5, 81, 20.0, 101)",
            "gold_call": "_oracle_compute_settled_lyapunov_exponent((-1.45, -0.55, -0.1, 0.35, 1.1), (0.14, 0.22, 0.27, 0.21, 0.16), 1.6, 2.0, 100, 2.5, 81, 20.0, 101)",
        },
        # --- Normal: a single bistable class split between its two attractors ---
        {
            "setup": header,
            "call": "compute_settled_lyapunov_exponent((0.04,), (1.0,), 0.5, 4.0, 100, 2.5, 61, 30.0, 401)",
            "gold_call": "_oracle_compute_settled_lyapunov_exponent((0.04,), (1.0,), 0.5, 4.0, 100, 2.5, 61, 30.0, 401)",
        },
        # --- Invalid: population fractions that do not sum to one ---
        {
            "setup": status,
            "call": "_status(lambda: compute_settled_lyapunov_exponent((-0.1, 0.4), (0.5, 0.3), 0.5, 4.0, 16, 2.5, 21, 10.0, 11))",
            "gold_call": "_status(lambda: _oracle_compute_settled_lyapunov_exponent((-0.1, 0.4), (0.5, 0.3), 0.5, 4.0, 16, 2.5, 21, 10.0, 11))",
        },
        # --- Invalid: one fraction per class is missing ---
        {
            "setup": status,
            "call": "_status(lambda: compute_settled_lyapunov_exponent((-0.1, 0.4, 1.0), (0.5, 0.5), 0.5, 4.0, 16, 2.5, 21, 10.0, 11))",
            "gold_call": "_status(lambda: _oracle_compute_settled_lyapunov_exponent((-0.1, 0.4, 1.0), (0.5, 0.5), 0.5, 4.0, 16, 2.5, 21, 10.0, 11))",
        },
        # --- Invalid: a single starting mark cannot span the interval ---
        {
            "setup": status,
            "call": "_status(lambda: compute_settled_lyapunov_exponent((0.04,), (1.0,), 0.5, 4.0, 1, 2.5, 21, 10.0, 11))",
            "gold_call": "_status(lambda: _oracle_compute_settled_lyapunov_exponent((0.04,), (1.0,), 0.5, 4.0, 1, 2.5, 21, 10.0, 11))",
        },
    ]
