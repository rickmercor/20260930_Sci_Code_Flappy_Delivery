"""
Run the complete adaptive pricing experiment and return the option value. Every free choice is fixed by the inputs: the contour shift, the refinement index, the two error prefactors, and how many quadrature nodes each refinement level receives.

The complete pricer fits separate algebraic quadrature-error covers for the level-zero integrand and the first correction, propagates the correction cover to finer levels, and allocates node counts by minimizing modeled work. The final value telescopes the coarse price and all level differences, evaluating the fine and coarse terms of each correction at identical nodes and weights so quadrature errors cancel as intended.

Returns
-------
float The option value V.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def hierarchical_fourier_price(model, market, tolerance, base_steps,
                               level_zero_index, correction_index,
                               cost_exponent, pilot_order, reference_order):
    """Return the option value produced by the prescribed level hierarchy.

    Split ``tolerance`` evenly between time bias and quadrature error.  Reuse
    the earlier public functions to select the decay rate, contour shift, and
    refinement depth.  Estimate the level-zero and first-correction algebraic
    error covers on the fixed pilot orders ``(2, 4, 6, 8, 10, 12, 14, 16)``
    against ``reference_order``.  Propagate the correction cover through the
    remaining levels with the prescribed time-step scaling.

    Allocate integer quadrature orders with the source's work-minimizing rule,
    using ``level_zero_index``, ``correction_index``, and ``cost_exponent``.
    Assemble the resulting telescoping price with a shared quadrature rule for
    both terms of each level difference.  Use the supplied parameters and the
    preceding public functions for every intermediate choice.

    Parameters
    ----------
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    market : sequence of float
        Four finite numbers ``(spot, strike, rate, horizon)`` with
        ``spot > 0``, ``strike > 0`` and ``horizon > 0``.
    tolerance : float
        Strictly positive total error budget.
    base_steps : int
        Strictly positive number of time intervals at level zero.
    level_zero_index : float
        Strictly positive algebraic index of the level-zero quadrature error.
    correction_index : float
        Strictly positive algebraic index shared by the level corrections.
    cost_exponent : float
        Strictly positive exponent of the per-level cost weights.
    pilot_order : int
        Strictly positive number of nodes used to measure the first bias gap.
    reference_order : int
        Strictly positive number of nodes used as the quadrature reference in
        the two prefactors.

    Returns
    -------
    float
        Finite call value produced by the complete level hierarchy.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, if ``tolerance``,
        ``level_zero_index``, ``correction_index`` or ``cost_exponent`` is not
        strictly positive, or if ``base_steps``, ``pilot_order`` or
        ``reference_order`` is not a positive integer.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _unpack_model(model):
    """Validate and unpack the six model parameters."""
    np = __import__("numpy")
    arr = np.asarray(model, dtype=float).reshape(-1)
    if arr.size != 6 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("model must hold six finite numbers")
    alpha, gamma, nu, rho, v0, theta = (float(v) for v in arr)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("model[0] must satisfy 0 < alpha <= 1")
    if gamma <= 0.0 or nu <= 0.0:
        raise ValueError("model[1] and model[2] must be positive")
    if not -1.0 < rho < 1.0:
        raise ValueError("model[3] must satisfy abs(rho) < 1")
    if v0 <= 0.0 or theta <= 0.0:
        raise ValueError("model[4] and model[5] must be positive")
    return alpha, gamma, nu, rho, v0, theta


def _unpack_market(market):
    """Validate and unpack the four market parameters."""
    np = __import__("numpy")
    arr = np.asarray(market, dtype=float).reshape(-1)
    if arr.size != 4 or not bool(np.all(np.isfinite(arr))):
        raise ValueError("market must hold four finite numbers")
    spot, strike, rate, horizon = (float(v) for v in arr)
    if spot <= 0.0 or strike <= 0.0:
        raise ValueError("market[0] and market[1] must be positive")
    if horizon <= 0.0:
        raise ValueError("market[3] must be positive")
    return spot, strike, rate, horizon


def _positive_int(value, name):
    """Return ``value`` as a positive Python int or raise ValueError."""
    np = __import__("numpy")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be a positive integer")
    if int(value) < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _line_argument(xi_real, xi_imag):
    """Return the flattened complex vector xi_real + 1j*xi_imag."""
    np = __import__("numpy")
    a = np.atleast_1d(np.asarray(xi_real, dtype=float)).reshape(-1)
    b = np.atleast_1d(np.asarray(xi_imag, dtype=float)).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("xi_real and xi_imag must have the same shape")
    if not (bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b)))):
        raise ValueError("xi_real and xi_imag must be finite")
    return a + 1j * b
def _payoff_values(xi, strike):
    """Return -strike**(1 - 1j*xi)/(xi**2 + 1j*xi) without an out-of-range power."""
    np = __import__("numpy")
    math = __import__("math")
    denom = xi * xi + 1j * xi
    if bool(np.any(denom == 0.0)):
        raise ValueError("xi must avoid the zeros of xi**2 + 1j*xi")
    return -np.exp((1.0 - 1j * xi) * math.log(strike)) / denom
def _riccati_rhs(xi, h, gamma, nu, rho):
    """Return F(xi, h) = -(xi**2 + 1j*xi)/2 + gamma*(1j*xi*rho*nu - 1)*h
    + (gamma*nu)**2 * h**2 / 2."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gamma * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gamma * nu) ** 2 * h * h)


def _nodal_solution(xi, horizon, steps, alpha, gamma, nu, rho):
    """Return the (steps+1, xi.size) array of nodal values of the integral equation."""
    np = __import__("numpy")
    math = __import__("math")
    dt = horizon / steps
    grid = np.arange(steps + 1, dtype=float)
    powered = grid ** alpha
    b_weights = powered[1:] - powered[:-1]
    powered1 = grid ** (alpha + 1.0)
    extended = np.arange(steps + 2, dtype=float) ** (alpha + 1.0)
    c_weights = extended[2:] + powered1[:-1] - 2.0 * powered1[1:]
    pref_predict = dt ** alpha / math.gamma(alpha + 1.0)
    pref_correct = dt ** alpha / math.gamma(alpha + 2.0)

    h = np.zeros((steps + 1, xi.size), dtype=complex)
    f = np.zeros((steps + 1, xi.size), dtype=complex)
    f[0] = _riccati_rhs(xi, 0.0 + 0.0j, gamma, nu, rho)
    for k in range(1, steps + 1):
        predicted = pref_predict * (b_weights[:k][::-1] @ f[:k])
        head = (k - 1.0) ** (alpha + 1.0) - (k - alpha - 1.0) * (float(k) ** alpha)
        acc = head * f[0]
        if k > 1:
            acc = acc + c_weights[:k - 1][::-1] @ f[1:k]
        acc = acc + _riccati_rhs(xi, predicted, gamma, nu, rho)
        h[k] = pref_correct * acc
        f[k] = _riccati_rhs(xi, h[k], gamma, nu, rho)
    return h
def _exponent_values(xi, h_nodes, horizon, spot, rate, gamma, nu, rho, v0, theta):
    """Return the fully discrete log-characteristic function on the line."""
    np = __import__("numpy")
    math = __import__("math")
    steps = h_nodes.shape[0] - 1
    dt = horizon / steps
    integrand = (theta * gamma * h_nodes
                 + v0 * _riccati_rhs(xi[None, :], h_nodes, gamma, nu, rho))
    total = dt * (0.5 * integrand[0]
                  + integrand[1:-1].sum(axis=0)
                  + 0.5 * integrand[-1])
    return 1j * xi * (math.log(spot) + rate * horizon) + total
def _decay_rate(alpha, gamma, nu, rho, v0, theta, horizon):
    """Return sqrt(1-rho**2)/(gamma*nu)*(gamma*theta*T + v0*T**(1-alpha)/Gamma(2-alpha))."""
    math = __import__("math")
    return (math.sqrt(1.0 - rho * rho) / (gamma * nu)) * (
        gamma * theta * horizon
        + v0 * horizon ** (1.0 - alpha) / math.gamma(2.0 - alpha)
    )
def _weight_rule(order, scale):
    """Return nodes and weights exact to degree 2*order-1 for exp(-scale*u) on (0, inf)."""
    special = __import__("scipy.special", fromlist=["roots_laguerre"])
    abscissas, coefficients = special.roots_laguerre(order)
    return abscissas / scale, coefficients / scale
def _level_value(steps, damping, nodes, weights, scale, model, market):
    """Return 2*sum_n weights_n*exp(scale*nodes_n)*g(nodes_n) at resolution ``steps``."""
    np = __import__("numpy")
    nodes = np.asarray(nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    values = _oracle_damped_fourier_integrand(nodes, damping, steps, model, market)
    return 2.0 * float(np.sum(weights * np.exp(scale * nodes) * values))
def _damping_objective(value, steps, model, market):
    """Return abs(g(0; value)) at resolution ``steps``, or infinity if it overflows."""
    math = __import__("math")
    out = abs(float(_oracle_damped_fourier_integrand(0.0, value, steps,
                                                     model, market)[0]))
    return out if math.isfinite(out) else float("inf")


def _minimise_damping(model, market, steps):
    """Return the rounded minimiser of abs(g(0; R)) over R in [-40, -1.5]."""
    math = __import__("math")
    _unpack_model(model)
    _unpack_market(market)
    steps = _positive_int(steps, "steps")
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    lower, upper = -40.0, -1.5

    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    f_left = _damping_objective(left, steps, model, market)
    f_right = _damping_objective(right, steps, model, market)
    for _ in range(200):
        if f_left < f_right:
            upper, right, f_right = right, left, f_left
            left = upper - ratio * (upper - lower)
            f_left = _damping_objective(left, steps, model, market)
        else:
            lower, left, f_left = left, right, f_right
            right = lower + ratio * (upper - lower)
            f_right = _damping_objective(right, steps, model, market)
        if abs(upper - lower) < 1e-12:
            break
    return round(0.5 * (lower + upper), 3)
def _level_count(model, market, base_steps, damping, scale,
                 convergence_order, eps_disc, pilot_order):
    """Return the level index L from the first level difference."""
    math = __import__("math")
    base_steps = _positive_int(base_steps, "base_steps")
    pilot_order = _positive_int(pilot_order, "pilot_order")
    if not float(scale) > 0.0:
        raise ValueError("scale must be positive")
    if not float(eps_disc) > 0.0:
        raise ValueError("eps_disc must be positive")
    if not float(convergence_order) > 0.0:
        raise ValueError("convergence_order must be positive")
    nodes, weights = _weight_rule(pilot_order, float(scale))
    coarse = _level_value(base_steps, damping, nodes, weights,
                          float(scale), model, market)
    fine = _level_value(2 * base_steps, damping, nodes, weights,
                        float(scale), model, market)
    gap = abs(fine - coarse)
    if gap <= 0.0:
        return 1
    p = float(convergence_order)
    raw = 1.0 + (1.0 / p) * math.log2(gap / ((2.0 ** p - 1.0) * float(eps_disc)))
    return max(1, int(math.ceil(raw)))


def _fit_orders():
    """Return the fixed pilot grid of quadrature orders used for the covers."""
    return (2, 4, 6, 8, 10, 12, 14, 16)


def _oracle_hierarchical_fourier_price(model, market, tolerance, base_steps,
                                       level_zero_index, correction_index,
                                       cost_exponent, pilot_order,
                                       reference_order):
    """Reference implementation of the full adaptive pipeline."""
    np = __import__("numpy")
    math = __import__("math")

    alpha, gamma, nu, rho, v0, theta = _unpack_model(model)
    spot, strike, rate, horizon = _unpack_market(market)
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    s0 = float(level_zero_index)
    s = float(correction_index)
    beta = float(cost_exponent)
    if not (s0 > 0.0 and s > 0.0 and beta > 0.0):
        raise ValueError("the algebraic indices and the cost exponent must be positive")
    base_steps = _positive_int(base_steps, "base_steps")
    pilot_order = _positive_int(pilot_order, "pilot_order")
    reference_order = _positive_int(reference_order, "reference_order")

    eps_disc = 0.5 * tolerance
    eps_quad = 0.5 * tolerance
    p = 1.0 + alpha

    scale = _oracle_integrand_decay_rate(model, horizon)
    damping = _oracle_select_damping_parameter(model, market, base_steps)
    levels = _oracle_select_discretization_level(model, market, base_steps, damping,
                                                 scale, p, eps_disc, pilot_order)

    def _weighted_sum(nodes, weights, steps):
        """Return 2*sum_n weights_n*exp(scale*nodes_n)*g(nodes_n) at ``steps``."""
        values = _oracle_damped_fourier_integrand(nodes, damping, steps,
                                                  model, market)
        return 2.0 * float(np.sum(weights * np.exp(scale * nodes) * values))

    def _quadrature(order, level):
        """Return Q(order, level) on the level-``level`` time grid."""
        nodes, weights = _oracle_exponential_weight_rule(order, scale)
        return _weighted_sum(nodes, weights, base_steps * 2 ** level)

    reference_zero = _quadrature(reference_order, 0)
    reference_gap = _quadrature(reference_order, 1) - reference_zero
    prefactor_zero = max(
        abs(reference_zero - _quadrature(order, 0)) * float(order) ** (0.5 * s0)
        for order in _fit_orders()
    )
    prefactor_gap = max(
        abs(reference_gap - (_quadrature(order, 1) - _quadrature(order, 0)))
        * float(order) ** (0.5 * s)
        for order in _fit_orders()
    )

    step = [horizon / (base_steps * 2 ** level) for level in range(levels + 1)]
    propagated = {level: prefactor_gap * (step[level] / step[1]) ** p
                  for level in range(1, levels + 1)}
    cost = {level: step[level] ** (-beta) for level in range(levels + 1)}

    total = sum(propagated[k] ** (2.0 / (s + 2.0))
                * (cost[k] + cost[k - 1]) ** (s / (s + 2.0))
                for k in range(1, levels + 1))
    count_zero = max(1, int(math.ceil((2.0 * prefactor_zero / eps_quad) ** (2.0 / s0))))
    counts = {
        level: max(1, int(math.ceil(
            (propagated[level] / (cost[level] + cost[level - 1])) ** (2.0 / (s + 2.0))
            * ((2.0 / eps_quad) * total) ** (2.0 / s)
        )))
        for level in range(1, levels + 1)
    }

    nodes, weights = _oracle_exponential_weight_rule(count_zero, scale)
    price = _weighted_sum(nodes, weights, base_steps)
    for level in range(1, levels + 1):
        nodes, weights = _oracle_exponential_weight_rule(counts[level], scale)
        fine = _weighted_sum(nodes, weights, base_steps * 2 ** level)
        coarse = _weighted_sum(nodes, weights, base_steps * 2 ** (level - 1))
        price += fine - coarse
    return float(price)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the target instance, a loose budget, a second market, and an
    invalid tolerance."""
    euros = (
        "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
        "market = [1000.0, 1000.0, 0.0, 2.0]\n"
    )
    return [
        {
            "setup": euros,
            "call": "round(float(hierarchical_fourier_price(model, market, 8.0e-4, 32, 8.0, 7.0, 2.0, 16, 64)), 9)",
            "gold_call": "round(float(_oracle_hierarchical_fourier_price(model, market, 8.0e-4, 32, 8.0, 7.0, 2.0, 16, 64)), 9)",
        },
        {
            "setup": euros,
            "call": "round(float(hierarchical_fourier_price(model, market, 4.0e-3, 32, 8.0, 7.0, 2.0, 16, 64)), 9)",
            "gold_call": "round(float(_oracle_hierarchical_fourier_price(model, market, 4.0e-3, 32, 8.0, 7.0, 2.0, 16, 64)), 9)",
        },
        {
            "setup": euros,
            "call": "round(float(hierarchical_fourier_price(model, market, 8.0e-4, 32, 6.0, 5.0, 2.0, 8, 48)), 9)",
            "gold_call": "round(float(_oracle_hierarchical_fourier_price(model, market, 8.0e-4, 32, 6.0, 5.0, 2.0, 8, 48)), 9)",
        },
        {
            "setup": (
                "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
                "market = [100.0, 90.0, 0.03, 0.75]\n"
            ),
            "call": "round(float(hierarchical_fourier_price(model, market, 2.0e-3, 16, 8.0, 7.0, 2.0, 16, 64)), 9)",
            "gold_call": "round(float(_oracle_hierarchical_fourier_price(model, market, 2.0e-3, 16, 8.0, 7.0, 2.0, 16, 64)), 9)",
        },
        {
            "setup": (
                euros
                + "def run_model():\n"
                "    try:\n"
                "        hierarchical_fourier_price(model, market, 0.0, 32, 8.0, 7.0, 2.0, 16, 64)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_hierarchical_fourier_price(model, market, 0.0, 32, 8.0, 7.0, 2.0, 16, 64)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
