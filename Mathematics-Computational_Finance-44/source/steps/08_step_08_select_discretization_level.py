"""
Decide how far the time grid has to be refined before the remaining bias sits inside a prescribed budget. The decision is taken from a single measured gap between the two coarsest grids and an assumed algebraic decay of that gap.

The multilevel method estimates the remaining time-discretization bias from the first difference between the two coarsest grids. Combining that measured gap with the empirical price-error rate yields a Richardson-style indicator and the smallest refinement level whose extrapolated bias fits inside the prescribed discretization budget.

Returns
-------
int The refinement index L, at least 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_discretization_level(model, market, base_steps, damping, scale,
                                convergence_order, eps_disc, pilot_order):
    """Return the prescribed refinement index for a bias budget.

    Measure the first gap between the two coarsest time grids with one shared
    ``pilot_order`` exponential-weight quadrature rule.  Apply the source's
    single-gap bias extrapolation using ``convergence_order`` and ``eps_disc``;
    level zero has ``base_steps`` intervals and every subsequent level doubles
    that count.  Return level one when the measured gap vanishes.  Reuse the
    earlier public quadrature and integrand functions.

    Parameters
    ----------
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    market : sequence of float
        Four finite numbers ``(spot, strike, rate, horizon)`` with
        ``spot > 0``, ``strike > 0`` and ``horizon > 0``.
    base_steps : int
        Strictly positive number of time intervals at level zero.
    damping : float
        Contour shift, strictly below ``-1``.
    scale : float
        Strictly positive decay rate of the quadrature weight.
    convergence_order : float
        Strictly positive assumed algebraic order of the bias in the step size.
    eps_disc : float
        Strictly positive bias budget.
    pilot_order : int
        Strictly positive number of quadrature nodes used for the measurement.

    Returns
    -------
    int
        Refinement index, at least one.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, if ``damping`` is
        not strictly below ``-1``, if ``base_steps`` or ``pilot_order`` is not a
        positive integer, or if ``scale``, ``convergence_order`` or ``eps_disc``
        is not strictly positive.
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


def _oracle_select_discretization_level(model, market, base_steps, damping, scale,
                                        convergence_order, eps_disc, pilot_order):
    """Reference implementation of the refinement-index rule."""
    np = __import__("numpy")
    _unpack_model(model)
    _unpack_market(market)
    base_steps = _positive_int(base_steps, "base_steps")
    pilot_order = _positive_int(pilot_order, "pilot_order")
    damping = float(damping)
    if not damping < -1.0:
        raise ValueError("damping must satisfy damping < -1")
    scale = float(scale)
    convergence_order = float(convergence_order)
    eps_disc = float(eps_disc)
    if not (np.isfinite(scale) and scale > 0.0):
        raise ValueError("scale must be finite and positive")
    if not (np.isfinite(convergence_order) and convergence_order > 0.0):
        raise ValueError("convergence_order must be finite and positive")
    if not (np.isfinite(eps_disc) and eps_disc > 0.0):
        raise ValueError("eps_disc must be finite and positive")
    return int(_level_count(model, market, base_steps, damping, scale,
                            convergence_order, eps_disc, pilot_order))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the target setting, a loose budget, a tight budget, a second
    market, and an invalid budget."""
    euros = (
        "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
        "market = [1000.0, 1000.0, 0.0, 2.0]\n"
        "scale = 2.666571209748119\n"
    )
    return [
        {
            "setup": euros,
            "call": "int(select_discretization_level(model, market, 32, -4.673, scale, 1.62, 4.0e-4, 16))",
            "gold_call": "int(_oracle_select_discretization_level(model, market, 32, -4.673, scale, 1.62, 4.0e-4, 16))",
        },
        {
            "setup": euros,
            "call": "int(select_discretization_level(model, market, 32, -4.673, scale, 1.62, 1.0e-2, 16))",
            "gold_call": "int(_oracle_select_discretization_level(model, market, 32, -4.673, scale, 1.62, 1.0e-2, 16))",
        },
        {
            "setup": euros,
            "call": "int(select_discretization_level(model, market, 32, -4.673, scale, 1.62, 2.0e-5, 32))",
            "gold_call": "int(_oracle_select_discretization_level(model, market, 32, -4.673, scale, 1.62, 2.0e-5, 32))",
        },
        {
            "setup": (
                "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
                "market = [100.0, 90.0, 0.03, 0.75]\n"
                "scale = 1.3986135505941166\n"
            ),
            "call": "int(select_discretization_level(model, market, 16, -6.36, scale, 1.62, 1.0e-4, 16))",
            "gold_call": "int(_oracle_select_discretization_level(model, market, 16, -6.36, scale, 1.62, 1.0e-4, 16))",
        },
        {
            "setup": (
                euros
                + "def run_model():\n"
                "    try:\n"
                "        select_discretization_level(model, market, 32, -4.673, scale, 1.62, 0.0, 16)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_select_discretization_level(model, market, 32, -4.673, scale, 1.62, 0.0, 16)\n"
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
