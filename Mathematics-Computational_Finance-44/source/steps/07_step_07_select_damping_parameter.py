"""
Choose the contour shift used by the damped Fourier representation. The shift is not free: it must keep the payoff transform integrable, and among the admissible values one is singled out by the size of the integrand at the origin.

A vertical contour shift makes the payoff transform integrable and controls the magnitude of the Fourier integrand. Because the rough Heston analyticity strip is not available in closed form, the source selects the admissible shift that minimizes the level-zero integrand magnitude at the origin and reports the optimizer at reproducible three-decimal precision.

Returns
-------
float The selected shift, a value in [-40, -1.5] rounded to three decimals.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def select_damping_parameter(model, market, steps):
    """Return the prescribed contour shift at one time resolution.

    Apply the source's constrained selection rule to the damped Fourier
    integrand at the origin using ``steps`` uniform time intervals.  Search the
    closed admissible interval ``[-40, -1.5]`` accurately enough that the
    selected shift is stable when rounded to three decimals.  Reuse the earlier
    public integrand function when evaluating the selection criterion.

    Parameters
    ----------
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    market : sequence of float
        Four finite numbers ``(spot, strike, rate, horizon)`` with
        ``spot > 0``, ``strike > 0`` and ``horizon > 0``.
    steps : int
        Strictly positive number of uniform time intervals.

    Returns
    -------
    float
        Selected shift in ``[-40, -1.5]``, rounded to three decimals.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, or if ``steps`` is
        not a positive integer.
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


def _oracle_select_damping_parameter(model, market, steps):
    """Reference implementation of the contour-shift selection."""
    np = __import__("numpy")
    _unpack_model(model)
    _unpack_market(market)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise ValueError("steps must be a positive integer")
    if int(steps) < 1:
        raise ValueError("steps must be a positive integer")
    return float(_minimise_damping(model, market, int(steps)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return four maturities and an invalid resolution."""
    euros = "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
    return [
        {
            "setup": euros + "market = [1000.0, 1000.0, 0.0, 2.0]\n",
            "call": "round(float(select_damping_parameter(model, market, 32)), 6)",
            "gold_call": "round(float(_oracle_select_damping_parameter(model, market, 32)), 6)",
        },
        {
            "setup": euros + "market = [1000.0, 1000.0, 0.0, 1.0]\n",
            "call": "round(float(select_damping_parameter(model, market, 32)), 6)",
            "gold_call": "round(float(_oracle_select_damping_parameter(model, market, 32)), 6)",
        },
        {
            "setup": euros + "market = [1000.0, 1000.0, 0.0, 10.0]\n",
            "call": "round(float(select_damping_parameter(model, market, 32)), 6)",
            "gold_call": "round(float(_oracle_select_damping_parameter(model, market, 32)), 6)",
        },
        {
            "setup": (
                "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
                "market = [100.0, 90.0, 0.03, 0.75]\n"
            ),
            "call": "round(float(select_damping_parameter(model, market, 24)), 6)",
            "gold_call": "round(float(_oracle_select_damping_parameter(model, market, 24)), 6)",
        },
        {
            "setup": (
                euros + "market = [1000.0, 1000.0, 0.0, 2.0]\n"
                "def run_model():\n"
                "    try:\n"
                "        select_damping_parameter(model, market, 0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_select_damping_parameter(model, market, 0)\n"
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
