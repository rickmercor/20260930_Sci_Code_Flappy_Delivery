"""
Assemble the real integrand whose integral over the positive half line prices a European call, with the contour shifted into the lower half plane. The integrand is built from the discrete pieces already available, so its value depends on the resolution of the time grid used to obtain them.

The damped half-line integrand multiplies the fully discrete characteristic function by the generalized payoff transform and the discount factor, then takes the real part required by Fourier inversion. Each evaluation remains tied to a specified time-grid resolution because its characteristic exponent is obtained from that level’s Riccati-Volterra nodes.

Returns
-------
numpy.ndarray Real array of shape (n,) holding g(u).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def damped_fourier_integrand(u, damping, steps, model, market):
    """Return the real damped Fourier integrand at a given time resolution.

    With ``rate = market[2]``, ``horizon = market[3]`` and
    ``xi = u + 1j * damping``, the integrand is

        g(u) = exp(-rate * horizon) / (2 * pi)
               * real( exp(G(xi)) * P(xi) ),

    where ``P`` is the European-call payoff transform for strike ``market[1]``
    and ``G`` is the discrete log-characteristic function assembled from the
    nodal values of the variance equation on a uniform grid of ``steps``
    intervals over ``[0, horizon]``.

    Implementation requirement: obtain ``P``, the nodal values and ``G`` by
    calling the previously defined public functions rather than reproducing
    their formulas here.

    Parameters
    ----------
    u : array_like
        Real evaluation points, interpreted as a one-dimensional sequence of
        length ``n`` with finite entries.
    damping : float
        Imaginary part of the contour; must satisfy ``damping < -1``.
    steps : int
        Strictly positive number of uniform time intervals.
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.
    market : sequence of float
        Four finite numbers ``(spot, strike, rate, horizon)`` with
        ``spot > 0``, ``strike > 0`` and ``horizon > 0``.

    Returns
    -------
    numpy.ndarray
        Real array of shape ``(n,)`` holding ``g(u)``.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, if ``damping`` is
        not strictly below ``-1``, if ``steps`` is not a positive integer, or if
        ``u`` holds a non-finite entry.
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
def _oracle_damped_fourier_integrand(u, damping, steps, model, market):
    """Reference implementation of the damped integrand, assembled from the
    step 1 to step 3 oracles rather than from private copies of them."""
    np = __import__("numpy")
    math = __import__("math")
    _unpack_model(model)
    spot, strike, rate, horizon = _unpack_market(market)
    steps = _positive_int(steps, "steps")
    damping = float(damping)
    if not damping < -1.0:
        raise ValueError("damping must satisfy damping < -1")
    grid = np.atleast_1d(np.asarray(u, dtype=float)).reshape(-1)
    if not bool(np.all(np.isfinite(grid))):
        raise ValueError("u must be finite")
    shift = np.full(grid.shape, damping)
    h_nodes = _oracle_riccati_volterra_nodes(grid, shift, model, horizon, steps)
    exponent = _oracle_characteristic_exponent(grid, shift, h_nodes, model, market)
    payoff = _oracle_payoff_fourier_transform(grid, shift, strike)
    return (math.exp(-rate * horizon) / (2.0 * math.pi)
            * (np.exp(exponent) * payoff).real)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return coarse, refined, alternative-market, and invalid-damping cases."""
    flat = (
        "def flat(x):\n"
        "    return np.round(np.asarray(x, dtype=float).reshape(-1), 9).tolist()\n"
    )
    euros = (
        "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
        "market = [1000.0, 1000.0, 0.0, 2.0]\n"
    )
    return [
        {
            "setup": flat + euros + "u = [0.0, 1.0, 2.0, 5.0]\n",
            "call": "flat(damped_fourier_integrand(u, -4.673, 32, model, market))",
            "gold_call": "flat(_oracle_damped_fourier_integrand(u, -4.673, 32, model, market))",
        },
        {
            "setup": flat + euros + "u = [0.0, 2.0, 7.25]\n",
            "call": "flat(damped_fourier_integrand(u, -4.673, 512, model, market))",
            "gold_call": "flat(_oracle_damped_fourier_integrand(u, -4.673, 512, model, market))",
        },
        {
            "setup": flat + (
                "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
                "market = [100.0, 90.0, 0.03, 0.75]\n"
                "u = np.linspace(0.0, 6.0, 7)\n"
            ),
            "call": "flat(damped_fourier_integrand(u, -3.5, 48, model, market))",
            "gold_call": "flat(_oracle_damped_fourier_integrand(u, -3.5, 48, model, market))",
        },
        {
            "setup": flat + euros + "u = [0.5]\n",
            "call": "flat(damped_fourier_integrand(u, -1.5, 16, model, market))",
            "gold_call": "flat(_oracle_damped_fourier_integrand(u, -1.5, 16, model, market))",
        },
        {
            "setup": euros + (
                "def run_model():\n"
                "    try:\n"
                "        damped_fourier_integrand([1.0], -0.5, 16, model, market)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_damped_fourier_integrand([1.0], -0.5, 16, model, market)\n"
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
