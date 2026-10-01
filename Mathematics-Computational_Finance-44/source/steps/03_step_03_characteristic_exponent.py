"""
Assemble the fully discrete log-characteristic function of the log-price from nodal values of the variance equation. Only the grid values already available are used; no continuous reconstruction of the underlying trajectory is performed.

The log-price characteristic exponent combines the initial log spot, the risk-free drift, and the variance feedback encoded by the Riccati-Volterra solution. A composite trapezoid over the available nodal values defines the fully discrete exponent at each level; preserving the endpoint at time zero is necessary because the Riccati right-hand side is nonzero there.

Returns
-------
numpy.ndarray Complex array of shape (n,) holding G(xi).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def characteristic_exponent(xi_real, xi_imag, h_nodes, model, market):
    """Return the discrete log-characteristic function on a complex line.

    Write ``xi = xi_real + 1j * xi_imag``, ``alpha = model[0]``,
    ``gamma = model[1]``, ``nu = model[2]``, ``rho = model[3]``,
    ``v0 = model[4]``, ``theta = model[5]``, ``spot = market[0]``,
    ``rate = market[2]``, ``horizon = market[3]``, and

        F(xi, h) = -(xi ** 2 + 1j * xi) / 2
                   + gamma * (1j * xi * rho * nu - 1) * h
                   + (gamma * nu) ** 2 * h ** 2 / 2.

    Let ``steps = h_nodes.shape[0] - 1`` and ``dt = horizon / steps``.  With

        J_j = theta * gamma * h_nodes[j] + v0 * F(xi, h_nodes[j]),

    formed at every grid index ``j = 0, ..., steps``, return

        G(xi) = 1j * xi * (log(spot) + rate * horizon)
                + dt * ( J_0 / 2 + sum_{j=1}^{steps-1} J_j + J_steps / 2 ).

    The index ``j = 0`` contributes even though ``h_nodes[0]`` vanishes.

    Parameters
    ----------
    xi_real : array_like
        Real parts of the evaluation points, one-dimensional of length ``n``.
    xi_imag : array_like
        Imaginary parts, same shape as ``xi_real``.
    h_nodes : numpy.ndarray
        Complex array of shape ``(steps + 1, n)`` holding the nodal values of
        the variance equation on the same uniform grid, with ``steps >= 1``.
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
        Complex array of shape ``(n,)`` holding ``G(xi)``.

    Raises
    ------
    ValueError
        If ``model`` or ``market`` violates a stated bound, if ``xi_real`` and
        ``xi_imag`` disagree in shape or hold a non-finite entry, or if
        ``h_nodes`` is not a two-dimensional array with ``n`` columns and at
        least two rows.
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


def _oracle_characteristic_exponent(xi_real, xi_imag, h_nodes, model, market):
    """Reference implementation of the discrete exponent."""
    np = __import__("numpy")
    alpha, gamma, nu, rho, v0, theta = _unpack_model(model)
    spot, strike, rate, horizon = _unpack_market(market)
    xi = _line_argument(xi_real, xi_imag)
    nodes = np.asarray(h_nodes, dtype=complex)
    if nodes.ndim != 2 or nodes.shape[0] < 2 or nodes.shape[1] != xi.size:
        raise ValueError("h_nodes must have shape (steps + 1, n) with steps >= 1")
    return _exponent_values(xi, nodes, horizon, spot, rate,
                            gamma, nu, rho, v0, theta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return coarse, refined, degenerate-variance, and invalid-shape cases."""
    flat = (
        "def flat(z):\n"
        "    z = np.asarray(z, dtype=complex).reshape(-1)\n"
        "    return np.round(np.concatenate([z.real, z.imag]), 9).tolist()\n"
    )
    euros = (
        "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
        "market = [1000.0, 1000.0, 0.0, 2.0]\n"
    )
    return [
        {
            "setup": flat + euros + (
                "xr = np.array([2.0])\nxi = np.array([-4.673])\n"
                "h = _nodal_solution(xr + 1j * xi, market[3], 32, model[0], model[1], model[2], model[3])\n"
            ),
            "call": "flat(characteristic_exponent(xr, xi, h, model, market))",
            "gold_call": "flat(_oracle_characteristic_exponent(xr, xi, h, model, market))",
        },
        {
            "setup": flat + euros + (
                "xr = np.array([0.0, 1.0, 6.5])\nxi = np.full(3, -4.673)\n"
                "h = _nodal_solution(xr + 1j * xi, market[3], 64, model[0], model[1], model[2], model[3])\n"
            ),
            "call": "flat(characteristic_exponent(xr, xi, h, model, market))",
            "gold_call": "flat(_oracle_characteristic_exponent(xr, xi, h, model, market))",
        },
        {
            "setup": flat + (
                "model = [0.62, 0.1, 1e-12, -0.681, 0.0392, 0.0392]\n"
                "market = [100.0, 90.0, 0.03, 0.75]\n"
                "xr = np.array([0.5, 4.0])\nxi = np.full(2, -3.25)\n"
                "h = _nodal_solution(xr + 1j * xi, market[3], 24, model[0], model[1], model[2], model[3])\n"
            ),
            "call": "flat(characteristic_exponent(xr, xi, h, model, market))",
            "gold_call": "flat(_oracle_characteristic_exponent(xr, xi, h, model, market))",
        },
        {
            "setup": flat + euros + (
                "xr = np.array([3.0])\nxi = np.array([-2.0])\n"
                "h = _nodal_solution(xr + 1j * xi, market[3], 1, model[0], model[1], model[2], model[3])\n"
            ),
            "call": "flat(characteristic_exponent(xr, xi, h, model, market))",
            "gold_call": "flat(_oracle_characteristic_exponent(xr, xi, h, model, market))",
        },
        {
            "setup": euros + (
                "bad = np.zeros((5, 3), dtype=complex)\n"
                "def run_model():\n"
                "    try:\n"
                "        characteristic_exponent([1.0], [-2.0], bad, model, market)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_characteristic_exponent([1.0], [-2.0], bad, model, market)\n"
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
