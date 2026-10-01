#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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


def payoff_fourier_transform(xi_real, xi_imag, strike):
    """Reference implementation of the payoff transform."""
    np = __import__("numpy")
    strike = float(strike)
    if not np.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be finite and positive")
    xi = _line_argument(xi_real, xi_imag)
    return _payoff_values(xi, strike)

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


def riccati_volterra_nodes(xi_real, xi_imag, model, horizon, steps):
    """Reference implementation of the nodal solution."""
    np = __import__("numpy")
    alpha, gamma, nu, rho, v0, theta = _unpack_model(model)
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    steps = _positive_int(steps, "steps")
    xi = _line_argument(xi_real, xi_imag)
    return _nodal_solution(xi, horizon, steps, alpha, gamma, nu, rho)

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


def characteristic_exponent(xi_real, xi_imag, h_nodes, model, market):
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
def _decay_rate(alpha, gamma, nu, rho, v0, theta, horizon):
    """Return sqrt(1-rho**2)/(gamma*nu)*(gamma*theta*T + v0*T**(1-alpha)/Gamma(2-alpha))."""
    math = __import__("math")
    return (math.sqrt(1.0 - rho * rho) / (gamma * nu)) * (
        gamma * theta * horizon
        + v0 * horizon ** (1.0 - alpha) / math.gamma(2.0 - alpha)
    )


def integrand_decay_rate(model, horizon):
    """Reference implementation of the decay rate."""
    np = __import__("numpy")
    alpha, gamma, nu, rho, v0, theta = _unpack_model(model)
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    return float(_decay_rate(alpha, gamma, nu, rho, v0, theta, horizon))

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
def _weight_rule(order, scale):
    """Return nodes and weights exact to degree 2*order-1 for exp(-scale*u) on (0, inf)."""
    special = __import__("scipy.special", fromlist=["roots_laguerre"])
    abscissas, coefficients = special.roots_laguerre(order)
    return abscissas / scale, coefficients / scale


def exponential_weight_rule(order, scale):
    """Reference implementation of the scaled quadrature rule."""
    np = __import__("numpy")
    order = _positive_int(order, "order")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    nodes, weights = _weight_rule(order, scale)
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)

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
def damped_fourier_integrand(u, damping, steps, model, market):
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
    h_nodes = riccati_volterra_nodes(grid, shift, model, horizon, steps)
    exponent = characteristic_exponent(grid, shift, h_nodes, model, market)
    payoff = payoff_fourier_transform(grid, shift, strike)
    return (math.exp(-rate * horizon) / (2.0 * math.pi)
            * (np.exp(exponent) * payoff).real)

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
    out = abs(float(damped_fourier_integrand(0.0, value, steps,
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


def select_damping_parameter(model, market, steps):
    """Reference implementation of the contour-shift selection."""
    np = __import__("numpy")
    _unpack_model(model)
    _unpack_market(market)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise ValueError("steps must be a positive integer")
    if int(steps) < 1:
        raise ValueError("steps must be a positive integer")
    return float(_minimise_damping(model, market, int(steps)))

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
    values = damped_fourier_integrand(nodes, damping, steps, model, market)
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


def select_discretization_level(model, market, base_steps, damping, scale,
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
    values = damped_fourier_integrand(nodes, damping, steps, model, market)
    return 2.0 * float(np.sum(weights * np.exp(scale * nodes) * values))
def _damping_objective(value, steps, model, market):
    """Return abs(g(0; value)) at resolution ``steps``, or infinity if it overflows."""
    math = __import__("math")
    out = abs(float(damped_fourier_integrand(0.0, value, steps,
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


def hierarchical_fourier_price(model, market, tolerance, base_steps,
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

    scale = integrand_decay_rate(model, horizon)
    damping = select_damping_parameter(model, market, base_steps)
    levels = select_discretization_level(model, market, base_steps, damping,
                                                 scale, p, eps_disc, pilot_order)

    def _weighted_sum(nodes, weights, steps):
        """Return 2*sum_n weights_n*exp(scale*nodes_n)*g(nodes_n) at ``steps``."""
        values = damped_fourier_integrand(nodes, damping, steps,
                                                  model, market)
        return 2.0 * float(np.sum(weights * np.exp(scale * nodes) * values))

    def _quadrature(order, level):
        """Return Q(order, level) on the level-``level`` time grid."""
        nodes, weights = exponential_weight_rule(order, scale)
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

    nodes, weights = exponential_weight_rule(count_zero, scale)
    price = _weighted_sum(nodes, weights, base_steps)
    for level in range(1, levels + 1):
        nodes, weights = exponential_weight_rule(counts[level], scale)
        fine = _weighted_sum(nodes, weights, base_steps * 2 ** level)
        coarse = _weighted_sum(nodes, weights, base_steps * 2 ** (level - 1))
        price += fine - coarse
    return float(price)
SCICODE_GOLD_EOF
