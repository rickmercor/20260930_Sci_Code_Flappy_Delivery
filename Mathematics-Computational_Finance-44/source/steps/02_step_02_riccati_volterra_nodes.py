"""
Solve the nonlinear fractional integral equation that carries the variance dynamics, and return its values on a uniform grid. The equation has a weakly singular power-law kernel, so the solution's entire history affects every grid value.

The rough Heston characteristic function is governed by a nonlinear fractional Riccati-Volterra equation. Its weakly singular kernel retains the full solution history, so the fractional Adams predictor-corrector evaluates every new time node from all preceding nodes and produces the distinct discrete trajectories needed by the multilevel hierarchy.

Returns
-------
numpy.ndarray Complex array of shape (steps + 1, n) whose row j holds h(xi, t_j); row 0 is identically zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def riccati_volterra_nodes(xi_real, xi_imag, model, horizon, steps):
    """Return the prescribed discrete variance-equation history.

    Interpret the two coordinate arrays as one complex Fourier line and apply
    the source's history-dependent discretization of the nonlinear fractional
    variance equation on a uniform grid over ``[0, horizon]``.  Preserve the
    complete prior history at every update and evaluate all frequencies in
    complex arithmetic.  The initial row corresponds to time zero and must
    vanish exactly.

    Parameters
    ----------
    xi_real : array_like
        Real parts of the evaluation points, a one-dimensional sequence of
        length ``n`` with finite entries.
    xi_imag : array_like
        Imaginary parts, same shape as ``xi_real``, finite.
    model : sequence of float
        Six finite numbers ``(alpha, gamma, nu, rho, v0, theta)`` with
        ``0 < alpha <= 1``, ``gamma > 0``, ``nu > 0``, ``abs(rho) < 1``,
        ``v0 > 0`` and ``theta > 0``.  Only the first four are used here.
    horizon : float
        Strictly positive end of the time interval.
    steps : int
        Strictly positive number of uniform grid intervals.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(steps + 1, n)`` containing the prescribed
        nodal history in increasing time order.

    Raises
    ------
    ValueError
        If ``model`` violates any stated bound, if ``horizon`` is not finite and
        strictly positive, if ``steps`` is not a positive integer, or if
        ``xi_real`` and ``xi_imag`` disagree in shape or hold a non-finite entry.
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


def _oracle_riccati_volterra_nodes(xi_real, xi_imag, model, horizon, steps):
    """Reference implementation of the nodal solution."""
    np = __import__("numpy")
    alpha, gamma, nu, rho, v0, theta = _unpack_model(model)
    horizon = float(horizon)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    steps = _positive_int(steps, "steps")
    xi = _line_argument(xi_real, xi_imag)
    return _nodal_solution(xi, horizon, steps, alpha, gamma, nu, rho)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return coarse, refined, multi-point, and invalid-step cases."""
    flat = (
        "def flat(z):\n"
        "    z = np.asarray(z, dtype=complex).reshape(-1)\n"
        "    return np.round(np.concatenate([z.real, z.imag]), 9).tolist()\n"
    )
    euros = "model = [0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156]\n"
    return [
        {
            "setup": flat + euros + "xr = [2.0]\nxi = [-4.673]\nhorizon = 2.0\nsteps = 32\n",
            "call": "flat(riccati_volterra_nodes(xr, xi, model, horizon, steps))",
            "gold_call": "flat(_oracle_riccati_volterra_nodes(xr, xi, model, horizon, steps))",
        },
        {
            "setup": flat + euros + "xr = [0.0, 1.0, 6.5]\nxi = [-4.673, -4.673, -4.673]\nhorizon = 2.0\nsteps = 64\n",
            "call": "flat(riccati_volterra_nodes(xr, xi, model, horizon, steps))",
            "gold_call": "flat(_oracle_riccati_volterra_nodes(xr, xi, model, horizon, steps))",
        },
        {
            "setup": flat + "model = [1.0, 0.9, 0.4, 0.25, 0.05, 0.09]\nxr = [3.0, 12.0]\nxi = [-2.5, -2.5]\nhorizon = 0.75\nsteps = 40\n",
            "call": "flat(riccati_volterra_nodes(xr, xi, model, horizon, steps))",
            "gold_call": "flat(_oracle_riccati_volterra_nodes(xr, xi, model, horizon, steps))",
        },
        {
            "setup": flat + euros + "xr = [1.5]\nxi = [-3.0]\nhorizon = 1.0\nsteps = 1\n",
            "call": "flat(riccati_volterra_nodes(xr, xi, model, horizon, steps))",
            "gold_call": "flat(_oracle_riccati_volterra_nodes(xr, xi, model, horizon, steps))",
        },
        {
            "setup": (
                euros
                + "def run_model():\n"
                "    try:\n"
                "        riccati_volterra_nodes([1.0], [-2.0], model, 2.0, 0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_riccati_volterra_nodes([1.0], [-2.0], model, 2.0, 0)\n"
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
