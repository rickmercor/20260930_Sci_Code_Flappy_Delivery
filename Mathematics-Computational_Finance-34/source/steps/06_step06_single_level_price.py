"""
Computes the single-level price V_(N, level): the scaled half-line quadrature rule with n_quad nodes and weight scale sigma applied to the level-level Fourier integrand. This step validates the coupling of quadrature and integrand at one fixed level. Deliberately excluded: level differences, pilot indicators, level selection, and point allocation.

Each hierarchy level defines an integrand whose exact integral is that level's price; the quadrature rule evaluates the integrand at scaled nodes and forms the doubled weighted sum. Because every node evaluation requires one fractional-Riccati solve on the level grid, the cost of V_(N, level) is the number of nodes times the per-solve cost of that level. For the frozen benchmark configuration with sigma = 6.254084328929: V_(16,1) = 19.123804457074, V_(16,0) = 19.118235714038, and with only 4 nodes V_(4,0) = 19.069170156296.

Returns
-------
float — the single-level scaled-quadrature price V_(n_quad, level) (a single scalar).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def single_level_price(level: int, n_quad: int, sigma: float, damp: float,
                       alpha: float, gam: float, nu: float, rho: float,
                       v0: float, theta: float, s0: float, strike: float,
                       r: float, big_t: float, m0: int) -> float:
    """Single-level scaled-quadrature price V_(n_quad, level).

    Parameters
    ----------
    level : int
        Discretization level, level >= 0.
    n_quad : int
        Number of quadrature nodes, n_quad >= 1.
    sigma : float
        Scale of the exponential quadrature weight, sigma > 0.
    damp : float
        Damping parameter, damp < -1.
    alpha, gam, nu, rho, v0, theta : float
        Rough Heston model parameters (admissible ranges as in steps 3-5).
    s0, strike : float
        Spot and strike, both > 0.
    r : float
        Risk-free rate.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.

    Returns
    -------
    float
        V_(n_quad, level).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _s06_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s06_adams(xi, num_steps, dt, alpha, gam, nu, rho):
    """Fractional Adams PECE (Diethelm-Ford-Freed) solve of the Riccati-
    Volterra equation with h(0) = 0 on a uniform grid t_j = j*dt.

    xi : 1-D complex array. Returns complex array of shape (num_steps+1, len(xi)).
    Product-rectangle predictor, product-trapezoidal corrector, one
    corrector pass per step.
    """
    import numpy as np
    from math import gamma as _gamma_fn
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    h = np.zeros((num_steps + 1, xi.shape[0]), dtype=complex)
    fh = np.zeros_like(h)
    fh[0] = _s06_F(xi, h[0], gam, nu, rho)
    k = np.arange(num_steps + 2, dtype=float)
    ka = k ** alpha
    kap = k ** (alpha + 1.0)
    b_all = ka[1:] - ka[:-1]
    c_all = kap[2:] + kap[:-2] - 2.0 * kap[1:-1]
    pre_p = dt ** alpha / _gamma_fn(alpha + 1.0)
    pre_c = dt ** alpha / _gamma_fn(alpha + 2.0)
    for n in range(num_steps):
        hp = pre_p * (b_all[: n + 1, None] * fh[n::-1]).sum(axis=0)
        a0w = kap[n] - (n - alpha) * ka[n + 1]
        hist = a0w * fh[0]
        if n >= 1:
            hist = hist + (c_all[:n, None] * fh[n:0:-1]).sum(axis=0)
        h[n + 1] = pre_c * (_s06_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s06_F(xi, h[n + 1], gam, nu, rho)
    return h


def _s06_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r, big_t, m0):
    """Fully discrete characteristic-function exponent G_level(xi):
    composite trapezoid (half weights at both endpoints, j = 0 included)
    of J(xi, h_j) = theta*gam*h_j + v0*F(xi, h_j) over the nodal Adams
    solution, plus the drift term i*xi*(log(s0) + r*T)."""
    import numpy as np
    from math import log
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    num_steps = m0 * 2 ** level
    dt = big_t / num_steps
    h = _s06_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    jv = theta * gam * h + v0 * _s06_F(xi, h, gam, nu, rho)
    trap = dt * (0.5 * jv[0] + jv[1:-1].sum(axis=0) + 0.5 * jv[-1])
    return 1j * xi * (log(s0) + r * big_t) + trap


def _s06_integrand(u_arr, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                 strike, r, big_t, m0):
    """Half-line Fourier integrand g_level(u) = e^{-rT}/(2 pi) *
    Re[exp(G_level(u + i*damp)) * Phat(u + i*damp)] with the damped call
    payoff transform Phat(xi) = -K^{1-i xi} / (xi^2 + i xi)."""
    import numpy as np
    from math import exp, pi, log
    u_arr = np.atleast_1d(np.asarray(u_arr, dtype=float))
    xi = u_arr + 1j * damp
    gv = _s06_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r,
                     big_t, m0)
    phat = -np.exp((1.0 - 1j * xi) * log(strike)) / (xi * xi + 1j * xi)
    return exp(-r * big_t) / (2.0 * pi) * (np.exp(gv) * phat).real


def _s06_lag_nodes(n_quad, sigma):
    """Nodes/weights of the n_quad-point Gauss-Laguerre rule with weight
    e^{-sigma u} on (0, inf): standard nodes/weights rescaled by 1/sigma."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return x / sigma, w / sigma


def _s06_quad_apply(vals, n_quad, sigma):
    """Q_N^sigma[g] = 2 * sum_n w_n * e^{sigma u_n} * g(u_n) (half line,
    factor 2 absorbed)."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return 2.0 * float(np.sum((w / sigma) * np.exp(x) * vals))


def _s06_level_price(level, n_quad, sigma, damp, alpha, gam, nu, rho, v0,
                   theta, s0, strike, r, big_t, m0):
    """Single-level scaled Gauss-Laguerre price V_(N, level)."""
    un, _ = _s06_lag_nodes(n_quad, sigma)
    gv = _s06_integrand(un, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                      strike, r, big_t, m0)
    return _s06_quad_apply(gv, n_quad, sigma)


def _oracle_single_level_price(level, n_quad, sigma, damp, alpha, gam, nu,
                               rho, v0, theta, s0, strike, r, big_t, m0):
    level = int(level)
    n_quad = int(n_quad)
    if level < 0 or n_quad < 1:
        raise ValueError("level must be >= 0 and n_quad >= 1")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if damp >= -1.0:
        raise ValueError("damp must be < -1")
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam < 0.0 or nu <= 0.0:
        raise ValueError("gam must be >= 0 and nu > 0")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if v0 < 0.0 or theta < 0.0:
        raise ValueError("v0 and theta must be non-negative")
    if s0 <= 0.0 or strike <= 0.0 or big_t <= 0.0 or int(m0) < 1:
        raise ValueError("s0, strike, big_t must be positive and m0 >= 1")
    return float(_s06_level_price(level, n_quad, sigma, damp, alpha, gam,
                                  nu, rho, v0, theta, s0, strike, r, big_t,
                                  int(m0)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "single_level_price(1, 16, 6.254084328929, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
            "gold_call": "_oracle_single_level_price(1, 16, 6.254084328929, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "single_level_price(0, 16, 6.254084328929, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
            "gold_call": "_oracle_single_level_price(0, 16, 6.254084328929, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "single_level_price(0, 4, 6.254084328929, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
            "gold_call": "_oracle_single_level_price(0, 4, 6.254084328929, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
        },
    ]
