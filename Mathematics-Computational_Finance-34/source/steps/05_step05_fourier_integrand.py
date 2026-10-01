"""
Evaluates the half-line Fourier pricing integrand g_level(u) for the European call at a single point u >= 0: the discounted real part of the product of the discrete characteristic function exp(G_level) and the generalized payoff transform of the call, both evaluated at the damped argument xi = u + i*damp. This step validates the integrand assembly (payoff transform, damping, discounting, real-part reduction). Deliberately excluded: quadrature in u, level differences, and any parameter selection.

Under a damped contour the call price becomes the integral of an even, real integrand over the half line, so pricing reduces to one real integral of a product of the characteristic function and a closed-form payoff transform with a second-order pole structure off the contour. When the mean-reversion speed is zero the variance is constant, the discrete exponent is exact at every level, and the integrand coincides with the Black-Scholes integrand with total variance v0*T — a closed-form reduction used for validation. For the frozen benchmark configuration, g_1(1.5) = 3.212851902941 and g_0(0.0) = 6.553918278685.

Returns
-------
float — the value of the damped half-line Fourier pricing integrand at the requested point and level (a single scalar).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fourier_integrand(u: float, damp: float, level: int, alpha: float,
                      gam: float, nu: float, rho: float, v0: float,
                      theta: float, s0: float, strike: float, r: float,
                      big_t: float, m0: int) -> float:
    """Half-line Fourier pricing integrand g_level(u) for the European call.

    Parameters
    ----------
    u : float
        Fourier variable, u >= 0.
    damp : float
        Damping (contour) parameter, damp < -1 for the call transform.
    level : int
        Discretization level, level >= 0.
    alpha, gam, nu, rho, v0, theta : float
        Rough Heston model parameters (admissible ranges as in steps 3-4).
    s0 : float
        Spot price, s0 > 0.
    strike : float
        Strike price, strike > 0.
    r : float
        Risk-free rate.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.

    Returns
    -------
    float
        g_level(u).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _s05_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s05_adams(xi, num_steps, dt, alpha, gam, nu, rho):
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
    fh[0] = _s05_F(xi, h[0], gam, nu, rho)
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
        h[n + 1] = pre_c * (_s05_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s05_F(xi, h[n + 1], gam, nu, rho)
    return h


def _s05_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r, big_t, m0):
    """Fully discrete characteristic-function exponent G_level(xi):
    composite trapezoid (half weights at both endpoints, j = 0 included)
    of J(xi, h_j) = theta*gam*h_j + v0*F(xi, h_j) over the nodal Adams
    solution, plus the drift term i*xi*(log(s0) + r*T)."""
    import numpy as np
    from math import log
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    num_steps = m0 * 2 ** level
    dt = big_t / num_steps
    h = _s05_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    jv = theta * gam * h + v0 * _s05_F(xi, h, gam, nu, rho)
    trap = dt * (0.5 * jv[0] + jv[1:-1].sum(axis=0) + 0.5 * jv[-1])
    return 1j * xi * (log(s0) + r * big_t) + trap


def _s05_integrand(u_arr, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                 strike, r, big_t, m0):
    """Half-line Fourier integrand g_level(u) = e^{-rT}/(2 pi) *
    Re[exp(G_level(u + i*damp)) * Phat(u + i*damp)] with the damped call
    payoff transform Phat(xi) = -K^{1-i xi} / (xi^2 + i xi)."""
    import numpy as np
    from math import exp, pi, log
    u_arr = np.atleast_1d(np.asarray(u_arr, dtype=float))
    xi = u_arr + 1j * damp
    gv = _s05_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r,
                     big_t, m0)
    phat = -np.exp((1.0 - 1j * xi) * log(strike)) / (xi * xi + 1j * xi)
    return exp(-r * big_t) / (2.0 * pi) * (np.exp(gv) * phat).real


def _oracle_fourier_integrand(u, damp, level, alpha, gam, nu, rho, v0,
                              theta, s0, strike, r, big_t, m0):
    if u < 0.0:
        raise ValueError("u must be non-negative")
    if damp >= -1.0:
        raise ValueError("damp must be < -1 for the call payoff transform")
    level = int(level)
    if level < 0:
        raise ValueError("level must be >= 0")
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
    vals = _s05_integrand(u, damp, level, alpha, gam, nu, rho, v0, theta,
                          s0, strike, r, big_t, int(m0))
    return float(vals[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "fourier_integrand(1.5, -4.0, 1, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
            "gold_call": "_oracle_fourier_integrand(1.5, -4.0, 1, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "fourier_integrand(8.0, -4.0, 0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
            "gold_call": "_oracle_fourier_integrand(8.0, -4.0, 0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "fourier_integrand(0.0, -4.0, 0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
            "gold_call": "_oracle_fourier_integrand(0.0, -4.0, 0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32)",
        },
    ]
