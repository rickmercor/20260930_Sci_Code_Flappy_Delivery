"""
Assembles the fully discrete characteristic-function exponent G_level(xi) at a single Fourier argument xi = u + i*damp from the nodal fractional-Riccati values on the level grid, combining the drift contribution with the time integral of the exponent integrand evaluated on the same nodal grid, and returns Re G_level(xi). This step validates the level-defining discrete exponent. Deliberately excluded: the payoff transform, the damped integrand prefactor, and all quadrature in the Fourier variable.

The characteristic function of the log price is the exponential of a time integral involving the Riccati solution; a fully discrete exponent replaces both the Riccati solution (nodal values) and the time integral (a quadrature over the same nodal grid) so that each hierarchy level is a well-defined computable object. At xi = 0 the exponent integrand vanishes identically along h = 0, so G(0) = 0 and the discrete construction must reproduce it exactly (the characteristic function equals 1 at the origin). For the frozen benchmark configuration at xi = 1.5 - 4.0i on the level-1 grid, Re G_1(xi) = 20.165708488902.

Returns
-------
float — the real part of the fully discrete characteristic-function exponent at the requested level (a single scalar).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cf_exponent_re(u: float, damp: float, level: int, alpha: float,
                   gam: float, nu: float, rho: float, v0: float,
                   theta: float, s0: float, r: float, big_t: float,
                   m0: int) -> float:
    """Real part of the fully discrete characteristic exponent G_level.

    Parameters
    ----------
    u : float
        Real part of the Fourier argument, u >= 0.
    damp : float
        Imaginary part of the Fourier argument (damping), damp <= 0.
    level : int
        Discretization level, level >= 0.
    alpha, gam, nu, rho : float
        Rough Heston model parameters (see step 3 for admissible ranges).
    v0 : float
        Initial variance, v0 >= 0.
    theta : float
        Long-run variance, theta >= 0.
    s0 : float
        Spot price, s0 > 0.
    r : float
        Risk-free rate.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.

    Returns
    -------
    float
        Re G_level(u + i*damp).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _s04_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s04_adams(xi, num_steps, dt, alpha, gam, nu, rho):
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
    fh[0] = _s04_F(xi, h[0], gam, nu, rho)
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
        h[n + 1] = pre_c * (_s04_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s04_F(xi, h[n + 1], gam, nu, rho)
    return h


def _s04_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r, big_t, m0):
    """Fully discrete characteristic-function exponent G_level(xi):
    composite trapezoid (half weights at both endpoints, j = 0 included)
    of J(xi, h_j) = theta*gam*h_j + v0*F(xi, h_j) over the nodal Adams
    solution, plus the drift term i*xi*(log(s0) + r*T)."""
    import numpy as np
    from math import log
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    num_steps = m0 * 2 ** level
    dt = big_t / num_steps
    h = _s04_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    jv = theta * gam * h + v0 * _s04_F(xi, h, gam, nu, rho)
    trap = dt * (0.5 * jv[0] + jv[1:-1].sum(axis=0) + 0.5 * jv[-1])
    return 1j * xi * (log(s0) + r * big_t) + trap


def _oracle_cf_exponent_re(u, damp, level, alpha, gam, nu, rho, v0, theta,
                           s0, r, big_t, m0):
    import numpy as np
    if u < 0.0:
        raise ValueError("u must be non-negative")
    if damp > 0.0:
        raise ValueError("damp must be <= 0")
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
    if s0 <= 0.0 or big_t <= 0.0 or int(m0) < 1:
        raise ValueError("s0, big_t must be positive and m0 >= 1")
    xi = np.array([u + 1j * damp])
    gv = _s04_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r,
                       big_t, int(m0))
    return float(gv[0].real)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "cf_exponent_re(1.5, -4.0, 1, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 0.02, 2.0, 32)",
            "gold_call": "_oracle_cf_exponent_re(1.5, -4.0, 1, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 0.02, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "cf_exponent_re(5.0, -4.0, 0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 0.02, 2.0, 32)",
            "gold_call": "_oracle_cf_exponent_re(5.0, -4.0, 0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 0.02, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "cf_exponent_re(0.0, 0.0, 0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 0.02, 2.0, 32)",
            "gold_call": "_oracle_cf_exponent_re(0.0, 0.0, 0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 0.02, 2.0, 32)",
        },
    ]
