"""
Solves the fractional Riccati equation of the rough Heston characteristic function at a single Fourier argument xi = u + i*damp on the uniform grid of the discretization hierarchy (level ell has m0 * 2^ell time steps on [0, T]), using the fractional Adams predictor-corrector (PECE) scheme of Diethelm, Ford and Freed with h(0) = 0, and returns the real part of the terminal nodal value h(xi, T). This step validates the time-stepping scheme only. Deliberately excluded: the characteristic-function exponent, the payoff transform, and all quadrature.

The rough Heston characteristic function is exponential-affine in a function h that solves a fractional Riccati equation, i.e. a nonlinear Volterra equation with a weakly singular power kernel. No closed form exists, so h is computed numerically per Fourier argument; the cost of one solve on a grid with M steps scales as O(M^2) because of the history convolution. At xi = 0 the right-hand side vanishes at h = 0, so h(0, t) = 0 identically and the scheme reproduces it exactly. For the frozen benchmark configuration at xi = 1.5 - 4.0i on the level-1 grid (64 steps), the real part of h(xi, T) is 4.776269585778.

Returns
-------
float — the real part of the terminal discrete Riccati solution at the requested level (a single scalar).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def riccati_terminal_re(u: float, damp: float, level: int, alpha: float,
                        gam: float, nu: float, rho: float, big_t: float,
                        m0: int) -> float:
    """Real part of the terminal fractional-Riccati solution h(u + i*damp, T).

    Parameters
    ----------
    u : float
        Real part of the Fourier argument, u >= 0.
    damp : float
        Imaginary part of the Fourier argument (damping), damp <= 0.
    level : int
        Discretization level, level >= 0; the grid has m0 * 2**level steps.
    alpha : float
        Roughness index, 0.5 < alpha < 1.
    gam : float
        Mean-reversion speed, gam >= 0.
    nu : float
        Vol-of-vol scale, nu > 0.
    rho : float
        Correlation, -1 < rho < 1.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Number of time steps at level zero, m0 >= 1.

    Returns
    -------
    float
        Re h(u + i*damp, T) on the level grid.

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _s03_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s03_adams(xi, num_steps, dt, alpha, gam, nu, rho):
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
    fh[0] = _s03_F(xi, h[0], gam, nu, rho)
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
        h[n + 1] = pre_c * (_s03_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s03_F(xi, h[n + 1], gam, nu, rho)
    return h


def _oracle_riccati_terminal_re(u, damp, level, alpha, gam, nu, rho, big_t,
                                m0):
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
    if big_t <= 0.0 or int(m0) < 1:
        raise ValueError("big_t must be > 0 and m0 >= 1")
    num_steps = int(m0) * 2 ** level
    dt = big_t / num_steps
    xi = np.array([u + 1j * damp])
    h = _s03_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    return float(h[-1, 0].real)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "riccati_terminal_re(1.5, -4.0, 1, 0.66, 0.5, 0.12, -0.55, 2.0, 32)",
            "gold_call": "_oracle_riccati_terminal_re(1.5, -4.0, 1, 0.66, 0.5, 0.12, -0.55, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "riccati_terminal_re(5.0, -4.0, 0, 0.66, 0.5, 0.12, -0.55, 2.0, 32)",
            "gold_call": "_oracle_riccati_terminal_re(5.0, -4.0, 0, 0.66, 0.5, 0.12, -0.55, 2.0, 32)",
        },
        {
            "setup": "",
            "call": "riccati_terminal_re(0.0, 0.0, 0, 0.66, 0.5, 0.12, -0.55, 2.0, 32)",
            "gold_call": "_oracle_riccati_terminal_re(0.0, 0.0, 0, 0.66, 0.5, 0.12, -0.55, 2.0, 32)",
        },
    ]
