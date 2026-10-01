"""
Orchestrates the complete pipeline and returns the multilevel benchmark price. It derives the weight scale from the model (step 1 oracle), verifies the scaled-rule invariant (step 2 oracle), computes the pilot level prices (step 6 oracle) and the selected level (step 7 oracle, cross-checked against an internal recomputation), obtains the total allocation (step 8 oracle, cross-checked against the internally recomputed allocation vector), verifies the probe values of the Riccati solution, the discrete exponent, and the integrand (step 3-5 oracles) against its own traced evaluations, and then assembles the telescoped price: the level-zero quadrature plus the sum of correction-level quadratures of level differences, each correction evaluated at its own allocated node count with both levels of the difference sampled at the same scaled nodes. The pipeline's rate convention (the empirical rate used by the selection rule and the prefactor propagation) is fixed internally. Raises on any cross-check inconsistency. A local re-trace of nodal integrand values is used for state that earlier steps do not expose.

The benchmark quantity is the deterministic multilevel approximation itself - the value the hierarchical method commits to at the prescribed tolerance - not the limiting exact price. The tolerance is split equally between discretization and quadrature; the discretization half fixes the finest level via the pilot rule, and the quadrature half is allocated over the level-zero term and the corrections. For the frozen benchmark configuration the pipeline selects L = 4 with allocation (21; 42, 24, 14, 8), a level-zero term of 19.118240793291 and a correction sum of 7.992027083734e-03.

Returns
-------
float — the multilevel benchmark price V_(N, L) of the European call under the frozen pipeline (a single scalar).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rough_heston_multilevel_price(eps: float, damp: float, alpha: float,
                                  gam: float, nu: float, rho: float,
                                  v0: float, theta: float, s0: float,
                                  strike: float, r: float, big_t: float,
                                  m0: int, n_pilot: int, a0: float,
                                  s0_idx: float, a1: float, s_idx: float,
                                  beta: float) -> float:
    """Multilevel scaled-quadrature benchmark price of the European call.

    Parameters
    ----------
    eps : float
        Total error tolerance, eps > 0 (split equally between
        discretization and quadrature).
    damp : float
        Damping parameter, damp < -1.
    alpha, gam, nu, rho, v0, theta : float
        Rough Heston model parameters; gam > 0 here because the weight
        scale is derived from the model.
    s0, strike : float
        Spot and strike, both > 0.
    r : float
        Risk-free rate.
    big_t : float
        Maturity, big_t > 0.
    m0 : int
        Level-zero step count, m0 >= 1.
    n_pilot : int
        Pilot node count for the level-selection indicator, n_pilot >= 1.
    a0, s0_idx, a1, s_idx : float
        Fitted algebraic quadrature-model constants (see step 8).
    beta : float
        Cost exponent of one integrand evaluation, beta > 0.

    Returns
    -------
    float
        The multilevel benchmark price V_(N, L).

    Raises
    ------
    ValueError
        If any parameter is outside its admissible range, or if an
        internal cross-check against a sub-step oracle fails.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _s09_F(xi, h, gam, nu, rho):
    """Riccati right-hand side F(xi, h) of the rough Heston model."""
    return (-0.5 * (xi * xi + 1j * xi)
            + gam * (1j * xi * rho * nu - 1.0) * h
            + 0.5 * (gam * nu) ** 2 * h * h)


def _s09_adams(xi, num_steps, dt, alpha, gam, nu, rho):
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
    fh[0] = _s09_F(xi, h[0], gam, nu, rho)
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
        h[n + 1] = pre_c * (_s09_F(xi, hp, gam, nu, rho) + hist)
        fh[n + 1] = _s09_F(xi, h[n + 1], gam, nu, rho)
    return h


def _s09_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r, big_t, m0):
    """Fully discrete characteristic-function exponent G_level(xi):
    composite trapezoid (half weights at both endpoints, j = 0 included)
    of J(xi, h_j) = theta*gam*h_j + v0*F(xi, h_j) over the nodal Adams
    solution, plus the drift term i*xi*(log(s0) + r*T)."""
    import numpy as np
    from math import log
    xi = np.atleast_1d(np.asarray(xi, dtype=complex))
    num_steps = m0 * 2 ** level
    dt = big_t / num_steps
    h = _s09_adams(xi, num_steps, dt, alpha, gam, nu, rho)
    jv = theta * gam * h + v0 * _s09_F(xi, h, gam, nu, rho)
    trap = dt * (0.5 * jv[0] + jv[1:-1].sum(axis=0) + 0.5 * jv[-1])
    return 1j * xi * (log(s0) + r * big_t) + trap


def _s09_integrand(u_arr, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                 strike, r, big_t, m0):
    """Half-line Fourier integrand g_level(u) = e^{-rT}/(2 pi) *
    Re[exp(G_level(u + i*damp)) * Phat(u + i*damp)] with the damped call
    payoff transform Phat(xi) = -K^{1-i xi} / (xi^2 + i xi)."""
    import numpy as np
    from math import exp, pi, log
    u_arr = np.atleast_1d(np.asarray(u_arr, dtype=float))
    xi = u_arr + 1j * damp
    gv = _s09_exponent(xi, level, alpha, gam, nu, rho, v0, theta, s0, r,
                     big_t, m0)
    phat = -np.exp((1.0 - 1j * xi) * log(strike)) / (xi * xi + 1j * xi)
    return exp(-r * big_t) / (2.0 * pi) * (np.exp(gv) * phat).real


def _s09_lag_nodes(n_quad, sigma):
    """Nodes/weights of the n_quad-point Gauss-Laguerre rule with weight
    e^{-sigma u} on (0, inf): standard nodes/weights rescaled by 1/sigma."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return x / sigma, w / sigma


def _s09_quad_apply(vals, n_quad, sigma):
    """Q_N^sigma[g] = 2 * sum_n w_n * e^{sigma u_n} * g(u_n) (half line,
    factor 2 absorbed)."""
    import numpy as np
    x, w = np.polynomial.laguerre.laggauss(int(n_quad))
    return 2.0 * float(np.sum((w / sigma) * np.exp(x) * vals))


def _s09_level_price(level, n_quad, sigma, damp, alpha, gam, nu, rho, v0,
                   theta, s0, strike, r, big_t, m0):
    """Single-level scaled Gauss-Laguerre price V_(N, level)."""
    un, _ = _s09_lag_nodes(n_quad, sigma)
    gv = _s09_integrand(un, damp, level, alpha, gam, nu, rho, v0, theta, s0,
                      strike, r, big_t, m0)
    return _s09_quad_apply(gv, n_quad, sigma)


def _s09_alloc_vector(big_l, eps_quad, a0, s0_idx, a1, s_idx, beta, p,
                    big_t, m0):
    """Allocation of quadrature points: level-zero and correction counts.

    One half of eps_quad is assigned to the level-zero term and one half to
    the corrections.  Correction prefactors are propagated from the first
    correction level, A_l = a1 * (dt_l / dt_1)^p.  Per-point correction cost
    is c_l = W_l + W_{l-1} with W_l = dt_l^{-beta}.  Real-valued minimisers
    are rounded up (ceiling)."""
    import numpy as np
    from math import ceil
    dt = np.array([big_t / (m0 * 2 ** l) for l in range(big_l + 1)])
    w_cost = dt ** (-beta)
    n0_star = (2.0 * a0 / eps_quad) ** (2.0 / s0_idx)
    a_l = np.array([a1 * (dt[l] / dt[1]) ** p for l in range(1, big_l + 1)])
    c_l = np.array([w_cost[l] + w_cost[l - 1] for l in range(1, big_l + 1)])
    ssum = float(np.sum(a_l ** (2.0 / (s_idx + 2.0))
                        * c_l ** (s_idx / (s_idx + 2.0))))
    n_star = ((a_l / c_l) ** (2.0 / (s_idx + 2.0))
              * ((2.0 / eps_quad) * ssum) ** (2.0 / s_idx))
    n0 = int(ceil(n0_star - 1e-12))
    nl = [max(1, int(ceil(v - 1e-12))) for v in n_star]
    return n0, nl


def _oracle_rough_heston_multilevel_price(eps, damp, alpha, gam, nu, rho,
                                          v0, theta, s0, strike, r, big_t,
                                          m0, n_pilot, a0, s0_idx, a1,
                                          s_idx, beta):
    import numpy as np
    from math import ceil, log2
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if damp >= -1.0:
        raise ValueError("damp must be < -1")
    if not (0.5 < alpha < 1.0):
        raise ValueError("alpha must satisfy 0.5 < alpha < 1")
    if gam <= 0.0 or nu <= 0.0:
        raise ValueError("gam and nu must be positive")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must satisfy -1 < rho < 1")
    if v0 < 0.0 or theta < 0.0:
        raise ValueError("v0 and theta must be non-negative")
    if s0 <= 0.0 or strike <= 0.0 or big_t <= 0.0:
        raise ValueError("s0, strike, big_t must be positive")
    m0 = int(m0)
    n_pilot = int(n_pilot)
    if m0 < 1 or n_pilot < 1:
        raise ValueError("m0 and n_pilot must be >= 1")
    if a0 <= 0.0 or a1 <= 0.0 or s0_idx < 1.0 or s_idx < 1.0:
        raise ValueError("invalid quadrature-model constants")
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    # empirical rate convention of the pipeline
    p_rate = 1.0 + alpha
    eps_disc = 0.5 * eps
    eps_quad = 0.5 * eps

    # step-1 oracle: weight scale
    sigma = _oracle_laguerre_scale(alpha, gam, nu, rho, v0, theta, big_t)

    # step-2 oracle: scaled-rule invariant  Q_N^sigma[e^-sigma u] = 2/sigma
    inv = _oracle_scaled_laguerre_exponential(sigma, 8, sigma)
    if abs(inv - 2.0 / sigma) > 1e-9:
        raise ValueError("scaled-rule invariant violated")

    # step-6 oracle: pilot level prices; pilot indicator
    v1p = _oracle_single_level_price(1, n_pilot, sigma, damp, alpha, gam,
                                     nu, rho, v0, theta, s0, strike, r,
                                     big_t, m0)
    v0p = _oracle_single_level_price(0, n_pilot, sigma, damp, alpha, gam,
                                     nu, rho, v0, theta, s0, strike, r,
                                     big_t, m0)
    d1 = abs(v1p - v0p)

    # step-7 oracle: selected level, cross-checked
    big_l = int(_oracle_select_level(eps_disc, p_rate, n_pilot, sigma,
                                     damp, alpha, gam, nu, rho, v0, theta,
                                     s0, strike, r, big_t, m0))
    if d1 == 0.0:
        big_l_int = 1
    else:
        big_l_int = max(1, int(ceil(
            1.0 + log2(d1 / ((2.0 ** p_rate - 1.0) * eps_disc)) / p_rate)))
    if big_l != big_l_int:
        raise ValueError("level-selection cross-check failed")

    # step-8 oracle: allocation, cross-checked against internal vector
    n0, nl = _s09_alloc_vector(big_l, eps_quad, a0, s0_idx, a1, s_idx,
                               beta, p_rate, big_t, m0)
    total = _oracle_allocate_points_total(big_l, eps_quad, a0, s0_idx, a1,
                                          s_idx, beta, p_rate, big_t, m0)
    if int(total) != n0 + sum(nl):
        raise ValueError("allocation cross-check failed")

    # step-3/4/5 oracles: probe cross-checks against the local trace
    u_probe = 1.5
    href = _oracle_riccati_terminal_re(u_probe, damp, 1, alpha, gam, nu,
                                       rho, big_t, m0)
    htr = _s09_adams(np.array([u_probe + 1j * damp]), 2 * m0,
                     big_t / (2 * m0), alpha, gam, nu, rho)[-1, 0].real
    if abs(href - htr) > 1e-9:
        raise ValueError("Riccati probe cross-check failed")
    gref = _oracle_cf_exponent_re(u_probe, damp, 1, alpha, gam, nu, rho,
                                  v0, theta, s0, r, big_t, m0)
    gtr = _s09_exponent(np.array([u_probe + 1j * damp]), 1, alpha, gam, nu,
                        rho, v0, theta, s0, r, big_t, m0)[0].real
    if abs(gref - gtr) > 1e-9:
        raise ValueError("exponent probe cross-check failed")
    iref = _oracle_fourier_integrand(u_probe, damp, 1, alpha, gam, nu, rho,
                                     v0, theta, s0, strike, r, big_t, m0)
    itr = float(_s09_integrand(u_probe, damp, 1, alpha, gam, nu, rho, v0,
                               theta, s0, strike, r, big_t, m0)[0])
    if abs(iref - itr) > 1e-9:
        raise ValueError("integrand probe cross-check failed")

    # telescoped multilevel price (local re-trace of nodal values)
    u0, _ = _s09_lag_nodes(n0, sigma)
    g0 = _s09_integrand(u0, damp, 0, alpha, gam, nu, rho, v0, theta, s0,
                        strike, r, big_t, m0)
    price = _s09_quad_apply(g0, n0, sigma)
    for lev in range(1, big_l + 1):
        un, _ = _s09_lag_nodes(nl[lev - 1], sigma)
        g_hi = _s09_integrand(un, damp, lev, alpha, gam, nu, rho, v0,
                              theta, s0, strike, r, big_t, m0)
        g_lo = _s09_integrand(un, damp, lev - 1, alpha, gam, nu, rho, v0,
                              theta, s0, strike, r, big_t, m0)
        price += _s09_quad_apply(g_hi - g_lo, nl[lev - 1], sigma)
    return float(price)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "rough_heston_multilevel_price(2.5e-4, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32, 16, 12.0, 8.0, 1.4, 7.0, 2.0)",
            "gold_call": "_oracle_rough_heston_multilevel_price(2.5e-4, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32, 16, 12.0, 8.0, 1.4, 7.0, 2.0)",
        },
        {
            "setup": "",
            "call": "rough_heston_multilevel_price(1.0e-3, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32, 16, 12.0, 8.0, 1.4, 7.0, 2.0)",
            "gold_call": "_oracle_rough_heston_multilevel_price(1.0e-3, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32, 16, 12.0, 8.0, 1.4, 7.0, 2.0)",
        },
        {
            "setup": "",
            "call": "rough_heston_multilevel_price(1.0e-2, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32, 16, 12.0, 8.0, 1.4, 7.0, 2.0)",
            "gold_call": "_oracle_rough_heston_multilevel_price(1.0e-2, -4.0, 0.66, 0.5, 0.12, -0.55, 0.07, 0.35, 100.0, 115.0, 0.02, 2.0, 32, 16, 12.0, 8.0, 1.4, 7.0, 2.0)",
        },
    ]
