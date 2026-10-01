#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

# ORACLE SOLUTION


def variance_riccati(delta: np.ndarray, tau: float, xi: float, c0: float, c1: float) -> np.ndarray:
    import numpy as np

    d = np.atleast_1d(np.asarray(delta, dtype=complex))
    if d.ndim != 1 or d.size < 1 or not np.all(np.isfinite(d)):
        raise ValueError("delta must be a finite one-dimensional array")
    for name, value in (("tau", tau), ("xi", xi), ("c0", c0), ("c1", c1)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(tau) < 0.0:
        raise ValueError("tau must be non-negative")
    if float(xi) <= 0.0:
        raise ValueError("xi must be positive")
    t, x = float(tau), float(xi)

    q = 1j * d + d * d
    c = float(c0) + 1j * d * float(c1)
    # The solution is invariant under disc -> -disc; the principal root keeps
    # exp(-disc * tau) bounded.
    disc = np.sqrt(c * c + x * x * q)
    decay = np.exp(-disc * t)
    y = -q * (1.0 - decay) / (2.0 * disc - (c + disc) * (1.0 - decay))
    return np.vstack([y.real, y.imag]).astype(float)

# ORACLE SOLUTION


def liquidity_coefficients(delta: np.ndarray, tau: float, a: float, b: float, eta: float,
                                   m: float, c_x: float, beta_sq: float, n_quad: int = 64) -> np.ndarray:
    import numpy as np

    d = np.atleast_1d(np.asarray(delta, dtype=complex))
    if d.ndim != 1 or d.size < 1 or not np.all(np.isfinite(d)):
        raise ValueError("delta must be a finite one-dimensional array")
    for name, value in (("tau", tau), ("a", a), ("b", b), ("eta", eta), ("m", m),
                        ("c_x", c_x), ("beta_sq", beta_sq)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(tau) < 0.0 or float(a) <= 0.0 or float(eta) <= 0.0 or float(beta_sq) < 0.0:
        raise ValueError("need tau >= 0, a > 0, eta > 0 and beta_sq >= 0")
    if isinstance(n_quad, bool) or not isinstance(n_quad, (int, np.integer)) or int(n_quad) < 2:
        raise ValueError("n_quad must be an integer >= 2")
    t, a_, b_, e_ = float(tau), float(a), float(b), float(eta)

    q = 1j * d + d * d
    lin = float(m) + 1j * d * float(c_x) * e_
    disc = np.sqrt(lin * lin + e_ * e_ * float(beta_sq) * q)

    def c_coef(s):
        e2 = np.exp(-2.0 * disc * s)
        return -(disc ** 2 - lin ** 2) * (1.0 - e2) / (2.0 * e_ ** 2 * (2.0 * disc - (disc + lin) * (1.0 - e2)))

    def b_coef(s):
        e1 = np.exp(-disc * s)
        e2 = e1 * e1
        return (-a_ * b_ * (disc ** 2 - lin ** 2) / (disc * e_ ** 2)
                * (1.0 - e1) ** 2 / (2.0 * disc - (disc + lin) * (1.0 - e2)))

    C = c_coef(t)
    B = b_coef(t)
    e2 = np.exp(-2.0 * disc * t)
    int_eta2_c = 0.5 * (-(disc + lin) * t - np.log((2.0 * disc - (lin + disc) * (1.0 - e2)) / (2.0 * disc)))
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    s = 0.5 * t * (nodes + 1.0)
    w = 0.5 * t * weights
    Bs = b_coef(s[:, None])
    rest = np.sum(w[:, None] * (a_ * b_ * Bs + 0.5 * e_ ** 2 * Bs ** 2), axis=0)
    const = int_eta2_c + rest
    return np.vstack([C.real, C.imag, B.real, B.imag, const.real, const.imag]).astype(float)

def regime_factor(delta: np.ndarray, tau: float, generator: np.ndarray, sig2: np.ndarray,
                          kappa1: float, theta1: np.ndarray, xi1: float, d_c0: float, d_c1: float,
                          kappa2: float, theta2: np.ndarray, xi2: float, e_c0: float, e_c1: float,
                          initial_regime: int, n_steps: int = 200) -> np.ndarray:
    import numpy as np

    d = np.atleast_1d(np.asarray(delta, dtype=complex))
    G = np.asarray(generator, dtype=float)
    s2 = np.atleast_1d(np.asarray(sig2, dtype=float))
    t1 = np.atleast_1d(np.asarray(theta1, dtype=float))
    t2 = np.atleast_1d(np.asarray(theta2, dtype=float))
    if G.ndim != 2 or G.shape[0] != G.shape[1] or G.shape[0] < 1:
        raise ValueError("generator must be a square matrix")
    k = G.shape[0]
    if s2.shape != (k,) or t1.shape != (k,) or t2.shape != (k,):
        raise ValueError("sig2, theta1 and theta2 must have one entry per regime")
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(s2)) and np.all(np.isfinite(t1)) and np.all(np.isfinite(t2))):
        raise ValueError("inputs must be finite")
    off = G - np.diag(np.diag(G))
    if np.any(off < 0.0) or not np.allclose(G.sum(axis=1), 0.0, atol=1e-12):
        raise ValueError("generator must have non-negative off-diagonal entries and zero row sums")
    if isinstance(initial_regime, bool) or not isinstance(initial_regime, (int, np.integer)) or not (1 <= int(initial_regime) <= k):
        raise ValueError("initial_regime must be an integer between 1 and k")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")
    for name, value in (("tau", tau), ("kappa1", kappa1), ("kappa2", kappa2)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(tau) < 0.0:
        raise ValueError("tau must be non-negative")

    q = 1j * d + d * d
    n = int(n_steps)
    h_step = float(tau) / n

    def g_at(s):
        Dr = variance_riccati(d, s, xi1, d_c0, d_c1)
        Er = variance_riccati(d, s, xi2, e_c0, e_c1)
        D = Dr[0] + 1j * Dr[1]
        E = Er[0] + 1j * Er[1]
        return (-0.5 * s2[:, None] * q[None, :] + float(kappa1) * t1[:, None] * D[None, :]
                + float(kappa2) * t2[:, None] * E[None, :])          # shape (k, n)

    def expm_batch(A):
        # Scaling and squaring with a Taylor series, for a stack of (k, k)
        # complex matrices of shape (n, k, k).
        norm = np.max(np.sum(np.abs(A), axis=2), axis=1)
        sq = np.maximum(0, np.ceil(np.log2(np.maximum(norm, 1e-300) / 0.5))).astype(int)
        X = A / (2.0 ** sq)[:, None, None]
        eye = np.broadcast_to(np.eye(k, dtype=complex), A.shape)
        out = eye.copy()
        term = eye.copy()
        for j in range(1, 19):
            term = term @ X / j
            out = out + term
        for i in range(int(sq.max()) if sq.size else 0):
            mask = sq > i
            out[mask] = out[mask] @ out[mask]
        return out

    hv = np.ones((d.size, k), dtype=complex)
    for step in range(n):
        gm = g_at((step + 0.5) * h_step).T                          # shape (n, k)
        A = h_step * (G[None, :, :] + gm[:, :, None] * np.eye(k)[None, :, :])
        hv = np.einsum("nij,nj->ni", expm_batch(A), hv)
    out = hv[:, int(initial_regime) - 1]
    return np.vstack([out.real, out.imag]).astype(float)

# ORACLE SOLUTION


def characteristic_function(delta: np.ndarray, tau: float, model: dict, n_quad: int = 64,
                                    n_steps: int = 200) -> np.ndarray:
    import numpy as np

    keys = ("s1", "s2", "nu1", "nu2", "alpha", "kappa1", "kappa2", "xi1", "xi2", "rho1", "rho2",
            "rho", "rho_tilde1", "rho_tilde2", "beta1", "beta2", "a", "b", "eta",
            "sigma1", "sigma2", "theta1", "theta2", "generator", "initial_regime")
    if not isinstance(model, dict) or any(k not in model for k in keys):
        raise ValueError("model must be a dict with keys " + ", ".join(keys))
    M = model
    for name in ("s1", "s2", "kappa1", "kappa2", "xi1", "xi2", "a", "eta"):
        if not np.isfinite(float(M[name])) or float(M[name]) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("nu1", "nu2", "beta1", "beta2"):
        if not np.isfinite(float(M[name])) or float(M[name]) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("rho", "rho1", "rho2", "rho_tilde1", "rho_tilde2"):
        if not (-1.0 <= float(M[name]) <= 1.0):
            raise ValueError(f"{name} must lie in [-1, 1]")
    d = np.atleast_1d(np.asarray(delta, dtype=complex))

    sig1 = np.asarray(M["sigma1"], dtype=float)
    sig2 = np.asarray(M["sigma2"], dtype=float)
    rho = float(M["rho"])
    # Total constant variance rate of ln(S1/S2) in each regime.
    sig2_ratio = sig1 ** 2 + sig2 ** 2 - 2.0 * rho * sig1 * sig2
    k1, k2 = float(M["kappa1"]), float(M["kappa2"])
    x1, x2 = float(M["xi1"]), float(M["xi2"])
    r1, r2 = float(M["rho1"]), float(M["rho2"])
    eta = float(M["eta"])
    b1, b2 = float(M["beta1"]), float(M["beta2"])
    rt1, rt2 = float(M["rho_tilde1"]), float(M["rho_tilde2"])

    # Variance of asset 1: unchanged drift, covariance +rho1 xi1 nu1 with x.
    d_c0, d_c1 = -k1, r1 * x1
    # Variance of asset 2: the numeraire adds rho2 xi2 nu2 to its drift and
    # its covariance with x is -rho2 xi2 nu2.
    e_c0, e_c1 = r2 * x2 - k2, -r2 * x2
    # Liquidity: the numeraire adds rho_tilde2 beta2 eta alpha to the drift;
    # covariance with x is (rho_tilde1 beta1 - rho_tilde2 beta2) eta alpha.
    m_alpha = rt2 * b2 * eta - float(M["a"])
    c_x = rt1 * b1 - rt2 * b2

    Dr = variance_riccati(d, tau, x1, d_c0, d_c1)
    Er = variance_riccati(d, tau, x2, e_c0, e_c1)
    Lr = liquidity_coefficients(d, tau, float(M["a"]), float(M["b"]), eta, m_alpha, c_x,
                                        b1 * b1 + b2 * b2, n_quad)
    Hr = regime_factor(d, tau, np.asarray(M["generator"], dtype=float), sig2_ratio,
                               k1, np.asarray(M["theta1"], dtype=float), x1, d_c0, d_c1,
                               k2, np.asarray(M["theta2"], dtype=float), x2, e_c0, e_c1,
                               int(M["initial_regime"]), n_steps)
    D = Dr[0] + 1j * Dr[1]
    E = Er[0] + 1j * Er[1]
    C = Lr[0] + 1j * Lr[1]
    B = Lr[2] + 1j * Lr[3]
    K = Lr[4] + 1j * Lr[5]
    H = Hr[0] + 1j * Hr[1]
    x = np.log(float(M["s1"]) / float(M["s2"]))
    al = float(M["alpha"])
    phi = np.exp(1j * d * x + K + B * al + C * al * al + D * float(M["nu1"]) + E * float(M["nu2"])) * H
    return np.vstack([phi.real, phi.imag]).astype(float)

# ORACLE SOLUTION


def exercise_probabilities(tau: float, model: dict, delta_max: float = 40.0, n_nodes: int = 240,
                                   n_quad: int = 64, n_steps: int = 200) -> np.ndarray:
    import numpy as np

    for name, value in (("tau", tau), ("delta_max", delta_max)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer >= 2")
    nodes, weights = np.polynomial.legendre.leggauss(int(n_nodes))
    dl = 0.5 * float(delta_max) * (nodes + 1.0)
    wl = 0.5 * float(delta_max) * weights

    f_mi = characteristic_function(np.array([-1j]), tau, model, n_quad, n_steps)
    phi_mi = f_mi[0][0] + 1j * f_mi[1][0]
    f2 = characteristic_function(dl, tau, model, n_quad, n_steps)
    f1 = characteristic_function(dl - 1j, tau, model, n_quad, n_steps)
    phi2 = f2[0] + 1j * f2[1]
    phi1 = f1[0] + 1j * f1[1]
    p2 = 0.5 + np.sum(wl * np.real(phi2 / (1j * dl))) / np.pi
    p1 = 0.5 + np.sum(wl * np.real(phi1 / (1j * dl * phi_mi))) / np.pi
    return np.array([phi_mi.real, p1, p2], dtype=float)

# ORACLE SOLUTION


def margrabe_price(s1: float, s2: float, variance_rate: float, tau: float) -> float:
    import numpy as np
    from math import erf, log, sqrt

    for name, value in (("s1", s1), ("s2", s2), ("variance_rate", variance_rate), ("tau", tau)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    s = sqrt(float(variance_rate) * float(tau))
    d1 = (log(float(s1) / float(s2)) + 0.5 * s * s) / s
    norm_cdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    return float(float(s1) * norm_cdf(d1) - float(s2) * norm_cdf(d1 - s))

# ORACLE SOLUTION


def price_comparisons(tau: float, model: dict, delta_max: float = 40.0, n_nodes: int = 240,
                              n_quad: int = 64, n_steps: int = 200) -> np.ndarray:
    import numpy as np

    if not isinstance(model, dict) or "generator" not in model or "initial_regime" not in model:
        raise ValueError("model must be a dict as in characteristic_function")
    G = np.asarray(model["generator"], dtype=float)
    if G.shape != (2, 2):
        raise ValueError("the comparison needs a two-regime model")

    def value(m):
        out = exercise_probabilities(tau, m, delta_max, n_nodes, n_quad, n_steps)
        return float(m["s2"]) * (out[0] * out[1] - out[2])

    full = value(model)
    other = value(dict(model, initial_regime=3 - int(model["initial_regime"])))
    frozen = value(dict(model, generator=[[0.0, 0.0], [0.0, 0.0]]))
    no_liquidity = value(dict(model, beta1=0.0, beta2=0.0))
    return np.array([full, other, frozen, no_liquidity], dtype=float)

# ORACLE SOLUTION


def exchange_option_price(model: dict = None, tau: float = 1.0, delta_max: float = 40.0,
                                  n_nodes: int = 240, n_quad: int = 64, n_steps: int = 200) -> float:
    import numpy as np

    if model is None:
        model = dict(s1=100.0, s2=95.0, nu1=0.1, nu2=0.1, alpha=0.3, kappa1=2.0, kappa2=2.0,
                     xi1=0.1, xi2=0.1, rho1=-0.5, rho2=-0.5, rho=-0.25, rho_tilde1=-0.7,
                     rho_tilde2=-0.7, beta1=0.5, beta2=0.5, a=0.2, b=0.3, eta=0.9,
                     sigma1=[0.1, 0.1], sigma2=[0.2, 0.2], theta1=[0.1, 0.3], theta2=[0.1, 0.3],
                     generator=[[-0.5, 0.5], [0.45, -0.45]], initial_regime=2)
    if isinstance(tau, bool) or not np.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("tau must be a finite positive number")

    # ---- exact properties of the building blocks --------------------------
    at_minus_i = np.array([-1j])
    if np.max(np.abs(variance_riccati(at_minus_i, tau, float(model["xi1"]), -float(model["kappa1"]), 0.0))) > 1e-12:
        raise ValueError("variance coefficient does not vanish at delta = -i")          # step 01
    liq = liquidity_coefficients(at_minus_i, tau, float(model["a"]), float(model["b"]),
                                         float(model["eta"]), -float(model["a"]), 0.0, 0.5, n_quad)
    if np.max(np.abs(liq)) > 1e-12:
        raise ValueError("liquidity coefficients do not vanish at delta = -i")          # step 02
    G = np.asarray(model["generator"], dtype=float)
    k = G.shape[0]
    fac = regime_factor(at_minus_i, tau, G, np.full(k, 0.05), 1.0, np.full(k, 0.1), 0.2,
                                -1.0, 0.0, 1.0, np.full(k, 0.1), 0.2, -1.0, 0.0, 1, 20)
    if abs(fac[0][0] - 1.0) > 1e-12 or abs(fac[1][0]) > 1e-12:
        raise ValueError("regime factor is not one at delta = -i")                      # step 03
    phi = characteristic_function(np.array([0.0, -1j]), tau, model, n_quad, n_steps)
    ratio = float(model["s1"]) / float(model["s2"])
    if abs(phi[0][0] - 1.0) > 1e-10 or abs(phi[0][1] - ratio) > 1e-8 * ratio:
        raise ValueError("characteristic function fails phi(0) = 1 or phi(-i) = S1/S2")  # step 04

    # ---- Margrabe limit of the transform pricing -----------------------------
    # The regime with the largest constant variance rate gives the best-
    # conditioned inversion for the check.
    rho = float(model["rho"])
    sig_a = np.asarray(model["sigma1"], dtype=float)
    sig_b = np.asarray(model["sigma2"], dtype=float)
    rates = sig_a ** 2 + sig_b ** 2 - 2.0 * rho * sig_a * sig_b
    kmax = int(np.argmax(rates))
    s1v, s2v = float(sig_a[kmax]), float(sig_b[kmax])
    var_rate = float(rates[kmax])
    limit = dict(model, nu1=0.0, nu2=0.0, alpha=0.0, beta1=0.0, beta2=0.0, b=0.0,
                 sigma1=[s1v], sigma2=[s2v], theta1=[0.0], theta2=[0.0], generator=[[0.0]],
                 initial_regime=1, rho1=0.0, rho2=0.0, rho_tilde1=0.0, rho_tilde2=0.0)
    pr = exercise_probabilities(tau, limit, delta_max, n_nodes, n_quad, n_steps)  # step 05
    transform_limit = float(model["s2"]) * (pr[0] * pr[1] - pr[2])
    reference = margrabe_price(float(model["s1"]), float(model["s2"]), var_rate, tau)   # step 06
    if abs(transform_limit - reference) > 1e-4 * max(1.0, reference):
        raise ValueError("transform pricing fails the Margrabe limit")

    # ---- full model ------------------------------------------------------------
    values = price_comparisons(tau, model, delta_max, n_nodes, n_quad, n_steps)     # step 07
    return float(values[0])
SCICODE_GOLD_EOF
