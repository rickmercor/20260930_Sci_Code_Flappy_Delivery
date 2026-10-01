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


def tanaka_drift_rate(s_grid: np.ndarray, strike: float, rate: float, vol: float, eps: float) -> np.ndarray:
    import numpy as np

    s = np.atleast_1d(np.asarray(s_grid, dtype=float))
    if s.ndim != 1 or s.size < 1 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("s_grid must be a finite, non-negative one-dimensional array")
    for name, value, low in (("strike", strike, 0.0), ("eps", eps, 0.0)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= low:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(rate, bool) or not np.isfinite(float(rate)):
        raise ValueError("rate must be finite")
    if isinstance(vol, bool) or not np.isfinite(float(vol)) or float(vol) < 0.0:
        raise ValueError("vol must be finite and non-negative")
    k, r, b, e = float(strike), float(rate), float(vol), float(eps)

    # Tanaka-Meyer: d(K - S)^+ = -1{S<K} dS + (1/2) dL^K, so with dS = r S dt + ...
    # mu - r G = -r S 1{S<K} - r (K - S)^+ + (1/2) dL/dt = -r K 1{S<K} + (1/2) dL/dt.
    below = np.where(s < k, 1.0, np.where(s == k, 0.5, 0.0))
    no_local_time = -r * k * below
    # Local-time density: Gaussian kernel at the strike times b^2 K^2.
    kernel = np.exp(-(s - k) ** 2 / (2.0 * e * e)) / (np.sqrt(2.0 * np.pi) * e)
    local_time = 0.5 * b * b * k * k * kernel
    return np.vstack([no_local_time, local_time]).astype(float)

# ORACLE SOLUTION


def bs_operator_bands(s_grid: np.ndarray, rate: float, vol: float) -> np.ndarray:
    import numpy as np

    s = np.asarray(s_grid, dtype=float)
    if s.ndim != 1 or s.size < 3 or not np.all(np.isfinite(s)) or s[0] < 0.0:
        raise ValueError("s_grid must be a finite, non-negative array of at least 3 nodes")
    d = np.diff(s)
    h = d[0]
    if h <= 0.0 or not np.allclose(d, h, rtol=1e-9, atol=0.0):
        raise ValueError("s_grid must be strictly increasing and uniformly spaced")
    if isinstance(rate, bool) or not np.isfinite(float(rate)):
        raise ValueError("rate must be finite")
    if isinstance(vol, bool) or not np.isfinite(float(vol)) or float(vol) < 0.0:
        raise ValueError("vol must be finite and non-negative")
    r, b = float(rate), float(vol)

    diffusion = 0.5 * b * b * s ** 2 / h ** 2
    drift = r * s / (2.0 * h)
    lower = diffusion - drift
    diag = -2.0 * diffusion - r
    upper = diffusion + drift
    return np.vstack([lower, diag, upper]).astype(float)

# ORACLE SOLUTION


def implicit_step(bands: np.ndarray, dt: float, rhs: np.ndarray, left_value: float,
                          right_value: float) -> np.ndarray:
    import numpy as np
    from scipy.linalg import solve_banded

    B = np.asarray(bands, dtype=float)
    f = np.asarray(rhs, dtype=float)
    if f.ndim != 1 or f.size < 3 or B.shape != (3, f.size):
        raise ValueError("bands must have shape (3, n) and rhs shape (n,), n >= 3")
    for name, value in (("dt", dt), ("left_value", left_value), ("right_value", right_value)):
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(dt) < 0.0:
        raise ValueError("dt must be non-negative")
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(f))):
        raise ValueError("bands and rhs must be finite")
    dt = float(dt)
    n = f.size

    # Banded storage for solve_banded((1, 1), ...): row 0 super, 1 diag, 2 sub.
    ab = np.zeros((3, n))
    ab[0, 1:] = -dt * B[2, :-1]
    ab[1, :] = 1.0 - dt * B[1, :]
    ab[2, :-1] = -dt * B[0, 1:]
    # Dirichlet rows.
    ab[1, 0] = 1.0
    ab[0, 1] = 0.0
    ab[1, -1] = 1.0
    ab[2, -2] = 0.0
    x = f.copy()
    x[0] = float(left_value)
    x[-1] = float(right_value)
    return solve_banded((1, 1), ab, x).astype(float)

# ORACLE SOLUTION


def american_put_exercise(s_grid: np.ndarray, strike: float, rate: float, vol: float,
                                  maturity: float, n_steps: int) -> np.ndarray:
    import numpy as np

    s = np.asarray(s_grid, dtype=float)
    if s.ndim != 1 or s.size < 3 or not np.all(np.isfinite(s)) or s[0] != 0.0:
        raise ValueError("s_grid must be a finite grid of at least 3 nodes starting at 0")
    for name, value in (("strike", strike), ("maturity", maturity)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or int(n_steps) < 1:
        raise ValueError("n_steps must be an integer >= 1")
    bands = bs_operator_bands(s, rate, vol)   # validates uniformity, rate and vol
    k = float(strike)
    m = int(n_steps)
    dt = float(maturity) / m

    payoff = np.maximum(k - s, 0.0)
    value = payoff.copy()
    out = np.zeros((m + 1, s.size))
    for level in range(m - 1, -1, -1):
        continuation = implicit_step(bands, dt, value, k, 0.0)
        exercise = (payoff > continuation) & (payoff > 0.0)
        value = np.maximum(continuation, payoff)
        out[1 + level] = exercise.astype(float)
    out[0] = value
    return out

# ORACLE SOLUTION


def stopped_drift_integral(s_grid: np.ndarray, sources: np.ndarray, exercise: np.ndarray,
                                   rate: float, vol: float, maturity: float) -> np.ndarray:
    import numpy as np

    s = np.asarray(s_grid, dtype=float)
    src = np.atleast_2d(np.asarray(sources, dtype=float))
    ex = np.asarray(exercise, dtype=float)
    if s.ndim != 1 or s.size < 3:
        raise ValueError("s_grid must be one-dimensional with at least 3 nodes")
    n = s.size
    if src.ndim != 2 or src.shape[1] != n or src.shape[0] < 1:
        raise ValueError("sources must have shape (k, n)")
    if ex.ndim != 2 or ex.shape[1] != n or ex.shape[0] < 1:
        raise ValueError("exercise must have shape (n_steps, n)")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(ex))):
        raise ValueError("sources and exercise must be finite")
    if not np.all((ex == 0.0) | (ex == 1.0)):
        raise ValueError("exercise indicators must be 0 or 1")
    if isinstance(maturity, bool) or not np.isfinite(float(maturity)) or float(maturity) <= 0.0:
        raise ValueError("maturity must be a finite positive number")
    bands = bs_operator_bands(s, rate, vol)   # validates the grid, rate and vol
    m = ex.shape[0]
    dt = float(maturity) / m
    stopped = ex > 0.5

    out = np.empty_like(src)
    for j in range(src.shape[0]):
        w = np.zeros(n)
        for level in range(m - 1, -1, -1):
            w = implicit_step(bands, dt, w + dt * src[j], 0.0, 0.0)
            w[stopped[level]] = 0.0
        out[j] = w
    return out

# ORACLE SOLUTION


def european_additive_value(spot: float, strike: float, rate: float, vol: float,
                                    maturity: float, eps: float) -> np.ndarray:
    import numpy as np
    from math import erf, exp, log, pi, sqrt
    from scipy import integrate

    for name, value in (("spot", spot), ("strike", strike), ("vol", vol),
                        ("maturity", maturity), ("eps", eps)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(rate, bool) or not np.isfinite(float(rate)):
        raise ValueError("rate must be finite")
    s0, k, r, b, t_mat, e = (float(x) for x in (spot, strike, rate, vol, maturity, eps))
    if e >= k / 10.0:
        raise ValueError("eps must be below strike / 10")

    mu = r - 0.5 * b * b
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / np.sqrt(2.0 * np.pi)
    norm_cdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def kernel_expectation(u):
        # E[ phi_eps(S_u - K) ]: integrate over whichever of the two laws is wider.
        if u <= 0.0:
            return exp(-(s0 - k) ** 2 / (2.0 * e * e)) / (sqrt(2.0 * pi) * e)
        sd = b * sqrt(u)
        if s0 * sd < e:
            s = s0 * np.exp(mu * u + sd * nodes)
            return float(np.sum(weights * np.exp(-(s - k) ** 2 / (2.0 * e * e)))) / (sqrt(2.0 * pi) * e)
        s = k + e * nodes
        x = np.log(s / s0)
        density = np.exp(-(x - mu * u) ** 2 / (2.0 * sd * sd)) / (s * sd * sqrt(2.0 * pi))
        return float(np.sum(weights * density))

    def drift_part(u):
        if u <= 0.0:
            below = 1.0 if s0 < k else (0.5 if s0 == k else 0.0)
        else:
            below = norm_cdf(-(log(s0 / k) + mu * u) / (b * sqrt(u)))
        return -r * k * exp(-r * u) * below

    def local_time_part(u):
        return 0.5 * b * b * k * k * exp(-r * u) * kernel_expectation(u)

    # u = v^2 resolves the square-root behaviour at the start.
    opts = dict(limit=800, epsabs=1e-13, epsrel=1e-12)
    top = sqrt(t_mat)
    d_int, _ = integrate.quad(lambda v: 2.0 * v * drift_part(v * v), 0.0, top, **opts)
    l_int, _ = integrate.quad(lambda v: 2.0 * v * local_time_part(v * v), 0.0, top, **opts)
    return np.array([max(k - s0, 0.0) + d_int + l_int, d_int, l_int], dtype=float)

# ORACLE SOLUTION


def additive_decomposition(spot: float, strike: float, rate: float, vol: float, maturity: float,
                                   eps: float, n_intervals: int, n_steps: int,
                                   smax_factor: float) -> np.ndarray:
    import numpy as np

    for name, value in (("spot", spot), ("strike", strike), ("vol", vol), ("maturity", maturity)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(smax_factor, bool) or not np.isfinite(float(smax_factor)) or float(smax_factor) <= 1.0:
        raise ValueError("smax_factor must exceed 1")
    if isinstance(n_intervals, bool) or not isinstance(n_intervals, (int, np.integer)) or int(n_intervals) < 2:
        raise ValueError("n_intervals must be an integer >= 2")
    k = float(strike)
    s_max = float(smax_factor) * k
    grid = np.linspace(0.0, s_max, int(n_intervals) + 1)
    h = grid[1] - grid[0]
    i0 = int(round(float(spot) / h))
    if not (0 < i0 < grid.size - 1) or abs(grid[i0] - float(spot)) > 1e-9 * max(1.0, float(spot)):
        raise ValueError("spot must be an interior grid node")

    # Stopping rule and American price (step 04).
    solved = american_put_exercise(grid, k, rate, vol, maturity, n_steps)
    price = solved[0][i0]
    exercise = solved[1:]
    # Drift-rate parts (step 01), integrated up to the stopping time (step 05).
    parts = tanaka_drift_rate(grid, k, rate, vol, eps)
    stopped = stopped_drift_integral(grid, parts, exercise, rate, vol, maturity)
    gain = max(k - float(spot), 0.0)
    no_local_time = gain + stopped[0][i0]
    local_time = stopped[1][i0]
    return np.array([no_local_time + local_time, no_local_time, local_time, price], dtype=float)

# ORACLE SOLUTION


def additive_american_value(spot: float = 100.0, strike: float = 100.0, rate: float = 0.05,
                                    vol: float = 0.2, maturity: float = 1.0, eps: float = 0.5,
                                    n_intervals: int = 4000, n_steps: int = 4000,
                                    smax_factor: float = 4.0) -> float:
    import numpy as np
    from math import erf, exp, log, sqrt

    for name, value in (("spot", spot), ("strike", strike), ("vol", vol),
                        ("maturity", maturity), ("eps", eps)):
        if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
    if isinstance(rate, bool) or not np.isfinite(float(rate)):
        raise ValueError("rate must be finite")
    if isinstance(smax_factor, bool) or not np.isfinite(float(smax_factor)) or float(smax_factor) <= 1.0:
        raise ValueError("smax_factor must exceed 1")
    s0, k, r, b, t_mat, e = (float(x) for x in (spot, strike, rate, vol, maturity, eps))
    if e >= k / 10.0:
        raise ValueError("eps must be below strike / 10")
    if s0 >= float(smax_factor) * k:
        raise ValueError("spot must lie inside the grid")

    # ---- Coarse companion checks, on a grid of spacing eps / 2 -----------------
    s_max = float(smax_factor) * k
    n_coarse = int(np.ceil(s_max / (0.5 * e)))
    grid = np.linspace(0.0, s_max, n_coarse + 1)
    m_coarse = 400
    dt = t_mat / m_coarse
    bands = bs_operator_bands(grid, r, b)                          # step 02
    euro = np.maximum(k - grid, 0.0)
    for level in range(m_coarse - 1, -1, -1):                              # step 03
        euro = implicit_step(bands, dt, euro, k * exp(-r * (t_mat - level * dt)), 0.0)
    norm_cdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
    d1 = (log(s0 / k) + (r + 0.5 * b * b) * t_mat) / (b * sqrt(t_mat))
    bs_put = k * exp(-r * t_mat) * norm_cdf(-(d1 - b * sqrt(t_mat))) - s0 * norm_cdf(-d1)
    euro_at_spot = float(np.interp(s0, grid, euro))
    if abs(euro_at_spot - bs_put) > 0.005 * max(1.0, bs_put):
        raise ValueError("implicit steps fail to reproduce the Black-Scholes put")

    parts = tanaka_drift_rate(grid, k, r, b, e)                    # step 01
    no_exercise = np.zeros((m_coarse, grid.size))
    euro_integral = stopped_drift_integral(grid, parts.sum(axis=0), no_exercise,
                                                   r, b, t_mat)            # step 05
    euro_grid_value = max(k - s0, 0.0) + float(np.interp(s0, grid, euro_integral[0]))
    euro_quad = european_additive_value(s0, k, r, b, t_mat, e)     # step 06
    if abs(euro_grid_value - euro_quad[0]) > 0.005 * max(1.0, abs(euro_quad[0])):
        raise ValueError("grid and quadrature disagree on the European representation")

    american = american_put_exercise(grid, k, r, b, t_mat, m_coarse)   # step 04
    am_at_spot = float(np.interp(s0, grid, american[0]))
    if am_at_spot < max(k - s0, 0.0) - 1e-9 or am_at_spot < euro_at_spot - 1e-6:
        raise ValueError("American price below the intrinsic or European value")

    # ---- Fine evaluation of the representation (step 07) ---------------------
    result = additive_decomposition(s0, k, r, b, t_mat, e, n_intervals, n_steps, smax_factor)
    return float(result[0])
SCICODE_GOLD_EOF
