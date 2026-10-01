#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def uniform_steady_states(phi: float, B: float) -> "np.ndarray":
    """Eqs. (3)-(4): interior equilibrium xi = -phi (2 beta - 1) = -phi B and the bistability
    threshold phi_c = 1 / B of the uniform pathway; returns [xi, phi_c]."""
    if isinstance(phi, bool) or isinstance(B, bool) or not np.isfinite(float(phi)) or not np.isfinite(float(B)):
        raise ValueError("phi and B must be finite numbers")
    p, b = float(phi), float(B)
    if abs(p) > 1.0 or b <= 1.0:
        raise ValueError("need |phi| <= 1 and B > 1")
    return np.array([-p * b, 1.0 / b], dtype=np.float64)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _as_vector(v, name):
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a non-empty 1-D array of finite numbers")
    return a


def _check_params(alpha, B, phi):
    a = _as_vector(alpha, "alpha"); b = _as_vector(B, "B"); p = _as_vector(phi, "phi")
    if not (a.size == b.size == p.size):
        raise ValueError("alpha, B and phi must have the same length")
    if np.any(a <= 0.0) or np.any(b <= 1.0) or np.any(np.abs(p) > 1.0):
        raise ValueError("need alpha > 0, B > 1 and |phi| <= 1 on every edge")
    return a, b, p


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def cascade_rate(x: "np.ndarray", x_in: float, alpha: "np.ndarray", B: "np.ndarray",
                         phi: "np.ndarray") -> "np.ndarray":
    """Eq. (1) for every node, with the constant upstream input x_in feeding node 1 and
    beta_i = (B_i + 1) / 2. Activation is driven by (1 + x_{i-1}) and weighted (1 + phi_i)/4,
    inactivation by (1 - x_{i-1}) and weighted (1 - phi_i)/4; the saturating factors are
    alpha beta (1 -/+ x_i) / (2 beta - (1 +/- x_i))."""
    a, b, p = _check_params(alpha, B, phi)
    xv = _as_vector(x, "x")
    if xv.size != a.size or np.any(np.abs(xv) >= b):
        raise ValueError("x must have one entry per edge with |x_i| < B_i (the saturating denominators must stay positive)")
    xin = _check_scalar(x_in, "x_in", -1.0, 1.0)
    beta = (b + 1.0) / 2.0
    up = np.concatenate(([xin], xv[:-1]))
    act = (1.0 + p) / 4.0 * (1.0 + up) * a * beta * (1.0 - xv) / (2.0 * beta - (1.0 + xv))
    ina = (1.0 - p) / 4.0 * (1.0 - up) * a * beta * (1.0 + xv) / (2.0 * beta - (1.0 - xv))
    return act - ina

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _check_params(alpha, B, phi):
    a = _as_vector(alpha, "alpha"); b = _as_vector(B, "B"); p = _as_vector(phi, "phi")
    if not (a.size == b.size == p.size):
        raise ValueError("alpha, B and phi must have the same length")
    if np.any(a <= 0.0) or np.any(b <= 1.0) or np.any(np.abs(p) > 1.0):
        raise ValueError("need alpha > 0, B > 1 and |phi| <= 1 on every edge")
    return a, b, p


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def simulate_cascade(alpha: "np.ndarray", B: "np.ndarray", phi: "np.ndarray", x_in: float,
                             x_init: float, dt: float, threshold: float, max_steps: int) -> "np.ndarray":
    """Integrate eq. (1) from the uniform state x_i(0) = x_init under the sustained input x_in and
    return the profiles at t_j = j dt, j = 0..J, as a (J + 1, N) array. J is the first sample at
    which the terminal node has moved by more than `threshold` from x_init (the source halts its
    velocity tracking the moment the front reaches the terminal node, Sec. 2.4); if that never
    happens within max_steps samples the last row is at j = max_steps."""
    a, b, p = _check_params(alpha, B, phi)
    xin = _check_scalar(x_in, "x_in", -1.0, 1.0)
    x0 = _check_scalar(x_init, "x_init", -1.0, 1.0)
    h = _check_scalar(dt, "dt", 0.0, strict_lo=True)
    thr = _check_scalar(threshold, "threshold", 0.0, strict_lo=True)
    if isinstance(max_steps, bool) or int(max_steps) != max_steps or int(max_steps) < 1:
        raise ValueError("max_steps must be a positive integer")
    x = np.full(a.size, x0, dtype=np.float64)
    rows = [x.copy()]
    for j in range(int(max_steps)):
        sol = solve_ivp(lambda t, y: cascade_rate(y, xin, a, b, p), (j * h, (j + 1) * h), x,
                        method="RK45", rtol=1e-9, atol=1e-11)
        x = sol.y[:, -1]
        rows.append(x.copy())
        if abs(x[-1] - x0) > thr:
            break
    return np.array(rows, dtype=np.float64)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _as_vector(v, name):
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a non-empty 1-D array of finite numbers")
    return a


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def _profile_interpolant(positions, profile, x_in):
    """Piecewise-linear interpolant of the profile with the input counted as node 0 at position 0;
    positions before node 0 take the input value (np.interp clamps)."""
    P = np.concatenate(([0.0], positions))
    V = np.concatenate(([x_in], profile))
    return lambda q: np.interp(q, P, V)


def _shift_misfit(c, f, pos, nxt, h):
    """Least-squares misfit between the later profile and the earlier one shifted downstream by c h."""
    return float(np.sum((nxt - f(pos - c * h)) ** 2))


def instantaneous_speed(profile_prev: "np.ndarray", profile_next: "np.ndarray",
                                positions: "np.ndarray", x_in: float, dt: float) -> float:
    """Eq. (14)/(16): c_j = argmin_{c >= 0} sum_i (x_i(t_{j+1}) - x~(pos_i - c dt, t_j))^2 with x~ the
    LINEAR interpolation of the earlier profile over the given node positions (input as node 0 at
    position 0). The minimisation is global over shifts up to the whole pathway length (a coarse
    grid of 2001 candidates followed by bounded refinement), because the objective is not unimodal
    once the front approaches the terminal node."""
    prev = _as_vector(profile_prev, "profile_prev"); nxt = _as_vector(profile_next, "profile_next")
    pos = _as_vector(positions, "positions")
    if not (prev.size == nxt.size == pos.size) or np.any(np.diff(pos) <= 0.0) or pos[0] <= 0.0:
        raise ValueError("profiles and positions must have equal length and positions must be increasing and positive")
    xin = _check_scalar(x_in, "x_in", -1.0, 1.0)
    h = _check_scalar(dt, "dt", 0.0, strict_lo=True)
    f = _profile_interpolant(pos, prev, xin)
    cmax = pos[-1] / h
    grid = np.linspace(0.0, cmax, 2001)
    vals = np.array([_shift_misfit(c, f, pos, nxt, h) for c in grid])
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    res = minimize_scalar(_shift_misfit, bounds=(lo, hi), args=(f, pos, nxt, h), method="bounded",
                          options={"xatol": 1e-13})
    return float(res.x)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _as_vector(v, name):
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a non-empty 1-D array of finite numbers")
    return a


def _check_params(alpha, B, phi):
    a = _as_vector(alpha, "alpha"); b = _as_vector(B, "B"); p = _as_vector(phi, "phi")
    if not (a.size == b.size == p.size):
        raise ValueError("alpha, B and phi must have the same length")
    if np.any(a <= 0.0) or np.any(b <= 1.0) or np.any(np.abs(p) > 1.0):
        raise ValueError("need alpha > 0, B > 1 and |phi| <= 1 on every edge")
    return a, b, p


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def _uniform_chain_rhs(t, y, x_in, av, bv, pv):
    """Right-hand side of a uniform reference chain (module-level so the oracle stays a single function)."""
    return cascade_rate(y, x_in, av, bv, pv)


def _uniform_chain_delays(n, alpha, B, phi, x_in, x_init, sgn):
    """Zero-crossing delays between consecutive nodes of an n-node uniform chain, from a tight DOP853 integration
    with root-located crossing times (dense output + Brent)."""
    av, bv, pv = np.full(n, alpha), np.full(n, B), np.full(n, phi)
    event = (lambda t, y, *args: sgn * y[-1])
    event.terminal = True
    event.direction = 1.0
    sol = solve_ivp(_uniform_chain_rhs, (0.0, 400.0 * n / alpha), np.full(n, x_init), args=(x_in, av, bv, pv),
                    method="DOP853", rtol=1e-13, atol=1e-15, dense_output=True, events=event)
    if sol.t_events[0].size == 0:
        raise ValueError("the front did not reach the end of the reference chain")
    grid = np.linspace(0.0, float(sol.t[-1]), 20001)
    X = sol.sol(grid)
    crossing = np.full(n, np.nan)
    for i in range(n):
        g = sgn * X[i]
        idx = np.where((g[:-1] < 0.0) & (g[1:] >= 0.0))[0]
        if idx.size == 0:
            continue
        k = int(idx[0])
        crossing[i] = brentq(lambda t, i=i: sgn * float(sol.sol(t)[i]), grid[k], grid[k + 1], xtol=1e-14)
    delays = np.diff(crossing)
    return delays[np.isfinite(delays)]


def intrinsic_edge_speed(alpha: float, B: float, phi: float, x_in: float, x_init: float,
                                 rel_tol: float) -> float:
    """Property-defined intrinsic speed c(alpha, B, phi): the asymptotic speed, in nodes per unit time, at which
    the front launched by the constant input x_in into the uniform resting state x_init travels through an
    infinitely long uniform pathway built from this edge, i.e. the speed at which the front profile translates
    without change of shape once the boundary transient has decayed. Returned to a relative accuracy of rel_tol
    or better.

    Construction: the reciprocal of the limiting per-node delay of the front. The delay between the zero crossings
    of consecutive nodes converges geometrically along the chain (contraction factor between 0.55 and 0.95 per node
    for bistable edges, slowest close to the bistability boundary), so a chain of 240, 480 or 960 nodes integrated
    with a tight tolerance and root-located crossing times (dense output + Brent, never sample interpolation) is
    extended until the last ten delays agree to rel_tol / 10; a ValueError is raised if that never happens."""
    for name, v in (("alpha", alpha), ("B", B), ("phi", phi), ("x_in", x_in), ("x_init", x_init), ("rel_tol", rel_tol)):
        if isinstance(v, bool) or not np.isfinite(float(v)):
            raise ValueError(f"{name} must be a finite number")
    a, b, p = float(alpha), float(B), float(phi)
    xin, x0, tol = float(x_in), float(x_init), float(rel_tol)
    if a <= 0.0 or b <= 1.0 or abs(p) >= 1.0 / b:
        raise ValueError("need alpha > 0, B > 1 and a bistable edge |phi| < 1 / B")
    if not (0.0 < tol <= 1e-6):
        raise ValueError("rel_tol must lie in (0, 1e-6]")
    if abs(abs(xin) - 1.0) > 0.0 or abs(abs(x0) - 1.0) > 0.0 or xin == x0:
        raise ValueError("the front must be launched from one saturated state into the other: x_in, x_init in {-1, +1}, x_in != x_init")
    sgn = 1.0 if xin > x0 else -1.0
    for n in (240, 480, 960):
        delays = _uniform_chain_delays(n, a, b, p, xin, x0, sgn)
        if delays.size < 20:
            raise ValueError("too few nodes crossed on the reference chain")
        tail = delays[-10:]
        # certificate: with a contraction factor below 0.95 per node, a last-ten spread under tol/10 bounds the
        # remaining deviation of the limiting delay by about tol/4
        if (tail.max() - tail.min()) <= 0.1 * tol * tail.mean():
            return float(1.0 / tail.mean())
    raise ValueError("the per-node delay did not converge to the requested accuracy on a 960-node chain")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _as_vector(v, name):
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a non-empty 1-D array of finite numbers")
    return a


def rescaled_positions(edge_speeds: "np.ndarray") -> "np.ndarray":
    """Eq. (15): s_0 = 0, s_{i+1} = s_i + cbar / c_i with cbar = (sum_k 1/c_k)^-1, so that the
    rescaled pathway has unit length; returns s_1..s_N."""
    c = _as_vector(edge_speeds, "edge_speeds")
    if np.any(c <= 0.0):
        raise ValueError("edge speeds must be positive")
    inv = 1.0 / c
    return np.cumsum(inv) / inv.sum()

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _as_vector(v, name):
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a non-empty 1-D array of finite numbers")
    return a


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def wave_centre(positions: "np.ndarray", profile: "np.ndarray", x_in: float) -> float:
    """Position at which the linearly interpolated profile (input as node 0 at position 0) first
    crosses zero coming from the input side; the terminal position if it never does."""
    pos = _as_vector(positions, "positions"); prof = _as_vector(profile, "profile")
    if pos.size != prof.size or np.any(np.diff(pos) <= 0.0) or pos[0] <= 0.0:
        raise ValueError("positions must be increasing and positive with one entry per profile value")
    xin = _check_scalar(x_in, "x_in", -1.0, 1.0)
    P = np.concatenate(([0.0], pos)); V = np.concatenate(([xin], prof))
    idx = np.where((V[:-1] > 0.0) & (V[1:] <= 0.0))[0]
    if idx.size == 0:
        return float(P[-1])
    i = int(idx[0])
    return float(P[i] + (0.0 - V[i]) * (P[i + 1] - P[i]) / (V[i + 1] - V[i]))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _as_vector(v, name):
    a = np.asarray(v, dtype=float)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a non-empty 1-D array of finite numbers")
    return a


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def velocity_ise(speeds: "np.ndarray", times: "np.ndarray", centres: "np.ndarray",
                         s_lo: float, s_hi: float) -> float:
    """Sec. 3.2 VISE: trapezoidal integral over the post-transient window of the squared deviation of
    the speed from its window mean. The window is the set of sampling intervals whose START time has
    the wave centre (a fraction of the pathway length) in [s_lo, s_hi); speeds[j] belongs to the
    interval starting at times[j]."""
    c = _as_vector(speeds, "speeds"); t = _as_vector(times, "times"); s = _as_vector(centres, "centres")
    if not (c.size == t.size == s.size) or np.any(np.diff(t) <= 0.0):
        raise ValueError("speeds, times and centres must have equal length and times must increase")
    lo = _check_scalar(s_lo, "s_lo", 0.0, 1.0); hi = _check_scalar(s_hi, "s_hi", 0.0, 1.0)
    if hi <= lo:
        raise ValueError("need s_lo < s_hi")
    mask = (s >= lo) & (s < hi)
    if mask.sum() < 2:
        raise ValueError("fewer than two sampling intervals fall in the window")
    cw = c[mask]; tw = t[mask]
    dev = cw - cw.mean()
    return float(np.trapezoid(dev ** 2, tw))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


def _check_params(alpha, B, phi):
    a = _as_vector(alpha, "alpha"); b = _as_vector(B, "B"); p = _as_vector(phi, "phi")
    if not (a.size == b.size == p.size):
        raise ValueError("alpha, B and phi must have the same length")
    if np.any(a <= 0.0) or np.any(b <= 1.0) or np.any(np.abs(p) > 1.0):
        raise ValueError("need alpha > 0, B > 1 and |phi| <= 1 on every edge")
    return a, b, p


def _check_scalar(v, name, lo=None, hi=None, strict_lo=False):
    if isinstance(v, bool) or not np.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number")
    v = float(v)
    if lo is not None and (v < lo or (strict_lo and v <= lo)):
        raise ValueError(f"{name} out of range")
    if hi is not None and v > hi:
        raise ValueError(f"{name} out of range")
    return v


def fluctuation_suppression_ratio(alpha: "np.ndarray", B: "np.ndarray", phi: "np.ndarray",
                                          x_in: float, x_init: float, dt: float, threshold: float,
                                          speed_rel_tol: float, s_lo: float, s_hi: float) -> float:
    """End-to-end: bistability guard (step 01), resting-state guard (02), heterogeneous simulation (03), intrinsic edge speeds
    (05: the asymptotic travelling-wave speed to speed_rel_tol, using c(alpha,B,phi) = alpha c(1,B,phi) since alpha
    only rescales time), rescaled
    coordinate (06), least-squares speeds in both coordinates (04), wave centres (07) and the two
    VISE values (08). Speeds are compared as fractions of the respective pathway length per unit
    time (c_j / N against c~_j / 1). Returns VISE(normalised) / VISE(original)."""
    a, b, p = _check_params(alpha, B, phi)
    for pi, bi in zip(p, b):
        xi_phic = uniform_steady_states(pi, bi)
        if abs(pi) >= xi_phic[1]:
            raise ValueError("every edge must lie in the bistable regime |phi| < 1/B")
    h = _check_scalar(dt, "dt", 0.0, strict_lo=True)
    n = a.size
    # the pathway must start at rest: the uniform state x_init driven by an input equal to x_init is stationary
    rest = cascade_rate(np.full(n, float(x_init)), float(x_init), a, b, p)
    if np.max(np.abs(rest)) > 1e-9:
        raise ValueError("x_init is not a uniform steady state of the pathway")
    profiles = simulate_cascade(a, b, p, x_in, x_init, h, threshold, 100000)
    n_int = profiles.shape[0] - 1
    times = h * np.arange(n_int)
    base = {}
    for bi, pi in set(zip(b.tolist(), p.tolist())):
        base[(bi, pi)] = intrinsic_edge_speed(1.0, bi, pi, x_in, x_init, speed_rel_tol)
    c_edge = np.array([a[i] * base[(b[i], p[i])] for i in range(n)])
    s = rescaled_positions(c_edge)
    pos = np.arange(1, n + 1, dtype=np.float64)
    xin = float(x_in)
    c_orig = np.array([instantaneous_speed(profiles[j], profiles[j + 1], pos, xin, h)
                       for j in range(n_int)]) / n
    c_norm = np.array([instantaneous_speed(profiles[j], profiles[j + 1], s, xin, h)
                       for j in range(n_int)]) / s[-1]
    centres = np.array([wave_centre(s, profiles[j], xin) for j in range(n_int)]) / s[-1]
    v_orig = velocity_ise(c_orig, times, centres, s_lo, s_hi)
    v_norm = velocity_ise(c_norm, times, centres, s_lo, s_hi)
    return v_norm / v_orig
SCICODE_GOLD_EOF
