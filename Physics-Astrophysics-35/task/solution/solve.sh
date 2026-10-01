#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_two_body_constants(m0: float, m1: float, a: float, e: float) -> np.ndarray:
    import numpy as np

    for name, val in (("m0", m0), ("m1", m1), ("a", a)):
        if not (isinstance(val, (int, float, np.floating)) and np.isfinite(val) and float(val) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(e, (int, float, np.floating)) and np.isfinite(e) and 0.0 < float(e) < 1.0):
        raise ValueError("e must be a finite number with 0 < e < 1")

    G = 4.0 * np.pi ** 2                      # au^3 / (Msun yr^2)
    m0, m1, a = float(m0), float(m1), float(a)
    M = m0 + m1
    eps = m1 / m0
    mu = m0 * m1 / M
    c0 = G * m0 ** 2 * m1 ** 2 / M            # Runge-Lenz coefficient mu * G * m0 * m1
    period = 2.0 * np.pi * np.sqrt(a ** 3 / (G * M))
    return np.array([eps, M, mu, c0, period], dtype=float)

def democratic_heliocentric_coordinates(masses: np.ndarray, positions: np.ndarray,
                                                momenta: np.ndarray) -> np.ndarray:
    import numpy as np

    masses = np.asarray(masses, dtype=float)
    positions = np.asarray(positions, dtype=float)
    momenta = np.asarray(momenta, dtype=float)
    if masses.ndim != 1 or masses.size < 2:
        raise ValueError("masses must be a one-dimensional array with at least 2 entries")
    n = masses.size
    if positions.shape != (n, 3) or momenta.shape != (n, 3):
        raise ValueError("positions and momenta must both have shape (N, 3)")
    if not (np.all(np.isfinite(masses)) and np.all(np.isfinite(positions)) and np.all(np.isfinite(momenta))):
        raise ValueError("all inputs must be finite")
    if np.any(masses <= 0.0):
        raise ValueError("every mass must be > 0")

    M = masses.sum()
    P0 = momenta.sum(axis=0)
    Q = np.empty((n, 3))
    P = np.empty((n, 3))
    Q[0] = (masses[:, None] * positions).sum(axis=0) / M      # centre of mass
    P[0] = P0                                                  # total momentum
    Q[1:] = positions[1:] - positions[0]                       # heliocentric positions
    P[1:] = momenta[1:] - (masses[1:, None] / M) * P0          # barycentric momenta
    return np.stack([Q, P])

def kepler_drift(q: np.ndarray, v: np.ndarray, gm: float, dt: float) -> np.ndarray:
    import numpy as np

    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    if q.shape != (3,) or v.shape != (3,):
        raise ValueError("q and v must both have shape (3,)")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(v))):
        raise ValueError("q and v must be finite")
    if not (isinstance(gm, (int, float, np.floating)) and np.isfinite(gm) and float(gm) > 0.0):
        raise ValueError("gm must be a finite number > 0")
    if not (isinstance(dt, (int, float, np.floating)) and np.isfinite(dt)):
        raise ValueError("dt must be a finite number")
    r0 = np.sqrt(q @ q)
    if r0 == 0.0:
        raise ValueError("q must be nonzero")
    gm, dt = float(gm), float(dt)
    if dt == 0.0:
        return np.stack([q.copy(), v.copy()])

    def stumpff(z):
        if z > 1e-6:
            s = np.sqrt(z)
            return (1.0 - np.cos(s)) / z, (s - np.sin(s)) / s ** 3
        if z < -1e-6:
            s = np.sqrt(-z)
            return (np.cosh(s) - 1.0) / (-z), (np.sinh(s) - s) / s ** 3
        # series for small |z|
        return (0.5 - z / 24.0 + z * z / 720.0 - z ** 3 / 40320.0,
                1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z ** 3 / 362880.0)

    sq = np.sqrt(gm)
    vr0 = (q @ v) / r0
    alpha = 2.0 / r0 - (v @ v) / gm            # reciprocal semimajor axis (< 0 for hyperbolic)
    # initial guess for the universal anomaly
    if alpha > 0.0:
        x = sq * alpha * dt
    else:
        x = np.sign(dt) * np.sqrt(abs(1.0 / alpha)) * np.log(1.0 + abs(dt) * sq / max(r0, 1e-300)) if alpha < 0.0 \
            else sq * dt / r0
    for _ in range(200):
        z = alpha * x * x
        C, S = stumpff(z)
        F = r0 * vr0 / sq * x * x * C + (1.0 - alpha * r0) * x ** 3 * S + r0 * x - sq * dt
        dF = r0 * vr0 / sq * x * (1.0 - alpha * x * x * S) + (1.0 - alpha * r0) * x * x * C + r0
        dx = F / dF
        x -= dx
        if abs(dx) <= 1e-15 * max(1.0, abs(x)):
            break
    else:
        raise ValueError("universal Kepler equation did not converge")
    z = alpha * x * x
    C, S = stumpff(z)
    f = 1.0 - x * x / r0 * C
    g = dt - x ** 3 / sq * S
    qn = f * q + g * v
    rn = np.sqrt(qn @ qn)
    fdot = sq / (rn * r0) * (alpha * x ** 3 * S - x)
    gdot = 1.0 - x * x / rn * C
    vn = fdot * q + gdot * v
    return np.stack([qn, vn])

def runge_lenz_vector(Q: np.ndarray, P: np.ndarray, m0: float, m1: float) -> np.ndarray:
    import numpy as np

    Q = np.asarray(Q, dtype=float)
    P = np.asarray(P, dtype=float)
    if Q.shape != (3,) or P.shape != (3,):
        raise ValueError("Q and P must both have shape (3,)")
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(P))):
        raise ValueError("Q and P must be finite")
    for name, val in (("m0", m0), ("m1", m1)):
        if not (isinstance(val, (int, float, np.floating)) and np.isfinite(val) and float(val) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    q = np.sqrt(Q @ Q)
    if q == 0.0:
        raise ValueError("Q must be nonzero")
    G = 4.0 * np.pi ** 2
    m0, m1 = float(m0), float(m1)
    c0 = G * m0 ** 2 * m1 ** 2 / (m0 + m1)
    L = np.cross(Q, P)
    return np.cross(P, L) - c0 * Q / q

def error_hamiltonian_rl_rate(Q: np.ndarray, P: np.ndarray, m0: float, m1: float,
                                      h: float) -> np.ndarray:
    import numpy as np

    Q = np.asarray(Q, dtype=float)
    P = np.asarray(P, dtype=float)
    if Q.shape != (3,) or P.shape != (3,):
        raise ValueError("Q and P must both have shape (3,)")
    if not (np.all(np.isfinite(Q)) and np.all(np.isfinite(P))):
        raise ValueError("Q and P must be finite")
    for name, val in (("m0", m0), ("m1", m1), ("h", h)):
        if not (isinstance(val, (int, float, np.floating)) and np.isfinite(val) and float(val) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    q = np.sqrt(Q @ Q)
    if q == 0.0:
        raise ValueError("Q must be nonzero")
    G = 4.0 * np.pi ** 2
    m0, m1, h = float(m0), float(m1), float(h)
    eps = m1 / m0
    c0 = G * m0 ** 2 * m1 ** 2 / (m0 + m1)
    L = np.cross(Q, P)
    PQ = P @ Q
    P2 = P @ P
    pref = G * h ** 2 * eps / 24.0
    # {R, H_B} with H_B = (h^2/24) {{V_sun, T0}, T0} = pref * (P^2/Q^3 - 3 (P.Q)^2 / Q^5)
    return pref * (np.cross(Q, L) * (3.0 * P2 / q ** 5 - 15.0 * PQ ** 2 / q ** 7)
                   + np.cross(P, L) * (6.0 * PQ / q ** 5)
                   + 2.0 * c0 * PQ * Q / q ** 6
                   - 2.0 * c0 * P / q ** 4)

def orbit_averaged_artificial_precession(m0: float, m1: float, a: float, e: float, h: float,
                                                 n_samples: int = 512) -> float:
    import numpy as np

    if not (isinstance(h, (int, float, np.floating)) and np.isfinite(h) and float(h) > 0.0):
        raise ValueError("h must be a finite number > 0")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or int(n_samples) < 16:
        raise ValueError("n_samples must be an integer >= 16")
    eps, M, mu, c0, period = compute_two_body_constants(m0, m1, a, e)
    G = 4.0 * np.pi ** 2
    m0, m1, a, e, h, n = float(m0), float(m1), float(a), float(e), float(h), int(n_samples)

    # relative orbit at pericentre, in the xy-plane, angular momentum along +z
    q_rel = np.array([a * (1.0 - e), 0.0, 0.0])
    v_rel = np.array([0.0, np.sqrt(G * M * (1.0 + e) / (a * (1.0 - e))), 0.0])
    positions = np.array([-(m1 / M) * q_rel, (m0 / M) * q_rel])
    momenta = np.array([-mu * v_rel, mu * v_rel])
    QP = democratic_heliocentric_coordinates(np.array([m0, m1]), positions, momenta)
    Q, P = QP[0, 1].copy(), QP[1, 1].copy()

    dt = period / n
    rates = np.empty(n)
    for k in range(n):
        R = runge_lenz_vector(Q, P, m0, m1)
        Rdot = error_hamiltonian_rl_rate(Q, P, m0, m1, h)
        L = np.cross(Q, P)
        zhat = L / np.sqrt(L @ L)
        rates[k] = (np.cross(R, Rdot) @ zhat) / (R @ R)
        qv = kepler_drift(Q, P / mu, G * M, dt)
        Q, P = qv[0], mu * qv[1]
    return float(rates.mean())

def gr_precession_rate(m0: float, m1: float, a: float, e: float) -> float:
    import numpy as np

    for name, val in (("m0", m0), ("m1", m1), ("a", a)):
        if not (isinstance(val, (int, float, np.floating)) and np.isfinite(val) and float(val) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(e, (int, float, np.floating)) and np.isfinite(e) and 0.0 < float(e) < 1.0):
        raise ValueError("e must be a finite number with 0 < e < 1")
    G = 4.0 * np.pi ** 2
    c = 299792458.0 * 365.25 * 86400.0 / 1.495978707e11      # speed of light in au / yr
    m0, m1, a, e = float(m0), float(m1), float(a), float(e)
    M = m0 + m1
    return float(3.0 * G ** 1.5 * m0 * np.sqrt(M) / (c ** 2 * a ** 2.5 * (1.0 - e ** 2)))

def compute_net_apsidal_precession(m0: float = 1.0, m1: float = 0.07, a: float = 0.06,
                                           e: float = 0.5, timestep_minutes: float = 14.0,
                                           n_samples: int = 512) -> float:
    import numpy as np

    if not (isinstance(timestep_minutes, (int, float, np.floating)) and np.isfinite(timestep_minutes)
            and float(timestep_minutes) > 0.0):
        raise ValueError("timestep_minutes must be a finite number > 0")
    h = float(timestep_minutes) / (60.0 * 24.0 * 365.25)            # years
    artificial = orbit_averaged_artificial_precession(m0, m1, a, e, h, n_samples)
    gr = gr_precession_rate(m0, m1, a, e)
    rad_per_yr_to_arcsec_per_century = 180.0 / np.pi * 3600.0 * 100.0
    return float((gr + artificial) * rad_per_yr_to_arcsec_per_century)
SCICODE_GOLD_EOF
