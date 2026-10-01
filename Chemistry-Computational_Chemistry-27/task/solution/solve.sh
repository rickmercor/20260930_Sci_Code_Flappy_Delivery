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


def _pade_coefficients(lam):
    """Range-dependent Pade coefficients c1, c2, c3."""
    import numpy as np
    s_ij = np.array([[-3.1649, 13.3501, -14.8057, 5.7029],
                     [43.0042, -191.6623, 273.8968, -128.9334],
                     [65.0419, -266.4627, 361.0431, -162.6996]], dtype=float)
    j = np.arange(1, 5, dtype=float)
    return s_ij @ (float(lam) ** (-j))


def _validate_phi_lam(phi, lam):
    """Shared argument validation for the volume fraction and the well range."""
    import numpy as np
    if not np.isfinite(phi) or not np.isfinite(lam):
        raise ValueError("phi and lam must be finite")
    if not (0.0 < float(phi) < 1.0):
        raise ValueError("phi must satisfy 0 < phi < 1")
    if float(lam) <= 1.0:
        raise ValueError("lam must be > 1")


def pade_effective_volume_fraction(phi: float, lam: float) -> float:
    import numpy as np
    _validate_phi_lam(phi, lam)
    phi = float(phi)
    c1, c2, c3 = _pade_coefficients(lam)
    denom = (1.0 + c3 * phi) ** 3
    if denom == 0.0:
        raise ValueError("degenerate Pade denominator")
    return float(phi * (c1 + c2 * phi) / denom)

import numpy as np


def _pade_derivatives(phi, lam):
    """phi' and its first four derivatives, from p * M = N by Leibniz."""
    import numpy as np
    c1, c2, c3 = _pade_coefficients(lam)
    phi = float(phi)
    N0 = c1 * phi + c2 * phi ** 2
    N1 = c1 + 2.0 * c2 * phi
    N2 = 2.0 * c2
    N3 = 0.0
    N4 = 0.0
    b = 1.0 + c3 * phi
    M0 = b ** 3
    M1 = 3.0 * c3 * b ** 2
    M2 = 6.0 * c3 ** 2 * b
    M3 = 6.0 * c3 ** 3
    M4 = 0.0
    if M0 == 0.0:
        raise ValueError("degenerate Pade denominator")
    p0 = pade_effective_volume_fraction(phi, lam)
    p1 = (N1 - p0 * M1) / M0
    p2 = (N2 - 2.0 * p1 * M1 - p0 * M2) / M0
    p3 = (N3 - 3.0 * p2 * M1 - 3.0 * p1 * M2 - p0 * M3) / M0
    p4 = (N4 - 4.0 * p3 * M1 - 6.0 * p2 * M2 - 4.0 * p1 * M3 - p0 * M4) / M0
    return p0, p1, p2, p3, p4


def contact_number_derivatives(phi: float, lam: float) -> "np.ndarray":
    import numpy as np
    _validate_phi_lam(phi, lam)
    phi = float(phi)
    p0, p1, p2, p3, p4 = _pade_derivatives(phi, lam)
    u = 1.0 - p0
    if u <= 0.0:
        raise ValueError("effective volume fraction reached 1; contact value diverges")
    # g and its p-derivatives, written in u = 1 - p where g = (u**-3 + u**-2) / 2
    g0 = 0.5 * (u ** -3 + u ** -2)
    g1 = 1.5 * u ** -4 + u ** -3
    g2 = 6.0 * u ** -5 + 3.0 * u ** -4
    g3 = 30.0 * u ** -6 + 12.0 * u ** -5
    g4 = 180.0 * u ** -7 + 60.0 * u ** -6
    # Faa di Bruno for G(phi) = g(p(phi))
    G0 = g0
    G1 = g1 * p1
    G2 = g2 * p1 ** 2 + g1 * p2
    G3 = g3 * p1 ** 3 + 3.0 * g2 * p1 * p2 + g1 * p3
    G4 = (g4 * p1 ** 4 + 6.0 * g3 * p1 ** 2 * p2 + 3.0 * g2 * p2 ** 2
          + 4.0 * g2 * p1 * p3 + g1 * p4)
    K = 8.0 * (float(lam) ** 3 - 1.0)
    return np.array([K * phi * G0,
                     K * (G0 + phi * G1),
                     K * (2.0 * G1 + phi * G2),
                     K * (3.0 * G2 + phi * G3),
                     K * (4.0 * G3 + phi * G4)], dtype=float)

import numpy as np


def hard_sphere_reference(phi: float) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(phi):
        raise ValueError("phi must be finite")
    phi = float(phi)
    if not (0.0 < phi < 1.0):
        raise ValueError("phi must satisfy 0 < phi < 1")
    om = 1.0 - phi
    a = phi * (np.log(phi) - 1.0) + (4.0 - 3.0 * phi) / om ** 2 * phi ** 2
    # h = P / V with P = 8 phi - 9 phi^2 + 3 phi^3 and V = (1 - phi)^3, by Leibniz
    P0 = 8.0 * phi - 9.0 * phi ** 2 + 3.0 * phi ** 3
    P1 = 8.0 - 18.0 * phi + 9.0 * phi ** 2
    P2 = -18.0 + 18.0 * phi
    V0 = om ** 3
    V1 = -3.0 * om ** 2
    V2 = 6.0 * om
    h0 = P0 / V0
    h1 = (P1 - h0 * V1) / V0
    h2 = (P2 - 2.0 * h1 * V1 - h0 * V2) / V0
    mu = np.log(phi) + h0
    pi = phi * (1.0 + phi + phi ** 2 - phi ** 3) / om ** 3
    return np.array([a, mu, pi, 1.0 / phi + h1, -1.0 / phi ** 2 + h2], dtype=float)

import numpy as np


def effective_well_depth(beta_eps_sw: float, alpha: float) -> float:
    import numpy as np
    if not np.isfinite(beta_eps_sw) or not np.isfinite(alpha):
        raise ValueError("beta_eps_sw and alpha must be finite")
    x = float(beta_eps_sw)
    a = float(alpha)
    if x < 0.0:
        raise ValueError("beta_eps_sw must be >= 0")
    if not (0.0 < a <= 1.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    h = 0.5 * x
    # log(alpha*e^h + 1-alpha) = h + log(alpha + (1-alpha)*e^-h) avoids overflow
    return float(2.0 * (h + np.log(a + (1.0 - a) * np.exp(-h))))

import numpy as np


def mean_field_term(phi: float, lam: float, beta_eps_eff: float) -> "np.ndarray":
    import numpy as np
    _validate_phi_lam(phi, lam)
    if not np.isfinite(beta_eps_eff):
        raise ValueError("beta_eps_eff must be finite")
    B = float(beta_eps_eff)
    if B < 0.0:
        raise ValueError("beta_eps_eff must be >= 0")
    phi = float(phi)
    n0, n1, n2, n3, _ = contact_number_derivatives(phi, lam)
    return np.array([-0.5 * B * n0 * phi,
                     -0.5 * B * (n0 + phi * n1),
                     -0.5 * B * (2.0 * n1 + phi * n2),
                     -0.5 * B * (3.0 * n2 + phi * n3)], dtype=float)

import numpy as np


def _compressibility_factor(phi):
    """d(phi)/d(pi_HS) and its first three derivatives, from D * Qd = (1-phi)^4."""
    import numpy as np
    phi = float(phi)
    om = 1.0 - phi
    U0 = om ** 4
    U1 = -4.0 * om ** 3
    U2 = 12.0 * om ** 2
    U3 = -24.0 * om
    Q0 = 1.0 + 4.0 * phi + 4.0 * phi ** 2 - 4.0 * phi ** 3 + phi ** 4
    Q1 = 4.0 + 8.0 * phi - 12.0 * phi ** 2 + 4.0 * phi ** 3
    Q2 = 8.0 - 24.0 * phi + 12.0 * phi ** 2
    Q3 = -24.0 + 24.0 * phi
    D0 = U0 / Q0
    D1 = (U1 - D0 * Q1) / Q0
    D2 = (U2 - 2.0 * D1 * Q1 - D0 * Q2) / Q0
    D3 = (U3 - 3.0 * D2 * Q1 - 3.0 * D1 * Q2 - D0 * Q3) / Q0
    return D0, D1, D2, D3


def second_order_term(phi: float, lam: float, beta_eps_eff: float) -> "np.ndarray":
    import numpy as np
    _validate_phi_lam(phi, lam)
    if not np.isfinite(beta_eps_eff):
        raise ValueError("beta_eps_eff must be finite")
    B = float(beta_eps_eff)
    if B < 0.0:
        raise ValueError("beta_eps_eff must be >= 0")
    phi = float(phi)
    _, n1, n2, n3, n4 = contact_number_derivatives(phi, lam)
    D0, D1, D2, D3 = _compressibility_factor(phi)
    # the well factor that restores the exact square-well second virial coefficient
    F = np.expm1(B) - B
    R0, R1, R2, R3 = phi ** 2, 2.0 * phi, 2.0, 0.0
    S0, S1, S2, S3 = n1, n2, n3, n4
    T0, T1, T2, T3 = D0, D1, D2, D3
    Q0 = R0 * S0 * T0
    Q1 = R1 * S0 * T0 + R0 * S1 * T0 + R0 * S0 * T1
    Q2 = (R2 * S0 * T0 + R0 * S2 * T0 + R0 * S0 * T2
          + 2.0 * (R1 * S1 * T0 + R1 * S0 * T1 + R0 * S1 * T1))
    Q3 = (R3 * S0 * T0 + R0 * S3 * T0 + R0 * S0 * T3
          + 3.0 * (R2 * S1 * T0 + R2 * S0 * T1 + R1 * S2 * T0
                   + R0 * S2 * T1 + R1 * S0 * T2 + R0 * S1 * T2)
          + 6.0 * R1 * S1 * T1)
    return np.array([-0.5 * F * Q0, -0.5 * F * Q1, -0.5 * F * Q2, -0.5 * F * Q3],
                    dtype=float)

import numpy as np


def reduced_potentials(phi: float, lam: float, beta_eps_sw: float,
                               alpha: float) -> "np.ndarray":
    import numpy as np
    B = effective_well_depth(beta_eps_sw, alpha)
    hs = hard_sphere_reference(phi)
    mf = mean_field_term(phi, lam, B)
    r2 = second_order_term(phi, lam, B)
    phi = float(phi)
    a = hs[0] + mf[0] + r2[0]
    mu = hs[1] + mf[1] + r2[1]
    dmu = hs[3] + mf[2] + r2[2]
    d2mu = hs[4] + mf[3] + r2[3]
    return np.array([a, mu, phi * mu - a, dmu, d2mu], dtype=float)

import numpy as np


def _bisect_root(f, lo, hi, iters=300):
    """Bisection refined to machine precision on a bracketed sign change."""
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return float(lo)
    if fhi == 0.0:
        return float(hi)
    if flo * fhi > 0.0:
        raise ValueError("no sign change in bracket")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0:
            return float(mid)
        if flo * fm < 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if (hi - lo) <= 5e-17 * max(1.0, abs(hi)):
            break
    return float(0.5 * (lo + hi))


def _validate_state(lam, eps_sw, alpha, temperature=1.0):
    """Shared validation for the physical state point."""
    import numpy as np
    for v in (lam, eps_sw, alpha, temperature):
        if not np.isfinite(v):
            raise ValueError("all state arguments must be finite")
    if float(lam) <= 1.0:
        raise ValueError("lam must be > 1")
    if float(eps_sw) <= 0.0:
        raise ValueError("eps_sw must be > 0")
    if not (0.0 < float(alpha) <= 1.0):
        raise ValueError("alpha must satisfy 0 < alpha <= 1")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be > 0")


def _spinodal_roots(T, lam, eps_sw, alpha, grid):
    """Inner and outer spinodal volume fractions at temperature T, or None above T_c."""
    import numpy as np
    x = float(eps_sw) / float(T)
    f = lambda p: reduced_potentials(p, lam, x, alpha)[3]
    v = np.array([f(p) for p in grid], dtype=float)
    idx = np.where(np.diff(np.sign(v)) != 0)[0]
    if idx.size < 2:
        return None
    return (_bisect_root(f, float(grid[int(idx[0])]), float(grid[int(idx[0]) + 1])),
            _bisect_root(f, float(grid[int(idx[-1])]), float(grid[int(idx[-1]) + 1])))


def _critical_residual(pc, T, lam, eps_sw, alpha):
    """The two critical conditions: first and second phi-derivatives of the potential."""
    import numpy as np
    v = reduced_potentials(pc, lam, float(eps_sw) / float(T), alpha)
    return np.array([v[3], v[4]], dtype=float)


def critical_point(lam: float, eps_sw: float, alpha: float) -> "np.ndarray":
    import numpy as np
    _validate_state(lam, eps_sw, alpha)
    eps_sw = float(eps_sw)
    grid = np.linspace(1e-6, 0.70, 2001)
    lo, hi = eps_sw / 50.0, 5.0 * eps_sw
    if _spinodal_roots(lo, lam, eps_sw, alpha, grid) is None:
        raise ValueError("no spinodal at the lower temperature bracket")
    if _spinodal_roots(hi, lam, eps_sw, alpha, grid) is not None:
        raise ValueError("spinodal still present at the upper temperature bracket")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _spinodal_roots(mid, lam, eps_sw, alpha, grid) is not None:
            lo = mid
        else:
            hi = mid
        if (hi - lo) <= 1e-13 * max(1.0, hi):
            break
    r = _spinodal_roots(lo, lam, eps_sw, alpha, grid)
    phi_c, t_c = 0.5 * (r[0] + r[1]), lo

    # polish on the two critical conditions so the answer is scan-independent
    hp, ht = 1e-7, 1e-5
    for _ in range(80):
        f0 = _critical_residual(phi_c, t_c, lam, eps_sw, alpha)
        jac = np.empty((2, 2))
        jac[:, 0] = (_critical_residual(phi_c + hp, t_c, lam, eps_sw, alpha)
                     - _critical_residual(phi_c - hp, t_c, lam, eps_sw, alpha)) / (2.0 * hp)
        jac[:, 1] = (_critical_residual(phi_c, t_c + ht, lam, eps_sw, alpha)
                     - _critical_residual(phi_c, t_c - ht, lam, eps_sw, alpha)) / (2.0 * ht)
        try:
            step = np.linalg.solve(jac, f0)
        except np.linalg.LinAlgError:
            break
        phi_c -= step[0]
        t_c -= step[1]
        if abs(step[0]) < 1e-15 and abs(step[1]) < 1e-12:
            break
    return np.array([float(phi_c), float(t_c)], dtype=float)

import numpy as np




def _binodal_outer_roots(m, lam, x, alpha, grid, i1, i2):
    """The dilute and dense volume fractions whose chemical potential equals m."""
    mu_of = lambda p: reduced_potentials(p, lam, x, alpha)[1]
    pI = _bisect_root(lambda p: mu_of(p) - m, float(grid[0]), float(grid[i1]))
    pII = _bisect_root(lambda p: mu_of(p) - m, float(grid[i2]), float(grid[-1]))
    return pI, pII


def _binodal_pressure_gap(m, lam, x, alpha, grid, i1, i2):
    """Osmotic-pressure difference between the two branches at equal chemical potential."""
    pi_of = lambda p: reduced_potentials(p, lam, x, alpha)[2]
    pI, pII = _binodal_outer_roots(m, lam, x, alpha, grid, i1, i2)
    return pi_of(pII) - pi_of(pI)


def llps_binodal(lam: float, eps_sw: float, alpha: float,
                         temperature: float) -> "np.ndarray":
    import numpy as np
    _validate_state(lam, eps_sw, alpha, temperature)
    x = float(eps_sw) / float(temperature)
    mu_of = lambda p: reduced_potentials(p, lam, x, alpha)[1]

    grid = np.linspace(1e-8, 0.74, 4001)
    mus = np.array([mu_of(p) for p in grid], dtype=float)
    d = np.diff(mus)
    dec = np.where(d < 0.0)[0]
    if dec.size == 0:
        raise ValueError("no van der Waals loop: state is at or above the critical temperature")
    i1 = int(dec[0])
    inc = np.where(d[i1:] > 0.0)[0]
    if inc.size == 0:
        raise ValueError("no van der Waals loop: state is at or above the critical temperature")
    i2 = i1 + int(inc[0])
    mu_hi, mu_lo = float(mus[i1]), float(mus[i2])
    if not (mu_lo < mu_hi):
        raise ValueError("degenerate loop: state is at or above the critical temperature")

    args = (lam, x, alpha, grid, i1, i2)
    pad = 1e-10 * max(1.0, abs(mu_hi - mu_lo))
    m_star = _bisect_root(lambda m: _binodal_pressure_gap(m, *args), mu_lo + pad, mu_hi - pad)
    pI, pII = _binodal_outer_roots(m_star, *args)
    if not (pII - pI > 1e-6):
        raise ValueError("no distinct two-phase solution at this state point")
    return np.array([pI, pII], dtype=float)

import numpy as np


def _crystal_chemical_potential(temperature, n_s, eps_s, ln_omega_s):
    """Reduced chemical potential of the incompressible crystal (mu0 = 0)."""
    return -float(n_s) * (float(eps_s) / float(temperature)) / 2.0 - float(ln_omega_s)


def crystal_solubility(lam: float, eps_sw: float, alpha: float, temperature: float,
                               n_s: float, eps_s: float, ln_omega_s: float) -> float:
    import numpy as np
    _validate_state(lam, eps_sw, alpha, temperature)
    for v in (n_s, eps_s, ln_omega_s):
        if not np.isfinite(v):
            raise ValueError("crystal parameters must be finite")
    if float(n_s) <= 0.0:
        raise ValueError("n_s must be > 0")
    if float(eps_s) <= 0.0:
        raise ValueError("eps_s must be > 0")

    x = float(eps_sw) / float(temperature)
    mu_s = _crystal_chemical_potential(temperature, n_s, eps_s, ln_omega_s)
    resid = lambda p: reduced_potentials(p, lam, x, alpha)[1] - mu_s

    grid = np.linspace(1e-12, 0.70, 4001)
    f = np.array([resid(p) for p in grid], dtype=float)
    idx = np.where(np.diff(np.sign(f)) != 0)[0]
    if idx.size == 0:
        raise ValueError("no solubility root in (0, 0.70)")
    i = int(idx[0])
    phi_s = float(_bisect_root(resid, float(grid[i]), float(grid[i + 1])))
    state = reduced_potentials(phi_s, lam, x, alpha)
    if not np.isfinite(state[3]) or float(state[3]) <= 0.0:
        raise ValueError("no locally stable solubility root in (0, 0.70)")
    spinodal_grid = np.linspace(1e-6, 0.70, 2001)
    spinodals = _spinodal_roots(temperature, lam, eps_sw, alpha, spinodal_grid)
    if spinodals is not None and phi_s >= float(spinodals[0]):
        raise ValueError("no solubility root on the stable dilute branch")
    return phi_s

import numpy as np


def crystallization_driving_force(lam: float, eps_sw: float, alpha: float,
                                          temperature: float, n_s: float, eps_s: float,
                                          ln_omega_s: float) -> float:
    import numpy as np
    phi_c, t_c = critical_point(lam, eps_sw, alpha)
    if float(temperature) >= t_c:
        raise ValueError("temperature is at or above the critical temperature; "
                         "no protein-rich coexisting phase exists")
    binodal = llps_binodal(lam, eps_sw, alpha, temperature)
    phi_dense = float(binodal[1])
    phi_sol = crystal_solubility(lam, eps_sw, alpha, temperature,
                                         n_s, eps_s, ln_omega_s)
    x = float(eps_sw) / float(temperature)
    mu_dense = reduced_potentials(phi_dense, lam, x, alpha)[1]
    mu_sol = reduced_potentials(phi_sol, lam, x, alpha)[1]
    return float(mu_dense - mu_sol)
SCICODE_GOLD_EOF
