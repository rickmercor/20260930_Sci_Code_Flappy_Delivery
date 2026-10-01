#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def displacement_levels(tenors, atm_vols):
    import numpy as np
    t = np.asarray(tenors, dtype=float)
    v = np.asarray(atm_vols, dtype=float)
    if t.ndim != 1 or v.ndim != 1 or len(t) != len(v) + 1 or len(v) < 1:
        raise ValueError("tenors must hold tau_0 and one entry per ATM volatility")
    if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("tenors must start at 0 and increase strictly")
    if np.any(v <= 0.0):
        raise ValueError("ATM volatilities must be positive")
    sigma0 = float(v[0])
    out = np.zeros(len(v))
    for k in range(1, len(v)):
        inc = v[k] ** 2 * t[k + 1] - v[k - 1] ** 2 * t[k]
        if inc < 0.0:
            raise ValueError("calendar arbitrage: total implied variance decreases")
        out[k] = -sigma0 + np.sqrt(inc / (t[k + 1] - t[k]))
    return out

def displacement_functionals(tenors, levels, tau):
    import numpy as np
    t = np.asarray(tenors, dtype=float)
    lv = np.asarray(levels, dtype=float)
    if t.ndim != 1 or lv.ndim != 1 or len(t) != len(lv) + 1:
        raise ValueError("tenors must hold tau_0 and one entry per level")
    if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("tenors must start at 0 and increase strictly")
    hits = np.nonzero(np.isclose(t, tau, rtol=0.0, atol=1e-15))[0]
    if len(hits) != 1 or hits[0] == 0:
        raise ValueError("tau must be one of the positive tenor grid points")
    n = int(hits[0])
    P = np.polynomial.polynomial
    # running values of the iterated integrals at the left end of the current interval
    I1 = 0.0; I2 = 0.0; K1 = 0.0; L1 = 0.0; M1 = 0.0
    A0 = A1 = A2 = A3 = A4 = B2 = B3a = B3b = 0.0
    for k in range(n):
        t0, h, l = t[k], t[k + 1] - t[k], lv[k]
        phi = np.array([l])                                          # phi~(s)          (in x = s - t0)
        pI1 = np.array([I1, l])                                      # int_0^s phi~
        pI2 = P.polyadd(np.array([I2]), P.polyint(pI1))              # int_0^s int_0^s1 phi~
        pK1 = P.polyadd(np.array([K1]), P.polyint(P.polymul(pI1, phi)))   # int_0^s (int_0^s1 phi~) phi~(s1)
        s_poly = np.array([t0, 1.0])                                 # s
        pL1 = P.polyadd(np.array([L1]), P.polyint(P.polymul(s_poly, phi)))  # int_0^s s1 phi~(s1)
        pM1 = P.polyadd(np.array([M1]), P.polyint(P.polymul(pK1, phi)))     # int_0^s K1(s1) phi~(s1)
        def integ(p):
            return float(P.polyval(h, P.polyint(p)))
        A0 += integ(P.polymul(phi, phi))          # int phi~^2
        A1 += integ(P.polymul(phi, pI1))          # int phi~(s) (int_0^s phi~)
        A2 += integ(pI1)                          # int int_0^s phi~
        A3 += integ(P.polymul(s_poly, phi))       # int s phi~(s)
        A4 += integ(pI2)                          # int int_0^s int_0^s1 phi~
        B2 += integ(P.polymul(pK1, phi))          # int K1(s) phi~(s)
        B3a += integ(P.polymul(pL1, phi))         # int L1(s) phi~(s)
        B3b += integ(P.polymul(pM1, phi))         # int M1(s) phi~(s)
        I1, I2, K1, L1, M1 = (float(P.polyval(h, pI1)), float(P.polyval(h, pI2)),
                              float(P.polyval(h, pK1)), float(P.polyval(h, pL1)),
                              float(P.polyval(h, pM1)))
    return np.array([A0, A1, A2, A3, A4, B2, B3a, B3b], dtype=float)

def expansion_blocks(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, functionals):
    import numpy as np
    F = np.asarray(functionals, dtype=float)
    if F.shape != (8,):
        raise ValueError("functionals must be the length-8 array of displacement_functionals")
    if tau <= 0.0 or sigma0 <= 0.0 or vov < 0.0 or not (-1.0 <= rho <= 1.0):
        raise ValueError("invalid model coefficients")
    A0, A1, A2, A3, A4, B2, B3a, B3b = [float(x) for x in F]
    u = np.asarray(u, dtype=complex)
    if u.ndim > 1:
        raise ValueError("u must be a scalar or a one-dimensional array of frequencies")
    u2 = u * u
    beta0 = vov * rho
    betap0 = vov * np.sqrt(1.0 - rho * rho)
    skew = -1j * u ** 3 * beta0 / (sigma0 * tau ** 1.5) * A1
    dblk = -u2 * delta0 / (sigma0 * tau) * A2
    ablk = -u2 * alpha0 / (sigma0 * tau) * A3
    eblk = u ** 4 * eta0 / (sigma0 * tau ** 2) * A4
    bblk = -beta0 ** 2 * u2 / (8.0 * sigma0 ** 2 * tau) * (
        2.0 * tau ** 2 + 4.0 * u2 * A1 - 24.0 * u2 / tau * B2
        - 12.0 * u2 / tau * (B3a - 2.0 * u2 / tau * B3b))
    pblk = -betap0 ** 2 * u2 / (2.0 * sigma0 ** 2 * tau) * (tau ** 2 / 2.0 - 2.0 * u2 / tau * B3a)
    return np.array([skew, dblk, ablk, eblk, bblk, pblk], dtype=complex)

def continuous_cf_expansion(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, tenors, shifts):
    import numpy as np
    sh = np.asarray(shifts, dtype=float)
    if sigma0 <= 0.0:
        raise ValueError("sigma0 must be positive")
    levels = 1.0 + sh / sigma0
    F = displacement_functionals(tenors, levels, tau)
    blocks = expansion_blocks(u, tau, sigma0, vov, rho, eta0, alpha0, delta0, F)
    u = np.asarray(u, dtype=complex)
    lead = np.exp(-u * u / 2.0 * F[0] / tau)
    out = lead * (1.0 + np.sum(blocks, axis=0))
    return complex(out) if out.ndim == 0 else np.asarray(out, dtype=complex)

def fourier_call_price(model, s0, strike, tau, n_nodes, u_max):
    import numpy as np
    if s0 <= 0.0 or strike <= 0.0 or tau <= 0.0:
        raise ValueError("s0, strike and tau must be positive")
    n_nodes = int(n_nodes)
    if n_nodes < 1 or u_max <= 0.0:
        raise ValueError("n_nodes must be positive and u_max positive")
    sigma0 = float(model["sigma0"])
    args = (sigma0, model["vov"], model["rho"], model["eta0"], model["alpha0"], model["delta0"],
            model["tenors"], model["shifts"])
    du = u_max / n_nodes
    grid = (np.arange(n_nodes) + 0.5) * du
    sq = sigma0 * np.sqrt(tau)
    d2 = (np.log(s0) - np.log(strike) - 0.5 * sigma0 * sigma0 * tau) / sq
    shift = 1j * sq
    norm = continuous_cf_expansion(-shift, tau, *args)
    psi_plain = continuous_cf_expansion(grid, tau, *args)
    psi_shift = continuous_cf_expansion(grid - shift, tau, *args)
    phase = np.exp(1j * grid * d2)
    p1 = 0.5 + du / np.pi * float(np.sum((phase * psi_shift / (1j * grid * norm)).real))
    p2 = 0.5 + du / np.pi * float(np.sum((phase * psi_plain / (1j * grid)).real))
    return float(s0 * p1 - strike * p2)

def implied_volatility(price, s0, strike, tau):
    import numpy as np
    from math import erf, sqrt, log
    if s0 <= 0.0 or strike <= 0.0 or tau <= 0.0:
        raise ValueError("s0, strike and tau must be positive")
    if not (max(s0 - strike, 0.0) <= price <= s0):
        raise ValueError("price violates the zero-rate no-arbitrage bounds")
    def ncdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    def bs_call(sig):
        d1 = (log(s0 / strike) + 0.5 * sig * sig * tau) / (sig * sqrt(tau))
        return s0 * ncdf(d1) - strike * ncdf(d1 - sig * sqrt(tau))
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs_call(mid) > price:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))

def risk_reversal(config):
    import numpy as np
    tenors = [float(x) for x in config["tenors"]]
    atm = [float(x) for x in config["atm_vols"]]
    s0 = float(config["s0"])
    n_nodes, u_max = int(config["n_nodes"]), float(config["u_max"])
    if s0 <= 0.0 or n_nodes < 1 or u_max <= 0.0:
        raise ValueError("s0 must be positive, n_nodes at least 1 and u_max positive")
    shifts = displacement_levels(tenors, atm)
    sigma0 = atm[0]
    tau = tenors[-1]
    # consistency gate: at zero frequency every correction block vanishes and the expansion
    # equals one, so the assembled pieces are checked before any price is formed
    levels = 1.0 + np.asarray(shifts, dtype=float) / sigma0
    F = displacement_functionals(tenors, levels, tau)
    if F[0] <= 0.0:
        raise ValueError("the squared displacement profile must have positive integral")
    blocks = expansion_blocks(0.0, tau, sigma0, config["vov"], config["rho"], config["eta0"],
                                      config["alpha0"], config["delta0"], F)
    psi0 = continuous_cf_expansion(0.0, tau, sigma0, config["vov"], config["rho"], config["eta0"],
                                           config["alpha0"], config["delta0"], tenors, shifts)
    if np.max(np.abs(blocks)) != 0.0 or abs(psi0 - 1.0) > 1e-12:
        raise ValueError("expansion does not equal one at zero frequency")
    model = dict(sigma0=sigma0, vov=float(config["vov"]), rho=float(config["rho"]),
                 eta0=float(config["eta0"]), alpha0=float(config["alpha0"]),
                 delta0=float(config["delta0"]), tenors=tenors, shifts=shifts)
    ivs = []
    for m in (float(config["m_put"]), float(config["m_call"])):
        strike = s0 * np.exp(m * atm[-1] * np.sqrt(tau))
        price = fourier_call_price(model, s0, strike, tau, n_nodes, u_max)
        ivs.append(implied_volatility(price, s0, strike, tau))
    return float(ivs[0] - ivs[1])
SCICODE_GOLD_EOF
