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


def _Q():
    return 1.602176634e-19


def _KB():
    return 1.380649e-23


def _E0():
    return 8.8541878128e-12


def device_parameters(
    thickness: float,
    permittivity: float,
    temperature: float,
    nc: float,
    nv: float,
    gap: float,
    barrier_anode: float,
    barrier_cathode: float,
    mobility_n: float,
    mobility_p: float,
    langevin_factor: float,
    generation_rate: float,
) -> "np.ndarray":
    """Parameter vector (SI) of the undoped thin-film cell with symmetric or asymmetric injection barriers."""
    d_nm = float(thickness)
    eps = float(permittivity)
    T = float(temperature)
    Nc = float(nc)
    Nv = float(nv)
    Eg = float(gap)

    pa = float(barrier_anode)
    pc = float(barrier_cathode)
    mun = float(mobility_n)
    mup = float(mobility_p)
    gam = float(langevin_factor)
    Gex = float(generation_rate)

    if not (
        d_nm > 0
        and eps > 0
        and T > 0
        and Nc > 0
        and Nv > 0
        and Eg > 0
        and mun > 0
        and mup > 0
        and gam >= 0
        and Gex >= 0
    ):
        raise ValueError(
            "thickness, permittivity, temperature, densities of states, gap "
            "and mobilities must be positive; the reduction factor and "
            "generation rate non-negative"
        )

    if not (0.0 <= pa and 0.0 <= pc and pa + pc < Eg):
        raise ValueError(
            "injection barriers must be non-negative and sum to less than the gap"
        )

    VT = _KB() * T / _Q()
    d = d_nm * 1e-9
    ee = eps * _E0()

    Nc_m = Nc * 1e6
    Nv_m = Nv * 1e6

    p_an = Nv_m * np.exp(-pa / VT)
    n_an = Nc_m * np.exp(-(Eg - pa) / VT)
    n_cat = Nc_m * np.exp(-pc / VT)
    p_cat = Nv_m * np.exp(-(Eg - pc) / VT)

    ni2 = Nc_m * Nv_m * np.exp(-Eg / VT)

    mun_m = mun * 1e-4
    mup_m = mup * 1e-4

    gb = gam * _Q() * (mun_m + mup_m) / ee
    G = Gex * 1e6

    return np.array(
        [
            VT,
            Eg - pa - pc,
            d,
            ee,
            p_an,
            n_an,
            n_cat,
            p_cat,
            ni2,
            gb,
            mun_m * VT,
            mup_m * VT,
            G,
            _Q() * G * d,
        ],
        dtype=float,
    )

import numpy as np


def _Q():
    return 1.602176634e-19


def _KB():
    return 1.380649e-23


def _E0():
    return 8.8541878128e-12


def _besseli1(x):
    """Modified Bessel function I1(x) by its power series."""
    x = float(x)
    term = x / 2.0
    s = term
    k = 0

    while k < 1000:
        k += 1
        term *= (x / 2.0) ** 2 / (k * (k + 1))
        s += term

        if term < 1e-17 * s:
            break

    return s


def onsager_braun_yield(
    field: float,
    zero_field_yield: float,
    permittivity: float,
    temperature: float,
) -> "np.ndarray":
    """Onsager-Braun field-assisted dissociation yield anchored to its zero-field value."""
    F = abs(float(field))
    P0 = float(zero_field_yield)
    eps = float(permittivity)
    T = float(temperature)

    if not (0.0 < P0 <= 1.0) or eps <= 0 or T <= 0:
        raise ValueError(
            "the zero-field yield must lie in (0, 1], "
            "the permittivity and temperature must be positive"
        )

    bF = (
        _Q() ** 3
        * F
        / (8.0 * np.pi * eps * _E0() * (_KB() * T) ** 2)
    )

    if bF > 0.0:
        u = np.sqrt(2.0 * bF)
        f = _besseli1(2.0 * u) / u
    else:
        f = 1.0

    Pgen = P0 * f / (P0 * f + 1.0 - P0)

    return np.array([Pgen, f, bF], dtype=float)

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19


def _bernoulli(x):
    """B(x) = x / (exp(x) - 1) with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-8
    out[s] = 1.0 - x[s] / 2.0 + x[s] ** 2 / 12.0
    xb = x[~s]; out[~s] = xb / np.expm1(xb)
    return out

def _dbernoulli(x):
    """dB/dx with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-5
    out[s] = -0.5 + x[s] / 6.0
    xb = x[~s]; ex = np.expm1(xb); out[~s] = (ex - xb * (ex + 1.0)) / ex ** 2
    return out

def _block_tridiagonal_solve(A, B, C, r):
    """Solve the block tridiagonal system with diagonal blocks A[i] (k x k), super-diagonal B[i] (row i to i+1) and
    sub-diagonal C[i] (row i to i-1) for the right-hand side r (M, k), through a LAPACK banded solve."""
    M, k, _ = A.shape; n = k * M; u = 2 * k - 1
    ab = np.zeros((2 * u + 1, n)); i = np.arange(M)
    for rr in range(k):
        for cc in range(k):
            rows = k * i + rr; cols = k * i + cc; ab[u + rows - cols, cols] = A[:, rr, cc]
            rows = k * i[:-1] + rr; cols = k * (i[:-1] + 1) + cc; ab[u + rows - cols, cols] = B[:-1, rr, cc]
            rows = k * i[1:] + rr; cols = k * (i[1:] - 1) + cc; ab[u + rows - cols, cols] = C[1:, rr, cc]
    return solve_banded((u, u), ab, r.reshape(-1)).reshape(M, k)

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step, with the generation scaled by s."""
    VT, Vbi, d, ee, p_an, n_an, n_cat, p_cat, ni2, gb, Dn, Dp, G, qGd = [float(v) for v in par]
    return dict(VT=VT, Vbi=Vbi, d=d, ee=ee, p_an=p_an, n_an=n_an, n_cat=n_cat, p_cat=p_cat, ni2=ni2, gamma=gb, Dn=Dn, Dp=Dp, G=G, qGd=qGd)

def _dd_unpack(state, prm, V):
    """Full node arrays (psi in V, n and p in 1/m^3) from the interior state (M, 3) = [psi/VT, ln n, ln p]."""
    VT = prm["VT"]
    psi = np.concatenate([[0.0], state[:, 0] * VT, [prm["Vbi"] - V]])
    n = np.concatenate([[prm["n_an"]], np.exp(state[:, 1]), [prm["n_cat"]]])
    p = np.concatenate([[prm["p_an"]], np.exp(state[:, 2]), [prm["p_cat"]]])
    return psi, n, p

def _dd_fluxes(psi, n, p, prm, h):
    """Scharfetter-Gummel electron and hole current densities (A/m^2) on the N cell midpoints."""
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q() * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q() * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _dd_residual(state, prm, V, h, G):
    """Scaled residuals (M, 3): Poisson in units of VT, continuity per carrier at each interior node; G in 1/(m^3 s)."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    U = prm["gamma"] * (n * p - prm["ni2"]) - G
    Fpsi = ((psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h ** 2 + _Q() / ee * (p[1:-1] - n[1:-1])) * h ** 2 / VT
    Fn = ((Jn[1:] - Jn[:-1]) / h - _Q() * U[1:-1]) * h ** 2 / (_Q() * prm["Dn"] * n[1:-1])
    Fp = ((Jp[1:] - Jp[:-1]) / h + _Q() * U[1:-1]) * h ** 2 / (_Q() * prm["Dp"] * p[1:-1])
    return np.stack([Fpsi, Fn, Fp], axis=1)

def _dd_jacobian(state, prm, V, h, G):
    """Blocks (M,3,3) of the Jacobian of the scaled residual with respect to [psi/VT, ln n, ln p]."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]; M = state.shape[0]; N = M + 1
    dl = np.diff(psi) / VT; Bp = _bernoulli(dl); Bm = _bernoulli(-dl); dBp = _dbernoulli(dl); dBm = -_dbernoulli(-dl)
    c = _Q() * h ** 2 / (ee * VT); an = h ** 2 * prm["gamma"] / prm["Dn"]; ap = h ** 2 * prm["gamma"] / prm["Dp"]
    gn = h ** 2 * G / prm["Dn"]; gp = h ** 2 * G / prm["Dp"]
    i = np.arange(1, N); ni = n[i]; pi_ = p[i]; nip = n[i + 1]; nim = n[i - 1]; pip = p[i + 1]; pim = p[i - 1]
    A = np.zeros((M, 3, 3)); B = np.zeros((M, 3, 3)); C = np.zeros((M, 3, 3))
    A[:, 0, 0] = -2.0; A[:, 0, 1] = -c * ni; A[:, 0, 2] = c * pi_; B[:, 0, 0] = 1.0; C[:, 0, 0] = 1.0
    fn = (nip * Bp[i] - ni * Bm[i]) - (ni * Bp[i - 1] - nim * Bm[i - 1]) - an * (ni * pi_ - prm["ni2"]) + gn
    A[:, 1, 1] = (-Bm[i] - Bp[i - 1]) - an * pi_ - fn / ni
    A[:, 1, 2] = -an * pi_
    B[:, 1, 1] = Bp[i] * nip / ni; C[:, 1, 1] = Bm[i - 1] * nim / ni
    dli = (nip * dBp[i] - ni * dBm[i]) / ni; dlim = -(ni * dBp[i - 1] - nim * dBm[i - 1]) / ni
    A[:, 1, 0] = -dli + dlim; B[:, 1, 0] = dli; C[:, 1, 0] = -dlim
    fp = (pi_ * Bp[i] - pip * Bm[i]) - (pim * Bp[i - 1] - pi_ * Bm[i - 1]) + ap * (ni * pi_ - prm["ni2"]) - gp
    A[:, 2, 2] = (Bp[i] + Bm[i - 1]) + ap * ni - fp / pi_
    A[:, 2, 1] = ap * ni
    B[:, 2, 2] = -Bm[i] * pip / pi_; C[:, 2, 2] = -Bp[i - 1] * pim / pi_
    dpi = (pi_ * dBp[i] - pip * dBm[i]) / pi_; dpim = -(pim * dBp[i - 1] - pi_ * dBm[i - 1]) / pi_
    A[:, 2, 0] = -dpi + dpim; B[:, 2, 0] = dpi; C[:, 2, 0] = -dpim
    return A, B, C

def _dd_equilibrium(prm, N):
    """Dark zero-bias state: Poisson-Boltzmann solution (damped Newton) in the state layout."""
    VT = prm["VT"]; d = prm["d"]; h = d / N; M = N - 1; c = _Q() * h ** 2 / (prm["ee"] * VT)
    psi = np.linspace(0.0, prm["Vbi"], N + 1)
    def _res(ps):
        n = prm["n_an"] * np.exp(ps / VT); p = prm["p_an"] * np.exp(-ps / VT)
        return (ps[2:] - 2.0 * ps[1:-1] + ps[:-2]) / VT + c * (p[1:-1] - n[1:-1]), n, p
    for it in range(300):
        F, n, p = _res(psi); nrm = np.max(np.abs(F))
        if nrm < 1e-13: break
        A = ((-2.0 - c * (p[1:-1] + n[1:-1])) / VT).reshape(M, 1, 1)
        Bq = np.full((M, 1, 1), 1.0 / VT); Cq = np.full((M, 1, 1), 1.0 / VT)
        dpsi = _block_tridiagonal_solve(A, Bq, Cq, -F.reshape(M, 1))[:, 0]
        lam = 1.0
        for _ in range(60):
            pt = psi.copy(); pt[1:-1] += lam * dpsi
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = _res(pt)[0]
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        psi = pt
    n = prm["n_an"] * np.exp(psi / VT); p = prm["p_an"] * np.exp(-psi / VT)
    return np.stack([psi[1:-1] / VT, np.log(n[1:-1]), np.log(p[1:-1])], axis=1)

def _dd_newton(state, prm, V, h, G, tol=1e-12, maxit=100):
    """Damped Newton on the scaled residual; stops below tol or when a full line search no longer reduces it."""
    U = state.copy()
    for it in range(maxit):
        F = _dd_residual(U, prm, V, h, G); nrm = np.max(np.abs(F))
        if nrm < tol: return U, nrm
        A, B, C = _dd_jacobian(U, prm, V, h, G)
        dU = _block_tridiagonal_solve(A, B, C, -F); lam = 1.0; ok = False
        for _ in range(40):
            Ut = U + lam * dU
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = _dd_residual(Ut, prm, V, h, G)
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: ok = True; break
            lam *= 0.5
        if not ok: return U, nrm
        U = Ut
    return U, nrm

def _dd_solve(prm, N, V, G, state0=None, V0=0.0, dvmax=0.05):
    """Steady state at bias V under the generation rate G: from the dark equilibrium with the generation ramped in at
    zero bias and continuation in steps of at most dvmax, or from a given state at bias V0."""
    h = prm["d"] / N
    if state0 is None:
        U = _dd_equilibrium(prm, N); V0 = 0.0
        if G > 0.0:
            for f in (1e-3, 1e-2, 1e-1, 0.3, 1.0):
                U, nrm = _dd_newton(U, prm, 0.0, h, G * f)
                if not nrm < 1e-10:
                    raise ValueError("generation ramp did not converge")
    else:
        U = state0.copy()
    nsteps = max(1, int(np.ceil(abs(V - V0) / dvmax - 1e-12)))
    for k in range(1, nsteps + 1):
        Vk = V0 + (V - V0) * k / nsteps
        U, nrm = _dd_newton(U, prm, Vk, h, G)
        if not nrm < 1e-10:
            raise ValueError("drift-diffusion solve did not converge at V = %g" % Vk)
    return U

def steady_state(voltage: float, generation_scale: float, intervals: int, par: "np.ndarray", start: "tuple[np.ndarray, float] | None" = None) -> "np.ndarray":
    """Converged steady state at bias V with the generation rate scaled by s (0 = dark), as [psi/VT, ln n, ln p];
    optionally continued from a converged state at another bias, start = (state, voltage0)."""
    par = np.asarray(par, dtype=float).ravel(); N = int(intervals); V = float(voltage); s = float(generation_scale)
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    if s < 0.0:
        raise ValueError("the generation scale must be non-negative")
    prm = _dd_params(par)
    if not (V < prm["Vbi"] + 0.5):
        raise ValueError("bias must lie below the built-in voltage plus 0.5 V")
    G = prm["G"] * s
    if start is None:
        U = _dd_solve(prm, N, V, G)
    else:
        st = np.asarray(start[0], dtype=float); V0 = float(start[1])
        if st.shape != (N + 1, 3) or abs(st[0, 0]) > 1e-9 or abs(st[-1, 0] * prm["VT"] - (prm["Vbi"] - V0)) > 1e-9:
            raise ValueError("start must be an (N + 1, 3) state satisfying the contact potentials at its bias")
        U = _dd_solve(prm, N, V, G, state0=st[1:-1].copy(), V0=V0)
    psi, n, p = _dd_unpack(U, prm, V)
    return np.stack([psi / prm["VT"], np.log(n), np.log(p)], axis=1)

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19


def _bernoulli(x):
    """B(x) = x / (exp(x) - 1) with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-8
    out[s] = 1.0 - x[s] / 2.0 + x[s] ** 2 / 12.0
    xb = x[~s]; out[~s] = xb / np.expm1(xb)
    return out

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step, with the generation scaled by s."""
    VT, Vbi, d, ee, p_an, n_an, n_cat, p_cat, ni2, gb, Dn, Dp, G, qGd = [float(v) for v in par]
    return dict(VT=VT, Vbi=Vbi, d=d, ee=ee, p_an=p_an, n_an=n_an, n_cat=n_cat, p_cat=p_cat, ni2=ni2, gamma=gb, Dn=Dn, Dp=Dp, G=G, qGd=qGd)

def _dd_unpack(state, prm, V):
    """Full node arrays (psi in V, n and p in 1/m^3) from the interior state (M, 3) = [psi/VT, ln n, ln p]."""
    VT = prm["VT"]
    psi = np.concatenate([[0.0], state[:, 0] * VT, [prm["Vbi"] - V]])
    n = np.concatenate([[prm["n_an"]], np.exp(state[:, 1]), [prm["n_cat"]]])
    p = np.concatenate([[prm["p_an"]], np.exp(state[:, 2]), [prm["p_cat"]]])
    return psi, n, p

def _dd_fluxes(psi, n, p, prm, h):
    """Scharfetter-Gummel electron and hole current densities (A/m^2) on the N cell midpoints."""
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q() * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q() * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _dd_residual(state, prm, V, h, G):
    """Scaled residuals (M, 3): Poisson in units of VT, continuity per carrier at each interior node; G in 1/(m^3 s)."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    U = prm["gamma"] * (n * p - prm["ni2"]) - G
    Fpsi = ((psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h ** 2 + _Q() / ee * (p[1:-1] - n[1:-1])) * h ** 2 / VT
    Fn = ((Jn[1:] - Jn[:-1]) / h - _Q() * U[1:-1]) * h ** 2 / (_Q() * prm["Dn"] * n[1:-1])
    Fp = ((Jp[1:] - Jp[:-1]) / h + _Q() * U[1:-1]) * h ** 2 / (_Q() * prm["Dp"] * p[1:-1])
    return np.stack([Fpsi, Fn, Fp], axis=1)

def _dd_current(state, prm, V, N):
    """Total current density (A/m^2) at the mid-cell, with the SG fluxes of the full node arrays."""
    h = prm["d"] / N; psi, n, p = _dd_unpack(state, prm, V)
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    return float((Jn + Jp)[N // 2]), Jn, Jp, psi, n, p

def loss_budget(state_light: "np.ndarray", state_dark: "np.ndarray", voltage: float, generation_scale: float, par: "np.ndarray") -> "np.ndarray":
    """Currents and the exact budget of the photogenerated carriers at one bias: collected, lost to first-order and
    second-order bulk recombination, and extracted at the wrong contacts; all in units of the generation current."""
    sl = np.asarray(state_light, dtype=float); sd = np.asarray(state_dark, dtype=float); par = np.asarray(par, dtype=float).ravel()
    V = float(voltage); s = float(generation_scale)
    if sl.ndim != 2 or sl.shape[1] != 3 or sl.shape[0] < 5 or sl.shape != sd.shape:
        raise ValueError("the two states must be (N + 1, 3) arrays of the same shape")
    if par.size != 14 or s <= 0.0:
        raise ValueError("par must be the 14-entry parameter vector and the generation scale positive")
    N = sl.shape[0] - 1; prm = _dd_params(par); h = prm["d"] / N; G = prm["G"] * s
    for st in (sl, sd):
        if abs(st[0, 0]) > 1e-9 or abs(st[-1, 0] * prm["VT"] - (prm["Vbi"] - V)) > 1e-9:
            raise ValueError("a state does not satisfy the contact potentials at this bias")
    Ul = sl[1:-1].copy(); Ud = sd[1:-1].copy()
    if np.max(np.abs(_dd_residual(Ul, prm, V, h, G))) > 1e-9 or np.max(np.abs(_dd_residual(Ud, prm, V, h, 0.0))) > 1e-9:
        raise ValueError("a state is not a converged steady state at this bias and generation")
    Jl, Jnl, Jpl, psil, nl, pl = _dd_current(Ul, prm, V, N)
    Jd, Jnd, Jpd, psid, nd, pd = _dd_current(Ud, prm, V, N)
    qGd = _Q() * G * prm["d"]
    dn = nl - nd; dp = pl - pd
    R1 = prm["gamma"] * (nd * dp + pd * dn); R2 = prm["gamma"] * dn * dp
    first = _Q() * h * R1[1:N].sum() / qGd; second = _Q() * h * R2[1:N].sum() / qGd
    # contact currents of the excess carriers: the first-interval flux plus the generation of the boundary half-cell
    anode = ((Jnl[0] - Jnd[0]) + _Q() * G * h / 2.0) / qGd        # electrons extracted by the anode
    cathode = ((Jpl[-1] - Jpd[-1]) + _Q() * G * h / 2.0) / qGd    # holes extracted by the cathode
    collected = -(Jl - Jd) / qGd
    return np.array([Jl / 10.0, Jd / 10.0, collected, first, second, anode, cathode], dtype=float)

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19


def _bernoulli(x):
    """B(x) = x / (exp(x) - 1) with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-8
    out[s] = 1.0 - x[s] / 2.0 + x[s] ** 2 / 12.0
    xb = x[~s]; out[~s] = xb / np.expm1(xb)
    return out

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step, with the generation scaled by s."""
    VT, Vbi, d, ee, p_an, n_an, n_cat, p_cat, ni2, gb, Dn, Dp, G, qGd = [float(v) for v in par]
    return dict(VT=VT, Vbi=Vbi, d=d, ee=ee, p_an=p_an, n_an=n_an, n_cat=n_cat, p_cat=p_cat, ni2=ni2, gamma=gb, Dn=Dn, Dp=Dp, G=G, qGd=qGd)

def _dd_fluxes(psi, n, p, prm, h):
    """Scharfetter-Gummel electron and hole current densities (A/m^2) on the N cell midpoints."""
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q() * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q() * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _Sweep(par, N):
    class _SweepState:
        """Cache of converged states of one cell keyed by (bias, generation scale); every new state is continued from
        the nearest cached one through the third step's oracle."""
        def __init__(self, par, N):
            self.par = np.asarray(par, dtype=float).ravel(); self.prm = _dd_params(self.par); self.N = N; self.cache = {}
    
        def _state(self, V, s):
            key = (round(float(V), 12), round(float(s), 12))
            if key in self.cache: return self.cache[key]
            same = [k for k in self.cache if k[1] == key[1]]
            lit = [k for k in self.cache if k[1] > 0.0]
            if same:
                k0 = min(same, key=lambda k: abs(k[0] - key[0]))
            elif s > 0.0 and lit:
                k0 = min(lit, key=lambda k: abs(k[0] - key[0]))
            else:
                k0 = None
            st = steady_state(V, s, self.N, self.par, None if k0 is None else (self.cache[k0], k0[0]))
            self.cache[key] = st
            return st
    
        def _current(self, V, s):
            st = self._state(V, s); prm = self.prm; h = prm["d"] / self.N
            psi = st[:, 0] * prm["VT"]; n = np.exp(st[:, 1]); p = np.exp(st[:, 2])
            Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
            return float((Jn + Jp)[self.N // 2])
    return _SweepState(par, N)

def jv_characteristics(voltages: "np.ndarray", generation_scale: float, intervals: int, par: "np.ndarray") -> "np.ndarray":
    """Current density (mA/cm^2) at each bias, by continuation outward from zero bias."""
    Vs = np.asarray(voltages, dtype=float).ravel(); par = np.asarray(par, dtype=float).ravel(); N = int(intervals); s = float(generation_scale)
    if Vs.size == 0 or par.size != 14 or N < 4 or N % 2 != 0 or s < 0.0:
        raise ValueError("voltages must be non-empty, par the 14-entry vector, intervals an even integer of at least 4 and the generation scale non-negative")
    prm = _dd_params(par)
    if np.any(Vs >= prm["Vbi"] + 0.5):
        raise ValueError("every bias must lie below the built-in voltage plus 0.5 V")
    sw = _Sweep(par, N)
    out = np.empty(Vs.size)
    order = np.argsort(np.abs(Vs))
    for k in order:
        out[k] = sw._current(Vs[k], s) / 10.0
    return out

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19
def _KB():
    return 1.380649e-23
def _E0():
    return 8.8541878128e-12


def _bernoulli(x):
    """B(x) = x / (exp(x) - 1) with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-8
    out[s] = 1.0 - x[s] / 2.0 + x[s] ** 2 / 12.0
    xb = x[~s]; out[~s] = xb / np.expm1(xb)
    return out

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step, with the generation scaled by s."""
    VT, Vbi, d, ee, p_an, n_an, n_cat, p_cat, ni2, gb, Dn, Dp, G, qGd = [float(v) for v in par]
    return dict(VT=VT, Vbi=Vbi, d=d, ee=ee, p_an=p_an, n_an=n_an, n_cat=n_cat, p_cat=p_cat, ni2=ni2, gamma=gb, Dn=Dn, Dp=Dp, G=G, qGd=qGd)

def _dd_fluxes(psi, n, p, prm, h):
    """Scharfetter-Gummel electron and hole current densities (A/m^2) on the N cell midpoints."""
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q() * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q() * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _Sweep(par, N):
    class _SweepState:
        """Cache of converged states of one cell keyed by (bias, generation scale); every new state is continued from
        the nearest cached one through the third step's oracle."""
        def __init__(self, par, N):
            self.par = np.asarray(par, dtype=float).ravel(); self.prm = _dd_params(self.par); self.N = N; self.cache = {}
    
        def _state(self, V, s):
            key = (round(float(V), 12), round(float(s), 12))
            if key in self.cache: return self.cache[key]
            same = [k for k in self.cache if k[1] == key[1]]
            lit = [k for k in self.cache if k[1] > 0.0]
            if same:
                k0 = min(same, key=lambda k: abs(k[0] - key[0]))
            elif s > 0.0 and lit:
                k0 = min(lit, key=lambda k: abs(k[0] - key[0]))
            else:
                k0 = None
            st = steady_state(V, s, self.N, self.par, None if k0 is None else (self.cache[k0], k0[0]))
            self.cache[key] = st
            return st
    
        def _current(self, V, s):
            st = self._state(V, s); prm = self.prm; h = prm["d"] / self.N
            psi = st[:, 0] * prm["VT"]; n = np.exp(st[:, 1]); p = np.exp(st[:, 2])
            Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
            return float((Jn + Jp)[self.N // 2])
    return _SweepState(par, N)

def _bisect(f, a, b, tol):
    """Root of f on [a, b] (sign change required) to an interval width below tol."""
    fa = f(a); fb = f(b)
    if not (fa * fb < 0):
        raise ValueError("no sign change on the bracket")
    while b - a > tol:
        m = 0.5 * (a + b); fm = f(m)
        if fa * fm <= 0: b, fb = m, fm
        else: a, fa = m, fm
    return 0.5 * (a + b)

def _golden_max(f, a, b, tol):
    """Maximiser of f on [a, b] by golden-section search to an interval width below tol; returns (x, f(x))."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - g * (b - a); d = a + g * (b - a); fc = f(c); fd = f(d)
    while b - a > tol:
        if fc > fd:
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)

def procedure_metrics(zero_field_yield: float, reverse_biases: "np.ndarray", intervals: int, par: "np.ndarray") -> "np.ndarray":
    """The Jph-Veff procedure and the light metrics of one device whose generation carries the Onsager-Braun yield
    evaluated at the nominal field (Vbi - V)/d."""
    P0 = float(zero_field_yield); Vrs = np.asarray(reverse_biases, dtype=float).ravel(); N = int(intervals); par = np.asarray(par, dtype=float).ravel()
    if not (0.0 < P0 <= 1.0) or Vrs.size == 0 or np.any(Vrs >= 0.0) or par.size != 14 or N < 4 or N % 2 != 0:
        raise ValueError("the zero-field yield must lie in (0, 1], the reverse biases must be negative, par the 14-entry vector and intervals an even integer of at least 4")
    prm = _dd_params(par); VT = prm["VT"]; Vbi = prm["Vbi"]; qGd = prm["qGd"]
    eps = prm["ee"] / _E0(); T = VT * _Q() / _KB()
    sw = _Sweep(par, N)
    def _yld(V):
        return 1.0 if P0 >= 1.0 else float(onsager_braun_yield((Vbi - V) / prm["d"], P0, eps, T)[0])
    def _jlight(V): return sw._current(V, _yld(V))
    def _jdark(V): return sw._current(V, 0.0)
    def _jph(V): return _jlight(V) - _jdark(V)
    Jsc = -_jlight(0.0); Jph0 = _jph(0.0)
    V0 = _bisect(_jph, 0.0, Vbi + 0.4, 1e-10)
    Voc = _bisect(_jlight, 0.0, Vbi + 0.4, 1e-10)
    Vm, Pm = _golden_max(lambda V: -V * _jlight(V), 0.0, Voc, 1e-9)
    FF = Pm / (Voc * Jsc)
    Pg_sc = _yld(0.0)
    out = [V0, Jsc / qGd, Voc, Vm, FF, Pg_sc, Jsc / (qGd * Pg_sc)]
    for Vr in Vrs:
        Jsat = _jph(Vr); out += [-Jsat / qGd, Jph0 / Jsat]
    return np.array(out, dtype=float)

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19
def _KB():
    return 1.380649e-23


def _bernoulli(x):
    """B(x) = x / (exp(x) - 1) with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-8
    out[s] = 1.0 - x[s] / 2.0 + x[s] ** 2 / 12.0
    xb = x[~s]; out[~s] = xb / np.expm1(xb)
    return out

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step, with the generation scaled by s."""
    VT, Vbi, d, ee, p_an, n_an, n_cat, p_cat, ni2, gb, Dn, Dp, G, qGd = [float(v) for v in par]
    return dict(VT=VT, Vbi=Vbi, d=d, ee=ee, p_an=p_an, n_an=n_an, n_cat=n_cat, p_cat=p_cat, ni2=ni2, gamma=gb, Dn=Dn, Dp=Dp, G=G, qGd=qGd)

def _dd_fluxes(psi, n, p, prm, h):
    """Scharfetter-Gummel electron and hole current densities (A/m^2) on the N cell midpoints."""
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q() * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q() * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _Sweep(par, N):
    class _SweepState:
        """Cache of converged states of one cell keyed by (bias, generation scale); every new state is continued from
        the nearest cached one through the third step's oracle."""
        def __init__(self, par, N):
            self.par = np.asarray(par, dtype=float).ravel(); self.prm = _dd_params(self.par); self.N = N; self.cache = {}
    
        def _state(self, V, s):
            key = (round(float(V), 12), round(float(s), 12))
            if key in self.cache: return self.cache[key]
            same = [k for k in self.cache if k[1] == key[1]]
            lit = [k for k in self.cache if k[1] > 0.0]
            if same:
                k0 = min(same, key=lambda k: abs(k[0] - key[0]))
            elif s > 0.0 and lit:
                k0 = min(lit, key=lambda k: abs(k[0] - key[0]))
            else:
                k0 = None
            st = steady_state(V, s, self.N, self.par, None if k0 is None else (self.cache[k0], k0[0]))
            self.cache[key] = st
            return st
    
        def _current(self, V, s):
            st = self._state(V, s); prm = self.prm; h = prm["d"] / self.N
            psi = st[:, 0] * prm["VT"]; n = np.exp(st[:, 1]); p = np.exp(st[:, 2])
            Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
            return float((Jn + Jp)[self.N // 2])
    return _SweepState(par, N)

def collection_ceiling(barriers: "np.ndarray", reduction_factors: "np.ndarray", mobility: float, intervals: int, material: "np.ndarray") -> "np.ndarray":
    """Short-circuit collection efficiency against the symmetric injection barrier for a balanced mobility and each
    Langevin reduction factor (columns), with the Sokel-Hughes diffusion-loss form as the last column."""
    Ph = np.asarray(barriers, dtype=float).ravel(); gs = np.asarray(reduction_factors, dtype=float).ravel(); mu = float(mobility)
    N = int(intervals); mat = [float(v) for v in np.asarray(material, dtype=float).ravel()]
    if Ph.size == 0 or gs.size == 0 or mu <= 0 or len(mat) != 7 or N < 4 or N % 2 != 0 or np.any(gs < 0) or np.any(Ph < 0):
        raise ValueError("barriers (non-negative) and reduction factors (non-negative) must be non-empty, the mobility positive, material the 7-entry tuple and intervals an even integer of at least 4")
    d_nm, eps, T, Nc, Nv, Eg, Gex = mat
    if np.any(2.0 * Ph >= Eg):
        raise ValueError("twice the barrier must lie below the gap")
    VT = _KB() * T / _Q()
    out = np.empty((Ph.size, gs.size + 1))
    for i, phi in enumerate(Ph):
        for j, g in enumerate(gs):
            pv = device_parameters(d_nm, eps, T, Nc, Nv, Eg, phi, phi, mu, mu, g, Gex)
            prm = _dd_params(pv); sw = _Sweep(pv, N)
            out[i, j] = -(sw._current(0.0, 1.0) - sw._current(0.0, 0.0)) / prm["qGd"]
        u = (Eg - 2.0 * phi) / VT
        out[i, gs.size] = 1.0 / np.tanh(u / 2.0) - 2.0 / u
    return out

import numpy as np
from scipy.linalg import solve_banded


def _bisect(f, a, b, tol):
    """Root of f on [a, b] (sign change required) to an interval width below tol."""
    fa = f(a); fb = f(b)
    if not (fa * fb < 0):
        raise ValueError("no sign change on the bracket")
    while b - a > tol:
        m = 0.5 * (a + b); fm = f(m)
        if fa * fm <= 0: b, fb = m, fm
        else: a, fa = m, fm
    return 0.5 * (a + b)

def transport_crossover(mobility_ideal: float, mobility_design: float, barrier: float, reduction_factor: float, intervals: int, material: "np.ndarray") -> "np.ndarray":
    """Short-circuit loss _budget against the balanced mobility: the mobility at which the second-order (bimolecular)
    loss equals the sum of the first-order and contact losses, and the budgets at the design and ideal mobilities."""
    mui = float(mobility_ideal); mud = float(mobility_design); phi = float(barrier); g = float(reduction_factor)
    N = int(intervals); mat = [float(v) for v in np.asarray(material, dtype=float).ravel()]
    if mui <= 0 or mud <= 0 or mud >= mui or phi < 0 or g <= 0 or len(mat) != 7 or N < 4 or N % 2 != 0:
        raise ValueError("0 < design mobility < ideal mobility, non-negative barrier, positive reduction factor, material the 7-entry tuple, intervals an even integer of at least 4")
    d_nm, eps, T, Nc, Nv, Eg, Gex = mat
    if 2.0 * phi >= Eg:
        raise ValueError("twice the barrier must lie below the _gap")
    def _budget(mu):
        pv = device_parameters(d_nm, eps, T, Nc, Nv, Eg, phi, phi, mu, mu, g, Gex)
        sl = steady_state(0.0, 1.0, N, pv); sd = steady_state(0.0, 0.0, N, pv)
        return loss_budget(sl, sd, 0.0, 1.0, pv)
    def _gap(lm):
        b = _budget(10.0 ** lm)
        return b[4] - (b[3] + b[5] + b[6])
    lm = _bisect(_gap, np.log10(mud) - 3.0, np.log10(mui), 1e-9)
    bd = _budget(mud); bi = _budget(mui)
    return np.array([lm] + list(bd[2:7]) + list(bi[2:7]), dtype=float)

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19


def _bernoulli(x):
    """B(x) = x / (exp(x) - 1) with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-8
    out[s] = 1.0 - x[s] / 2.0 + x[s] ** 2 / 12.0
    xb = x[~s]; out[~s] = xb / np.expm1(xb)
    return out

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step, with the generation scaled by s."""
    VT, Vbi, d, ee, p_an, n_an, n_cat, p_cat, ni2, gb, Dn, Dp, G, qGd = [float(v) for v in par]
    return dict(VT=VT, Vbi=Vbi, d=d, ee=ee, p_an=p_an, n_an=n_an, n_cat=n_cat, p_cat=p_cat, ni2=ni2, gamma=gb, Dn=Dn, Dp=Dp, G=G, qGd=qGd)

def _dd_fluxes(psi, n, p, prm, h):
    """Scharfetter-Gummel electron and hole current densities (A/m^2) on the N cell midpoints."""
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q() * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q() * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _Sweep(par, N):
    class _SweepState:
        """Cache of converged states of one cell keyed by (bias, generation scale); every new state is continued from
        the nearest cached one through the third step's oracle."""
        def __init__(self, par, N):
            self.par = np.asarray(par, dtype=float).ravel(); self.prm = _dd_params(self.par); self.N = N; self.cache = {}
    
        def _state(self, V, s):
            key = (round(float(V), 12), round(float(s), 12))
            if key in self.cache: return self.cache[key]
            same = [k for k in self.cache if k[1] == key[1]]
            lit = [k for k in self.cache if k[1] > 0.0]
            if same:
                k0 = min(same, key=lambda k: abs(k[0] - key[0]))
            elif s > 0.0 and lit:
                k0 = min(lit, key=lambda k: abs(k[0] - key[0]))
            else:
                k0 = None
            st = steady_state(V, s, self.N, self.par, None if k0 is None else (self.cache[k0], k0[0]))
            self.cache[key] = st
            return st
    
        def _current(self, V, s):
            st = self._state(V, s); prm = self.prm; h = prm["d"] / self.N
            psi = st[:, 0] * prm["VT"]; n = np.exp(st[:, 1]); p = np.exp(st[:, 2])
            Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
            return float((Jn + Jp)[self.N // 2])
    return _SweepState(par, N)

def intensity_dependence(intensities: "np.ndarray", mobilities: "np.ndarray", barrier: float, reduction_factor: float, intervals: int, material: "np.ndarray") -> "np.ndarray":
    """Short-circuit collection efficiency against the illumination intensity for each balanced mobility (rows)."""
    Is = np.asarray(intensities, dtype=float).ravel(); mus = np.asarray(mobilities, dtype=float).ravel()
    phi = float(barrier); g = float(reduction_factor); N = int(intervals); mat = [float(v) for v in np.asarray(material, dtype=float).ravel()]
    if Is.size == 0 or np.any(Is <= 0) or mus.size == 0 or np.any(mus <= 0) or phi < 0 or g < 0 or len(mat) != 7 or N < 4 or N % 2 != 0:
        raise ValueError("positive intensities and mobilities, non-negative barrier and reduction factor, material the 7-entry tuple, intervals an even integer of at least 4")
    d_nm, eps, T, Nc, Nv, Eg, Gex = mat
    if 2.0 * phi >= Eg:
        raise ValueError("twice the barrier must lie below the gap")
    out = np.empty((mus.size, Is.size))
    for i, mu in enumerate(mus):
        pv = device_parameters(d_nm, eps, T, Nc, Nv, Eg, phi, phi, mu, mu, g, Gex)
        prm = _dd_params(pv); sw = _Sweep(pv, N); Jd = sw._current(0.0, 0.0)
        for j, s in enumerate(Is):
            out[i, j] = -(sw._current(0.0, s) - Jd) / (prm["qGd"] * s)
    return out

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19
def _KB():
    return 1.380649e-23
def _E0():
    return 8.8541878128e-12


def _NAMES():
    return ["V0", "eta_sc", "Voc", "Vmpp", "FF", "Pgen_sc", "eta_coll", "Jsat_1", "Papp_1", "Jsat_2", "Papp_2",
             "loss_first", "loss_second", "loss_anode", "loss_cathode"]

def procedure_audit(zero_field_yields: "np.ndarray", reverse_biases: "np.ndarray", barriers: "np.ndarray", reduction_factors: "np.ndarray", mobility_ideal: float, intensities: "np.ndarray", intervals: int, device: "tuple[float, ...]") -> "np.ndarray":
    """Run the chain for every zero-field yield; head row [apparent dissociation probability of the unity-yield device
    at the first reverse bias, the ideal-transport ceiling at the design barrier, the largest ceiling on the barrier grid
    and its barrier, the Sokel-Hughes value at the design barrier, log10 of the crossover mobility, the low-intensity
    collection efficiency at the design and ideal mobilities, J_sc and the two saturation currents of the unity cell in
    mA/cm^2 from the sweep step, the short-circuit yield of the smallest zero-field yield] followed by one row per yield."""
    P0s = np.asarray(zero_field_yields, dtype=float).ravel(); Vrs = np.asarray(reverse_biases, dtype=float).ravel()
    dev = [float(v) for v in np.asarray(device, dtype=float).ravel()]; N = int(intervals)
    if P0s.size == 0 or Vrs.size != 2 or len(dev) != 12 or N < 4 or N % 2 != 0:
        raise ValueError("yields must be non-empty, two reverse biases, device the 12-entry tuple of step 1 and intervals an even integer of at least 4")
    if not np.any(np.abs(P0s - 1.0) < 1e-12):
        raise ValueError("the unity yield must be among the zero-field yields")
    par = device_parameters(*dev)
    d_nm, eps, T, Nc, Nv, Eg, pa, pc, mun, mup, gam, Gex = dev
    material = (d_nm, eps, T, Nc, Nv, Eg, Gex)
    rows = []
    for P0 in P0s:
        m = procedure_metrics(P0, Vrs, N, par)
        s = m[5]
        sl = steady_state(0.0, s, N, par); sd = steady_state(0.0, 0.0, N, par)
        lb = loss_budget(sl, sd, 0.0, s, par)
        rows.append(list(m[:11]) + list(lb[3:7]))
    ceil = collection_ceiling(barriers, reduction_factors, mobility_ideal, N, material)
    gs = np.asarray(reduction_factors, dtype=float).ravel(); Ph = np.asarray(barriers, dtype=float).ravel()
    gcol = int(np.argmin(np.abs(gs - gam))); col = ceil[:, gcol]; im = int(np.argmax(col)); ip = int(np.argmin(np.abs(Ph - pa)))
    cx = transport_crossover(mobility_ideal, mun, pa, gam, N, material)
    inten = intensity_dependence(intensities, [mun, mobility_ideal], pa, gam, N, material)
    iu = int(np.argmin(np.abs(P0s - 1.0)))
    jl = jv_characteristics([Vrs[1], Vrs[0], 0.0], 1.0, N, par); jd = jv_characteristics([Vrs[1], Vrs[0], 0.0], 0.0, N, par)
    Jsc = -jl[2]; Jsat1 = -(jl[1] - jd[1]); Jsat2 = -(jl[0] - jd[0])
    pmin = float(P0s.min()); VT = par[0]; T = VT * _Q() / _KB(); eps = par[3] / _E0()
    pg_min = float(onsager_braun_yield(par[1] / par[2], pmin, eps, T)[0]) if pmin < 1.0 else 1.0
    head = [rows[iu][8], col[ip], col[im], Ph[im], ceil[ip, gs.size], cx[0], inten[0, 0], inten[1, 0], Jsc, Jsat1, Jsat2, pg_min]
    T_ = np.zeros((len(rows) + 1, len(_NAMES())))
    T_[0, :len(head)] = head
    T_[1:] = np.array(rows)
    return T_
SCICODE_GOLD_EOF
