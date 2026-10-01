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


_Q = 1.602176634e-19

_KB = 1.380649e-23

_E0 = 8.8541878128e-12

def diode_parameters(thickness: float, permittivity: float, temperature: float, nc: float, nv: float, gap: float, barrier_anode: float, barrier_cathode: float, mobility_n: float, mobility_p: float, langevin_factor: float) -> "np.ndarray":
    d_nm = float(thickness); eps = float(permittivity); T = float(temperature); Nc = float(nc); Nv = float(nv); Eg = float(gap)
    pa = float(barrier_anode); pc = float(barrier_cathode); mun = float(mobility_n); mup = float(mobility_p); zeta = float(langevin_factor)
    if not (d_nm > 0.0) or not (eps > 0.0) or not (T > 0.0) or not (Nc > 0.0) or not (Nv > 0.0):
        raise ValueError("thickness, permittivity, temperature and densities of states must be positive")
    if not (pa >= 0.0) or not (pc >= 0.0) or not (Eg > pa + pc):
        raise ValueError("barriers must be non-negative and smaller than the gap in total")
    if not (mun > 0.0) or not (mup > 0.0) or not (zeta > 0.0):
        raise ValueError("mobilities and the Langevin factor must be positive")
    VT = _KB * T / _Q
    p_an = Nv * np.exp(-pa / VT); n_cat = Nc * np.exp(-pc / VT)
    n_an = Nc * np.exp(-(Eg - pa) / VT); p_cat = Nv * np.exp(-(Eg - pc) / VT)
    Vbi0 = Eg - pa - pc
    d = d_nm * 1e-9; ee = eps * _E0
    cgeo = ee / d * 1e5                                             # nF/cm^2
    lam_an = np.sqrt(2.0 * ee * VT / (_Q * p_an)) * 1e9; lam_cat = np.sqrt(2.0 * ee * VT / (_Q * n_cat)) * 1e9
    gam = zeta * _Q * (mun + mup) / ee * 1e18                        # 1e-18 m^3/s
    return np.array([VT, Vbi0, cgeo, lam_an, lam_cat, np.log(p_an), np.log(n_cat), np.log(n_an), np.log(p_cat), gam, d_nm, eps, mun, mup], dtype=float)

import numpy as np


def effective_built_in_potential(voltage: float, par: "np.ndarray") -> "np.ndarray":
    par = np.asarray(par, dtype=float).ravel()
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    V = float(voltage); VT, Vbi0 = par[0], par[1]; d = par[10] * 1e-9; lam_an = par[3] * 1e-9; lam_cat = par[4] * 1e-9
    if not (V < Vbi0):
        raise ValueError("the injected-carrier model requires V below the nominal built-in potential")
    Vb = Vbi0
    for it in range(10000):
        ga = np.sqrt(1.0 + (2.0 * VT * d / ((Vb - V) * lam_an)) ** 2)
        gc = np.sqrt(1.0 + (2.0 * VT * d / ((Vb - V) * lam_cat)) ** 2)
        Vn = Vbi0 - 2.0 * VT * np.log(0.5 * (ga + 1.0)) - 2.0 * VT * np.log(0.5 * (gc + 1.0))
        if not (Vn > V):
            raise ValueError("no physical solution of the effective built-in potential at this bias")
        if abs(Vn - Vb) < 1e-15:
            Vb = Vn; break
        Vb = Vn
    ga = np.sqrt(1.0 + (2.0 * VT * d / ((Vb - V) * lam_an)) ** 2); gc = np.sqrt(1.0 + (2.0 * VT * d / ((Vb - V) * lam_cat)) ** 2)
    eta_an = 1.0 - 1.0 / ga; eta_cat = 1.0 - 1.0 / gc
    ebulk = (V - Vb) / d
    w_an = 2.0 * VT * d / (Vb - V) * eta_an; w_cat = 2.0 * VT * d / (Vb - V) * eta_cat
    k = abs(ebulk) / (2.0 * VT)                                                     # q|E_bulk|/(2kT), 1/m
    aa = np.arcsinh(abs(ebulk) * lam_an / (2.0 * VT)); ac = np.arcsinh(abs(ebulk) * lam_cat / (2.0 * VT))
    xs = 0.5 * d + (ac - aa) / (2.0 * k)                                            # crossover of the two field branches
    return np.array([Vb, eta_an, eta_cat, ebulk * 1e-6, w_an * 1e9, w_cat * 1e9, xs * 1e9], dtype=float)

import numpy as np


def field_and_carrier_profiles(positions: "np.ndarray", voltage: float, par: "np.ndarray") -> "np.ndarray":
    par = np.asarray(par, dtype=float).ravel(); x = np.atleast_1d(np.asarray(positions, dtype=float)) * 1e-9
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    d = par[10] * 1e-9
    if x.ndim != 1 or x.size == 0 or np.any(x < 0.0) or np.any(x > d):
        raise ValueError("positions must lie inside the active layer 0 <= x <= d")
    VT = par[0]; lam_an = par[3] * 1e-9; lam_cat = par[4] * 1e-9
    Vb, eta_an, eta_cat, eb, wa, wc, xs_nm = effective_built_in_potential(voltage, par)
    E = eb * 1e6; Ea = abs(E); xs = xs_nm * 1e-9
    k = Ea / (2.0 * VT)
    aa = np.arcsinh(Ea * lam_an / (2.0 * VT)); ac = np.arcsinh(Ea * lam_cat / (2.0 * VT))
    left = x <= xs
    Ex = np.where(left, E / np.tanh(k * x + aa), E / np.tanh(k * (d - x) + ac))
    phi_left = -(E / k) * (np.log(np.sinh(k * x + aa)) - np.log(np.sinh(aa)))           # phi(x) = -int_0^x E
    phi_right = (E / k) * (np.log(np.sinh(k * (d - x) + ac)) - np.log(np.sinh(ac)))      # phi(x) - phi(d)
    p_rel = np.where(left, np.exp(-phi_left / VT), 0.0)                                   # p / p_an in the hole region
    n_rel = np.where(left, 0.0, np.exp(phi_right / VT))                                   # n / n_cat in the electron region
    return np.stack([Ex * 1e-6, p_rel, n_rel], axis=1)

import numpy as np


_Q = 1.602176634e-19

_KB = 1.380649e-23

_E0 = 8.8541878128e-12

def analytic_capacitance(voltage: float, par: "np.ndarray") -> "np.ndarray":
    par = np.asarray(par, dtype=float).ravel()
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    V = float(voltage); VT = par[0]; Vbi0 = par[1]; d = par[10] * 1e-9
    Vb, eta_an, eta_cat, eb, wa, wc, xs = effective_built_in_potential(V, par)
    eta = eta_an + eta_cat
    c_rel = 1.0 / (1.0 - 2.0 * VT / (Vb - V) * eta)
    if not (c_rel > 0.0):
        raise ValueError("the analytic capacitance is not defined this close to the built-in potential")
    dc_rel = c_rel - 1.0
    inv = ((Vb - V) - 2.0 * eta * VT) / (2.0 * eta * VT)               # (dC/Cgeo)^-1 = q/(2 eta kT) [Vbi(V) - V - 2 eta kT/q]
    p_an = np.exp(par[5]); n_cat = np.exp(par[6]); ee = par[11] * _E0
    dc_weak = 2.0 * (_KB * (VT * _Q / _KB)) ** 2 * d / (_Q * (Vbi0 - V) ** 3) * (p_an + n_cat) / (ee / d)   # Eq. 13, relative to Cgeo
    return np.array([c_rel, dc_rel, inv, eta, dc_weak], dtype=float)

import numpy as np


_Q = 1.602176634e-19

_E0 = 8.8541878128e-12

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
    """Solve the block tridiagonal system with diagonal blocks A[i], super-diagonal B[i] (row i to i+1)
    and sub-diagonal C[i] (row i to i-1); r has shape (M, k). Block Thomas algorithm."""
    M = r.shape[0]
    Ap = A.copy(); rp = r.copy()
    for i in range(1, M):
        L = np.linalg.solve(Ap[i - 1].T, C[i].T).T
        Ap[i] = Ap[i] - L @ B[i - 1]
        rp[i] = rp[i] - L @ rp[i - 1]
    x = np.empty_like(rp)
    x[M - 1] = np.linalg.solve(Ap[M - 1], rp[M - 1])
    for i in range(M - 2, -1, -1):
        x[i] = np.linalg.solve(Ap[i], rp[i] - B[i] @ x[i + 1])
    return x

def _dd_unpack(state, prm, V):
    """Full node arrays (psi in V, n and p in 1/m^3) from the interior state (M, 3) = [psi/VT, ln n, ln p]."""
    VT, Vbi0, n_an, p_an, n_cat, p_cat = prm["VT"], prm["Vbi0"], prm["n_an"], prm["p_an"], prm["n_cat"], prm["p_cat"]
    psi = np.concatenate([[0.0], state[:, 0] * VT, [Vbi0 - V]])
    n = np.concatenate([[n_an], np.exp(state[:, 1]), [n_cat]])
    p = np.concatenate([[p_an], np.exp(state[:, 2]), [p_cat]])
    return psi, n, p

def _dd_fluxes(psi, n, p, prm, h):
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _dd_residual(state, prm, V, h):
    """Scaled residuals (M, 3): Poisson in units of VT, continuity per carrier at each node."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    R = prm["gamma"] * (n * p - prm["ni2"])
    Fpsi = ((psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h ** 2 + _Q / ee * (p[1:-1] - n[1:-1])) * h ** 2 / VT
    Fn = ((Jn[1:] - Jn[:-1]) / h - _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dn"] * n[1:-1])
    Fp = ((Jp[1:] - Jp[:-1]) / h + _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dp"] * p[1:-1])
    return np.stack([Fpsi, Fn, Fp], axis=1)

def _dd_jacobian(state, prm, V, h):
    """Blocks (M,3,3) of the Jacobian of the scaled residual with respect to [psi/VT, ln n, ln p], plus dF/dV (M,3)."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]; M = state.shape[0]; N = M + 1
    dl = np.diff(psi) / VT; Bp = _bernoulli(dl); Bm = _bernoulli(-dl); dBp = _dbernoulli(dl); dBm = -_dbernoulli(-dl)
    c = _Q * h ** 2 / (ee * VT); an = h ** 2 * prm["gamma"] / prm["Dn"]; ap = h ** 2 * prm["gamma"] / prm["Dp"]
    i = np.arange(1, N); ni = n[i]; pi_ = p[i]; nip = n[i + 1]; nim = n[i - 1]; pip = p[i + 1]; pim = p[i - 1]
    A = np.zeros((M, 3, 3)); B = np.zeros((M, 3, 3)); C = np.zeros((M, 3, 3)); dV = np.zeros((M, 3))
    # Poisson row: (psi_{i+1} - 2 psi_i + psi_{i-1})/VT + c (p_i - n_i); unknown psi/VT
    A[:, 0, 0] = -2.0; A[:, 0, 1] = -c * ni; A[:, 0, 2] = c * pi_; B[:, 0, 0] = 1.0; C[:, 0, 0] = 1.0
    dV[-1, 0] = -1.0 / VT
    # electron row divided by n_i
    fn = (nip * Bp[i] - ni * Bm[i]) - (ni * Bp[i - 1] - nim * Bm[i - 1]) - an * (ni * pi_ - prm["ni2"])
    A[:, 1, 1] = (-Bm[i] - Bp[i - 1]) - an * pi_ - fn / ni
    A[:, 1, 2] = -an * pi_
    B[:, 1, 1] = Bp[i] * nip / ni; C[:, 1, 1] = Bm[i - 1] * nim / ni
    dli = (nip * dBp[i] - ni * dBm[i]) / ni; dlim = -(ni * dBp[i - 1] - nim * dBm[i - 1]) / ni
    A[:, 1, 0] = -dli + dlim; B[:, 1, 0] = dli; C[:, 1, 0] = -dlim
    dV[-1, 1] = -dli[-1] / VT
    # hole row divided by p_i
    fp = (pi_ * Bp[i] - pip * Bm[i]) - (pim * Bp[i - 1] - pi_ * Bm[i - 1]) + ap * (ni * pi_ - prm["ni2"])
    A[:, 2, 2] = (Bp[i] + Bm[i - 1]) + ap * ni - fp / pi_
    A[:, 2, 1] = ap * ni
    B[:, 2, 2] = -Bm[i] * pip / pi_; C[:, 2, 2] = -Bp[i - 1] * pim / pi_
    dpi = (pi_ * dBp[i] - pip * dBm[i]) / pi_; dpim = -(pim * dBp[i - 1] - pi_ * dBm[i - 1]) / pi_
    A[:, 2, 0] = -dpi + dpim; B[:, 2, 0] = dpi; C[:, 2, 0] = -dpim
    dV[-1, 2] = -dpi[-1] / VT
    return A, B, C, dV

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step."""
    VT, Vbi0, cgeo, lam_an, lam_cat, ln_pan, ln_ncat, ln_nan, ln_pcat, gam, d_nm, eps, mun, mup = par
    d = d_nm * 1e-9
    return dict(VT=VT, Vbi0=Vbi0, d=d, ee=eps * _E0, Cgeo=eps * _E0 / d, p_an=np.exp(ln_pan), n_cat=np.exp(ln_ncat),
                n_an=np.exp(ln_nan), p_cat=np.exp(ln_pcat), ni2=np.exp(ln_nan + ln_pan), gamma=gam * 1e-18,
                Dn=mun * VT, Dp=mup * VT, lam_an=lam_an * 1e-9, lam_cat=lam_cat * 1e-9)

def _dd_equilibrium(prm, N):
    """Zero-bias state: Poisson-Boltzmann solution (Newton with a line search) in the state layout."""
    VT = prm["VT"]; d = prm["d"]; h = d / N; M = N - 1; c = _Q * h ** 2 / (prm["ee"] * VT)
    psi = np.linspace(0.0, prm["Vbi0"], N + 1)
    def res(ps):
        n = prm["n_an"] * np.exp(ps / VT); p = prm["p_an"] * np.exp(-ps / VT)
        return (ps[2:] - 2.0 * ps[1:-1] + ps[:-2]) / VT + c * (p[1:-1] - n[1:-1]), n, p
    for it in range(300):
        F, n, p = res(psi); nrm = np.max(np.abs(F))
        if nrm < 1e-14: break
        A = ((-2.0 - c * (p[1:-1] + n[1:-1])) / VT).reshape(M, 1, 1)
        Bq = np.full((M, 1, 1), 1.0 / VT); Cq = np.full((M, 1, 1), 1.0 / VT)
        dpsi = _block_tridiagonal_solve(A, Bq, Cq, -F.reshape(M, 1))[:, 0]
        lam = 1.0
        for _ in range(60):
            pt = psi.copy(); pt[1:-1] += lam * dpsi
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = res(pt)[0]
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        psi = pt
    n = prm["n_an"] * np.exp(psi / VT); p = prm["p_an"] * np.exp(-psi / VT)
    return np.stack([psi[1:-1] / VT, np.log(n[1:-1]), np.log(p[1:-1])], axis=1)

def _dd_newton(state, prm, V, h, tol=1e-13, maxit=200):
    U = state.copy()
    for it in range(maxit):
        F = _dd_residual(U, prm, V, h); nrm = np.max(np.abs(F))
        if nrm < tol: return U, nrm
        A, B, C, _ = _dd_jacobian(U, prm, V, h)
        dU = _block_tridiagonal_solve(A, B, C, -F); lam = 1.0
        for _ in range(60):
            Ut = U + lam * dU
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = _dd_residual(Ut, prm, V, h)
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        U = Ut
    return U, nrm

def _dd_continuation(prm, N, V, state0=None, V0=0.0, dvmax=0.05):
    """Steady state at bias V by continuation in steps of at most dvmax from (V0, state0) (equilibrium if None)."""
    h = prm["d"] / N
    U = _dd_equilibrium(prm, N) if state0 is None else state0.copy()
    nsteps = max(1, int(np.ceil(abs(V - V0) / dvmax - 1e-12)))
    for k in range(1, nsteps + 1):
        Vk = V0 + (V - V0) * k / nsteps
        U, nrm = _dd_newton(U, prm, Vk, h)
        if not (nrm < 1e-12):
            raise ValueError("drift-diffusion solve did not converge at V = %g" % Vk)
    return U

def drift_diffusion_state(voltage: float, intervals: int, par: "np.ndarray") -> "np.ndarray":
    par = np.asarray(par, dtype=float).ravel(); N = int(intervals); V = float(voltage)
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    prm = _dd_params(par)
    if not (V < prm["Vbi0"]):
        raise ValueError("bias must lie below the nominal built-in potential")
    U = _dd_continuation(prm, N, V)
    psi, n, p = _dd_unpack(U, prm, V)
    return np.stack([psi / prm["VT"], np.log(n), np.log(p)], axis=1)

import numpy as np


_Q = 1.602176634e-19

_E0 = 8.8541878128e-12

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
    """Solve the block tridiagonal system with diagonal blocks A[i], super-diagonal B[i] (row i to i+1)
    and sub-diagonal C[i] (row i to i-1); r has shape (M, k). Block Thomas algorithm."""
    M = r.shape[0]
    Ap = A.copy(); rp = r.copy()
    for i in range(1, M):
        L = np.linalg.solve(Ap[i - 1].T, C[i].T).T
        Ap[i] = Ap[i] - L @ B[i - 1]
        rp[i] = rp[i] - L @ rp[i - 1]
    x = np.empty_like(rp)
    x[M - 1] = np.linalg.solve(Ap[M - 1], rp[M - 1])
    for i in range(M - 2, -1, -1):
        x[i] = np.linalg.solve(Ap[i], rp[i] - B[i] @ x[i + 1])
    return x

def _dd_unpack(state, prm, V):
    """Full node arrays (psi in V, n and p in 1/m^3) from the interior state (M, 3) = [psi/VT, ln n, ln p]."""
    VT, Vbi0, n_an, p_an, n_cat, p_cat = prm["VT"], prm["Vbi0"], prm["n_an"], prm["p_an"], prm["n_cat"], prm["p_cat"]
    psi = np.concatenate([[0.0], state[:, 0] * VT, [Vbi0 - V]])
    n = np.concatenate([[n_an], np.exp(state[:, 1]), [n_cat]])
    p = np.concatenate([[p_an], np.exp(state[:, 2]), [p_cat]])
    return psi, n, p

def _dd_fluxes(psi, n, p, prm, h):
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _dd_residual(state, prm, V, h):
    """Scaled residuals (M, 3): Poisson in units of VT, continuity per carrier at each node."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    R = prm["gamma"] * (n * p - prm["ni2"])
    Fpsi = ((psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h ** 2 + _Q / ee * (p[1:-1] - n[1:-1])) * h ** 2 / VT
    Fn = ((Jn[1:] - Jn[:-1]) / h - _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dn"] * n[1:-1])
    Fp = ((Jp[1:] - Jp[:-1]) / h + _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dp"] * p[1:-1])
    return np.stack([Fpsi, Fn, Fp], axis=1)

def _dd_jacobian(state, prm, V, h):
    """Blocks (M,3,3) of the Jacobian of the scaled residual with respect to [psi/VT, ln n, ln p], plus dF/dV (M,3)."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]; M = state.shape[0]; N = M + 1
    dl = np.diff(psi) / VT; Bp = _bernoulli(dl); Bm = _bernoulli(-dl); dBp = _dbernoulli(dl); dBm = -_dbernoulli(-dl)
    c = _Q * h ** 2 / (ee * VT); an = h ** 2 * prm["gamma"] / prm["Dn"]; ap = h ** 2 * prm["gamma"] / prm["Dp"]
    i = np.arange(1, N); ni = n[i]; pi_ = p[i]; nip = n[i + 1]; nim = n[i - 1]; pip = p[i + 1]; pim = p[i - 1]
    A = np.zeros((M, 3, 3)); B = np.zeros((M, 3, 3)); C = np.zeros((M, 3, 3)); dV = np.zeros((M, 3))
    # Poisson row: (psi_{i+1} - 2 psi_i + psi_{i-1})/VT + c (p_i - n_i); unknown psi/VT
    A[:, 0, 0] = -2.0; A[:, 0, 1] = -c * ni; A[:, 0, 2] = c * pi_; B[:, 0, 0] = 1.0; C[:, 0, 0] = 1.0
    dV[-1, 0] = -1.0 / VT
    # electron row divided by n_i
    fn = (nip * Bp[i] - ni * Bm[i]) - (ni * Bp[i - 1] - nim * Bm[i - 1]) - an * (ni * pi_ - prm["ni2"])
    A[:, 1, 1] = (-Bm[i] - Bp[i - 1]) - an * pi_ - fn / ni
    A[:, 1, 2] = -an * pi_
    B[:, 1, 1] = Bp[i] * nip / ni; C[:, 1, 1] = Bm[i - 1] * nim / ni
    dli = (nip * dBp[i] - ni * dBm[i]) / ni; dlim = -(ni * dBp[i - 1] - nim * dBm[i - 1]) / ni
    A[:, 1, 0] = -dli + dlim; B[:, 1, 0] = dli; C[:, 1, 0] = -dlim
    dV[-1, 1] = -dli[-1] / VT
    # hole row divided by p_i
    fp = (pi_ * Bp[i] - pip * Bm[i]) - (pim * Bp[i - 1] - pi_ * Bm[i - 1]) + ap * (ni * pi_ - prm["ni2"])
    A[:, 2, 2] = (Bp[i] + Bm[i - 1]) + ap * ni - fp / pi_
    A[:, 2, 1] = ap * ni
    B[:, 2, 2] = -Bm[i] * pip / pi_; C[:, 2, 2] = -Bp[i - 1] * pim / pi_
    dpi = (pi_ * dBp[i] - pip * dBm[i]) / pi_; dpim = -(pim * dBp[i - 1] - pi_ * dBm[i - 1]) / pi_
    A[:, 2, 0] = -dpi + dpim; B[:, 2, 0] = dpi; C[:, 2, 0] = -dpim
    dV[-1, 2] = -dpi[-1] / VT
    return A, B, C, dV

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step."""
    VT, Vbi0, cgeo, lam_an, lam_cat, ln_pan, ln_ncat, ln_nan, ln_pcat, gam, d_nm, eps, mun, mup = par
    d = d_nm * 1e-9
    return dict(VT=VT, Vbi0=Vbi0, d=d, ee=eps * _E0, Cgeo=eps * _E0 / d, p_an=np.exp(ln_pan), n_cat=np.exp(ln_ncat),
                n_an=np.exp(ln_nan), p_cat=np.exp(ln_pcat), ni2=np.exp(ln_nan + ln_pan), gamma=gam * 1e-18,
                Dn=mun * VT, Dp=mup * VT, lam_an=lam_an * 1e-9, lam_cat=lam_cat * 1e-9)

def _dd_charge_capacitance_current(state, prm, V, N):
    """Displaced charge (Ramo-Shockley form), its exact derivative dQ/dV and the current density (SI)."""
    h = prm["d"] / N; d = prm["d"]; M = N - 1
    psi, n, p = _dd_unpack(state, prm, V)
    x = np.arange(N + 1) * h; tw = np.full(N + 1, h); tw[0] = tw[-1] = 0.5 * h
    Q = prm["Cgeo"] * (V - prm["Vbi0"]) + _Q * np.sum(tw * ((x / d) * p + (1.0 - x / d) * n))
    A, B, C, dV = _dd_jacobian(state, prm, V, h)
    dUdV = _block_tridiagonal_solve(A, B, C, -dV)
    xi = np.arange(1, N) * h
    dQ = np.zeros((M, 3)); dQ[:, 1] = _Q * h * (1.0 - xi / d) * n[1:-1]; dQ[:, 2] = _Q * h * (xi / d) * p[1:-1]
    Cap = float(np.sum(dQ * dUdV) + prm["Cgeo"])
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    return Q, Cap, float((Jn + Jp)[N // 2]), psi, n, p

def exact_capacitance(state: "np.ndarray", voltage: float, par: "np.ndarray") -> "np.ndarray":
    st = np.asarray(state, dtype=float); par = np.asarray(par, dtype=float).ravel(); V = float(voltage)
    if st.ndim != 2 or st.shape[1] != 3 or st.shape[0] < 5:
        raise ValueError("state must be an (N + 1, 3) array")
    N = st.shape[0] - 1; prm = _dd_params(par)
    if abs(st[0, 0]) > 1e-9 or abs(st[-1, 0] * prm["VT"] - (prm["Vbi0"] - V)) > 1e-9:
        raise ValueError("state does not satisfy the contact potentials at this bias")
    U = st[1:-1].copy()
    F = _dd_residual(U, prm, V, prm["d"] / N)
    if np.max(np.abs(F)) > 1e-9:
        raise ValueError("state is not a converged steady state")
    Q, Cap, J, psi, n, p = _dd_charge_capacitance_current(U, prm, V, N)
    Emid = -(psi[N // 2 + 1] - psi[N // 2]) / (prm["d"] / N)
    return np.array([Q / prm["Cgeo"], Cap / prm["Cgeo"], Q / (prm["ee"] * Emid)], dtype=float)

import numpy as np


_Q = 1.602176634e-19

_E0 = 8.8541878128e-12

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
    """Solve the block tridiagonal system with diagonal blocks A[i], super-diagonal B[i] (row i to i+1)
    and sub-diagonal C[i] (row i to i-1); r has shape (M, k). Block Thomas algorithm."""
    M = r.shape[0]
    Ap = A.copy(); rp = r.copy()
    for i in range(1, M):
        L = np.linalg.solve(Ap[i - 1].T, C[i].T).T
        Ap[i] = Ap[i] - L @ B[i - 1]
        rp[i] = rp[i] - L @ rp[i - 1]
    x = np.empty_like(rp)
    x[M - 1] = np.linalg.solve(Ap[M - 1], rp[M - 1])
    for i in range(M - 2, -1, -1):
        x[i] = np.linalg.solve(Ap[i], rp[i] - B[i] @ x[i + 1])
    return x

def _dd_unpack(state, prm, V):
    """Full node arrays (psi in V, n and p in 1/m^3) from the interior state (M, 3) = [psi/VT, ln n, ln p]."""
    VT, Vbi0, n_an, p_an, n_cat, p_cat = prm["VT"], prm["Vbi0"], prm["n_an"], prm["p_an"], prm["n_cat"], prm["p_cat"]
    psi = np.concatenate([[0.0], state[:, 0] * VT, [Vbi0 - V]])
    n = np.concatenate([[n_an], np.exp(state[:, 1]), [n_cat]])
    p = np.concatenate([[p_an], np.exp(state[:, 2]), [p_cat]])
    return psi, n, p

def _dd_fluxes(psi, n, p, prm, h):
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _dd_residual(state, prm, V, h):
    """Scaled residuals (M, 3): Poisson in units of VT, continuity per carrier at each node."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    R = prm["gamma"] * (n * p - prm["ni2"])
    Fpsi = ((psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h ** 2 + _Q / ee * (p[1:-1] - n[1:-1])) * h ** 2 / VT
    Fn = ((Jn[1:] - Jn[:-1]) / h - _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dn"] * n[1:-1])
    Fp = ((Jp[1:] - Jp[:-1]) / h + _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dp"] * p[1:-1])
    return np.stack([Fpsi, Fn, Fp], axis=1)

def _dd_jacobian(state, prm, V, h):
    """Blocks (M,3,3) of the Jacobian of the scaled residual with respect to [psi/VT, ln n, ln p], plus dF/dV (M,3)."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]; M = state.shape[0]; N = M + 1
    dl = np.diff(psi) / VT; Bp = _bernoulli(dl); Bm = _bernoulli(-dl); dBp = _dbernoulli(dl); dBm = -_dbernoulli(-dl)
    c = _Q * h ** 2 / (ee * VT); an = h ** 2 * prm["gamma"] / prm["Dn"]; ap = h ** 2 * prm["gamma"] / prm["Dp"]
    i = np.arange(1, N); ni = n[i]; pi_ = p[i]; nip = n[i + 1]; nim = n[i - 1]; pip = p[i + 1]; pim = p[i - 1]
    A = np.zeros((M, 3, 3)); B = np.zeros((M, 3, 3)); C = np.zeros((M, 3, 3)); dV = np.zeros((M, 3))
    # Poisson row: (psi_{i+1} - 2 psi_i + psi_{i-1})/VT + c (p_i - n_i); unknown psi/VT
    A[:, 0, 0] = -2.0; A[:, 0, 1] = -c * ni; A[:, 0, 2] = c * pi_; B[:, 0, 0] = 1.0; C[:, 0, 0] = 1.0
    dV[-1, 0] = -1.0 / VT
    # electron row divided by n_i
    fn = (nip * Bp[i] - ni * Bm[i]) - (ni * Bp[i - 1] - nim * Bm[i - 1]) - an * (ni * pi_ - prm["ni2"])
    A[:, 1, 1] = (-Bm[i] - Bp[i - 1]) - an * pi_ - fn / ni
    A[:, 1, 2] = -an * pi_
    B[:, 1, 1] = Bp[i] * nip / ni; C[:, 1, 1] = Bm[i - 1] * nim / ni
    dli = (nip * dBp[i] - ni * dBm[i]) / ni; dlim = -(ni * dBp[i - 1] - nim * dBm[i - 1]) / ni
    A[:, 1, 0] = -dli + dlim; B[:, 1, 0] = dli; C[:, 1, 0] = -dlim
    dV[-1, 1] = -dli[-1] / VT
    # hole row divided by p_i
    fp = (pi_ * Bp[i] - pip * Bm[i]) - (pim * Bp[i - 1] - pi_ * Bm[i - 1]) + ap * (ni * pi_ - prm["ni2"])
    A[:, 2, 2] = (Bp[i] + Bm[i - 1]) + ap * ni - fp / pi_
    A[:, 2, 1] = ap * ni
    B[:, 2, 2] = -Bm[i] * pip / pi_; C[:, 2, 2] = -Bp[i - 1] * pim / pi_
    dpi = (pi_ * dBp[i] - pip * dBm[i]) / pi_; dpim = -(pim * dBp[i - 1] - pi_ * dBm[i - 1]) / pi_
    A[:, 2, 0] = -dpi + dpim; B[:, 2, 0] = dpi; C[:, 2, 0] = -dpim
    dV[-1, 2] = -dpi[-1] / VT
    return A, B, C, dV

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step."""
    VT, Vbi0, cgeo, lam_an, lam_cat, ln_pan, ln_ncat, ln_nan, ln_pcat, gam, d_nm, eps, mun, mup = par
    d = d_nm * 1e-9
    return dict(VT=VT, Vbi0=Vbi0, d=d, ee=eps * _E0, Cgeo=eps * _E0 / d, p_an=np.exp(ln_pan), n_cat=np.exp(ln_ncat),
                n_an=np.exp(ln_nan), p_cat=np.exp(ln_pcat), ni2=np.exp(ln_nan + ln_pan), gamma=gam * 1e-18,
                Dn=mun * VT, Dp=mup * VT, lam_an=lam_an * 1e-9, lam_cat=lam_cat * 1e-9)

def _dd_equilibrium(prm, N):
    """Zero-bias state: Poisson-Boltzmann solution (Newton with a line search) in the state layout."""
    VT = prm["VT"]; d = prm["d"]; h = d / N; M = N - 1; c = _Q * h ** 2 / (prm["ee"] * VT)
    psi = np.linspace(0.0, prm["Vbi0"], N + 1)
    def res(ps):
        n = prm["n_an"] * np.exp(ps / VT); p = prm["p_an"] * np.exp(-ps / VT)
        return (ps[2:] - 2.0 * ps[1:-1] + ps[:-2]) / VT + c * (p[1:-1] - n[1:-1]), n, p
    for it in range(300):
        F, n, p = res(psi); nrm = np.max(np.abs(F))
        if nrm < 1e-14: break
        A = ((-2.0 - c * (p[1:-1] + n[1:-1])) / VT).reshape(M, 1, 1)
        Bq = np.full((M, 1, 1), 1.0 / VT); Cq = np.full((M, 1, 1), 1.0 / VT)
        dpsi = _block_tridiagonal_solve(A, Bq, Cq, -F.reshape(M, 1))[:, 0]
        lam = 1.0
        for _ in range(60):
            pt = psi.copy(); pt[1:-1] += lam * dpsi
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = res(pt)[0]
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        psi = pt
    n = prm["n_an"] * np.exp(psi / VT); p = prm["p_an"] * np.exp(-psi / VT)
    return np.stack([psi[1:-1] / VT, np.log(n[1:-1]), np.log(p[1:-1])], axis=1)

def _dd_newton(state, prm, V, h, tol=1e-13, maxit=200):
    U = state.copy()
    for it in range(maxit):
        F = _dd_residual(U, prm, V, h); nrm = np.max(np.abs(F))
        if nrm < tol: return U, nrm
        A, B, C, _ = _dd_jacobian(U, prm, V, h)
        dU = _block_tridiagonal_solve(A, B, C, -F); lam = 1.0
        for _ in range(60):
            Ut = U + lam * dU
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = _dd_residual(Ut, prm, V, h)
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        U = Ut
    return U, nrm

def _dd_continuation(prm, N, V, state0=None, V0=0.0, dvmax=0.05):
    """Steady state at bias V by continuation in steps of at most dvmax from (V0, state0) (equilibrium if None)."""
    h = prm["d"] / N
    U = _dd_equilibrium(prm, N) if state0 is None else state0.copy()
    nsteps = max(1, int(np.ceil(abs(V - V0) / dvmax - 1e-12)))
    for k in range(1, nsteps + 1):
        Vk = V0 + (V - V0) * k / nsteps
        U, nrm = _dd_newton(U, prm, Vk, h)
        if not (nrm < 1e-12):
            raise ValueError("drift-diffusion solve did not converge at V = %g" % Vk)
    return U

def _dd_charge_capacitance_current(state, prm, V, N):
    """Displaced charge (Ramo-Shockley form), its exact derivative dQ/dV and the current density (SI)."""
    h = prm["d"] / N; d = prm["d"]; M = N - 1
    psi, n, p = _dd_unpack(state, prm, V)
    x = np.arange(N + 1) * h; tw = np.full(N + 1, h); tw[0] = tw[-1] = 0.5 * h
    Q = prm["Cgeo"] * (V - prm["Vbi0"]) + _Q * np.sum(tw * ((x / d) * p + (1.0 - x / d) * n))
    A, B, C, dV = _dd_jacobian(state, prm, V, h)
    dUdV = _block_tridiagonal_solve(A, B, C, -dV)
    xi = np.arange(1, N) * h
    dQ = np.zeros((M, 3)); dQ[:, 1] = _Q * h * (1.0 - xi / d) * n[1:-1]; dQ[:, 2] = _Q * h * (xi / d) * p[1:-1]
    Cap = float(np.sum(dQ * dUdV) + prm["Cgeo"])
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    return Q, Cap, float((Jn + Jp)[N // 2]), psi, n, p

def built_in_extraction(fit_voltages: "np.ndarray", intervals: int, par: "np.ndarray") -> "np.ndarray":
    vs = np.atleast_1d(np.asarray(fit_voltages, dtype=float)); par = np.asarray(par, dtype=float).ravel(); N = int(intervals)
    if vs.ndim != 1 or vs.size < 3 or np.any(np.diff(vs) <= 0.0):
        raise ValueError("fit voltages must be at least three strictly increasing values")
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    prm = _dd_params(par); VT = par[0]
    if not (vs[-1] < prm["Vbi0"]):
        raise ValueError("fit voltages must lie below the nominal built-in potential")
    U0 = _dd_equilibrium(prm, N); U0, nrm = _dd_newton(U0, prm, 0.0, prm["d"] / N)
    c0 = _dd_charge_capacitance_current(U0, prm, 0.0, N)[1] / prm["Cgeo"]
    y = np.empty(vs.size)
    for sign in (1.0, -1.0):                                   # march away from equilibrium in both directions
        idx = [k for k in range(vs.size) if sign * vs[k] > 0.0] if sign > 0 else [k for k in range(vs.size) if vs[k] < 0.0][::-1]
        U, Vc = U0, 0.0
        for k in idx:
            U = _dd_continuation(prm, N, vs[k], state0=U, V0=Vc); Vc = vs[k]
            y[k] = 1.0 / (_dd_charge_capacitance_current(U, prm, vs[k], N)[1] / prm["Cgeo"] - 1.0)
    for k in range(vs.size):
        if vs[k] == 0.0:
            y[k] = 1.0 / (c0 - 1.0)
    A = np.vstack([np.ones_like(vs), vs]).T
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    intercept, slope = coef[0], coef[1]
    S = -slope; Vstar = intercept / S
    eta_ext = c0 / (2.0 * S * VT)                                   # the source's limiting-case reading of the slope
    vbi_ext = Vstar + (1.0 + c0) / S                                # the source's built-in potential formula V_bi = V* + [1 + C(0)/C_geo] / S
    # exact first-order expansion of the analytic inverse excess capacitance about V = 0 (arbitrary contacts)
    Vb0, ea, ec, eb, wa, wc, xs = effective_built_in_potential(0.0, par); eta0 = ea + ec
    ca0 = analytic_capacitance(0.0, par)[0]
    y0 = (Vb0 - 2.0 * eta0 * VT) / (2.0 * eta0 * VT)
    deta = ca0 * (ea * (2.0 - ea) * (1.0 - ea) + ec * (2.0 - ec) * (1.0 - ec)) / Vb0      # d eta / dV at V = 0
    dy = -(ca0 / eta0 + Vb0 * deta / eta0 ** 2) / (2.0 * VT)                          # d y / dV at V = 0
    S_lin = -dy; Vstar_lin = y0 / S_lin
    fS = eta0 ** 2 - 6.0 * (eta0 + 1.0 / eta0) + 12.0                                   # the source's one-ohmic-contact factor
    return np.array([S, Vstar, eta_ext, vbi_ext, vbi_ext - Vb0, S_lin, Vstar_lin, fS], dtype=float)

import numpy as np


_Q = 1.602176634e-19

_KB = 1.380649e-23

_E0 = 8.8541878128e-12

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
    """Solve the block tridiagonal system with diagonal blocks A[i], super-diagonal B[i] (row i to i+1)
    and sub-diagonal C[i] (row i to i-1); r has shape (M, k). Block Thomas algorithm."""
    M = r.shape[0]
    Ap = A.copy(); rp = r.copy()
    for i in range(1, M):
        L = np.linalg.solve(Ap[i - 1].T, C[i].T).T
        Ap[i] = Ap[i] - L @ B[i - 1]
        rp[i] = rp[i] - L @ rp[i - 1]
    x = np.empty_like(rp)
    x[M - 1] = np.linalg.solve(Ap[M - 1], rp[M - 1])
    for i in range(M - 2, -1, -1):
        x[i] = np.linalg.solve(Ap[i], rp[i] - B[i] @ x[i + 1])
    return x

def _dd_unpack(state, prm, V):
    """Full node arrays (psi in V, n and p in 1/m^3) from the interior state (M, 3) = [psi/VT, ln n, ln p]."""
    VT, Vbi0, n_an, p_an, n_cat, p_cat = prm["VT"], prm["Vbi0"], prm["n_an"], prm["p_an"], prm["n_cat"], prm["p_cat"]
    psi = np.concatenate([[0.0], state[:, 0] * VT, [Vbi0 - V]])
    n = np.concatenate([[n_an], np.exp(state[:, 1]), [n_cat]])
    p = np.concatenate([[p_an], np.exp(state[:, 2]), [p_cat]])
    return psi, n, p

def _dd_fluxes(psi, n, p, prm, h):
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _dd_residual(state, prm, V, h):
    """Scaled residuals (M, 3): Poisson in units of VT, continuity per carrier at each node."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    R = prm["gamma"] * (n * p - prm["ni2"])
    Fpsi = ((psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h ** 2 + _Q / ee * (p[1:-1] - n[1:-1])) * h ** 2 / VT
    Fn = ((Jn[1:] - Jn[:-1]) / h - _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dn"] * n[1:-1])
    Fp = ((Jp[1:] - Jp[:-1]) / h + _Q * R[1:-1]) * h ** 2 / (_Q * prm["Dp"] * p[1:-1])
    return np.stack([Fpsi, Fn, Fp], axis=1)

def _dd_jacobian(state, prm, V, h):
    """Blocks (M,3,3) of the Jacobian of the scaled residual with respect to [psi/VT, ln n, ln p], plus dF/dV (M,3)."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]; M = state.shape[0]; N = M + 1
    dl = np.diff(psi) / VT; Bp = _bernoulli(dl); Bm = _bernoulli(-dl); dBp = _dbernoulli(dl); dBm = -_dbernoulli(-dl)
    c = _Q * h ** 2 / (ee * VT); an = h ** 2 * prm["gamma"] / prm["Dn"]; ap = h ** 2 * prm["gamma"] / prm["Dp"]
    i = np.arange(1, N); ni = n[i]; pi_ = p[i]; nip = n[i + 1]; nim = n[i - 1]; pip = p[i + 1]; pim = p[i - 1]
    A = np.zeros((M, 3, 3)); B = np.zeros((M, 3, 3)); C = np.zeros((M, 3, 3)); dV = np.zeros((M, 3))
    # Poisson row: (psi_{i+1} - 2 psi_i + psi_{i-1})/VT + c (p_i - n_i); unknown psi/VT
    A[:, 0, 0] = -2.0; A[:, 0, 1] = -c * ni; A[:, 0, 2] = c * pi_; B[:, 0, 0] = 1.0; C[:, 0, 0] = 1.0
    dV[-1, 0] = -1.0 / VT
    # electron row divided by n_i
    fn = (nip * Bp[i] - ni * Bm[i]) - (ni * Bp[i - 1] - nim * Bm[i - 1]) - an * (ni * pi_ - prm["ni2"])
    A[:, 1, 1] = (-Bm[i] - Bp[i - 1]) - an * pi_ - fn / ni
    A[:, 1, 2] = -an * pi_
    B[:, 1, 1] = Bp[i] * nip / ni; C[:, 1, 1] = Bm[i - 1] * nim / ni
    dli = (nip * dBp[i] - ni * dBm[i]) / ni; dlim = -(ni * dBp[i - 1] - nim * dBm[i - 1]) / ni
    A[:, 1, 0] = -dli + dlim; B[:, 1, 0] = dli; C[:, 1, 0] = -dlim
    dV[-1, 1] = -dli[-1] / VT
    # hole row divided by p_i
    fp = (pi_ * Bp[i] - pip * Bm[i]) - (pim * Bp[i - 1] - pi_ * Bm[i - 1]) + ap * (ni * pi_ - prm["ni2"])
    A[:, 2, 2] = (Bp[i] + Bm[i - 1]) + ap * ni - fp / pi_
    A[:, 2, 1] = ap * ni
    B[:, 2, 2] = -Bm[i] * pip / pi_; C[:, 2, 2] = -Bp[i - 1] * pim / pi_
    dpi = (pi_ * dBp[i] - pip * dBm[i]) / pi_; dpim = -(pim * dBp[i - 1] - pi_ * dBm[i - 1]) / pi_
    A[:, 2, 0] = -dpi + dpim; B[:, 2, 0] = dpi; C[:, 2, 0] = -dpim
    dV[-1, 2] = -dpi[-1] / VT
    return A, B, C, dV

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step."""
    VT, Vbi0, cgeo, lam_an, lam_cat, ln_pan, ln_ncat, ln_nan, ln_pcat, gam, d_nm, eps, mun, mup = par
    d = d_nm * 1e-9
    return dict(VT=VT, Vbi0=Vbi0, d=d, ee=eps * _E0, Cgeo=eps * _E0 / d, p_an=np.exp(ln_pan), n_cat=np.exp(ln_ncat),
                n_an=np.exp(ln_nan), p_cat=np.exp(ln_pcat), ni2=np.exp(ln_nan + ln_pan), gamma=gam * 1e-18,
                Dn=mun * VT, Dp=mup * VT, lam_an=lam_an * 1e-9, lam_cat=lam_cat * 1e-9)

def _dd_equilibrium(prm, N):
    """Zero-bias state: Poisson-Boltzmann solution (Newton with a line search) in the state layout."""
    VT = prm["VT"]; d = prm["d"]; h = d / N; M = N - 1; c = _Q * h ** 2 / (prm["ee"] * VT)
    psi = np.linspace(0.0, prm["Vbi0"], N + 1)
    def res(ps):
        n = prm["n_an"] * np.exp(ps / VT); p = prm["p_an"] * np.exp(-ps / VT)
        return (ps[2:] - 2.0 * ps[1:-1] + ps[:-2]) / VT + c * (p[1:-1] - n[1:-1]), n, p
    for it in range(300):
        F, n, p = res(psi); nrm = np.max(np.abs(F))
        if nrm < 1e-14: break
        A = ((-2.0 - c * (p[1:-1] + n[1:-1])) / VT).reshape(M, 1, 1)
        Bq = np.full((M, 1, 1), 1.0 / VT); Cq = np.full((M, 1, 1), 1.0 / VT)
        dpsi = _block_tridiagonal_solve(A, Bq, Cq, -F.reshape(M, 1))[:, 0]
        lam = 1.0
        for _ in range(60):
            pt = psi.copy(); pt[1:-1] += lam * dpsi
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = res(pt)[0]
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        psi = pt
    n = prm["n_an"] * np.exp(psi / VT); p = prm["p_an"] * np.exp(-psi / VT)
    return np.stack([psi[1:-1] / VT, np.log(n[1:-1]), np.log(p[1:-1])], axis=1)

def _dd_newton(state, prm, V, h, tol=1e-13, maxit=200):
    U = state.copy()
    for it in range(maxit):
        F = _dd_residual(U, prm, V, h); nrm = np.max(np.abs(F))
        if nrm < tol: return U, nrm
        A, B, C, _ = _dd_jacobian(U, prm, V, h)
        dU = _block_tridiagonal_solve(A, B, C, -F); lam = 1.0
        for _ in range(60):
            Ut = U + lam * dU
            with np.errstate(over="ignore", invalid="ignore"):
                Ft = _dd_residual(Ut, prm, V, h)
            if np.all(np.isfinite(Ft)) and np.max(np.abs(Ft)) < nrm: break
            lam *= 0.5
        U = Ut
    return U, nrm

def _dd_continuation(prm, N, V, state0=None, V0=0.0, dvmax=0.05):
    """Steady state at bias V by continuation in steps of at most dvmax from (V0, state0) (equilibrium if None)."""
    h = prm["d"] / N
    U = _dd_equilibrium(prm, N) if state0 is None else state0.copy()
    nsteps = max(1, int(np.ceil(abs(V - V0) / dvmax - 1e-12)))
    for k in range(1, nsteps + 1):
        Vk = V0 + (V - V0) * k / nsteps
        U, nrm = _dd_newton(U, prm, Vk, h)
        if not (nrm < 1e-12):
            raise ValueError("drift-diffusion solve did not converge at V = %g" % Vk)
    return U

def _dd_charge_capacitance_current(state, prm, V, N):
    """Displaced charge (Ramo-Shockley form), its exact derivative dQ/dV and the current density (SI)."""
    h = prm["d"] / N; d = prm["d"]; M = N - 1
    psi, n, p = _dd_unpack(state, prm, V)
    x = np.arange(N + 1) * h; tw = np.full(N + 1, h); tw[0] = tw[-1] = 0.5 * h
    Q = prm["Cgeo"] * (V - prm["Vbi0"]) + _Q * np.sum(tw * ((x / d) * p + (1.0 - x / d) * n))
    A, B, C, dV = _dd_jacobian(state, prm, V, h)
    dUdV = _block_tridiagonal_solve(A, B, C, -dV)
    xi = np.arange(1, N) * h
    dQ = np.zeros((M, 3)); dQ[:, 1] = _Q * h * (1.0 - xi / d) * n[1:-1]; dQ[:, 2] = _Q * h * (xi / d) * p[1:-1]
    Cap = float(np.sum(dQ * dUdV) + prm["Cgeo"])
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    return Q, Cap, float((Jn + Jp)[N // 2]), psi, n, p

def validity_threshold(level: float, intervals: int, par: "np.ndarray", march_step: float) -> "np.ndarray":
    lev = float(level); N = int(intervals); par = np.asarray(par, dtype=float).ravel(); dv = float(march_step)
    if par.size != 14:
        raise ValueError("par must be the 14-entry parameter vector")
    if not (lev > 0.0) or not (dv > 0.0):
        raise ValueError("level and march step must be positive")
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    prm = _dd_params(par); h = prm["d"] / N
    def deviation(V, U0, V0):
        Us = _dd_continuation(prm, N, V, state0=U0, V0=V0, dvmax=dv)
        Q, Cap, J, psi, n, p = _dd_charge_capacitance_current(Us, prm, V, N)
        ca = analytic_capacitance(V, par)[0] * prm["Cgeo"]
        return 100.0 * (ca / Cap - 1.0), Us, J
    U = _dd_equilibrium(prm, N); U, nrm = _dd_newton(U, prm, 0.0, h)
    dev0, U, J = deviation(0.0, U, 0.0)
    if dev0 >= lev:
        raise ValueError("deviation already above the level at zero bias")
    Va = 0.0; Ua = U
    while True:
        Vn = Va + dv
        if not (Vn < prm["Vbi0"] - 4.0 * prm["VT"]):
            raise ValueError("deviation never reaches the level below the built-in potential")
        devn, Un, J = deviation(Vn, Ua, Va)
        if devn >= lev:
            a, b = Va, Vn
            break
        Va, Ua = Vn, Un
    for it in range(200):
        m = 0.5 * (a + b)
        devm, Um, Jm = deviation(m, Ua, a)
        if devm < lev:
            a, Ua = m, Um
        else:
            b = m
        if b - a < 1e-12:
            break
    Vt = 0.5 * (a + b)
    devt, Ut, Jt = deviation(Vt, Ua, a)
    Vb = effective_built_in_potential(Vt, par)[0]
    z = _Q * (Vb - Vt) / (2.0 * _KB * (prm["VT"] * _Q / _KB))
    return np.array([Vt, Vb - Vt, z], dtype=float)

import numpy as np


def cv_audit(design_table: list, intervals: int, level: float, fit_voltages: "np.ndarray", march_step: float) -> "np.ndarray":
    rows = [tuple(float(v) for v in r) for r in design_table]
    if not rows or any(len(r) != 11 for r in rows):
        raise ValueError("design_table rows must hold the 11 diode parameters")
    N = int(intervals)
    if N < 4 or N != intervals or N % 2 != 0:
        raise ValueError("intervals must be an even integer of at least 4")
    fv = np.atleast_1d(np.asarray(fit_voltages, dtype=float))
    NC = 22
    out = np.zeros((1 + len(rows), NC))
    for i, r in enumerate(rows, 1):
        par = diode_parameters(*r)
        eff0 = effective_built_in_potential(0.0, par)
        ca0 = analytic_capacitance(0.0, par)
        st0 = drift_diffusion_state(0.0, N, par)
        ex0 = exact_capacitance(st0, 0.0, par)
        Vp = 0.5 * eff0[0]
        prof = field_and_carrier_profiles([0.5 * par[10]], Vp, par)
        cap = analytic_capacitance(Vp, par)
        stp = drift_diffusion_state(Vp, N, par)
        exp_ = exact_capacitance(stp, Vp, par)
        ext = built_in_extraction(fv, N, par)
        thr = validity_threshold(level, N, par, march_step)
        out[i] = [par[1], eff0[0], eff0[1] + eff0[2], ca0[0], ex0[1], 100.0 * (ca0[0] / ex0[1] - 1.0),
                  Vp, cap[0], exp_[1], 100.0 * (cap[0] / exp_[1] - 1.0), exp_[2], prof[0, 0], eff0[6],
                  cap[4], cap[1] / cap[4], ext[0], ext[2], ext[3], ext[4], thr[0], thr[1], thr[2]]
    out[0, 0] = out[1, 19]
    out[0, 1] = len(rows); out[0, 2] = N; out[0, 3] = float(level)
    return out
SCICODE_GOLD_EOF
