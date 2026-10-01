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


def _im() -> "np.ndarray":
    """Six-component column of the identity tensor."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def elastic_operator(E: float, nu: float) -> "np.ndarray":
    if not np.isfinite(E) or E <= 0.0:
        raise ValueError("E must be a positive finite modulus")
    if not np.isfinite(nu) or nu <= -1.0 or nu >= 0.5:
        raise ValueError("nu must lie in (-1, 0.5)")
    im = _im()
    K = E / (3.0 * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = K * np.outer(im, im) + 2.0 * mu * (np.eye(6) - np.outer(im, im) / 3.0)
    out = np.zeros((7, 6))
    out[:6, :] = D
    out[6, 0] = K
    out[6, 1] = mu
    return out

import numpy as np


def degradation_state(phi: float, kappa: float) -> "np.ndarray":
    if not np.isfinite(phi) or phi < 0.0 or phi > 1.0:
        raise ValueError("phi must lie in [0, 1]")
    if not np.isfinite(kappa) or kappa <= 0.0 or kappa >= 1.0:
        raise ValueError("kappa must lie in (0, 1)")
    d = (1.0 - kappa) * (1.0 - phi) ** 2 + kappa
    dd = -2.0 * (1.0 - kappa) * (1.0 - phi)
    return np.array([float(d), float(dd), float(2.0 * (1.0 - kappa))])

import numpy as np


def _im() -> "np.ndarray":
    """Six-component column of the identity tensor."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _tr(v: "np.ndarray") -> float:
    return float(v[0] + v[1] + v[2])


def _dev(v: "np.ndarray") -> "np.ndarray":
    return v - (_tr(v) / 3.0) * _im()


def _unit(v: "np.ndarray") -> "np.ndarray":
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0.0 else np.zeros(6)


def faceted_directions(eps: "np.ndarray") -> "np.ndarray":
    eps = np.asarray(eps, dtype=float)
    if eps.shape != (6,) or not np.all(np.isfinite(eps)):
        raise ValueError("eps must be a finite six-component array")
    t = _tr(eps)
    s = (t / abs(t)) if t != 0.0 else 0.0
    G1 = (1.0 / 3.0) * s * _im()
    G2 = _unit(_dev(eps))
    return np.vstack([G1, G2])

import numpy as np


def _im() -> "np.ndarray":
    """Six-component column of the identity tensor."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _tr(v: "np.ndarray") -> float:
    return float(v[0] + v[1] + v[2])


def _dev(v: "np.ndarray") -> "np.ndarray":
    return v - (_tr(v) / 3.0) * _im()


def faceted_potential_gradients(eta: "np.ndarray", ft: float, fs: float) -> "np.ndarray":
    eta = np.asarray(eta, dtype=float)
    if eta.shape != (6,) or not np.all(np.isfinite(eta)):
        raise ValueError("eta must be a finite six-component array")
    if ft <= 0.0 or fs <= 0.0:
        raise ValueError("ft and fs must be positive strengths")
    im = _im()
    de = _dev(eta)
    nd = float(np.linalg.norm(de))
    dFd = ft * (1.0 if _tr(eta) > 0.0 else 0.0) * im
    if nd > 0.0:
        dFd = dFd + fs * de / nd
    dFi = (-1.0e6) * ft * (1.0 if _tr(eta) <= 0.0 else 0.0) * im
    return np.vstack([dFd, dFi])

import numpy as np


def _im() -> "np.ndarray":
    """Six-component column of the identity tensor."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _tr(v: "np.ndarray") -> float:
    return float(v[0] + v[1] + v[2])


def _dev(v: "np.ndarray") -> "np.ndarray":
    return v - (_tr(v) / 3.0) * _im()


def _unit(v: "np.ndarray") -> "np.ndarray":
    n = float(np.linalg.norm(v))
    return (v / n) if n > 0.0 else np.zeros(6)


def smooth_direction_gradient(eps: "np.ndarray", eta: "np.ndarray", ft: float, fs: float, eps_ref: float) -> "np.ndarray":
    eps = np.asarray(eps, dtype=float)
    eta = np.asarray(eta, dtype=float)
    if eps.shape != (6,) or eta.shape != (6,):
        raise ValueError("eps and eta must be six-component arrays")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(eta))):
        raise ValueError("eps and eta must be finite")
    if ft <= 0.0 or fs <= 0.0 or eps_ref <= 0.0:
        raise ValueError("ft, fs and eps_ref must be positive")
    de_e = _dev(eps)
    de_n = _dev(eta)
    nd = float(np.linalg.norm(de_n))
    if _tr(eps) >= 0.0:
        G = _unit(eps)
        den = np.sqrt(ft ** 2 * _tr(eta) ** 2 + fs ** 2 * nd ** 2)
        dFd = np.zeros(6) if den == 0.0 else (ft ** 2 * _tr(eta) * _im() + fs ** 2 * de_n) / den
        Fd = float(den)
        branch = 1.0
    else:
        G = _unit(de_e)
        fac = fs * (1.0 - _tr(eps) / eps_ref)
        dFd = (fac * de_n / nd) if nd > 0.0 else np.zeros(6)
        Fd = float(fac * nd)
        branch = -1.0
    return np.vstack([G, dFd, np.array([Fd, branch, 0.0, 0.0, 0.0, 0.0])])

import numpy as np


def _solve_multiplier(residual: "Callable[[float], float]", hi0: float = 1.0e-3) -> "tuple[float, bool]":
    """Smallest lam >= 0 with residual(lam) = 0; lam = 0 when the state is inside."""
    probe = 1.0e-14
    if residual(probe) >= 0.0:
        return 0.0, False
    hi = hi0
    for _ in range(400):
        if residual(hi) >= 0.0:
            break
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


def smooth_return_map(eps: "np.ndarray", phi: float, E: float, nu: float, ft: float, fs: float, eps_ref: float, kappa: float, kappa_t: float) -> "np.ndarray":
    if np.asarray(eps, dtype=float).shape != (6,):
        raise ValueError("eps must be a six-component array")
    if kappa_t < 0.0 or not np.isfinite(kappa_t):
        raise ValueError("kappa_t must be a non-negative finite factor")
    blk0 = elastic_operator(E, nu)
    D = blk0[:6, :]
    K = float(blk0[6, 0])
    dv = degradation_state(phi, kappa)[0]
    G = smooth_direction_gradient(eps, np.zeros(6), ft, fs, eps_ref)[0]

    def _residual(lam: float) -> float:
        eta = lam * G
        blk = smooth_direction_gradient(eps, eta, ft, fs, eps_ref)
        sig = D @ (eps - eta)
        return float(G @ (dv * blk[1] - sig) + kappa_t * K * lam)

    lam, active = _solve_multiplier(_residual)
    eta = lam * G
    blk = smooth_direction_gradient(eps, eta, ft, fs, eps_ref)
    sig = D @ (eps - eta)
    return np.concatenate([[lam, 1.0 if active else 0.0, float(blk[2][0])], eta, sig])

import numpy as np


def _im() -> "np.ndarray":
    """Six-component column of the identity tensor."""
    return np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _tr(v: "np.ndarray") -> float:
    return float(v[0] + v[1] + v[2])


def _dev(v: "np.ndarray") -> "np.ndarray":
    return v - (_tr(v) / 3.0) * _im()


def _solve_multiplier(residual: "Callable[[float], float]", hi0: float = 1.0e-3) -> "tuple[float, bool]":
    """Smallest lam >= 0 with residual(lam) = 0; lam = 0 when the state is inside."""
    probe = 1.0e-14
    if residual(probe) >= 0.0:
        return 0.0, False
    hi = hi0
    for _ in range(400):
        if residual(hi) >= 0.0:
            break
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


def faceted_return_map(eps: "np.ndarray", phi: float, E: float, nu: float, ft: float, fs: float, kappa: float, kappa_t: float) -> "np.ndarray":
    if np.asarray(eps, dtype=float).shape != (6,):
        raise ValueError("eps must be a six-component array")
    if kappa_t < 0.0 or not np.isfinite(kappa_t):
        raise ValueError("kappa_t must be a non-negative finite factor")
    blk0 = elastic_operator(E, nu)
    D = blk0[:6, :]
    K = float(blk0[6, 0])
    dv = degradation_state(phi, kappa)[0]
    Gs = faceted_directions(eps)
    lam = np.zeros(2)

    def _facet_residual(i: int, value: float, other: float) -> float:
        lm = np.zeros(2)
        lm[i] = value
        lm[1 - i] = other
        eta = lm[0] * Gs[0] + lm[1] * Gs[1]
        gr = faceted_potential_gradients(eta, ft, fs)
        sig = D @ (eps - eta)
        return float(Gs[i] @ (dv * gr[0] + gr[1] - sig) + kappa_t * K * value)

    active = np.zeros(2)
    for _ in range(60):
        new = lam.copy()
        for i in (0, 1):
            li, act = _solve_multiplier(lambda v, i=i: _facet_residual(i, v, lam[1 - i]))
            new[i] = li
            active[i] = 1.0 if act else 0.0
        if float(np.max(np.abs(new - lam))) < 1.0e-15:
            lam = new
            break
        lam = new
    eta = lam[0] * Gs[0] + lam[1] * Gs[1]
    sig = D @ (eps - eta)
    Fd = ft * max(_tr(eta), 0.0) + fs * float(np.linalg.norm(_dev(eta)))
    return np.concatenate([lam, active, [Fd], eta, sig])

import numpy as np


def consistent_tangent(dirs: "np.ndarray", active: "np.ndarray", E: float, nu: float, kappa_t: float) -> "np.ndarray":
    dirs = np.asarray(dirs, dtype=float)
    active = np.asarray(active, dtype=float)
    if dirs.ndim != 2 or dirs.shape[1] != 6:
        raise ValueError("dirs must be a (m, 6) array of directions")
    if active.shape != (dirs.shape[0],):
        raise ValueError("active must carry one flag per direction")
    if kappa_t < 0.0 or not np.isfinite(kappa_t):
        raise ValueError("kappa_t must be a non-negative finite factor")
    blk0 = elastic_operator(E, nu)
    D = blk0[:6, :]
    K = float(blk0[6, 0])
    g = np.array(dirs, dtype=float).T.copy()
    m = g.shape[1]
    S = g.T @ D @ g + kappa_t * K * np.eye(m)
    for i in range(m):
        if active[i] <= 0.0:
            g[:, i] = 0.0
            S[i, :] = 0.0
            S[:, i] = 0.0
            S[i, i] = 1.0
    return D - D @ g @ np.linalg.solve(S, g.T @ D)

import numpy as np


def phase_field_update(F_hist: float, Gc: float, ell: float, source_coeff: float, lap_phi: float) -> float:
    if not np.isfinite(F_hist) or F_hist < 0.0:
        raise ValueError("F_hist must be a non-negative finite driving force")
    if Gc <= 0.0 or ell <= 0.0:
        raise ValueError("Gc and ell must be positive")
    if not np.isfinite(source_coeff) or source_coeff <= 0.0:
        raise ValueError("source_coeff must be a positive finite coefficient")
    if not np.isfinite(lap_phi):
        raise ValueError("lap_phi must be finite")
    pre = source_coeff * F_hist
    return float((Gc * ell * lap_phi + pre) / (Gc / ell + pre))

import numpy as np


def _pack(t: "np.ndarray") -> "np.ndarray":
    """Tensor components [xx, yy, zz, yz, xz, xy] -> the source's six-component column."""
    r2 = np.sqrt(2.0)
    return np.array([t[0], t[1], t[2], r2 * t[3], r2 * t[4], r2 * t[5]], dtype=float)


def _configuration() -> "tuple[np.ndarray, np.ndarray, dict]":
    """The fixed strain path (tensor components), the supplied Laplacians and the material data."""
    deps = np.array([
        [ 6.0e-4, -2.0e-4, 0.0, 0.0,    0.0,  6.0e-4],
        [-1.8e-3, -1.1e-3, 0.0, 0.0,    0.0,  8.0e-4],
        [-6.0e-4, -4.0e-4, 0.0, 0.0,    0.0,  5.0e-4],
        [ 4.0e-4,  3.0e-4, 0.0, 0.0,    0.0, -1.1e-3],
        [ 1.3e-3,  9.0e-4, 0.0, 2.0e-4, 0.0,  4.0e-4],
        [ 9.0e-4,  7.0e-4, 0.0, 0.0,    0.0,  5.0e-4],
        [ 1.1e-3,  8.0e-4, 0.0, 0.0,    0.0,  6.0e-4],
    ])
    laps = np.array([0.0, 10.0, 20.0, 15.0, 40.0, 25.0, 30.0])
    pars = dict(E=200.0, nu=0.3, ft=0.150, fs=0.150, eps_ref=0.01,
                kappa=1.0e-3, kappa_t=1.0e-9, Gc=1.0e-4, ell=0.05)
    return deps, laps, pars


def cohesive_audit(load_scale: float) -> "np.ndarray":
    if not np.isfinite(load_scale) or load_scale <= 0.0:
        raise ValueError("load_scale must be a positive finite number")
    deps, laps, P = _configuration()
    D_el = elastic_operator(P["E"], P["nu"])[:6, :]
    rows = []
    for crit in ("smooth", "faceted"):
        eps = np.zeros(6); phi = 0.0; Fh = 0.0; sig_prev = np.zeros(6)
        for n in range(deps.shape[0]):
            d_eps = load_scale * _pack(deps[n])
            eps = eps + d_eps
            if crit == "smooth":
                r = smooth_return_map(eps, phi, P["E"], P["nu"], P["ft"], P["fs"],
                                              P["eps_ref"], P["kappa"], P["kappa_t"])
                lam = np.array([r[0], 0.0]); act = np.array([r[1], 0.0])
                Fd = float(r[2]); sig = r[9:15]
                dirs = np.vstack([smooth_direction_gradient(
                    eps, r[3:9], P["ft"], P["fs"], P["eps_ref"])[0], np.zeros(6)])
            else:
                r = faceted_return_map(eps, phi, P["E"], P["nu"], P["ft"], P["fs"],
                                               P["kappa"], P["kappa_t"])
                lam = r[0:2]; act = r[2:4]; sig = r[11:17]
                dirs = faceted_directions(eps)
                eta = r[5:11]
                # Fd of Eq. (17a) is positively homogeneous of degree one in eta,
                # so Euler's identity recovers it exactly from its own gradient.
                Fd = float(faceted_potential_gradients(eta, P["ft"], P["fs"])[0] @ eta)
            if float(np.max(act)) <= 0.0:
                Dt = D_el
            else:
                Dt = consistent_tangent(dirs, act, P["E"], P["nu"], P["kappa_t"])
            Fh = max(Fd, Fh)
            c2 = degradation_state(phi, P["kappa"])[2]
            phi = phase_field_update(Fh, P["Gc"], P["ell"], c2, laps[n])
            gap = float(np.linalg.norm((sig - sig_prev) - Dt @ d_eps))
            sig_prev = sig
            rows.append([lam[0], lam[1], float(np.linalg.norm(sig)), Fd, Fh, phi,
                         float(np.linalg.norm(Dt)), gap])
    return np.array(rows)
SCICODE_GOLD_EOF
