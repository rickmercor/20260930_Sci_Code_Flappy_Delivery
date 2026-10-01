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
 
 
def external_potential(a: float, M: int) -> "np.ndarray":
    """The optical-lattice potential of the source's Sec IV B on the uniform periodic grid
    x_j = j a / M, j = 0..M-1:  v_ext(x) = -sin^2(pi x / a)."""
    a, M = float(a), int(M)
    if a <= 0.0 or M < 8:
        raise ValueError("need a > 0 and M >= 8")
    x = np.arange(M) * a / M
    return -np.sin(np.pi * x / a) ** 2

import numpy as np
 
 
def _wavenumbers(a, M):
    return 2.0 * np.pi * np.fft.fftfreq(int(M), d=1.0 / int(M)) / float(a)
 
def _coef(f, a):
    """Orthonormal-plane-wave Fourier coefficients of grid values f."""
    M = f.size
    return np.fft.fft(f) * (float(a) / M) / np.sqrt(float(a))
 
def _from_coef(c, a):
    M = c.size
    return np.real(np.fft.ifft(c) * np.sqrt(float(a)) * M / float(a))
 
 
def duality_map(f: "np.ndarray", a: float) -> "np.ndarray":
    """Eq 12: the duality (Riesz) map J : X -> X* of the periodic homogeneous Sobolev pair,
    J(f) = Σ_{G≠0} (f̂_G / |G|^2) e_G, i.e. the zero-mean solution of −J(f)'' = f − mean(f).
    The G = 0 coefficient of f carries no norm and is dropped."""
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 1 or f.size < 8 or a <= 0.0:
        raise ValueError("f must be a 1-D grid array with at least 8 points and a > 0")
    G = _wavenumbers(a, f.size)
    c = _coef(f, a)
    out = np.zeros_like(c)
    nz = G != 0.0
    out[nz] = c[nz] / G[nz] ** 2
    return _from_coef(out, a)

import numpy as np
 
 
def _wavenumbers(a, M):
    return 2.0 * np.pi * np.fft.fftfreq(int(M), d=1.0 / int(M)) / float(a)
 
def _coef(f, a):
    """Orthonormal-plane-wave Fourier coefficients of grid values f."""
    M = f.size
    return np.fft.fft(f) * (float(a) / M) / np.sqrt(float(a))
 
 
def density_norm(f: "np.ndarray", a: float) -> float:
    """Eq 11: the norm of the density space X = H^{-1}_{per,hom},
    ||f||_X = sqrt(Σ_{G≠0} |f̂_G|^2 / |G|^2) = sqrt(<f, J f>). The G = 0 coefficient (the mass)
    carries no norm, which is why X is used for density DIFFERENCES."""
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 1 or f.size < 8 or a <= 0.0:
        raise ValueError("f must be a 1-D grid array with at least 8 points and a > 0")
    G = _wavenumbers(a, f.size)
    c = _coef(f, a)
    nz = G != 0.0
    return float(np.sqrt(np.sum(np.abs(c[nz]) ** 2 / G[nz] ** 2)))

import numpy as np
 
 
def _wavenumbers(a, M):
    return 2.0 * np.pi * np.fft.fftfreq(int(M), d=1.0 / int(M)) / float(a)
 
def _coef(f, a):
    """Orthonormal-plane-wave Fourier coefficients of grid values f."""
    M = f.size
    return np.fft.fft(f) * (float(a) / M) / np.sqrt(float(a))
 
 
def potential_norm(v: "np.ndarray", a: float) -> float:
    """Eq 9: the norm of the potential space X* = H^1_{per,hom},
    ||v||_{X*} = sqrt(Σ_G |G|^2 |v̂_G|^2) = sqrt(∫ |v'(x)|^2 dx). A constant shift of v has
    norm zero, which is the gauge freedom of potentials."""
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or v.size < 8 or a <= 0.0:
        raise ValueError("v must be a 1-D grid array with at least 8 points and a > 0")
    G = _wavenumbers(a, v.size)
    c = _coef(v, a)
    return float(np.sqrt(np.sum(G ** 2 * np.abs(c) ** 2)))

import numpy as np
 
 
def _wavenumbers(a, M):
    return 2.0 * np.pi * np.fft.fftfreq(int(M), d=1.0 / int(M)) / float(a)
 
def _coef(f, a):
    """Orthonormal-plane-wave Fourier coefficients of grid values f."""
    M = f.size
    return np.fft.fft(f) * (float(a) / M) / np.sqrt(float(a))
 
def _from_coef(c, a):
    M = c.size
    return np.real(np.fft.ifft(c) * np.sqrt(float(a)) * M / float(a))
 
def _integral(f, a):
    return float(np.sum(f) * float(a) / f.size)
 
def _dense_ops(a, M):
    """Dense spectral −d²/dx² (Lap) and duality map (Jm) on the grid, for Newton solves."""
    G = _wavenumbers(a, M)
    I = np.eye(M)
    Lap = np.stack([_from_coef(G ** 2 * _coef(I[:, j], a), a) for j in range(M)], axis=1)
    Jm = np.stack([duality_map(I[:, j], a) for j in range(M)], axis=1)
    return G, 0.5 * (Lap + Lap.T), 0.5 * (Jm + Jm.T)
 
def _energy(phi, a, G, vext, gF, rho_gs, eps, Jm):
    """E(φ) = ½∫|φ'|² + ∫ v_ext φ² + (gF/2)∫φ⁴ [+ (1/2ε)||φ² − ρ_gs||_X²]."""
    E = 0.5 * float(np.sum(G ** 2 * np.abs(_coef(phi, a)) ** 2)) + _integral(vext * phi ** 2, a) \
        + 0.5 * gF * _integral(phi ** 4, a)
    if rho_gs is not None:
        d = phi ** 2 - rho_gs
        E += _integral(d * (Jm @ d), a) / (2.0 * eps)     # ||d||_X^2 = <d, J d>
    return E
 
def _gradient(phi, a, G, vext, gF, rho_gs, eps, Jm):
    gr = _from_coef(G ** 2 * _coef(phi, a), a) + 2.0 * (vext + gF * phi ** 2) * phi
    if rho_gs is not None:
        gr = gr + 2.0 * ((Jm @ (phi ** 2 - rho_gs)) / eps) * phi
    return gr
 
def _descend(phi, a, G, vext, gF, rho_gs, eps, Jm, its, tol):
    """Riemannian gradient descent on the unit sphere with an inverse-Laplacian
    preconditioner and Armijo backtracking; the energy decreases monotonically.
    Three details keep the cost bounded and platform-independent. The accepted step length
    is carried into the next iteration: the Moreau-Yosida curvature (1/eps)|G|^-2 fixes the
    scale of the step, so restarting the backtracking at t = 1 rediscovers the same small t
    at a cost of ~28 wasted energy evaluations per iteration. A candidate is taken only when
    it actually lowers the energy, so a line search with nowhere left to go ends the descent
    rather than stepping where Armijo refused. And the descent stops once the energy stops
    moving, because `tol` is an absolute residual that sits inside this method's own stall
    band -- whether it is ever reached is decided by rounding, not by convergence."""
    w = float(a) / phi.size
    E = _energy(phi, a, G, vext, gF, rho_gs, eps, Jm)
    t = 1.0
    stall = 0
    for _ in range(its):
        gr = _gradient(phi, a, G, vext, gF, rho_gs, eps, Jm)
        mu = w * np.sum(gr * phi) / 2.0
        r = gr - 2.0 * mu * phi
        if np.sqrt(w * np.sum(r ** 2)) < tol:
            break
        pc = _from_coef(_coef(r, a) / (1.0 + 0.5 * G ** 2), a)
        pc -= w * np.sum(pc * phi) * phi
        slope = w * np.sum(r * pc)
        t = min(1.0, 2.0 * t)
        ok = False
        while t > 1e-14:
            cand = phi - t * pc
            cand /= np.sqrt(w * np.sum(cand ** 2))
            Ec = _energy(cand, a, G, vext, gF, rho_gs, eps, Jm)
            if Ec <= E - 1e-4 * t * slope:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        dE = E - Ec
        phi, E = cand, Ec
        stall = stall + 1 if dE <= 1e-15 * max(1.0, abs(E)) else 0
        if stall >= 5:
            break
    return phi, E
 
def _newton(phi, mu, a, G, Lap, Jm, vext, gF, rho_gs, eps, tol=1e-12, its=60):
    """Damped Newton on the Euler–Lagrange system  H(φ)φ = μφ,  ∫φ² = 1."""
    M = phi.size
    w = float(a) / M
 
    def _resid(p, m):
        veff = vext + gF * p ** 2
        if rho_gs is not None:
            veff = veff + (Jm @ (p ** 2 - rho_gs)) / eps
        R = 0.5 * (Lap @ p) + veff * p - m * p
        return np.concatenate([R, [w * np.sum(p ** 2) - 1.0]]), veff
 
    F, veff = _resid(phi, mu)
    for _ in range(its):
        nF = np.linalg.norm(F)
        if nF < tol:
            break
        Jac = 0.5 * Lap + np.diag(veff - mu + 2.0 * gF * phi ** 2)
        if rho_gs is not None:
            Jac = Jac + (phi[:, None] * Jm * (2.0 * phi[None, :])) / eps
        A = np.zeros((M + 1, M + 1))
        A[:M, :M] = Jac
        A[:M, M] = -phi
        A[M, :M] = 2.0 * w * phi
        step = np.linalg.solve(A, -F)
        t = 1.0
        ok = False
        while t > 1e-6:
            p2, m2 = phi + t * step[:M], mu + t * step[M]
            F2, veff2 = _resid(p2, m2)
            if np.linalg.norm(F2) < nF:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        phi, mu, F, veff = p2, m2, F2, veff2
    return phi, mu, np.linalg.norm(F)
 
 
def gpe_ground_state(vext: "np.ndarray", a: float, g: float) -> "np.ndarray":
    """Ground-state density of the periodic Gross–Pitaevskii equation (Eq 34),
    (−½ d²/dx² + v_ext + g ρ) φ = E φ, ρ = φ², ∫ρ = 1, on the grid of v_ext: the minimiser of
    ½∫|φ'|² + ∫v_ext φ² + (g/2)∫φ⁴. Damped SCF on the lowest eigenpair, then Newton polish."""
    vext = np.asarray(vext, dtype=np.float64)
    if vext.ndim != 1 or vext.size < 8 or a <= 0.0 or g < 0.0:
        raise ValueError("v_ext must be a 1-D grid array with at least 8 points, a > 0, g >= 0")
    M = vext.size
    a = float(a)
    G, Lap, Jm = _dense_ops(a, M)
    w = a / M
    rho = np.ones(M) / a
    phi = np.sqrt(rho)
    for _ in range(400):
        H = 0.5 * Lap + np.diag(vext + g * rho)
        _, V = np.linalg.eigh(H)
        phi = V[:, 0] / np.sqrt(w)
        rho_new = phi ** 2
        if np.sqrt(w * np.sum((rho_new - rho) ** 2)) < 1e-8:
            rho = rho_new
            break
        rho = 0.7 * rho + 0.3 * rho_new
    phi, _ = _descend(phi, a, G, vext, g, None, None, Jm, 500, 1e-11)
    mu0 = w * np.sum(phi * (0.5 * (Lap @ phi) + (vext + g * phi ** 2) * phi))
    phi, _, nF = _newton(phi, mu0, a, G, Lap, Jm, vext, g, None, None)
    if not np.isfinite(nF) or nF > 1e-9:
        raise ValueError("ground-state solve did not converge")
    return phi ** 2

import numpy as np
 
 
def _wavenumbers(a, M):
    return 2.0 * np.pi * np.fft.fftfreq(int(M), d=1.0 / int(M)) / float(a)
 
def _coef(f, a):
    """Orthonormal-plane-wave Fourier coefficients of grid values f."""
    M = f.size
    return np.fft.fft(f) * (float(a) / M) / np.sqrt(float(a))
 
def _from_coef(c, a):
    M = c.size
    return np.real(np.fft.ifft(c) * np.sqrt(float(a)) * M / float(a))
 
def _integral(f, a):
    return float(np.sum(f) * float(a) / f.size)
 
def _dense_ops(a, M):
    """Dense spectral −d²/dx² (Lap) and duality map (Jm) on the grid, for Newton solves."""
    G = _wavenumbers(a, M)
    I = np.eye(M)
    Lap = np.stack([_from_coef(G ** 2 * _coef(I[:, j], a), a) for j in range(M)], axis=1)
    Jm = np.stack([duality_map(I[:, j], a) for j in range(M)], axis=1)
    return G, 0.5 * (Lap + Lap.T), 0.5 * (Jm + Jm.T)
 
def _energy(phi, a, G, vext, gF, rho_gs, eps, Jm):
    """E(φ) = ½∫|φ'|² + ∫ v_ext φ² + (gF/2)∫φ⁴ [+ (1/2ε)||φ² − ρ_gs||_X²]."""
    E = 0.5 * float(np.sum(G ** 2 * np.abs(_coef(phi, a)) ** 2)) + _integral(vext * phi ** 2, a) \
        + 0.5 * gF * _integral(phi ** 4, a)
    if rho_gs is not None:
        d = phi ** 2 - rho_gs
        E += _integral(d * (Jm @ d), a) / (2.0 * eps)     # ||d||_X^2 = <d, J d>
    return E
 
def _gradient(phi, a, G, vext, gF, rho_gs, eps, Jm):
    gr = _from_coef(G ** 2 * _coef(phi, a), a) + 2.0 * (vext + gF * phi ** 2) * phi
    if rho_gs is not None:
        gr = gr + 2.0 * ((Jm @ (phi ** 2 - rho_gs)) / eps) * phi
    return gr
 
def _descend(phi, a, G, vext, gF, rho_gs, eps, Jm, its, tol):
    """Riemannian gradient descent on the unit sphere with an inverse-Laplacian
    preconditioner and Armijo backtracking; the energy decreases monotonically.
    Three details keep the cost bounded and platform-independent. The accepted step length
    is carried into the next iteration: the Moreau-Yosida curvature (1/eps)|G|^-2 fixes the
    scale of the step, so restarting the backtracking at t = 1 rediscovers the same small t
    at a cost of ~28 wasted energy evaluations per iteration. A candidate is taken only when
    it actually lowers the energy, so a line search with nowhere left to go ends the descent
    rather than stepping where Armijo refused. And the descent stops once the energy stops
    moving, because `tol` is an absolute residual that sits inside this method's own stall
    band -- whether it is ever reached is decided by rounding, not by convergence."""
    w = float(a) / phi.size
    E = _energy(phi, a, G, vext, gF, rho_gs, eps, Jm)
    t = 1.0
    stall = 0
    for _ in range(its):
        gr = _gradient(phi, a, G, vext, gF, rho_gs, eps, Jm)
        mu = w * np.sum(gr * phi) / 2.0
        r = gr - 2.0 * mu * phi
        if np.sqrt(w * np.sum(r ** 2)) < tol:
            break
        pc = _from_coef(_coef(r, a) / (1.0 + 0.5 * G ** 2), a)
        pc -= w * np.sum(pc * phi) * phi
        slope = w * np.sum(r * pc)
        t = min(1.0, 2.0 * t)
        ok = False
        while t > 1e-14:
            cand = phi - t * pc
            cand /= np.sqrt(w * np.sum(cand ** 2))
            Ec = _energy(cand, a, G, vext, gF, rho_gs, eps, Jm)
            if Ec <= E - 1e-4 * t * slope:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        dE = E - Ec
        phi, E = cand, Ec
        stall = stall + 1 if dE <= 1e-15 * max(1.0, abs(E)) else 0
        if stall >= 5:
            break
    return phi, E
 
def _newton(phi, mu, a, G, Lap, Jm, vext, gF, rho_gs, eps, tol=1e-12, its=60):
    """Damped Newton on the Euler–Lagrange system  H(φ)φ = μφ,  ∫φ² = 1."""
    M = phi.size
    w = float(a) / M
 
    def _resid(p, m):
        veff = vext + gF * p ** 2
        if rho_gs is not None:
            veff = veff + (Jm @ (p ** 2 - rho_gs)) / eps
        R = 0.5 * (Lap @ p) + veff * p - m * p
        return np.concatenate([R, [w * np.sum(p ** 2) - 1.0]]), veff
 
    F, veff = _resid(phi, mu)
    for _ in range(its):
        nF = np.linalg.norm(F)
        if nF < tol:
            break
        Jac = 0.5 * Lap + np.diag(veff - mu + 2.0 * gF * phi ** 2)
        if rho_gs is not None:
            Jac = Jac + (phi[:, None] * Jm * (2.0 * phi[None, :])) / eps
        A = np.zeros((M + 1, M + 1))
        A[:M, :M] = Jac
        A[:M, M] = -phi
        A[M, :M] = 2.0 * w * phi
        step = np.linalg.solve(A, -F)
        t = 1.0
        ok = False
        while t > 1e-6:
            p2, m2 = phi + t * step[:M], mu + t * step[M]
            F2, veff2 = _resid(p2, m2)
            if np.linalg.norm(F2) < nF:
                ok = True
                break
            t *= 0.5
        if not ok:
            break
        phi, mu, F, veff = p2, m2, F2, veff2
    return phi, mu, np.linalg.norm(F)
 
 
def proximal_density(
    rho_gs: "np.ndarray",
    vext: "np.ndarray",
    a: float,
    eps: float,
) -> "np.ndarray":
    """Eq 35: the proximal density of the Moreau–Yosida-regularised guiding functional
    F(ρ) = T(ρ) + <v_ext, ρ>  (T = ½∫|φ'|², ρ = φ², ∫ρ = 1; no interaction term — Sec IV B),
        ρ_ε = argmin_ρ  F(ρ) + (1/2ε) ||ρ − ρ_gs||_X²,   ||d||_X² = <d, J d>  (Eq 11–12).
    Direct minimisation over φ (preconditioned descent), then Newton on the Euler–Lagrange
    equation to machine precision; the minimiser is unique (the source's Prop. 4)."""
    rho_gs = np.asarray(rho_gs, dtype=np.float64)
    vext = np.asarray(vext, dtype=np.float64)
    if rho_gs.shape != vext.shape or rho_gs.ndim != 1 or rho_gs.size < 8:
        raise ValueError("rho_gs and v_ext must be 1-D grid arrays of one length >= 8")
    if a <= 0.0 or eps <= 0.0:
        raise ValueError("need a > 0 and eps > 0")
    if np.any(rho_gs < 0.0) or abs(_integral(rho_gs, a) - 1.0) > 1e-8:
        raise ValueError("rho_gs must be a nonnegative density of unit mass")
    M = rho_gs.size
    a, eps = float(a), float(eps)
    G, Lap, Jm = _dense_ops(a, M)
    w = a / M
    phi = np.sqrt(np.maximum(rho_gs, 0.0))
    phi /= np.sqrt(w * np.sum(phi ** 2))
    phi, E1 = _descend(phi, a, G, vext, 0.0, rho_gs, eps, Jm, 3000, 1e-7)
    mu0 = w * np.sum(phi * (0.5 * (Lap @ phi) + (vext + (Jm @ (phi ** 2 - rho_gs)) / eps) * phi))
    phi2, _, nF = _newton(phi, mu0, a, G, Lap, Jm, vext, 0.0, rho_gs, eps)
    E2 = _energy(phi2, a, G, vext, 0.0, rho_gs, eps, Jm)
    if not (np.isfinite(nF) and nF < 1e-9 and E2 <= E1 + 1e-10):
        phi2, _ = _descend(phi, a, G, vext, 0.0, rho_gs, eps, Jm, 3000, 1e-11)
        E2 = _energy(phi2, a, G, vext, 0.0, rho_gs, eps, Jm)
        mu0 = w * np.sum(phi2 * (0.5 * (Lap @ phi2) + (vext + (Jm @ (phi2 ** 2 - rho_gs)) / eps) * phi2))
        phi2, _, nF = _newton(phi2, mu0, a, G, Lap, Jm, vext, 0.0, rho_gs, eps)
        if not (np.isfinite(nF) and nF < 1e-9):
            raise ValueError("proximal solve did not converge")
    return phi2 ** 2

import numpy as np
 
 
def inversion_error(a: float, M: int, g: float, eps: float) -> float:
    """ORCHESTRATOR (Sec IV B): reference GPE density on the M-point grid, proximal density at
    the given ε, inverted potential v_ε = (1/ε) J(ρ_ε − ρ_gs) (Eq 36), and its error against
    the sought potential v_g = g ρ_gs measured in X* = H^1_{per,hom}."""
    a, M, g, eps = float(a), int(M), float(g), float(eps)
    if a <= 0.0 or M < 8 or g < 0.0 or eps <= 0.0:
        raise ValueError("need a > 0, M >= 8, g >= 0, eps > 0")
    vext = external_potential(a, M)
    rho_gs = gpe_ground_state(vext, a, g)
    rho_eps = proximal_density(rho_gs, vext, a, eps)
    dist = density_norm(rho_eps - rho_gs, a)
    if not np.isfinite(dist):
        raise ValueError("the proximal-density displacement is not finite")
    v_eps = duality_map(rho_eps - rho_gs, a) / eps
    return potential_norm(v_eps - g * rho_gs, a)
SCICODE_GOLD_EOF
