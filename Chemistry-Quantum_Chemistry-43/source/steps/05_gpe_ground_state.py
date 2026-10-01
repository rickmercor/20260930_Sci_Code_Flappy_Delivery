"""
Given the grid values vext of the external potential on the uniform periodic grid of the cell [0, a) and the coupling constant g, return the ground-state density rho = |phi|^2 of the periodic Gross-Pitaevskii equation (-1/2 d^2/dx^2 + vext + g rho) phi = E phi with unit normalisation (integral of rho over the cell equal to 1), computed with spectral derivatives and converged to at least 1e-10 in the density. Raise ValueError if vext is not a one-dimensional array with at least 8 points, if a is not positive, if g is negative, or if the numerical solve does not converge.

The Gross-Pitaevskii equation is a one-orbital nonlinear eigenvalue problem with the same structure as the Kohn-Sham equations, which is why the source uses it to study its inversion algorithms: the reference density is the exact ground-state density of a known potential.

Returns
-------
ndarray of float64 with shape (M,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gpe_ground_state(vext: "np.ndarray", a: float, g: float) -> "np.ndarray":
    """ndarray of float64 with shape (M,)."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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
    Jm = np.stack([_oracle_duality_map(I[:, j], a) for j in range(M)], axis=1)
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
 
 
def _oracle_gpe_ground_state(vext: "np.ndarray", a: float, g: float) -> "np.ndarray":
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "a, g = 10.0, 1.0\nvext = _oracle_external_potential(a, 128)",
            "call": "gpe_ground_state(vext, a, g)",
            "gold_call": "_oracle_gpe_ground_state(vext, a, g)",
        },
        {
            "setup": "a, g = 8.0, 0.5\nvext = _oracle_external_potential(a, 96)",
            "call": "gpe_ground_state(vext, a, g)",
            "gold_call": "_oracle_gpe_ground_state(vext, a, g)",
        },
        {
            "setup": "a, g = 10.0, 0.0\nvext = _oracle_external_potential(a, 64)",
            "call": "gpe_ground_state(vext, a, g)",
            "gold_call": "_oracle_gpe_ground_state(vext, a, g)",
        },
    ]
