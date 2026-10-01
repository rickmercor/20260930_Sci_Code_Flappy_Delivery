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


def _pair(v):
    a = np.atleast_1d(np.asarray(v, float))
    return (float(a.flat[0]), float(a.flat[-1]))


def sqpfc_laplacian_symbol(mesh: "int | Sequence[int]",
                                   cell: "float | Sequence[float]") -> "np.ndarray":
    """Fourier symbol of -Delta_h on the signed index set."""
    n = np.atleast_1d(np.asarray(mesh, float))
    c = np.atleast_1d(np.asarray(cell, float))
    if n.size not in (1, 2) or c.size not in (1, 2):
        raise ValueError("mesh and cell must be scalars or length-2 sequences")
    if np.any(n < 1.0) or np.any(n != np.round(n)):
        raise ValueError("mesh must contain positive whole numbers")
    if np.any(c <= 0.0) or not np.all(np.isfinite(c)):
        raise ValueError("cell edge lengths must be positive and finite")
    Mx, My = (int(round(t)) for t in _pair(mesh))
    Lx, Ly = _pair(cell)
    kx = (np.fft.fftfreq(Mx) * Mx).reshape(Mx, 1)
    ky = (np.fft.fftfreq(My) * My).reshape(1, My)
    return (4.0 * float(np.pi) ** 2
            * ((kx / Lx) ** 2 + (ky / Ly) ** 2) * np.ones((Mx, My)))

import numpy as np


def sqpfc_bdf_kernels(q: int) -> "np.ndarray":
    """BDFq convolution kernel and the extrapolation kernel."""
    if q != int(q) or int(q) not in (3, 4, 5):
        raise ValueError("q must be 3, 4 or 5")
    q = int(q)
    beta = {3: [11.0 / 6.0, -7.0 / 6.0, 1.0 / 3.0],
            4: [25.0 / 12.0, -23.0 / 12.0, 13.0 / 12.0, -1.0 / 4.0],
            5: [137.0 / 60.0, -163.0 / 60.0, 137.0 / 60.0,
                -21.0 / 20.0, 1.0 / 5.0]}[q]
    a = [1.0]
    for j in range(1, q):
        a.append(-a[-1] * (q - j) / j)             # (-1)^j C(q-1, j)
    g = [1.0 / beta[0]]                            # DOC kernel: conv inverse
    for m in range(1, q):
        s = sum(g[i] * beta[m - i] for i in range(m) if m - i < q)
        g.append(-s / beta[0])
    return np.stack([np.array(beta, float), np.array(a, float),
                     np.array(g, float)])

import numpy as np


def _pair(v):
    a = np.atleast_1d(np.asarray(v, float))
    return (float(a.flat[0]), float(a.flat[-1]))


def sqpfc_spectral_gradient(phi: "np.ndarray",
                                    cell: "float | Sequence[float]") -> "np.ndarray":
    """(D_x phi, D_y phi), Nyquist dropped from the odd-order multipliers."""
    phi = np.asarray(phi, float)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional array")
    c = np.atleast_1d(np.asarray(cell, float))
    if c.size not in (1, 2) or np.any(c <= 0.0):
        raise ValueError("cell must be a positive scalar or length-2 sequence")
    Mx, My = phi.shape
    Lx, Ly = _pair(cell)
    kx = np.fft.fftfreq(Mx) * Mx
    ky = np.fft.fftfreq(My) * My
    kx = np.where(np.abs(kx) == Mx // 2, 0.0, kx)
    ky = np.where(np.abs(ky) == My // 2, 0.0, ky)
    f = np.fft.fft2(phi)
    pi = float(np.pi)
    dx = np.fft.ifft2((2.0 * pi / Lx * 1j * kx).reshape(Mx, 1) * f).real
    dy = np.fft.ifft2((2.0 * pi / Ly * 1j * ky).reshape(1, My) * f).real
    return np.stack([dx, dy])

import numpy as np


def _pair(v):
    a = np.atleast_1d(np.asarray(v, float))
    return (float(a.flat[0]), float(a.flat[-1]))


def _area(phi, L):
    Lx, Ly = _pair(L)
    return (Lx / phi.shape[0]) * (Ly / phi.shape[1])


def _lapf(phi, L):
    """Delta_h phi, through the symbol of step 1."""
    lam = sqpfc_laplacian_symbol(phi.shape, L)
    return np.fft.ifft2(-lam * np.fft.fft2(phi)).real


def sqpfc_discrete_energy(phi: "np.ndarray", cell: "float | Sequence[float]",
                                  eps: float) -> float:
    """Original discrete free energy; gradient from step 3, symbol from step 1."""
    phi = np.asarray(phi, float)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional array")
    eps = float(eps)
    a = _area(phi, cell)
    g = sqpfc_spectral_gradient(phi, cell)
    g2 = g[0] ** 2 + g[1] ** 2
    w = phi + _lapf(phi, cell)
    return (0.25 * a * float(np.sum(g2 ** 2))
            - 0.5 * eps * a * float(np.sum(phi ** 2))
            + 0.5 * a * float(np.sum(w ** 2)))

import numpy as np


def sqpfc_quartic_divergence(phi: "np.ndarray",
                                     cell: "float | Sequence[float]") -> "np.ndarray":
    """N(phi) = -div_h(|grad_h phi|^2 grad_h phi); every derivative is step 3."""
    phi = np.asarray(phi, float)
    if phi.ndim != 2:
        raise ValueError("phi must be a two-dimensional array")
    g = sqpfc_spectral_gradient(phi, cell)
    g2 = g[0] ** 2 + g[1] ** 2
    return -(sqpfc_spectral_gradient(g2 * g[0], cell)[0]
             + sqpfc_spectral_gradient(g2 * g[1], cell)[1])

import numpy as np


def sqpfc_convex_split_step(hist: "np.ndarray", cell: "float | Sequence[float]",
                                    eps: float, S: float, tau: float,
                                    tol: float = 1e-13,
                                    max_iter: int = 200) -> "np.ndarray":
    """One BDFq convex-splitting step: kernels from step 2, symbol from step 1,
    nonlinear term from step 5."""
    hist = np.asarray(hist, float)
    if hist.ndim != 3:
        raise ValueError("hist must have shape (q, Mx, My)")
    if hist.shape[0] not in (3, 4, 5):
        raise ValueError("the leading axis of hist must be q = 3, 4 or 5")
    if not float(tau) > 0.0:
        raise ValueError("tau must be positive")
    if float(S) < 0.0:
        raise ValueError("S must be non-negative")
    if not float(tol) > 0.0 or int(max_iter) < 1:
        raise ValueError("tol must be positive and max_iter at least 1")
    q = hist.shape[0]
    eps = float(eps)
    S = float(S)
    tau = float(tau)
    _k = sqpfc_bdf_kernels(q)
    beta, alph = _k[0], _k[1]
    lam = sqpfc_laplacian_symbol(hist.shape[1:], cell)

    ex = np.zeros(q + 1)
    ex[0] += 1.0
    for j in range(q):
        ex[j] -= alph[j]
        ex[j + 1] += alph[j]
    phi_hat = sum(ex[j] * hist[q - j] for j in range(1, q + 1))

    dc = np.zeros(q + 1)
    for j in range(q):
        dc[j] += beta[j]
        dc[j + 1] -= beta[j]
    g = sum(dc[j] * hist[q - j] for j in range(1, q + 1)) / tau

    P = 1.0 + S * tau ** q * lam ** 2
    A = (beta[0] / tau) * P + lam * (1.0 - lam) ** 2
    rhs = eps * lam * np.fft.fft2(phi_hat) - P * np.fft.fft2(g)

    tol = float(tol)
    phi = phi_hat.copy()
    for _ in range(int(max_iter)):
        nl = sqpfc_quartic_divergence(phi, cell)
        new = np.fft.ifft2((-lam * np.fft.fft2(nl) + rhs) / A).real
        d = float(np.max(np.abs(new - phi)))
        phi = new
        if d < tol:
            return phi
    raise RuntimeError("the Picard iteration did not meet tol within max_iter "
                       "sweeps; the last iterate is not the step")

import numpy as np


def _pair(v):
    a = np.atleast_1d(np.asarray(v, float))
    return (float(a.flat[0]), float(a.flat[-1]))


def _area(phi, L):
    Lx, Ly = _pair(L)
    return (Lx / phi.shape[0]) * (Ly / phi.shape[1])


def _lapf(phi, L):
    """Delta_h phi, through the symbol of step 1."""
    lam = sqpfc_laplacian_symbol(phi.shape, L)
    return np.fft.ifft2(-lam * np.fft.fft2(phi)).real


def _ipm1(u, w, L):
    """<u, w>_{-1}, zero mode excluded; the symbol comes from step 1."""
    lam = sqpfc_laplacian_symbol(u.shape, L)
    inv = np.zeros_like(lam)
    nz = lam > 0
    inv[nz] = 1.0 / lam[nz]
    iu = np.fft.ifft2(inv * np.fft.fft2(u)).real
    return _area(u, L) * float(np.sum(iu * w))


def sqpfc_modified_energy(hist: "np.ndarray", cell: "float | Sequence[float]",
                                  eps: float, S: float, tau: float,
                                  G: "np.ndarray", J: "np.ndarray") -> float:
    """Modified discrete energy; E from step 4, the gradients from step 3."""
    hist = np.asarray(hist, float)
    G = np.asarray(G, float)
    J = np.asarray(J, float)
    if G.ndim != 2 or G.shape[0] != G.shape[1] or J.shape != G.shape:
        raise ValueError("G and J must be square matrices of the same shape")
    if hist.ndim != 3 or hist.shape[0] != G.shape[0] + 1:
        raise ValueError("hist must have shape (q+1, Mx, My) with q = G.shape[0]")
    if not float(tau) > 0.0:
        raise ValueError("tau must be positive")
    if float(S) < 0.0:
        raise ValueError("S must be non-negative")
    q = G.shape[0]
    eps = float(eps)
    a = _area(hist[0], cell)
    d = hist[1:] - hist[:-1]
    gr = np.stack([sqpfc_spectral_gradient(z, cell) for z in d])

    t1 = t2 = t3 = 0.0
    for i in range(q):
        for j in range(q):
            if G[i, j] != 0.0:
                t1 += G[i, j] * _ipm1(d[j], d[i], cell)
                t2 += G[i, j] * a * float(np.sum(gr[j] * gr[i]))
            if J[i, j] != 0.0:
                t3 += J[i, j] * a * float(np.sum(d[j] * d[i]))
    return (sqpfc_discrete_energy(hist[-1], cell, eps)
            + t1 / float(tau) + float(S) * float(tau) ** (q - 1) * t2 + eps * t3)

import numpy as np


def sqpfc_stabilization_certificate(q: int, kappa_q: float,
                                            eta_q: float, eps: float,
                                            S: float) -> "np.ndarray":
    """[C_q, C_required, S_min, margin]."""
    if q != int(q) or int(q) < 3:
        raise ValueError("q must be a whole number and at least 3")
    if not float(kappa_q) > 0.0:
        raise ValueError("kappa_q must be positive")
    if float(S) < 0.0:
        raise ValueError("S must be non-negative")
    q = int(q)
    kappa_q = float(kappa_q)
    eta_q = float(eta_q)
    eps = float(eps)
    S = float(S)
    pref = ((q - 1.0) / (q - 2.0)) ** (q - 2) * kappa_q * (q - 1.0)
    C = (pref * S) ** (1.0 / (q - 1.0))
    C_req = (eps * eta_q + eps / 2.0 + 5.0 / 8.0) ** 2 / kappa_q
    S_min = C_req ** (q - 1.0) / pref
    return np.array([C, C_req, S_min, C - C_req])

import numpy as np


def _seed_field(kind, M, L):
    a = np.atleast_1d(np.asarray(M, float))
    Mx, My = int(round(float(a.flat[0]))), int(round(float(a.flat[-1])))
    b = np.atleast_1d(np.asarray(L, float))
    Lx, Ly = float(b.flat[0]), float(b.flat[-1])
    X, Y = np.meshgrid(np.arange(Mx) * Lx / Mx, np.arange(My) * Ly / My,
                       indexing="ij")
    if kind == "trig":
        return 0.07 + 0.60 * (np.cos(X) * np.cos(Y)
                              + 0.40 * np.sin(2.0 * X) * np.cos(Y)
                              + 0.30 * np.cos(X - 2.0 * Y))
    r = np.sqrt((X - 0.5 * Lx) ** 2 + (Y - 0.5 * Ly) ** 2)
    return 2.5 * (1.0 - np.tanh(0.5 * (r - 2.0)))


def sqpfc_terminal_energy_sum(configs: "Sequence[str] | None" = None,
                                      mesh: "int | Sequence[int] | None" = None,
                                      tau: "float | None" = None,
                                      n_steps: "int | None" = None,
                                      S: "float | None" = None) -> float:
    G3 = np.array([[0.0, 0.0, 0.0],
                   [0.0, 1.0 / 6.0, -7.0 / 24.0],
                   [0.0, -7.0 / 24.0, 65.0 / 96.0]])
    J3 = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.5, -0.5],
                   [0.0, -0.5, 1.0]])
    kappa3 = 95.0 / 48.0
    eta3 = 0.5
    qq = 3
    table = (
        ("P1", "trig",    (8.0 * np.pi, 8.0 * np.pi),  (32, 32), 0.25, 200.0, 0.05, 12),
        ("P2", "nucleus", (25.0, 25.0),                (32, 32), 0.50, 100.0, 0.10, 20),
        ("P3", "trig",    (6.0 * np.pi, 10.0 * np.pi), (24, 40), 0.40, 250.0, 0.04, 15),
        ("P4", "nucleus", (32.0, 20.0),                (48, 24), 0.30,  50.0, 0.08, 18),
        ("P5", "trig",    (10.0 * np.pi, 4.0 * np.pi), (40, 24), 0.20, 120.0, 0.06, 14),
    )
    names = [c[0] for c in table] if configs is None else list(configs)
    known = [c[0] for c in table]
    for nm in names:
        if nm not in known:
            raise ValueError("unknown configuration tag: %r" % (nm,))
    if n_steps is not None and int(n_steps) < 0:
        raise ValueError("n_steps must be non-negative")
    if tau is not None and not float(tau) > 0.0:
        raise ValueError("tau must be positive")
    if S is not None and float(S) < 0.0:
        raise ValueError("S must be non-negative")
    total = 0.0
    for nm, kind, L, Mc, eps, sfac, tc, nc in table:
        if nm not in names:
            continue
        Mv = Mc if mesh is None else mesh
        tv = tc if tau is None else float(tau)
        nv = nc if n_steps is None else int(n_steps)
        cert = sqpfc_stabilization_certificate(qq, kappa3, eta3,
                                                       eps, 1.0)
        Sv = sfac * float(cert[2]) if S is None else float(S)
        u = _seed_field(kind, Mv, L)
        hist = [u] * (qq + 1)
        for _ in range(nv):
            hist.append(sqpfc_convex_split_step(
                np.stack(hist[-qq:]), L, eps, Sv, tv))
        total += sqpfc_modified_energy(
            np.stack(hist[-(qq + 1):]), L, eps, Sv, tv, G3, J3)
    return total
SCICODE_GOLD_EOF
