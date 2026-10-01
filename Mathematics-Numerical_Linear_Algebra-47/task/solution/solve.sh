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


def compute_split_operators(y: np.ndarray, D_nodes: np.ndarray, dx: float,
                                    c: float, r: float, eps: float, A: float,
                                    B: float) -> np.ndarray:
    """Reference implementation."""
    y = np.asarray(y, dtype=float)
    D_nodes = np.asarray(D_nodes, dtype=float)
    if y.ndim != 1 or y.size % 3 != 0:
        raise ValueError("y must be a 1D array whose length is a multiple of 3")
    n = y.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if D_nodes.ndim != 1 or D_nodes.size != n:
        raise ValueError("D_nodes must be a 1D array of length y.size // 3")
    if not np.isfinite(y).all() or not np.isfinite(D_nodes).all():
        raise ValueError("y and D_nodes must contain only finite values")
    if not (np.isscalar(dx) or np.ndim(dx) == 0) or float(dx) <= 0.0:
        raise ValueError("dx must be a positive scalar")
    if not (np.isscalar(eps) or np.ndim(eps) == 0) or float(eps) <= 0.0:
        raise ValueError("eps must be a positive scalar")
    for _name, _val in (("c", c), ("r", r), ("A", A), ("B", B)):
        if not (np.isscalar(_val) or np.ndim(_val) == 0) or not np.isfinite(float(_val)):
            raise ValueError(_name + " must be a finite scalar")
    dx = float(dx)
    c = float(c)
    r = float(r)
    eps = float(eps)
    A = float(A)
    B = float(B)

    D_face = 0.5 * (D_nodes[:-1] + D_nodes[1:])
    f_adv = np.zeros(3 * n)
    f_dif = np.zeros(3 * n)
    for k in range(3):
        s = y[k * n:(k + 1) * n]
        adv = np.zeros(n)
        dif = np.zeros(n)
        adv[1:-1] = -c * (s[2:] - s[:-2]) / (2.0 * dx)
        dif[1:-1] = (D_face[1:] * (s[2:] - s[1:-1])
                     - D_face[:-1] * (s[1:-1] - s[:-2])) / dx ** 2
        f_adv[k * n:(k + 1) * n] = adv
        f_dif[k * n:(k + 1) * n] = dif

    u = y[:n]
    v = y[n:2 * n]
    w = y[2 * n:]
    f_rx = np.concatenate([r * (A - (w + 1.0) * u + v * u ** 2),
                           r * (w * u - v * u ** 2),
                           r * ((B - w) / eps - w * u)])
    for k in range(3):
        f_rx[k * n] = 0.0
        f_rx[(k + 1) * n - 1] = 0.0

    return np.vstack([f_adv, f_dif, f_rx])

import numpy as np


def estimate_dominant_eigenvalue(y0: np.ndarray, D_nodes: np.ndarray,
                                         dx: float, seed: int, rtol: float,
                                         atol: float, tau: float, q_lambda: float,
                                         max_iter: int = 200) -> np.ndarray:
    """Reference implementation."""
    y0 = np.asarray(y0, dtype=float)
    D_nodes = np.asarray(D_nodes, dtype=float)
    if y0.ndim != 1 or y0.size % 3 != 0:
        raise ValueError("y0 must be a 1D array whose length is a multiple of 3")
    n = y0.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if D_nodes.ndim != 1 or D_nodes.size != n:
        raise ValueError("D_nodes must be a 1D array of length y0.size // 3")
    if not np.isfinite(y0).all() or not np.isfinite(D_nodes).all():
        raise ValueError("y0 and D_nodes must contain only finite values")
    for _name, _val in (("dx", dx), ("rtol", rtol), ("atol", atol),
                        ("tau", tau), ("q_lambda", q_lambda)):
        if not (np.isscalar(_val) or np.ndim(_val) == 0) or not np.isfinite(float(_val)):
            raise ValueError(_name + " must be a finite scalar")
        if float(_val) <= 0.0:
            raise ValueError(_name + " must be positive")
    if int(max_iter) < 2:
        raise ValueError("max_iter must be at least 2")
    dx = float(dx)
    rtol = float(rtol)
    atol = float(atol)
    tau = float(tau)
    q_lambda = float(q_lambda)

    D_face = 0.5 * (D_nodes[:-1] + D_nodes[1:])

    def _diffusion(vec):
        out = np.zeros(3 * n)
        for k in range(3):
            s = vec[k * n:(k + 1) * n]
            g = np.zeros(n)
            g[1:-1] = (D_face[1:] * (s[2:] - s[1:-1])
                       - D_face[:-1] * (s[1:-1] - s[:-2])) / dx ** 2
            out[k * n:(k + 1) * n] = g
        return out

    weights = rtol * np.abs(y0) + atol
    rng = np.random.default_rng(int(seed))
    v = rng.standard_normal(y0.size)
    base = _diffusion(y0)
    lam_prev = 0.0
    lam = 0.0
    iters = 0
    converged = False
    for k in range(int(max_iter)):
        iters = k + 1
        sigma = 1.0 / np.sqrt(np.mean((v / weights) ** 2))
        jv = (_diffusion(y0 + sigma * v) - base) / sigma
        lam = float((v @ jv) / (v @ v))
        nrm = np.linalg.norm(jv)
        if nrm == 0.0:
            raise ValueError("power iteration produced a zero image vector")
        v = jv / nrm
        if k > 0 and abs(lam - lam_prev) < tau * abs(lam):
            converged = True
            break
        lam_prev = lam
    if not converged:
        raise ValueError("power iteration did not converge within max_iter")

    return np.array([lam, q_lambda * abs(lam), float(iters)])

import numpy as np


def select_rkl_stage_counts(h: float, delta_c: np.ndarray,
                                    lam_eff: float) -> np.ndarray:
    """Reference implementation."""
    delta_c = np.asarray(delta_c, dtype=float)
    if not (np.isscalar(h) or np.ndim(h) == 0) or not np.isfinite(float(h)) \
            or float(h) <= 0.0:
        raise ValueError("h must be a finite positive scalar")
    if not (np.isscalar(lam_eff) or np.ndim(lam_eff) == 0) \
            or not np.isfinite(float(lam_eff)):
        raise ValueError("lam_eff must be a finite scalar")
    if float(lam_eff) < 0.0:
        raise ValueError("lam_eff must be a non-negative magnitude")
    if delta_c.ndim != 1 or delta_c.size < 1:
        raise ValueError("delta_c must be a non-empty 1D array")
    if not np.isfinite(delta_c).all():
        raise ValueError("delta_c must contain only finite values")
    if np.any(delta_c <= 0.0):
        raise ValueError("every entry of delta_c must be positive")
    h = float(h)
    lam_eff = float(lam_eff)

    counts = []
    for dc in delta_c:
        arg = 0.5 * (np.sqrt(9.0 + 8.0 * float(dc) * h * lam_eff) - 1.0)
        counts.append(max(2, int(np.ceil(arg))))

    return np.array(counts, dtype=float)

import numpy as np


def rkl2_super_step(y: np.ndarray, H: float, s: int, forcing: np.ndarray,
                            D_nodes: np.ndarray, dx: float) -> np.ndarray:
    """Reference implementation."""
    y = np.asarray(y, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    D_nodes = np.asarray(D_nodes, dtype=float)
    if y.ndim != 1 or y.size % 3 != 0:
        raise ValueError("y must be a 1D array whose length is a multiple of 3")
    n = y.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if forcing.shape != y.shape:
        raise ValueError("forcing must have the same shape as y")
    if D_nodes.ndim != 1 or D_nodes.size != n:
        raise ValueError("D_nodes must be a 1D array of length y.size // 3")
    if not (np.isfinite(y).all() and np.isfinite(forcing).all()
            and np.isfinite(D_nodes).all()):
        raise ValueError("y, forcing and D_nodes must contain only finite values")
    if not (np.isscalar(H) or np.ndim(H) == 0) or not np.isfinite(float(H)) \
            or float(H) <= 0.0:
        raise ValueError("H must be a finite positive scalar")
    if not (np.isscalar(dx) or np.ndim(dx) == 0) or not np.isfinite(float(dx)) \
            or float(dx) <= 0.0:
        raise ValueError("dx must be a finite positive scalar")
    if int(s) != s or int(s) < 2:
        raise ValueError("s must be an integer of at least 2")
    H = float(H)
    dx = float(dx)
    s = int(s)

    D_face = 0.5 * (D_nodes[:-1] + D_nodes[1:])

    def _slope(vec):
        out = np.zeros(3 * n)
        for k in range(3):
            sp = vec[k * n:(k + 1) * n]
            g = np.zeros(n)
            g[1:-1] = (D_face[1:] * (sp[2:] - sp[1:-1])
                       - D_face[:-1] * (sp[1:-1] - sp[:-2])) / dx ** 2
            out[k * n:(k + 1) * n] = g
        return out + forcing

    b = np.zeros(s + 1)
    b[0] = 1.0 / 3.0
    b[1] = 1.0 / 3.0
    for j in range(2, s + 1):
        b[j] = (j * j + j - 2.0) / (2.0 * j * (j + 1.0))
    a = 1.0 - b
    w1 = 4.0 / (s * s + s - 2.0)

    f0 = _slope(y)
    y_jm2 = y
    y_jm1 = y + b[1] * w1 * H * f0
    for j in range(2, s + 1):
        mu = (2.0 * j - 1.0) / j * b[j] / b[j - 1]
        nu = -(j - 1.0) / j * b[j] / b[j - 2]
        mu_t = mu * w1
        gamma_t = -a[j - 1] * mu_t
        y_new = (mu * y_jm1 + nu * y_jm2 + (1.0 - mu - nu) * y
                 + mu_t * H * _slope(y_jm1) + gamma_t * H * f0)
        y_jm2 = y_jm1
        y_jm1 = y_new

    return y_jm1

import numpy as np


def solve_reaction_stage(rhs: np.ndarray, h_gamma: float, guess: np.ndarray,
                                 r: float, eps: float, A: float, B: float,
                                 tol: float = 1e-13,
                                 max_iter: int = 60) -> np.ndarray:
    """Reference implementation."""
    rhs = np.asarray(rhs, dtype=float)
    guess = np.asarray(guess, dtype=float)
    if rhs.ndim != 1 or rhs.size % 3 != 0:
        raise ValueError("rhs must be a 1D array whose length is a multiple of 3")
    n = rhs.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if guess.shape != rhs.shape:
        raise ValueError("guess must have the same shape as rhs")
    if not (np.isfinite(rhs).all() and np.isfinite(guess).all()):
        raise ValueError("rhs and guess must contain only finite values")
    for _name, _val in (("h_gamma", h_gamma), ("r", r), ("A", A), ("B", B)):
        if not (np.isscalar(_val) or np.ndim(_val) == 0) \
                or not np.isfinite(float(_val)):
            raise ValueError(_name + " must be a finite scalar")
    if float(h_gamma) < 0.0:
        raise ValueError("h_gamma must be non-negative")
    if not (np.isscalar(eps) or np.ndim(eps) == 0) or not np.isfinite(float(eps)) \
            or float(eps) <= 0.0:
        raise ValueError("eps must be a finite positive scalar")
    if not (np.isscalar(tol) or np.ndim(tol) == 0) or float(tol) <= 0.0:
        raise ValueError("tol must be a positive scalar")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1")
    hg = float(h_gamma)
    r = float(r)
    eps = float(eps)
    A = float(A)
    B = float(B)
    tol = float(tol)

    def _reaction(vec):
        u = vec[:n]
        v = vec[n:2 * n]
        w = vec[2 * n:]
        out = np.concatenate([r * (A - (w + 1.0) * u + v * u ** 2),
                              r * (w * u - v * u ** 2),
                              r * ((B - w) / eps - w * u)])
        for k in range(3):
            out[k * n] = 0.0
            out[(k + 1) * n - 1] = 0.0
        return out

    mask = np.ones(n)
    mask[0] = 0.0
    mask[-1] = 0.0

    z = guess.copy()
    converged = False
    for _ in range(int(max_iter)):
        resid = z - hg * _reaction(z) - rhs
        if np.max(np.abs(resid)) < tol:
            converged = True
            break
        u = z[:n]
        v = z[n:2 * n]
        w = z[2 * n:]
        J = np.zeros((n, 3, 3))
        J[:, 0, 0] = r * (-(w + 1.0) + 2.0 * v * u) * mask
        J[:, 0, 1] = r * u ** 2 * mask
        J[:, 0, 2] = -r * u * mask
        J[:, 1, 0] = r * (w - 2.0 * v * u) * mask
        J[:, 1, 1] = -r * u ** 2 * mask
        J[:, 1, 2] = r * u * mask
        J[:, 2, 0] = -r * w * mask
        J[:, 2, 2] = r * (-1.0 / eps - u) * mask
        M = np.tile(np.eye(3), (n, 1, 1)) - hg * J
        Fn = np.stack([resid[:n], resid[n:2 * n], resid[2 * n:]], axis=1)
        try:
            dz = np.linalg.solve(M, Fn[..., None])[..., 0]
        except np.linalg.LinAlgError:
            raise ValueError("nodewise Newton system is singular")
        z = z - np.concatenate([dz[:, 0], dz[:, 1], dz[:, 2]])
        if not np.isfinite(z).all():
            raise ValueError("nodewise Newton iteration produced non-finite values")
    if not converged:
        resid = z - hg * _reaction(z) - rhs
        if np.max(np.abs(resid)) >= tol:
            raise ValueError("nodewise Newton did not converge within max_iter")

    return z

import numpy as np


def assemble_coupling_vector(h: float, A_impl: np.ndarray,
                                     A_expl: np.ndarray, d_impl: np.ndarray,
                                     d_expl: np.ndarray, row_index: int,
                                     f_impl: np.ndarray,
                                     f_expl: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    A_impl = np.asarray(A_impl, dtype=float)
    A_expl = np.asarray(A_expl, dtype=float)
    d_impl = np.asarray(d_impl, dtype=float)
    d_expl = np.asarray(d_expl, dtype=float)
    f_impl = np.asarray(f_impl, dtype=float)
    f_expl = np.asarray(f_expl, dtype=float)
    if A_impl.ndim != 2 or A_impl.shape[0] != A_impl.shape[1]:
        raise ValueError("A_impl must be a square 2D array")
    if A_expl.shape != A_impl.shape:
        raise ValueError("A_expl must have the same shape as A_impl")
    s = A_impl.shape[0]
    if s < 3:
        raise ValueError("the tableau must have at least 3 stages")
    if d_impl.ndim != 1 or d_impl.size != s or d_expl.shape != d_impl.shape:
        raise ValueError("d_impl and d_expl must be 1D arrays of length s")
    if f_impl.ndim != 2 or f_impl.shape[0] != s:
        raise ValueError("f_impl must be a 2D array with one row per tableau stage")
    if f_expl.shape != f_impl.shape:
        raise ValueError("f_expl must have the same shape as f_impl")
    if not (np.isfinite(A_impl).all() and np.isfinite(A_expl).all()
            and np.isfinite(d_impl).all() and np.isfinite(d_expl).all()
            and np.isfinite(f_impl).all() and np.isfinite(f_expl).all()):
        raise ValueError("all tableau and stage-value inputs must be finite")
    if not (np.isscalar(h) or np.ndim(h) == 0) or not np.isfinite(float(h)) \
            or float(h) <= 0.0:
        raise ValueError("h must be a finite positive scalar")
    if int(row_index) != row_index:
        raise ValueError("row_index must be an integer")
    row_index = int(row_index)
    if row_index < 1 or row_index > s:
        raise ValueError("row_index must lie between 1 and the number of stages")
    h = float(h)

    if row_index < s:
        upper_i = A_impl[row_index]
        upper_e = A_expl[row_index]
        lower_i = A_impl[row_index - 1]
        lower_e = A_expl[row_index - 1]
        n_terms = row_index
    else:
        upper_i = d_impl
        upper_e = d_expl
        lower_i = A_impl[s - 2]
        lower_e = A_expl[s - 2]
        n_terms = s

    g = np.zeros(f_impl.shape[1])
    for j in range(n_terms):
        g = g + ((upper_i[j] - lower_i[j]) * f_impl[j]
                 + (upper_e[j] - lower_e[j]) * f_expl[j])

    return h * g

import numpy as np


def extsts_step(y: np.ndarray, h: float, lam_eff: float,
                        D_nodes: np.ndarray, dx: float, c: float, r: float,
                        eps: float, A: float, B: float) -> np.ndarray:
    """Reference implementation."""
    y = np.asarray(y, dtype=float)
    D_nodes = np.asarray(D_nodes, dtype=float)
    if y.ndim != 1 or y.size % 3 != 0:
        raise ValueError("y must be a 1D array whose length is a multiple of 3")
    n = y.size // 3
    if n < 3:
        raise ValueError("each species must have at least 3 nodes")
    if D_nodes.ndim != 1 or D_nodes.size != n:
        raise ValueError("D_nodes must be a 1D array of length y.size // 3")
    if not (np.isfinite(y).all() and np.isfinite(D_nodes).all()):
        raise ValueError("y and D_nodes must contain only finite values")
    if not (np.isscalar(h) or np.ndim(h) == 0) or not np.isfinite(float(h)) \
            or float(h) <= 0.0:
        raise ValueError("h must be a finite positive scalar")
    if not (np.isscalar(lam_eff) or np.ndim(lam_eff) == 0) \
            or not np.isfinite(float(lam_eff)):
        raise ValueError("lam_eff must be a finite scalar")
    if float(lam_eff) < 0.0:
        raise ValueError("lam_eff must be a non-negative magnitude")
    if not (np.isscalar(dx) or np.ndim(dx) == 0) or not np.isfinite(float(dx)) \
            or float(dx) <= 0.0:
        raise ValueError("dx must be a finite positive scalar")
    if not (np.isscalar(eps) or np.ndim(eps) == 0) or not np.isfinite(float(eps)) \
            or float(eps) <= 0.0:
        raise ValueError("eps must be a finite positive scalar")
    for _name, _val in (("c", c), ("r", r), ("A", A), ("B", B)):
        if not (np.isscalar(_val) or np.ndim(_val) == 0) \
                or not np.isfinite(float(_val)):
            raise ValueError(_name + " must be a finite scalar")
    h = float(h)
    lam_eff = float(lam_eff)
    dx = float(dx)
    c = float(c)
    r = float(r)
    eps = float(eps)
    A = float(A)
    B = float(B)

    sq2 = np.sqrt(2.0)
    gam = (2.0 - sq2) / 2.0
    dl = sq2 / 4.0
    Cv = np.array([0.0, 2.0 * gam, 2.0 * gam, 1.0, 1.0, 1.0])
    AI = np.zeros((6, 6))
    AE = np.zeros((6, 6))
    AE[1, 0] = 2.0 * gam
    AE[2, 0] = 2.0 * gam
    AE[3, 0] = (3.0 - 2.0 * sq2) / 6.0
    AE[3, 2] = (3.0 + 2.0 * sq2) / 6.0
    AE[4, 0] = (3.0 - 2.0 * sq2) / 6.0
    AE[4, 2] = (3.0 + 2.0 * sq2) / 6.0
    AE[5, 0] = dl
    AE[5, 2] = dl
    AE[5, 4] = gam
    AI[1, 0] = 2.0 * gam
    AI[2, 0] = gam
    AI[2, 2] = gam
    AI[3, 2] = 1.0
    AI[4, 0] = dl
    AI[4, 2] = dl
    AI[4, 4] = gam
    AI[5, 0] = dl
    AI[5, 2] = dl
    AI[5, 4] = gam
    dI = np.array([(4.0 - sq2) / 8.0, 0.0, (4.0 - sq2) / 8.0, 0.0, dl, 0.0])
    dE = dI.copy()

    D_face = 0.5 * (D_nodes[:-1] + D_nodes[1:])

    def _advective(vec):
        out = np.zeros(3 * n)
        for k in range(3):
            sp = vec[k * n:(k + 1) * n]
            g = np.zeros(n)
            g[1:-1] = -c * (sp[2:] - sp[:-2]) / (2.0 * dx)
            out[k * n:(k + 1) * n] = g
        return out

    def _diffusive(vec):
        out = np.zeros(3 * n)
        for k in range(3):
            sp = vec[k * n:(k + 1) * n]
            g = np.zeros(n)
            g[1:-1] = (D_face[1:] * (sp[2:] - sp[1:-1])
                       - D_face[:-1] * (sp[1:-1] - sp[:-2])) / dx ** 2
            out[k * n:(k + 1) * n] = g
        return out

    def _reactive(vec):
        u = vec[:n]
        v = vec[n:2 * n]
        w = vec[2 * n:]
        out = np.concatenate([r * (A - (w + 1.0) * u + v * u ** 2),
                              r * (w * u - v * u ** 2),
                              r * ((B - w) / eps - w * u)])
        for k in range(3):
            out[k * n] = 0.0
            out[(k + 1) * n - 1] = 0.0
        return out

    def _stage_count(H):
        arg = 0.5 * (np.sqrt(9.0 + 8.0 * H * lam_eff) - 1.0)
        return max(2, int(np.ceil(arg)))

    def _super_step(y_in, H, s_cnt, forcing):
        b = np.zeros(s_cnt + 1)
        b[0] = 1.0 / 3.0
        b[1] = 1.0 / 3.0
        for j in range(2, s_cnt + 1):
            b[j] = (j * j + j - 2.0) / (2.0 * j * (j + 1.0))
        a_co = 1.0 - b
        w1 = 4.0 / (s_cnt * s_cnt + s_cnt - 2.0)
        slope = lambda vec: _diffusive(vec) + forcing
        f0 = slope(y_in)
        y_jm2 = y_in
        y_jm1 = y_in + b[1] * w1 * H * f0
        for j in range(2, s_cnt + 1):
            mu = (2.0 * j - 1.0) / j * b[j] / b[j - 1]
            nu = -(j - 1.0) / j * b[j] / b[j - 2]
            mu_t = mu * w1
            gamma_t = -a_co[j - 1] * mu_t
            y_new = (mu * y_jm1 + nu * y_jm2 + (1.0 - mu - nu) * y_in
                     + mu_t * H * slope(y_jm1) + gamma_t * H * f0)
            y_jm2 = y_jm1
            y_jm1 = y_new
        return y_jm1

    def _reaction_stage(rhs, hg, guess):
        mask = np.ones(n)
        mask[0] = 0.0
        mask[-1] = 0.0
        z = guess.copy()
        for _ in range(60):
            resid = z - hg * _reactive(z) - rhs
            if np.max(np.abs(resid)) < 1e-13:
                break
            u = z[:n]
            v = z[n:2 * n]
            w = z[2 * n:]
            J = np.zeros((n, 3, 3))
            J[:, 0, 0] = r * (-(w + 1.0) + 2.0 * v * u) * mask
            J[:, 0, 1] = r * u ** 2 * mask
            J[:, 0, 2] = -r * u * mask
            J[:, 1, 0] = r * (w - 2.0 * v * u) * mask
            J[:, 1, 1] = -r * u ** 2 * mask
            J[:, 1, 2] = r * u * mask
            J[:, 2, 0] = -r * w * mask
            J[:, 2, 2] = r * (-1.0 / eps - u) * mask
            M = np.tile(np.eye(3), (n, 1, 1)) - hg * J
            Fn = np.stack([resid[:n], resid[n:2 * n], resid[2 * n:]], axis=1)
            dz = np.linalg.solve(M, Fn[..., None])[..., 0]
            z = z - np.concatenate([dz[:, 0], dz[:, 1], dz[:, 2]])
        return z

    m = y.size
    Z = np.zeros((6, m))
    FR = np.zeros((6, m))
    FA = np.zeros((6, m))
    Z[0] = y
    FA[0] = _advective(y)
    FR[0] = _reactive(y)
    for i in range(1, 6):
        g = np.zeros(m)
        for j in range(i):
            g = g + ((AI[i, j] - AI[i - 1, j]) * FR[j]
                     + (AE[i, j] - AE[i - 1, j]) * FA[j])
        g = h * g
        dci = Cv[i] - Cv[i - 1]
        if AI[i, i] == 0.0 and dci > 0.0:
            H = dci * h
            Z[i] = _super_step(Z[i - 1], H, _stage_count(H), g / H)
        elif AI[i, i] == 0.0:
            Z[i] = Z[i - 1] + g
        else:
            Z[i] = _reaction_stage(Z[i - 1] + g, h * AI[i, i], Z[i - 1])
        FA[i] = _advective(Z[i])
        FR[i] = _reactive(Z[i])

    g_emb = np.zeros(m)
    for j in range(6):
        g_emb = g_emb + ((dI[j] - AI[4, j]) * FR[j] + (dE[j] - AE[4, j]) * FA[j])
    g_emb = h * g_emb

    return np.vstack([Z[5], Z[4] + g_emb])

import numpy as np


def run_pipeline(n_nodes: int, n_steps: int, T: float, c: float, d: float,
                         r: float, eps: float, A: float, B: float, d_var: float,
                         seed: int, rtol: float, atol: float, tau: float,
                         q_lambda: float, target_index: int) -> float:
    """Reference implementation."""
    if int(n_nodes) != n_nodes or int(n_nodes) < 3:
        raise ValueError("n_nodes must be an integer of at least 3")
    if int(n_steps) != n_steps or int(n_steps) < 1:
        raise ValueError("n_steps must be a positive integer")
    if not (np.isscalar(T) or np.ndim(T) == 0) or not np.isfinite(float(T)) \
            or float(T) <= 0.0:
        raise ValueError("T must be a finite positive scalar")
    if not (np.isscalar(d) or np.ndim(d) == 0) or not np.isfinite(float(d)) \
            or float(d) <= 0.0:
        raise ValueError("d must be a finite positive scalar")
    if int(target_index) != target_index \
            or not (0 <= int(target_index) < 3 * int(n_nodes)):
        raise ValueError("target_index must be an integer index into the state vector")
    n_nodes = int(n_nodes)
    n_steps = int(n_steps)
    T = float(T)
    target_index = int(target_index)

    sq2 = np.sqrt(2.0)
    gam = (2.0 - sq2) / 2.0
    dl = sq2 / 4.0
    Cv = np.array([0.0, 2.0 * gam, 2.0 * gam, 1.0, 1.0, 1.0])
    AI = np.zeros((6, 6))
    AE = np.zeros((6, 6))
    AE[1, 0] = 2.0 * gam
    AE[2, 0] = 2.0 * gam
    AE[3, 0] = (3.0 - 2.0 * sq2) / 6.0
    AE[3, 2] = (3.0 + 2.0 * sq2) / 6.0
    AE[4, 0] = (3.0 - 2.0 * sq2) / 6.0
    AE[4, 2] = (3.0 + 2.0 * sq2) / 6.0
    AE[5, 0] = dl
    AE[5, 2] = dl
    AE[5, 4] = gam
    AI[1, 0] = 2.0 * gam
    AI[2, 0] = gam
    AI[2, 2] = gam
    AI[3, 2] = 1.0
    AI[4, 0] = dl
    AI[4, 2] = dl
    AI[4, 4] = gam
    AI[5, 0] = dl
    AI[5, 2] = dl
    AI[5, 4] = gam
    dI = np.array([(4.0 - sq2) / 8.0, 0.0, (4.0 - sq2) / 8.0, 0.0, dl, 0.0])
    dE = dI.copy()

    x = np.linspace(0.0, 1.0, n_nodes)
    dx = x[1] - x[0]
    D_nodes = float(d) * (1.0 + float(d_var) * np.sin(2.0 * np.pi * x))
    s0 = 0.1 * np.sin(2.0 * np.pi * x)
    y = np.concatenate([A + s0, B / A + s0, B + s0])

    comp0 = compute_split_operators(y, D_nodes, dx, c, r, eps, A, B)
    if comp0.shape != (3, 3 * n_nodes):
        raise ValueError("split operator returned an unexpected shape")
    for k in range(3):
        if comp0[2][k * n_nodes] != 0.0 or comp0[2][(k + 1) * n_nodes - 1] != 0.0:
            raise ValueError("stationary boundary condition violated in the "
                             "reaction component")

    eig = estimate_dominant_eigenvalue(y, D_nodes, dx, seed, rtol, atol, tau, q_lambda)
    lam = float(eig[0])
    lam_eff = float(eig[1])
    if lam >= 0.0:
        raise ValueError("diffusion eigenvalue estimate must be negative")

    h = T / n_steps
    dcs = np.array([Cv[1] - Cv[0], Cv[3] - Cv[2]])
    counts = select_rkl_stage_counts(h, dcs, lam_eff)
    if np.any(counts < 2):
        raise ValueError("stage counts must be at least 2")

    m = y.size
    Z = np.zeros((6, m))
    FR = np.zeros((6, m))
    FA = np.zeros((6, m))
    Z[0] = y
    FA[0] = comp0[0]
    FR[0] = comp0[2]
    sts_seen = 0
    for i in range(1, 6):
        g = assemble_coupling_vector(h, AI, AE, dI, dE, i, FR, FA)
        dci = Cv[i] - Cv[i - 1]
        if AI[i, i] == 0.0 and dci > 0.0:
            H = dci * h
            Z[i] = rkl2_super_step(Z[i - 1], H, int(counts[sts_seen]), g / H,
                                           D_nodes, dx)
            sts_seen += 1
        elif AI[i, i] == 0.0:
            Z[i] = Z[i - 1] + g
        else:
            Z[i] = solve_reaction_stage(Z[i - 1] + g, h * AI[i, i], Z[i - 1],
                                                r, eps, A, B)
        comp = compute_split_operators(Z[i], D_nodes, dx, c, r, eps, A, B)
        FA[i] = comp[0]
        FR[i] = comp[2]
    if sts_seen != counts.size:
        raise ValueError("stage count vector does not match the tableau structure")
    y = Z[5]

    for _ in range(n_steps - 1):
        out = extsts_step(y, h, lam_eff, D_nodes, dx, c, r, eps, A, B)
        y = out[0]

    return float(y[target_index])
SCICODE_GOLD_EOF
