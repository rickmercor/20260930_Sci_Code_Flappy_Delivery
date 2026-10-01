#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def basis_integrals(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                            potential_width: float, cap_onset: float) -> "np.ndarray":
    import numpy as np
    from scipy.special import erfc
    if not np.isfinite(alpha0) or alpha0 <= 0.0:
        raise ValueError("alpha0 must be a positive finite number")
    if not np.isfinite(beta) or beta <= 1.0:
        raise ValueError("beta must be greater than 1")
    if int(n_basis) != n_basis or n_basis < 2:
        raise ValueError("n_basis must be an integer of at least 2")
    if not np.isfinite(barrier_strength) or barrier_strength < 0.0:
        raise ValueError("barrier_strength must be a non-negative finite number")
    if not np.isfinite(potential_width) or potential_width <= 0.0:
        raise ValueError("potential_width must be a positive finite number")
    if not np.isfinite(cap_onset) or cap_onset < 0.0:
        raise ValueError("cap_onset must be a non-negative finite number")

    n = int(n_basis)
    alpha = float(alpha0) * float(beta) ** np.arange(n)
    p = alpha[:, None] + alpha[None, :]

    S = np.sqrt(np.pi / p)
    T = (alpha[:, None] * alpha[None, :]) * np.sqrt(np.pi) / p ** 1.5
    q = p + float(potential_width)
    V = float(barrier_strength) * np.sqrt(np.pi) / (2.0 * q ** 1.5)
    X2 = np.sqrt(np.pi) / (2.0 * p ** 1.5)

    c = float(cap_onset)
    J0 = 0.5 * np.sqrt(np.pi / p) * erfc(c * np.sqrt(p))
    J1 = np.exp(-p * c * c) / (2.0 * p)
    J2 = c * np.exp(-p * c * c) / (2.0 * p) + J0 / (2.0 * p)
    W = 2.0 * (J2 - 2.0 * c * J1 + c * c * J0)

    return np.stack([S, T + V, W, X2])

def canonical_transform(overlap: "np.ndarray", threshold: float) -> "np.ndarray":
    import numpy as np
    S = np.asarray(overlap, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("overlap must be a square two-dimensional array")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be a positive finite number")

    s, U = np.linalg.eigh(S)
    keep = s > float(threshold)
    if not keep.any():
        raise ValueError("no overlap eigenvalue exceeds the threshold")

    X = U[:, keep] / np.sqrt(s[keep])
    for k in range(X.shape[1]):
        col = X[:, k]
        lead = int(np.argmax(np.abs(col)))
        if col[lead] < 0.0:
            X[:, k] = -col
    return X

def _check_cap_inputs(integrals, transform, eta):
    import numpy as np
    I = np.asarray(integrals, dtype=float)
    X = np.asarray(transform, dtype=float)
    if I.ndim != 3 or I.shape[0] != 4 or I.shape[1] != I.shape[2]:
        raise ValueError("integrals must have shape (4, n, n)")
    if X.ndim != 2 or X.shape[0] != I.shape[1]:
        raise ValueError("transform must have as many rows as the primitive basis size")
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be a non-negative finite number")
    return I, X, float(eta)


def _cap_solve(integrals, transform, eta):
    """Sorted eigenvalues and c-product normalised primitive-basis eigenvectors."""
    import numpy as np
    import scipy.linalg as sla
    I, X, e = _check_cap_inputs(integrals, transform, eta)
    S, H0, W = I[0], I[1], I[2]
    vals, Cp = sla.eig(X.T @ (H0 - 1j * e * W) @ X)
    C = X @ Cp
    norms = np.einsum('ij,jk,ki->i', C.T, S, C)
    C = C / np.sqrt(norms)
    order = np.lexsort((vals.imag, vals.real))
    return vals[order], C[:, order]


def cap_eigenvalues(integrals: "np.ndarray", transform: "np.ndarray", eta: float) -> "np.ndarray":
    import numpy as np
    return _cap_solve(integrals, transform, eta)[0]

def resonance_extents(integrals: "np.ndarray", transform: "np.ndarray", eta: float) -> "np.ndarray":
    import numpy as np
    I = np.asarray(integrals, dtype=float)
    # Step 3 defines the public ordering of the CAP states. The eigensolver
    # also supplies the vectors needed here, so align those vectors to the
    # eigenvalues returned by Step 3 before forming the extents.
    target = np.asarray(cap_eigenvalues(integrals, transform, eta), dtype=complex)
    vals, C_raw = _cap_solve(integrals, transform, eta)
    remaining = list(range(vals.size))
    order = []
    for value in target:
        j = min(remaining, key=lambda k: abs(vals[k] - value))
        order.append(j)
        remaining.remove(j)
    C = C_raw[:, order]
    return np.einsum('ij,jk,ki->i', C.T, I[3], C).real

def _check_grid(etas, e_max):
    import numpy as np
    g = np.asarray(etas, dtype=float)
    if g.ndim != 1 or g.size < 3:
        raise ValueError("etas must be a one-dimensional grid of at least 3 points")
    if not np.all(np.isfinite(g)) or g[0] < 0.0:
        raise ValueError("etas must be finite and non-negative")
    if not np.all(np.diff(g) > 0.0):
        raise ValueError("etas must be strictly increasing")
    if not np.isfinite(e_max) or e_max <= 0.0:
        raise ValueError("e_max must be a positive finite number")
    return g


def _track_branch(integrals, transform, etas, eta_seed, e_max):
    """Follow the resonance outwards from the seed; return E, dE/deta and the vectors."""
    import numpy as np
    g = _check_grid(etas, e_max)
    I = np.asarray(integrals, dtype=float)
    S, W = I[0], I[2]

    js = int(np.argmin(np.abs(g - float(eta_seed))))
    vals, C = _cap_solve(integrals, transform, g[js])
    r2 = resonance_extents(integrals, transform, g[js])
    window = np.where((vals.real > 0.0) & (vals.real < float(e_max)))[0]
    if window.size == 0:
        raise ValueError("no eigenvalue at the seed lies between zero and e_max")
    k = int(window[int(np.argmin(r2[window]))])

    n = g.size
    E = np.empty(n, dtype=complex)
    D = np.empty(n, dtype=complex)
    vecs = [None] * n

    def _record(i, vals_i, C_i, idx):
        c = C_i[:, idx]
        E[i] = vals_i[idx]
        D[i] = -1j * (c @ W @ c)
        vecs[i] = c
        return c

    prev = _record(js, vals, C, k)
    for i in range(js - 1, -1, -1):
        vals_i, C_i = _cap_solve(integrals, transform, g[i])
        idx = int(np.argmax(np.abs(prev @ S @ C_i)))
        prev = _record(i, vals_i, C_i, idx)
    prev = vecs[js]
    for i in range(js + 1, n):
        vals_i, C_i = _cap_solve(integrals, transform, g[i])
        idx = int(np.argmax(np.abs(prev @ S @ C_i)))
        prev = _record(i, vals_i, C_i, idx)
    return E, D, vecs


def branch_energies(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                            eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    return _track_branch(integrals, transform, etas, eta_seed, e_max)[0]

def branch_velocity(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                            eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    g = np.asarray(etas, dtype=float)
    I = np.asarray(integrals, dtype=float)
    W = I[2]
    # Step 5 supplies the energy of the tracked state at every grid point.
    # Match those energies to the eigensystems and evaluate the analytic
    # Hellmann-Feynman derivative for the selected state.
    energies = np.asarray(
        branch_energies(integrals, transform, etas, eta_seed, e_max),
        dtype=complex,
    )
    D = np.empty(g.size, dtype=complex)
    for i, eta in enumerate(g):
        vals, C = _cap_solve(integrals, transform, float(eta))
        k = int(np.argmin(np.abs(vals - energies[i])))
        D[i] = -1j * (C[:, k] @ W @ C[:, k])
    return np.abs(g * D)

def _velocity_minima(velocity):
    import numpy as np
    v = np.asarray(velocity, dtype=float)
    return [i for i in range(1, v.size - 1) if v[i] < v[i - 1] and v[i] < v[i + 1]]


def _parabolic_eta(etas, velocity, i):
    """Vertex of the parabola through the three bracketing points in ln(eta)."""
    import numpy as np
    x0, x1, x2 = np.log(etas[i - 1]), np.log(etas[i]), np.log(etas[i + 1])
    y0, y1, y2 = velocity[i - 1], velocity[i], velocity[i + 1]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / den
    return float(np.exp(-b / (2.0 * a)))


def solution_etas(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                          eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    g = np.asarray(etas, dtype=float)
    v = branch_velocity(integrals, transform, etas, eta_seed, e_max)
    return np.array([_parabolic_eta(g, v, i) for i in _velocity_minima(v)], dtype=float)

def deperturbed_widths(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                               eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    g = np.asarray(etas, dtype=float)
    I = np.asarray(integrals, dtype=float)
    S, W = I[0], I[2]
    _, D, vecs = _track_branch(integrals, transform, etas, eta_seed, e_max)
    v = np.abs(g * D)

    widths = []
    for i in _velocity_minima(v):
        eta_opt = _parabolic_eta(g, v, i)
        vals, C = _cap_solve(integrals, transform, eta_opt)
        idx = int(np.argmax(np.abs(vecs[i] @ S @ C)))
        c = C[:, idx]
        energy = vals[idx]
        deriv = -1j * (c @ W @ c)
        widths.append(-2.0 * (energy - eta_opt * deriv).imag)
    return np.array(widths, dtype=float)

def _slope_at(integrals, transform, eta, reference):
    """dv/dln(eta) of the state of largest c-product overlap with reference, and its vector."""
    import numpy as np
    I = np.asarray(integrals, dtype=float)
    S, W = I[0], I[2]
    vals, C = _cap_solve(integrals, transform, eta)
    k = int(np.argmax(np.abs(reference @ S @ C)))
    Wm = C.T @ W @ C
    d1 = -1j * Wm[k, k]
    gaps = vals[k] - vals
    gaps[k] = np.inf
    d2 = -2.0 * np.sum(Wm[k, :] ** 2 / gaps)
    q = eta * d1
    dq = eta * d1 + eta * eta * d2
    return float((np.conj(q) * dq).real / abs(q)), C[:, k]


def branch_velocity_slope(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                                  eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    g = np.asarray(etas, dtype=float)
    if g.ndim == 1 and g.size >= 1 and np.all(np.isfinite(g)) and g[0] <= 0.0:
        raise ValueError("etas must be strictly positive")
    _, _, vecs = _track_branch(integrals, transform, etas, eta_seed, e_max)
    return np.array([_slope_at(integrals, transform, float(g[i]), vecs[i])[0]
                     for i in range(g.size)], dtype=float)

def _remover_shift(integrals, lam):
    import numpy as np
    J = np.array(integrals, dtype=float, copy=True)
    J[1] = J[1] + float(lam) * J[2]
    return J


def _branch_from(integrals, transform, etas, index, reference):
    """Velocity and eigenvectors of the branch through the state nearest reference at etas[index]."""
    import numpy as np
    I = np.asarray(integrals, dtype=float)
    S, W = I[0], I[2]
    g = np.asarray(etas, dtype=float)
    n = g.size
    vel = np.empty(n, dtype=float)
    vecs = [None] * n

    def _record(i, C_i, idx):
        c = C_i[:, idx]
        vel[i] = abs(g[i] * (-1j) * (c @ W @ c))
        vecs[i] = c
        return c

    vals, C = _cap_solve(integrals, transform, g[index])
    prev = _record(index, C, int(np.argmax(np.abs(np.asarray(reference) @ S @ C))))
    for i in range(index - 1, -1, -1):
        vals_i, C_i = _cap_solve(integrals, transform, g[i])
        prev = _record(i, C_i, int(np.argmax(np.abs(prev @ S @ C_i))))
    prev = vecs[index]
    for i in range(index + 1, n):
        vals_i, C_i = _cap_solve(integrals, transform, g[i])
        prev = _record(i, C_i, int(np.argmax(np.abs(prev @ S @ C_i))))
    return vel, vecs


def _fold_window(etas, velocity, eta_ref, slope=None):
    """Grid minimum nearest eta_ref in ln(eta), plus the maximum immediately below it."""
    import numpy as np
    g = np.asarray(etas, dtype=float)
    v = np.asarray(velocity, dtype=float)
    minima = _velocity_minima(v)
    if slope is not None:
        s = np.asarray(slope, dtype=float)
        if s.shape != v.shape:
            raise ValueError("velocity and slope must have the same shape")
        # A sampled minimum represents a smooth stationary minimum only when
        # the analytic slope changes from negative to positive nearby.
        minima = [i for i in minima
                  if np.min(s[max(0, i - 1):i + 1]) <= 0.0
                  and np.max(s[i:min(s.size, i + 2)]) >= 0.0]
    if len(minima) == 0:
        return None
    i = min(minima, key=lambda m: abs(np.log(g[m]) - np.log(float(eta_ref))))
    left = i
    while left > 0 and v[left - 1] >= v[left]:
        left -= 1
    return float(g[left]), float(g[i]), int(i)


def _fold_depth(integrals, transform, etas, vecs, eta_lo, eta_hi):
    """Smallest value of the smooth slope dv/dln(eta) over [eta_lo, eta_hi]."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    g = np.asarray(etas, dtype=float)
    idx = np.where((g >= eta_lo) & (g <= eta_hi))[0]
    if idx.size == 0:
        raise ValueError("the window contains no grid point")
    slope = np.array([_slope_at(integrals, transform, g[i], vecs[i])[0] for i in idx])
    j = int(np.argmin(slope))
    s = np.log(g)
    a = s[idx[max(j - 1, 0)]]
    b = s[idx[min(j + 1, idx.size - 1)]]
    best, eta_best = float(slope[j]), float(g[idx[j]])
    if b > a:
        res = minimize_scalar(lambda t: _slope_at(integrals, transform, float(np.exp(t)), vecs[idx[j]])[0],
                              bounds=(a, b), method='bounded', options={'xatol': 1e-13})
        if float(res.fun) < best:
            best, eta_best = float(res.fun), float(np.exp(res.x))
    return eta_best, best


def solution_fold_strength(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                                   potential_width: float, cap_onset: float, threshold: float,
                                   eta_min: float, eta_max: float, n_eta: int, eta_seed: float,
                                   eta_ref: float, kappa_step: float, kappa_max: float) -> float:
    import numpy as np
    from scipy.optimize import brentq
    if not np.isfinite(eta_ref) or eta_ref <= 0.0:
        raise ValueError("eta_ref must be positive")
    if not np.isfinite(kappa_step) or kappa_step <= 0.0:
        raise ValueError("kappa_step must be positive")
    if not np.isfinite(kappa_max) or kappa_max <= kappa_step:
        raise ValueError("kappa_max must exceed kappa_step")
    if not np.isfinite(eta_min) or not np.isfinite(eta_max) or eta_min <= 0.0 or eta_max <= eta_min:
        raise ValueError("need 0 < eta_min < eta_max")
    if int(n_eta) < 3:
        raise ValueError("n_eta must be at least 3")
    e_max = float(barrier_strength) / (float(potential_width) * np.e)
    I = basis_integrals(alpha0, beta, n_basis, barrier_strength, potential_width, cap_onset)
    X = canonical_transform(I[0].copy(), threshold)
    etas = np.geomspace(float(eta_min), float(eta_max), int(n_eta))

    energies = branch_energies(I, X, etas, eta_seed, e_max)
    velocity = branch_velocity(I, X, etas, eta_seed, e_max)
    slope = branch_velocity_slope(I, X, etas, eta_seed, e_max)
    window = _fold_window(etas, velocity, eta_ref, slope)
    if window is None:
        raise ValueError("the unshifted problem has no CAP solution")
    lo, hi, anchor = window
    vals, C = _cap_solve(I, X, float(etas[anchor]))
    cref = C[:, int(np.argmin(np.abs(vals - energies[anchor])))]

    kappa = 0.0
    step = float(kappa_step)
    upper = float(kappa_max)
    while kappa < upper:
        nxt = min(kappa + step, upper)
        J = _remover_shift(I, nxt)
        vel_n, vecs_n = _branch_from(J, X, etas, anchor, cref)
        if _fold_depth(J, X, etas, vecs_n, lo, hi)[1] >= 0.0:
            break
        kappa = nxt
        window = _fold_window(etas, vel_n, etas[anchor])
        if window is None:
            raise ValueError("the followed branch was lost before it coalesced")
        lo, hi, anchor = window
        cref = vecs_n[anchor]
    else:
        raise ValueError("the chosen solution still exists at kappa_max")

    def _depth(lam):
        J = _remover_shift(I, lam)
        _, vv = _branch_from(J, X, etas, anchor, cref)
        return _fold_depth(J, X, etas, vv, lo, hi)[1]

    f_lo, f_hi = _depth(kappa), _depth(nxt)
    if not (f_lo < 0.0 < f_hi):
        raise ValueError("the solution does not vanish by coalescence inside the bracketing interval")
    kappa_c = float(brentq(_depth, kappa, nxt, xtol=1e-14, rtol=4.0 * np.finfo(float).eps))
    if abs(_depth(kappa_c)) > 1e-8:
        raise ValueError("the coalescence condition is not satisfied at the located strength")
    return kappa_c

def remover_survival_strength(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                                      potential_width: float, cap_onset: float, threshold: float,
                                      eta_min: float, eta_max: float, n_eta: int, eta_seed: float,
                                      kappa_step: float, kappa_max: float) -> float:
    import numpy as np
    e_max = float(barrier_strength) / (float(potential_width) * np.e)
    I = basis_integrals(alpha0, beta, n_basis, barrier_strength, potential_width, cap_onset)
    X = canonical_transform(I[0].copy(), threshold)
    etas = np.geomspace(float(eta_min), float(eta_max), int(n_eta))
    eta_opt = np.asarray(solution_etas(I, X, etas, eta_seed, e_max), dtype=float)
    if eta_opt.size == 0:
        raise ValueError("the unshifted problem has no CAP solution")
    widths = np.asarray(deperturbed_widths(I, X, etas, eta_seed, e_max), dtype=float)
    if widths.size != eta_opt.size:
        raise ValueError("the solutions and their first-order widths disagree in number")
    order = np.argsort(-widths, kind='stable')
    best, best_width = None, None
    for k in order:
        fold = solution_fold_strength(alpha0, beta, n_basis, barrier_strength, potential_width,
                                              cap_onset, threshold, eta_min, eta_max, n_eta, eta_seed,
                                              float(eta_opt[k]), kappa_step, kappa_max)
        if best is None or fold > best + 1e-12 or (abs(fold - best) <= 1e-12 and widths[k] > best_width):
            best, best_width = fold, float(widths[k])
    return float(best)
SCICODE_GOLD_EOF
