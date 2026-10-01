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


def _jmul(a, b):
    return np.stack(
        (
            a[0] * b[0],
            a[1] * b[0] + a[0] * b[1],
            a[2] * b[0] + 2 * a[1] * b[1] + a[0] * b[2],
        )
    )


def _jmat(a, b):
    return np.stack(
        (
            a[0] @ b[0],
            a[1] @ b[0] + a[0] @ b[1],
            a[2] @ b[0] + 2 * a[1] @ b[1] + a[0] @ b[2],
        )
    )


def _jtranspose(a):
    return a.swapaxes(-1, -2)


def _jinverse(a):
    v = np.linalg.inv(a[0])
    first = -v @ a[1] @ v
    second = 2 * v @ a[1] @ v @ a[1] @ v - v @ a[2] @ v
    return np.stack((v, first, second))


def _gradient_hessian(r, A, C, cubic, beta, force):
    t = C @ r
    g = A @ r - force + C.T @ (cubic * t * t + beta * t**3)
    H = A + C.T @ ((2 * cubic * t + 3 * beta * t * t)[:, None] * C)
    return g, H


def relaxed_reference_response(
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    guess: "np.ndarray",
    tol: float,
    maxiter: int,
) -> "np.ndarray":
    r = guess.copy()
    threshold = tol * (1 + np.linalg.norm(force[0], ord=np.inf))
    for iteration in range(maxiter + 1):
        g, H = _gradient_hessian(r, A[0], C, cubic, beta[0], force[0])
        residual = np.linalg.norm(g, ord=np.inf)
        if residual <= threshold:
            break
        if iteration == maxiter:
            raise ValueError("equilibrium solve failed")
        delta = np.linalg.solve(H, -g)
        step = 1.0
        for _ in range(30):
            trial = r + step * delta
            newg, _ = _gradient_hessian(trial, A[0], C, cubic, beta[0], force[0])
            if np.linalg.norm(newg, ord=np.inf) <= (1 - 1e-4 * step) * residual:
                r = trial
                break
            step *= 0.5
        else:
            raise ValueError("equilibrium line search failed")
    t = C @ r
    gl = A[1] @ r - force[1] + C.T @ (beta[1] * t**3)
    r1 = np.linalg.solve(H, -gl)
    t1 = C @ r1
    Hl = A[1] + C.T @ ((3 * beta[1] * t * t)[:, None] * C)
    gll = A[2] @ r - force[2] + C.T @ (beta[2] * t**3)
    r2 = np.linalg.solve(
        H, -gll - 2 * Hl @ r1 - C.T @ ((2 * cubic + 6 * beta[0] * t) * t1 * t1)
    )
    t2 = C @ r2
    H1 = A[1] + C.T @ (
        ((2 * cubic + 6 * beta[0] * t) * t1 + 3 * beta[1] * t * t)[:, None] * C
    )
    H2 = A[2] + C.T @ (
        (
            (2 * cubic + 6 * beta[0] * t) * t2
            + 6 * beta[0] * t1 * t1
            + 12 * beta[1] * t * t1
            + 3 * beta[2] * t * t
        )[:, None]
        * C
    )
    return np.concatenate(
        (np.stack((r, r1, r2))[:, :, None], np.stack((H, H1, H2))), axis=2
    )

import numpy as np
from scipy.linalg import solve_sylvester


def _sqrt_jet(a):
    e, U = np.linalg.eigh(a[0])
    if e[0] <= 0:
        raise ValueError("positive definite square-root argument required")
    S = (U * np.sqrt(e)) @ U.T
    S1 = solve_sylvester(S, S, a[1])
    S2 = solve_sylvester(S, S, a[2] - 2 * S1 @ S1)
    return np.stack((S, S1, S2))


def anchored_band_response(
    eq: "np.ndarray", mass: "np.ndarray", anchor: "np.ndarray", lo: float, hi: float
) -> "np.ndarray":
    s, k = anchor.shape
    inv = mass[0] ** -0.5
    sj = np.stack(
        [
            np.diag(inv),
            np.diag(-0.5 * mass[1] * inv),
            np.diag((0.25 * mass[1] ** 2 - 0.5 * mass[2]) * inv),
        ]
    )
    H = eq[:, :, 1:]
    D = _jmat(sj, _jmat(H, sj))
    ev, U = np.linalg.eigh(D[0])
    selected = (ev >= lo**2 - 1e-12) & (ev <= hi**2 + 1e-12)
    if selected.sum() != k:
        raise ValueError("anchor width must equal retained band dimension")
    p = selected.astype(float)
    D1 = U.T @ D[1] @ U
    D2 = U.T @ D[2] @ U
    cross = selected[:, None] != selected[None, :]
    gaps = ev[:, None] - ev[None, :]
    if np.any(np.abs(gaps[cross]) < 1e-9):
        raise ValueError("retained and excluded spectra must be separated")
    P0 = np.diag(p)
    P1 = np.zeros((s, s))
    P1[cross] = ((p[:, None] - p[None, :]) * D1)[cross] / gaps[cross]
    square = P1 @ P1
    P2 = np.zeros_like(P1)
    aa = selected[:, None] & selected[None, :]
    oo = ~selected[:, None] & ~selected[None, :]
    P2[aa] = -2 * square[aa]
    P2[oo] = 2 * square[oo]
    rhs = (p[:, None] - p[None, :]) * D2 + 2 * (P1 @ D1 - D1 @ P1)
    P2[cross] = rhs[cross] / gaps[cross]
    P = np.stack([U @ x @ U.T for x in (P0, P1, P2)])
    X = P @ anchor
    gram = _jmat(_jtranspose(X), X)
    W = _jmat(X, _jinverse(_sqrt_jet(gram)))
    B = _jmat(sj, W)
    K = _jmat(_jtranspose(B), _jmat(H, B))
    return np.concatenate((K, B), axis=1)

import numpy as np


def projected_force_response(
    eq: "np.ndarray",
    ref: "np.ndarray",
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    qjet: "np.ndarray",
) -> "np.ndarray":
    k = qjet.shape[1]
    B = ref[:, k:]
    displacement = _jmat(B, qjet)
    r = eq[:, :, 0] + displacement
    t = r @ C.T
    t2 = _jmul(t, t)
    t3 = _jmul(t2, t)
    grad = _jmat(A, r) - force + (cubic * t2 + _jmul(beta, t3)) @ C
    hweights = 2 * cubic * t + 3 * _jmul(beta, t2)
    H = A + np.stack([C.T @ (v[:, None] * C) for v in hweights])
    Href = eq[:, :, 1:]
    f = _jmat(_jtranspose(B), -grad + _jmat(Href, displacement))
    G = _jmat(_jtranspose(B), _jmat(Href - H, B))
    return np.concatenate((f[:, :, None], G), axis=2)

import numpy as np
from scipy.linalg import expm


def harmonic_frechet_response(
    stiffness: "np.ndarray", h: float
) -> "np.ndarray":
    k = stiffness.shape[1]
    L = np.zeros((3, 2 * k, 2 * k))
    L[0, :k, k:] = np.eye(k)
    L[:, k:, :k] = -stiffness
    size = 2 * k
    lift = np.zeros((3 * size, 3 * size))
    for i in range(3):
        lift[i * size : (i + 1) * size, i * size : (i + 1) * size] = L[0]
    lift[:size, size : 2 * size] = L[1]
    lift[size : 2 * size, 2 * size :] = L[1]
    lift[:size, 2 * size :] = L[2] / 2
    E = expm(h * lift)
    result = np.stack(
        (E[:size, :size], E[:size, size : 2 * size], 2 * E[:size, 2 * size :])
    )
    return result

import numpy as np


def _kick_response(jet, eq, ref, A, C, cubic, beta, force, drive, tau):
    k = jet.shape[1] // 2
    response = projected_force_response(
        eq, ref, A, C, cubic, beta, force, jet[:, :k, 0]
    )
    f = response[:, :, 0] + _jmat(_jtranspose(ref[:, k:]), drive)
    G = response[:, :, 1:]
    out = jet.copy()
    out[:, k:, 0] += tau * f
    out[:, k:, 1:] += tau * _jmat(G, jet[:, :k, 1:])
    return out


def driven_band_jet_step(
    jet: "np.ndarray",
    eq: "np.ndarray",
    ref: "np.ndarray",
    R: "np.ndarray",
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    drive0: "np.ndarray",
    drive1: "np.ndarray",
    h: float,
) -> "np.ndarray":
    first = _kick_response(jet, eq, ref, A, C, cubic, beta, force, drive0, h / 2)
    middle = _jmat(R, first)
    result = _kick_response(middle, eq, ref, A, C, cubic, beta, force, drive1, h / 2)
    return result

import numpy as np


def _period_response(initial, eq, ref, R, A, C, cubic, beta, force, drives, h):
    dim = initial.shape[1]
    jet = np.zeros((3, dim, 1 + dim))
    jet[:, :, 0] = initial
    jet[0, :, 1:] = np.eye(dim)
    for i in range(len(drives) - 1):
        jet = driven_band_jet_step(
            jet, eq, ref, R, A, C, cubic, beta, force, drives[i], drives[i + 1], h
        )
    return jet


def periodic_orbit_response(
    eq: "np.ndarray",
    ref: "np.ndarray",
    R: "np.ndarray",
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    drives: "np.ndarray",
    h: float,
    tol: float,
    maxiter: int,
) -> "np.ndarray":
    k = ref.shape[2]
    dim = 2 * k
    initial = np.zeros((3, dim))
    for amplitude in np.arange(1, 9) / 8:
        drive = drives * amplitude
        for iteration in range(maxiter + 1):
            end = _period_response(
                initial, eq, ref, R, A, C, cubic, beta, force, drive, h
            )
            residual = end[0, :, 0] - initial[0]
            norm = np.linalg.norm(residual, ord=np.inf)
            if norm <= tol:
                break
            if iteration == maxiter:
                raise ValueError("periodic shooting failed")
            delta = np.linalg.solve(end[0, :, 1:] - np.eye(dim), -residual)
            alpha = 1.0
            for _ in range(25):
                trial = initial.copy()
                trial[0] += alpha * delta
                trialend = _period_response(
                    trial, eq, ref, R, A, C, cubic, beta, force, drive, h
                )
                trialnorm = np.linalg.norm(trialend[0, :, 0] - trial[0], ord=np.inf)
                if trialnorm <= (1 - 1e-4 * alpha) * norm:
                    initial = trial
                    break
                alpha *= 0.5
            else:
                raise ValueError("shooting line search failed")
    end = _period_response(initial, eq, ref, R, A, C, cubic, beta, force, drives, h)
    boundary = np.eye(dim) - end[0, :, 1:]
    initial[1] = np.linalg.solve(boundary, end[1, :, 0])
    first = _period_response(initial, eq, ref, R, A, C, cubic, beta, force, drives, h)
    initial[2] = np.linalg.solve(boundary, first[2, :, 0])
    total = _period_response(initial, eq, ref, R, A, C, cubic, beta, force, drives, h)
    result = total.copy()
    result[:, :, 0] = initial
    return result

import numpy as np


def metric_stretching_curvature(
    monodromy: "np.ndarray", stiffness: "np.ndarray", T: float
) -> "np.ndarray":
    k = stiffness.shape[1]
    root = _sqrt_jet(stiffness)
    D = np.zeros((3, 2 * k, 2 * k))
    D[:, :k, :k] = root
    D[0, k:, k:] = np.eye(k)
    scaled = _jmat(D, _jmat(monodromy, _jinverse(D)))
    gram = _jmat(_jtranspose(scaled), scaled)
    vals, V = np.linalg.eigh(gram[0])
    lam = vals[-1]
    v = V[:, -1]
    if lam <= 0 or lam - vals[-2] <= 1e-9 * lam:
        raise ValueError("positive simple leading squared singular value required")
    first = v @ gram[1] @ v
    rhs = gram[1] @ v - first * v
    v1 = np.linalg.solve(lam * np.eye(2 * k) - gram[0] + np.outer(v, v), rhs)
    second = v @ gram[2] @ v + 2 * v1 @ gram[1] @ v
    return np.array(
        [
            np.sqrt(lam),
            np.log(lam) / (2 * T),
            first / (2 * T * lam),
            (second / lam - (first / lam) ** 2) / (2 * T),
        ]
    )

import numpy as np


def phase_averaged_curvature(
    A: "np.ndarray",
    C: "np.ndarray",
    cubic: "np.ndarray",
    beta: "np.ndarray",
    force: "np.ndarray",
    mass: "np.ndarray",
    anchor: "np.ndarray",
    lo: float,
    hi: float,
    cosdrive: "np.ndarray",
    sindrive: "np.ndarray",
    h: float,
    n: int,
) -> float:
    eq = relaxed_reference_response(
        A, C, cubic, beta, force, np.zeros(A.shape[1]), 1e-13, 40
    )
    ref = anchored_band_response(eq, mass, anchor, lo, hi)
    k = anchor.shape[1]
    stiffness = ref[:, :k]
    R = harmonic_frechet_response(stiffness, h)
    phase = 2 * np.pi * np.arange(n + 1) / n
    drives = (
        np.cos(phase)[:, None, None] * cosdrive
        + np.sin(phase)[:, None, None] * sindrive
    )
    drives[-1] = drives[0]
    orbit = periodic_orbit_response(
        eq, ref, R, A, C, cubic, beta, force, drives, h, 1e-12, 30
    )
    prefix = np.zeros_like(orbit)
    prefix[:, :, 0] = orbit[:, :, 0]
    prefix[0, :, 1:] = np.eye(2 * k)
    total = 0.0
    for j in range(n):
        P = prefix[:, :, 1:]
        shifted = _jmat(P, _jmat(orbit[:, :, 1:], _jinverse(P)))
        metrics = metric_stretching_curvature(shifted, stiffness, n * h)
        total += metrics[3]
        prefix = driven_band_jet_step(
            prefix, eq, ref, R, A, C, cubic, beta, force, drives[j], drives[j + 1], h
        )
    return float(total / n)
SCICODE_GOLD_EOF
