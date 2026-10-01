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
def assemble_action_derivatives(beads: "np.ndarray", surface: "np.ndarray", beta: float, tau: float,
                                        order: int) -> "np.ndarray":
    """Reference implementation: closed-form surface derivatives on y = 0 and tau-differentiated coefficients."""
    import math
    import numpy as np

    X = np.asarray(beads, dtype=float)
    s = np.asarray(surface, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2 or X.shape[0] < 4 or X.shape[0] % 2 or not np.all(np.isfinite(X)):
        raise ValueError("beads must be a finite (N, 2) array with even N >= 4")
    if np.any(X[:, 1] != 0.0):
        raise ValueError("every bead must lie on the line y = 0")
    if s.shape != (7,) or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("surface must hold seven finite positive numbers")
    if not (math.isfinite(beta) and math.isfinite(tau) and 0.0 < tau < beta):
        raise ValueError("tau must satisfy 0 < tau < beta")
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order <= 4:
        raise ValueError("order must be an integer from 0 to 4")
    v0, a, m, omega, chi_inf, chi_0, sigma = s
    n, half, order = X.shape[0], X.shape[0] // 2, int(order)
    tail = beta - tau
    # Spring constants m / d scale as 1 / tau and 1 / (beta - tau); potential weights are linear in tau.
    k_a = math.factorial(order) * (-1) ** order * m * half / tau ** (order + 1)
    k_b = math.factorial(order) * m * half / tail ** (order + 1)
    springs = np.array([k_a] * half + [k_b] * half)
    w_a = (tau / half, 1.0 / half, 0.0, 0.0, 0.0)[order]
    w_b = (tail / half, -1.0 / half, 0.0, 0.0, 0.0)[order]
    weights = np.array([0.5 * (w_a + w_b)] + [w_a] * (half - 1) + [0.5 * (w_a + w_b)] + [w_b] * (half - 1))
    x = X[:, 0]
    e = np.exp(-2.0 * np.abs(x) / a)
    sech2 = 4.0 * e / (1.0 + e) ** 2
    t = np.tanh(x / a)
    eck = [sech2, -2.0 * t * sech2, sech2 * (6.0 * t**2 - 2.0), t * sech2 * (16.0 - 24.0 * t**2),
           sech2 * (16.0 - 120.0 * t**2 + 120.0 * t**4)]
    eck = [v0 * d / a**k for k, d in enumerate(eck)]
    bump = np.exp(-x**2 / (2.0 * sigma**2))
    chi = chi_inf + (chi_0 - chi_inf) * bump
    dchi = -(x / sigma**2) * (chi_0 - chi_inf) * bump
    # Morse Taylor coefficients on y = 0: -D b^3 y^3 and (7/12) D b^4 y^4 with D b^2 = m omega^2 / 2.
    cubic = -1.5 * omega * (2.0 * m * omega) ** 1.5
    third = np.zeros((n, 2, 2, 2))
    third[:, 0, 0, 0] = eck[3]
    third[:, 1, 1, 1] = cubic * np.sqrt(chi)
    fourth = np.zeros((n, 2, 2, 2, 2))
    fourth[:, 0, 0, 0, 0] = eck[4]
    fourth[:, 1, 1, 1, 1] = 14.0 * m**2 * omega**3 * chi
    for idx in ((0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)):
        fourth[(slice(None),) + idx] = cubic * dchi / (2.0 * np.sqrt(chi))
    step = np.roll(X, -1, axis=0) - X
    action = 0.5 * np.sum(springs * np.sum(step**2, axis=1)) + np.sum(weights * eck[0])
    grad = np.roll(springs, 1)[:, None] * (X - np.roll(X, 1, axis=0)) - springs[:, None] * step
    grad[:, 0] += weights * eck[1]
    hess = np.zeros((2 * n, 2 * n))
    for i in range(n):
        j = (i + 1) % n
        local = np.diag([eck[2][i], m * omega**2])
        hess[2 * i:2 * i + 2, 2 * i:2 * i + 2] += (springs[i] + springs[i - 1]) * np.eye(2) + weights[i] * local
        hess[2 * i:2 * i + 2, 2 * j:2 * j + 2] -= springs[i] * np.eye(2)
        hess[2 * j:2 * j + 2, 2 * i:2 * i + 2] -= springs[i] * np.eye(2)
    return np.concatenate([[action], grad.ravel(), hess.ravel(),
                           (weights[:, None, None, None] * third).ravel(),
                           (weights[:, None, None, None, None] * fourth).ravel()])

def locate_flux_instanton(surface: "np.ndarray", beta: float, tau: float, n_beads: int) -> "np.ndarray":
    """Reference implementation: Newton search from the continuum Eckart orbit, then implicit differentiation."""
    import math
    import numpy as np

    if isinstance(n_beads, bool) or not isinstance(n_beads, (int, np.integer)) or n_beads < 4 or n_beads % 2:
        raise ValueError("n_beads must be an even integer of at least 4")
    s = np.asarray(surface, dtype=float)
    if s.shape != (7,) or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("surface must hold seven finite positive numbers")
    n, half = int(n_beads), int(n_beads) // 2
    v0, a, m = s[0], s[1], s[2]
    gamma = math.pi * math.sqrt(2.0 * m * a**2 * v0)
    if not (math.isfinite(beta) and beta * v0 > gamma):
        raise ValueError("beta must exceed 2 pi / omega_b")

    def _pieces(order, Y):
        flat = assemble_action_derivatives(Y, s, beta, tau, order)
        k = 1 + 2 * n + 4 * n * n
        return flat[1:1 + 2 * n], flat[1 + 2 * n:k].reshape(2 * n, 2 * n), flat[k:k + 8 * n].reshape(n, 2, 2, 2)

    # The continuum periodic orbit of the Eckart barrier is sinh(x/a) proportional to sin(2 pi t / beta).
    energy = gamma**2 / (v0 * beta**2)
    X = np.zeros((n, 2))
    X[:, 0] = a * np.arcsinh(math.sqrt((v0 - energy) / energy) * np.sin(2.0 * math.pi * np.arange(n) / n))
    X[0, 0] = X[half, 0] = 0.0
    free = np.ones(2 * n, bool)
    free[[0, 2 * half]] = False
    size = np.inf
    for _ in range(60):
        grad, hess, _ = _pieces(0, X)
        move = np.zeros(2 * n)
        move[free] = np.linalg.solve(hess[np.ix_(free, free)], -grad[free])
        X = X + move.reshape(n, 2)
        previous, size = size, float(np.max(np.abs(move)))
        if size < 1e-15 * a or (size < 1e-11 * a and size >= previous):
            break
    else:
        raise ValueError("stationary path search did not converge")
    if not (np.all(X[1:half, 0] > 0.0) and np.all(X[half + 1:, 0] < 0.0)):
        raise ValueError("no stationary path with the required sign pattern")
    _, hess0, third = _pieces(0, X)
    grad1, hess1, _ = _pieces(1, X)
    grad2, _, _ = _pieces(2, X)
    inverse = np.linalg.inv(hess0[np.ix_(free, free)])
    first = np.zeros(2 * n)
    first[free] = -inverse @ grad1[free]
    V1 = first.reshape(n, 2)
    drive = np.einsum("iabc,ib,ic->ia", third, V1, V1).ravel() + 2.0 * hess1 @ first + grad2
    second = np.zeros(2 * n)
    second[free] = -inverse @ drive[free]
    return np.stack([X, V1, second.reshape(n, 2)])

def compute_spatial_correction(surface: "np.ndarray", beta: float, tau: float, n_beads: int) -> float:
    """Reference implementation: Green's-function contractions of the bead-local derivative tensors."""
    import numpy as np

    s = np.asarray(surface, dtype=float)
    X = locate_flux_instanton(s, beta, tau, n_beads)[0]
    n, half, m = X.shape[0], X.shape[0] // 2, s[2]
    flat = assemble_action_derivatives(X, s, beta, tau, 0)
    k = 1 + 2 * n + 4 * n * n
    hess = flat[1 + 2 * n:k].reshape(2 * n, 2 * n)
    third = flat[k:k + 8 * n].reshape(n, 2, 2, 2)
    fourth = flat[k + 8 * n:].reshape(n, 2, 2, 2, 2)
    free = np.ones(2 * n, bool)
    free[[0, 2 * half]] = False
    # The inverse Hessian over the free coordinates, padded with zeros on the pinned ones.
    G = np.zeros_like(hess)
    G[np.ix_(free, free)] = np.linalg.inv(hess[np.ix_(free, free)])
    blocks = G.reshape(n, 2, n, 2).transpose(0, 2, 1, 3)
    local = blocks[np.arange(n), np.arange(n)]
    quartic = -0.125 * np.einsum("iabcd,iab,icd->", fourth, local, local)
    tadpole = np.einsum("iabc,ibc->ia", third, local).ravel()
    cubic = 0.125 * tadpole @ G @ tadpole
    cubic += sum(np.einsum("abc,jad,jbe,jcf,jdef->", third[i], blocks[i], blocks[i], blocks[i], third)
                 for i in range(n)) / 12.0
    d_a, d_b = 2.0 * tau / n, 2.0 * (beta - tau) / n
    ends = ((0, 1, m / d_a), (half - 1, half, m / d_a), (half, half + 1, m / d_b), (n - 1, 0, m / d_b))
    p = np.zeros(4)
    dp = np.zeros((4, 2 * n))
    for j, (i0, i1, scale) in enumerate(ends):
        p[j] = scale * (X[i1, 0] - X[i0, 0])
        dp[j, 2 * i1] += scale
        dp[j, 2 * i0] -= scale
    pair = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=float)
    phi = 0.5 * p @ pair @ p
    dphi = dp.T @ pair @ p
    ddphi = dp.T @ pair @ dp
    flux = -0.5 * (dphi @ G @ tadpole) / phi + 0.5 * np.sum(ddphi * G) / phi
    return float(quartic + cubic + flux)

def compute_action_time_derivatives(surface: "np.ndarray", beta: float, tau: float,
                                            n_beads: int) -> "np.ndarray":
    """Reference implementation: chain rule with the stationarity condition eliminating higher path derivatives."""
    import numpy as np

    s = np.asarray(surface, dtype=float)
    X, V1, V2 = locate_flux_instanton(s, beta, tau, n_beads)
    n = X.shape[0]
    k = 1 + 2 * n + 4 * n * n
    parts = []
    for order in range(5):
        flat = assemble_action_derivatives(X, s, beta, tau, order)
        parts.append((flat[0], flat[1:1 + 2 * n], flat[1 + 2 * n:k].reshape(2 * n, 2 * n),
                      flat[k:k + 8 * n].reshape(n, 2, 2, 2), flat[k + 8 * n:].reshape(n, 2, 2, 2, 2)))
    v1, v2 = V1.ravel(), V2.ravel()

    def _cube(t3, A, B, C):
        return np.einsum("iabc,ia,ib,ic->", t3, A, B, C)

    # Terms carrying the third and fourth path derivatives cancel because the path is stationary.
    w1 = parts[1][0]
    w2 = parts[2][0] + parts[1][1] @ v1
    w3 = (parts[3][0] + 3.0 * parts[2][1] @ v1 + 3.0 * v1 @ parts[1][2] @ v1
          + _cube(parts[0][3], V1, V1, V1))
    w4 = (parts[4][0] + 4.0 * parts[3][1] @ v1 + 6.0 * v1 @ parts[2][2] @ v1 + 6.0 * parts[2][1] @ v2
          + 4.0 * _cube(parts[1][3], V1, V1, V1) + 12.0 * v2 @ parts[1][2] @ v1
          + np.einsum("iabcd,ia,ib,ic,id->", parts[0][4], V1, V1, V1, V1)
          + 6.0 * _cube(parts[0][3], V2, V1, V1) + 3.0 * v2 @ parts[0][2] @ v2)
    return np.array([parts[0][0], w1, w2, w3, w4])

import numpy as np
def compute_prefactor_time_derivatives(surface: "np.ndarray", beta: float, tau: float,
                                               n_beads: int) -> "np.ndarray":
    """Reference implementation: Jacobi's formula for the determinant and bilinear flux derivatives."""
    import numpy as np

    s = np.asarray(surface, dtype=float)
    X, V1, V2 = locate_flux_instanton(s, beta, tau, n_beads)
    n, half, m = X.shape[0], X.shape[0] // 2, s[2]
    k = 1 + 2 * n + 4 * n * n
    hess, third, fourth = [], [], []
    for order in range(3):
        flat = assemble_action_derivatives(X, s, beta, tau, order)
        hess.append(flat[1 + 2 * n:k].reshape(2 * n, 2 * n))
        third.append(flat[k:k + 8 * n].reshape(n, 2, 2, 2))
        fourth.append(flat[k + 8 * n:].reshape(n, 2, 2, 2, 2))

    def _bead_blocks(tensors):
        out = np.zeros((2 * n, 2 * n))
        for i in range(n):
            out[2 * i:2 * i + 2, 2 * i:2 * i + 2] = tensors[i]
        return out

    free = np.ones(2 * n, bool)
    free[[0, 2 * half]] = False
    sel = np.ix_(free, free)
    inverse = np.linalg.inv(hess[0][sel])
    # Total tau derivatives of the Hessian along the stationary family.
    dJ = hess[1] + _bead_blocks(np.einsum("iabc,ic->iab", third[0], V1))
    ddJ = (hess[2] + 2.0 * _bead_blocks(np.einsum("iabc,ic->iab", third[1], V1))
           + _bead_blocks(np.einsum("iabcd,ic,id->iab", fourth[0], V1, V1))
           + _bead_blocks(np.einsum("iabc,ic->iab", third[0], V2)))
    K = inverse @ dJ[sel]
    log1 = n / tau - n / (beta - tau) + np.trace(K)
    log2 = -n / tau**2 - n / (beta - tau) ** 2 + np.trace(inverse @ ddJ[sel]) - np.sum(K * K.T)
    d_a, d_b = 2.0 * tau / n, 2.0 * (beta - tau) / n
    # Each end momentum is (scale) x (bead difference); scale_a ~ 1/tau and scale_b ~ 1/(beta - tau).
    ends = ((0, 1, m / d_a, -1.0 / tau), (half - 1, half, m / d_a, -1.0 / tau),
            (half, half + 1, m / d_b, 1.0 / (beta - tau)), (n - 1, 0, m / d_b, 1.0 / (beta - tau)))
    p, q1, q2 = np.zeros(4), np.zeros(4), np.zeros(4)
    for j, (i0, i1, scale, rate) in enumerate(ends):
        gap, gap1, gap2 = X[i1, 0] - X[i0, 0], V1[i1, 0] - V1[i0, 0], V2[i1, 0] - V2[i0, 0]
        p[j] = scale * gap
        q1[j] = scale * (rate * gap + gap1)
        q2[j] = scale * (2.0 * rate**2 * gap + 2.0 * rate * gap1 + gap2)
    pair = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=float)
    phi = 0.5 * p @ pair @ p
    r1 = (p @ pair @ q1) / phi
    r2 = (q1 @ pair @ q1 + p @ pair @ q2) / phi
    return np.array([-0.5 * log1 + r1, -0.5 * log2 + r2 - r1**2])

def compute_reactant_correction(surface: "np.ndarray", beta: float, n_beads: int) -> float:
    """Reference implementation: Green's-function contractions on the collapsed reactant path."""
    import math
    import numpy as np

    if isinstance(n_beads, bool) or not isinstance(n_beads, (int, np.integer)) or n_beads < 4 or n_beads % 2:
        raise ValueError("n_beads must be an even integer of at least 4")
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError("beta must be positive and finite")
    s = np.asarray(surface, dtype=float)
    if s.shape != (7,) or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("surface must hold seven finite positive numbers")
    n = int(n_beads)
    X = np.zeros((n, 2))
    # Fifty ranges or widths out, the Eckart term is below 1e-40 of its height and the Gaussian bump has vanished.
    X[:, 0] = -50.0 * max(s[1], s[6])
    flat = assemble_action_derivatives(X, s, beta, 0.5 * beta, 0)
    k = 1 + 2 * n + 4 * n * n
    hess = flat[1 + 2 * n:k].reshape(2 * n, 2 * n)
    third = flat[k:k + 8 * n].reshape(n, 2, 2, 2)
    fourth = flat[k + 8 * n:].reshape(n, 2, 2, 2, 2)
    free = np.ones(2 * n, bool)
    free[0] = False
    G = np.zeros_like(hess)
    G[np.ix_(free, free)] = np.linalg.inv(hess[np.ix_(free, free)])
    blocks = G.reshape(n, 2, n, 2).transpose(0, 2, 1, 3)
    local = blocks[np.arange(n), np.arange(n)]
    quartic = -0.125 * np.einsum("iabcd,iab,icd->", fourth, local, local)
    tadpole = np.einsum("iabc,ibc->ia", third, local).ravel()
    cubic = 0.125 * tadpole @ G @ tadpole
    cubic += sum(np.einsum("abc,jad,jbe,jcf,jdef->", third[i], blocks[i], blocks[i], blocks[i], third)
                 for i in range(n)) / 12.0
    return float(quartic + cubic)

def compute_corrected_rate(barrier: "np.ndarray", stretch_lines: "np.ndarray", width: float, mass: float,
                                   beta: float, n_beads: int) -> float:
    """Reference implementation: leading-order rate, three first-order pieces, cumulant resummation."""
    import math
    import numpy as np

    top = np.asarray(barrier, dtype=float)
    lines = np.asarray(stretch_lines, dtype=float)
    if top.shape != (2,) or lines.shape != (2, 2) or not np.all(np.isfinite(top)) or not np.all(np.isfinite(lines)):
        raise ValueError("barrier must have shape (2,) and stretch_lines shape (2, 2), all finite")
    if np.any(top <= 0.0) or np.any(lines <= 0.0) or not (width > 0.0 and mass > 0.0):
        raise ValueError("all spectroscopic inputs, the width and the mass must be positive")
    harmonic = 3.0 * lines[:, 0] - lines[:, 1]
    if abs(harmonic[0] - harmonic[1]) > 1e-9 * abs(harmonic).max():
        raise ValueError("the two sites must share one harmonic frequency")
    omega = float(harmonic.mean())
    chi_inf, chi_0 = (2.0 * lines[:, 0] - lines[:, 1]) / (2.0 * omega)
    if not (omega > 0.0 and chi_inf > 0.0 and chi_0 > 0.0):
        raise ValueError("each site must give a positive harmonic frequency and anharmonicity constant")
    v0, omega_b = top
    surface = np.array([v0, math.sqrt(2.0 * v0 / mass) / omega_b, mass, omega, chi_inf, chi_0, width])
    tau = 0.5 * beta
    X = locate_flux_instanton(surface, beta, tau, n_beads)[0]
    n, half = X.shape[0], X.shape[0] // 2
    spatial = compute_spatial_correction(surface, beta, tau, n)
    w0, _, w2, w3, w4 = compute_action_time_derivatives(surface, beta, tau, n)
    a1, l2 = compute_prefactor_time_derivatives(surface, beta, tau, n)
    reactant = compute_reactant_correction(surface, beta, n)
    # Real-time derivatives are imaginary-time ones up to powers of i; the signs are absorbed below.
    a2 = l2 + a1**2
    temporal = -w4 / (8.0 * w2**2) + 5.0 * w3**2 / (24.0 * w2**3) - a1 * w3 / (2.0 * w2**2) + a2 / (2.0 * w2)
    k = 1 + 2 * n + 4 * n * n
    hess = assemble_action_derivatives(X, surface, beta, tau, 0)[1 + 2 * n:k].reshape(2 * n, 2 * n)
    free = np.ones(2 * n, bool)
    free[[0, 2 * half]] = False
    far = np.zeros((n, 2))
    far[:, 0] = -50.0 * max(surface[1], surface[6])
    flat_r = assemble_action_derivatives(far, surface, beta, tau, 0)
    hess_r = flat_r[1 + 2 * n:k].reshape(2 * n, 2 * n)
    free_r = np.ones(2 * n, bool)
    free_r[0] = False
    logdet = np.linalg.slogdet(hess[np.ix_(free, free)])[1]
    logdet_r = np.linalg.slogdet(hess_r[np.ix_(free_r, free_r)])[1]
    d = beta / n
    gaps = np.array([X[1, 0] - X[0, 0], X[half, 0] - X[half - 1, 0],
                     X[half + 1, 0] - X[half, 0], X[0, 0] - X[n - 1, 0]]) * mass / d
    phi = gaps[0] * gaps[2] + gaps[0] * gaps[1] + gaps[2] * gaps[3] + gaps[3] * gaps[1]
    # The (m / 2 pi)^N and time-step factors of c_ff and Z_r cancel at tau = beta / 2.
    log_k0 = (math.log(abs(phi) / (4.0 * mass**2)) + 0.5 * (logdet_r - logdet) - 0.5 * math.log(-w2)
              - (w0 - flat_r[0]))
    return float(1e12 * math.exp(log_k0 + spatial + temporal - reactant))
SCICODE_GOLD_EOF
