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


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]

    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")

    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )

    if g <= 0.0:
        raise ValueError("g must be strictly positive")

    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta

    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def bo_hamiltonian(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    s1 = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    s3 = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])

    s2_imag = np.array([
        [0.0, -1.0],
        [1.0, 0.0],
    ])

    real = (
        0.5 * K * Q * Q * np.eye(2)
        + g * Q * (
            s3 * np.cos(theta)
            - s1 * np.sin(theta)
        )
    )

    imag = Delta * s2_imag

    return np.stack([real, imag])

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]

    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")

    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )

    if g <= 0.0:
        raise ValueError("g must be strictly positive")

    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta

    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def adiabatic_states(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    stacked = bo_hamiltonian(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    H = stacked[0] + 1j * stacked[1]
    _, vecs = np.linalg.eigh(H)

    out = np.empty((2, 2), dtype=complex)

    for k in range(2):
        v = vecs[:, k]
        out[:, k] = v * np.exp(
            -1j * np.angle(v[1])
        )

    return np.stack([out.real, out.imag])

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]

    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")

    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )

    if g <= 0.0:
        raise ValueError("g must be strictly positive")

    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta

    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def berry_connections(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    stacked = adiabatic_states(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    lower_second = stacked[0][1, 0]
    upper_second = stacked[0][1, 1]
    da = _grad_alpha(Q, theta, g, Delta)

    return np.stack([
        -(1.0 - lower_second**2) * da,
        -(1.0 - upper_second**2) * da,
    ])

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]

    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")

    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )

    if g <= 0.0:
        raise ValueError("g must be strictly positive")

    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta

    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def bo_geometric_phase(
    Q: float,
    g: float,
    K: float,
    Delta: float,
    n_theta: int,
) -> float:
    _check_point(Q, 0.0, g, K, Delta)

    if (
        np.ndim(n_theta) != 0
        or int(n_theta) != n_theta
        or int(n_theta) < 8
    ):
        raise ValueError("n_theta must be an integer of at least 8")

    n = int(n_theta)
    grid = 2.0 * np.pi * np.arange(n) / n
    total = 0.0

    for th in grid:
        total += berry_connections(
            Q,
            th,
            g,
            K,
            Delta,
        )[0, 1]

    return float(total * Q * 2.0 * np.pi / n)

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]

    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")

    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )

    if g <= 0.0:
        raise ValueError("g must be strictly positive")

    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta

    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def interstate_coupling(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    stacked = adiabatic_states(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    V = stacked[0] + 1j * stacked[1]
    lower, upper = V[:, 0], V[:, 1]

    hs = bo_hamiltonian(
        Q,
        theta,
        g,
        K,
        Delta,
    )
    energies = np.linalg.eigvalsh(
        hs[0] + 1j * hs[1]
    )

    s1 = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    s3 = np.array([
        [1.0, 0.0],
        [0.0, -1.0],
    ])

    dH_dQ = (
        K * Q * np.eye(2)
        + g * (
            s3 * np.cos(theta)
            - s1 * np.sin(theta)
        )
    )

    dH_dtheta_over_Q = g * (
        -s3 * np.sin(theta)
        - s1 * np.cos(theta)
    )

    denom = energies[0] - energies[1]

    b = np.array([
        np.vdot(upper, dH_dQ @ lower) / denom,
        np.vdot(upper, dH_dtheta_over_Q @ lower) / denom,
    ])

    return np.stack([b.real, b.imag])

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]

    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")

    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )

    if g <= 0.0:
        raise ValueError("g must be strictly positive")

    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta

    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def coupling_divergence(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    c = np.cos(theta)
    s = np.sin(theta)

    W2 = g * g * Q * Q + Delta * Delta
    W = np.sqrt(W2)

    S2 = g * g * Q * Q * s * s + Delta * Delta
    S = np.sqrt(S2)

    radial = (
        (Delta * Delta * g * c / 2.0)
        * (
            W2 * S2
            - g * g * Q * Q
            * (2.0 * S2 + W2 * s * s)
        )
        / (W2 * W2 * S2 * S)
        + 1j
        * (Delta * g * s / 2.0)
        * (
            W2 * S2
            - g * g * Q * Q
            * (S2 + W2 * s * s)
        )
        / (W2 * W * S2 * S)
    )

    angular = (
        -(g / 2.0)
        * (c * Delta * Delta / (S2 * S))
        - 1j
        * (Delta * g / 2.0)
        * (
            s
            * (
                S2
                + g * g * Q * Q * c * c
            )
            / (W * S2 * S)
        )
    )

    value = (radial + angular) / Q

    return np.array([
        value.real,
        value.imag,
    ])

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]

    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")

    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )

    if g <= 0.0:
        raise ValueError("g must be strictly positive")

    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta

    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def diagonal_correction(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
    M0: float,
) -> float:
    Q, theta, g, K, Delta = _check_point(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    if (
        np.ndim(M0) != 0
        or not np.isfinite(float(M0))
        or float(M0) <= 0.0
    ):
        raise ValueError(
            "M0 must be a finite positive scalar"
        )

    M0 = float(M0)

    bs = interstate_coupling(
        Q,
        theta,
        g,
        K,
        Delta,
    )

    b = bs[0] + 1j * bs[1]

    return float(
        (abs(b[0]) ** 2 + abs(b[1]) ** 2)
        / (2.0 * M0)
    )

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]
    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")
    Q, theta, g, K, Delta = (float(v) for v in vals)
    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")
    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )
    if g <= 0.0:
        raise ValueError("g must be strictly positive")
    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )
    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta
    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def _check_nuclear(m, M0, mu):
    if np.ndim(m) != 0 or np.ndim(M0) != 0 or np.ndim(mu) != 0:
        raise ValueError("m, M0 and mu must be scalars")
    if int(m) != m:
        raise ValueError("m must be an integer angular momentum quantum number")

    M0, mu = float(M0), float(mu)

    if not np.isfinite(M0) or M0 <= 0.0:
        raise ValueError("M0 must be a finite positive scalar")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a finite positive scalar")

    return int(m), M0, mu


def _check_grid(Q_min, Q_max, n_grid):
    if (
        np.ndim(Q_min) != 0
        or np.ndim(Q_max) != 0
        or np.ndim(n_grid) != 0
    ):
        raise ValueError("Q_min, Q_max and n_grid must be scalars")

    Q_min, Q_max = float(Q_min), float(Q_max)

    if (
        not (np.isfinite(Q_min) and np.isfinite(Q_max))
        or Q_min <= 0.0
        or Q_max <= Q_min
    ):
        raise ValueError("require 0 < Q_min < Q_max, both finite")

    if int(n_grid) != n_grid or int(n_grid) < 16:
        raise ValueError("n_grid must be an integer of at least 16")

    return Q_min, Q_max, int(n_grid)


def _ground_state(diag, off):
    """Lowest eigenpair of a real symmetric tridiagonal matrix."""
    try:
        from scipy.linalg import eigh_tridiagonal
    except Exception:
        eigh_tridiagonal = None

    if eigh_tridiagonal is not None:
        val, vec = eigh_tridiagonal(
            diag,
            off,
            select="i",
            select_range=(0, 0),
        )
        return float(val[0]), vec[:, 0]

    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    val, vec = np.linalg.eigh(H)
    return float(val[0]), vec[:, 0]


def nuclear_radial_state(
    g: float,
    K: float,
    Delta: float,
    m: int,
    M0: float,
    mu: float,
    Q_min: float,
    Q_max: float,
    n_grid: int,
) -> "np.ndarray":
    _check_point(1.0, 0.0, g, K, Delta)
    m, M0, mu = _check_nuclear(m, M0, mu)
    Q_min, Q_max, n_grid = _check_grid(Q_min, Q_max, n_grid)

    Q = np.linspace(Q_min, Q_max, n_grid)
    h = Q[1] - Q[0]
    kinetic = mu / (2.0 * M0)

    W = np.sqrt(g * g * Q * Q + Delta * Delta)
    phase = np.pi * (1.0 - Delta / W)
    A_theta = phase / (2.0 * np.pi * Q)
    surface = 0.5 * K * Q * Q - W

    corr = np.array([
        diagonal_correction(q, 0.0, g, K, Delta, M0)
        for q in Q
    ])

    V = (
        kinetic
        * (
            (m / Q + A_theta) ** 2
            - 1.0 / (4.0 * Q * Q)
        )
        + surface
        + mu * corr
    )

    diag = 2.0 * kinetic / (h * h) + V
    off = -kinetic / (h * h) * np.ones(n_grid - 1)

    _, u = _ground_state(diag, off)

    if u[int(np.argmax(np.abs(u)))] < 0.0:
        u = -u

    R = u / np.sqrt(Q)
    R = R / np.abs(R).max()

    return np.stack([Q, R])

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]
    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")
    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )
    if g <= 0.0:
        raise ValueError("g must be strictly positive")
    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta
    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def _check_nuclear(m, M0, mu):
    if np.ndim(m) != 0 or np.ndim(M0) != 0 or np.ndim(mu) != 0:
        raise ValueError("m, M0 and mu must be scalars")
    if int(m) != m:
        raise ValueError("m must be an integer angular momentum quantum number")

    M0, mu = float(M0), float(mu)

    if not np.isfinite(M0) or M0 <= 0.0:
        raise ValueError("M0 must be a finite positive scalar")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a finite positive scalar")

    return int(m), M0, mu


def _check_grid(Q_min, Q_max, n_grid):
    if (
        np.ndim(Q_min) != 0
        or np.ndim(Q_max) != 0
        or np.ndim(n_grid) != 0
    ):
        raise ValueError("Q_min, Q_max and n_grid must be scalars")

    Q_min, Q_max = float(Q_min), float(Q_max)

    if (
        not (np.isfinite(Q_min) and np.isfinite(Q_max))
        or Q_min <= 0.0
        or Q_max <= Q_min
    ):
        raise ValueError("require 0 < Q_min < Q_max, both finite")

    if int(n_grid) != n_grid or int(n_grid) < 16:
        raise ValueError("n_grid must be an integer of at least 16")

    return Q_min, Q_max, int(n_grid)


def nuclear_momentum(
    Q: float,
    g: float,
    K: float,
    Delta: float,
    m: int,
    M0: float,
    mu: float,
    Q_min: float,
    Q_max: float,
    n_grid: int,
) -> "np.ndarray":
    Q, _, g, K, Delta = _check_point(Q, 0.0, g, K, Delta)
    m, M0, mu = _check_nuclear(m, M0, mu)
    Q_min, Q_max, n_grid = _check_grid(Q_min, Q_max, n_grid)

    state = nuclear_radial_state(
        g,
        K,
        Delta,
        m,
        M0,
        mu,
        Q_min,
        Q_max,
        n_grid,
    )
    grid, R = state[0], state[1]
    h = grid[1] - grid[0]
    k = int(round((Q - Q_min) / h))

    if (
        k < 1
        or k > n_grid - 2
        or abs(grid[k] - Q) > 1e-9 * max(1.0, abs(Q))
    ):
        raise ValueError("Q must be an interior point of the radial grid")

    log_derivative = (
        (R[k + 1] - R[k - 1])
        / (2.0 * h)
        / R[k]
    )

    W = np.sqrt(g * g * Q * Q + Delta * Delta)
    A_theta = (
        np.pi * (1.0 - Delta / W)
        / (2.0 * np.pi * Q)
    )

    p = np.array(
        [-1j * log_derivative, m / Q + A_theta],
        dtype=complex,
    )

    return np.stack([p.real, p.imag])

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]
    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")
    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )
    if g <= 0.0:
        raise ValueError("g must be strictly positive")
    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta
    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def _check_momentum(momentum):
    p = np.asarray(momentum, dtype=float)

    if p.shape != (2, 2) or not np.all(np.isfinite(p)):
        raise ValueError(
            "momentum must be a finite real array of shape (2, 2)"
        )

    return p[0] + 1j * p[1]


def correlation_coupling(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
    M0: float,
    momentum: "np.ndarray",
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(Q, theta, g, K, Delta)

    if (
        np.ndim(M0) != 0
        or not np.isfinite(float(M0))
        or float(M0) <= 0.0
    ):
        raise ValueError("M0 must be a finite positive scalar")

    M0 = float(M0)
    p = _check_momentum(momentum)

    bs = interstate_coupling(Q, theta, g, K, Delta)
    b = bs[0] + 1j * bs[1]

    conn = berry_connections(Q, theta, g, K, Delta)
    dA = conn[1] - conn[0]

    dv = coupling_divergence(Q, theta, g, K, Delta)
    div = dv[0] + 1j * dv[1]

    value = (
        (-div - 1j * np.dot(b, dA)) / (2.0 * M0)
        - 1j * np.dot(p, b) / M0
    )

    return np.array([value.real, value.imag])

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]
    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")
    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )
    if g <= 0.0:
        raise ValueError("g must be strictly positive")
    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta
    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def _check_momentum(momentum):
    p = np.asarray(momentum, dtype=float)

    if p.shape != (2, 2) or not np.all(np.isfinite(p)):
        raise ValueError(
            "momentum must be a finite real array of shape (2, 2)"
        )

    return p[0] + 1j * p[1]


def connection_correction(
    Q: float,
    theta: float,
    g: float,
    K: float,
    Delta: float,
    M0: float,
    mu: float,
    momentum: "np.ndarray",
) -> "np.ndarray":
    Q, theta, g, K, Delta = _check_point(Q, theta, g, K, Delta)

    if (
        np.ndim(mu) != 0
        or not np.isfinite(float(mu))
        or float(mu) <= 0.0
    ):
        raise ValueError("mu must be a finite positive scalar")

    mu = float(mu)

    el = correlation_coupling(
        Q,
        theta,
        g,
        K,
        Delta,
        M0,
        momentum,
    )
    element = el[0] + 1j * el[1]

    hs = bo_hamiltonian(Q, theta, g, K, Delta)
    energies = np.linalg.eigvalsh(hs[0] + 1j * hs[1])

    kappa = -mu * element / (energies[1] - energies[0])

    bs = interstate_coupling(Q, theta, g, K, Delta)
    b = bs[0] + 1j * bs[1]

    return 2.0 * np.real(np.conj(kappa) * (-1j * b))

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]
    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")
    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )
    if g <= 0.0:
        raise ValueError("g must be strictly positive")
    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta
    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def _check_nuclear(m, M0, mu):
    if np.ndim(m) != 0 or np.ndim(M0) != 0 or np.ndim(mu) != 0:
        raise ValueError("m, M0 and mu must be scalars")
    if int(m) != m:
        raise ValueError("m must be an integer angular momentum quantum number")

    M0, mu = float(M0), float(mu)

    if not np.isfinite(M0) or M0 <= 0.0:
        raise ValueError("M0 must be a finite positive scalar")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError("mu must be a finite positive scalar")

    return int(m), M0, mu


def _check_grid(Q_min, Q_max, n_grid):
    if (
        np.ndim(Q_min) != 0
        or np.ndim(Q_max) != 0
        or np.ndim(n_grid) != 0
    ):
        raise ValueError("Q_min, Q_max and n_grid must be scalars")

    Q_min, Q_max = float(Q_min), float(Q_max)

    if (
        not (np.isfinite(Q_min) and np.isfinite(Q_max))
        or Q_min <= 0.0
        or Q_max <= Q_min
    ):
        raise ValueError("require 0 < Q_min < Q_max, both finite")

    if int(n_grid) != n_grid or int(n_grid) < 16:
        raise ValueError("n_grid must be an integer of at least 16")

    return Q_min, Q_max, int(n_grid)


def nonadiabatic_phase_ratio(
    Q: float = 0.35,
    g: float = 0.6,
    K: float = 1.0,
    Delta: float = 0.12,
    m: int = 3,
    M0: float = 100.0,
    mu: float = 0.01,
    Q_min: float = 0.02,
    Q_max: float = 2.02,
    n_grid: int = 4001,
    n_theta: int = 2048,
) -> float:
    Q, _, g, K, Delta = _check_point(Q, 0.0, g, K, Delta)
    m, M0, mu = _check_nuclear(m, M0, mu)
    Q_min, Q_max, n_grid = _check_grid(
        Q_min,
        Q_max,
        n_grid,
    )

    if (
        np.ndim(n_theta) != 0
        or int(n_theta) != n_theta
        or int(n_theta) < 8
    ):
        raise ValueError("n_theta must be an integer of at least 8")

    n = int(n_theta)

    momentum = nuclear_momentum(
        Q,
        g,
        K,
        Delta,
        m,
        M0,
        mu,
        Q_min,
        Q_max,
        n_grid,
    )

    zeroth = bo_geometric_phase(
        Q,
        g,
        K,
        Delta,
        n,
    )

    grid = 2.0 * np.pi * np.arange(n) / n
    total = 0.0

    for th in grid:
        total += connection_correction(
            Q,
            th,
            g,
            K,
            Delta,
            M0,
            mu,
            momentum,
        )[1]

    first = total * Q * 2.0 * np.pi / n
    return float(100.0 * first / zeroth)
SCICODE_GOLD_EOF
