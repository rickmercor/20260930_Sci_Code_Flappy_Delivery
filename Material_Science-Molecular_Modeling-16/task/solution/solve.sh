#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def compute_moments(h, hb, beta, order):
    import numpy as np

    h = np.asarray(h, dtype=complex)
    hb = np.asarray(hb, dtype=complex)

    if (
        hb.ndim != 2
        or hb.shape[0] != hb.shape[1]
        or hb.shape[0] == 0
        or h.shape != (2 * hb.shape[0], 2 * hb.shape[0])
        or not np.isfinite(h).all()
        or not np.isfinite(hb).all()
        or not np.allclose(h, h.conj().T, atol=1e-12, rtol=0)
        or not np.allclose(hb, hb.conj().T, atol=1e-12, rtol=0)
        or not np.isfinite(beta)
        or beta < 0
        or not isinstance(order, (int, np.integer))
        or not 0 <= order <= 12
    ):
        raise ValueError(
            "Invalid Hamiltonian, temperature, or moment order."
        )

    energy, vectors = np.linalg.eigh(hb)
    weights = np.exp(-beta * (energy - energy.min()))
    weights = weights / weights.sum()
    rho = (vectors * weights) @ vectors.conj().T

    pauli = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, -1j], [1j, 0]],
            [[1, 0], [0, -1]],
        ],
        dtype=complex,
    ) / np.sqrt(2.0)

    bath_identity = np.eye(hb.shape[0])

    operators = np.array(
        [np.kron(a, bath_identity) for a in pauli]
    )
    dual = np.array(
        [np.kron(a, rho) for a in pauli]
    )

    result = np.empty((order + 1, 4, 4), dtype=float)

    for n in range(order + 1):
        result[n] = np.einsum(
            "iab,jba->ij", operators, dual
        ).real

        operators = 1j * (
            h @ operators - operators @ h
        )

    return result

def kernel_coefficients(moments):
    import math
    import numpy as np

    w = np.asarray(moments, dtype=float)

    if (
        w.shape != (13, 4, 4)
        or not np.isfinite(w).all()
        or not np.allclose(w[0], np.eye(4), atol=1e-10, rtol=0)
    ):
        raise ValueError(
            "Require thirteen finite moments with identity zeroth moment."
        )

    w = w.copy()

    previous = np.array([
        w[n + 1] - w[1] @ w[n]
        for n in range(1, 12)
    ])

    coefficients = [previous[2].copy()]

    for derivative_order in range(1, 9):
        current = np.array([
            previous[n] - previous[0] @ w[n]
            for n in range(1, len(previous))
        ])

        coefficients.append(
            current[2] / math.factorial(derivative_order)
        )

        previous = current

    return np.array(coefficients)

def fit_closure(coefficients):
    import numpy as np

    c = np.asarray(coefficients, dtype=float)

    if c.shape != (9, 4, 4) or not np.isfinite(c).all():
        raise ValueError(
            "Require nine finite 4 by 4 Taylor coefficients."
        )

    result = np.zeros((2, 5, 4, 4))
    result[1, 0] = 1.0

    for i in range(4):
        for j in range(4):
            a = c[:, i, j]

            matrix = np.array([
                [a[k - m] for m in range(1, 5)]
                for k in range(5, 9)
            ])

            rhs = -a[5:9]

            q = np.linalg.lstsq(
                matrix, rhs, rcond=1e-12
            )[0]

            scale = max(1.0, np.max(np.abs(a)))

            if np.max(np.abs(matrix @ q - rhs)) > 1e-10 * scale:
                raise ValueError(
                    "Inconsistent normalized Pade system."
                )

            result[1, 1:, i, j] = q

            result[0, :, i, j] = [
                a[k] + sum(
                    q[m - 1] * a[k - m]
                    for m in range(1, k + 1)
                )
                for k in range(5)
            ]

    return result

def propagate_kernel(moments, closure, duration, steps):
    import numpy as np

    w = np.asarray(moments, dtype=float)
    pq = np.asarray(closure, dtype=float)

    if (
        w.shape != (13, 4, 4)
        or pq.shape != (2, 5, 4, 4)
        or not np.isfinite(w).all()
        or not np.isfinite(pq).all()
        or not np.isfinite(duration)
        or duration <= 0
        or not isinstance(steps, (int, np.integer))
        or not 1 <= steps <= 4096
    ):
        raise ValueError("Invalid kernel propagation inputs.")

    h = duration / steps
    times = np.arange(2 * steps + 1) * h / 2

    numerator = np.polynomial.polynomial.polyval(
        times, pq[0]
    ).transpose(2, 0, 1)

    denominator = np.polynomial.polynomial.polyval(
        times, pq[1]
    ).transpose(2, 0, 1)

    if np.min(np.abs(denominator)) <= 1e-10:
        raise ValueError(
            "Closure denominator vanishes at a stage time."
        )

    values = numerator / denominator

    x = w[2] - w[1] @ w[1]
    y = w[3] - w[1] @ w[2]

    result = np.empty((steps + 1, 4, 4))
    result[0] = x

    for n in range(steps):
        ax = y - x @ w[1]
        ay = values[2 * n] - x @ w[2]

        bx = y + h * ay / 2 - (x + h * ax / 2) @ w[1]
        by = (
            values[2 * n + 1]
            - (x + h * ax / 2) @ w[2]
        )

        cx = y + h * by / 2 - (x + h * bx / 2) @ w[1]
        cy = (
            values[2 * n + 1]
            - (x + h * bx / 2) @ w[2]
        )

        dx = y + h * cy - (x + h * cx) @ w[1]
        dy = (
            values[2 * n + 2]
            - (x + h * cx) @ w[2]
        )

        x = x + h * (ax + 2 * bx + 2 * cx + dx) / 6
        y = y + h * (ay + 2 * by + 2 * cy + dy) / 6

        result[n + 1] = x

    return result

def propagate_correlation(kernel, drift, dt):
    import numpy as np

    k = np.asarray(kernel, dtype=float)
    a = np.asarray(drift, dtype=float)

    if (
        k.ndim != 3
        or len(k) < 2
        or k.shape[1] == 0
        or k.shape[1] != k.shape[2]
        or not np.isfinite(k).all()
        or a.shape != k.shape[1:]
        or not np.isfinite(a).all()
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Invalid kernel grid or time step.")

    identity = np.eye(k.shape[1])
    lhs = identity - dt * a / 2 - dt * dt * k[0] / 4

    if np.linalg.cond(lhs) > 1e12:
        raise ValueError(
            "Implicit endpoint matrix is ill-conditioned."
        )

    correlation = np.empty_like(k)
    correlation[0] = identity
    derivative = a.copy()

    for n in range(1, len(k)):
        interior = np.zeros_like(identity)

        for j in range(1, n):
            interior += k[n - j] @ correlation[j]

        known = dt * (k[n] / 2 + interior)

        correlation[n] = np.linalg.solve(
            lhs,
            correlation[n - 1]
            + dt * (derivative + known) / 2,
        )

        derivative = (
            a @ correlation[n]
            + known
            + dt * k[0] @ correlation[n] / 2
        )

    return correlation

def measure_population(correlation, initial, axis):
    import numpy as np

    c = np.asarray(correlation, dtype=float)
    r = np.asarray(initial, dtype=float)
    a = np.asarray(axis, dtype=float)

    if (
        c.shape != (4, 4)
        or r.shape != (3,)
        or a.shape != (3,)
        or not all(
            np.isfinite(value).all()
            for value in (c, r, a)
        )
        or np.linalg.norm(r) > 1 + 1e-12
        or abs(np.linalg.norm(a) - 1) > 1e-12
    ):
        raise ValueError(
            "Invalid correlation matrix or Bloch vectors."
        )

    return float(
        np.r_[1.0, a] @ c @ np.r_[1.0, r] / 2
    )

def solve(h, hb, beta, initial, axis, duration, steps):
    moments = compute_moments(
        h, hb, beta, 12
    )

    coefficients = kernel_coefficients(
        moments
    )

    closure = fit_closure(
        coefficients
    )

    kernel = propagate_kernel(
        moments, closure, duration, steps
    )

    correlation = propagate_correlation(
        kernel, moments[1], duration / steps
    )

    return measure_population(
        correlation[-1], initial, axis
    )
SCICODE_GOLD_EOF
