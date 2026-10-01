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

MU = 1.0
B0 = 0.5
AL = 0.4
CC = 0.2
E0 = 7.5e-6
AA = 0.6
RHO_SCALE = 1.0e-14


def field_samples(points, which):
    """Reference implementation of field_samples (deterministic)."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")
    if which not in ("B", "E", "rho"):
        raise ValueError("which must be one of 'B', 'E', 'rho'")

    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    if which == "B":
        return np.stack([(y - 2.0) ** 2 - MU + z ** 2 + AL * x,
                         -x - AL * y + CC * z,
                         np.full_like(x, B0)], axis=-1)
    if which == "E":
        return E0 * np.stack([np.zeros_like(x), 2.0 * z, 3.0 - 2.0 * y],
                             axis=-1)
    return 1.0 + AA * ((y - 2.0) ** 2 + z ** 2)

import numpy as np

AL = 0.4
CC = 0.2


def gradient_tensor(points):
    """Reference implementation of gradient_tensor (deterministic)."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")

    y = p[:, 1]
    z = p[:, 2]
    n = p.shape[0]
    G = np.zeros((n, 3, 3), dtype=np.float64)
    G[:, 0, 0] = AL
    G[:, 0, 1] = 2.0 * (y - 2.0)
    G[:, 0, 2] = 2.0 * z
    G[:, 1, 0] = -1.0
    G[:, 1, 1] = -AL
    G[:, 1, 2] = CC
    return G

import numpy as np


def alignment_residual(points):
    """Reference implementation of alignment_residual (deterministic).

    Composes the step-01 and step-02 oracle functions directly, so this
    reference value never depends on a submitted implementation.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")

    B = np.asarray(field_samples(p, "B"), dtype=np.float64)
    G = np.asarray(gradient_tensor(p), dtype=np.float64)

    GB = np.einsum("nij,nj->ni", G, B)
    residual = np.cross(B, GB)

    out = np.empty((p.shape[0], 4), dtype=np.float64)
    out[:, 0:3] = residual
    for n in range(p.shape[0]):
        ev = np.real(np.linalg.eigvals(G[n]))
        ev = np.sort(ev)[::-1]
        out[n, 3] = -(ev[0] * ev[-1])
    return out

import numpy as np

AL = 0.4
B0 = 0.5
CC = 0.2
MU = 1.0


def candidate_line(z_values):
    """Reference implementation of candidate_line (deterministic)."""
    z = np.asarray(z_values, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("z_values must be one-dimensional with length >= 1")
    if not np.all(np.isfinite(z)):
        raise ValueError("z_values must be finite")

    disc = AL ** 4 + 8.0 * AL ** 2 + 4.0 * MU + 4.0 * CC * B0 \
        - 4.0 * z ** 2 - 4.0 * AL * CC * z
    if np.any(disc <= 0.0):
        raise ValueError(
            "every z must satisfy 4*z**2 + 4*AL*CC*z < "
            "AL**4 + 8*AL**2 + 4*MU + 4*CC*B0")

    d = np.sqrt(disc)
    y = 2.0 + 0.5 * AL ** 2 - 0.5 * d
    by = B0 * (2.0 * z + AL * CC) / d
    x = -by - AL * y + CC * z
    return np.stack([x, y, z], axis=-1)

import numpy as np

AL = 0.4
CC = 0.2


def seed_point(z_lo, z_hi):
    """Reference implementation of seed_point (deterministic).

    Composes the step-03 and step-04 oracle functions directly, so this
    reference value never depends on a submitted implementation.
    """
    try:
        lo = float(z_lo)
        hi = float(z_hi)
    except (TypeError, ValueError):
        raise ValueError("z_lo and z_hi must be real numbers")
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("z_lo and z_hi must be finite")
    if not (lo < hi):
        raise ValueError("z_lo must be strictly less than z_hi")

    # The discriminant along the step-04 curve is the square root of a downward
    # parabola in the height, so it is strictly concave and its maximiser over
    # a closed interval is either the stationary point or one of the two ends.
    # All three are evaluated with the step-03 discriminant itself and the
    # largest is selected, so the selection rests on the measured quantity
    # rather than on the algebra used to shortlist the candidates. Passing all
    # three heights to step 04 at once also enforces existence over the whole
    # closed interval, not merely at the maximiser.
    z_star = -AL * CC / 2.0
    heights = np.array([min(max(z_star, lo), hi), lo, hi], dtype=np.float64)
    curve = candidate_line(heights)
    phi = np.asarray(alignment_residual(curve), dtype=np.float64)[:, 3]
    return np.asarray(curve[int(np.argmax(phi))], dtype=np.float64)

import math

import numpy as np

_RK_STEP = 2.0 ** -12


def quasi_x_line(z_values, seed):
    """Reference implementation of quasi_x_line (deterministic).

    Composes the step-01 oracle function directly, so this reference value
    never depends on a submitted implementation.
    """
    z = np.asarray(z_values, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("z_values must be one-dimensional with length >= 1")
    if not np.all(np.isfinite(z)):
        raise ValueError("z_values must be finite")

    s = np.asarray(seed, dtype=np.float64)
    if s.ndim != 1 or s.size != 3:
        raise ValueError("seed must have shape (3,)")
    if not np.all(np.isfinite(s)):
        raise ValueError("seed must be finite")

    z0 = float(s[2])

    def _slope(zz, u):
        pt = np.array([[u[0], u[1], zz]], dtype=np.float64)
        b = np.asarray(field_samples(pt, "B"), dtype=np.float64)[0]
        return np.array([b[0] / b[2], b[1] / b[2]], dtype=np.float64)

    out = np.empty((z.size, 3), dtype=np.float64)
    order = np.argsort(z)
    upper = [int(i) for i in order if z[i] > z0]
    lower = [int(i) for i in order[::-1] if z[i] < z0]

    for run in (upper, lower):
        u = np.array([s[0], s[1]], dtype=np.float64)
        cur = z0
        for idx in run:
            target = float(z[idx])
            nsub = max(1, int(math.ceil(abs(target - cur) / _RK_STEP)))
            h = (target - cur) / nsub
            for _ in range(nsub):
                k1 = _slope(cur, u)
                k2 = _slope(cur + 0.5 * h, u + 0.5 * h * k1)
                k3 = _slope(cur + 0.5 * h, u + 0.5 * h * k2)
                k4 = _slope(cur + h, u + h * k3)
                u = u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                cur = cur + h
            out[idx] = [u[0], u[1], target]

    for i in range(z.size):
        if z[i] == z0:
            out[i] = [s[0], s[1], z0]
    return out

import numpy as np


def parallel_electric_field(points):
    """Reference implementation of parallel_electric_field (deterministic).

    Composes the step-01 oracle function directly, so this reference value
    never depends on a submitted implementation.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 1:
        raise ValueError("points must have shape (N, 3) with N >= 1")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")

    B = np.asarray(field_samples(p, "B"), dtype=np.float64)
    E = np.asarray(field_samples(p, "E"), dtype=np.float64)
    mag = np.linalg.norm(B, axis=-1)
    if np.any(mag <= 0.0):
        raise ValueError("magnetic field magnitude must be positive")
    return np.sum(E * B, axis=-1) / mag

import numpy as np


def line_average(points, values):
    """Reference implementation of line_average (deterministic)."""
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("points must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")
    if v.ndim == 1:
        if v.shape[0] != p.shape[0]:
            raise ValueError("values must have length N")
    elif v.ndim == 2:
        if v.shape != (p.shape[0], 3):
            raise ValueError("vector values must have shape (N, 3)")
    else:
        raise ValueError("values must have shape (N,) or (N, 3)")
    if not np.all(np.isfinite(v)):
        raise ValueError("values must be finite")

    ds = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    total = float(np.sum(ds))
    if total == 0.0:
        raise ValueError("total sampled arclength is zero")

    w = ds if v.ndim == 1 else ds[:, None]
    integral = np.sum(0.5 * (v[1:] + v[:-1]) * w, axis=0)
    average = integral / total
    if v.ndim == 1:
        return float(average), total
    return np.asarray(average, dtype=np.float64), total

import math

import numpy as np

RHO_REF = 1.0e-14


def inflow_quantities(points, delta):
    """Reference implementation of inflow_quantities (deterministic).

    Composes the step-01 and step-08 oracle functions directly, so this
    reference value never depends on a submitted implementation.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < 2:
        raise ValueError("points must have shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(p)):
        raise ValueError("points must be finite")
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        raise ValueError("delta must be a real number")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be a positive finite real")

    shift = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    b_means = []
    r_means = []
    for sign in (1.0, -1.0):
        q = p + sign * delta * shift
        b_avg, _ = line_average(
            p, np.asarray(field_samples(q, "B"), dtype=np.float64))
        r_avg, _ = line_average(
            p, np.asarray(field_samples(q, "rho"), dtype=np.float64))
        b_means.append(np.asarray(b_avg, dtype=np.float64))
        r_means.append(float(r_avg))

    b_vec = 0.5 * (b_means[0] + b_means[1])
    rho_s = 0.5 * (r_means[0] + r_means[1])
    if not (rho_s > 0.0):
        raise ValueError("inflow density must be positive")
    b_in = float(np.linalg.norm(b_vec))
    v_a = b_in / math.sqrt(4.0 * math.pi * rho_s * RHO_REF)
    return b_in, rho_s, v_a

import numpy as np

C_LIGHT = 2.99792458e10


def reconnection_rate(z_lo=-0.8, z_hi=0.8, n_points=81, delta=0.05):
    """Reference implementation of reconnection_rate (deterministic).

    Composes the step-05, step-06, step-07, step-08 and step-09 oracle
    functions directly, so this reference value never depends on a submitted
    implementation.
    """
    try:
        lo = float(z_lo)
        hi = float(z_hi)
    except (TypeError, ValueError):
        raise ValueError("z_lo and z_hi must be real numbers")
    if not (np.isfinite(lo) and np.isfinite(hi)) or not (lo < hi):
        raise ValueError("require finite z_lo < z_hi")
    if isinstance(n_points, bool) or not isinstance(n_points, (int, np.integer)):
        raise ValueError("n_points must be an integer")
    if int(n_points) < 2:
        raise ValueError("n_points must be at least 2")
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        raise ValueError("delta must be a real number")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be a positive finite real")

    seed = seed_point(lo, hi)
    zs = np.linspace(lo, hi, int(n_points))
    pts = quasi_x_line(zs, seed)

    epar = parallel_electric_field(pts)
    r0, total_length = line_average(pts, epar)
    b_in, rho_s, v_a = inflow_quantities(pts, delta)

    # Certify the delivered configuration before reporting. The candidate curve
    # must actually solve the alignment condition; the curve that is sampled
    # must carry a nowhere-vanishing field and a finite gradient tensor, and it
    # must stay on the positive-discriminant branch along its whole length,
    # which is the filter the extraction ends with.
    candidate = candidate_line(zs)
    residual = np.asarray(alignment_residual(candidate), dtype=np.float64)
    if not np.all(np.abs(residual[:, 0:3]) < 1.0e-9):
        raise ValueError("candidate curve does not solve the alignment condition")

    field_on_line = np.asarray(field_samples(pts, "B"), dtype=np.float64)
    tensor_on_line = np.asarray(gradient_tensor(pts), dtype=np.float64)
    if np.any(np.linalg.norm(field_on_line, axis=-1) <= 0.0):
        raise ValueError("magnetic field vanishes on the sampled curve")
    if not np.all(np.isfinite(tensor_on_line)):
        raise ValueError("non-finite gradient tensor on the sampled curve")

    diagnostics = np.asarray(alignment_residual(pts), dtype=np.float64)
    if not np.all(diagnostics[:, 3] > 0.0):
        raise ValueError("sampled curve leaves the positive-discriminant branch")

    if not (total_length > 0.0 and b_in > 0.0 and rho_s > 0.0 and v_a > 0.0):
        raise ValueError("degenerate curve or inflow normalisation")
    if not np.isfinite(r0):
        raise ValueError("non-finite arclength-averaged parallel electric field")

    return float(r0) / (b_in * v_a / C_LIGHT)
SCICODE_GOLD_EOF
