#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def measurement_pencil_jets(energies: 'np.ndarray', weights: 'np.ndarray', dt: float, dimension: int, batches: int, noise_h: float, noise_s: float) -> 'np.ndarray':
    import numpy as np
    from numbers import Integral, Real
    try:
        e_raw = np.asarray(energies)
        if np.iscomplexobj(e_raw) and np.any(e_raw.imag != 0):
            raise ValueError('energies must be real-valued; nonzero imaginary parts are invalid.')
        e = np.asarray(e_raw.real if np.iscomplexobj(e_raw) else e_raw, dtype=float)
        w_raw = np.asarray(weights)
        if np.iscomplexobj(w_raw) and np.any(w_raw.imag != 0):
            raise ValueError('weights must be real-valued; nonzero imaginary parts are invalid.')
        w = np.asarray(w_raw.real if np.iscomplexobj(w_raw) else w_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Real level energies and weights are required.') from exc
    if e.ndim != 1 or e.size < 1 or w.shape != e.shape or not np.all(np.isfinite(e)) or not np.all(np.isfinite(w)) or np.any(w < 0) or not 0 < w.sum() < np.inf:
        raise ValueError('Energies and nonnegative weights must be finite matching vectors with positive total weight.')
    for v in (dt, noise_h, noise_s):
        if isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) or v < 0:
            raise ValueError('The time spacing and noise scales must be finite and nonnegative.')
    for v, low, high in ((dimension, 2, 32), (batches, 1, 64)):
        if isinstance(v, bool) or not isinstance(v, Integral) or not low <= v <= high:
            raise ValueError('Dimension or batch count is outside its supported integer range.')
    w = w / w.sum()
    v = np.sqrt(w[:, None]) * np.exp(-1j * np.outer(e, np.arange(dimension) * dt))
    h = v.conj().T @ (e[:, None] * v)
    s = v.conj().T @ v
    result = np.zeros((3, 2, batches, dimension, dimension), dtype=complex)
    for q in range(batches):
        a = np.zeros_like(h)
        b = np.zeros_like(h)
        c = np.zeros_like(h)
        for i in range(dimension):
            a[i, i] = np.cos((q + 1) * (i + 1))
            b[i, i] = np.sin((q + 2) * (i + 1))
            for j in range(i + 1, dimension):
                z = 1 + j - i
                a[i, j] = (np.sin((q + 1) * (i + j + 2)) + 1j * np.cos((q + 2) * (j - i))) / z
                b[i, j] = (np.cos((q + 2) * (i + j + 2)) + 1j * np.sin((q + 1) * (j - i))) / z
                c[i, j] = (np.cos((q + 1) * (i + j + 2)) + 1j * np.sin((q + 2) * (j - i))) / z
                a[j, i] = a[i, j].conjugate()
                b[j, i] = b[i, j].conjugate()
                c[j, i] = c[i, j].conjugate()
        result[0, 0, q] = h + noise_h * a
        result[1, 0, q] = noise_h * b
        result[2, 0, q] = -noise_h * a
        result[0, 1, q] = s + noise_s * c
    return result

def nearest_physical_overlap(overlap: 'np.ndarray', tolerance: float = 1e-13, max_iterations: int = 2000) -> 'np.ndarray':
    import numpy as np
    from numbers import Integral, Real
    try:
        a = np.array(overlap, dtype=complex, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError('A numerical overlap matrix is required.') from exc
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1 or not np.all(np.isfinite(a)) or not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
        raise ValueError('The overlap must be finite, square and Hermitian.')
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or not 0 < tolerance <= 1e-6:
        raise ValueError('Tolerance must lie in (0, 1e-6].')
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError('The iteration limit must be a positive integer.')
    y = (a + a.conj().T) / 2
    x = y.copy()
    correction = np.zeros_like(y)
    for _ in range(max_iterations):
        old_x, old_y = x, y
        r = y - correction
        values, vectors = np.linalg.eigh((r + r.conj().T) / 2)
        x = (vectors * np.maximum(values, 0)) @ vectors.conj().T
        correction = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        scale = max(1.0, np.linalg.norm(y, 'fro'))
        residual = max(np.linalg.norm(x - old_x, 'fro'), np.linalg.norm(y - old_y, 'fro'), np.linalg.norm(y - x, 'fro')) / scale
        if residual <= tolerance:
            return (y + y.conj().T) / 2
    raise ValueError('Physical-overlap projection did not converge within the iteration limit.')

def rotated_ground_energy(hamiltonian: 'np.ndarray', overlap: 'np.ndarray', theta: float, cutoff: float) -> float:
    import numpy as np
    from numbers import Real
    from scipy.linalg import eigh
    try:
        h = np.asarray(hamiltonian, dtype=complex)
        s = np.asarray(overlap, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError('Numerical matrices are required.') from exc
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 1 or s.shape != h.shape:
        raise ValueError('The pencil matrices must be nonempty and equally sized square matrices.')
    for a in (h, s):
        if not np.all(np.isfinite(a)) or not np.allclose(a, a.conj().T, atol=1e-10, rtol=0):
            raise ValueError('The pencil matrices must be finite and Hermitian.')
    for a in (theta, cutoff):
        if isinstance(a, bool) or not isinstance(a, Real) or not np.isfinite(a):
            raise ValueError('Angle and cutoff must be finite real scalars.')
    if cutoff < 0:
        raise ValueError('The cutoff must be nonnegative.')
    c, t = np.cos(theta), np.sin(theta)
    a = c * h - t * s
    b = c * s + t * h
    beta, u = eigh(b)
    u = u[:, beta > cutoff]
    if u.shape[1] == 0:
        raise ValueError('The retained overlap subspace is empty.')
    ac, bc = u.conj().T @ a @ u, u.conj().T @ b @ u
    lam = eigh((ac + ac.conj().T) / 2, (bc + bc.conj().T) / 2, eigvals_only=True)
    denominator = c - t * lam
    good = np.abs(denominator) > 1e-12
    if not np.any(good):
        raise ValueError('Every retained root lies at the excluded inverse-rotation pole.')
    physical = (c * lam[good] + t) / denominator[good]
    if not np.all(np.isfinite(physical)):
        raise ValueError('The recovered energies are nonfinite.')
    return float(np.min(physical))

def batch_energy_moments(energies: 'np.ndarray') -> 'np.ndarray':
    import numpy as np
    try:
        e_raw = np.asarray(energies)
        if np.iscomplexobj(e_raw) and np.any(e_raw.imag != 0):
            raise ValueError('energies must be real-valued; nonzero imaginary parts are invalid.')
        e = np.asarray(e_raw.real if np.iscomplexobj(e_raw) else e_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('A real energy vector is required.') from exc
    if e.ndim != 1 or e.size == 0 or not np.all(np.isfinite(e)):
        raise ValueError('The batch energies must form a nonempty finite vector.')
    mean = float(np.mean(e))
    centered = e - mean
    variance = float(np.mean(centered * centered))
    if not np.isfinite(variance):
        raise ValueError('The population variance must be representable.')
    return np.array([mean, np.sqrt(variance), variance], dtype=float)

def dimension_convergence(previous: 'np.ndarray', current: 'np.ndarray', gamma: float, energy_tolerance: float) -> 'np.ndarray':
    import numpy as np
    from numbers import Real
    try:
        p_raw = np.asarray(previous)
        if np.iscomplexobj(p_raw) and np.any(p_raw.imag != 0):
            raise ValueError('previous must be real-valued; nonzero imaginary parts are invalid.')
        p = np.asarray(p_raw.real if np.iscomplexobj(p_raw) else p_raw, dtype=float)
        q_raw = np.asarray(current)
        if np.iscomplexobj(q_raw) and np.any(q_raw.imag != 0):
            raise ValueError('current must be real-valued; nonzero imaginary parts are invalid.')
        q = np.asarray(q_raw.real if np.iscomplexobj(q_raw) else q_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Numerical moment triples are required.') from exc
    if p.shape != (3,) or q.shape != (3,) or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)) or np.any(p[1:] < 0) or np.any(q[1:] < 0):
        raise ValueError('Each moment triple must contain a finite mean and nonnegative dispersion entries.')
    for a in (gamma, energy_tolerance):
        if isinstance(a, bool) or not isinstance(a, Real) or not np.isfinite(a) or a < 0:
            raise ValueError('The convergence parameters must be finite and nonnegative.')
    difference = abs(float(q[0] - p[0]))
    threshold = max(float(energy_tolerance), float(gamma) * max(float(p[1]), float(q[1])))
    if not np.isfinite(threshold):
        raise ValueError('The convergence threshold must be representable.')
    return np.array([difference, threshold, float(difference < threshold)])

def select_variance_angle(angles: 'np.ndarray', batch_energies: 'np.ndarray') -> 'np.ndarray':
    import numpy as np
    try:
        a_raw = np.asarray(angles)
        if np.iscomplexobj(a_raw) and np.any(a_raw.imag != 0):
            raise ValueError('angles must be real-valued; nonzero imaginary parts are invalid.')
        a = np.asarray(a_raw.real if np.iscomplexobj(a_raw) else a_raw, dtype=float)
        e_raw = np.asarray(batch_energies)
        if np.iscomplexobj(e_raw) and np.any(e_raw.imag != 0):
            raise ValueError('batch_energies must be real-valued; nonzero imaginary parts are invalid.')
        e = np.asarray(e_raw.real if np.iscomplexobj(e_raw) else e_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Real angles and energies are required.') from exc
    if a.ndim != 1 or a.size == 0 or e.ndim != 2 or e.shape[0] != a.size or e.shape[1] == 0 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(e)):
        raise ValueError('Angles and batch-energy rows must be finite, nonempty and matched.')
    centered = e - np.mean(e, axis=1)[:, None]
    variances = np.mean(centered * centered, axis=1)
    if not np.all(np.isfinite(variances)):
        raise ValueError('Each population variance must be representable.')
    index = int(np.argmin(variances))
    return np.array([float(index), a[index], variances[index]], dtype=float)

def cutoff_projector_jet(denominator_jet: 'np.ndarray', cutoff: float, gap_tolerance: float = 1e-10) -> 'np.ndarray':
    import numpy as np
    from numbers import Real
    try:
        b = np.asarray(denominator_jet, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError('A numerical Hermitian matrix jet is required.') from exc
    if b.ndim != 3 or b.shape[0] != 3 or b.shape[1] != b.shape[2] or b.shape[1] < 1 or not np.all(np.isfinite(b)) or not np.allclose(b, b.conj().transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError('The denominator jet must have shape (3, n, n), with finite Hermitian slices.')
    for x in (cutoff, gap_tolerance):
        if isinstance(x, bool) or not isinstance(x, Real) or not np.isfinite(x) or x < 0:
            raise ValueError('Cutoff and gap tolerance must be finite and nonnegative.')
    beta, u = np.linalg.eigh(b[0])
    if np.min(np.abs(beta - cutoff)) <= gap_tolerance:
        raise ValueError('A denominator eigenvalue is too close to the cutoff for this derivative contract.')
    retained = np.flatnonzero(beta > cutoff)
    discarded = np.flatnonzero(beta < cutoff)
    e = u.conj().T @ b[1] @ u
    f = u.conj().T @ b[2] @ u
    p0 = np.diag((beta > cutoff).astype(float)).astype(complex)
    p1 = np.zeros_like(p0)
    for i in discarded:
        for j in retained:
            p1[i, j] = e[i, j] / (beta[j] - beta[i])
            p1[j, i] = p1[i, j].conjugate()
    square = p1 @ p1
    p2 = np.zeros_like(p0)
    p2[np.ix_(retained, retained)] = -2 * square[np.ix_(retained, retained)]
    p2[np.ix_(discarded, discarded)] = 2 * square[np.ix_(discarded, discarded)]
    rhs = f + 2 * (e @ p1 - p1 @ e)
    for i in discarded:
        for j in retained:
            p2[i, j] = rhs[i, j] / (beta[j] - beta[i])
            p2[j, i] = p2[i, j].conjugate()
    return np.stack([u @ p @ u.conj().T for p in (p0, p1, p2)])

def projected_energy_jet(numerator_jet: 'np.ndarray', denominator_jet: 'np.ndarray', projector_jet: 'np.ndarray', theta: float, gap_tolerance: float = 1e-10) -> 'np.ndarray':
    import numpy as np
    from numbers import Real
    from scipy.linalg import eigh
    try:
        a = np.asarray(numerator_jet, dtype=complex)
        b = np.asarray(denominator_jet, dtype=complex)
        p = np.asarray(projector_jet, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError('Numerical matrix jets are required.') from exc
    if a.ndim != 3 or a.shape[0] != 3 or a.shape[1] != a.shape[2] or a.shape[1] < 1 or b.shape != a.shape or p.shape != a.shape:
        raise ValueError('All three jets must have the same shape (3, n, n).')
    for x in (a, b, p):
        if not np.all(np.isfinite(x)) or not np.allclose(x, x.conj().transpose(0, 2, 1), atol=1e-10, rtol=0):
            raise ValueError('Jet slices must be finite and Hermitian.')
    if isinstance(theta, bool) or not isinstance(theta, Real) or not np.isfinite(theta):
        raise ValueError('The angle must be a finite real scalar.')
    if isinstance(gap_tolerance, bool) or not isinstance(gap_tolerance, Real) or not np.isfinite(gap_tolerance) or gap_tolerance < 0:
        raise ValueError('The gap tolerance must be finite and nonnegative.')
    identities = (p[0] @ p[0] - p[0], p[0] @ p[1] + p[1] @ p[0] - p[1], p[0] @ p[2] + p[2] @ p[0] + 2 * p[1] @ p[1] - p[2])
    if any(np.linalg.norm(x, 'fro') > 1e-7 for x in identities):
        raise ValueError('The supplied projector jet violates differentiated idempotency.')
    values, vectors = np.linalg.eigh(p[0])
    u0 = vectors[:, values > 0.5]
    if u0.shape[1] == 0:
        raise ValueError('The retained subspace is empty.')
    k = p[1] @ p[0] - p[0] @ p[1]
    dk = p[2] @ p[0] - p[0] @ p[2]
    u1 = k @ u0
    u2 = (dk + k @ k) @ u0
    compressed = []
    for m in (a, b):
        m0 = u0.conj().T @ m[0] @ u0
        m1 = u1.conj().T @ m[0] @ u0 + u0.conj().T @ m[1] @ u0 + u0.conj().T @ m[0] @ u1
        m2 = u2.conj().T @ m[0] @ u0 + u0.conj().T @ m[0] @ u2 + 2 * u1.conj().T @ m[0] @ u1 + 2 * u1.conj().T @ m[1] @ u0 + 2 * u0.conj().T @ m[1] @ u1 + u0.conj().T @ m[2] @ u0
        compressed.append([(x + x.conj().T) / 2 for x in (m0, m1, m2)])
    aa, bb = compressed
    if np.min(np.linalg.eigvalsh(bb[0])) <= 0:
        raise ValueError('The compressed denominator must be positive definite.')
    lam, z = eigh(aa[0], bb[0])
    c, s = np.cos(theta), np.sin(theta)
    den = c - s * lam
    good = np.flatnonzero(np.abs(den) > 1e-12)
    if good.size == 0:
        raise ValueError('Every retained root is at the excluded inverse-rotation pole.')
    physical = (c * lam[good] + s) / den[good]
    g = int(good[np.argmin(physical)])
    other = [j for j in range(len(lam)) if j != g]
    if other and np.min(np.abs(lam[g] - lam[other])) <= gap_tolerance:
        raise ValueError('The selected generalized eigenvalue is not sufficiently isolated.')
    target = z[:, g]
    residual = aa[1] - lam[g] * bb[1]
    first = float(np.real(target.conj() @ residual @ target))
    second = float(np.real(target.conj() @ (aa[2] - lam[g] * bb[2]) @ target))
    second -= 2 * first * float(np.real(target.conj() @ bb[1] @ target))
    for j in other:
        coupling = z[:, j].conj() @ residual @ target
        second += 2 * abs(coupling) ** 2 / (lam[g] - lam[j])
    result = np.array([(c * lam[g] + s) / den[g], first / den[g] ** 2, second / den[g] ** 2 + 2 * s * first ** 2 / den[g] ** 3], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError('The physical energy jet is nonfinite.')
    return result

def rotated_krylov_curvature(energies: 'np.ndarray', weights: 'np.ndarray', dt: float = 0.6, dimension: int = 6, batches: int = 4, noise_h: float = 0.035, noise_s: float = 0.065, cutoff: float = 0.08, angles: 'np.ndarray | tuple[float, ...]' = (0.0, 0.2, 0.45, 0.7, 0.95, 1.2, 1.4), gamma: float = 1.3, energy_tolerance: float = 1e-4) -> float:
    import numpy as np
    from numbers import Integral, Real
    if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 3:
        raise ValueError('The complete growth experiment requires at least three basis states.')
    for x in (gamma, energy_tolerance):
        if isinstance(x, bool) or not isinstance(x, Real) or not np.isfinite(x) or x < 0:
            raise ValueError('Convergence parameters must be finite and nonnegative.')
    try:
        theta_grid_raw = np.asarray(angles)
        if np.iscomplexobj(theta_grid_raw) and np.any(theta_grid_raw.imag != 0):
            raise ValueError('angles must be real-valued; nonzero imaginary parts are invalid.')
        theta_grid = np.asarray(theta_grid_raw.real if np.iscomplexobj(theta_grid_raw) else theta_grid_raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('A real angle grid is required.') from exc
    if theta_grid.ndim != 1 or theta_grid.size == 0 or not np.all(np.isfinite(theta_grid)):
        raise ValueError('The angle grid must be a nonempty finite vector.')
    data = measurement_pencil_jets(energies, weights, dt, dimension, batches, noise_h, noise_s)
    previous = None
    for size in range(2, dimension + 1):
        repaired = np.stack([nearest_physical_overlap(data[0, 1, q, :size, :size]) for q in range(batches)])
        sample = np.array([rotated_ground_energy(data[0, 0, q, :size, :size], repaired[q], 0.0, cutoff) for q in range(batches)])
        current = batch_energy_moments(sample)
        if previous is not None:
            certificate = dimension_convergence(previous, current, gamma, energy_tolerance)
            if certificate[2] == 1:
                break
        previous = current
    table = np.array([[rotated_ground_energy(data[0, 0, q, :size, :size], repaired[q], theta, cutoff) for q in range(batches)] for theta in theta_grid])
    choice = select_variance_angle(theta_grid, table)
    theta = float(choice[1])
    h = data[:, 0, :, :size, :size].mean(axis=1)
    s = np.zeros_like(h)
    s[0] = nearest_physical_overlap(data[0, 1, :, :size, :size].mean(axis=0))
    c, t = np.cos(theta), np.sin(theta)
    numerator = c * h - t * s
    denominator = c * s + t * h
    projector = cutoff_projector_jet(denominator, cutoff)
    result = projected_energy_jet(numerator, denominator, projector, theta)
    return float(result[2])
SCICODE_GOLD_EOF
