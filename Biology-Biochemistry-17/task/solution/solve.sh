#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def metropolis_segment(start: "np.ndarray", beta: float, rng: "np.random.Generator",
                               max_steps: int) -> "np.ndarray":
    """Metropolis walk on the lattice from start until A or B is entered or max_steps steps are made."""
    import numpy as np
    i, j = int(start[0]), int(start[1])
    frames = [(i, j)]
    u_old = ((-1.45 + 0.1 * i) ** 2 - 1.0) ** 2 + (-0.95 + 0.1 * j) ** 2
    steps = 0
    while 5 < i < 24 and steps < max_steps:
        d = int(rng.integers(4))
        u = rng.random()
        ni, nj = i + (1, -1, 0, 0)[d], j + (0, 0, 1, -1)[d]
        if 0 <= ni < 30 and 0 <= nj < 20:
            u_new = ((-1.45 + 0.1 * ni) ** 2 - 1.0) ** 2 + (-0.95 + 0.1 * nj) ** 2
            if u < np.exp(-beta * (u_new - u_old)):
                i, j, u_old = ni, nj, u_new
        frames.append((i, j))
        steps += 1
    return np.array(frames, dtype=int)

def tis_shooting_trial(path: "np.ndarray", set_index: int, interface: float, beta: float,
                               rng: "np.random.Generator", max_steps: int) -> "np.ndarray":
    """One two-way shooting trial in the ensemble of paths leaving A that cross the given interface."""
    import numpy as np
    path = np.asarray(path, dtype=int)
    n_old = len(path)
    s = 1 + int(rng.integers(n_old - 2))
    back = metropolis_segment(path[s], beta, rng, max_steps)
    if back[-1, 0] > 5:
        return path
    fwd = metropolis_segment(path[s], beta, rng, max_steps)
    if 5 < fwd[-1, 0] < 24:
        return path
    new = np.concatenate([back[::-1], fwd[1:]])
    x = -1.45 + 0.1 * new[:, 0]
    y = -0.95 + 0.1 * new[:, 1]
    theta = np.deg2rad(5.0)
    cvs = (x, x * np.cos(theta) + y * np.sin(theta), x + 0.1 * np.sin(2.0 * np.pi * y))
    if cvs[set_index - 1].max() <= interface:
        return path
    if rng.random() < min(1.0, (n_old - 2) / (len(new) - 2)):
        return new
    return path

def sample_interface_ensemble(set_index: int, k: int, n_equil: int, n_samples: int, beta: float,
                                      seed: int, max_steps: int) -> "np.ndarray":
    """Maxima of the three order parameters over the stored paths of TIS ensemble (set_index, k)."""
    import numpy as np
    interfaces = ([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2],
                  [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2],
                  [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2])
    lam = interfaces[set_index - 1][k - 1]
    rng = np.random.default_rng((seed, set_index, k))
    theta = np.deg2rad(5.0)
    y0 = -0.95 + 0.1 * 9
    top = 5
    while True:
        x0 = -1.45 + 0.1 * top
        value = (x0, x0 * np.cos(theta) + y0 * np.sin(theta), x0 + 0.1 * np.sin(2.0 * np.pi * y0))[set_index - 1]
        if value > lam:
            break
        top += 1
    cols = list(range(5, top + 1)) + list(range(top - 1, 4, -1))
    path = np.array([(c, 9) for c in cols], dtype=int)
    out = np.empty((n_samples, 3))
    for t in range(n_equil + n_samples):
        path = tis_shooting_trial(path, set_index, lam, beta, rng, max_steps)
        if t >= n_equil:
            x = -1.45 + 0.1 * path[:, 0]
            y = -0.95 + 0.1 * path[:, 1]
            out[t - n_equil] = (x.max(), (x * np.cos(theta) + y * np.sin(theta)).max(),
                                (x + 0.1 * np.sin(2.0 * np.pi * y)).max())
    return out

def joint_log_partition_sums(maxima: "np.ndarray", interfaces: list, counts: list) -> "np.ndarray":
    """Log conditional partition sums of every interface ensemble of every set, from all pooled paths."""
    import numpy as np
    maxima = np.asarray(maxima, dtype=float)
    m = len(interfaces)
    levels = np.stack([np.searchsorted(np.asarray(interfaces[i], dtype=float), maxima[:, i], side="left")
                       for i in range(m)], axis=1)
    if np.any(levels.max(axis=1) == 0):
        raise ValueError("a pooled path crosses no interface of any set")
    cells, mult = np.unique(levels, axis=0, return_counts=True)
    n = [np.asarray(c, dtype=float) for c in counts]
    z = [np.ones(len(c)) for c in n]
    for _ in range(100000):
        denom = np.zeros(len(cells))
        for i in range(m):
            denom += np.concatenate([[0.0], np.cumsum(n[i] / z[i])])[cells[:, i]]
        w = mult / denom
        w /= w.sum()
        new = [np.array([w[cells[:, i] >= kk].sum() for kk in range(1, len(n[i]) + 1)]) for i in range(m)]
        change = max(np.max(np.abs(a / b - 1.0)) for a, b in zip(new, z))
        z = new
        if change < 1e-13:
            return np.log(np.concatenate(z))
    raise ValueError("the self-consistent equations did not converge")

def crossing_probability(maxima: "np.ndarray", interfaces: list, counts: list, log_z: "np.ndarray",
                                 lam_ref: float, lams: "np.ndarray") -> "np.ndarray":
    """Log of the reweighted probability that a path crossing lam_ref along lambda1 = x also crosses each lam."""
    import numpy as np
    maxima = np.asarray(maxima, dtype=float)
    log_z = np.asarray(log_z, dtype=float)
    denom = np.zeros(len(maxima))
    start = 0
    for i in range(len(interfaces)):
        size = len(interfaces[i])
        levels = np.searchsorted(np.asarray(interfaces[i], dtype=float), maxima[:, i], side="left")
        cum = np.concatenate([[0.0], np.cumsum(np.asarray(counts[i], dtype=float) * np.exp(-log_z[start:start + size]))])
        denom += cum[levels]
        start += size
    w = 1.0 / denom
    w /= w.sum()
    ref = maxima[:, 0] > lam_ref
    total = w[ref].sum()
    return np.log(np.array([w[ref & (maxima[:, 0] > lam)].sum() / total for lam in np.atleast_1d(lams)]))

def rescaled_crossing_probability(maxima_by_set: list, interfaces: list, counts: list,
                                          lam_ref: float) -> float:
    """Log crossing probability into B from independently reweighted sets rescaled to unit reactive weight."""
    import numpy as np
    weights, pooled = [], []
    for i in range(len(interfaces)):
        mx = np.asarray(maxima_by_set[i], dtype=float)
        log_z = joint_log_partition_sums(mx[:, i:i + 1], [interfaces[i]], [counts[i]])
        levels = np.searchsorted(np.asarray(interfaces[i], dtype=float), mx[:, i], side="left")
        cum = np.concatenate([[0.0], np.cumsum(np.asarray(counts[i], dtype=float) * np.exp(-log_z))])
        w = 1.0 / cum[levels]
        reactive = mx[:, 0] > 0.9
        if not reactive.any():
            raise ValueError(f"set {i + 1} has no path that reaches B")
        weights.append(w / w[reactive].mean())
        pooled.append(mx)
    w = np.concatenate(weights)
    mx = np.concatenate(pooled)
    ref = mx[:, 0] > lam_ref
    return float(np.log(w[ref & (mx[:, 0] > 0.9)].sum() / w[ref].sum()))

def exact_crossing_probability(beta: float, lam_ref: float) -> float:
    """Exact log probability that a path leaving A which crosses x = lam_ref reaches B, for the lattice chain."""
    import numpy as np
    nx, ny = 30, 20
    x = -1.45 + 0.1 * np.arange(nx)
    y = -0.95 + 0.1 * np.arange(ny)
    u = ((x[:, None] ** 2 - 1.0) ** 2 + y[None, :] ** 2).ravel()
    ii = np.repeat(np.arange(nx), ny)
    jj = np.tile(np.arange(ny), nx)
    size = nx * ny
    kmat = np.zeros((size, size))
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ni, nj = ii + di, jj + dj
        ok = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
        src = np.nonzero(ok)[0]
        dst = ni[ok] * ny + nj[ok]
        kmat[src, dst] += 0.25 * np.minimum(1.0, np.exp(-beta * (u[dst] - u[src])))
    kmat[np.arange(size), np.arange(size)] += 1.0 - kmat.sum(axis=1)
    in_a = x[ii] < -0.9
    hits = []
    for target in (x[ii] > 0.9, (x[ii] > lam_ref) & ~in_a):
        free = ~(in_a | target)
        h = target.astype(float)
        h[free] = np.linalg.solve(np.eye(free.sum()) - kmat[np.ix_(free, free)], kmat[np.ix_(free, target)].sum(axis=1))
        hits.append(h)
    flux = np.exp(-beta * u[in_a])[:, None] * kmat[in_a] * (~in_a)[None, :]
    return float(np.log((flux @ hits[0]).sum() / (flux @ hits[1]).sum()))

def compare_crossing_estimates(n_equil: int, n_samples: int, beta: float, seed: int,
                                       max_steps: int) -> "np.ndarray":
    """1e5 x [joint (3 sets), joint (sets 1-2), set 1 alone, rescaled (3 sets), exact]."""
    import numpy as np
    interfaces = [np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2]),
                  np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2]),
                  np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2])]
    by_set = [np.concatenate([sample_interface_ensemble(i + 1, k + 1, n_equil, n_samples, beta, seed, max_steps)
                              for k in range(len(interfaces[i]))]) for i in range(3)]
    counts = [np.full(len(lams), n_samples) for lams in interfaces]
    out = []
    for m in (3, 2, 1):
        mx = np.concatenate(by_set[:m])[:, :m]
        log_z = joint_log_partition_sums(mx, interfaces[:m], counts[:m])
        out.append(np.exp(crossing_probability(mx, interfaces[:m], counts[:m], log_z, -0.8, np.array([0.9]))[0]))
    out.append(np.exp(rescaled_crossing_probability(by_set, interfaces, counts, -0.8)))
    out.append(np.exp(exact_crossing_probability(beta, -0.8)))
    return 1e5 * np.array(out)
SCICODE_GOLD_EOF
