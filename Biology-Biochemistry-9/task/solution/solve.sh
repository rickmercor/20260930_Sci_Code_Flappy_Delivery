#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def perturbed_weights(
    weights: "np.ndarray", factors: "np.ndarray"
) -> "np.ndarray":
    import numpy as np

    g = np.asarray(weights, dtype=float)
    f = np.asarray(factors, dtype=float)
    if (
        g.ndim != 2
        or f.ndim != 2
        or min(g.shape + f.shape) == 0
        or (g.shape[1] != f.shape[1])
        or (not np.isfinite(g).all())
        or (not np.isfinite(f).all())
        or np.any(g <= 0)
        or np.any(f <= 0)
    ):
        raise ValueError("Positive finite aligned matrices required")
    result = g[:, None, :] * f[None, :, :]
    if not np.isfinite(result).all():
        raise ValueError("Product overflow")
    return result

def balance_rows(
    stoichiometry: "np.ndarray", demand: "np.ndarray"
) -> "np.ndarray":
    import numpy as np

    a = np.asarray(stoichiometry, dtype=float)
    b = np.asarray(demand, dtype=float)
    if (
        a.ndim != 2
        or min(a.shape) == 0
        or b.shape != (a.shape[0],)
        or (not np.isfinite(a).all())
        or (not np.isfinite(b).all())
    ):
        raise ValueError("Finite balance matrix and aligned demand required")
    selected = []
    rank = 0
    for i in range(len(a)):
        r = np.linalg.matrix_rank(a[selected + [i]], tol=1e-10)
        if r > rank:
            selected.append(i)
            rank = r
    if np.linalg.matrix_rank(np.c_[a, b], tol=1e-10) > rank:
        raise ValueError("Inconsistent balance equations")
    return np.c_[a[selected], b[selected]].reshape(rank, a.shape[1] + 1)

def infer_net_panel(
    balance: "np.ndarray", weights: "np.ndarray", irreversible: "np.ndarray"
) -> "np.ndarray":
    import numpy as np
    from itertools import combinations
    from scipy.linalg import null_space

    a = np.asarray(balance, dtype=float)
    g = np.asarray(weights, dtype=float)
    ids = np.asarray(irreversible)
    if (
        g.ndim != 3
        or min(g.shape) == 0
        or a.ndim != 2
        or (a.shape[1] != g.shape[-1] + 1)
        or (not np.isfinite(a).all())
        or (not np.isfinite(g).all())
        or np.any(g <= 0)
        or (ids.ndim != 1)
        or (not np.issubdtype(ids.dtype, np.integer))
        or (len(ids) > 6)
        or (len(set(ids.tolist())) != len(ids))
        or np.any(ids < 0)
        or np.any(ids >= g.shape[-1])
    ):
        raise ValueError("Invalid panel, balance, or direction indices")
    n = g.shape[-1]
    A = a[:, :n]
    b = a[:, n]
    if np.linalg.matrix_rank(A, tol=1e-10) != len(A):
        raise ValueError("Independent balance rows required")
    faces = []
    for k in range(len(ids) + 1):
        for active in combinations(ids.tolist(), k):
            C = np.vstack([A, np.eye(n)[list(active)]])
            d = np.r_[b, np.zeros(k)]
            x = np.linalg.lstsq(C, d, rcond=1e-12)[0]
            if np.max(np.abs(C @ x - d), initial=0) > 1e-09:
                continue
            N = null_space(C, rcond=1e-12)
            faces.append((x, N))
    output = np.empty_like(g)
    for index in np.ndindex(g.shape[:-1]):
        scale = 2 * g[index] / np.e
        best = None
        best_f = np.inf

        def _cost(v):
            return float(np.sum(v * np.arcsinh(v / scale) - np.hypot(v, scale)))

        for origin, N in faces:
            v = origin.copy()
            for iteration in range(100):
                grad = N.T @ np.arcsinh(v / scale)
                if grad.size == 0 or np.max(np.abs(grad)) < 2e-12:
                    break
                hess = N.T / np.hypot(v, scale) @ N
                step = -N @ np.linalg.solve(hess, grad)
                decrement = float(grad @ np.linalg.solve(hess, grad))
                alpha = 1.0
                old = _cost(v)
                while alpha > 2 ** (-40):
                    trial = v + alpha * step
                    if _cost(trial) <= old - 0.0001 * alpha * decrement + 2e-13:
                        break
                    alpha *= 0.5
                v = trial
            else:
                raise RuntimeError("Stationary solve did not converge")
            if np.any(v[ids] < -2e-09):
                continue
            obj = _cost(v)
            if obj < best_f:
                best_f = obj
                best = v.copy()
        if best is None:
            raise ValueError("No state satisfies direction evidence")
        best[np.abs(best) < 2e-11] = 0.0
        output[index] = best
    return output

def directional_panel(
    nets: "np.ndarray", weights: "np.ndarray"
) -> "np.ndarray":
    import numpy as np

    v = np.asarray(nets, dtype=float)
    g = np.asarray(weights, dtype=float)
    if (
        v.shape != g.shape
        or v.size == 0
        or (not np.isfinite(v).all())
        or (not np.isfinite(g).all())
        or np.any(g <= 0)
    ):
        raise ValueError("Aligned finite nets and positive weights required")
    a = g / np.e
    large = (np.hypot(v, 2 * a) + np.abs(v)) / 2
    small = a * (a / large)
    f = np.where(v >= 0, large, small)
    r = np.where(v >= 0, small, large)
    result = np.stack([f, r], axis=-1)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("Directional values outside representable range")
    return result

def energy_panel(
    directional: "np.ndarray", gas_constant: float, temperature: float
) -> "np.ndarray":
    import numpy as np

    p = np.asarray(directional, dtype=float)
    R = float(gas_constant)
    T = float(temperature)
    if (
        p.ndim < 1
        or p.shape[-1] != 2
        or p.size == 0
        or (not np.isfinite(p).all())
        or np.any(p <= 0)
        or (not np.isfinite(R))
        or (not np.isfinite(T))
        or (R <= 0)
        or (T <= 0)
    ):
        raise ValueError("Positive paired rates, R and T required")
    return R * T * (np.log(p[..., 1]) - np.log(p[..., 0]))

def active_directions(
    nets: "np.ndarray", activity_threshold: float
) -> "np.ndarray":
    import numpy as np

    v = np.asarray(nets, dtype=float)
    t = float(activity_threshold)
    if v.size == 0 or not np.isfinite(v).all() or (not np.isfinite(t)) or (t <= 0):
        raise ValueError("Finite nets and positive activity threshold required")
    return np.where(np.abs(v) > t, np.sign(v), 0.0).astype(float)

def physiological_ranges(
    stoichiometry: "np.ndarray",
    directions: "np.ndarray",
    standard_mean: "np.ndarray",
    standard_sd: "np.ndarray",
    log_bounds: "np.ndarray",
    contrasts: "np.ndarray",
    contrast_bounds: "np.ndarray",
    gas_constant: float,
    temperature: float,
    driving_floor: float,
) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import linprog

    S = np.asarray(stoichiometry, dtype=float)
    z = np.asarray(directions, dtype=float)
    mu = np.asarray(standard_mean, dtype=float)
    sd = np.asarray(standard_sd, dtype=float)
    lo = np.asarray(log_bounds, dtype=float)
    H = np.asarray(contrasts, dtype=float)
    hb = np.asarray(contrast_bounds, dtype=float)
    R = float(gas_constant)
    T = float(temperature)
    eps = float(driving_floor)
    if S.ndim != 2 or z.ndim != 3 or min(S.shape + z.shape) == 0:
        raise ValueError("Invalid stoichiometry or direction panel")
    m, n = S.shape
    C, P, nz = z.shape
    if (
        nz != n
        or mu.shape != (n,)
        or sd.shape != (n,)
        or (lo.shape != (C, m, 2))
        or (H.ndim != 2)
        or (H.shape[1] != m)
        or (hb.shape != (C, H.shape[0], 2))
        or any((not np.isfinite(x).all() for x in (S, z, mu, sd, lo, H, hb)))
        or np.any(sd < 0)
        or np.any(lo[:, :, 0] > lo[:, :, 1])
        or np.any(hb[:, :, 0] > hb[:, :, 1])
        or (not np.isin(z, [-1, 0, 1]).all())
        or (not np.isfinite([R, T, eps]).all())
        or (min(R, T, eps) <= 0)
    ):
        raise ValueError("Invalid physiological inputs")
    D = np.c_[np.eye(n), R * T * S.T]
    J = np.c_[np.zeros((len(H), n)), H]
    result = np.zeros((C, P, 1 + 2 * n))
    options = {
        "primal_feasibility_tolerance": 1e-09,
        "dual_feasibility_tolerance": 1e-09,
    }
    for c in range(C):
        bounds = list(zip(mu - 2.58 * sd, mu + 2.58 * sd)) + list(map(tuple, lo[c]))
        cache = {}
        for p in range(P):
            key = tuple(z[c, p])
            if key in cache:
                result[c, p] = cache[key]
                continue
            active = np.flatnonzero(z[c, p])
            A = np.vstack([z[c, p, active, None] * D[active], J, -J])
            b = np.r_[np.full(len(active), -eps), hb[c, :, 1], -hb[c, :, 0]]

            def _run(obj):
                return linprog(
                    obj,
                    A_ub=A,
                    b_ub=b,
                    bounds=bounds,
                    method="highs",
                    options=options,
                )

            first = _run(np.zeros(n + m))
            row = np.zeros(1 + 2 * n)
            if first.status == 2:
                cache[key] = row
                continue
            if not first.success:
                raise RuntimeError(first.message)
            row[0] = 1.0
            for j in range(n):
                left = _run(D[j])
                right = _run(-D[j])
                if not left.success or not right.success:
                    raise RuntimeError("Unresolved finite range")
                row[1 + j] = left.fun
                row[1 + n + j] = -right.fun
            result[c, p] = row
            cache[key] = row
    return result

def compatibility_panel(
    energies: "np.ndarray",
    directions: "np.ndarray",
    ranges: "np.ndarray",
    allowed_excess: float,
) -> "np.ndarray":
    import numpy as np

    e = np.asarray(energies, dtype=float)
    z = np.asarray(directions, dtype=float)
    t = np.asarray(ranges, dtype=float)
    a = float(allowed_excess)
    if (
        e.ndim != 3
        or min(e.shape) == 0
        or z.shape != e.shape
        or (t.shape != e.shape[:-1] + (1 + 2 * e.shape[-1],))
        or any((not np.isfinite(x).all() for x in (e, z, t)))
        or (not np.isin(z, [-1, 0, 1]).all())
        or (not np.isin(t[..., 0], [0, 1]).all())
        or (not np.isfinite(a))
        or (a < 0)
    ):
        raise ValueError("Invalid compatibility inputs")
    n = e.shape[-1]
    lower = t[..., 1 : 1 + n]
    upper = t[..., 1 + n :]
    if np.any((lower > upper) & (t[..., 0, None] == 1)):
        raise ValueError("Reversed feasible ranges")
    distance = np.maximum(np.maximum(lower - e, e - upper), 0.0)
    distance = np.where(z != 0, distance, 0.0)
    excess = np.max(distance, axis=-1)
    excess = np.where(t[..., 0] == 1, excess, 0.0)
    passed = (t[..., 0] == 1) & (excess <= a)
    return np.stack([passed.astype(float), excess], axis=-1)

def select_robust_candidate(
    nets: "np.ndarray",
    compatibility: "np.ndarray",
    target: int,
    normalization: float,
) -> "np.ndarray":
    import numpy as np

    v = np.asarray(nets, dtype=float)
    k = np.asarray(compatibility, dtype=float)
    d = float(normalization)
    if (
        v.ndim != 3
        or min(v.shape) == 0
        or k.shape != v.shape[:-1] + (2,)
        or (not np.isfinite(v).all())
        or (not np.isfinite(k).all())
        or (not np.isin(k[..., 0], [0, 1]).all())
        or (not isinstance(target, (int, np.integer)))
        or (not 0 <= target < v.shape[-1])
        or (not np.isfinite(d))
        or (d <= 0)
    ):
        raise ValueError("Invalid selection inputs")
    eligible = np.all(k[..., 0] == 1, axis=1)
    if not eligible.any():
        raise ValueError("No eligible candidate")
    scores = np.min(v[:, :, target], axis=1) / d
    maximum = np.max(scores[eligible])
    winner = np.flatnonzero(eligible & (scores >= maximum - 1e-10))[0]
    row = v[winner, :, target] / d
    scenario = np.flatnonzero(row <= scores[winner] + 1e-10)[0]
    return np.r_[
        float(winner),
        float(scenario),
        scores[winner],
        eligible.astype(float),
        scores,
    ]

def resolve_metabolic_panel(data: dict | None = None) -> float:
    if data is None:
        data = _benchmark_inputs()
    if not isinstance(data, dict):
        raise ValueError("Input must be a data dictionary")
    required = {
        "S",
        "b",
        "weights",
        "factors",
        "irreversible",
        "standard_mean",
        "standard_sd",
        "log_bounds",
        "contrasts",
        "contrast_bounds",
        "R",
        "T",
        "tau",
        "epsilon",
        "excess",
        "target",
        "normalization",
    }
    if set(data) != required:
        raise ValueError("Input keys do not match the contract")
    weights = perturbed_weights(data["weights"], data["factors"])
    balance = balance_rows(data["S"], data["b"])
    nets = infer_net_panel(balance, weights, data["irreversible"])
    pairs = directional_panel(nets, weights)
    energies = energy_panel(pairs, data["R"], data["T"])
    directions = active_directions(nets, data["tau"])
    ranges = physiological_ranges(
        data["S"],
        directions,
        data["standard_mean"],
        data["standard_sd"],
        data["log_bounds"],
        data["contrasts"],
        data["contrast_bounds"],
        data["R"],
        data["T"],
        data["epsilon"],
    )
    compatibility = compatibility_panel(
        energies, directions, ranges, data["excess"]
    )
    selected = select_robust_candidate(
        nets, compatibility, data["target"], data["normalization"]
    )
    return float(selected[2])


def _benchmark_inputs():
    import numpy as np

    edges = [
        (0, 1),
        (0, 2),
        (2, 1),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 6),
        (6, 1),
        (5, 2),
        (0, 6),
    ]
    S = np.zeros((7, 14))
    for j, (left, right) in enumerate(edges):
        S[left, j] = -1.0
        S[right, j] = 1.0
    weights = np.array(
        [
            [
                4.8,
                13.1,
                2.4,
                9.7,
                3.9,
                12.6,
                2.1,
                4.3,
                7.2,
                16.5,
                5.4,
                8.1,
                3.6,
                1.1,
            ],
            [
                8.4,
                10.2,
                6.3,
                5.1,
                9.8,
                11.3,
                7.4,
                3.6,
                4.9,
                14.2,
                6.7,
                4.5,
                9.3,
                1.7,
            ],
            [
                12.2,
                7.6,
                3.1,
                13.4,
                6.8,
                8.9,
                4.2,
                7.1,
                2.7,
                18.4,
                3.9,
                6.8,
                5.1,
                2.4,
            ],
            [
                6.1,
                14.8,
                9.2,
                6.3,
                12.1,
                15.7,
                3.6,
                5.8,
                10.4,
                13.7,
                7.8,
                3.2,
                8.6,
                0.9,
            ],
            [
                10.7,
                9.3,
                4.6,
                11.9,
                5.2,
                10.8,
                8.1,
                6.4,
                3.8,
                15.9,
                4.7,
                7.3,
                2.9,
                1.4,
            ],
            [
                7.9,
                12.4,
                7.1,
                8.6,
                7.7,
                13.9,
                5.5,
                4.9,
                6.3,
                17.1,
                5.9,
                5.6,
                6.2,
                1.9,
            ],
        ]
    )
    factors = np.ones((6, 14))
    changes = [
        [(0, 0.22), (8, 1.8)],
        [(3, 0.28), (13, 1.5)],
        [(5, 0.55), (10, 1.4)],
        [(6, 0.18), (9, 0.75)],
        [(1, 0.35), (3, 1.6)],
    ]
    for row, entries in enumerate(changes, 1):
        for column, value in entries:
            factors[row, column] = value
    H = np.zeros((3, 7))
    H[0, 1] = 1.0
    H[0, 2] = -1.0
    H[1, 6] = 1.0
    H[1, 4] = -1.0
    H[2, [0, 3]] = 1.0
    H[2, [1, 4]] = -1.0
    hb = np.tile([[-6.0, 6.0], [-6.0, 6.0], [-12.0, 12.0]], (6, 1, 1))
    hb[0, 0] = [0.18, 0.42]
    hb[3, 1] = [-0.58, -0.22]
    hb[4, 2] = [2.25, 2.9]
    return dict(
        S=S,
        b=np.array([-10.0, 0.0, 0.0, 0.0, 0.0, 2.0, 8.0]),
        weights=weights,
        factors=factors,
        irreversible=np.array([2, 11, 12]),
        standard_mean=np.zeros(14),
        standard_sd=np.full(14, 0.04),
        log_bounds=np.tile([-10.5, -4.5], (6, 7, 1)),
        contrasts=H,
        contrast_bounds=hb,
        R=0.00831446261815324,
        T=298.15,
        tau=1e-07,
        epsilon=0.001,
        excess=0.1,
        target=9,
        normalization=8.0,
    )
SCICODE_GOLD_EOF
