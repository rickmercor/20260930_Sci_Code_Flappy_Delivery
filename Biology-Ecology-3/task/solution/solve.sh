#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def estimate_rates(
    release_records: "numpy.typing.ArrayLike",
    reproduction_records: "numpy.typing.ArrayLike",
    terminal_age: int = 13,
) -> tuple:
    """Reference implementation for estimate_rates."""
    import numpy as np

    def _curve(age, s0, sp, k, extra, terminal_age, model="late"):
        """The stated survival curves. The pooled class takes the terminal-age value."""
        import numpy as np

        a = np.minimum(np.asarray(age, dtype=float), float(terminal_age))
        base = s0 + (sp - s0) * (1.0 - np.exp(-k * a))
        if model == "late":
            return base * np.exp(-extra * np.maximum(0.0, a - 8.0))
        return base / (1.0 + np.exp((a - 11.0) / extra))


    def _unpack(theta):
        import numpy as np

        s0, sp, p = 1.0 / (1.0 + np.exp(-theta[:3]))
        k, d = np.exp(theta[3]), np.exp(theta[4])
        return s0, sp, k, d, p


    def _census_schedule(all_years, k_cells):
        """The census years implied by a printed table: the years present, plus the
        consecutive years after the last one. Missing years inside the span are the
        paused censuses, so a row's columns follow the census sequence, not the years."""
        import numpy as np

        present = sorted({int(y) for y in all_years})
        pset = set(present)
        schedule = [y for y in range(present[0], present[-1] + 1) if y in pset]
        schedule += list(range(present[-1] + 1, present[-1] + 1 + k_cells))
        return schedule


    def _intervals(all_years, k_cells):
        """Per row, the years from its release census to each of the following K."""
        import numpy as np

        schedule = _census_schedule(all_years, k_cells)
        out = np.empty((len(all_years), k_cells))
        for i, y in enumerate(all_years):
            following = [c for c in schedule if c > y][:k_cells]
            out[i] = [c - y for c in following]
        return out


    def _apply_rules(m):
        """Confirmation, then validity, then recovery removal. Returns effective data."""
        import numpy as np

        years = m[:, 0].astype(float)
        births = m[:, 1].astype(float)
        released = m[:, 2].astype(float)
        dead = m[:, 3].astype(float)
        cols = m[:, 4:].astype(float).copy()
        deltas = _intervals(years, cols.shape[1])
        for i in range(len(cols)):
            nz = np.nonzero(cols[i])[0]
            if len(nz):
                cols[i, nz[-1]] = 0.0
        totals = released - dead
        keep = cols.sum(axis=1) > 0
        if np.any(totals[keep] - cols[keep].sum(axis=1) < -1e-9):
            raise ValueError("a group has fewer females at risk than confirmed captures")
        return years[keep], births[keep], totals[keep], cols[keep], deltas[keep]


    def _likelihood(theta, eff, terminal_age, model="late"):
        """Negative log-likelihood of the usable records under a candidate model."""
        import numpy as np

        years, births, totals, cols, deltas = eff
        s0, sp, k, extra, p = _unpack(theta)
        if not (0.0 < p < 1.0) or k <= 0.0 or extra <= 0.0:
            return 1e12
        k_cells = cols.shape[1]
        ages = np.minimum(years - births - 1, terminal_age).astype(float)
        surv = np.ones(len(years))
        probs = np.empty((len(years), k_cells))
        max_delta = int(deltas.max())
        for t in range(k_cells):
            lo = deltas[:, t - 1] if t else np.zeros(len(years))
            hi = deltas[:, t]
            for u in range(max_delta):
                surv = np.where(
                    (lo <= u) & (u < hi),
                    surv * _curve(ages + u, s0, sp, k, extra, terminal_age, model),
                    surv,
                )
            probs[:, t] = surv * p * (1.0 - p) ** t
        if np.any(probs < 0.0) or np.any(probs.sum(axis=1) > 1.0 + 1e-9):
            return 1e12
        never = 1.0 - probs.sum(axis=1)
        counts = np.concatenate([cols, (totals - cols.sum(axis=1))[:, None]], axis=1)
        vals = np.clip(np.concatenate([probs, never[:, None]], axis=1), 1e-12, 1.0)
        return -float((counts * np.log(vals)).sum())


    def _nelder_mead(fun, x0, step=0.25, max_iter=4000, tol=1e-10):
        """Deterministic derivative-free minimization (numpy only)."""
        import numpy as np

        n = len(x0)
        simplex = [np.asarray(x0, dtype=float).copy()]
        for i in range(n):
            x = np.asarray(x0, dtype=float).copy()
            x[i] += step
            simplex.append(x)
        simplex = np.array(simplex)
        fvals = np.array([fun(x) for x in simplex])
        for _ in range(max_iter):
            order = np.argsort(fvals)
            simplex, fvals = simplex[order], fvals[order]
            if np.max(np.abs(simplex[1:] - simplex[0])) < tol:
                break
            centroid = simplex[:-1].mean(axis=0)
            xr = centroid + (centroid - simplex[-1])
            fr = fun(xr)
            if fr < fvals[0]:
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = fun(xe)
                simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = fun(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                    fvals[1:] = np.array([fun(x) for x in simplex[1:]])
        order = np.argsort(fvals)
        return simplex[order[0]]


    m = np.asarray(release_records)
    if m.ndim != 2 or m.shape[0] == 0 or m.shape[1] < 4 + 2:
        raise ValueError("release_records must be a non-empty 2-D array with at "
                         "least six columns")
    if not np.all(np.isfinite(m)):
        raise ValueError("release_records entries must be finite")
    if np.any(m != np.round(m)):
        raise ValueError("release_records entries must be integer-valued")
    if np.any(m < 0):
        raise ValueError("release_records entries must be non-negative")
    if np.any(m[:, 0] <= m[:, 1]):
        raise ValueError("a release year must follow the birth year")
    if np.any(m[:, 4:].sum(axis=1) > m[:, 2]):
        raise ValueError("a release group cannot be recaptured more often than released")
    if not isinstance(terminal_age, (int, np.integer)) or isinstance(terminal_age, bool) \
            or terminal_age < 2:
        raise ValueError("terminal_age must be an integer of at least 2")

    eff = _apply_rules(m)
    ages = np.minimum(eff[0] - eff[1] - 1, int(terminal_age))
    if any(a not in ages for a in range(int(terminal_age) + 1)):
        raise ValueError("every survival class must appear among the usable release groups")

    r = np.asarray(reproduction_records)
    if r.ndim != 2 or r.shape[1] != 5 or r.size == 0:
        raise ValueError("reproduction_records must be a non-empty 2-D array with 5 columns")
    if not np.all(np.isfinite(r)):
        raise ValueError("reproduction_records entries must be finite")
    if np.any(r != np.round(r)):
        raise ValueError("reproduction_records entries must be integer-valued")
    if np.any(r < 0):
        raise ValueError("reproduction_records entries must be non-negative")
    if np.any(r[:, 3] > r[:, 2]):
        raise ValueError("more females cannot give birth than were present")

    T = int(terminal_age)
    candidates = {
        "late": [(0.60, 0.90, 0.8, 0.10, 0.80),
                 (0.50, 0.85, 0.4, 0.20, 0.70),
                 (0.70, 0.95, 1.5, 0.05, 0.85)],
        "logistic": [(0.60, 0.90, 0.8, 2.0, 0.80),
                     (0.50, 0.85, 1.2, 3.0, 0.70),
                     (0.70, 0.95, 0.5, 1.0, 0.85)],
    }
    selected = None
    for model, starts in candidates.items():
        for s0, sp, k, extra, p0 in starts:
            x0 = np.array([np.log(s0 / (1 - s0)), np.log(sp / (1 - sp)),
                           np.log(p0 / (1 - p0)), np.log(k), np.log(extra)])
            xb = _nelder_mead(
                lambda th: _likelihood(th, eff, T, model), x0, max_iter=8000)
            f = _likelihood(xb, eff, T, model)
            if selected is None or f < selected[0]:
                selected = (f, xb, model)
    s0, sp, k, extra, p = _unpack(selected[1])
    phi = [float(_curve(a, s0, sp, k, extra, T, selected[2])) for a in range(T + 1)]

    fec_obs = {a: [] for a in range(T + 1)}
    for pulse_year, birth_year, present, _gave, fawns in r:
        age = pulse_year - birth_year
        if age < 2:
            continue
        fec_obs[min(int(age), T)].append((present, fawns * 0.5))
    fecundity = [0.0, 0.0]
    for a in range(2, T + 1):
        pairs = fec_obs[a]
        if not pairs:
            raise ValueError(f"no reproduction records for age class {a}")
        fecundity.append(round(sum(n for _p, n in pairs) / sum(p for p, _n in pairs), 3))

    survival = [round(float(v), 3) for v in phi]
    return np.array(survival, dtype=float), np.array(fecundity, dtype=float)

def transition_matrix(
    survival: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reference implementation for transition_matrix."""
    import numpy as np

    s = np.asarray(survival, dtype=float)
    if s.ndim != 1 or s.size < 2:
        raise ValueError("survival must be a 1-D sequence of at least two classes")
    if not np.all(np.isfinite(s)):
        raise ValueError("survival probabilities must be finite")
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("survival probabilities must lie in [0, 1]")

    n = s.size
    u = np.zeros((n, n), dtype=float)
    for a in range(n - 1):
        u[a + 1, a] = s[a]
    u[n - 1, n - 1] = s[n - 1]
    return u

def birth_pulse_operator(
    fecundity: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reference implementation for birth_pulse_operator."""
    import numpy as np

    r = np.asarray(fecundity, dtype=float)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("fecundity must be a 1-D sequence of at least two classes")
    if not np.all(np.isfinite(r)):
        raise ValueError("fecundities must be finite")
    if np.any(r < 0.0):
        raise ValueError("fecundities must be non-negative")

    n = r.size
    raw = np.zeros((n, n), dtype=float)
    raw[0, :] = r
    return np.eye(n, dtype=float) + raw

def survival_ageing_decomposition(
    transition: "numpy.typing.ArrayLike",
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Reference implementation for survival_ageing_decomposition."""
    import numpy as np

    u = np.asarray(transition, dtype=float)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("the transition matrix must be square")
    if u.shape[0] < 2:
        raise ValueError("the transition matrix must hold at least two classes")
    if not np.all(np.isfinite(u)):
        raise ValueError("transition entries must be finite")
    if np.any(u < 0.0) or np.any(u > 1.0):
        raise ValueError("transition entries must lie in [0, 1]")

    n = u.shape[0]
    ageing = np.zeros((n, n), dtype=float)
    for a in range(n - 1):
        ageing[a + 1, a] = 1.0
    ageing[n - 1, n - 1] = 1.0
    if np.any(u[ageing == 0.0] != 0.0):
        raise ValueError("only the first sub-diagonal and the last diagonal entry may "
                         "be non-zero")

    survival = np.array([u[a + 1, a] for a in range(n - 1)] + [u[n - 1, n - 1]])
    survival_operator = np.diag(survival)
    return survival_operator, ageing

def fractional_survival_powers(
    survival: "numpy.typing.ArrayLike",
    months: float,
) -> "tuple[numpy.ndarray, numpy.ndarray]":
    """Reference implementation for fractional_survival_powers."""
    import numpy as np

    s = np.asarray(survival, dtype=float)
    if s.ndim != 1 or s.size < 2:
        raise ValueError("survival must be a 1-D sequence of at least two classes")
    if not np.all(np.isfinite(s)):
        raise ValueError("survival probabilities must be finite")
    if np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("survival probabilities must lie in [0, 1]")
    if not np.isfinite(months) or months < 0.0 or months > 12.0:
        raise ValueError("months must be finite and lie in [0, 12]")

    exponent = float(months) / 12.0
    S_a, S_b = np.diag(s ** exponent), np.diag(s ** (1.0 - exponent))
    return S_a, S_b

def intermediate_projection(
    survival: "numpy.typing.ArrayLike",
    fecundity: "numpy.typing.ArrayLike",
    months: float = 11,
) -> "numpy.ndarray":
    """Reference implementation for intermediate_projection."""
    import numpy as np

    transition = transition_matrix(survival)
    survival_operator, ageing = survival_ageing_decomposition(transition)
    survival_elapsed, survival_remaining = fractional_survival_powers(
        np.diag(survival_operator), months)
    pulse = birth_pulse_operator(fecundity)
    if pulse.shape[0] != ageing.shape[0]:
        raise ValueError("survival and fecundity must cover the same age classes")
    L = survival_elapsed @ pulse @ ageing @ survival_remaining
    return L

def dominant_eigenpair(
    matrix: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reference implementation for dominant_eigenpair."""
    import numpy as np

    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    if m.shape[0] < 2:
        raise ValueError("matrix must hold at least two classes")
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix entries must be finite")
    if np.any(m < 0.0):
        raise ValueError("projection matrix entries must be non-negative")

    values, vectors = np.linalg.eig(m)
    k = int(np.argmax(values.real))
    growth = values[k]
    if abs(growth.imag) > 1e-8 * max(1.0, abs(growth.real)) or growth.real <= 0.0:
        raise ValueError("the dominant eigenvalue must be real and positive")
    growth = float(growth.real)

    shares = np.asarray(vectors[:, k].real, dtype=float)
    total = shares.sum()
    if abs(total) < 1e-12:
        raise ValueError("the stable vector cannot be normalized")
    shares = shares / total
    if shares.sum() < 0:
        shares = -shares
    if np.any(shares < -1e-9):
        raise ValueError("the stable vector must be non-negative")
    shares = np.clip(shares, 0.0, None)
    shares = shares / shares.sum()
    return np.concatenate([[growth], shares])

def reproductive_values(
    matrix: "numpy.typing.ArrayLike",
) -> "numpy.ndarray":
    """Reference implementation for reproductive_values."""
    import numpy as np

    m = np.asarray(matrix, dtype=float)
    packed = dominant_eigenpair(m)
    growth, stable = float(packed[0]), packed[1:]

    values_t, vectors_t = np.linalg.eig(m.T)
    k = int(np.argmin(np.abs(values_t - growth)))
    left = np.asarray(vectors_t[:, k].real, dtype=float)
    left = left / (left @ stable)
    if abs(left[0]) < 1e-12:
        raise ValueError("the reproductive value vector cannot be anchored")
    rv = left / left[0]
    if np.any(rv < -1e-9):
        raise ValueError("the reproductive values must be non-negative")
    rv = np.clip(rv, 0.0, None)
    return rv

def relative_reproductive_value(
    release_records: "numpy.typing.ArrayLike",
    reproduction_records: "numpy.typing.ArrayLike",
    age: int = 8,
    months: float = 11,
) -> float:
    """Reference implementation for relative_reproductive_value."""
    import numpy as np

    survival, fecundity = estimate_rates(release_records, reproduction_records)
    projection = intermediate_projection(survival, fecundity, months)
    rv = reproductive_values(projection)
    if not isinstance(age, (int, np.integer)) or isinstance(age, bool):
        raise ValueError("age must be an integer")
    if age < 0 or age > rv.size - 1:
        raise ValueError("age must be from 0 to the number of classes minus one")
    return float(rv[int(age)])
SCICODE_GOLD_EOF
