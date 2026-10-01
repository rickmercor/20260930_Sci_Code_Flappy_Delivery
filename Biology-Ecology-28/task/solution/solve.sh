#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def regularized_gamma(a: float, x: float) -> list:
    import math

    def _real(value, name):
        if isinstance(value, complex):
            if value.imag != 0.0:
                raise ValueError(f"{name} must be a real number")
            value = value.real
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a real number") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    a = _real(a, "a")
    x = _real(x, "x")
    if a <= 0.0:
        raise ValueError("a must be positive")
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return [0.0, 1.0]

    tiny = 1e-300
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        shape = a
        for _ in range(100000):
            shape += 1.0
            term *= x / shape
            total += term
            if abs(term) <= abs(total) * 1e-17:
                break
        else:
            raise ValueError("incomplete gamma series did not converge")
        lower = min(math.exp(log_prefactor + math.log(total)), 1.0)
        return [float(lower), float(1.0 - lower)]

    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= 1e-15:
            break
    else:
        raise ValueError("incomplete gamma continued fraction did not converge")
    upper = min(math.exp(log_prefactor + math.log(h)), 1.0)
    return [float(1.0 - upper), float(upper)]

def regularized_beta(a: float, b: float, x: float) -> list:
    import math

    def _real(value, name):
        if isinstance(value, complex):
            if value.imag != 0.0:
                raise ValueError(f"{name} must be a real number")
            value = value.real
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a real number") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    a = _real(a, "a")
    b = _real(b, "b")
    x = _real(x, "x")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x < 0.0 or x > 1.0:
        raise ValueError("x must lie in [0, 1]")
    if x == 0.0:
        return [0.0, 1.0]
    if x == 1.0:
        return [1.0, 0.0]

    tiny = 1e-300

    def _continued_fraction(p, r, z):
        qab = p + r
        qap = p + 1.0
        qam = p - 1.0
        c = 1.0
        d = 1.0 - qab * z / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 100000):
            m2 = 2.0 * m
            aa = m * (r - m) * z / ((qam + m2) * (p + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            aa = -(p + m) * (qab + m) * z / ((p + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) <= 1e-15:
                return h
        raise ValueError("incomplete beta continued fraction did not converge")

    log_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        lower = min(math.exp(log_front + math.log(_continued_fraction(a, b, x))) / a, 1.0)
        return [float(lower), float(1.0 - lower)]
    upper = min(math.exp(log_front + math.log(_continued_fraction(b, a, 1.0 - x))) / b, 1.0)
    return [float(1.0 - upper), float(upper)]

def poisson_adjusted_moments(distances: list, C: float, ell: int) -> list:
    import math

    def _real(value, name):
        if isinstance(value, complex):
            if value.imag != 0.0:
                raise ValueError(f"{name} must be a real number")
            value = value.real
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a real number") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _order(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer >= 1")
        number = _real(value, name)
        if number != math.floor(number) or number < 1.0:
            raise ValueError(f"{name} must be an integer >= 1")
        return int(number)

    def _survey(table, radius):
        if isinstance(table, (str, bytes)):
            raise ValueError("distances must be a sequence of rows")
        try:
            rows = list(table)
        except TypeError:
            raise ValueError("distances must be a sequence of rows") from None
        if not rows:
            raise ValueError("distances must contain at least one sampling point")
        observed = []
        censored = 0
        width = None
        for row in rows:
            if isinstance(row, (str, bytes)):
                raise ValueError("each sampling point must be a sequence of sector distances")
            try:
                entries = list(row)
            except TypeError:
                raise ValueError("each sampling point must be a sequence of sector distances") from None
            if not entries:
                raise ValueError("each sampling point needs at least one sector")
            if width is None:
                width = len(entries)
            elif len(entries) != width:
                raise ValueError("all sampling points must have the same number of sectors")
            for entry in entries:
                if entry is None:
                    censored += 1
                    continue
                r = _real(entry, "distance")
                if r <= 0.0:
                    raise ValueError("distances must be positive")
                if r > radius:
                    censored += 1
                else:
                    observed.append(r)
        if not observed:
            raise ValueError("every sector is censored")
        return observed, censored, len(rows), width

    C = _real(C, "C")
    if C <= 0.0:
        raise ValueError("C must be positive")
    ell = _order(ell, "ell")
    observed, censored, n_points, q = _survey(distances, C)
    total = n_points * q
    first = math.fsum(observed) / total
    second = math.fsum(r * r for r in observed) / total
    if censored == 0:
        return [math.inf, float(first), float(second)]

    p_obs = (total - censored) / total

    def _lower(shape, m):
        return regularized_gamma(shape, m)[0]

    hi = max(1.0, float(ell))
    while _lower(ell, hi) < p_obs:
        hi *= 2.0
        if hi > 1e300:
            raise ValueError("censoring quantile is not representable")
    lo = 0.0
    for _ in range(4000):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _lower(ell, mid) < p_obs:
            lo = mid
        else:
            hi = mid
    m_hat = lo if abs(_lower(ell, lo) - p_obs) < abs(_lower(ell, hi) - p_obs) else hi

    moment1 = first / _lower(ell + 0.5, m_hat)
    moment2 = second / _lower(ell + 1.0, m_hat)
    return [float(m_hat), float(moment1), float(moment2)]

def censored_poisson_densities(
    distances: list, C: float, ell: int
) -> list:
    """Oracle implementation of censored_poisson_densities."""
    import math

    moments = poisson_adjusted_moments(distances, C, ell)
    M_1 = moments[1]
    M_2 = moments[2]
    n = len(distances)
    q = len(distances[0])

    lambda_C = q * ell / (4.0 * M_1**2)
    lambda_P = (n * q * ell - 1.0) / (math.pi * n * M_2)

    return [float(lambda_C), float(lambda_P)]

def csr_tail_moment(u: float, ell: int, q: int, lam: float, C: float) -> float:
    import math

    def _real(value, name):
        if isinstance(value, complex):
            if value.imag != 0.0:
                raise ValueError(f"{name} must be a real number")
            value = value.real
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a real number") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _order(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer >= 1")
        number = _real(value, name)
        if number != math.floor(number) or number < 1.0:
            raise ValueError(f"{name} must be an integer >= 1")
        return int(number)

    u = _real(u, "u")
    ell = _order(ell, "ell")
    q = _order(q, "q")
    lam = _real(lam, "lam")
    C = _real(C, "C")
    if lam <= 0.0 or C <= 0.0:
        raise ValueError("lam and C must be positive")
    shape = ell + 0.5 * u
    if shape <= 0.0:
        raise ValueError("ell + u / 2 must be positive")
    scale = math.pi * lam / q
    T = scale * C * C
    if not math.isfinite(T):
        raise ValueError("pi * lam * C**2 / q must be finite")

    tiny = 1e-300

    def _log_upper_gamma(s, x):
        if x == 0.0:
            return math.lgamma(s)
        if x < s + 1.0:
            lower = regularized_gamma(s, x)[0]
            if lower >= 1.0:
                raise ValueError("upper incomplete gamma function underflows")
            return math.lgamma(s) + math.log1p(-lower)
        b = x + 1.0 - s
        c = 1.0 / tiny
        d = 1.0 / b
        h = d
        for i in range(1, 100000):
            an = -i * (i - s)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) <= 1e-15:
                break
        else:
            raise ValueError("incomplete gamma continued fraction did not converge")
        return -x + s * math.log(x) + math.log(h)

    log_moment = -0.5 * u * math.log(scale) + _log_upper_gamma(shape, T) - _log_upper_gamma(float(ell), T)
    try:
        moment = math.exp(log_moment)
    except OverflowError:
        raise ValueError("tail moment is not representable as a finite float") from None
    if not math.isfinite(moment) or moment == 0.0:
        raise ValueError("tail moment is not representable as a finite float")
    return float(moment)

def censored_shen_estimates(distances: list, C: float, ell: int) -> list:
    import math

    _, lam_init = censored_poisson_densities(distances, C, ell)
    if not lam_init > 0.0:
        raise ValueError("the censored Pollard-type density must be positive")
    radius = float(C)
    order = int(float(ell))
    rows = [list(row) for row in distances]
    n_points = len(rows)
    q = len(rows[0])
    total = n_points * q
    observed = []
    censored = 0
    for row in rows:
        for entry in row:
            if entry is None or float(entry) > radius:
                censored += 1
            else:
                observed.append(float(entry))

    adjusted = {}
    for u in (-1.0, 1.0, 2.0):
        tail = csr_tail_moment(u, order, q, lam_init, radius) if censored else 0.0
        adjusted[u] = (math.fsum(r ** u for r in observed) + censored * tail) / total

    lam_n = (q * (2 * order - 1) * adjusted[-1.0] / (math.pi * adjusted[1.0])
             - q * order / (math.pi * adjusted[2.0]))
    ratio = adjusted[-1.0] * adjusted[2.0] / adjusted[1.0]
    denominator = (2 * order - 1) * ratio - 2 * order
    if denominator == 0.0:
        raise ValueError("aggregation estimate is undefined for this moment ratio")
    k_n = ((2 * order - 1) * ratio - order) / denominator
    return [float(lam_n), float(k_n)]

def nbd_censored_log_likelihood(distances: list, C: float, ell: int, lam: float, k: float) -> float:
    import math

    def _real(value, name):
        if isinstance(value, complex):
            if value.imag != 0.0:
                raise ValueError(f"{name} must be a real number")
            value = value.real
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a real number") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _order(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer >= 1")
        number = _real(value, name)
        if number != math.floor(number) or number < 1.0:
            raise ValueError(f"{name} must be an integer >= 1")
        return int(number)

    def _survey(table, radius):
        if isinstance(table, (str, bytes)):
            raise ValueError("distances must be a sequence of rows")
        try:
            rows = list(table)
        except TypeError:
            raise ValueError("distances must be a sequence of rows") from None
        if not rows:
            raise ValueError("distances must contain at least one sampling point")
        observed = []
        censored = 0
        width = None
        for row in rows:
            if isinstance(row, (str, bytes)):
                raise ValueError("each sampling point must be a sequence of sector distances")
            try:
                entries = list(row)
            except TypeError:
                raise ValueError("each sampling point must be a sequence of sector distances") from None
            if not entries:
                raise ValueError("each sampling point needs at least one sector")
            if width is None:
                width = len(entries)
            elif len(entries) != width:
                raise ValueError("all sampling points must have the same number of sectors")
            for entry in entries:
                if entry is None:
                    censored += 1
                    continue
                r = _real(entry, "distance")
                if r <= 0.0:
                    raise ValueError("distances must be positive")
                if r > radius:
                    censored += 1
                else:
                    observed.append(r)
        if not observed:
            raise ValueError("every sector is censored")
        return observed, censored, len(rows), width

    C = _real(C, "C")
    lam = _real(lam, "lam")
    k = _real(k, "k")
    if C <= 0.0 or lam <= 0.0 or k <= 0.0:
        raise ValueError("C, lam and k must be positive")
    ell = _order(ell, "ell")
    observed, censored, _, q = _survey(distances, C)

    rate = math.pi * lam / q
    constant = (math.log(2.0) + ell * math.log(rate) + math.lgamma(ell + k) - math.lgamma(k)
                - math.lgamma(ell) - ell * math.log(k))
    terms = [constant + (2 * ell - 1) * math.log(r) - (ell + k) * math.log1p(rate * r * r / k) for r in observed]
    if censored:
        mass = rate * C * C
        w = mass / (mass + k)
        survival = regularized_beta(float(ell), k, w)[1]
        if survival <= 0.0:
            raise ValueError("censoring probability underflows to zero")
        terms.append(censored * math.log(survival))
    value = math.fsum(terms)
    if not math.isfinite(value):
        raise ValueError("log-likelihood is not finite")
    return float(value)

def nbd_censored_mle(distances: list, C: float, ell: int) -> list:
    import math
 
    radius = float(C)
    order = int(float(ell)) if not isinstance(ell, bool) else ell
    # validates every input (raises ValueError for invalid surveys, radius or order)
    nbd_censored_log_likelihood(distances, C, ell, 1.0, 1.0)
    rows = [list(row) for row in distances]
    q = len(rows[0])
    observed = []
    censored = 0
    for row in rows:
        for entry in row:
            if entry is None or float(entry) > radius:
                censored += 1
            else:
                observed.append(float(entry))
 
    def _loglik(s, t):
        return nbd_censored_log_likelihood(distances, radius, order, math.exp(s), math.exp(t))
 
    def _gradient(s, t):
        lam = math.exp(s)
        k = math.exp(t)
        rate = math.pi * lam / q
        harmonic = math.fsum(1.0 / (k + i) for i in range(order))
        gs = []
        gt = []
        for r in observed:
            z = rate * r * r / k
            frac = z / (1.0 + z)
            gs.append(order - (order + k) * frac)
            gt.append(k * harmonic - order - k * math.log1p(z) + (order + k) * frac)
        if censored:
            mass = rate * radius * radius
            w = mass / (mass + k)
            log_w = math.log(w)
            log_1mw = math.log1p(-w)
            log_terms = [math.lgamma(k + j) - math.lgamma(k) - math.lgamma(j + 1) + j * log_w + k * log_1mw
                         for j in range(order)]
            top = max(log_terms)
            log_survival = top + math.log(math.fsum(math.exp(v - top) for v in log_terms))
            partial = [math.fsum(1.0 / (k + i) for i in range(j)) for j in range(order)]
            dk_fixed_w = math.fsum(math.exp(v - log_survival) * (partial[j] + log_1mw)
                                   for j, v in enumerate(log_terms))
            log_density = ((order - 1) * log_w + (k - 1.0) * log_1mw
                           - (math.lgamma(order) + math.lgamma(k) - math.lgamma(order + k)))
            density_ratio = math.exp(log_density - log_survival) * w * (1.0 - w)
            gs.append(-censored * density_ratio)
            gt.append(censored * (k * dk_fixed_w + density_ratio))
        return [math.fsum(gs), math.fsum(gt)]
 
    try:
        lam0 = censored_shen_estimates(distances, radius, order)
        start_lam = lam0[0] if lam0[0] > 0.0 else censored_poisson_densities(distances, radius, order)[1]
        start_k = min(max(lam0[1], 0.5), 1e3) if lam0[1] > 0.0 else 1.0
    except ValueError:
        start_lam = 0.0
        start_k = 1.0
    if not start_lam > 0.0:
        mean_sq = math.fsum(r * r for r in observed) / len(observed)
        start_lam = order * q / (math.pi * mean_sq)
 
    s = math.log(start_lam)
    t = math.log(start_k)
    f = _loglik(s, t)
    h = 1e-5
    t_max = math.log(1e8)
    converged = False
    for _ in range(2000):
        g = _gradient(s, t)
        gp = _gradient(s + h, t)
        gm = _gradient(s - h, t)
        hs0 = (gp[0] - gm[0]) / (2.0 * h)
        hs1 = (gp[1] - gm[1]) / (2.0 * h)
        gp = _gradient(s, t + h)
        gm = _gradient(s, t - h)
        ht0 = (gp[0] - gm[0]) / (2.0 * h)
        ht1 = (gp[1] - gm[1]) / (2.0 * h)
        h00 = hs0
        h11 = ht1
        h01 = 0.5 * (hs1 + ht0)
        det = h00 * h11 - h01 * h01
        if h00 < 0.0 and det > 0.0:
            ds = -(h11 * g[0] - h01 * g[1]) / det
            dt = -(-h01 * g[0] + h00 * g[1]) / det
            newton = True
        else:
            norm = max(abs(g[0]), abs(g[1]), 1e-300)
            ds = g[0] / norm
            dt = g[1] / norm
            newton = False
        longest = max(abs(ds), abs(dt))
        if longest > 1.0:
            ds /= longest
            dt /= longest
        if newton and longest < 1e-11:
            converged = True
            break
        step = 1.0
        accepted = False
        while step > 1e-14:
            sn = s + step * ds
            tn = t + step * dt
            try:
                fn = _loglik(sn, tn)
            except ValueError:
                fn = -math.inf
            if fn >= f - 1e-12 * (1.0 + abs(f)):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        s, t, f = sn, tn, fn
        if t > t_max:
            raise ValueError("the log-likelihood has no interior maximum with k <= 1e8")
    if not converged:
        g = _gradient(s, t)
        if not (max(abs(g[0]), abs(g[1])) <= 1e-7 * (1.0 + len(observed) + censored)):
            raise ValueError("censored NBD maximization did not converge")
    if t > t_max:
        raise ValueError("the log-likelihood has no interior maximum with k <= 1e8")
    return [float(math.exp(s)), float(math.exp(t)), float(_loglik(s, t))]

def nbd_truncated_moment(u: float, lam: float, k: float, q: int, ell: int, C: float) -> list:
    import math

    def _real(value, name):
        if isinstance(value, complex):
            if value.imag != 0.0:
                raise ValueError(f"{name} must be a real number")
            value = value.real
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a real number") from None
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _order(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer >= 1")
        number = _real(value, name)
        if number != math.floor(number) or number < 1.0:
            raise ValueError(f"{name} must be an integer >= 1")
        return int(number)

    u = _real(u, "u")
    lam = _real(lam, "lam")
    k = _real(k, "k")
    C = _real(C, "C")
    q = _order(q, "q")
    ell = _order(ell, "ell")
    if lam <= 0.0 or k <= 0.0 or C <= 0.0:
        raise ValueError("lam, k and C must be positive")
    shape_a = ell + 0.5 * u
    shape_b = k - 0.5 * u
    if shape_a <= 0.0 or shape_b <= 0.0:
        raise ValueError("the moment requires ell + u/2 > 0 and k - u/2 > 0")

    mass = math.pi * lam * C * C / q
    if not math.isfinite(mass):
        raise ValueError("pi * lam * C**2 / q must be finite")
    w = mass / (mass + k)
    log_complete = (0.5 * u * math.log(k * q / (math.pi * lam)) + math.lgamma(shape_a) + math.lgamma(shape_b)
                    - math.lgamma(ell) - math.lgamma(k))
    try:
        complete = math.exp(log_complete)
    except OverflowError:
        raise ValueError("moment is not representable as a finite float") from None
    partial = complete * regularized_beta(shape_a, shape_b, w)[0]
    censoring = regularized_beta(float(ell), k, w)[1]
    if not math.isfinite(partial):
        raise ValueError("moment is not representable as a finite float")
    return [float(partial), float(censoring)]

def asymptotic_shen_bias(lam: float, k: float, q: int, ell: int, C: float) -> list:
    import math

    partial = {}
    p0 = None
    for u in (-1.0, 1.0, 2.0):
        partial[u], p0 = nbd_truncated_moment(u, lam, k, q, ell, C)
    lam = float(lam)
    k = float(k)
    q = int(float(q))
    ell = int(float(ell))
    C = float(C)
    mass = math.pi * lam * C * C / q
    w = mass / (mass + k)
    p_obs = regularized_beta(float(ell), k, w)[0]

    if p0 == 0.0:
        second = partial[2.0]
        lam_init = ell * q / (math.pi * second)
        adjusted = dict(partial)
    else:
        if p_obs <= 0.0:
            raise ValueError("the limiting uncensored fraction underflows to zero")

        def _lower(shape, m):
            return regularized_gamma(shape, m)[0]

        hi = max(1.0, float(ell))
        while _lower(ell, hi) < p_obs:
            hi *= 2.0
            if hi > 1e300:
                raise ValueError("censoring quantile is not representable")
        lo = 0.0
        for _ in range(4000):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if _lower(ell, mid) < p_obs:
                lo = mid
            else:
                hi = mid
        m_inf = lo if abs(_lower(ell, lo) - p_obs) < abs(_lower(ell, hi) - p_obs) else hi
        second = partial[2.0] / _lower(ell + 1.0, m_inf)
        lam_init = ell * q / (math.pi * second)
        adjusted = {u: partial[u] + p0 * csr_tail_moment(u, ell, q, lam_init, C) for u in (-1.0, 1.0, 2.0)}

    lam_n = (q * (2 * ell - 1) * adjusted[-1.0] / (math.pi * adjusted[1.0])
             - q * ell / (math.pi * adjusted[2.0]))
    return [float(lam_init), float(lam_n), float(lam_n / lam - 1.0)]

def censored_design_bias(distances: list, C: float, ell: int) -> float:
    lam_hat, k_hat, _ = nbd_censored_mle(distances, C, ell)
    if not k_hat > 1.0:
        raise ValueError("the fitted aggregation parameter must exceed 1")
    q = len(list(list(distances)[0]))
    return float(asymptotic_shen_bias(lam_hat, k_hat, q, ell, C)[2])
SCICODE_GOLD_EOF
