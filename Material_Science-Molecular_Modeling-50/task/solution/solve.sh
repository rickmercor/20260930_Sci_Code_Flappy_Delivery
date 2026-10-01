#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def fjc_log_density(n: int, r: float) -> float:
    nn = _links(n); rr = _length(r, "r")
    if rr >= nn: raise ValueError("r must be smaller than the contour length n")
    v = _lnp(nn, rr)
    if not np.isfinite(v): raise ValueError("non-finite density")
    return float(v)

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def kuhn_grun_free_energy(n: int, beps: float, x: float, y: float) -> float:
    nn = _links(n); b = _energy(beps); xx = _length(x, "x"); yy = _length(y, "y", strict=False)
    v = _kg_free(nn, b, xx, yy)
    if not np.isfinite(v): raise ValueError("non-finite free energy")
    return float(v)

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def breakable_chain_free_energy(n: int, beps: float, x: float, y: float) -> float:
    nn = _links(n); b = _energy(beps); xx = _length(x, "x"); yy = _length(y, "y", strict=False)
    v = _free(nn, b, xx, yy)[0]
    if not np.isfinite(v): raise ValueError("non-finite free energy")
    return float(v)

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def chain_landscape(n: int, beps: float, y: float) -> np.ndarray:
    nn = _links(n); b = _energy(beps); yy = _length(y, "y", strict=False)
    x1, xt, x2, Es, Eh = _barriers(nn, b, yy)
    out = np.array([x1, xt, x2, Es, Eh], dtype=np.float64)
    if not np.all(np.isfinite(out)): raise ValueError("non-finite landscape")
    return out

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def intact_branch_force(n: int, beps: float, y: float) -> float:
    nn = _links(n); b = _energy(beps); yy = _length(y, "y", strict=False)
    v = _force(nn, b, yy)
    if not np.isfinite(v): raise ValueError("non-finite force")
    return float(v)

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def chain_critical_point(n: int, beps: float) -> np.ndarray:
    nn = _links(n); b = _energy(beps)
    yc, xc, fc = _fold(nn, b)
    out = np.array([yc, xc, fc], dtype=np.float64)
    if not np.all(np.isfinite(out)): raise ValueError("non-finite critical point")
    return out

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def equilibrium_probability(n: int, beps: float, y: float) -> float:
    nn = _links(n); b = _energy(beps); yy = _length(y, "y", strict=False)
    v = _peq(nn, b, yy)
    if not np.isfinite(v): raise ValueError("non-finite probability")
    return float(v)

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def equilibrium_rupture_statistics(n: int, beps: float) -> np.ndarray:
    nn = _links(n); b = _energy(beps)
    out = np.array(_eq_stats(nn, b), dtype=np.float64)
    if not np.all(np.isfinite(out)): raise ValueError("non-finite statistics")
    return out

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def pulled_chain_survival(n: int, beps: float, gbar: float, y: float) -> float:
    nn = _links(n); b = _energy(beps); g = _pull_rate(gbar); yy = _length(y, "y", strict=False)
    if yy == 0.0: return 1.0
    if yy >= _fold(nn, b)[0]: return 0.0
    v = _pull(nn, b, g, ystop=yy)[0]
    if not np.isfinite(v): raise ValueError("non-finite probability")
    return float(v)

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def final_scission_statistics(n: int, beps: float, gbar: float) -> np.ndarray:
    nn = _links(n); b = _energy(beps); g = _pull_rate(gbar)
    P, ym, fm, norm = _pull(nn, b, g)
    out = np.array([ym, fm], dtype=np.float64)
    if not np.all(np.isfinite(out)): raise ValueError("non-finite statistics")
    return out

import numpy as np, math
from scipy.optimize import brentq, minimize_scalar, root
from scipy.integrate import quad, quad_vec, solve_ivp

def _fin(x, name):
    try: v = float(x)
    except Exception: raise ValueError(f"{name} must be a real number")
    if not np.isfinite(v): raise ValueError(f"{name} must be finite")
    return v
def _int(x, name, lo=0):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    v = int(x)
    if v < lo: raise ValueError(f"{name} must be at least {lo}")
    return v
def _links(n):
    return _int(n, "n", 2)
def _energy(beps):
    v = _fin(beps, "beps")
    if v <= 0.0: raise ValueError("beps must be positive")
    return v
def _pull_rate(gbar):
    v = _fin(gbar, "gbar")
    if not (0.0 < v <= 1e-2): raise ValueError("gbar must be positive and at most 1e-2")
    return v
def _length(x, name, strict=True):
    v = _fin(x, name)
    if v < 0.0 or (strict and v == 0.0): raise ValueError(f"{name} must be {'positive' if strict else 'non-negative'}")
    return v

def _treloar(n, r):
    """Exact end-to-end statistics of a freely jointed chain of n unit rigid links (n >= 2), for 0 <= r < n:
    G(r) = int_r^n r' p_n(r') dr',  H(r) = r p_n(r),  H'(r); p_n is the 3D density (int p_n d^3r = 1).
    The alternating Rayleigh/Treloar sums are evaluated in exact integer arithmetic (r taken as the exact
    binary rational it is) and rounded once to double, so no cancellation is lost for any n."""
    if r >= n: return 0.0, 0.0, 0.0
    m, e = math.frexp(r)                     # r = m 2^e
    s = 53 - e                               # r = M / 2^s exactly
    M = int(m * (1 << 53))
    sG = sH = sHp = 0
    for k in range(n // 2 + 1):
        t = (n - 2 * k) * (1 << s) - M       # 2^s (n - r - 2k)
        if t <= 0: break
        c = (-1) ** k * math.comb(n, k)
        if n >= 3:
            p3 = t ** (n - 3); sHp += c * p3; p2 = p3 * t
        else:
            p2 = 1
        sH += c * p2
        sG += c * p2 * t
    den = (1 << (n + 1)) * math.factorial(n - 2)
    G = (sG / (den * (n - 1) * (1 << (s * (n - 1))))) / math.pi
    H = (sH / (den * (1 << (s * (n - 2))))) / math.pi
    Hp = 0.0 if n == 2 else -(n - 2) * (sHp / (den * (1 << (s * (n - 3))))) / math.pi
    return G, H, Hp

def _lnp(n, r):
    """ln p_n(r), 0 < r < n"""
    if r <= 0.0 or r >= n: raise ValueError("r must lie strictly between 0 and n")
    H = _treloar(n, r)[1]
    if H <= 0.0: raise ValueError("density underflow")
    return math.log(H) - math.log(r)

def _lj(x, beps):
    """Lennard-Jones link in units of kT and l: V, V', V''"""
    x6 = x ** -6; x12 = x6 * x6
    return beps * (x12 - 2.0 * x6), beps * 12.0 * (x ** -7 - x ** -13), beps * (156.0 * x ** -14 - 84.0 * x ** -8)

def _free(n, beps, x, y):
    """beta A(x; y) = beta V(x) - ln[x^2 int_0^pi p_n(|y - x|_phi) sin(phi) dphi] with the exact normalised p_n,
    together with dA/dx and dA/dy; raises ValueError if the rigid fragments cannot span |y - x| (A infinite)."""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V, Vp, Vpp = _lj(x, beps)
    if y == 0.0:
        G, H, Hp = _treloar(n, x)
        if H <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
        return V - math.log(2.0 * x * H), Vp - 1.0 / x - Hp / H, 0.0
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0: raise ValueError("the free energy is infinite: the rigid fragments cannot span the gap")
    s = 1.0 if x > y else -1.0                # d|y - x|/dx
    A = V - math.log(D) - math.log(x / y)
    Ax = Vp - 1.0 / x - (Hb - s * Ha) / D
    Ay = 1.0 / y - (Hb + s * Ha) / D
    return A, Ax, Ay

def _Ax(n, beps, x, y):
    try: return _free(n, beps, x, y)[1]
    except ValueError: return math.inf

def _landscape(n, beps, y, fine=False):
    """critical points of beta A(x; y): (x1, xt, x2, A1, At, A2) or None when the intact well does not exist"""
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    if fine:
        xs = np.concatenate([np.linspace(xlo, 1.8, 4000), np.linspace(1.8, 3.0, 200)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    else:
        xs = np.concatenate([np.linspace(xlo, 1.8, 70), np.linspace(1.8, 3.0, 30)[1:], np.geomspace(3.0, xmax, 40)[1:]])
    f = lambda x: _Ax(n, beps, x, y)
    v = [f(x) for x in xs]
    idx = [i for i in range(1, len(xs)) if np.isfinite(v[i - 1]) and np.isfinite(v[i]) and (v[i - 1] > 0.0) != (v[i] > 0.0)]
    if len(idx) < 3 and not fine:
        return _landscape(n, beps, y, fine=True)
    if len(idx) == 1: return None
    if len(idx) != 3: raise ValueError("unexpected free-energy landscape")
    r = [brentq(f, xs[i - 1], xs[i], xtol=1e-14, rtol=1e-15, maxiter=200) for i in idx]
    if not (v[idx[0] - 1] < 0.0): return None       # the first crossing must be a minimum
    x1, xt, x2 = r
    A1 = _free(n, beps, x1, y)[0]; At = _free(n, beps, xt, y)[0]; A2 = _free(n, beps, x2, y)[0]
    return x1, xt, x2, A1, At, A2

def _barriers(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: raise ValueError("no intact well at this extension")
    x1, xt, x2, A1, At, A2 = L
    return x1, xt, x2, At - A1, At - A2

def _force(n, beps, y):
    """f = d beta A(x1(y); y) / dy along the intact branch (envelope theorem: partial derivative at x1)"""
    x1 = _barriers(n, beps, y)[0]
    return _free(n, beps, x1, y)[2]

def _wall(n, beps, y):
    """largest value of dA/dx between the intact minimum and the transition state (positive while the well exists)"""
    xlo = max(0.9, y - n + 1e-7)
    xs = np.linspace(xlo, 2.5, 240); v = [_Ax(n, beps, x, y) for x in xs]
    i = int(np.argmax(v))
    r = minimize_scalar(lambda x: -_Ax(n, beps, x, y), bounds=(xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]), method='bounded', options={'xatol': 1e-13})
    return -r.fun, r.x

def _Axx(n, beps, x, y):
    """Analytic bond-length curvature of the exact displacement-controlled free energy."""
    if y <= 0.0 or x == y:
        raise ValueError("fold curvature requires y > 0 and x != y")
    a = abs(y - x); b = y + x
    Ga, Ha, Hpa = _treloar(n, a); Gb, Hb, Hpb = _treloar(n, b)
    D = Ga - Gb
    if D <= 0.0:
        raise ValueError("the free energy is infinite at the proposed fold")
    s = 1.0 if x > y else -1.0
    Dx = Hb - s * Ha
    return _lj(x, beps)[2] + 1.0 / (x * x) - (Hpb - Hpa) / D + (Dx / D) ** 2

def _fold(n, beps, _cache={}):
    """(yc, xc, fc): the extension at which the intact minimum merges with the transition state, the merged bond
    length and the branch force there"""
    key = (n, beps)
    if key in _cache: return _cache[key]
    if _landscape(n, beps, 0.0) is None: raise ValueError("the link has no intact well even at zero extension")
    grid = [0.0] + [n * u for u in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)] + [n + 0.5, n + 1.0, n + 1.5]
    prev = 0.0
    for y in grid[1:]:
        if _wall(n, beps, y)[0] < 0.0: break
        prev = y
    else:
        raise ValueError("no critical extension found")
    yc = brentq(lambda t: _wall(n, beps, t)[0], prev, y, xtol=1e-13, rtol=1e-15, maxiter=200)
    xc0 = _wall(n, beps, yc)[1]
    def fold_equations(z):
        xx, yy = z
        return (_Ax(n, beps, xx, yy), _Axx(n, beps, xx, yy))
    refined = root(fold_equations, (xc0, yc), method="hybr", options={"xtol": 1e-12})
    if not np.all(np.isfinite(refined.x)) or np.linalg.norm(refined.fun, ord=np.inf) > 1e-8:
        raise ValueError("the critical fold did not converge")
    xc, yc = map(float, refined.x)
    fc = _free(n, beps, xc, yc)[2]
    _cache[key] = (yc, xc, fc)
    return _cache[key]

def _peq(n, beps, y):
    L = _landscape(n, beps, y)
    if L is None: return 0.0
    x1, xt, x2, A1, At, A2 = L
    return 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))

def _eq_density(n, beps, y):
    """rho_eq(y) = -dPeq/dy = Peq (1 - Peq) (f1 - f2) with f1, f2 the y-derivatives of A at the two minima; also f1"""
    L = _landscape(n, beps, y)
    if L is None: return 0.0, 0.0
    x1, xt, x2, A1, At, A2 = L
    P = 1.0 / (1.0 + (n + 1) * math.exp(-(A2 - A1)))
    f1 = _free(n, beps, x1, y)[2]; f2 = _free(n, beps, x2, y)[2]
    return P * (1.0 - P) * (f1 - f2), f1

def _y_half(n, beps):
    yc = _fold(n, beps)[0]
    g = lambda y: math.log(_peq(n, beps, y) / (1.0 - _peq(n, beps, y))) if 0.0 < _peq(n, beps, y) < 1.0 else (1.0 if _peq(n, beps, y) >= 1.0 else -1.0)
    return brentq(g, 0.0, yc - 1e-4, xtol=1e-13, rtol=1e-15, maxiter=200)

def _eq_stats(n, beps):
    """Half extension and conditional equilibrium means: normalize rho_eq = -dPeq/dy by its mass P_eq(0)-P_eq(yc-)."""
    yh = _y_half(n, beps); T = _pull_tables(n, beps); ye = T.yend
    def integrand(y):
        Es, Eh, f1, f2 = T.at(y, 4)
        P = 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
        rho = P * (1.0 - P) * (f1 - f2)
        return np.array([rho, y * rho, f1 * rho])
    tot = np.zeros(3)
    for a, b in ((0.0, yh), (yh, ye)):
        tot += quad_vec(integrand, a, b, epsabs=1e-14, epsrel=1e-13, limit=400, norm='max')[0]
    if abs(tot[0] - (_peq(n, beps, 0.0) - _peq(n, beps, ye))) > 1e-8:
        raise ValueError("the equilibrium scission density does not integrate to the total probability change")
    return yh, tot[1] / tot[0], tot[2] / tot[0]

# ---- the Kuhn-Grun free energy of the source (Cohen Pade inverse Langevin) ----
def _kg_h(f):
    """h(f) = f coth f + ln(f / sinh f), vectorised and overflow-safe"""
    f = np.asarray(f, dtype=np.float64)
    out = np.zeros_like(f)
    m = f > 1e-8
    fm = f[m]
    out[m] = fm / np.tanh(fm) + np.log(fm) - (fm + np.log((1.0 - np.exp(-2.0 * fm)) / 2.0))
    return out
def _kg_nodes(_cache={}):
    if 'gl' not in _cache: _cache['gl'] = np.polynomial.legendre.leggauss(300)
    return _cache['gl']
def _kg_free(n, beps, x, y):
    """beta A_KG(x; y) = beta V(x) - ln[x^2 I_n], I_n = int_{-1}^{1} exp(-n h(L^{-1}(r/n))) dc, r^2 = x^2 + y^2 - 2 x y c"""
    if x <= 0.0 or y < 0.0: raise ValueError("x must be positive and y non-negative")
    V = _lj(x, beps)[0]
    if y == 0.0:
        e = x / n
        if e >= 1.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
        I = 2.0 * math.exp(-n * float(_kg_h((3.0 * e - e ** 3) / (1.0 - e * e))))
    else:
        c, w = _kg_nodes()
        r = np.sqrt(np.maximum(x * x + y * y - 2.0 * x * y * c, 0.0)); e = r / n
        ok = e < 1.0
        val = np.zeros_like(e)
        ee = e[ok]
        val[ok] = np.exp(-n * _kg_h((3.0 * ee - ee ** 3) / (1.0 - ee * ee)))
        I = float(np.dot(w, val))
        if I <= 0.0: raise ValueError("the free energy is infinite: the fragments cannot span the gap")
    return V - 2.0 * math.log(x) - math.log(I)

def _kg_landscape(n, beps, y):
    """critical points of the Kuhn-Grun landscape by bracketed extremisation of A itself"""
    def A(x):
        try: return _kg_free(n, beps, x, y)
        except ValueError: return math.inf
    xlo = max(0.85, y - n + 1e-9); xmax = y + min(n - 1.0, 4.0 * math.sqrt(n) + 2.0)
    xs = np.concatenate([np.linspace(xlo, 1.8, 160), np.linspace(1.8, 3.0, 60)[1:], np.geomspace(3.0, xmax, 80)[1:]])
    v = np.array([A(x) for x in xs])
    d = np.diff(v)
    ext = [i for i in range(1, len(d)) if np.isfinite(v[i - 1]) and np.isfinite(v[i + 1]) and (d[i - 1] > 0.0) != (d[i] > 0.0)]
    if len(ext) < 3: return None
    i1, it, i2 = ext[:3]
    def refine(i, sign):
        r = minimize_scalar(lambda x: sign * A(x), bounds=(xs[i - 1], xs[i + 1]), method='bounded', options={'xatol': 1e-12, 'maxiter': 500})
        return r.x, sign * r.fun
    x1, A1 = refine(i1, 1.0); xt, At = refine(it, -1.0); x2, A2 = refine(i2, 1.0)
    return x1, xt, x2, A1, At, A2

def _kg_y_half(n, beps):
    def g(y):
        L = _kg_landscape(n, beps, y)
        if L is None: return -1e3
        return (L[5] - L[3]) - math.log(n + 1)
    return brentq(g, 0.0, n + 0.99, xtol=1e-11, rtol=1e-15, maxiter=200)

# ---- constant-rate pulling ----
def _cheb_nodes(N):
    k = np.arange(N)
    return np.cos(math.pi * (2 * k + 1) / (2 * N))           # first kind, ascending after reversal
def _cheb_eval(vals, t):
    """barycentric interpolation on Chebyshev nodes of the first kind; t in [-1, 1]"""
    N = len(vals); k = np.arange(N)
    x = np.cos(math.pi * (2 * k + 1) / (2 * N))
    w = (-1.0) ** k * np.sin(math.pi * (2 * k + 1) / (2 * N))
    d = t - x
    j = np.argmin(np.abs(d))
    if abs(d[j]) < 1e-15: return vals[j]
    q = w / d
    return float(np.dot(q, vals) / q.sum())

def _pull_tables(n, beps, _cache={}):
    class _Pull:
        """rate-independent tables of the barriers and the branch force on geometric panels up to the fold"""
        def __init__(self, n, beps, N=20):
            self.n, self.beps = n, beps
            yc, xc, fc = _fold(n, beps); self.yc, self.xc, self.fc = yc, xc, fc
            self.yend = yc - 1e-6
            # panels of width at most 2 (the density is a piecewise polynomial with knots at integer spacings, so wide
            # panels interpolate poorly), refined geometrically towards the fold
            edges = list(np.arange(0.0, yc - 3.0, 3.0)); d = yc - edges[-1]
            while d > 4e-5:
                d *= 0.5; edges.append(yc - d)
            edges.append(self.yend)
            self.edges = np.array(edges); self.N = N
            self.tab = []
            t = _cheb_nodes(N)
            for a, b in zip(edges[:-1], edges[1:]):
                ys = 0.5 * (a + b) + 0.5 * (b - a) * t
                rows = []
                for y in ys:
                    x1, xt, x2, Es, Eh = _barriers(n, beps, y)
                    rows.append((Es, Eh, _free(n, beps, x1, y)[2], _free(n, beps, x2, y)[2]))
                self.tab.append(np.array(rows))
        def at(self, y, k=3):
            """(Es, Eh, f1[, f2]) at y in [0, yend]: barriers, intact-branch force and broken-branch force"""
            i = int(np.searchsorted(self.edges, y, side='right') - 1)
            i = min(max(i, 0), len(self.tab) - 1)
            a, b = self.edges[i], self.edges[i + 1]
            t = (2.0 * y - a - b) / (b - a)
            T = self.tab[i]
            return tuple(_cheb_eval(T[:, j], t) for j in range(k))
    key = (n, beps)
    if key not in _cache: _cache[key] = _Pull(n, beps)
    return _cache[key]

def _pull(n, beps, gbar, ystop=None, method='Radau', rtol=1e-12):
    """probability of an intact chain and the final-scission moments when pulled at constant normalised rate gbar
    from y = 0 (P = 1): returns (P(ystop), <y_s>, <f_s>, normalisation) with the moments over the whole pull"""
    T = _pull_tables(n, beps); yc = T.yc; yend = T.yend
    def rates(y):
        Es, Eh, f = T.at(min(max(y, 0.0), yend))
        return (n + 1) * math.exp(-Es) / gbar, math.exp(-Eh) / gbar, f
    def peq(y):
        Es, Eh, f = T.at(y)
        return 1.0 / (1.0 + (n + 1) * math.exp(Eh - Es))
    # while the rates exceed the pulling rate by eight orders of magnitude the chain follows its equilibrium
    # probability to better than 1e-14; the kinetic integration starts where that stops being true
    if rates(0.0)[0] + rates(0.0)[1] <= 1e8:
        ys = 0.0; s = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    else:
        yh = brentq(lambda y: peq(y) - 0.5, 0.0, yend, xtol=1e-10, maxiter=200)
        ys = brentq(lambda y: math.log(rates(y)[0] + rates(y)[1]) - math.log(1e8), 0.0, yh, xtol=1e-10, maxiter=200)
        s = np.array([peq(ys), 0.0, 0.0, 0.0, 0.0])
    # reference point for the healing survival factor: int_yref^yend kh/gbar = 60 (Phi below 1e-26 further left)
    def Ltail(y0, eps):
        return quad(lambda y: rates(y)[1], y0, yend, limit=400, epsabs=0.0, epsrel=eps)[0]
    if Ltail(ys, 1e-6) <= 60.0:
        yref = ys
    else:
        yref = brentq(lambda y: Ltail(y, 1e-6) - 60.0, ys, yend, xtol=1e-6, maxiter=200)
    Lref = Ltail(yref, 1e-13)
    def make(active):
        # active: healing survival factor and its exponent are integrated (segments at and beyond yref); the
        # inactive form (Phi = 0) is used before yref, so that no segment carries a discontinuous right-hand side
        def rhs(y, v):
            P, C = v[0], v[1]
            ks, kh, f = rates(y)
            if active:
                Phi = math.exp(min(C - Lref, 0.0)); dC = kh
            else:
                Phi = 0.0; dC = 0.0
            w = P * ks * Phi
            return [-ks * P + kh * (1.0 - P), dC, w, y * w, f * w]
        def jac(y, v):
            ks, kh, f = rates(y)
            Phi = math.exp(min(v[1] - Lref, 0.0)) if active else 0.0
            P = v[0]; w = ks * Phi
            return [[-(ks + kh), 0.0, 0.0, 0.0, 0.0], [0.0] * 5, [w, P * w, 0.0, 0.0, 0.0], [y * w, y * P * w, 0.0, 0.0, 0.0], [f * w, f * P * w, 0.0, 0.0, 0.0]]
        return rhs, jac
    Pstop = None
    if ystop is not None and ystop <= ys: Pstop = peq(ystop) if ystop > 0.0 else 1.0
    # the tables are only continuous to rounding at the panel edges, so the integration restarts there; once the
    # intact probability has fallen below 1e-30 past the scission peak nothing further contributes
    stops = sorted(set([yref, yend] + [float(e) for e in T.edges if ys < e < yend] + ([ystop] if (ystop is not None and ystop > ys) else [])))
    def dead(y, v): return v[0] - 1e-30
    dead.terminal = True; dead.direction = -1
    y0 = ys; alive = True
    for y1 in stops:
        if y1 <= y0: continue
        if alive:
            rhs, jac = make(y0 >= yref)
            sol = solve_ivp(rhs, (y0, y1), s, method=method, rtol=rtol, atol=1e-16, jac=jac, first_step=1e-4 * (y1 - y0), events=dead)
            if not sol.success: raise ValueError(f"the pulling kinetics did not converge on [{y0}, {y1}] state {s}: " + sol.message)
            s = sol.y[:, -1]
            if sol.status == 1: alive = False; s[0] = 0.0
        y0 = y1
        if ystop is not None and y1 == ystop: Pstop = s[0]
    P, C, M0, Y1, F1 = s
    # whatever survives to the fold breaks there with the fold force
    M0 += P; Y1 += P * yc; F1 += P * T.fc
    if abs(M0 - 1.0) > 1e-7: raise ValueError("the final-scission density does not integrate to one")
    return Pstop, Y1 / M0, F1 / M0, M0

def chain_audit(n: int, beps: float, gbar: float) -> np.ndarray:
    nn = _links(n); b = _energy(beps); g = _pull_rate(gbar)
    L0 = chain_landscape(nn, b, 0.0)
    # the density step and the free-energy step must agree at zero extension
    for xx in (1.05, 2.3):
        lhs = breakable_chain_free_energy(nn, b, xx, 0.0)
        rhs = _lj(xx, b)[0] - math.log(2.0 * xx * xx) - fjc_log_density(nn, xx)
        if abs(lhs - rhs) > 1e-9 * max(1.0, abs(lhs)): raise ValueError("the free energy and the density disagree at zero extension")
    eq = equilibrium_rupture_statistics(nn, b); yh = float(eq[0])
    if abs(equilibrium_probability(nn, b, yh) - 0.5) > 1e-8: raise ValueError("the half-probability extension is inconsistent")
    cp = chain_critical_point(nn, b)
    fh = intact_branch_force(nn, b, yh)
    if not np.isfinite(fh): raise ValueError("non-finite force")
    Ph = pulled_chain_survival(nn, b, g, yh)
    fs = final_scission_statistics(nn, b, g)
    ykg = _kg_y_half(nn, b)
    Lkg = _kg_landscape(nn, b, yh)
    if Lkg is None: raise ValueError("the Kuhn-Grun landscape has no intact well at the exact half-probability extension")
    dkg = (Lkg[5] - Lkg[3]) - math.log(nn + 1)
    kgA = kuhn_grun_free_energy(nn, b, Lkg[0], yh)
    if abs(kgA - Lkg[3]) > 1e-9 * max(1.0, abs(kgA)): raise ValueError("the Kuhn-Grun landscape is inconsistent with its free energy")
    out = np.array([L0[3], L0[4], yh, eq[1], eq[2], cp[0], cp[2], Ph, fs[0], fs[1], fs[1] / eq[2], ykg, dkg], dtype=np.float64)
    if not np.all(np.isfinite(out)): raise ValueError("non-finite audit data")
    return out
SCICODE_GOLD_EOF
