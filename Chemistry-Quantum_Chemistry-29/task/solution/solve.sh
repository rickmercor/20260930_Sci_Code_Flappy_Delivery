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

def _vwn_fit(x, A, x0, b, c):
    import numpy as np
    X = x * x + b * x + c
    X0 = x0 * x0 + b * x0 + c
    Q = np.sqrt(4.0 * c - b * b)
    at = np.arctan(Q / (2.0 * x + b))
    g = A * (np.log(x * x / X) + 2.0 * b / Q * at
             - b * x0 / X0 * (np.log((x - x0) ** 2 / X) + 2.0 * (b + 2.0 * x0) / Q * at))
    dX = 2.0 * x + b
    dat = -2.0 * Q / (dX * dX + Q * Q)
    dg = A * (2.0 / x - dX / X + 2.0 * b / Q * dat
              - b * x0 / X0 * (2.0 / (x - x0) - dX / X + 2.0 * (b + 2.0 * x0) / Q * dat))
    return g, dg

def lsd_exchange_correlation(rho_up: "np.ndarray", rho_down: "np.ndarray") -> "np.ndarray":
    import numpy as np
    ru = np.asarray(rho_up, dtype=float)
    rd = np.asarray(rho_down, dtype=float)
    if ru.shape != rd.shape:
        raise ValueError("spin densities must have the same shape")
    if not (np.all(np.isfinite(ru)) and np.all(np.isfinite(rd))) or np.any(ru < 0.0) or np.any(rd < 0.0):
        raise ValueError("spin densities must be finite and non-negative")
    out = np.zeros((3,) + ru.shape)
    pos = (ru + rd) > 0.0
    u = ru[pos]
    d = rd[pos]
    n = u + d
    one_plus = 2.0 * u / n
    one_minus = 2.0 * d / n
    z = one_plus - 1.0
    cx = (6.0 / np.pi) ** (1.0 / 3.0)
    ex_energy = -0.75 * cx * (u * np.cbrt(u) + d * np.cbrt(d))
    x = np.sqrt(np.cbrt(3.0 / (4.0 * np.pi)) / np.cbrt(n))
    eP, dP = _vwn_fit(x, 0.0310907, -0.10498, 3.72744, 12.9352)
    eF, dF = _vwn_fit(x, 0.01554535, -0.32500, 7.06042, 18.0578)
    eA, dA = _vwn_fit(x, -1.0 / (6.0 * np.pi ** 2), -0.0047584, 1.13107, 13.0045)
    fden = 2.0 ** (4.0 / 3.0) - 2.0
    f = (one_plus * np.cbrt(one_plus) + one_minus * np.cbrt(one_minus) - 2.0) / fden
    df = (4.0 / 3.0) * (np.cbrt(one_plus) - np.cbrt(one_minus)) / fden
    fpp0 = 8.0 / (9.0 * fden)
    z3 = z ** 3
    z4 = z3 * z
    ec = eP + eA * f / fpp0 * (1.0 - z4) + (eF - eP) * f * z4
    dec_dx = dP + dA * f / fpp0 * (1.0 - z4) + (dF - dP) * f * z4
    dec_dz = eA / fpp0 * (df * (1.0 - z4) - 4.0 * z3 * f) + (eF - eP) * (df * z4 + 4.0 * z3 * f)
    common = ec - x / 6.0 * dec_dx
    out[0][pos] = ex_energy / n + ec
    out[1][pos] = -cx * np.cbrt(u) + common + one_minus * dec_dz
    out[2][pos] = -cx * np.cbrt(d) + common - one_plus * dec_dz
    return out

import numpy as np

def _radial_orbitals(L_int, wr, V, n):
    import numpy as np
    if n == 0:
        return np.zeros((0, V.size)), np.zeros(0)
    e, U = np.linalg.eig(-0.5 * L_int + np.diag(V))
    e = e.real
    U = U.real
    idx = np.argsort(e)[:n]
    U = U[:, idx]
    U = U / np.sqrt((wr[:, None] * U * U).sum(0))
    return U.T, e[idx]

def _radial_densities(L_int, wr, r, V_up, V_down, n_up, n_down):
    import numpy as np
    Uu, eu = _radial_orbitals(L_int, wr, V_up, n_up)
    Ud, ed = _radial_orbitals(L_int, wr, V_down, n_down)
    ru = (Uu ** 2).sum(0) / (4.0 * np.pi * r * r)
    rd = (Ud ** 2).sum(0) / (4.0 * np.pi * r * r) if n_down else np.zeros_like(r)
    return ru, rd, Uu, Ud, eu, ed

def _radial_potentials(L_int, L_inf, r, Z, n_el, ru, rd):
    import numpy as np
    U = np.linalg.solve(L_int, -4.0 * np.pi * r * (ru + rd) - L_inf * n_el)
    xc = lsd_exchange_correlation(np.maximum(ru, 0.0), np.maximum(rd, 0.0))
    return -Z / r + U / r + xc[1], -Z / r + U / r + xc[2]

def _radial_lsd_atom(Z, n_up, n_down, N=140):
    import numpy as np
    cache = _radial_lsd_atom.__dict__.setdefault("cache", {})
    key = (float(Z), int(n_up), int(n_down), int(N))
    if key in cache:
        return cache[key]
    k = np.arange(N + 1)
    t = np.cos(np.pi * k / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** k
    T = np.tile(t, (N + 1, 1)).T
    D = np.outer(c, 1.0 / c) / (T - T.T + np.eye(N + 1))
    D -= np.diag(D.sum(1))
    a = 2.0 / np.sqrt(Z)
    ti = t[1:N]
    r = a * (1.0 + ti) / (1.0 - ti)
    rt = 2.0 * a / (1.0 - ti) ** 2
    rtt = 4.0 * a / (1.0 - ti) ** 3
    L = (1.0 / rt ** 2)[:, None] * (D @ D)[1:N, :] - (rtt / rt ** 3)[:, None] * D[1:N, :]
    L_int = L[:, 1:N]
    L_inf = L[:, 0]
    th = np.pi * k / N
    wcc = np.zeros(N + 1)
    v = np.ones(N - 1)
    for j in range(1, N // 2):
        v -= 2.0 * np.cos(2.0 * j * th[1:N]) / (4.0 * j * j - 1.0)
    v -= np.cos(N * th[1:N]) / (N * N - 1.0)
    wcc[1:N] = 2.0 * v / N
    wr = wcc[1:N] * rt
    n_el = n_up + n_down
    V_up = -Z / r
    V_down = -Z / r
    xs, fs = [], []
    best = None
    for it in range(400):
        ru, rd, _, _, _, _ = _radial_densities(L_int, wr, r, V_up, V_down, n_up, n_down)
        new_up, new_down = _radial_potentials(L_int, L_inf, r, Z, n_el, ru, rd)
        x = np.concatenate([V_up, V_down])
        f = np.concatenate([new_up, new_down]) - x
        res = np.max(np.abs(f))
        if best is None or res < best[0]:
            best = (res, V_up.copy(), V_down.copy())
        if res < 1e-11:
            break
        xs.append(x)
        fs.append(f)
        xs, fs = xs[-8:], fs[-8:]
        if len(xs) > 1:
            dX = np.array([xs[i + 1] - xs[i] for i in range(len(xs) - 1)]).T
            dF = np.array([fs[i + 1] - fs[i] for i in range(len(fs) - 1)]).T
            g = np.linalg.lstsq(dF, f, rcond=None)[0]
            x = x + 0.5 * f - (dX + 0.5 * dF) @ g
        else:
            x = x + 0.3 * f
        V_up, V_down = x[:N - 1], x[N - 1:]
    ru, rd, Uu, Ud, eu, ed = _radial_densities(L_int, wr, r, best[1], best[2], n_up, n_down)
    uu = np.zeros((n_up, N + 1))
    uu[:, 1:N] = Uu
    ud = np.zeros((n_down, N + 1))
    ud[:, 1:N] = Ud
    bw = np.hstack([0.5, np.ones(N - 1), 0.5]) * (-1.0) ** k
    atom = {"t": t, "a": a, "bw": bw, "uu": uu, "ud": ud, "eu": eu, "ed": ed}
    cache[key] = atom
    return atom

def radial_lsd_spin_densities(Z: float, n_up: int, n_down: int, radii: "np.ndarray") -> "np.ndarray":
    import numpy as np
    if not np.isfinite(Z) or Z <= 0.0:
        raise ValueError("Z must be finite and positive")
    for n in (n_up, n_down):
        if isinstance(n, bool) or int(n) != n:
            raise ValueError("occupation numbers must be integers")
    n_up, n_down = int(n_up), int(n_down)
    if n_up < 1 or n_up > 2 or n_down < 0 or n_down > n_up or n_up + n_down > Z + 1e-12:
        raise ValueError("need 1 <= n_up <= 2, 0 <= n_down <= n_up and n_up + n_down <= Z")
    r = np.asarray(radii, dtype=float)
    if r.ndim != 1 or not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("radii must be a 1-D array of finite positive distances")
    atom = _radial_lsd_atom(float(Z), n_up, n_down)
    tt = (r - atom["a"]) / (r + atom["a"])
    diff = tt[:, None] - atom["t"][None, :]
    hit = np.abs(diff) < 1e-15
    diff[hit] = 1.0
    C = atom["bw"][None, :] / diff
    out = np.zeros((2, r.size))
    for s, U in enumerate((atom["uu"], atom["ud"])):
        if U.shape[0] == 0:
            continue
        val = (U @ C.T) / C.sum(1)
        for i in np.nonzero(hit.any(1))[0]:
            val[:, i] = U[:, np.nonzero(hit[i])[0][0]]
        out[s] = (val ** 2).sum(0) / (4.0 * np.pi * r * r)
    return out

import numpy as np

def _sgauss_check(exponents):
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    if a.ndim != 1 or a.size < 1 or not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("exponents must be a 1-D array of finite positive numbers")
    if np.unique(a).size != a.size:
        raise ValueError("exponents must be distinct")
    return a

def _radial_grid(exponents):
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    r_min = 1e-5 / np.sqrt(a.max())
    r_max = max(np.sqrt(200.0 / a.min()), 80.0)
    x = np.linspace(np.log(r_min), np.log(r_max), 6001)
    r = np.exp(x)
    w = 4.0 * np.pi * r ** 3 * (x[1] - x[0])
    w[0] *= 0.5
    w[-1] *= 0.5
    return r, w

def _basis_on_grid(exponents, r):
    import numpy as np
    a = np.asarray(exponents, dtype=float)
    return (2.0 * a[:, None] / np.pi) ** 0.75 * np.exp(-np.minimum(a[:, None] * r[None, :] ** 2, 745.0))

def _xc_on_grid(Z, n_up, n_down, r):
    import numpy as np
    rho = radial_lsd_spin_densities(Z, n_up, n_down, r)
    return lsd_exchange_correlation(np.maximum(rho[0], 0.0), np.maximum(rho[1], 0.0))

def xc_potential_matrices(Z: float, n_up: int, n_down: int, exponents: "np.ndarray") -> "np.ndarray":
    import numpy as np
    a = _sgauss_check(exponents)
    r, w = _radial_grid(a)
    G = _basis_on_grid(a, r)
    xc = _xc_on_grid(Z, n_up, n_down, r)
    V_up = (G * (w * xc[1])) @ G.T
    V_down = (G * (w * xc[2])) @ G.T
    return np.stack([0.5 * (V_up + V_up.T), 0.5 * (V_down + V_down.T)])

import numpy as np

def product_overlap_matrix(exponents: "np.ndarray") -> "np.ndarray":
    import numpy as np
    a = _sgauss_check(exponents)
    n = (2.0 * a / np.pi) ** 0.75
    iu, ju = np.triu_indices(a.size)
    pa = a[iu] + a[ju]
    pn = n[iu] * n[ju]
    return pn[:, None] * pn[None, :] * (np.pi / (pa[:, None] + pa[None, :])) ** 1.5

import numpy as np

def _select_products(V, exponents, eps):
    import numpy as np
    a = _sgauss_check(exponents)
    V = np.asarray(V, dtype=float)
    K = a.size
    if V.shape != (K, K) or not np.all(np.isfinite(V)) or not np.allclose(V, V.T, rtol=1e-12, atol=1e-14):
        raise ValueError("V must be a finite symmetric matrix matching the basis")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    W = product_overlap_matrix(a)
    iu, ju = np.triu_indices(K)
    target = V[iu, ju]
    chosen = []
    coef = np.zeros(0)
    fit = np.zeros_like(target)
    while len(chosen) < target.size:
        res = np.abs(fit - target)
        q = int(np.argmax(res))
        if res[q] < eps:
            break
        chosen.append(q)
        coef = np.linalg.solve(W[np.ix_(chosen, chosen)], target[chosen])
        fit = W[:, chosen] @ coef
    return np.array(chosen, dtype=int), coef, iu, ju

def importance_selected_products(V: "np.ndarray", exponents: "np.ndarray", eps: float) -> "np.ndarray":
    import numpy as np
    chosen, coef, iu, ju = _select_products(V, exponents, eps)
    if chosen.size == 0:
        return np.zeros((0, 2), dtype=int)
    return np.stack([iu[chosen], ju[chosen]], axis=1).astype(int)

import numpy as np

def reconstructed_potential(V: "np.ndarray", exponents: "np.ndarray", eps: float, radii: "np.ndarray") -> "np.ndarray":
    import numpy as np
    r = np.asarray(radii, dtype=float)
    if not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("radii must be finite and non-negative")
    pairs = importance_selected_products(V, exponents, eps)
    chosen, coef, iu, ju = _select_products(V, exponents, eps)
    a = np.asarray(exponents, dtype=float)
    n = (2.0 * a / np.pi) ** 0.75
    if pairs.shape[0] == 0:
        return np.zeros_like(r)
    i, j = pairs[:, 0], pairs[:, 1]
    pa = a[i] + a[j]
    pn = n[i] * n[j]
    flat = r.reshape(-1)
    vals = np.exp(-np.minimum(pa[None, :] * flat[:, None] ** 2, 745.0)) @ (pn * coef)
    return vals.reshape(r.shape)

import numpy as np

def reconstruction_l2_error(Z: float, n_up: int, n_down: int, exponents: "np.ndarray", spin: int, eps: float) -> float:
    import numpy as np
    if isinstance(spin, bool) or spin not in (0, 1):
        raise ValueError("spin must be 0 or 1")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    a = _sgauss_check(exponents)
    V = xc_potential_matrices(Z, n_up, n_down, a)[spin]
    r, w = _radial_grid(a)
    v = _xc_on_grid(Z, n_up, n_down, r)[1 + spin]
    vr = reconstructed_potential(V, a, eps, r)
    return float(np.sqrt(np.sum(w * (vr - v) ** 2)))
SCICODE_GOLD_EOF
