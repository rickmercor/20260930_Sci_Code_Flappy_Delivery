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

def pi_hamiltonian(n_sites: int, s: float, mode: str, params: dict) -> "np.ndarray":
    import numpy as np
    if isinstance(n_sites, bool) or int(n_sites) != n_sites or n_sites < 4 or n_sites % 2:
        raise ValueError("n_sites must be an even integer >= 4")
    if not np.isfinite(s) or s < 0.0 or s > 1.0:
        raise ValueError("s must lie in [0, 1]")
    if mode not in ("con", "dis"):
        raise ValueError("mode must be 'con' or 'dis'")
    n = int(n_sites)
    phi = 0.5 * np.pi * s
    r = params["r_open"] - (params["r_open"] - params["r_closed"]) * s
    h = np.zeros((n, n))
    for i in range(n - 1):
        h[i, i + 1] = h[i + 1, i] = params["beta_double"] if i % 2 == 0 else params["beta_single"]
    h[0, 1] = h[1, 0] = params["beta_double"] * np.cos(phi)
    h[n - 2, n - 1] = h[n - 1, n - 2] = params["beta_double"] * np.cos(phi)
    a = np.array([np.sin(phi), 0.0, np.cos(phi)])
    b = np.array([np.sin(phi) if mode == "con" else -np.sin(phi), 0.0, np.cos(phi)])
    t_pi = params["tau_pi"] * np.exp(-params["zeta_pi"] * (r - params["r_closed"]))
    t_sigma = params["tau_sigma"] * np.exp(-params["zeta_sigma"] * (r - params["r_closed"]))
    h[0, n - 1] = h[n - 1, 0] = t_pi * (a @ b) + (t_sigma - t_pi) * a[0] * b[0]
    return h

import numpy as np

def _string_tables(n, k):
    import numpy as np
    from itertools import combinations
    cache = _string_tables.__dict__.setdefault("cache", {})
    if (n, k) not in cache:
        strs = [sum(1 << i for i in c) for c in combinations(range(n), k)]
        pos = {x: i for i, x in enumerate(strs)}
        rows = []
        for i, x in enumerate(strs):
            for p in range(n):
                if not x >> p & 1:
                    continue
                for q in range(n):
                    if q == p or x >> q & 1:
                        continue
                    y = x ^ (1 << p) ^ (1 << q)
                    lo, hi = min(p, q), max(p, q)
                    sign = -1.0 if bin(x & ((1 << hi) - (1 << (lo + 1)))).count("1") % 2 else 1.0
                    rows.append((i, pos[y], q, p, sign))
        occ = np.array([[x >> p & 1 for p in range(n)] for x in strs], dtype=float)
        cache[(n, k)] = (strs, pos, occ, np.array(rows, dtype=float).reshape(-1, 5))
    return cache[(n, k)]


def _check_h_U(h, U):
    import numpy as np
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2 or h.shape[0] % 2 or h.shape[0] > 10:
        raise ValueError("h must be a square matrix of even size between 2 and 10")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-12):
        raise ValueError("h must be finite and symmetric")
    if not np.isfinite(U) or U < 0.0:
        raise ValueError("U must be finite and >= 0")
    return h, float(U)


def _fci_states(h, U, nroots):
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    n = h.shape[0]
    k = n // 2
    strs, pos, occ, rows = _string_tables(n, k)
    m = len(strs)
    src, tgt, q, p, sgn = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2].astype(int), rows[:, 3].astype(int), rows[:, 4]
    A = sp.coo_matrix((sgn * h[q, p], (tgt, src)), shape=(m, m)).tocsr() + sp.diags(occ @ np.diag(h))
    I = sp.identity(m, format="csr")
    H = sp.kron(A, I, format="csr") + sp.kron(I, A, format="csr") + sp.diags((U * occ @ occ.T).ravel())
    if m * m <= 100:
        E, V = np.linalg.eigh(H.toarray())
        return E[:nroots], V[:, :nroots]
    E, V = spl.eigsh(H, k=nroots, which="SA", tol=1e-14, v0=np.ones(m * m))
    o = np.argsort(E)
    return E[o], V[:, o]


def _spin_squared(c, n):
    import numpy as np
    k = n // 2
    strs, pos, occ, _ = _string_tables(n, k)
    up_strs, up_pos, _, _ = _string_tables(n, k + 1)
    dn_strs, dn_pos, _, _ = _string_tables(n, k - 1)
    m = len(strs)
    C = c.reshape(m, m)
    out = np.zeros((len(up_strs), len(dn_strs)))
    for ia, a in enumerate(strs):
        for ib, b in enumerate(strs):
            if C[ia, ib] == 0.0:
                continue
            for p in range(n):
                if (b >> p & 1) and not (a >> p & 1):
                    na, nb = a | (1 << p), b ^ (1 << p)
                    sign = (-1.0) ** (bin(a & ((1 << p) - 1)).count("1") + bin(b & ((1 << p) - 1)).count("1") + bin(a).count("1"))
                    out[up_pos[na], dn_pos[nb]] += sign * C[ia, ib]
    return float(np.sum(out * out))


def _one_rdm(c, n):
    import numpy as np
    k = n // 2
    strs, pos, occ, rows = _string_tables(n, k)
    m = len(strs)
    C = c.reshape(m, m)
    Pa = C @ C.T
    Pb = C.T @ C
    g = np.diag(occ.T @ np.diag(Pa) + occ.T @ np.diag(Pb))
    src, tgt, q, p, sgn = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2].astype(int), rows[:, 3].astype(int), rows[:, 4]
    np.add.at(g, (q, p), sgn * (Pa[tgt, src] + Pb[tgt, src]))
    return 0.5 * (g + g.T)


def singlet_ground_state(h: "np.ndarray", U: float) -> "np.ndarray":
    import numpy as np
    h, U = _check_h_U(h, U)
    n = h.shape[0]
    nroots = min(6, (len(_string_tables(n, n // 2)[0])) ** 2)
    E, V = _fci_states(h, U, nroots)
    for i in range(len(E)):
        if _spin_squared(V[:, i], n) < 0.5:
            occ = np.sort(np.linalg.eigvalsh(_one_rdm(V[:, i], n)))[::-1]
            return np.concatenate([[E[i]], occ])
    raise ValueError("no singlet among the lowest states")

import numpy as np

def frontier_mixing_descriptors(h: "np.ndarray", U: float) -> "np.ndarray":
    import numpy as np
    res = singlet_ground_state(h, U)
    n = len(res) - 1
    n_h = min(max(res[n // 2], 0.0), 2.0)
    theta = np.arccos(np.sqrt(n_h / 2.0))
    return np.array([np.degrees(theta), 0.5 * (1.0 - np.sin(2.0 * theta)), 0.5 * (1.0 + np.sin(2.0 * theta))])

import numpy as np

def _path_energy(n_sites, s, mode, U, params):
    return singlet_ground_state(pi_hamiltonian(n_sites, s, mode, params), U)[0]


def path_transition_state(n_sites: int, mode: str, U: float, params: dict) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import minimize_scalar
    if not np.isfinite(U) or U <= 0.0:
        raise ValueError("U must be finite and > 0")
    if isinstance(n_sites, bool) or int(n_sites) != n_sites or n_sites > 10:
        raise ValueError("n_sites must be an even integer between 4 and 10")
    grid = np.linspace(0.0, 1.0, 41)
    E = np.array([_path_energy(n_sites, s, mode, U, params) for s in grid])
    k = int(np.argmax(E))
    if k == 0 or k == len(grid) - 1:
        raise ValueError("the energy profile has no interior maximum")
    res = minimize_scalar(lambda s: -_path_energy(n_sites, s, mode, U, params),
                          bounds=(grid[k - 1], grid[k + 1]), method="bounded", options={"xatol": 1e-11})
    s_ts = float(res.x)
    e_ts = _path_energy(n_sites, s_ts, mode, U, params)
    th_0 = frontier_mixing_descriptors(pi_hamiltonian(n_sites, 0.0, mode, params), U)[0]
    th_ts = frontier_mixing_descriptors(pi_hamiltonian(n_sites, s_ts, mode, params), U)[0]
    return np.array([s_ts, e_ts - E[0], th_0, th_ts, th_ts - th_0])

import numpy as np

def _uhf_run(h, U, Da, Db, iters=3000):
    import numpy as np
    k = h.shape[0] // 2
    E_old = np.inf
    for _ in range(iters):
        ea, Ca = np.linalg.eigh(h + U * np.diag(np.diag(Db)))
        eb, Cb = np.linalg.eigh(h + U * np.diag(np.diag(Da)))
        Dan, Dbn = Ca[:, :k] @ Ca[:, :k].T, Cb[:, :k] @ Cb[:, :k].T
        Da, Db = 0.5 * (Da + Dan), 0.5 * (Db + Dbn)
        E = np.sum(h * (Da + Db)) + U * np.sum(np.diag(Da) * np.diag(Db))
        if abs(E - E_old) < 1e-14 and max(np.abs(Dan - Da).max(), np.abs(Dbn - Db).max()) < 1e-11:
            break
        E_old = E
    return E, Da, Db


def yamaguchi_mixing_angle(h: "np.ndarray", U: float) -> "np.ndarray":
    import numpy as np
    h, U = _check_h_U(h, U)
    n = h.shape[0]
    k = n // 2
    e, C = np.linalg.eigh(h)
    D0 = C[:, :k] @ C[:, :k].T
    starts = [(D0, D0)]
    grid = np.arange(n * n).reshape(n, n)
    for j in range(1, 20):
        Qa = np.linalg.qr(np.eye(n) + 0.6 * np.sin(1.7 * j + 2.3 * grid + 0.37 * grid ** 2))[0]
        Qb = np.linalg.qr(np.eye(n) + 0.6 * np.cos(2.9 * j - 1.1 * grid + 0.53 * grid ** 2))[0]
        Ca, Cb = C @ Qa, C @ Qb
        starts.append((Ca[:, :k] @ Ca[:, :k].T, Cb[:, :k] @ Cb[:, :k].T))
    best = None
    for Da, Db in starts:
        E, Da, Db = _uhf_run(h, U, Da, Db)
        if max(np.abs(Da @ Da - Da).max(), np.abs(Db @ Db - Db).max()) > 1e-8:
            continue
        if best is None or E < best[0] - 1e-10:
            best = (E, Da, Db)
    E, Da, Db = best
    occ = np.sort(np.linalg.eigvalsh(Da + Db))[::-1]
    T = min(max(occ[k - 1] - 1.0, 0.0), 1.0)
    y = 1.0 - 2.0 * T / (1.0 + T * T)
    return np.array([E, occ[k - 1], y, np.degrees(np.arcsin(np.sqrt(y / 2.0)))])

import numpy as np

def _fit_box_map(z, bounds):
    import numpy as np
    (u_low, u_high), (b_low, b_high) = bounds
    return (u_low + (u_high - u_low) / (1.0 + np.exp(-z[0])), b_low + (b_high - b_low) / (1.0 + np.exp(-z[1])))


def _fit_residual(z, n_sites, targets, params, bounds):
    U, b = _fit_box_map(z, bounds)
    p = dict(params, beta_single=b)
    return [path_transition_state(n_sites, "con", U, p)[4] - targets[0],
            path_transition_state(n_sites, "dis", U, p)[4] - targets[1]]


def calibrate_model(n_sites: int, target_con: float, target_dis: float, params: dict, U_bounds: tuple,
                            beta_bounds: tuple) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import fsolve
    U_low, U_high = float(U_bounds[0]), float(U_bounds[1])
    b_low, b_high = float(beta_bounds[0]), float(beta_bounds[1])
    if not all(np.isfinite([target_con, target_dis, U_low, U_high, b_low, b_high])):
        raise ValueError("targets and bounds must be finite")
    if U_low <= 0.0 or U_high <= U_low or b_high <= b_low or b_high >= 0.0:
        raise ValueError("need 0 < U_low < U_high and beta_low < beta_high < 0")
    bounds = ((U_low, U_high), (b_low, b_high))
    sol, info, ier, msg = fsolve(_fit_residual, [0.0, 0.0],
                                 args=(n_sites, (target_con, target_dis), params, bounds),
                                 full_output=True, xtol=1e-13, epsfcn=1e-4)
    U, b = _fit_box_map(sol, bounds)
    if max(abs(v) for v in info["fvec"]) > 1e-5 or not (U_low < U < U_high and b_low < b < b_high):
        raise ValueError("the search does not converge inside the rectangle")
    return np.array([U, b])

import numpy as np

def calibrated_puhf_angle(n_sites: int, mode: str, target_con: float, target_dis: float, params: dict,
                                  U_bounds: tuple, beta_bounds: tuple) -> float:
    if mode not in ("con", "dis"):
        raise ValueError("mode must be 'con' or 'dis'")
    U_star, beta_star = calibrate_model(n_sites, target_con, target_dis, params, U_bounds, beta_bounds)
    fitted = dict(params, beta_single=beta_star)
    s_ts = path_transition_state(n_sites, mode, U_star, fitted)[0]
    return float(yamaguchi_mixing_angle(pi_hamiltonian(n_sites, s_ts, mode, fitted), U_star)[3])
SCICODE_GOLD_EOF
